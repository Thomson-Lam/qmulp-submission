"""Run the whole pipeline, stage by stage, inside one process tree; no scheduler.

    python -m scripts.run_all                        # both datasets, every stage, then figures and tables
    python -m scripts.run_all --dataset hmeq         # one dataset
    python -m scripts.run_all --from-stage 04        # resume from a stage
    python -m scripts.run_all --smoke                # trimmed configuration, nothing reportable
    python -m scripts.run_all --to-stage 05a         # stop after a stage
    python -m scripts.run_all --dataset fmac --from-stage 03 --to-stage 03 --skip-figures   # one stage of one dataset
    python -m scripts.run_all --require-clean        # refuse to run from a tree with uncommitted changes

Stages, in order: ``01`` data, ``02`` network, ``03`` classical, ``04``
quantum (limit, tau*, scores), ``05`` evaluation (agreement, decision,
strata, placebo, downstream), ``05a`` mechanism (after ``05``: it reads the
placebo summary ``05`` writes), ``06`` hardware (the noise
grid, then the assembly), ``07`` counterexample (Table EC.1), ``08``
synthetic (S1 to S3 and the S3b sensitivity run), ``09`` variability (both
datasets), ``10`` runtime; then ``scripts/make_figures.py`` and
``scripts/make_manuscript_tables.py``.

A stage notebook is one notebook per stage with a cell tagged ``parameters``
that holds ``DATASET``; this driver replaces that cell in memory for each
dataset, executes the notebook with ``nbclient`` and writes the executed copy to
``outputs/executed_notebooks/<stage>_<dataset>.ipynb``. The tracked notebooks
stay stripped. The data stage keeps one notebook per dataset (the cleaning
narrative differs). Before the stage-06 notebook, the noise-grid instances without a
complete shard are simulated one subprocess each (``scripts.warm --stage 06``): after
the array jobs of a batch chain every shard is complete and this step does nothing
(the stage then checks each shard's identity as it assembles), and in one single job
it is what computes the grid.
Every circuit result is a simulation; nothing here submits to a device.

A dirty working tree never blocks a run: the manifests record the commit, the
dirty flag and ``source_sha256``, which identifies the code that produced the
results whether or not it is committed. ``--require-clean`` is opt-in.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DATASET_NAMES  # noqa: E402
from paths import OUTPUTS_DIR  # noqa: E402

STAGES = ("01", "02", "03", "04", "05", "05a", "06", "07", "08", "09", "10")
# The data stage is dataset-specific narrative; every other notebook stage is one notebook with a parameters cell.
DATASET_NOTEBOOKS = {("01", "hmeq"): "01_hmeq_cleaning.ipynb", ("01", "fmac"): "01_fmac_cleaning.ipynb"}
STAGE_NOTEBOOKS = {"02": "02_network.ipynb", "03": "03_classical.ipynb", "04": "04_quantum.ipynb", "05": "05_evaluation.ipynb",
                   "05a": "05a_mechanism.ipynb", "06": "06_hardware.ipynb"}
PARAMETERS_TAG = "parameters"
SCRIPTS = {"07": ["-m", "scripts.ec_counterexample"], "08": ["-m", "scripts.run_synthetic"], "09": ["-m", "scripts.run_multisplit"],
           "10": ["-m", "scripts.benchmark_runtime", "--scaling"]}
# The counterexample is an eight-individual exact computation, identical in smoke and real runs: it takes no --smoke flag.
NO_SMOKE_FLAG = frozenset({"07"})


def notebook_for(stage: str, dataset: str) -> str:
    """The notebook file that runs ``stage`` for ``dataset``."""
    return DATASET_NOTEBOOKS.get((stage, dataset)) or STAGE_NOTEBOOKS[stage]


def set_dataset(nb, dataset: str) -> bool:
    """Replace the cell tagged ``parameters`` by ``DATASET = "<dataset>"`` in memory; whether such a cell exists."""
    for cell in nb.cells:
        if cell.cell_type == "code" and PARAMETERS_TAG in cell.metadata.get("tags", []):
            cell.source = f'DATASET = "{dataset}"\n'
            return True
    return False


def execute_notebook(stage: str, dataset: str, smoke: bool) -> float:
    """Execute the stage's notebook for one dataset into ``outputs/executed_notebooks/<stage>_<dataset>.ipynb``.

    The source notebook is untouched; the dataset is set in memory. A stage notebook without its parameters cell is an error.
    """
    import nbformat
    from nbclient import NotebookClient

    name = notebook_for(stage, dataset)
    source = ROOT / "notebooks" / name
    target_dir = OUTPUTS_DIR / "executed_notebooks"
    target_dir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(source, as_version=4)
    parameterised = set_dataset(nb, dataset)
    if (stage, dataset) not in DATASET_NOTEBOOKS and not parameterised:
        raise AssertionError(f"{name} has no cell tagged {PARAMETERS_TAG!r}; a stage notebook is parameterised on DATASET")
    began = time.perf_counter()
    client = NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": str(source.parent)}})
    os.environ["QMULP_SMOKE"] = "1" if smoke else ""
    client.execute()
    nbformat.write(nb, target_dir / f"{stage}_{dataset}.ipynb")
    return time.perf_counter() - began


def run_script(args: list[str], smoke: bool, extra: list[str], *, pass_flag: bool = True) -> float:
    began = time.perf_counter()
    command = [sys.executable, *args, *extra, *(["--smoke"] if smoke and pass_flag else [])]
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ, "QMULP_SMOKE": "1" if smoke else ""})
    return time.perf_counter() - began


def shard_is_complete(path: Path) -> bool:
    """Whether a noise-grid shard file exists and says it is complete (its identity is checked when the stage assembles it)."""
    try:
        return bool(json.loads(path.read_text()).get("complete"))
    except (OSError, ValueError):
        return False


def run_noise_grid(dataset: str, smoke: bool) -> float:
    """Simulate every retained instance of ``dataset`` that has no complete shard, one subprocess per instance (the array tasks).

    After the array jobs of a batch chain every shard is complete and this step does nothing.
    """
    from config import DEFAULT_CONFIG
    from fingerprint import resolve_config
    from scripts.warm import tasks_for_stage

    config, _ = resolve_config(DEFAULT_CONFIG, smoke=smoke)
    began = time.perf_counter()
    for task in tasks_for_stage("06", config, dataset=dataset):
        if shard_is_complete(task.path):
            continue
        run_script(["-m", "scripts.warm", "--stage", "06", "--dataset", dataset, "--task-id", str(task.index)], smoke, [], pass_flag=False)
    return time.perf_counter() - began


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASET_NAMES, action="append")
    parser.add_argument("--from-stage", choices=STAGES, default="01")
    parser.add_argument("--to-stage", choices=STAGES, default="10")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--require-clean", action="store_true", help="refuse to run from a tree with uncommitted changes (off by default)")
    args = parser.parse_args(argv)
    datasets = args.dataset or list(DATASET_NAMES)
    start, stop = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > stop:
        parser.error("--from-stage must not come after --to-stage")
    from evaluation.provenance import git_state

    state = git_state()
    if state["dirty"]:
        if args.require_clean:
            print("run_all: the working tree has uncommitted changes and --require-clean was given; nothing was run", file=sys.stderr)
            return 1
        print("run_all: warning: the working tree has uncommitted changes; the manifests record the commit, the dirty flag "
              "and source_sha256", file=sys.stderr)
    timings: dict[str, float] = {}
    for stage in STAGES[start:stop + 1]:
        if stage in SCRIPTS:
            if stage == "09":
                for ds in datasets:
                    timings[f"{stage}:{ds}"] = run_script(SCRIPTS[stage], args.smoke, ["--dataset", ds])
            elif stage == "10":
                timings[stage] = run_script(SCRIPTS[stage], args.smoke, [arg for ds in datasets for arg in ("--dataset", ds)])
            else:
                timings[stage] = run_script(SCRIPTS[stage], args.smoke, [], pass_flag=stage not in NO_SMOKE_FLAG)
            continue
        for ds in datasets:
            if stage == "06":
                print(f"== stage 06 {ds}: noise grid")
                timings[f"06grid:{ds}"] = run_noise_grid(ds, args.smoke)
            print(f"== stage {stage} {ds}: {notebook_for(stage, ds)}")
            timings[f"{stage}:{ds}"] = execute_notebook(stage, ds, args.smoke)
    if not args.skip_figures:
        timings["figures"] = run_script(["-m", "scripts.make_figures"], False, [])
        timings["tables"] = run_script(["-m", "scripts.make_manuscript_tables"], False, [])
    for key, seconds in timings.items():
        print(f"{key:>12}: {seconds:9.1f} s")
    # One file per invocation, so the stage jobs of a batch chain do not overwrite each other.
    label = f"{'+'.join(datasets)}_{args.from_stage}-{args.to_stage}"
    timings_dir = OUTPUTS_DIR / "timings"
    timings_dir.mkdir(parents=True, exist_ok=True)
    (timings_dir / f"run_all_{label}.json").write_text(json.dumps(timings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
