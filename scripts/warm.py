"""Array-task dispatcher: compute exactly one expensive artifact of stage 04 or 06.

    python -m scripts.warm --stage 04 --list          # print the task table, compute nothing
    python -m scripts.warm --stage 04 --task-id 2     # compute exactly task 2
    python -m scripts.warm --stage 04 --dataset fmac --list
    python -m scripts.warm --stage 06 --list
    python -m scripts.warm --stage 06 --dataset fmac --task-id 3

Each task warms the cache the stage already reads -- one checkpointed
full-operator reduction-check walk (stage 04) or one retained instance's
noise-grid shard (stage 06) -- by calling the same named function the stage
calls, then exits. No manifest, merged CSV or table is written here; those
are the stage's job. ``tasks_for_stage`` is the one list the sbatch
``--array`` bounds are derived from.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from config import DEFAULT_CONFIG, ExperimentConfig
from datasets import DATASET_NAMES, get_dataset
from fingerprint import resolve_config, sha256_path
from paths import relative_to_repo

STAGES = ("04", "06")


@dataclass(frozen=True)
class Task:
    index: int
    stage: str
    kind: str
    network: str
    tag: str
    path: Path
    tau: int | None = None
    n_classes: int | None = None
    n_full: int | None = None
    dataset: str = "hmeq"


def _recorded_class_count(data) -> int | None:
    """The primary network's lumped class count as stage 04 recorded it; ``None`` before it has run."""
    import json

    path = data.stage_path("quantum") / f"convergence_{data.primary}.json"
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["lumped"]["n_classes"])


def tasks_for_stage(stage: str, config: ExperimentConfig, *, n_full: int | None = None, dataset: str = "hmeq") -> list[Task]:
    """Ordered, deterministic task list of one stage; the single source of truth for ``--array``.

    Stage 04: one task per planned reduction-check walk of ``dataset``. Stage
    06: one task per retained instance of the dataset's primary network;
    ``n_full`` is the lumped class count, read from stage 04's convergence
    record when it exists and otherwise taken as the largest configured
    sweep entry (the worker asserts the real count before computing).
    """
    data = get_dataset(dataset)
    if stage == "04":
        from paths import quantum_history_path
        from quantum.inspired.walk_cache import planned_walks

        quantum_stage = data.stage("quantum")
        return [Task(i, stage, w["kind"], w["network"], w["tag"],
                     quantum_history_path(w["network"], w["tag"], stage=quantum_stage, create=False), tau=w["tau"], dataset=dataset)
                for i, w in enumerate(planned_walks(config, ladder=data.ladder, primary=data.primary))]
    if stage == "06":
        from quantum.hardware.class_candidates import candidate_label, sweep_sizes
        from quantum.hardware.shards import shard_path

        if n_full is None:
            n_full = _recorded_class_count(data) or max(config.hardware_class_sweep)
        output = data.stage_path("hardware")
        return [Task(i, stage, "noise_grid_instance", data.primary, candidate_label(n, n_full), shard_path(output, n),
                     n_classes=n, n_full=n_full, dataset=dataset) for i, n in enumerate(sweep_sizes(config, n_full))]
    raise ValueError(f"stage must be one of {STAGES}; got {stage!r}")


def print_tasks(tasks: list[Task]) -> None:
    print(f"{'index':>5}  {'kind':<22}{'network':<16}{'tag':<18}{'horizon/classes':<18}output")
    for t in tasks:
        size = f"tau={t.tau}" if t.tau is not None else f"n_classes={t.n_classes} (of {t.n_full})"
        print(f"{t.index:>5}  {t.kind:<22}{t.network:<16}{t.tag:<18}{size:<18}{relative_to_repo(t.path)}")
    print(f"{len(tasks)} tasks -> sbatch --array=0-{len(tasks) - 1}")


def run_walk_task(task: Task, config: ExperimentConfig) -> list[Path]:
    from quantum.inspired import stage, walk_cache

    data = get_dataset(task.dataset)
    plan = walk_cache.dataset_networks(data, resolve_specs=True)
    output = data.stage_dir("quantum")
    directory = walk_cache.checkpoint_dir(output, task.network, task.tag)
    if task.path.exists() and (directory / "completed.npz").exists():
        print(f"task {task.index}: {task.network}/{task.tag} already complete; the stage re-checks its fingerprint")
    else:
        print(f"task {task.index}: {task.network}/{task.tag} to tau={task.tau}")
        stage.run_walk(task.network, task.tag, task.tau, config, stage.load_classical_digests(stage=plan["classical"]), {},
                       output, specs=plan["specs"], stage=plan["quantum"])
    return [task.path, directory / "completed.npz"]


def run_grid_task(task: Task, config: ExperimentConfig) -> list[Path]:
    """Simulate one retained instance. An existing shard is checked against this run's identity (a stale one raises),
    kept when complete and simulated again when partial; the check lives in ``stages.hardware.run``."""
    from stages import hardware

    print(f"task {task.index}: noise grid of the {task.tag} instance ({task.n_classes} of {task.n_full} classes)", flush=True)
    result = hardware.run(task.dataset, config, instances=[int(task.n_classes)], assemble=False)
    if not result["instances"]:
        raise AssertionError(f"task {task.index}: {task.n_classes} classes is not a retained instance of the {task.dataset} primary network "
                             "(stage 04 has not recorded its class count, or the sweep changed); regenerate the --array bounds from --list")
    return list(result["artifacts"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--dataset", default="hmeq", choices=DATASET_NAMES)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print the task table and exit; computes nothing")
    group.add_argument("--task-id", type=int, help="compute exactly this task (the SLURM array index)")
    args = parser.parse_args(argv)

    config, smoke = resolve_config(DEFAULT_CONFIG)
    tasks = tasks_for_stage(args.stage, config, dataset=args.dataset)
    if args.list:
        print(f"stage {args.stage} | dataset {args.dataset} | smoke={smoke} | tasks derived from the frozen configuration")
        print_tasks(tasks)
        return 0
    if not 0 <= args.task_id < len(tasks):
        print(f"task id {args.task_id} is outside 0..{len(tasks) - 1} for stage {args.stage} (smoke={smoke}); "
              "regenerate the --array bounds from --list", file=sys.stderr)
        return 2
    task = tasks[args.task_id]
    print(f"smoke={smoke}")
    written = run_walk_task(task, config) if args.stage == "04" else run_grid_task(task, config)
    for path in written:
        if path.exists():
            print(f"{sha256_path(path)}  {relative_to_repo(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
