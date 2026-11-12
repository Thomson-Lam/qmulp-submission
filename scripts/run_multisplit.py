"""The variability study (Table 2 columns): HMEQ re-splits, Freddie Mac intervals and reference resamples.

    python -m scripts.run_multisplit --dataset hmeq      # 50 splits + the frozen one, every HMEQ network
    python -m scripts.run_multisplit --dataset fmac      # bootstrap intervals + 50 reference resamples
    python -m scripts.run_multisplit --dataset hmeq --smoke

Writes ``EC_multisplit_long.csv`` and ``T2_variability.csv`` under
``outputs/results/<dataset>/`` and a manifest under
``outputs/notebook_pipeline/09_multisplit_<dataset>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from datasets import DATASET_NAMES, get_dataset  # noqa: E402
from evaluation.context import Inputs  # noqa: E402
from evaluation.multisplit import (  # noqa: E402
    HMEQ_FROZEN_SEED,
    LONG_COLUMNS,
    fmac_bootstrap_intervals,
    fmac_reference_resamples,
    hmeq_multisplit,
    resample_summary,
    split_seeds,
    summary_columns,
    variability_summary,
)
from evaluation.provenance import RunTimer, write_manifest, write_result  # noqa: E402
from fingerprint import resolve_config  # noqa: E402
from paths import stage_dir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-tau-star", action="store_true", help="skip tau* on the primary network per split")
    args = parser.parse_args(argv)
    config, smoke = resolve_config(DEFAULT_CONFIG, smoke=args.smoke)
    dataset = get_dataset(args.dataset)
    stage = f"09_multisplit_{dataset.name}"
    timer = RunTimer()
    output = stage_dir(stage)
    outputs = []
    single = [n for n, layers in dataset.networks.items() if len(layers) == 1]
    if dataset.split == "stratified":
        rows = hmeq_multisplit(dataset, config, with_tau_star=not args.no_tau_star)
        summary = variability_summary(rows, frozen_seed=HMEQ_FROZEN_SEED)
        seeds = split_seeds(config)
        columns = summary_columns("summary", single)
        extra = {"kind": "resplit", "n_splits": len(seeds), "seeds": seeds, "frozen_seed": HMEQ_FROZEN_SEED}
    else:
        data = Inputs(config, dataset)
        intervals = fmac_bootstrap_intervals(data)
        rows = fmac_reference_resamples(dataset, config)
        # Two summaries, two name families in one file: resample_* (reference resamples) and boot_* (bootstrap over test loans).
        summary = resample_summary(variability_summary(rows, frozen_seed=None), intervals)
        columns = summary_columns("resample", single)
        outputs.append(write_result(dataset.name, "T2_variability_bootstrap", pl.DataFrame(intervals), stage=stage,
                                    columns=summary_columns("bootstrap")))
        extra = {"kind": "reference_resample", "n_draws": int(config.n_reference_resamples_fmac)}
    long_path = write_result(dataset.name, "EC_multisplit_long", pl.DataFrame(rows, strict=False, infer_schema_length=None), stage=stage,
                             columns=LONG_COLUMNS)
    summary_path = write_result(dataset.name, "T2_variability", pl.DataFrame(summary, strict=False, infer_schema_length=None), stage=stage,
                                columns=columns)
    outputs += [long_path, summary_path]
    write_manifest(output, arm_name=stage, config=config,
                   inputs=[dataset.processed_path(), dataset.frozen_split_path()], outputs=outputs,
                   seeds={"master_seed": config.master_seed}, timer=timer, execution_mode="simulation",
                   extra={"dataset": dataset.name, "smoke": smoke, **extra})
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
