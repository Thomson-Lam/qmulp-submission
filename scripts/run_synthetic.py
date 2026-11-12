"""Run the synthetic experiments S1-S3 and the S3b sensitivity run.

Writes ``outputs/results/synthetic/EC_synthetic_S*.csv`` and their summaries.

    python -m scripts.run_synthetic                  # 30 replicates, n = 3,000
    python -m scripts.run_synthetic --smoke          # 2 replicates, n = 300, nothing reportable
    python -m scripts.run_synthetic --experiment S3b # one experiment only

S3b re-scores the S3 populations (same seeds) at m = 0 and 0.3 with the development references placed per category instead of per
joint cell, and reports the correlation between the arms at m = 0 under both placements (printed, and in ``EC_synthetic_keys.json``).
Also verifies, on three small instances, that the lumped operator reproduces the unlumped step marginals
(``lumped_equals_full.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.provenance import RunTimer, write_manifest, write_result  # noqa: E402
from paths import stage_dir  # noqa: E402
from quantum.mechanism.scores_stage import full_vs_reduced_error  # noqa: E402
from quantum.mechanism.variants import primary_variant, route_variant  # noqa: E402
from synthetic import experiments as ex  # noqa: E402
from synthetic.generator import generate_attribute_graph  # noqa: E402

STAGE = "08_synthetic"
SCOPE = "synthetic"
N_REPLICATES = 30
N_INDIVIDUALS = 3_000
SMOKE = {"replicates": 2, "n": 300}
SUMMARY_KEYS = {
    "S1": (("mode", "dangling_share"), ("max_change_classical", "max_change_qppr", "rho_order_qppr", "rho_qc", "d_prauc")),
    "S2": (("levels", "size_dispersion", "ref_share_spread"), ("one_minus_rho", "abs_d_prauc", "levels_q", "levels_c")),
    "S3": (("config", "partial_share", "kappa"), ("rho_qc_all", "rho_qc_fully_observed", "partial_share_top_q", "partial_share_top_c", "d_prauc")),
    "S3b": (("config", "partial_share", "kappa"),
            ("rho_qc_all", "rho_qc_fully_observed", "partial_share_top_q", "partial_share_top_c", "d_prauc", "rho_qc_all_joint_cell",
             "d_rho_qc_all", "ref_share_gap_category", "ref_share_gap_joint_cell")),
}


def _fmt(value) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def lumped_equals_full(config, n: int) -> list[dict]:
    """The lumped step marginals equal the unlumped ones on three small instances."""
    out = []
    for rep, (L, levels, m) in enumerate(((1, (4,), 0.0), (2, (3, 2), 0.0), (2, (4, 3), 0.2))):
        instance = generate_attribute_graph(n, L, levels, 0.5, 0.2, 0.03, 0.2, m, 1.0, 0.05, 0.1, config.seed_synthetic("check", rep))
        partitions = instance.partitions()
        variant = primary_variant(partitions, instance.spec, config, partitions.test_idx)
        routed = route_variant(variant, config, keep_full=True)
        error = full_vs_reduced_error(routed, variant, int(config.reduction_check_steps))
        out.append({"instance": rep, "L": L, "levels": list(levels), "route": routed.route, **routed.record(), **error})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--experiment", choices=("S1", "S2", "S3", "S3b"), default=None)
    parser.add_argument("--replicates", type=int, default=None)
    args = parser.parse_args(argv)
    config = DEFAULT_CONFIG
    replicates = args.replicates or (SMOKE["replicates"] if args.smoke else N_REPLICATES)
    n = SMOKE["n"] if args.smoke else N_INDIVIDUALS
    timer = RunTimer()
    output = stage_dir(STAGE)
    outputs = []
    check = lumped_equals_full(config, 60 if args.smoke else 200)
    (output / "lumped_equals_full.json").write_text(json.dumps(check, indent=2, default=str))
    worst = max(c["full_vs_reduced_error"] for c in check)
    print(f"lumped == full on {len(check)} instances: worst step error {worst:.3e}")
    summaries = {}
    runners = {"S1": ex.s1_rows, "S2": ex.s2_rows, "S3": ex.s3_rows, "S3b": ex.s3b_rows}
    for name, runner in runners.items():
        if args.experiment and name != args.experiment:
            continue
        rows = runner(config, n=n, replicates=replicates)
        keys, metrics = SUMMARY_KEYS[name]
        summary = ex.summarise(rows, keys, metrics)
        long_path = write_result(SCOPE, f"EC_synthetic_{name}", pl.DataFrame(rows, strict=False, infer_schema_length=None), stage=STAGE,
                                 columns=ex.ROW_COLUMNS[name])
        summary_path = write_result(SCOPE, f"EC_synthetic_{name}_summary", pl.DataFrame(summary, strict=False, infer_schema_length=None),
                                    stage=STAGE, columns=ex.summary_columns(name, keys, metrics))
        outputs += [long_path, summary_path]
        summaries[name] = summary
        print(f"{name}: {len(rows)} rows -> {long_path.name}, {summary_path.name}")
    if not args.experiment:
        keys = ex.headline_keys(summaries["S1"], summaries["S2"], summaries["S3"], summaries["S3b"])
        keys_path = output / "EC_synthetic_keys.json"
        keys_path.write_text(json.dumps(keys, indent=2, default=str))
        outputs.append(keys_path)
        print(json.dumps(keys, indent=1, default=str))
        for cfg in ex.S3_CONFIGS:
            value = {q: keys.get(f"{cfg}.{q}") for q in ("s3_rho_m0", "s3b_rho_m0_category", "s3b_d_rho_m0")}
            print(f"m = 0, {cfg}: rho(Q, C) per joint cell {_fmt(value['s3_rho_m0'])}, per category {_fmt(value['s3b_rho_m0_category'])}, "
                  f"paired difference {_fmt(value['s3b_d_rho_m0'])}")
    write_manifest(output, arm_name=STAGE, config=config, inputs=[], outputs=[*outputs, output / "lumped_equals_full.json"],
                   seeds={"synthetic": "config.seed_synthetic(experiment, replicate, ...)"}, timer=timer,
                   execution_mode="simulation", extra={"replicates": replicates, "n": n, "smoke": bool(args.smoke),
                                                       "lumped_equals_full_worst": worst})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
