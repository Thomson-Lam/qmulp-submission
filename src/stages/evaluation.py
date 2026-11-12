"""Stage 05: the agreement, decision, strata and placebo tables, and the downstream robustness check.

Reads the scoring stage's ``scores_<network>.parquet`` files through
``evaluation.context.Inputs`` for the agreement and decision tables, the
strata and the downstream models; rebuilds the lumped operators through the
router for the placebo draws and the fold operators.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from datasets import get_dataset
from evaluation.context import DatasetContext, Inputs
from evaluation.controls import run_placebo, write_placebo_tables
from evaluation.downstream import crossfit_scores, evaluate_feature_sets, feature_frame, feature_importance, write_downstream_tables
from evaluation.provenance import RunTimer, write_manifest
from evaluation.tables import write_agreement_tables, write_decision_tables, write_strata_table
from fingerprint import resolve_config

STAGE_KEY = "downstream"


def run(dataset_name: str, config: ExperimentConfig | None = None, *, placebo: bool = True, downstream: bool = True) -> dict[str, Any]:
    """Run the evaluation stage of one dataset."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    source = dataset.stage(STAGE_KEY)
    data = Inputs(config, dataset)
    ctx = DatasetContext.load(dataset, config, stage_key=STAGE_KEY)
    output = ctx.output
    outputs: list[Path] = []
    written: dict[str, Any] = {}
    written["agreement"] = write_agreement_tables(config, data, source_stage=source)
    written["decision"] = write_decision_tables(config, data, source_stage=source)
    written["strata"] = write_strata_table(config, data, source_stage=source)
    extra: dict[str, Any] = {"dataset": dataset.name, "network": dataset.primary, "smoke": smoke}
    if placebo:
        result = run_placebo(ctx, dataset.primary)
        written["placebo"] = write_placebo_tables(ctx, result, source_stage=source)
        seeds_path = output / f"placebo_seed_sets_{dataset.primary}.json"
        seeds_path.write_text(json.dumps(result["seed_sets"]))
        outputs.append(seeds_path)
        extra["placebo"] = {"n_permutations": int(config.n_placebo_permutations), "summary": result["summary"]}
    if downstream:
        raw = crossfit_scores(ctx, data, dataset.primary)
        feats = feature_frame(ctx, data, dataset.primary, raw)
        rows = evaluate_feature_sets(ctx, data, dataset.primary, feats["features"])
        importance = feature_importance(ctx, dataset.primary)
        written["downstream"] = write_downstream_tables(ctx, rows, importance, feats["diagnostics"], inputs=data.inputs,
                                                        source_stage=source)
        extra["downstream"] = {"diagnostics": feats["diagnostics"], "bound": written["downstream"]["bound_row"]}
    for group in written.values():
        items = group.values() if isinstance(group, dict) else [group]
        outputs += [item for item in items if isinstance(item, Path)]
    manifest = write_manifest(output, arm_name=source, config=config, inputs=list(data.inputs), outputs=outputs,
                              seeds={"master_seed": config.master_seed, "placebo_seeds": [config.seed_placebo(p) for p in range(int(config.n_placebo_permutations))],
                                     "fold_seed": config.seed_folds()},
                              timer=timer, execution_mode="simulation", dataset=dataset, extra=extra)
    return {"written": written, "artifacts": [*outputs, manifest], "extra": extra}
