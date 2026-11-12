"""Stage 05a: the mechanism on the real networks -- the eta sweep, the variants, the partially observed, the single layers.

Runs on the lumped operators only, through the router, for one dataset:
``T_eta`` (Figure 1), ``T3_variants`` with the classical grid, ``T4_partial``
with the label-free control and the placebo Jaccards, and
``EC_single_layer_share`` for the single-layer networks.

Stage 05 (evaluation) comes first: ``T4_partial`` prints the placebo Jaccard and
p-value that stage 05 writes to ``EC_placebo``, and this stage raises
``FileNotFoundError`` when that table is absent instead of leaving the columns empty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from config import DEFAULT_MECHANISM, ExperimentConfig, MechanismConfig
from datasets import get_dataset
from evaluation.context import DatasetContext
from evaluation.provenance import RunTimer, read_result, write_manifest
from fingerprint import resolve_config
from quantum.mechanism.single_layer_share import single_layer_rows, write_single_layer_table
from quantum.mechanism.studies import (
    eta_rows,
    partial_rows,
    variant_rows,
    write_eta_table,
    write_partial_table,
    write_variant_tables,
)

STAGE_KEY = "mechanism"


def _placebo_summary(dataset) -> list[dict[str, Any]]:
    """Stage 05's placebo summary of this dataset. Table 4 prints its Jaccard and p-value, so this stage cannot run without it."""
    frame = read_result(dataset.name, "EC_placebo")
    if frame is None:
        raise FileNotFoundError(f"stage 05 (evaluation) has not written EC_placebo for {dataset.name}; run stage 05 first")
    return frame.to_dicts()


def run(dataset_name: str, config: ExperimentConfig | None = None, mech: MechanismConfig | None = None) -> dict[str, Any]:
    """Run the mechanism stage of one dataset."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    # The smoke trim keeps both ends of the eta grid and two damping values (the frozen alpha among them).
    mech = mech or (MechanismConfig(eta_grid=(0.0, 1.0), alpha_grid=(0.5, float(config.resolved_alpha())), smoke=True)
                    if smoke else DEFAULT_MECHANISM)
    dataset = get_dataset(dataset_name)
    placebo = _placebo_summary(dataset)      # stage 05 must have finished: refuse before any work, never continue without it
    timer = RunTimer()
    ctx = DatasetContext.load(dataset, config, stage_key=STAGE_KEY)
    source = dataset.stage(STAGE_KEY)
    primary = dataset.primary
    outputs: list[Path] = []
    rows_eta, scored_eta = eta_rows(ctx, primary, mech)
    eta_table = write_eta_table(ctx, rows_eta, mech=mech, source_stage=source)
    rows_var, grid, grid_summary = variant_rows(ctx, primary, mech, scored_eta)
    var_tables = write_variant_tables(ctx, rows_var, grid, mech=mech, source_stage=source)
    rows_partial, label_free = partial_rows(ctx, primary, placebo)
    partial_table = write_partial_table(ctx, rows_partial, source_stage=source)
    single_rows, single_summaries = [], {}
    for network in dataset.ladder:
        if ctx.specs[network].n_layers == 1:
            rows, summary = single_layer_rows(ctx, network)
            single_rows += rows
            single_summaries[network] = {k: v for k, v in summary.items() if k != "pairs"}
            single_summaries[network]["pairs"] = summary["pairs"]
    single_table = write_single_layer_table(ctx, single_rows, source_stage=source)
    outputs += [eta_table, var_tables["variants"], var_tables["grid"], partial_table, single_table]
    extra = {"dataset": dataset.name, "network": primary, "mechanism": mech.as_dict(), "smoke": smoke,
             "grid_summary": grid_summary, "label_free": label_free, "single_layer": single_summaries,
             "eta_routes": {str(r["eta"]): r["route"] for r in rows_eta}}
    (ctx.output / "mechanism_summary.json").write_text(json.dumps(extra, indent=2, default=str))
    manifest = write_manifest(ctx.output, arm_name=source, config=config, inputs=ctx.inputs, outputs=outputs,
                              seeds={"master_seed": config.master_seed}, timer=timer, execution_mode="simulation", dataset=ctx.dataset,
                              extra=extra)
    return {"eta": rows_eta, "variants": rows_var, "partial": rows_partial, "single_layer": single_rows,
            "summary": extra, "artifacts": [*outputs, manifest]}
