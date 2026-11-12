"""Stage 04: the limit, ``tau*`` and both arms' scores per network, the reduction check, the convergence tables.

Per network: the full-operator reduction-check walk on the production
operator (stage 03's masked frame, operator identity asserted), then the
routed lumped operator, its limit, the running average to ``tau_max``,
``tau*``, the scores file, the reduction residuals and the convergence
tables. The unlumped step check is run for every HMEQ network and for the
Freddie Mac primary network (``FULL_CHECK``); elsewhere the residuals alone
are reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from config import ExperimentConfig
from datasets import get_dataset
from evaluation.context import DatasetContext
from evaluation.provenance import RunTimer, write_stage_manifest
from fingerprint import resolve_config
from quantum.inspired import stage as walks
from quantum.mechanism.convergence import write_convergence_tables
from quantum.mechanism.scores_stage import (
    SCORE_COLUMN_DEFINITIONS,
    compute_network,
    write_network_artifacts,
    write_reduction_table,
)

STAGE_KEY = "quantum"
# Networks whose unlumped operator is walked step by step against the lumped one.
FULL_CHECK = {"hmeq": ("NET_JOB", "NET_REASON", "NET_2L"), "fmac": ("NET_FMAC_2L",)}


def run(dataset_name: str, config: ExperimentConfig | None = None, *, networks=None, run_full_walks: bool = True) -> dict[str, Any]:
    """Run the scoring stage of one dataset; ``networks`` restricts the ladder (an array task runs one)."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    ctx = DatasetContext.load(dataset, config, stage_key=STAGE_KEY)
    output = ctx.output
    ladder = list(networks or dataset.ladder)
    outputs: list[Path] = []
    walk_records: dict[str, Any] = {}
    if run_full_walks:
        classical = walks.load_classical_digests(stage=dataset.stage("classical"))
        quantum = walks.load_quantum_digests(output)
        for network in ladder:
            walk_records[network] = walks.reduction_check_walk(network, config, classical, quantum, output,
                                                              specs=ctx.specs, stage=dataset.stage(STAGE_KEY))
        outputs.append(walks.write_quantum_digests(output, quantum))
    results = {}
    records, curves, sensitivity, reduction_rows = [], [], [], []
    for network in ladder:
        check_full = network in FULL_CHECK.get(dataset.name, ())
        result = compute_network(ctx, network, check_full=check_full)
        results[network] = result
        outputs += write_network_artifacts(ctx, result)
        records.append(result.convergence)
        curves += result.curves
        sensitivity += result.sensitivity
        reduction_rows.append(result.reduction)
    tables = write_convergence_tables(config, dataset, records, curves, sensitivity, inputs=ctx.inputs,
                                      source_stage=dataset.stage(STAGE_KEY))
    reduction = write_reduction_table(reduction_rows, dataset=dataset.name, source_stage=dataset.stage(STAGE_KEY))
    outputs += [*tables.values(), reduction]
    summary = {network: {"tau_star": r.convergence["tau_star"], "reached": r.convergence["reached"],
                         "route": r.convergence["route"], "R": r.convergence["R"],
                         "levels_q": r.convergence["levels_q"], "levels_c": r.convergence["levels_c"],
                         "full_vs_reduced_error": r.reduction["full_vs_reduced_error"],
                         "full_vs_reduced_error_ranker": r.reduction["full_vs_reduced_error_ranker"]}
               for network, r in results.items()}
    manifest = write_stage_manifest(dataset, "quantum", config, outputs=outputs, timer=timer, output_dir=output,
                                    extra={"quantum_score": config.quantum_score, "networks_run": ladder, "summary": summary,
                                           "reduction_check_walks": walk_records, "score_columns": SCORE_COLUMN_DEFINITIONS,
                                           "smoke": smoke})
    (output / "stage_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {"results": results, "summary": summary, "tables": tables, "reduction": reduction,
            "artifacts": [*outputs, manifest], "frame": pl.DataFrame(reduction_rows, strict=False, infer_schema_length=None)}
