"""Stage 02: what each network is -- sizes, degrees, level ceiling, lumped classes -- and the exact-lumpability check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from classical import network
from classical.network_diagnostics import RESOURCE_PLAN_STATUS
from config import ExperimentConfig
from datasets import get_dataset
from evaluation.provenance import RunTimer, write_stage_manifest
from fingerprint import load_frozen_partitions, resolve_config

STAGE_KEY = "network"


def run(dataset_name: str, config: ExperimentConfig | None = None, *, calibrate: bool = True) -> dict[str, Any]:
    """Run the network stage of one dataset."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    output = dataset.stage_dir(STAGE_KEY)
    partitions = load_frozen_partitions(dataset)
    specs = dataset.network_specs(partitions.frame)
    observed_states = dataset.assert_expected_states(specs)
    masked_levels = partitions.masked_labels(partitions.test_idx)
    built: dict[str, Any] = {}
    files: list[Path] = []
    for name in dataset.ladder:
        built[name] = network.describe_one_network(partitions, specs[name], config, masked_levels, dataset)
        files += network.write_network_files(output, name, built[name]["facts"], built[name]["nodes"], built[name]["edges"])
    summary = pl.DataFrame([network.network_summary_row(specs[n], built[n]["facts"], partitions.frame.height) for n in dataset.ladder])
    check = network.lumpability_check(partitions, specs[dataset.primary], config, built[dataset.primary]["nodes"], built[dataset.primary]["edges"])
    if not check["passed"]:
        raise AssertionError(f"exact lumpability failed: {check}")
    calibration = network.calibrate_step_constant(config, dataset, [specs[n] for n in dataset.ladder], n_steps=3) if calibrate else None
    spec_facts = {name: built[name]["facts"] for name in dataset.ladder}
    inputs = [dataset.processed_path(), dataset.frozen_split_path()]
    artifacts = network.write_network_artifacts(output, spec_facts, check, summary, [specs[n] for n in dataset.ladder], partitions.frame.height)
    artifacts.append(network.write_edge_coverage(output, built, dataset.ladder))
    table = network.write_network_summary_table(config, dataset, summary, inputs)
    artifacts += [table, *files]
    manifest = write_stage_manifest(dataset, "network", config, outputs=artifacts, timer=timer, execution_mode="analysis",
                                    extra={"lumpability_check": check, "smoke": smoke,
                                           "resource_plan_status": RESOURCE_PLAN_STATUS, "step_time_calibration": calibration,
                                           "layer_states": observed_states})
    return {"summary": summary, "check": check, "calibration": calibration, "artifacts": [*artifacts, manifest]}
