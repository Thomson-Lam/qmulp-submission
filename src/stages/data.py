"""Stage 01: the processed frame, the frozen split, the shared bootstrap, the observation flags, Table 1.

HMEQ is cleaned here (``hmeq.cleaning``); the Freddie Mac extract is encoded
by its cleaning notebook cell by cell, so for ``fmac`` this stage runs only
what follows the cleaning and refuses to start before the processed frame
and the frozen split exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from config import ExperimentConfig
from datasets import get_dataset
from evaluation.provenance import RunTimer, write_manifest
from fingerprint import load_frozen_partitions, observation_flags, resolve_config, sha256_path
from preprocessing.common import (
    data_rows,
    expected_counts_note,
    write_bootstrap_matrix,
    write_data_table,
    write_observation_flags,
)

STAGE_KEY = "data"
# Counts the plan records for the two datasets; a difference is reported, never adjusted.
EXPECTED: dict[tuple[str, str], Any] = {
    ("hmeq", "n_raw"): 5960, ("hmeq", "n_dropped"): 107, ("hmeq", "n_borrowers"): 5853,
    ("hmeq", "n_dev"): 4682, ("hmeq", "n_test"): 1171, ("hmeq", "n_ref"): 943, ("hmeq", "n_test_pos"): 236,
    ("hmeq", "n_missing_l1"): 172, ("hmeq", "n_missing_l2"): 145, ("hmeq", "n_partial_test"): 58,
    ("fmac", "n_dev"): 3045, ("fmac", "n_test"): 3199, ("fmac", "n_ref"): 91, ("fmac", "n_test_pos"): 103,
    ("fmac", "levels_l1"): 24, ("fmac", "levels_l2"): 19, ("fmac", "n_partial_test"): 105,
}


def run(dataset_name: str, config: ExperimentConfig | None = None) -> dict[str, Any]:
    """Run the data stage of one dataset and return its artifacts, counts and expectation notes."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    output = dataset.stage_dir(STAGE_KEY)
    extra: dict[str, Any] = {"dataset": dataset.name, "split": dataset.split, "smoke": smoke}
    inputs: list[Path] = []
    outputs: list[Path] = []
    if dataset.name == "hmeq":
        from hmeq import cleaning as hmeq_stage
        from hmeq.config import RAW_CSV

        result = hmeq_stage.clean_and_freeze(config, RAW_CSV)
        counts = hmeq_stage.observed_counts(result)
        hmeq_stage.assert_expected_counts(counts)
        inputs.append(RAW_CSV)
        extra.update(counts=counts, raw_sha256=result.raw_sha256, processed_sha256=result.processed_sha256,
                     imputation_medians=result.imputation_medians, missing_flag_columns=result.missing_flag_columns)
        outputs.append(hmeq_stage.write_cleaning_report(result, output))
    else:
        for path in (dataset.processed_path(), dataset.frozen_split_path()):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing; run notebooks/01_fmac_cleaning.ipynb first")
        from fmac.config import raw_csvs

        try:
            raws = raw_csvs()
        except FileNotFoundError:
            raws = {}                                   # the digests are then recorded as absent, never invented
        inputs += list(raws.values())
        extra.update(raw_sha256={cohort: sha256_path(path) for cohort, path in raws.items()} or None,
                     processed_sha256=sha256_path(dataset.processed_path()))
    partitions = load_frozen_partitions(dataset)
    observed_states = dataset.assert_expected_states(dataset.network_specs(partitions.frame))
    bootstrap_path, indices = write_bootstrap_matrix(config, partitions, dataset)
    flags = observation_flags(partitions.frame, dataset)
    flags_path = write_observation_flags(partitions, flags, output)
    rows = data_rows(dataset, config, partitions, flags, n_raw=None, n_dropped=None)
    notes = expected_counts_note(rows, {k: v for k, v in EXPECTED.items() if k[0] == dataset.name})
    table = write_data_table(rows, dataset=dataset, source_stage=dataset.stage(STAGE_KEY))
    outputs += [dataset.processed_path(), dataset.frozen_split_path(), bootstrap_path, flags_path, table]
    extra.update(layer_states=observed_states, expectation_differences=notes,
                 bootstrap={"reps": int(indices.shape[0]), "n_test": int(indices.shape[1]), "stratified": True,
                            "seed": config.seed_bootstrap()},
                 flag_counts={name: int(sum(bool(x) for x in flags[name])) for name in
                              ("blank_field", "structural_none", "unseen_level", "partially_observed")},
                 blank_field_columns=flags["blank_field_columns"], unseen_level_columns=flags["unseen_level_columns"])
    manifest = write_manifest(output, arm_name=dataset.stage(STAGE_KEY), config=config, inputs=inputs, outputs=outputs,
                              seeds={"split_seed": config.split_seed, "bootstrap_seed": config.seed_bootstrap()},
                              timer=timer, execution_mode="preprocessing", dataset=dataset, extra=extra)
    return {"rows": pl.DataFrame(rows), "notes": notes, "artifacts": [*outputs, manifest], "flags": flags}
