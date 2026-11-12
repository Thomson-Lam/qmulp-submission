"""What both datasets' stage 01 shares after the cleaning: the frozen split record, the data-stage files and Table 1.

The cleaning itself is dataset-specific and lives in the dataset's package
(``hmeq.cleaning``; ``fmac.cleaning`` with its notebook). Each dataset builds
its own frozen split record (``hmeq.cleaning.HMEQCleaningResult.frozen_split_record``,
``fmac.cleaning.frozen_split_record``) under the one schema name
:data:`evaluation.masked_eval.FROZEN_SPLIT_SCHEMA` that
:func:`evaluation.masked_eval.load_partitions` validates;
:func:`write_frozen_split` is the one writer of that record.

What follows the cleaning is the same for both: one outcome-stratified
resample matrix over the test partition, drawn once from the master seed, and
one per-borrower observation record.

Table 1 inputs: what each dataset is, counted on the frozen artifacts. One row
per quantity and dataset: individuals, the split, the reference set, the test
positives, prevalence, the layers and their levels, the individuals missing a
category per layer, the partially observed test individuals, the records
dropped before the split. Every count is read off the processed frame, the
frozen split and the observation flags, never typed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config import ExperimentConfig
from datasets import DatasetConfig
from evaluation.masked_eval import Partitions
from evaluation.provenance import write_result
from fingerprint import bootstrap_indices


# ---------------------------------------------------------------------------
# The frozen split record
# ---------------------------------------------------------------------------


def write_frozen_split(record: dict[str, Any], path: str | Path) -> Path:
    """Write the split record; every later stage reads it and none redraws it."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1, sort_keys=True))
    return output


# ---------------------------------------------------------------------------
# The bootstrap matrix and the observation flags
# ---------------------------------------------------------------------------


def write_bootstrap_matrix(
    config: ExperimentConfig, partitions: Partitions, dataset: DatasetConfig
) -> tuple[Path, np.ndarray]:
    """The one shared ``(reps, n_test)`` resample matrix of ``dataset``, drawn from ``config.seed_bootstrap()``.

    Every interval in every table of that dataset is computed on it, so
    differences between arms are paired by construction. Each dataset has its
    own file: the matrices are drawn over different test partitions and must
    never be read in each other's place.
    """
    y_test = partitions.y_true[partitions.test_idx]
    indices = bootstrap_indices(y_test, config.bootstrap_reps, config.seed_bootstrap())
    path = dataset.bootstrap_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, indices)
    return path, indices


def write_observation_flags(partitions: Partitions, flags: dict[str, Any], output: Path) -> Path:
    """``observation_flags.csv``: the four per-individual flags and the blank-field count."""
    path = output / "observation_flags.csv"
    pl.DataFrame(
        {
            "borrower_id": np.arange(partitions.frame.height, dtype=int),
            "partition": partitions.partition,
            "blank_field": np.asarray(flags["blank_field"], dtype=bool),
            "blank_field_count": np.asarray(flags["blank_field_count"], dtype=int),
            "structural_none": np.asarray(flags["structural_none"], dtype=bool),
            "unseen_level": np.asarray(flags["unseen_level"], dtype=bool),
            "partially_observed": np.asarray(flags["partially_observed"], dtype=bool),
            **{f"missing_{layer}": np.asarray(mask, dtype=bool) for layer, mask in flags["missing_per_layer"].items()},
        }
    ).write_csv(path)
    return path


# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------


TABLE_NAME = "T1_data"


def split_description(dataset: DatasetConfig, config: ExperimentConfig, record: dict[str, Any]) -> str:
    if dataset.split == "stratified":
        return (f"one stratified {int(round(100 * (1 - config.test_frac)))}/{int(round(100 * config.test_frac))} "
                f"split on the outcome, seed {record.get('split_seed', config.split_seed)}")
    return (f"temporal: development = first payment {record.get('development_cohort')}, "
            f"test = first payment {record.get('test_cohort')}")


def data_rows(dataset: DatasetConfig, config: ExperimentConfig, partitions: Partitions, flags: dict[str, Any],
              *, n_raw: int | None, n_dropped: int | None) -> list[dict[str, Any]]:
    """The Table 1 quantities of one dataset as ``(dataset, key, value, note)`` rows."""
    y = partitions.y_true
    dev, test = partitions.development_idx, partitions.test_idx
    record = json.loads(Path(partitions.split_path).read_text()) if partitions.split_path else {}
    if n_raw is None:
        raw = record.get("n_raw_rows")
        n_raw = int(sum(raw.values())) if isinstance(raw, dict) else (int(raw) if raw is not None else None)
    if n_dropped is None and record.get("n_unconnectable_dropped") is not None:
        n_dropped = int(record["n_unconnectable_dropped"])
    specs = dataset.network_specs(partitions.frame)
    layers = [layer for layer in specs[dataset.primary].layers]
    missing = flags["missing_per_layer"]
    rows = [
        ("n_raw", n_raw, "records in the supplied file(s)"),
        ("n_dropped", n_dropped, "records dropped before the split (no category in any layer)"),
        ("n_borrowers", partitions.frame.height, "individuals retained"),
        ("n_dev", int(dev.size), "development partition"),
        ("n_test", int(test.size), "test partition"),
        ("n_ref", int(y[dev].sum()), "development references (the reference set)"),
        ("n_test_pos", int(y[test].sum()), "test positives (masked from every operator)"),
        ("prev_dev", float(y[dev].mean()), "development prevalence"),
        ("prev_test", float(y[test].mean()), "test prevalence"),
        ("n_layers", len(layers), "layers of the two-layer network"),
    ]
    for i, layer in enumerate(layers, start=1):
        rows.append((f"layer_l{i}", layer.name, "layer name"))
        rows.append((f"levels_l{i}", int(layer.n_states), "category levels (development vocabulary)"))
        rows.append((f"n_missing_l{i}", int(np.asarray(missing[layer.name]).sum()),
                     f"individuals with no category in {layer.name} ({flags['layer_absence_means']})"))
        rows.append((f"n_missing_l{i}_test", int(np.asarray(missing[layer.name])[test].sum()),
                     f"test individuals with no category in {layer.name}"))
    partial = np.asarray(flags["partially_observed"], dtype=bool)
    rows += [
        ("n_partial", int(partial.sum()), "partially observed individuals (no edge in at least one layer)"),
        ("n_partial_test", int(partial[test].sum()), "partially observed test individuals"),
        ("n_blank_field_test", int(np.asarray(flags["blank_field"], dtype=bool)[test].sum()),
         "test individuals with at least one blank field"),
        ("n_unseen_level_test", int(np.asarray(flags["unseen_level"], dtype=bool)[test].sum()),
         "test individuals carrying a level outside the development vocabulary"),
        ("n_structural_none_test", int(np.asarray(flags["structural_none"], dtype=bool)[test].sum()),
         "test individuals with a structural none (not a blank)"),
        ("split", split_description(dataset, config, record), "how the partitions were formed"),
        ("split_seed", record.get("split_seed"), "seed of the split, null for a temporal split"),
    ]
    return [{"dataset": dataset.name, "key": key, "value": value, "note": note} for key, value, note in rows]


def write_data_table(rows, *, dataset: DatasetConfig, source_stage: str) -> Path:
    """``outputs/results/<dataset>/T1_data.csv``: one ``(dataset, key, value, note)`` row per quantity.

    Each dataset writes its own file; the manuscript assembler reads both
    scopes and never a file of the other dataset.
    """
    frame = pl.DataFrame([{**r, "value": str(r["value"])} for r in rows])
    return write_result(
        dataset.name, TABLE_NAME, frame, stage=source_stage,
        columns={"dataset": "dataset name", "key": "quantity name (see the note column)",
                 "value": "the counted value, as text (counts, rates, names and the split description share one column)",
                 "note": "what the quantity is"},
    )


def expected_counts_note(rows, expectations: dict[tuple[str, str], Any]) -> list[str]:
    """Human-readable differences between the counted rows and the expected values the plan records."""
    lookup = {(r["dataset"], r["key"]): r["value"] for r in rows}
    notes = []
    for key, expected in expectations.items():
        observed = lookup.get(key)
        if observed != expected:
            notes.append(f"{key[0]}.{key[1]}: expected {expected}, counted {observed}")
    return notes
