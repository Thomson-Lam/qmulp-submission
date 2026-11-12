"""Freddie Mac cleaning helpers: sanitised level names, the pooled frame, the frozen split.

The cleaning notebook encodes each origination cohort; these helpers turn the
two encoded cohorts into the artifacts every later stage reads and none
recomputes: the pooled network frame (2023 first, so a row position *is* the
``borrower_id``) and the frozen split record in the schema
:func:`evaluation.masked_eval.load_partitions` already validates.

The split is temporal, not drawn: nothing here randomises it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from evaluation.masked_eval import FROZEN_SPLIT_SCHEMA
from fingerprint import sha256_path
from fmac.config import DATASET, DEVELOPMENT_COHORT, TEST_COHORT
from paths import relative_to_repo
from preprocessing.missing_levels import (
    MISSING_INDICATOR_SUFFIX,
    MISSING_SENTINEL,
    split_missing_columns,
)

# The label column name the operator reads. ``classical.mulp`` and
# ``quantum.lumping`` both look for "BAD", so the pooled frame carries the
# outcome under that name; the raw column is called "target".
LABEL_COL = DATASET.label_col
SOURCE_LABEL_COL = "target"


# ---------------------------------------------------------------------------
# One-hot column names
# ---------------------------------------------------------------------------


def sanitize_level(value: Any) -> str:
    """A one-hot column suffix that survives being read back out of a name.

    Upper-cases and collapses every run of non-alphanumeric characters into one
    underscore, so a seller's legal name becomes a stable token. The reserved
    absence level passes through untouched: the repository identifies an
    absence marker by the exact ``_Missing`` suffix
    (:mod:`preprocessing.missing_levels`), so a level that sanitised to
    ``MISSING`` would silently become a category node.
    """
    text = str(value)
    if text == MISSING_SENTINEL:
        return MISSING_SENTINEL
    token = "_".join(
        part
        for part in "".join(ch if ch.isalnum() else "_" for ch in text.upper()).split("_")
        if part
    )
    if not token:
        raise ValueError(f"level {value!r} sanitises to an empty column suffix")
    if token.upper() == MISSING_SENTINEL.upper():
        raise ValueError(f"level {value!r} collides with the reserved {MISSING_SENTINEL!r} level")
    return token


def sanitized_levels(levels) -> dict[str, str]:
    """``{level: column suffix}``, refusing a sanitisation that merges two levels.

    Two seller names that differ only in punctuation would otherwise become one
    category node, silently linking borrowers who share nothing.
    """
    mapping = {str(level): sanitize_level(level) for level in levels}
    if len(set(mapping.values())) != len(mapping):
        seen: dict[str, str] = {}
        merged = []
        for level, token in mapping.items():
            if token in seen:
                merged.append((seen[token], level))
            seen[token] = level
        raise ValueError(f"sanitising merges distinct levels: {merged}")
    return mapping


# ---------------------------------------------------------------------------
# Pooling the two cohorts into one frame
# ---------------------------------------------------------------------------


def pool_cohorts(
    development: pd.DataFrame, test: pd.DataFrame
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """One frame, development cohort first, plus the two partition index arrays.

    The outcome is renamed to :data:`LABEL_COL` in place, keeping its position
    as the last column. Row order is the contract: ``borrower_id`` is the row
    position, so the partitions are contiguous ranges by construction.
    """
    if list(development.columns) != list(test.columns):
        raise ValueError("the two cohorts must carry identical columns in identical order")
    if SOURCE_LABEL_COL not in development.columns:
        raise ValueError(f"neither cohort carries the outcome column {SOURCE_LABEL_COL!r}")
    pooled = pd.concat([development, test], ignore_index=True, copy=True)
    pooled = pooled.rename(columns={SOURCE_LABEL_COL: LABEL_COL})
    development_idx = np.arange(len(development), dtype=int)
    test_idx = np.arange(len(development), len(pooled), dtype=int)
    return pl.from_pandas(pooled), development_idx, test_idx


def layer_genuine_columns(
    frame: pl.DataFrame, layer_names=DATASET.layer_order
) -> dict[str, list[str]]:
    """Genuine (non-absence) one-hot columns per layer, in state order."""
    columns: dict[str, list[str]] = {}
    for name in layer_names:
        prefix = f"{name}_"
        found = sorted(
            c
            for c in frame.columns
            if c.startswith(prefix) and not c.endswith(MISSING_INDICATOR_SUFFIX)
        )
        genuine, _markers = split_missing_columns(found)
        if not genuine:
            raise ValueError(f"layer {name!r} has no genuine category columns in the frame")
        columns[name] = genuine
    return columns


def unconnectable_mask(frame: pl.DataFrame, layer_names=DATASET.layer_order) -> np.ndarray:
    """True where a borrower has no genuine category in any layer.

    Such a borrower touches no category node, so it earns no edge anywhere and
    is a dangling column in every operator built from the frame. HMEQ drops the
    same rows once, in ``hmeq.cleaning.drop_unconnectable``.
    """
    columns = layer_genuine_columns(frame, layer_names)
    flat = [c for cols in columns.values() for c in cols]
    active = frame.select(flat).to_numpy().astype(int)
    return ~active.astype(bool).any(axis=1)


def drop_unconnectable(
    frame: pl.DataFrame,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    layer_names=DATASET.layer_order,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, list[int]]:
    """Drop unconnectable rows and renumber both partitions onto the kept rows."""
    mask = unconnectable_mask(frame, layer_names)
    dropped = [int(i) for i in np.flatnonzero(mask)]
    if not dropped:
        return frame, np.asarray(development_idx, dtype=int), np.asarray(test_idx, dtype=int), []
    keep = ~mask
    remap = np.full(frame.height, -1, dtype=int)
    remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()), dtype=int)
    kept = frame.filter(pl.Series("keep", keep))
    development = remap[np.asarray(development_idx, dtype=int)]
    test = remap[np.asarray(test_idx, dtype=int)]
    return kept, development[development >= 0], test[test >= 0], dropped


# ---------------------------------------------------------------------------
# The pooled frame and the frozen split
# ---------------------------------------------------------------------------


def write_network_frame(frame: pl.DataFrame, path: str | Path) -> tuple[Path, str]:
    """Write the pooled frame and return its path and digest."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output)
    return output, sha256_path(output)


def frozen_split_record(
    frame: pl.DataFrame,
    development_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    processed_path: str | Path,
    processed_sha256: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The split record, in the schema ``load_partitions`` already validates.

    ``split_seed`` is null and ``stratify`` is the cohort field rather than the
    outcome: this split is the calendar, not a draw, so there is no seed that
    would reproduce it and nothing to stratify.
    """
    development_idx = np.asarray(development_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    y = frame[LABEL_COL].to_numpy().astype(int)
    record: dict[str, Any] = {
        "schema": FROZEN_SPLIT_SCHEMA,
        "dataset": DATASET.name,
        "split": DATASET.split,
        "stratify": None,
        "split_seed": None,
        "development_cohort": DEVELOPMENT_COHORT,
        "test_cohort": TEST_COHORT,
        "n_rows": int(frame.height),
        "n_development": int(development_idx.size),
        "n_test": int(test_idx.size),
        "test_frac": float(test_idx.size / frame.height),
        "n_defaults": int(y.sum()),
        "n_development_defaults": int(y[development_idx].sum()),
        "n_test_defaults": int(y[test_idx].sum()),
        "development_idx": [int(i) for i in development_idx],
        "test_idx": [int(i) for i in test_idx],
        "processed": relative_to_repo(processed_path),
        "processed_sha256": processed_sha256,
    }
    if extra:
        record.update(extra)
    return record
