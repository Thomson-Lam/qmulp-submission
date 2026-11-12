"""HMEQ's stage 01: one processed frame, one frozen split, in an enforced order, and the counts it must reproduce.

Three artifacts every later stage reads and none recomputes:

- ``data/hmeq_processed.parquet`` -- one row per retained borrower; the
  row position *is* the ``borrower_id`` every graph index is built from;
- ``outputs/frozen_split.json`` -- borrower indices per partition, counts, the
  raw checksum and the seed;
- ``outputs/bootstrap_indices.npy`` -- the one ``(reps, n_test)`` paired,
  outcome-stratified resample matrix every table's interval is computed on
  (drawn by ``preprocessing.common.write_bootstrap_matrix``).

The cleaning record
-------------------
:class:`HMEQCleaningResult` is the record one cleaning run produces: the
processed frame, its frozen split, and every fitted quantity. The frozen split
file and the data stage's cleaning report are written from it.

The cleaning
------------
The order of operations, from the raw CSV, is the contract every downstream
stage relies on and is executed exactly as listed by :func:`clean_hmeq_frame`:

1. **Load** the raw CSV and record its SHA-256.
2. **Drop unconnectable borrowers** -- ``JOB`` *and* ``REASON`` both missing.
   Such a borrower cannot attach to either layer: its layer copies would form
   an isolated pair that no walk can reach and whose own mass can never leave.
   Removed before anything else, so the split proportions stay exact and the
   drop does not depend on the seed. Count and raw indices are recorded.
3. **Stratified split** on ``BAD``, 80/20, one fixed seed. Written once to
   ``outputs/frozen_split.json``; every later stage reads that file and no
   stage recomputes a split.
4. **Median imputation, fit on development, applied to both.** Every numeric
   column with a missing value gets the *development* median and a
   ``<COL>_was_missing`` flag. The flag is a first-class covariate and survives
   into the downstream feature matrix. The fit/apply boundary is two separate
   functions, :func:`fit_medians` and :func:`apply_imputation`.
5. **One-hot ``JOB`` and ``REASON``.** A borrower missing exactly one of the two
   keeps its edge in the other layer and has no edge in the missing one. The
   ``<LAYER>_Missing`` column is written so the encoding stays one-hot, but it
   is an absence marker, never a category node (``preprocessing.missing_levels``).

Step 3 precedes step 4 in code although the specification lists imputation
first: the medians are development-only quantities and cannot be fitted before
the development partition exists.

Vocabulary: ``partition`` takes the values ``"development"`` and ``"test"``.
The seed set and the scoring cohort are cross-fitting devices applied *within*
the development partition; see ``evaluation.masked_eval``.

The data stage
--------------
The cleaning steps ``stages.data`` calls for HMEQ, and the counts the cleaner
must reproduce. Two missing-value treatments coexist and are not
interchangeable. **Network layers**: a blank JOB or REASON earns no edge; it is
neither imputed nor a category node. **Downstream covariates**: median-imputed
on the development partition with a retained ``<col>_was_missing`` flag per
imputed column.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split

from config import ExperimentConfig
from evaluation.masked_eval import FROZEN_SPLIT_SCHEMA
from fingerprint import sha256_path
from hmeq.config import DATASET, JOB_COLUMNS, NUMERIC_COLS, RAW_CSV, REASON_COLUMNS
from paths import assert_no_absolute_paths, relative_to_repo
from preprocessing.common import write_frozen_split
from preprocessing.missing_levels import MISSING_INDICATOR_SUFFIX, MISSING_SENTINEL, assert_absence_consistent


# ---------------------------------------------------------------------------
# The cleaning record
# ---------------------------------------------------------------------------


TARGET_COL = "BAD"


@dataclass(frozen=True)
class HMEQCleaningResult:
    """One processed frame over every retained borrower, plus its frozen split.

    ``frame`` row position *is* the ``borrower_id`` used throughout the graph
    index, so no ``borrower_id`` column is added to it -- that would leak a row
    index into every downstream covariate matrix. The mapping lives in the
    frozen split instead.
    """

    frame: pd.DataFrame
    partition: np.ndarray
    development_idx: np.ndarray
    test_idx: np.ndarray
    # Every fitted quantity below is estimated on the development partition only.
    imputation_medians: dict[str, float]
    missing_flag_columns: list[str]
    categorical_levels: dict[str, list[str]]
    n_missing_per_column: dict[str, int]
    n_raw_rows: int
    n_unconnectable_dropped: int
    dropped_raw_indices: list[int]
    raw_sha256: str | None
    test_frac: float
    split_seed: int
    processed_path: Path | None = None
    frozen_split_path: Path | None = None
    processed_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(len(self.frame))

    def default_counts(self) -> dict[str, int]:
        y = self.frame[TARGET_COL].to_numpy().astype(int)
        return {
            "n_defaults": int(y.sum()),
            "n_development_defaults": int(y[self.development_idx].sum()),
            "n_test_defaults": int(y[self.test_idx].sum()),
        }

    def frozen_split_record(self) -> dict[str, Any]:
        """Everything a downstream stage needs to reproduce the partition exactly."""
        counts = self.default_counts()
        return {
            "schema": FROZEN_SPLIT_SCHEMA,
            "raw_sha256": self.raw_sha256,
            "n_raw_rows": int(self.n_raw_rows),
            "n_unconnectable_dropped": int(self.n_unconnectable_dropped),
            "dropped_raw_indices": [int(i) for i in self.dropped_raw_indices],
            "n_rows": self.n_rows,
            "stratify": TARGET_COL,
            "test_frac": float(self.test_frac),
            "split_seed": int(self.split_seed),
            "n_development": int(self.development_idx.size),
            "n_test": int(self.test_idx.size),
            **counts,
            "development_idx": [int(i) for i in self.development_idx],
            "test_idx": [int(i) for i in self.test_idx],
            "processed": relative_to_repo(self.processed_path) if self.processed_path else None,
            "processed_sha256": self.processed_sha256,
            "imputation_medians": {k: float(v) for k, v in self.imputation_medians.items()},
            "missing_flag_columns": list(self.missing_flag_columns),
            "n_missing_per_column": {k: int(v) for k, v in self.n_missing_per_column.items()},
            "categorical_levels": {k: list(v) for k, v in self.categorical_levels.items()},
        }

    def report(self) -> pd.DataFrame:
        """A ``(field, value)`` provenance table with repo-relative paths only."""
        counts = self.default_counts()
        rows = [
            (
                "processed_path",
                relative_to_repo(self.processed_path) if self.processed_path else "",
            ),
            (
                "frozen_split_path",
                relative_to_repo(self.frozen_split_path) if self.frozen_split_path else "",
            ),
            ("raw_sha256", self.raw_sha256 or ""),
            ("n_raw_rows", str(self.n_raw_rows)),
            ("n_unconnectable_dropped", str(self.n_unconnectable_dropped)),
            ("n_retained", str(self.n_rows)),
            ("n_defaults", str(counts["n_defaults"])),
            ("n_development", str(int(self.development_idx.size))),
            ("n_test", str(int(self.test_idx.size))),
            ("n_development_defaults", str(counts["n_development_defaults"])),
            ("n_test_defaults", str(counts["n_test_defaults"])),
            ("test_frac", str(self.test_frac)),
            ("split_seed", str(self.split_seed)),
        ]
        assert_no_absolute_paths([value for _field, value in rows], context="01_cleaning_report")
        return pd.DataFrame(rows, columns=["field", "value"])


# ---------------------------------------------------------------------------
# The cleaning
# ---------------------------------------------------------------------------


CATEGORICAL_COLS = list(DATASET.layer_order)
EXPECTED_RAW_COLUMNS = [
    TARGET_COL,
    "LOAN",
    "MORTDUE",
    "VALUE",
    "REASON",
    "JOB",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC",
]
# The one-hot block the graph is built from. Every level must appear in the
# development partition; the marker columns are absence markers, not nodes.
EXPECTED_ONE_HOT_COLUMNS = [*JOB_COLUMNS, *REASON_COLUMNS]

DEVELOPMENT_PARTITION = "development"
TEST_PARTITION = "test"
PARTITION_COL = "partition"
BORROWER_ID_COL = "borrower_id"

DEFAULT_PROCESSED = DATASET.processed
DEFAULT_FROZEN_SPLIT = DATASET.frozen_split


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def _validate_raw_columns(df: pd.DataFrame) -> None:
    missing = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Raw HMEQ data is missing required columns: {missing}")


def _assert_no_literal_sentinel(df: pd.DataFrame) -> None:
    """No raw categorical value may already be the fill sentinel.

    If a raw value were literally ``"Missing"``, ``get_dummies`` would emit one
    ``JOB_Missing`` column carrying both a genuine category and the absence
    marker, and ``preprocessing.missing_levels`` would strip a real category from the
    graph without anything reporting it.
    """
    for col in CATEGORICAL_COLS:
        assert not (df[col].dropna().astype(str) == MISSING_SENTINEL).any(), (
            f"{col} contains the literal sentinel {MISSING_SENTINEL!r}; it cannot be "
            "distinguished from an absent value"
        )


def load_raw(input_csv: str | Path) -> tuple[pd.DataFrame, str]:
    """Step 1: the raw CSV and its checksum."""
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Raw HMEQ CSV not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {path}")
    df = pd.read_csv(path)
    _validate_raw_columns(df)
    _assert_no_literal_sentinel(df)
    return df, sha256_path(path)


def drop_unconnectable(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[int]]:
    """Step 2: remove borrowers with both ``JOB`` and ``REASON`` missing.

    Borrowers missing *one* attribute are kept: they still connect through the
    recorded one. Returns the frame with a fresh 0..n-1 index, the count, and
    the raw row indices removed.
    """
    unconnectable = df[CATEGORICAL_COLS].isna().all(axis=1)
    dropped = [int(i) for i in np.flatnonzero(unconnectable.to_numpy())]
    kept = df.loc[~unconnectable].reset_index(drop=True)
    return kept, len(dropped), dropped


def stratified_split(
    y: np.ndarray, *, test_frac: float, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Step 3: one stratified 80/20 split of row positions, both sides sorted."""
    if not 0.0 < float(test_frac) < 1.0:
        raise ValueError("test_frac must be strictly between zero and one")
    positions = np.arange(len(y), dtype=int)
    development, test = train_test_split(
        positions,
        test_size=float(test_frac),
        stratify=np.asarray(y, dtype=int),
        random_state=int(split_seed),
    )
    return np.sort(np.asarray(development, dtype=int)), np.sort(np.asarray(test, dtype=int))


def fit_medians(
    df: pd.DataFrame, development_idx: np.ndarray, columns=NUMERIC_COLS
) -> dict[str, float]:
    """Step 4a (fit): the median of each numeric column on the development rows only."""
    development = df.iloc[np.asarray(development_idx, dtype=int)]
    medians: dict[str, float] = {}
    for col in columns:
        value = development[col].median()
        if pd.isna(value):
            raise ValueError(f"{col}: every development value is missing; no median exists")
        medians[col] = float(value)
    return medians


def apply_imputation(
    df: pd.DataFrame, medians: dict[str, float]
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Step 4b (apply): fill nulls with the fitted medians and add retained flags.

    A ``<COL>_was_missing`` flag is created for every column that has at least
    one missing value in the frame handed in. Applied to development and test
    alike; nothing here is fitted.
    """
    out = df.copy()
    flags: list[str] = []
    n_missing: dict[str, int] = {}
    for col, median in medians.items():
        missing = out[col].isna()
        n_missing[col] = int(missing.sum())
        if n_missing[col]:
            flag = f"{col}{MISSING_INDICATOR_SUFFIX}"
            out[flag] = missing.astype("int64").to_numpy()
            flags.append(flag)
            out[col] = out[col].fillna(median)
    return out, flags, n_missing


def encode_categoricals(
    df: pd.DataFrame, development_idx: np.ndarray
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Step 5: one-hot ``JOB``/``REASON`` with an absence marker per layer.

    Level sets are read from the development partition. A level appearing only
    in the test partition would have to be either fitted on test data or
    dropped, and both are refused.
    """
    out = df.copy()
    development_mask = np.zeros(len(out), dtype=bool)
    development_mask[np.asarray(development_idx, dtype=int)] = True
    levels: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLS:
        filled = out[col].fillna(MISSING_SENTINEL).astype(str)
        out[col] = filled
        development_levels = sorted(set(filled[development_mask].tolist()))
        levels[col] = development_levels
        unseen = sorted(set(filled.tolist()) - set(development_levels))
        if unseen:
            raise ValueError(
                f"{col} has level(s) {unseen} that appear only in the test partition. "
                "Encoding them would mean fitting on test data; dropping them would "
                "leave those rows with no active category. Re-split or merge the level."
            )
    encoded = pd.get_dummies(out, columns=CATEGORICAL_COLS, drop_first=False, dtype="int64")
    for col, col_levels in levels.items():
        for level in col_levels:
            name = f"{col}_{level}"
            if name not in encoded.columns:
                encoded[name] = 0
    return encoded, levels


def clean_hmeq_frame(
    df: pd.DataFrame,
    *,
    test_frac: float = 0.2,
    split_seed: int = 42,
    raw_sha256: str | None = None,
) -> HMEQCleaningResult:
    """Steps 2-5 on an already-loaded raw frame, in that order."""
    _validate_raw_columns(df)
    _assert_no_literal_sentinel(df)
    n_raw_rows = int(len(df))

    # 2. drop the unconnectable, before the split
    frame, n_dropped, dropped = drop_unconnectable(df)

    # 3. the one stratified split
    y = frame[TARGET_COL].to_numpy().astype(int)
    development_idx, test_idx = stratified_split(y, test_frac=test_frac, split_seed=split_seed)
    partition = np.full(len(frame), DEVELOPMENT_PARTITION, dtype=object)
    partition[test_idx] = TEST_PARTITION

    # 4. impute: fit on development, apply to both, keep the flags
    medians = fit_medians(frame, development_idx)
    frame, flags, n_missing = apply_imputation(frame, medians)

    # 5. one-hot the two layer variables; the marker stays a column, not a node
    encoded, levels = encode_categoricals(frame, development_idx)

    missing_one_hot = [c for c in EXPECTED_ONE_HOT_COLUMNS if c not in encoded.columns]
    if missing_one_hot:
        raise ValueError(f"Cleaned data is missing expected one-hot columns: {missing_one_hot}")
    ordered = [*NUMERIC_COLS, *EXPECTED_ONE_HOT_COLUMNS, *flags, TARGET_COL]
    extra = [c for c in encoded.columns if c not in ordered]
    if extra:
        raise ValueError(f"Cleaned data has unexpected columns: {extra}")
    encoded = encoded.reindex(columns=ordered).reset_index(drop=True)
    if encoded.isna().sum().sum() != 0:
        raise ValueError("Cleaned data contains null values.")

    # Every borrower has exactly one active level per layer across genuine
    # categories + the absence marker; that is what the graph builder relies on.
    for col in CATEGORICAL_COLS:
        group = [c for c in EXPECTED_ONE_HOT_COLUMNS if c.startswith(f"{col}_")]
        genuine = [c for c in group if not c.endswith(f"_{MISSING_SENTINEL}")]
        markers = [c for c in group if c.endswith(f"_{MISSING_SENTINEL}")]
        assert_absence_consistent(encoded, genuine, markers, col)
    if (encoded["JOB_Missing"].to_numpy() & encoded["REASON_Missing"].to_numpy()).any():
        raise AssertionError("a borrower with both layers missing survived the drop")

    return HMEQCleaningResult(
        frame=encoded,
        partition=partition.astype(str),
        development_idx=development_idx,
        test_idx=test_idx,
        imputation_medians=medians,
        missing_flag_columns=flags,
        categorical_levels=levels,
        n_missing_per_column=n_missing,
        n_raw_rows=n_raw_rows,
        n_unconnectable_dropped=n_dropped,
        dropped_raw_indices=dropped,
        raw_sha256=raw_sha256,
        test_frac=float(test_frac),
        split_seed=int(split_seed),
    )


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    # Polars writes Parquet without the optional pandas engine.
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(frame.to_dict(orient="list")).write_parquet(path)


def clean_hmeq(
    input_csv: str | Path,
    processed_output: str | Path = DEFAULT_PROCESSED,
    frozen_split_output: str | Path = DEFAULT_FROZEN_SPLIT,
    *,
    test_frac: float = 0.2,
    split_seed: int = 42,
) -> HMEQCleaningResult:
    """Steps 1-5, then write the processed frame and the frozen split."""
    df, raw_sha256 = load_raw(input_csv)
    result = clean_hmeq_frame(df, test_frac=test_frac, split_seed=split_seed, raw_sha256=raw_sha256)

    processed_output = Path(processed_output)
    frozen_split_output = Path(frozen_split_output)
    _write_parquet(result.frame, processed_output)
    result = HMEQCleaningResult(
        **{
            **result.__dict__,
            "processed_path": processed_output,
            "frozen_split_path": frozen_split_output,
            "processed_sha256": sha256_path(processed_output),
        }
    )
    write_frozen_split(result.frozen_split_record(), frozen_split_output)
    return result


# ---------------------------------------------------------------------------
# The data stage
# ---------------------------------------------------------------------------


# Counts the cleaner is expected to produce on the raw HMEQ file. A mismatch is
# a finding about the data or the cleaner, never something to adjust.
EXPECTED = {
    "n_raw_rows": 5960,
    "n_unconnectable_dropped": 107,
    "n_retained": 5853,
    "n_defaults": 1179,
    "n_development": 4682,
    "n_test": 1171,
    "n_development_defaults": 943,
    "n_test_defaults": 236,
}


def clean_and_freeze(config: ExperimentConfig, raw_csv: str | Path = RAW_CSV) -> HMEQCleaningResult:
    """Steps 1-5 of the cleaner, writing the processed frame and the frozen split."""
    return clean_hmeq(
        raw_csv,
        processed_output=DATASET.processed_path(),
        frozen_split_output=DATASET.frozen_split_path(),
        test_frac=config.test_frac,
        split_seed=config.split_seed,
    )


def observed_counts(result: HMEQCleaningResult) -> dict[str, int]:
    counts = result.default_counts()
    return {
        "n_raw_rows": int(result.n_raw_rows),
        "n_unconnectable_dropped": int(result.n_unconnectable_dropped),
        "n_retained": int(result.n_rows),
        "n_defaults": counts["n_defaults"],
        "n_development": int(result.development_idx.size),
        "n_test": int(result.test_idx.size),
        "n_development_defaults": counts["n_development_defaults"],
        "n_test_defaults": counts["n_test_defaults"],
    }


def assert_expected_counts(observed: dict[str, int]) -> None:
    mismatched = {k: (EXPECTED[k], observed[k]) for k in EXPECTED if EXPECTED[k] != observed[k]}
    if mismatched:
        detail = "; ".join(f"{k}: expected {a}, got {b}" for k, (a, b) in mismatched.items())
        raise AssertionError(
            f"cleaned HMEQ does not match the recorded counts ({detail}). "
            "Report the discrepancy; do not adjust the expected values."
        )


def write_cleaning_report(result: HMEQCleaningResult, output: Path) -> Path:
    """``cleaning_report.csv`` under the stage directory: the cleaner's per-step report."""
    path = output / "cleaning_report.csv"
    pl.DataFrame(result.report().to_dict(orient="list")).write_csv(path)
    return path
