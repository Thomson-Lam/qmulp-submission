"""Partitions, label hiding, and the guard against evaluating on your own seeds.

Vocabulary
----------
Three devices, kept apart (Óskarsdóttir & Bravo, *Omega* 105:102520, 2021):
the ``seed_set`` (``D``), the borrowers whose observed default builds the
diffusion operator; the ``scoring_cohort``, borrowers whose labels are masked so
their score is *out of influence*, computed by an operator that never saw their
own default (temporal in the baseline; HMEQ has no dates, so stratified
cross-fitting stands in for the rolling window); and the ``test_partition`` /
``development_partition``, the ordinary split on which downstream performance
is reported, owned by the dataset's cleaner (``hmeq.cleaning``, the Freddie
Mac cleaning notebook) and read back here by :func:`load_partitions`.

One implementation of label hiding
----------------------------------
:meth:`Partitions.mask_labels` is the only place a label is hidden. The
cross-fitting loop and the test-partition scoring both go through it, so the
two cannot drift into hiding labels differently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from evaluation.reporting import ranking_point_metrics
from evaluation.ties import canonicalize, tie_report
from config import DEFAULT_CONFIG

# Default for the tie handling every ranking metric applies. It lives in the
# frozen configuration; this name exists so a caller that does not hold a
# config object still gets the frozen value rather than inventing its own.
TIE_DECIMALS = DEFAULT_CONFIG.tie_tolerance

# The schema name every frozen split record carries and load_partitions checks.
FROZEN_SPLIT_SCHEMA = "qmulp.frozen_split.v1"

DEVELOPMENT_PARTITION = "development"
TEST_PARTITION = "test"
PARTITION_COL = "partition"
BORROWER_ID_COL = "borrower_id"

# The masked default indicator as the operator sees it: 1 for a visible
# default, 0 for a visible non-default, UNKNOWN_LABEL for every borrower whose
# label is hidden (the whole test partition, and the held-out fold under
# cross-fitting). The operator itself only distinguishes seed from non-seed;
# the third level exists so the lumping key never reads a hidden true label.
UNKNOWN_LABEL = -1


# ---------------------------------------------------------------------------
# The partition table written by preprocessing.clean_hmeq
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Partitions:
    """The processed frame plus its development/test assignment.

    Deliberately does *not* split anything: the split is a recorded artifact
    (``outputs/frozen_split.json``), not something each arm recomputes. Two
    arms that both call ``train_test_split`` with "the same" parameters are one
    library upgrade away from disagreeing silently, and nothing downstream
    would notice.
    """

    frame: pl.DataFrame
    borrower_ids: np.ndarray
    partition: np.ndarray
    target_col: str
    processed_path: Path | None
    split_path: Path | None

    @property
    def development_idx(self) -> np.ndarray:
        return np.flatnonzero(self.partition == DEVELOPMENT_PARTITION)

    @property
    def test_idx(self) -> np.ndarray:
        return np.flatnonzero(self.partition == TEST_PARTITION)

    @property
    def y_true(self) -> np.ndarray:
        return self.frame[self.target_col].to_numpy().astype(int)

    def _check_ids(self, ids) -> np.ndarray:
        ids = np.asarray(list(ids), dtype=int)
        if ids.size and (ids.min() < 0 or ids.max() >= self.frame.height):
            raise ValueError(
                f"ids outside [0, {self.frame.height}): min={int(ids.min())} max={int(ids.max())}"
            )
        return ids

    def frame_with_labels(self, labels) -> pl.DataFrame:
        """The frame with ``target_col`` replaced by ``labels`` (0/1 per borrower)."""
        labels = np.asarray(labels, dtype=int)
        if labels.shape != (self.frame.height,):
            raise ValueError(f"labels must have shape ({self.frame.height},); got {labels.shape}")
        return self.frame.with_columns(pl.Series(self.target_col, labels))

    def mask_labels(self, exclude_ids) -> pl.DataFrame:
        """Return the frame with ``target_col`` set to 0 for ``exclude_ids``.

        The single primitive for label hiding in this repo. A masked borrower
        contributes no restart column, so the operator built from the result
        cannot have seen its default -- which is the whole point of the
        scoring cohort, and of the baseline's rolling window before it.
        """
        exclude = self._check_ids(exclude_ids)
        y = self.y_true.copy()
        if exclude.size:
            y[exclude] = 0
        return self.frame_with_labels(y)

    def masked_labels(self, exclude_ids) -> np.ndarray:
        """The three-level masked indicator: true label where visible, else UNKNOWN_LABEL.

        This is what the lumping key reads. It never contains a hidden
        borrower's true label, so no class assignment can depend on it.
        """
        exclude = self._check_ids(exclude_ids)
        levels = self.y_true.copy()
        if exclude.size:
            levels[exclude] = UNKNOWN_LABEL
        return levels

    def placebo_labels(self, seed_ids) -> np.ndarray:
        """A label vector with 1 on ``seed_ids`` and 0 elsewhere.

        The seed-permutation placebo: the seed set is reassigned to borrowers
        chosen independently of the outcome, holding its size fixed, and the
        operator is built from these labels while every metric still grades
        against ``y_true``.
        """
        seeds = self._check_ids(seed_ids)
        labels = np.zeros(self.frame.height, dtype=int)
        labels[seeds] = 1
        return labels

    def seed_ids(self, exclude_ids=()) -> np.ndarray:
        """Borrower ids that still carry a 1 after masking -- the seed set D."""
        return np.flatnonzero(
            self.mask_labels(exclude_ids)[self.target_col].to_numpy().astype(int) == 1
        )


def load_partitions(
    processed_path: str | Path,
    split_path: str | Path,
    target_col: str = "BAD",
) -> Partitions:
    """Read the processed frame and ``outputs/frozen_split.json`` into a :class:`Partitions`.

    Never calls ``train_test_split``: the split happened once, in
    ``preprocessing.clean_hmeq``, and was written down. Everything the frozen
    split recorded about the frame -- row count, default counts per partition,
    the processed file's checksum -- is checked here, so a stage cannot read a
    frame the split was not made for.
    """
    processed_path = Path(processed_path)
    split_path = Path(split_path)
    for path in (processed_path, split_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    frame = pl.read_parquet(processed_path)
    record = json.loads(split_path.read_text())
    if record.get("schema") != FROZEN_SPLIT_SCHEMA:
        raise ValueError(
            f"{split_path} is not a qmulp frozen split (schema={record.get('schema')!r})"
        )

    if target_col not in frame.columns:
        raise ValueError(f"Missing target column {target_col!r} in {processed_path}")
    if int(record["n_rows"]) != frame.height:
        raise ValueError(
            f"frozen split records {record['n_rows']} rows but the processed frame has "
            f"{frame.height}; borrower_id is the frame's row position, so they must match"
        )

    development_idx = np.asarray(record["development_idx"], dtype=int)
    test_idx = np.asarray(record["test_idx"], dtype=int)
    covered = np.sort(np.concatenate([development_idx, test_idx]))
    if not np.array_equal(covered, np.arange(frame.height)):
        raise ValueError("frozen split does not partition every borrower exactly once")

    partition = np.full(frame.height, DEVELOPMENT_PARTITION, dtype=object)
    partition[test_idx] = TEST_PARTITION
    partition = partition.astype(str)

    y = frame[target_col].to_numpy().astype(int)
    if set(np.unique(y)).difference({0, 1}):
        raise ValueError(f"{target_col!r} must be binary")
    observed = {
        "n_development_defaults": int(y[development_idx].sum()),
        "n_test_defaults": int(y[test_idx].sum()),
    }
    for key, value in observed.items():
        if key in record and int(record[key]) != value:
            raise ValueError(
                f"frozen split records {key}={record[key]} but the processed frame has "
                f"{value}; the frame is not the one the split was made for"
            )
    if record.get("processed_sha256"):
        actual = split_sha256(processed_path)
        if actual != record["processed_sha256"]:
            raise ValueError(
                f"{processed_path} has sha256 {actual}, but the frozen split was written "
                f"for {record['processed_sha256']}; re-run stage 01 so the two agree"
            )

    return Partitions(
        frame=frame,
        borrower_ids=np.arange(frame.height, dtype=int),
        partition=partition,
        target_col=target_col,
        processed_path=processed_path,
        split_path=split_path,
    )


def split_sha256(split_path: str | Path) -> str:
    """SHA-256 of a file as written to disk (the frozen split, or the frame)."""
    digest = hashlib.sha256()
    with open(split_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The out-of-influence guard
# ---------------------------------------------------------------------------


def assert_out_of_influence(score_ids, seed_ids) -> None:
    """No reported metric may be computed on a population containing its seeds.

    A seed's own default built the operator that scores it, so its score is not
    a prediction. Measured on the pre-fix artifacts, seed defaulters scored in
    ``[0.001874, 0.002063]`` while everyone else sat in ``[0.000329,
    0.000468]`` -- disjoint bands, in-sample AUC exactly 1.0000. That is the
    failure mode the whole label-hiding device exists to prevent, which is why
    it is an assertion at the metric entry point rather than a convention
    someone has to remember.
    """
    score_ids = np.asarray(list(score_ids), dtype=int)
    seed_ids = np.asarray(list(seed_ids), dtype=int)
    overlap = np.intersect1d(score_ids, seed_ids)
    if overlap.size:
        raise AssertionError(
            f"{overlap.size} of {score_ids.size} scored borrowers are their own seeds "
            f"(e.g. {overlap[:10].tolist()}). A ranking metric on such a population is "
            "trivially perfect and reports nothing. Mask those labels via "
            "Partitions.mask_labels before building the operator."
        )


# ---------------------------------------------------------------------------
# The quick-look ranking report (the paper's tables come from the evaluation.table_* modules)
# ---------------------------------------------------------------------------


def _ranking_report(
    scores,
    y_true,
    evaluation_idx,
    seed_idx,
    *,
    k_values,
    model_label: str,
    decimals: int = TIE_DECIMALS,
) -> dict[str, float | int | str]:
    """The quick-look ranking report; every metric is the one ``evaluation.reporting`` defines."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    evaluation_idx = np.asarray(evaluation_idx, dtype=int)
    seed_idx = np.asarray(seed_idx, dtype=int)
    if scores.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("Scores and labels must be one-dimensional and have equal length")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values")

    # Every reported ranking metric passes through here, so this is the one
    # place the out-of-influence guarantee has to hold.
    assert_out_of_influence(evaluation_idx, seed_idx)

    y_eval = y_true[evaluation_idx]
    # Rounded before any metric touches it: automorphic borrowers carry equal
    # scores up to solver noise, and an unrounded sort orders them by that
    # noise. See evaluation.ties.
    score_eval = canonicalize(scores[evaluation_idx], decimals)
    if len(np.unique(y_eval)) < 2:
        raise ValueError("Ranking evaluation requires both target classes")

    if any(int(raw_k) <= 0 for raw_k in k_values):
        raise ValueError("k_values must contain positive integers")
    # PR-AUC, ROC-AUC, and precision and lift at k (the exact expectations over
    # tie-breakings of the borrowers on the rank-k boundary) are computed by
    # evaluation.reporting alone, so this quick look and the paper's tables
    # cannot disagree. The boundary composition is reported beside them.
    point = ranking_point_metrics(score_eval, y_eval, [int(raw_k) for raw_k in k_values], decimals=decimals)
    report: dict[str, float | int | str] = {
        "model": model_label,
        "tie_decimals": int(decimals),
        "distinct_levels": point["distinct_levels"],
        "pr_auc": point["pr_auc"],
        "roc_auc": point["roc_auc"],
        "n_scoring_cohort": int(len(evaluation_idx)),
        "n_scoring_cohort_defaulters": int(y_eval.sum()),
        "n_seed": int(len(seed_idx)),
        "n_seed_defaulters": int(y_true[seed_idx].sum()),
        "scoring_cohort_base_rate": float(y_eval.mean()),
    }
    ties = tie_report(score_eval, decimals, k_values)
    for raw_k in k_values:
        k = int(raw_k)
        report[f"effective_k_{k}"] = min(k, int(score_eval.size))
        report[f"precision_at_{k}"] = point[f"precision_at_{k}"]
        report[f"lift_at_{k}"] = point[f"lift_at_{k}"]
        report[f"n_strictly_above_{k}"] = ties[f"k_{k}"]["n_strictly_above"]
        report[f"n_tied_at_boundary_{k}"] = ties[f"k_{k}"]["n_tied_at_boundary"]
    return report


def evaluate_ranking_on_subset(
    scores,
    y_true,
    scoring_cohort_idx,
    seed_idx,
    *,
    k_values=(5, 10, 25),
    model_label: str,
    decimals: int = TIE_DECIMALS,
) -> dict[str, float | int | str]:
    return _ranking_report(
        scores,
        y_true,
        scoring_cohort_idx,
        seed_idx,
        k_values=k_values,
        model_label=model_label,
        decimals=decimals,
    )
