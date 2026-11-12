"""Cross-fitted scores: the primary construction, not a repair.

The baseline (Óskarsdóttir & Bravo, *Omega* 105:102520, 2021) seeds its network
from defaulters observed in the first four years and eleven months of a
five-year window and scores the *new* borrowers of the final month, then rolls
the window forward. Every borrower therefore ends up scored by an operator that
never saw its own default. HMEQ records neither an origination date nor a
default date, so that rolling window cannot be reproduced; stratified K-fold
cross-fitting is its non-temporal analogue and is used for the same reason, not
as a correction applied afterwards.

Without it the feature is degenerate rather than merely optimistic. Training
defaulters *are* the seeds, so their scores land in a band disjoint from
everyone else's -- measured at ``[0.001874, 0.002063]`` against ``[0.000329,
0.000468]`` -- giving in-sample AUC exactly 1.0000, and every downstream model
fitted on it collapses to predicting the majority class
(``accuracy = 0.80168``, ``precision = recall = 0``).
:func:`assert_crossfit_validity` fails loudly on that band rather than letting
it through.

The cross-fitted development scores are consumed only as a *feature* of the
downstream models (notebook 05), never as a ranking of their own: the fold
operators have different seed sets, so raw scores from different folds are not
on a common scale and are never pooled or ranked across folds. There is
deliberately no function here that does either, and ``tests/test_metrics.py``
asserts none appears.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from evaluation.masked_eval import Partitions

# In-sample AUC of a cross-fitted feature against the labels it was built
# without. Anything at or above this is the seed-band failure, not a strong
# feature.
MAX_IN_SAMPLE_AUC = 0.95

NO_FOLD = -1


class Scorer(Protocol):
    """Builds an operator on a label-masked frame and scores every borrower.

    ``masked_frame`` has ``BAD`` set to zero for every borrower whose label
    must stay hidden, so the operator's seed set is exactly the visible
    defaults. The returned array covers **all** borrowers in frame-row order;
    the caller keeps only the slice it is entitled to.
    """

    def __call__(self, masked_frame: pl.DataFrame, tag: str) -> np.ndarray: ...


@dataclass
class CrossfitResult:
    """Development-partition scores plus the fold each one came from.

    ``scores`` is NaN outside the development partition and ``fold_index`` is
    :data:`NO_FOLD` there, so a caller cannot silently evaluate a cross-fitted
    score on the test partition. ``fold_index`` travels with the scores because
    every metric on them has to be computed inside a fold.
    """

    scores: np.ndarray
    fold_index: np.ndarray
    n_folds: int
    seed: int
    fold_seed_counts: dict[int, int] = field(default_factory=dict)
    fold_sizes: dict[int, int] = field(default_factory=dict)

    @property
    def scored_idx(self) -> np.ndarray:
        return np.flatnonzero(self.fold_index != NO_FOLD)

    def fold_members(self, fold: int) -> np.ndarray:
        return np.flatnonzero(self.fold_index == int(fold))

    def as_frame(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "borrower_id": np.arange(self.scores.size, dtype=int),
                "score": self.scores,
                "fold_index": self.fold_index,
            }
        )

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "n_scored": int(self.scored_idx.size),
            "fold_sizes": {str(k): v for k, v in sorted(self.fold_sizes.items())},
            "fold_seed_counts": {str(k): v for k, v in sorted(self.fold_seed_counts.items())},
        }


def fold_assignment(partitions: Partitions, folds: int, seed: int) -> np.ndarray:
    """Stratified fold id per borrower; :data:`NO_FOLD` outside development.

    Derived from the recorded split and the master seed alone, so two arms that
    call it with the same configuration get identical folds without having to
    exchange a file.
    """
    development = partitions.development_idx
    y = partitions.y_true
    assignment = np.full(y.size, NO_FOLD, dtype=int)
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    for fold, (_train, test) in enumerate(
        splitter.split(development.reshape(-1, 1), y[development])
    ):
        assignment[development[test]] = fold
    return assignment


def assert_same_folds(left: CrossfitResult, right: CrossfitResult) -> None:
    """Arms must cross-fit on identical folds or their comparison is uncontrolled."""
    if not np.array_equal(left.fold_index, right.fold_index):
        differing = int((left.fold_index != right.fold_index).sum())
        raise AssertionError(
            f"cross-fit fold assignments differ for {differing} borrowers; the two "
            "arms would be scored by operators with different seed sets and the "
            "comparison between them would measure the folds"
        )


CACHE_KEY_FILE = "cache_key.json"


def _assert_cache_matches(cache: Path, cache_key: str | None) -> None:
    """Refuse a cache built under a different configuration.

    Fold scores are stored as ``fold_<k>.npy`` and validated only against the
    borrower count, which does not change between configurations: the same
    5,853 borrowers are scored under every network, every ``tau`` and every fold
    count. A two-fold smoke run therefore leaves files that a five-fold run
    would silently adopt for folds 0 and 1 -- scores from a different operator
    and a different fold partition, mixed into the reported column with no
    error anywhere.

    The key is written on first use and compared on every later one.
    """
    recorded = cache / CACHE_KEY_FILE
    existing = sorted(p.name for p in cache.glob("fold_*.npy"))
    if cache_key is None:
        if existing:
            raise ValueError(
                f"{cache} holds {existing} but no {CACHE_KEY_FILE}, so there is no way "
                "to tell which configuration produced them. Delete the directory and "
                "re-run, or pass cache_key."
            )
        return
    if recorded.exists():
        stored = json.loads(recorded.read_text()).get("cache_key")
        if stored != cache_key:
            raise ValueError(
                f"{cache} was written under configuration {stored!r} but this run is "
                f"{cache_key!r}. Reusing it would mix scores from two different "
                "operators into one column. Delete the directory and re-run."
            )
        return
    if existing:
        raise ValueError(
            f"{cache} holds {existing} from a run that recorded no configuration key. "
            "Delete the directory and re-run so every fold is rebuilt."
        )
    recorded.write_text(json.dumps({"cache_key": cache_key}, indent=2))


def crossfit_scores(
    partitions: Partitions,
    scorer: Scorer,
    folds: int,
    seed: int,
    target_col: str = "BAD",
    *,
    cache_dir: str | Path | None = None,
    fold_index: np.ndarray | None = None,
    cache_key: str | None = None,
) -> CrossfitResult:
    """Score the development partition, each fold by an operator that never saw it.

    For fold ``F_k`` the seed set is the development defaulters outside ``F_k``;
    ``F_k``'s labels and **all** test-partition labels are masked through
    :meth:`Partitions.mask_labels`, the operator is built, every borrower is
    scored, and only ``F_k``'s scores are kept.

    Costs ``folds`` walks. ``cache_dir`` makes it resumable: each fold's scores
    are written as soon as they exist, so a job killed at fold 3 of 5 resumes
    at fold 3.
    """
    if target_col != partitions.target_col:
        raise ValueError(f"partitions carry target {partitions.target_col!r}, not {target_col!r}")
    assignment = (
        fold_assignment(partitions, folds, seed)
        if fold_index is None
        else np.asarray(fold_index, dtype=int)
    )
    y = partitions.y_true
    test_idx = partitions.test_idx

    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        _assert_cache_matches(cache, cache_key)

    scores = np.full(y.size, np.nan, dtype=float)
    fold_seed_counts: dict[int, int] = {}
    fold_sizes: dict[int, int] = {}

    for fold in range(int(folds)):
        members = np.flatnonzero(assignment == fold)
        if members.size == 0:
            raise ValueError(f"fold {fold} is empty; reduce crossfit_folds")
        hidden = np.concatenate([members, test_idx])
        seed_ids = partitions.seed_ids(hidden)
        fold_seed_counts[fold] = int(seed_ids.size)
        fold_sizes[fold] = int(members.size)

        cached = cache / f"fold_{fold}.npy" if cache is not None else None
        if cached is not None and cached.exists():
            fold_scores = np.load(cached)
            if fold_scores.shape != y.shape:
                raise ValueError(
                    f"cached fold {fold} has shape {fold_scores.shape}, expected {y.shape}; "
                    "delete the cache directory and re-run"
                )
        else:
            masked = partitions.mask_labels(hidden)
            visible = masked[target_col].to_numpy().astype(int)
            if visible[members].sum() or visible[test_idx].sum():
                raise AssertionError(
                    f"fold {fold}: mask_labels left a visible label inside the held-out "
                    "fold or the test partition"
                )
            fold_scores = np.asarray(scorer(masked, f"fold_{fold}"), dtype=float)
            if fold_scores.shape != y.shape:
                raise ValueError(
                    f"scorer returned {fold_scores.shape} scores, expected {y.shape} "
                    "(one per borrower, in frame-row order)"
                )
            if cached is not None:
                np.save(cached, fold_scores)

        scores[members] = fold_scores[members]

    result = CrossfitResult(
        scores=scores,
        fold_index=assignment,
        n_folds=int(folds),
        seed=int(seed),
        fold_seed_counts=fold_seed_counts,
        fold_sizes=fold_sizes,
    )
    if cache is not None:
        (cache / "crossfit_folds.json").write_text(
            json.dumps(result.manifest_entry(), indent=2, sort_keys=True)
        )
    return result


def score_test_partition(
    partitions: Partitions, scorer: Scorer, *, tag: str = "test_operator"
) -> np.ndarray:
    """One operator, every development default as a seed, all test labels masked.

    No cross-fitting is needed here: a test borrower is never a seed, so its
    score is already out of influence. Returns scores for every borrower; the
    primary report reads the test slice.

    Named ``score_test_partition`` rather than ``test_partition_scores``: pytest
    collects any module-level callable whose name begins with ``test_``, so the
    latter was picked up as a test case in every module that imported it and
    reported as an error for a missing ``partitions`` fixture.
    """
    test_idx = partitions.test_idx
    masked = partitions.mask_labels(test_idx)
    visible = masked[partitions.target_col].to_numpy().astype(int)
    if visible[test_idx].sum():
        raise AssertionError("mask_labels left a visible test-partition label")
    scores = np.asarray(scorer(masked, tag), dtype=float)
    if scores.shape != partitions.y_true.shape:
        raise ValueError(
            f"scorer returned {scores.shape} scores, expected {partitions.y_true.shape}"
        )
    return scores


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_crossfit_validity(
    result: CrossfitResult, y, *, max_auc: float = MAX_IN_SAMPLE_AUC
) -> dict[str, Any]:
    """The cross-fitted feature must not separate its own labels perfectly.

    A cross-fitted score is still correlated with the label -- that is the
    point -- but an AUC at or near 1.0 means the seeds are being scored by
    their own defaults again. Reports the seed and non-seed score bands so the
    failure is diagnosable from the exception text alone.
    """
    y = np.asarray(y, dtype=int)
    idx = result.scored_idx
    scores = result.scores[idx]
    labels = y[idx]
    if np.isnan(scores).any():
        raise AssertionError("cross-fitted scores contain NaN inside the scored population")
    if len(np.unique(labels)) < 2:
        raise AssertionError("cross-fitted validation needs both outcome classes")

    auc = float(roc_auc_score(labels, scores))
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    report = {
        "in_sample_auc": auc,
        "n_scored": int(idx.size),
        "defaulter_band": [float(positives.min()), float(positives.max())],
        "non_defaulter_band": [float(negatives.min()), float(negatives.max())],
        "bands_disjoint": bool(positives.min() > negatives.max()),
        "max_auc": float(max_auc),
    }
    if auc >= float(max_auc):
        raise AssertionError(
            f"in-sample AUC of the cross-fitted feature is {auc:.4f} (limit {max_auc}); "
            f"defaulters score in {report['defaulter_band']} and everyone else in "
            f"{report['non_defaulter_band']}. The seeds are scoring themselves: the "
            "fold's labels reached the operator."
        )
    return report
