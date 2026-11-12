"""Stage 03's named steps: the classical MuLP personalized PageRank arm on the full operator.

For every network in the ladder, one test operator (all development defaults
as seeds, every test label masked) gives ``omega_C_test``: the stationary
score on the production sparse operator, which the lumped stationary score of
the scoring stage must reproduce exactly. On the primary network it
additionally produces ``omega_C_crossfit`` (+ ``fold_index``): K operators
over stratified folds of the development partition, each fold scored by an
operator built without its labels (:mod:`evaluation.crossfit`), against which
the downstream stage's lumped fold scores are checked.

Every masked frame is written once into ``outputs/notebook_pipeline/masked/``
and its operator digests recorded, so the quantum arm can prove it built the
same operator. Every solve is sparse ARPACK on the matrix-free operator. The
placebo and the classical running-average control live with the lumped
operators (``evaluation.controls``, ``quantum.mechanism.scores_stage``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config import ExperimentConfig, NetworkSpec
from datasets import DatasetConfig
from evaluation.crossfit import (
    CrossfitResult,
    assert_crossfit_validity,
    crossfit_scores,
    fold_assignment,
    score_test_partition,
)
from evaluation.masked_eval import Partitions, evaluate_ranking_on_subset
from fingerprint import (
    TEST_OPERATOR_TAG,
    OperatorBundle,
    config_fingerprint,
    fold_tag,
    sha256_path,
    solve_classical,
)
from paths import masked_frame_path


def digest_path(output: Path) -> Path:
    return Path(output) / "operator_digests.json"


def load_digests(output: Path) -> dict[str, dict[str, str]]:
    path = digest_path(output)
    return json.loads(path.read_text()) if path.exists() else {}


def digest_key(spec_name: str, tag: str) -> str:
    return f"{spec_name}/{tag}"


def assert_digests_complete(
    digest_log: dict[str, dict[str, str]], config: ExperimentConfig, dataset: DatasetConfig
) -> list[str]:
    """Stage 04 asserts operator identity per tag, so every tag must be here."""
    required = [digest_key(spec, TEST_OPERATOR_TAG) for spec in dataset.ladder]
    required += [
        digest_key(dataset.primary, fold_tag(k)) for k in range(int(config.crossfit_folds))
    ]
    missing = [tag for tag in required if tag not in digest_log]
    if missing:
        raise AssertionError(
            f"operator digests are missing for {missing}. Stage 04 compares its operator "
            "against these tag by tag. This happens when a fold's scores were restored from "
            "crossfit_cache/ but its digest was not: delete the crossfit_cache_* directories "
            "and re-run."
        )
    return required


def make_scorer(
    spec: NetworkSpec,
    config: ExperimentConfig,
    digest_log: dict[str, dict[str, str]],
    holder: dict[str, Any],
):
    """A :class:`evaluation.crossfit.Scorer` that solves the classical operator.

    Writes the masked frame to the shared directory before building anything,
    records the operator's digests under ``spec/tag``, and keeps the bundle in
    ``holder`` so the finite-horizon control can reuse the test operator.
    """

    def scorer(masked_frame: pl.DataFrame, tag: str) -> np.ndarray:
        path = masked_frame_path(spec.name, tag)
        masked_frame.write_parquet(path)
        bundle = solve_classical(path, spec, config)
        digest_log[digest_key(spec.name, tag)] = {
            **bundle.digests,
            "masked_frame": sha256_path(path),
        }
        holder[tag] = bundle
        return bundle.omega

    return scorer


def operator_diagnostics(bundle: OperatorBundle) -> dict[str, Any]:
    """The checks a referee wants to see: column-stochastic T, seed mass, a valid pi."""
    ranker = bundle.ranker
    transition = ranker.last_transition_diagnostics
    v = ranker._dangling_patch_vector()
    return {
        "N": int(ranker.N),
        "n_layers": int(ranker.n_layers),
        "transition_max_column_sum_error": float(transition.max_column_sum_error),
        "transition_dangling_columns": int(transition.dangling_column_count),
        "restart_seed_columns": int(ranker.restart_matrix.n_seed_columns),
        "seed_mass_total": float(v.sum()),
        "seed_mass_positive": bool(v.sum() > 0 and (v >= 0).all()),
        "pi_min": float(bundle.pi.min()),
        "pi_sum": float(bundle.pi.sum()),
        "omega_sum": float(bundle.omega.sum()),
    }


def assert_operator_diagnostics(diagnostics: dict[str, Any]) -> None:
    if diagnostics["transition_max_column_sum_error"] > 1e-10:
        raise AssertionError("transition matrix is not column-stochastic")
    if not diagnostics["seed_mass_positive"]:
        raise AssertionError("personalisation vector carries no positive seed mass")
    if diagnostics["pi_min"] < -1e-10 or abs(diagnostics["pi_sum"] - 1.0) > 1e-10:
        raise AssertionError("stationary vector is not a probability distribution")


def score_test_operator(
    partitions: Partitions, spec: NetworkSpec, config: ExperimentConfig, scorer
) -> np.ndarray:
    """One operator, every development default a seed, all test labels masked."""
    return score_test_partition(partitions, scorer, tag=TEST_OPERATOR_TAG)


def holdout_ranking_row(
    scores: np.ndarray, partitions: Partitions, config: ExperimentConfig, label: str
) -> dict[str, Any]:
    """Tie-aware quick look at the holdout ranking (the paper tables come from the shared bootstrap)."""
    y = partitions.y_true
    seed_idx = np.flatnonzero((y == 1) & (partitions.partition == "development"))
    return evaluate_ranking_on_subset(
        scores,
        y,
        partitions.test_idx,
        seed_idx,
        k_values=config.top_k,
        model_label=label,
        decimals=config.tie_tolerance,
    )


def crossfit_classical(
    partitions: Partitions, spec: NetworkSpec, config: ExperimentConfig, scorer, output: Path
) -> tuple[CrossfitResult, dict[str, Any]]:
    """K fold operators on the primary network; each fold scored without its labels."""
    folds = fold_assignment(partitions, config.crossfit_folds, config.seed_folds())
    # The key carries the frame and the split the fold operators are built on:
    # a cache written for another frame or split is refused, not reused.
    crossfit = crossfit_scores(
        partitions,
        scorer,
        folds=config.crossfit_folds,
        seed=config.seed_folds(),
        cache_dir=output / f"crossfit_cache_{spec.name}",
        fold_index=folds,
        cache_key=config_fingerprint(
            config, "03_classical", network_spec=spec.name,
            extra={"processed_sha256": sha256_path(partitions.processed_path),
                   "split_sha256": sha256_path(partitions.split_path)},
        ),
    )
    validity = assert_crossfit_validity(crossfit, partitions.y_true)
    return crossfit, validity


def write_classical_scores(
    output: Path,
    spec_name: str,
    partitions: Partitions,
    omega_test: np.ndarray,
    crossfit: CrossfitResult | None = None,
) -> Path:
    columns: dict[str, Any] = {
        "borrower_id": np.arange(partitions.frame.height, dtype=int),
        "partition": partitions.partition,
        "BAD": partitions.y_true,
        "omega_C_test": omega_test,
    }
    if crossfit is not None:
        columns["omega_C_crossfit"] = crossfit.scores
        columns["fold_index"] = crossfit.fold_index
    path = output / f"classical_scores_{spec_name}.parquet"
    pl.DataFrame(columns).write_parquet(path)
    return path


def write_digests(output: Path, digest_log: dict[str, dict[str, str]]) -> Path:
    path = digest_path(output)
    path.write_text(json.dumps(digest_log, indent=2, sort_keys=True))
    return path
