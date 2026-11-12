"""Shared plumbing both halves of the study import: the frozen inputs, the
operator builder, its digests, and the cache fingerprint.

Everything here exists so that no notebook defines a parameter or a convention
of its own. ``alpha``, ``stickiness``, the score definition, the shots,
``test_frac`` and every seed come from :class:`config.ExperimentConfig`; which frame, split and
bootstrap matrix a stage reads comes from its :class:`datasets.DatasetConfig`;
and the masked frame an arm builds its operator from is read by digest
(:func:`operator_cache_extra`).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from scipy.sparse import issparse

from config import ExperimentConfig, NetworkSpec
from datasets import DatasetConfig
from evaluation.masked_eval import Partitions, load_partitions
from preprocessing.missing_levels import MISSING_INDICATOR_SUFFIX, MISSING_SENTINEL

if TYPE_CHECKING:
    from classical.mulp.mulp import MultiLayerRanker

TEST_OPERATOR_TAG = "test_operator"


def fold_tag(fold: int) -> str:
    return f"fold_{int(fold)}"


def smoke_enabled() -> bool:
    """``QMULP_SMOKE=1`` trims every sweep to one small point."""
    return os.environ.get("QMULP_SMOKE", "").strip() not in ("", "0", "false", "False")


def sha256_bytes(*chunks: Any) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(np.ascontiguousarray(chunk).tobytes())
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Inputs every stage shares
# ---------------------------------------------------------------------------


def load_frozen_partitions(dataset: DatasetConfig) -> Partitions:
    """The processed frame and the frozen split of ``dataset``, resolved against the repo root."""
    return load_partitions(dataset.processed_path(), dataset.frozen_split_path())


def stratified_resample_indices(y: np.ndarray, reps: int, rng, *, stratify: bool = True):
    """``reps`` index arrays, each the same length as ``y``, yielded one at a time.

    Stratified on the outcome: an unstratified resample occasionally draws a
    replicate whose positive count is far enough off that PR-AUC is not
    comparable across replicates.
    """
    y = np.asarray(y, dtype=int)
    n = y.size
    if not stratify:
        for _ in range(int(reps)):
            yield rng.integers(0, n, size=n)
        return

    groups = [np.flatnonzero(y == value) for value in np.unique(y)]
    for _ in range(int(reps)):
        yield np.concatenate(
            [group[rng.integers(0, group.size, size=group.size)] for group in groups]
        )


def bootstrap_indices(y, reps: int, seed: int, *, stratify: bool = True) -> np.ndarray:
    """The one ``(reps, n)`` resample matrix every arm shares.

    Drawn once by the data stage from ``config.seed_bootstrap()`` and written
    to the dataset's bootstrap file; every interval in every table is computed
    on it, so differences between arms are paired by construction.
    """
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(int(seed))
    out = np.empty((int(reps), y.size), dtype=np.int32)
    for rep, idx in enumerate(stratified_resample_indices(y, reps, rng, stratify=stratify)):
        out[rep] = idx
    return out


def load_bootstrap_indices(
    config: ExperimentConfig, partitions: Partitions, dataset: DatasetConfig
) -> np.ndarray:
    """The one shared ``(reps, n_test)`` resample matrix the data stage of ``dataset`` wrote.

    Each dataset's matrix is drawn over its own test partition and must never
    be read in another's place; the shape check below is what catches it.
    """
    path = dataset.bootstrap_path()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the data stage writes the shared bootstrap indices"
        )
    indices = np.load(path)
    n_test = int(partitions.test_idx.size)
    if indices.ndim != 2 or indices.shape[1] != n_test:
        raise ValueError(
            f"bootstrap indices have shape {indices.shape}; expected (reps, {n_test}). "
            "Re-run the data stage so the resample matrix matches the frozen split."
        )
    if indices.shape[0] != int(config.bootstrap_reps):
        raise ValueError(
            f"bootstrap indices hold {indices.shape[0]} replicates but the configuration asks "
            f"for {config.bootstrap_reps}; re-run the data stage under this configuration"
        )
    return indices.astype(int)


def layer_columns_for(frame: pl.DataFrame, spec: NetworkSpec) -> dict[str, list[str]]:
    """Genuine (non-absence) one-hot columns per layer, in state order."""
    columns: dict[str, list[str]] = {}
    for layer in spec.layers:
        found = sorted(
            c
            for c in frame.columns
            if c.startswith(layer.prefix)
            and not c.endswith(MISSING_INDICATOR_SUFFIX)
            and not c.endswith(f"_{MISSING_SENTINEL}")
        )
        if not found:
            raise ValueError(
                f"layer {layer.name!r} has no columns in the frame; run the data stage first"
            )
        columns[layer.name] = found
    return columns


def layer_marker_columns_for(frame: pl.DataFrame, spec: NetworkSpec) -> dict[str, list[str]]:
    """The absence-marker column per layer, where the frame has one."""
    markers: dict[str, list[str]] = {}
    for layer in spec.layers:
        if layer.marker_column in frame.columns:
            markers[layer.name] = [layer.marker_column]
    return markers


# The state a borrower takes in a layer where none of its genuine columns is active.
UNRECORDED = -1


def layer_state_ids(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """State index per borrower in one layer; ``UNRECORDED`` where blank."""
    values = frame.select(columns).to_numpy().astype(int)
    ids = values.argmax(axis=1)
    ids[~values.astype(bool).any(axis=1)] = UNRECORDED
    return ids.astype(int)


def observation_flags(frame: pl.DataFrame, dataset: DatasetConfig) -> dict[str, Any]:
    """Four per-individual flags that keep three different absences apart.

    ``blank_field`` -- a genuinely absent raw value: every ``<col>_was_missing``
    flag, plus an active layer marker where the dataset says such a marker is
    a blank (HMEQ). ``structural_none`` -- the absence level of a family whose
    null means "none" (``DatasetConfig.structural_none_columns``); not
    missingness. ``unseen_level`` -- an active marker of a one-hot family whose
    value lay outside the development vocabulary (Freddie Mac: the layers and
    every other fitted family except the structural-none ones).
    ``partially_observed`` -- no membership edge in at least one layer of the
    dataset, whatever the reason. ``blank_field_count`` and
    ``blank_field_pattern`` count and lay out the blank fields alone; they are
    what the blank-field baseline, the strata and the R^2 regressors use.
    """
    was_missing = [c for c in frame.columns if c.endswith(MISSING_INDICATOR_SUFFIX)]
    markers = [c for c in frame.columns if c.endswith(f"_{MISSING_SENTINEL}")]
    layer_markers = [f"{layer}_{MISSING_SENTINEL}" for layer in dataset.layer_order]
    missing_layer_markers = [m for m in layer_markers if m not in frame.columns]
    if missing_layer_markers:
        raise ValueError(f"the processed frame lacks the layer markers {missing_layer_markers}")
    structural = [f"{family}_{MISSING_SENTINEL}" for family in dataset.structural_none_columns]
    structural = [c for c in structural if c in frame.columns]
    other_markers = [m for m in markers if m not in layer_markers and m not in structural]

    def active(columns) -> np.ndarray:
        if not columns:
            return np.zeros((frame.height, 0), dtype=int)
        return frame.select(columns).to_numpy().astype(int)

    layer_active = active(layer_markers)
    if dataset.layer_absence_means == "blank_field":
        blank_columns = was_missing + layer_markers
        unseen_columns: list[str] = []
    else:
        blank_columns = list(was_missing)
        unseen_columns = layer_markers + other_markers
    blank_pattern = active(blank_columns)
    return {
        "blank_field_columns": blank_columns,
        "blank_field_pattern": blank_pattern,
        "blank_field_count": blank_pattern.sum(axis=1).astype(int),
        "blank_field": blank_pattern.sum(axis=1) > 0,
        "structural_none": active(structural).sum(axis=1) > 0,
        "structural_none_columns": structural,
        "unseen_level": active(unseen_columns).sum(axis=1) > 0,
        "unseen_level_columns": unseen_columns,
        "partially_observed": layer_active.sum(axis=1) > 0,
        "missing_per_layer": {layer: layer_active[:, i].astype(bool) for i, layer in enumerate(dataset.layer_order)},
        "layer_absence_means": dataset.layer_absence_means,
    }


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


@dataclass
class OperatorBundle:
    """A built classical operator plus everything needed to compare it."""

    ranker: "MultiLayerRanker"
    pi: np.ndarray
    omega: np.ndarray
    digests: dict[str, str]
    dataset_path: Path


def operator_digests(ranker: "MultiLayerRanker") -> dict[str, str]:
    """SHA-256 of the transition matrix, the factored restart matrix, and ``v``.

    The manuscript's controlled-comparison claim is that the classical and
    quantum arms differ only in the diffusion applied to an identical operator.
    Comparing these digests turns that claim into a check.
    """
    if ranker.transition_matrix is None or ranker.restart_matrix is None:
        raise ValueError("build the operator (buildGoogleOperator) before taking its digest")

    def matrix_digest(matrix) -> str:
        if issparse(matrix):
            csr = matrix.tocsr()
            csr.sort_indices()
            return sha256_bytes(csr.data, csr.indices, csr.indptr, np.array(csr.shape))
        return sha256_bytes(np.asarray(matrix, dtype=float))

    return {
        "transition_matrix": matrix_digest(ranker.transition_matrix),
        "restart_matrix": sha256_bytes(*ranker.restart_matrix.digest_parts()),
        "personalization_vector": sha256_bytes(ranker._dangling_patch_vector()),
        "N": str(ranker.N),
        "n_layers": str(ranker.n_layers),
    }


def build_ranker(
    dataset_path: str | Path, spec: NetworkSpec, config: ExperimentConfig
) -> "MultiLayerRanker":
    """A ``MultiLayerRanker`` configured entirely from the frozen config: sparse, always."""
    from classical.mulp.mulp import MultiLayerRanker

    return MultiLayerRanker(
        dataset=str(dataset_path),
        bidirectional=True,
        sparse=True,
        influence_type=config.influence_type,
        stickiness=config.stickiness,
        layer_names=spec.layer_names,
    )


def solve_classical(
    dataset_path: str | Path, spec: NetworkSpec, config: ExperimentConfig
) -> OperatorBundle:
    """Build the operator, solve for the stationary distribution, aggregate."""
    ranker = build_ranker(dataset_path, spec, config)
    ranker.buildAdjMatrix()
    ranker.buildPersonalizationMatrix()
    pi = ranker.pageRank(alpha=config.resolved_alpha())
    omega = ranker.aggregateBorrowers(pi)
    return OperatorBundle(
        ranker=ranker,
        pi=np.asarray(pi, dtype=float),
        omega=np.asarray(omega, dtype=float),
        digests=operator_digests(ranker),
        dataset_path=Path(dataset_path),
    )


def assert_operator_identity(left: dict[str, str], right: dict[str, str], context: str) -> None:
    """Two arms must have built the same operator, digest for digest."""
    differing = sorted(k for k in left if left[k] != right.get(k))
    if differing:
        detail = ", ".join(
            f"{k}: {left[k][:12]}... vs {str(right.get(k))[:12]}..." for k in differing
        )
        raise AssertionError(
            f"{context}: the classical and quantum arms built different operators "
            f"({detail}). The controlled comparison does not hold."
        )


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    """The trimmed configuration ``QMULP_SMOKE=1`` selects.

    A smoke result is a check that the pipeline runs, never a number to
    report -- the manifest records ``smoke: true`` so the two cannot be confused.
    """
    return config.with_overrides(
        reduction_check_steps=4,
        conv_tau_max={"hmeq": 2_000, "fmac": 2_000},
        hardware_shots=256,
        hardware_steps=(1, 2),
        crossfit_folds=2,
        bootstrap_reps=25,
        n_placebo_permutations=2,
        n_splits_hmeq=2,
        n_reference_resamples_fmac=2,
        hardware_class_sweep=(6, 4),
    )


def resolve_config(
    config: ExperimentConfig, smoke: bool | None = None
) -> tuple[ExperimentConfig, bool]:
    """``(config, smoke)`` with the smoke trim applied when it is requested."""
    smoke = smoke_enabled() if smoke is None else bool(smoke)
    return (smoke_config(config) if smoke else config), smoke


def config_fingerprint(
    config: ExperimentConfig,
    arm: str,
    *,
    network_spec: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Identity of the operator and score definition a cached artifact was built under.

    Everything here changes the operator, the fold partition, or the walk it
    caches, so two runs that differ in any of it must not share a cache.
    ``network_spec`` is the network's name; it is what keeps two datasets'
    caches apart. A caller that holds the operator passes its digest and the
    masked frame's checksum through ``extra`` (``operator_cache_extra``), so a
    cache keyed on the configuration alone can never be reused for a different
    operator.
    """
    parts = {
        "arm": arm,
        "network_spec": network_spec,
        "alpha": config.resolved_alpha(),
        "stickiness": float(config.stickiness),
        "influence_type": config.influence_type,
        "quantum_score": str(config.quantum_score),
        "reduction_check_steps": int(config.reduction_check_steps),
        "crossfit_folds": int(config.crossfit_folds),
        "master_seed": int(config.master_seed),
    }
    if extra:
        parts.update(extra)
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def operator_cache_extra(digests: dict[str, str], masked_frame_path: str | Path) -> dict[str, str]:
    """The ``extra`` a cache key must carry: the operator digests and the masked frame's checksum.

    A cache keyed on the configuration alone was reusable for a different
    operator (another masked frame, another restart); keyed on these it is not.
    """
    required = ("transition_matrix", "restart_matrix", "personalization_vector")
    missing = [key for key in required if key not in digests]
    if missing:
        raise KeyError(f"operator digests lack {missing}; take them from operator_digests()")
    return {
        "operator_transition": str(digests["transition_matrix"]),
        "operator_restart": str(digests["restart_matrix"]),
        "operator_v": str(digests["personalization_vector"]),
        "masked_frame_sha256": sha256_path(masked_frame_path),
    }
