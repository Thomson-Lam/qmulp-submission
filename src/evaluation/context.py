"""One dataset's frozen inputs, twice: to build and score operators, and to read the scores the stages saved.

Both contexts share one evaluation population, the test partition of the
frozen split, and record every artifact they open so the tables and the
manifest can list every input (:class:`Population`).

Building and scoring operators (:class:`DatasetContext`)
--------------------------------------------------------
Every block that scores operators reads the frozen split, the network
registry on the processed frame and the observation flags through this
object. Every score vector is canonicalized at the tie tolerance before any
agreement measure sees it. The primary operator of each network is routed
once and cached.

Reading saved scores (:class:`Inputs`), and the paired comparison the tables share
----------------------------------------------------------------------------------
The scoring stage (``quantum.mechanism.scores_stage``) writes one
``scores_<network>.parquet`` per network with the two arms' scores on every
borrower -- ``classical`` (stationary), ``qppr_limit`` (the QPPR score),
``qppr_limit_destination``, ``qppr_tau_star`` (a check) and
``classical_tau_star`` (a control) -- beside the observation flags. The
tables read those files through :class:`Inputs`, restricted to the test
partition and canonicalized at the tie tolerance. Every interval comes from
the one shared bootstrap matrix the data stage wrote, and every comparative
row is a **paired difference** on those common replicates
(:func:`paired_difference_row`), never two marginal intervals.

Nothing here slices a walk history: there is no fixed horizon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config import ExperimentConfig
from datasets import DatasetConfig
from evaluation.masked_eval import Partitions
from evaluation.reporting import (
    agreement_with,
    bootstrap_replicates,
    expected_top_k_overlap,
    interval,
    metrics_with_ci,
    paired_difference,
    ranking_point_metrics,
)
from evaluation.ties import canonicalize, tie_report
from fingerprint import load_bootstrap_indices, load_frozen_partitions, observation_flags
from paths import classical_scores_path, convergence_path, scores_path
from quantum.lumped_operator import unfold_class_scores
from quantum.lumping_router import RoutedOperator
from quantum.mechanism.classical import stationary
from quantum.mechanism.limit import LimitingScores, limiting_scores
from quantum.mechanism.variants import OperatorVariant, primary_variant, route_variant


# ---------------------------------------------------------------------------
# The evaluation population both contexts share
# ---------------------------------------------------------------------------


class Population:
    """The test partition of the frozen split, its observation flags and the record of every artifact opened.

    A subclass sets ``partitions``, ``flags`` and ``inputs``.
    """

    @property
    def test_idx(self) -> np.ndarray:
        return self.partitions.test_idx

    def flag_test(self, name: str) -> np.ndarray:
        """One observation flag restricted to the test partition."""
        return np.asarray(self.flags[name])[self.test_idx]

    def _record(self, path: Path) -> Path:
        if path not in self.inputs:
            self.inputs.append(path)
        return path


# ---------------------------------------------------------------------------
# Building and scoring operators
# ---------------------------------------------------------------------------


AGREEMENT_MEASURES = ("spearman", "kendall", "normalized_l1", "top_k_overlap")


@dataclass
class ScoredOperator:
    """One routed operator with both arms scored on it, every borrower."""

    variant: OperatorVariant
    routed: RoutedOperator
    quantum: LimitingScores
    # Stationary distribution over compact positions, and unfolded to the covered borrowers.
    classical_positions: np.ndarray
    classical: np.ndarray

    @property
    def borrower_ids(self) -> np.ndarray:
        return self.quantum.borrower_ids

    @property
    def nodes(self):
        return self.variant.nodes


def score_operator(variant: OperatorVariant, config: ExperimentConfig, *, keep_full: bool = False) -> ScoredOperator:
    """Route ``variant`` and score both arms on it: the limiting QPPR and the stationary classical score."""
    routed = route_variant(variant, config, keep_full=keep_full)
    quantum = limiting_scores(variant.nodes, routed, config)
    pi = stationary(routed.G)
    _ids, classical = unfold_class_scores(variant.nodes, pi)
    return ScoredOperator(variant=variant, routed=routed, quantum=quantum, classical_positions=pi, classical=classical)


@dataclass
class DatasetContext(Population):
    """The frozen inputs of one dataset and the helpers every block shares."""

    dataset: DatasetConfig
    config: ExperimentConfig
    partitions: Partitions
    specs: dict[str, Any]
    observed_states: dict[str, int]
    flags: dict[str, Any]
    output: Path
    inputs: list[Path] = field(default_factory=list)
    _primary: dict[str, ScoredOperator] = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: DatasetConfig, config: ExperimentConfig, *, stage_key: str = "mechanism"):
        partitions = load_frozen_partitions(dataset)
        specs = dataset.network_specs(partitions.frame)
        observed = dataset.assert_expected_states(specs)
        return cls(
            dataset=dataset, config=config, partitions=partitions, specs=specs, observed_states=observed,
            flags=observation_flags(partitions.frame, dataset), output=dataset.stage_dir(stage_key),
            inputs=[dataset.processed_path(), dataset.frozen_split_path()],
        )

    # -- population --------------------------------------------------------

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def primary_network(self) -> str:
        return self.dataset.primary

    @property
    def hidden_ids(self) -> np.ndarray:
        """The borrowers whose labels every operator may not see: the test partition."""
        return self.partitions.test_idx

    @property
    def y_test(self) -> np.ndarray:
        return self.partitions.y_true[self.test_idx]

    @property
    def n_borrowers(self) -> int:
        return int(self.partitions.frame.height)

    @property
    def k(self) -> int:
        return int(self.config.primary_k)

    @property
    def decimals(self) -> int:
        return int(self.config.tie_tolerance)

    @property
    def tau_max(self) -> int:
        return self.config.tau_max_for(self.name)

    def evaluate(self, scores: np.ndarray) -> np.ndarray:
        """A full-length borrower vector restricted to the test partition and canonicalized."""
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.n_borrowers,):
            raise ValueError(f"expected one score per borrower; got shape {scores.shape}")
        return canonicalize(scores[self.test_idx], self.decimals)

    def test(self, borrower_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """``scores`` aligned with ``borrower_ids`` (a subset of borrowers) on the test members, canonicalized."""
        ids = np.asarray(borrower_ids, dtype=int)
        mask = np.isin(ids, self.test_idx)
        return canonicalize(np.asarray(scores, dtype=float)[mask], self.decimals)

    def agreement(self, scores, reference, suffix: str) -> dict[str, float]:
        """Spearman, Kendall, mean absolute difference and the top-k overlap of ``scores`` against ``reference``."""
        return agreement_with(scores, reference, self.k, decimals=self.decimals, suffix=suffix)

    def row(self, network: str, **values: Any) -> dict[str, Any]:
        return {"dataset": self.name, "network": network, **values}

    # -- the primary operators ------------------------------------------------

    def primary(self, network: str, *, keep_full: bool = False) -> ScoredOperator:
        """The primary operator of ``network`` with both arms scored; routed once and cached."""
        if network not in self._primary or (keep_full and self._primary[network].routed.G_full is None):
            variant = primary_variant(self.partitions, self.specs[network], self.config, self.hidden_ids)
            self._primary[network] = score_operator(variant, self.config, keep_full=keep_full)
        return self._primary[network]

    def drop_full(self, network: str) -> None:
        """Release the unlumped operator of a cached primary (it is 8 N^2 bytes)."""
        scored = self._primary.get(network)
        if scored is not None and scored.routed.G_full is not None:
            scored.routed = RoutedOperator(G=scored.routed.G, v=scored.routed.v, K=scored.routed.K,
                                           route=scored.routed.route, D_bar=scored.routed.D_bar,
                                           H=scored.routed.H, residuals=scored.routed.residuals)


# ---------------------------------------------------------------------------
# Reading saved scores
# ---------------------------------------------------------------------------


CLASSICAL = "classical"
QPPR = "qppr_limit"
QPPR_DESTINATION = "qppr_limit_destination"
QPPR_TAU_STAR = "qppr_tau_star"
CLASSICAL_TAU_STAR = "classical_tau_star"
SCORE_NAMES = (CLASSICAL, QPPR, QPPR_DESTINATION, QPPR_TAU_STAR, CLASSICAL_TAU_STAR)
PAIRED = "qppr_minus_classical"
POINT_METRICS = ("pr_auc", "roc_auc")


def tie_columns(scores: np.ndarray, decimals: int, k: int) -> dict[str, int]:
    """Distinct levels and the crowd on the rank-k boundary, as headline columns."""
    report = tie_report(scores, decimals, (k,))
    return {"distinct_levels": report["distinct_levels"],
            f"n_tied_at_{int(k)}": report[f"k_{int(k)}"]["n_tied_at_boundary"]}


# ---------------------------------------------------------------------------
# Loading stage outputs
# ---------------------------------------------------------------------------


def require_artifact(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run stage {stage} first")
    return path


class Inputs(Population):
    """Everything the evaluation tables read for one dataset, loaded once."""

    def __init__(self, config: ExperimentConfig, dataset: DatasetConfig) -> None:
        self.config = config
        self.dataset = dataset
        self.quantum_stage = dataset.stage("quantum")
        self.classical_stage = dataset.stage("classical")
        self.partitions = load_frozen_partitions(dataset)
        self.y = self.partitions.y_true
        self.y_test = self.y[self.test_idx]
        self.boot = load_bootstrap_indices(config, self.partitions, dataset)
        self.k = int(config.primary_k)
        self.decimals = int(config.tie_tolerance)
        self.flags = observation_flags(self.partitions.frame, dataset)
        self.inputs: list[Path] = [dataset.processed_path(), dataset.frozen_split_path(), dataset.bootstrap_path()]
        self._frames: dict[str, pl.DataFrame] = {}

    def specs(self):
        """The dataset's network specifications, discovered on the loaded frame where needed."""
        return self.dataset.network_specs(self.partitions.frame)

    def frame(self, network: str) -> pl.DataFrame:
        """The scoring stage's per-borrower frame of ``network``."""
        if network not in self._frames:
            path = require_artifact(scores_path(network, stage=self.quantum_stage), "04")
            self._frames[network] = pl.read_parquet(self._record(path))
        return self._frames[network]

    def full(self, network: str, name: str) -> np.ndarray:
        """One score column over every borrower, uncanonicalized."""
        if name not in SCORE_NAMES:
            raise KeyError(f"unknown score {name!r}; expected one of {SCORE_NAMES}")
        return self.frame(network)[name].to_numpy().astype(float)

    def test(self, network: str, name: str) -> np.ndarray:
        """One score column on the test partition, canonicalized at the tie tolerance."""
        return canonicalize(self.full(network, name)[self.test_idx], self.decimals)

    def classical(self, network: str) -> np.ndarray:
        return self.test(network, CLASSICAL)

    def qppr(self, network: str) -> np.ndarray:
        return self.test(network, QPPR)

    def convergence(self, network: str) -> dict[str, Any]:
        path = require_artifact(convergence_path(network, stage=self.quantum_stage), "04")
        return json.loads(self._record(path).read_text())

    def classical_crossfit(self, network: str) -> tuple[np.ndarray, np.ndarray]:
        """Stage 03's cross-fitted classical scores and fold index over every borrower."""
        path = require_artifact(classical_scores_path(network, stage=self.classical_stage), "03")
        table = pl.read_parquet(self._record(path))
        if "omega_C_crossfit" not in table.columns:
            raise FileNotFoundError(f"{path} carries no cross-fitted column; run the folds first")
        return table["omega_C_crossfit"].to_numpy().astype(float), table["fold_index"].to_numpy().astype(int)

    # -- metrics on the shared bootstrap ----------------------------------------

    def metrics(self, scores_test: np.ndarray) -> dict[str, Any]:
        return metrics_with_ci(scores_test, self.y_test, self.boot, k=self.k, decimals=self.decimals)

    def replicates(self, scores_test: np.ndarray) -> dict[str, np.ndarray]:
        return bootstrap_replicates(scores_test, self.y_test, self.boot, k=self.k, decimals=self.decimals)

    def point(self, scores_test: np.ndarray) -> dict[str, float]:
        return ranking_point_metrics(scores_test, self.y_test, (self.k,), decimals=self.decimals)

    def overlap_replicates(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Expected top-k overlap of two rankings on every shared resample."""
        out = np.full(self.boot.shape[0], np.nan)
        for rep, idx in enumerate(self.boot):
            out[rep] = expected_top_k_overlap(a[idx], b[idx], min(self.k, int(idx.size)))
        return out

    def agreement_replicates(self, a: np.ndarray, b: np.ndarray, measure) -> np.ndarray:
        """Any pairwise agreement ``measure(a, b)`` on every shared resample."""
        out = np.full(self.boot.shape[0], np.nan)
        for rep, idx in enumerate(self.boot):
            out[rep] = measure(a[idx], b[idx])
        return out


def paired_difference_row(
    data: Inputs, quantum: np.ndarray, classical: np.ndarray, *, base: dict[str, Any], overlap_key: str
) -> dict[str, Any]:
    """``qppr - classical`` on the shared replicates, one row in the metric columns.

    PR-AUC, ROC-AUC, precision@k and lift@k each get the observed difference,
    its 95% interval and ``p_delta_pos / zero / neg``. The top-k overlap
    between the two rankings is itself a paired quantity: its entry is the
    observed overlap with its bootstrap interval under ``overlap_key``.
    """
    k = data.k
    rep_q, rep_c = data.replicates(quantum), data.replicates(classical)
    point_q, point_c = data.point(quantum), data.point(classical)
    row: dict[str, Any] = dict(base)
    for metric in ("pr_auc", "roc_auc", f"precision_at_{k}", f"lift_at_{k}"):
        rep_key = metric.replace(f"_{k}", "_k")
        diff = paired_difference(rep_q[rep_key], rep_c[rep_key], point_q[metric], point_c[metric])
        row[metric] = diff["observed"]
        row[f"{metric}_ci_lo"] = diff["ci_lo"]
        row[f"{metric}_ci_hi"] = diff["ci_hi"]
        for sign in ("pos", "zero", "neg"):
            row[f"{metric}_p_delta_{sign}"] = diff[f"p_delta_{sign}"]
    lo, hi = interval(data.overlap_replicates(quantum, classical))
    row[overlap_key] = expected_top_k_overlap(quantum, classical, k)
    row[f"{overlap_key}_ci_lo"] = lo
    row[f"{overlap_key}_ci_hi"] = hi
    return row
