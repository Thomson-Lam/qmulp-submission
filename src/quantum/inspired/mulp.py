"""The quantum analogue of ``MultiLayerRanker``, and the finite-shot estimator of its score.

:class:`QuantumWalkRanker` reuses the classical MuLP pipeline and adds the
personalization vector, the matrix-free Szegedy walk, its initial state and
the time-averaged quantum PageRank.

The finite-shot estimator
-------------------------
The finite-shot estimator of the Szegedy score, and its ensemble over sampling
seeds. Sampling happens **per recorded step**, **over the N supra-nodes**, and
**before** both the layer aggregation and the time average, because that is
what a device would do: one measurement of the position register per shot, per
circuit. The exact per-step marginals are taken as an argument so every shot
budget and every seed can be drawn from one evolution.
"""

from __future__ import annotations

import logging

import numpy as np

from classical.mulp.mulp import MultiLayerRanker
from quantum.inspired.operator_terms import build_restart_matrix_from_influence, derive_personalization_vector
from quantum.inspired.szegedy_walk import SzegedyWalk


_logger = logging.getLogger("quantum.inspired.mulp")


class QuantumWalkRanker:
    """
    Quantum analogue of MultiLayerRanker.

    It reuses the classical MuLP pipeline:

        buildAdjMatrix()
        buildPersonalizationMatrix()
        buildTransitionMatrix()

    Then adds:

        buildPersonalizationVector()
        buildSzegedyWalk()
        buildInitialState()
        quantumPageRank()

    The quantum walk is one ``quantum.inspired.szegedy_walk.SzegedyWalk`` instance
    built through the matrix-free Google-operator constructor
    (``SzegedyWalk.implicit(...)``): dense G, sqrt(G) and W are never built.
    The restart matrix is the same factored object the classical arm solves
    with. The state Psi is still N x N because Szegedy's walk lives in
    C^N tensor C^N.
    """

    def __init__(
        self,
        dataset: str | None = None,
        ranker: MultiLayerRanker | None = None,
        bidirectional: bool = True,
        sparse: bool = True,
        influence_type: str = "combined",
        stickiness: float = 1.0,
        alpha: float = 0.85,
        layer_names=("JOB", "REASON"),
    ):
        if ranker is None and dataset is None:
            raise ValueError("Provide either dataset or ranker")

        if ranker is None:
            ranker = MultiLayerRanker(
                dataset=dataset,
                bidirectional=bidirectional,
                sparse=sparse,
                influence_type=influence_type,
                stickiness=stickiness,
                layer_names=layer_names,
            )

        self.ranker = ranker

        self.dataset = ranker.dataset
        self.df = ranker.df

        self.sparse = ranker.sparse
        self.bidirectional = ranker.bidirectional
        self.influence_type = ranker.influence_type
        self.stickiness = ranker.stickiness

        self.n_rows = ranker.n_rows
        self.N = ranker.N
        self.job_offset = ranker.job_offset
        self.reason_offset = ranker.reason_offset
        self.labels = ranker.labels

        self.alpha = float(alpha)
        self.measure_register = "source"

        self.adj_matrix = None
        self.personalization_matrix = None
        self.transition_matrix = None
        self.personalization_vector = None

        self.szegedy_walk: SzegedyWalk | None = None
        self.initial_state = None
        self.last_operator_diagnostics = None

        # One supra-node index array per layer; the borrower score sums over
        # all of them, matching MultiLayerRanker.aggregateBorrowers.
        borrowers = np.arange(self.n_rows, dtype=int)
        self.borrower_layer_indices = [
            borrowers + ranker.index.offset(layer) for layer in range(ranker.index.n_layers)
        ]
        self.borrower_layer1_indices = self.borrower_layer_indices[0]
        self.borrower_layer2_indices = (
            self.borrower_layer_indices[1] if len(self.borrower_layer_indices) > 1 else None
        )

    # ============================================================
    # Classical-style build methods
    # ============================================================

    def buildAdjMatrix(self):
        if self.ranker.adj_matrix is None:
            self.ranker.buildAdjMatrix()
        self.adj_matrix = self.ranker.adj_matrix
        return self.adj_matrix

    def buildPersonalizationMatrix(self):
        if self.ranker.personalization_matrix is None:
            self.ranker.buildPersonalizationMatrix()
        self.personalization_matrix = self.ranker.personalization_matrix
        return self.personalization_matrix

    def buildTransitionMatrix(self):
        if self.ranker.transition_matrix is None:
            self.ranker.buildTransitionMatrix()
        self.transition_matrix = self.ranker.transition_matrix
        return self.transition_matrix

    def buildPersonalizationVector(self):
        """v from the MuLP personalization matrix R (row-sums of R, normalized)."""
        if self.personalization_matrix is None:
            self.buildPersonalizationMatrix()
        self.personalization_vector = derive_personalization_vector(
            self.personalization_matrix, n=self.N
        )
        return self.personalization_vector

    def buildSzegedyWalk(self) -> SzegedyWalk:
        """The matrix-free Google-operator walk, without materializing dense G."""
        if self.transition_matrix is None:
            self.buildTransitionMatrix()
        if self.personalization_vector is None:
            self.buildPersonalizationVector()

        restart_matrix = build_restart_matrix_from_influence(
            self.personalization_matrix,
            self.personalization_vector,
        )
        self.szegedy_walk = SzegedyWalk.implicit(
            self.transition_matrix,
            self.personalization_vector,
            self.borrower_layer1_indices,
            self.borrower_layer2_indices,
            alpha=self.alpha,
            measure_register=self.measure_register,
            restart_matrix=restart_matrix,
        )
        self.last_operator_diagnostics = self.szegedy_walk.operator_diagnostics

        T = self.transition_matrix
        nnz = int(T.nnz) if hasattr(T, "nnz") else None
        _logger.info(
            "built matrix-free Szegedy walk: borrowers=%d supranodes=%d layers=%d alpha=%.3f "
            "stickiness=%s influence=%s evolution=W^2 measure_register=%s T_nnz=%s "
            "state_bytes~%.1fMiB",
            self.n_rows,
            self.szegedy_walk.n,
            len(self.borrower_layer_indices),
            self.alpha,
            self.stickiness,
            self.influence_type,
            self.measure_register,
            nnz,
            (self.szegedy_walk.n**2) * 16 / (1024**2),
        )
        return self.szegedy_walk

    def buildInitialState(self):
        """Psi_0 = sum_j sqrt(v_j) |j>|g_j>, stored as Psi[source, target]."""
        if self.szegedy_walk is None:
            self.buildSzegedyWalk()
        self.initial_state = self.szegedy_walk._build_initial_state()
        return self.initial_state

    # ============================================================
    # Public scoring methods
    # ============================================================

    def quantumPageRank(
        self, tau: int = 50, verbose: bool = False, validate_probabilities: bool = False
    ) -> np.ndarray:
        """Quantum analogue of ranker.pageRank(): borrower-level omega_Q."""
        if self.szegedy_walk is None:
            self.buildSzegedyWalk()
        return self.szegedy_walk.run(
            tau=tau, verbose=verbose, validate_probabilities=validate_probabilities
        )

    def quantumPageRankWithHistory(
        self, tau: int = 50, *, validate_probabilities: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Time-averaged borrower scores and every recorded borrower marginal."""
        if self.szegedy_walk is None:
            self.buildSzegedyWalk()
        return self.szegedy_walk.run_with_history(
            tau=tau, validate_probabilities=validate_probabilities
        )

    def run(self, tau: int = 50, verbose: bool = False, validate_probabilities: bool = False):
        """Alias for compatibility with existing qwalk code."""
        return self.quantumPageRank(
            tau=tau, verbose=verbose, validate_probabilities=validate_probabilities
        )

    def quantumPageRankFiniteShot(self, tau: int, shots: int, rng) -> np.ndarray:
        """Finite-shot borrower scores; see :func:`run_finite_shot`."""
        if self.szegedy_walk is None:
            self.buildSzegedyWalk()
        return run_finite_shot(self.szegedy_walk, tau=tau, shots=shots, rng=rng)

    def finiteShotEnsemble(self, tau: int, shots: int, seeds, *, supra_history=None):
        """``(mean, sd, draws)`` of the finite-shot estimator over sampling seeds."""
        if self.szegedy_walk is None:
            self.buildSzegedyWalk()
        return finite_shot_ensemble(
            self.szegedy_walk, tau=tau, shots=shots, seeds=seeds, supra_history=supra_history
        )


# ---------------------------------------------------------------------------
# The finite-shot estimator
# ---------------------------------------------------------------------------


_shot_logger = logging.getLogger("quantum.inspired.szegedy")


def sample_finite_shot(walk, supra_history: np.ndarray, shots: int, rng) -> np.ndarray:
    """Draw one finite-shot measurement record from recorded step marginals.

    Per recorded step ``t``: draw ``Multinomial(shots, p_t)`` over all ``N``
    supra-nodes, aggregate the counts over the borrower's ``L`` layer copies,
    divide by ``shots``. The ``tau`` estimates are then averaged, exactly as
    ``SzegedyWalk.run`` averages the exact per-step marginals.

    Sampling once at the end from a renormalized borrower distribution -- the
    earlier implementation -- both discards the per-step variance and spends
    the whole shot budget on a single distribution: at 8,192 shots over 1,192
    borrowers that is about seven expected counts each, roughly 38% relative
    error, and a Spearman correlation against the exact scores of 0.162.

    ``supra_history`` is the exact, deterministic marginal per step. The walk
    is deterministic and only the measurement is random, so two seeds sharing
    ``p_t`` is the same state measured twice, which is exactly what the
    ensemble is meant to model.
    """
    if int(shots) <= 0:
        raise ValueError(f"shots must be positive; got {shots}")
    shots = int(shots)
    supra_history = np.asarray(supra_history, dtype=float)
    if supra_history.ndim != 2 or supra_history.shape[1] != walk.n:
        raise ValueError(
            f"supra_history must have shape (tau, {walk.n}); got {supra_history.shape}"
        )

    estimate = np.zeros(walk.n_borrowers, dtype=float)
    max_renormalization = 0.0
    for probabilities in supra_history:
        # The marginal sums to 1 only up to floating-point error, and
        # multinomial sampling rejects a vector that sums above 1.
        probabilities = np.clip(probabilities, 0.0, None)
        total = float(probabilities.sum())
        max_renormalization = max(max_renormalization, abs(total - 1.0))

        counts = rng.multinomial(shots, probabilities / total)
        per_step = np.zeros(walk.n_borrowers, dtype=float)
        for layer_indices in walk.borrower_layer_indices:
            per_step += counts[layer_indices]
        estimate += per_step / shots

    if max_renormalization > max(walk.atol, 1e-8):
        raise AssertionError(
            "source-register marginal drifted from unit mass by "
            f"{max_renormalization:.3e} during finite-shot sampling"
        )
    return estimate / supra_history.shape[0]


def run_finite_shot(walk, tau: int, shots: int, rng) -> np.ndarray:
    """Evolve the walk for ``tau`` steps and take one finite-shot record.

    Convenience wrapper: it runs the exact evolution itself, so it costs a
    full walk. When more than one record is wanted from the same walk, take
    the marginals once and call :func:`sample_finite_shot` per draw.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}")
    return sample_finite_shot(walk, walk.supra_node_history(tau), shots, rng)


def finite_shot_ensemble(
    walk, tau: int, shots: int, seeds, *, supra_history: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean, sd, draws)`` over independent sampling seeds.

    Sampling noise is a property of the estimator, not of the network, so a
    single seed reports a draw rather than a result. Pass ``supra_history`` to
    reuse an evolution already run; omitting it runs the walk once here.
    """
    if supra_history is None:
        supra_history = walk.supra_node_history(tau)
    supra_history = np.asarray(supra_history, dtype=float)[:tau]
    if supra_history.shape[0] != tau:
        raise ValueError(
            f"supra_history holds {supra_history.shape[0]} steps, fewer than tau={tau}"
        )
    draws = np.stack(
        [
            sample_finite_shot(walk, supra_history, shots, np.random.default_rng(int(seed)))
            for seed in seeds
        ]
    )
    _shot_logger.info(
        "finite-shot ensemble: tau=%d shots=%d seeds=%d (one evolution reused)",
        tau, int(shots), draws.shape[0],
    )
    return draws.mean(axis=0), draws.std(axis=0, ddof=1 if draws.shape[0] > 1 else 0), draws
