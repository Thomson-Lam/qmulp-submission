from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import LinearOperator, eigs

from preprocessing.missing_levels import assert_absence_consistent

from .adjacency import SupraAdjacencyBuilder
from .config import normalize_influence_type, validate_dataset_path, validate_stickiness
from .dataset import DEFAULT_LAYER_NAMES, MuLPDatasetLoader
from .indexing import MuLPGraphIndex
from .personalization import (
    FactoredRestart,
    PersonalizationMatrixBuilder,
    assert_personalization_vector,
    normalized_restart,
)
from .transition import TransitionMatrixNormalizer

# The stationary vector must solve G pi = pi to this residual, be a probability
# vector (nonnegative, unit sum), and not merely have survived ARPACK's maxiter.
RESIDUAL_TOL = 1e-9
SIGN_TOL = 1e-10


class MultiLayerRanker:
    """Personalized multilayer PageRank over the MuLP supra-graph.

    Parameters
    ----------
    dataset : path to the processed parquet.
    bidirectional : make membership edges undirected (interlayer edges always are).
    sparse : scipy sparse internals rather than dense numpy.
    influence_type : ``combined`` | ``intra`` | ``inter``; how a seed's restart
        column links its layer copies.
    stickiness : interlayer borrower-copy edge weight S; finite S > 0 or +inf.
    layer_names : ``("JOB", "REASON")`` (the paper's network, the default),
        ``("JOB",)`` or ``("REASON",)`` (the single-layer ablations).

    Layer-indexed state lives on ``self.index`` (:class:`MuLPGraphIndex`);
    ``job_*``/``reason_*`` name the two layers and are empty/``None`` for a
    layer the network does not have.
    """

    def __init__(
        self,
        dataset: str,
        bidirectional: bool = True,
        sparse: bool = True,
        influence_type: str = "combined",
        stickiness: float = 1.0,  # default 1.0
        layer_names=DEFAULT_LAYER_NAMES,
    ):
        parquet_path = validate_dataset_path(dataset)

        self.sparse = sparse
        self.bidirectional = bidirectional
        self.influence_type = normalize_influence_type(influence_type)
        self.stickiness = validate_stickiness(stickiness)

        self.dataset = dataset
        self.dataset_state = MuLPDatasetLoader.load(parquet_path, layer_names)
        self.df = self.dataset_state.df
        self.layer_names = list(self.dataset_state.layer_names)
        self.layer_cols = [cols for _name, cols in self.dataset_state.layer_columns]
        self.layer_missing_cols = [cols for _name, cols in self.dataset_state.layer_missing_columns]

        # Each layer's columns hold only genuine categories, so a row has at
        # most one active value per layer. Zero is legitimate and means the
        # field was never recorded: the borrower earns no membership edge there
        # and reaches the graph through its copy in the other layer. Borrowers
        # with no recorded field in either layer are already gone, dropped once
        # by the dataset's cleaner.
        self.n_unrecorded_per_layer = [
            MuLPDatasetLoader.check_at_most_one_hot(self.df, cols, name)
            for name, cols in zip(self.layer_names, self.layer_cols)
        ]
        for name, genuine, markers in zip(
            self.layer_names, self.layer_cols, self.layer_missing_cols
        ):
            assert_absence_consistent(self.df, genuine, markers, name)

        self.index = MuLPGraphIndex.from_dataset(self.dataset_state)
        self.n_rows = self.index.n_rows
        self.n_layers = self.index.n_layers
        self.N = self.index.N

        # Two-layer names, retained across the repository.
        self.job_cols = self.index.job_cols
        self.reason_cols = self.index.reason_cols
        self.job_missing_cols = list(self.dataset_state.job_missing_cols)
        self.reason_missing_cols = list(self.dataset_state.reason_missing_cols)
        unrecorded = dict(zip(self.layer_names, self.n_unrecorded_per_layer))
        self.n_unrecorded_job = unrecorded.get("JOB", 0)
        self.n_unrecorded_reason = unrecorded.get("REASON", 0)
        self.n_job = self.index.n_job
        self.n_reason = self.index.n_reason
        self.job_offset = self.index.job_offset
        self.reason_offset = self.index.reason_offset

        self.adj_matrix = None
        self.labels = list(self.index.labels)

        self.personalization_matrix: csr_matrix | np.ndarray | None = None
        self.restart_matrix: FactoredRestart | None = None
        self.transition_matrix = None
        self.leading_eigenvector_norm = None
        self.pagerank_operator: LinearOperator | None = None
        self.alpha: float | None = None

        self.adjacency_builder = SupraAdjacencyBuilder()
        self.personalization_builder = PersonalizationMatrixBuilder()
        self.transition_normalizer = TransitionMatrixNormalizer()

        # Public trace hooks for debugging component behavior without reaching
        # into builder internals.
        self.last_adjacency_diagnostics = None
        self.last_personalization_diagnostics = None
        self.last_transition_diagnostics = None

    def buildAdjMatrix(self):
        self.adj_matrix = self.adjacency_builder.build(
            self.df,
            self.index,
            sparse=self.sparse,
            bidirectional=self.bidirectional,
            stickiness=self.stickiness,
        )
        self.last_adjacency_diagnostics = self.adjacency_builder.diagnostics
        return self.adj_matrix

    # Exports adjacency matrix to Df then to parquet
    def toDf(self, matrix=None, f: Optional[str] = None):
        matrix = self.adj_matrix if matrix is None else matrix
        if matrix is None:
            raise ValueError("Adj matrix is none")

        dense = matrix.toarray() if issparse(matrix) else matrix
        out = pd.DataFrame(data=dense, columns=self.labels, index=self.labels)
        if f:
            out.to_parquet(f)
        return out

    def buildPersonalizationMatrix(self):
        if self.personalization_matrix is not None:
            raise ValueError(
                "Personalization Matrix is not None. It may have been initialized already."
            )

        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix does not exist and needs to be built first.")

        if "BAD" not in self.df.columns:
            raise ValueError("Column BAD is not in dataframe!")

        self.personalization_matrix = self.personalization_builder.build(
            self.df,
            self.index,
            sparse=self.sparse,
            influence_type=self.influence_type,
        )
        self.last_personalization_diagnostics = self.personalization_builder.diagnostics
        return self.personalization_matrix

    def setInfluenceType(self, new_type: str):
        """PageRank must be re-run afterwards for the scores to reflect the new type."""
        self.influence_type = normalize_influence_type(new_type)

    def _dangling_patch_vector(self) -> np.ndarray:
        """Probability vector used for dangling columns and for the restart fill: v."""
        return self.transition_normalizer.dangling_patch_vector(
            self.personalization_matrix,
            self.N,
        )

    def _seed_borrower_ids(self) -> list[int]:
        """Row positions whose observed default builds the operator."""
        return (
            self.df.with_row_index("borrower_id")
            .filter(pl.col("BAD").cast(pl.Int8) == 1)
            .get_column("borrower_id")
            .to_list()
        )

    def _column_normalize_adjacency(self):
        """Column-normalize the raw supra-adjacency, stickiness included."""
        transition = self.transition_normalizer.normalize(
            self.adj_matrix,
            self.personalization_matrix,
            self.N,
        )
        self.last_transition_diagnostics = self.transition_normalizer.diagnostics
        return transition

    def buildTransitionMatrix(self):
        """Public alias for the normalized MuLP transition matrix builder."""
        self.transition_matrix = self._column_normalize_adjacency()
        return self.transition_matrix

    def _normalized_restart_matrix(self, v: np.ndarray) -> FactoredRestart:
        """The restart matrix in factored form: nonempty R columns normalised, the
        rest marked to take ``v``. The dense ``v 1^T`` is never built."""
        if self.personalization_matrix is None:
            raise ValueError("The personalization matrix has not been built yet")
        return normalized_restart(self.personalization_matrix, v)

    def buildGoogleOperator(self, alpha: float = 0.85) -> LinearOperator:
        """``G = alpha*T + (1-alpha)*P`` as a matrix-free operator.

        ``T`` is sparse and ``P`` is factored, so ``G x`` costs one sparse
        matvec plus a rank-one correction and no ``N x N`` dense object exists.
        """
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix has not been built yet")
        if self.personalization_matrix is None:
            raise ValueError("Personalization matrix has not been built yet")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha must be strictly between zero and one")

        self.transition_matrix = self._column_normalize_adjacency()
        if self.personalization_matrix.sum() == 0:
            raise ValueError("The sum of the personalization matrix is 0, division by zero error")

        v = self._dangling_patch_vector()
        if self.influence_type == "combined":
            # Only "combined" links every copy of a seed, so only it puts equal
            # mass on all L. The other two are legitimate variants with a
            # different, deliberate support.
            assert_personalization_vector(v, self.index, self._seed_borrower_ids())

        self.restart_matrix = self._normalized_restart_matrix(v)
        self.restart_matrix.assert_column_stochastic()
        self.alpha = float(alpha)

        T = self.transition_matrix
        restart = self.restart_matrix
        a = float(alpha)

        def matvec(x):
            x = np.asarray(x).reshape(-1)
            return a * (T @ x) + (1.0 - a) * restart.matvec(x)

        def rmatvec(y):
            y = np.asarray(y).reshape(-1)
            return a * (T.T @ y) + (1.0 - a) * restart.rmatvec(y)

        self.pagerank_operator = LinearOperator(
            (self.N, self.N), matvec=matvec, rmatvec=rmatvec, dtype=np.float64
        )
        return self.pagerank_operator

    def pageRank(self, alpha=0.85):
        """Compute personalized PageRank via G = alpha*T + (1-alpha)*P."""
        G = self.buildGoogleOperator(alpha)

        # ARPACK's implicitly restarted Arnoldi method (Lehoucq, Sorensen & Yang
        # 1998) on the matrix-free operator; the leading eigenvector (eigenvalue
        # 1, Perron-Frobenius) is the stationary distribution. A deterministic
        # initial vector keeps reports stable across repeated runs.
        v0 = np.full(self.N, 1.0 / self.N, dtype=float)
        _, leading_eigenvectors = eigs(G, 1, v0=v0)
        leading_eigenvector = leading_eigenvectors[:, 0].real
        pi = leading_eigenvector / leading_eigenvector.sum()

        # eigs() reports neither convergence nor accuracy unless asked, and the
        # residual alone passes for a vector with mixed signs. Every score in
        # the paper is read off this vector, so it must solve G pi = pi *and*
        # be a probability distribution.
        residual = float(np.linalg.norm(G @ pi - pi, 1))
        assert residual < RESIDUAL_TOL, (
            f"ARPACK returned a vector with ||G pi - pi||_1 = {residual:.3e}; "
            "the Arnoldi iteration did not converge to the stationary distribution"
        )
        assert float(pi.min()) >= -SIGN_TOL, (
            f"stationary vector has a negative entry ({float(pi.min()):.3e}); "
            "the eigenvector is not a probability distribution"
        )
        assert (
            abs(float(pi.sum()) - 1.0) < 1e-12
        ), f"stationary vector sums to {float(pi.sum())!r}, not 1"
        pi = np.maximum(pi, 0.0)
        self.leading_eigenvector_norm = pi / pi.sum()
        return self.leading_eigenvector_norm

    def finiteHorizon(self, tau: int, alpha: float | None = None) -> np.ndarray:
        """The classical finite-horizon control ``(1/tau) sum_{t<tau} G^t v``.

        Same operator, same start vector and same horizon as the quantum arm,
        so a difference between the two cannot be attributed to truncating the
        walk rather than to the diffusion. Returns the ``(tau, N)`` history of
        ``G^t v``; the score at horizon ``tau`` is its mean over the first axis.
        """
        if tau <= 0:
            raise ValueError(f"tau must be positive; got {tau}")
        if self.pagerank_operator is None or (alpha is not None and alpha != self.alpha):
            self.buildGoogleOperator(0.85 if alpha is None else alpha)
        x = self._dangling_patch_vector()
        history = np.zeros((int(tau), self.N), dtype=float)
        for t in range(int(tau)):
            history[t] = x
            x = self.pagerank_operator @ x
        return history

    def getLeadingEigenvectorNorm(self):
        return self.leading_eigenvector_norm

    def aggregateBorrowers(self, vec: np.ndarray) -> np.ndarray:
        """Sum a supra-node vector over each borrower's layer copies."""
        vec = np.asarray(vec, dtype=float)
        if vec.shape[0] != self.N:
            raise ValueError(f"vector has {vec.shape[0]} entries, expected {self.N}")
        total = np.zeros(self.n_rows, dtype=float)
        for layer in range(self.n_layers):
            start = self.index.offset(layer)
            total += vec[start : start + self.n_rows]
        return total

    def _source_neighbors(self, source_idx: int, sparse: bool) -> np.ndarray:
        """Nodes adjacent to ``source_idx`` (column slice; symmetric under bidirectional)."""
        if self.adj_matrix is None:
            raise ValueError("Please build the adjacency matrix first and run PageRank.")

        if sparse:
            rows, _ = self.adj_matrix[:, source_idx].nonzero()
            return rows
        return np.nonzero(self.adj_matrix[:, source_idx])[0]

    def getLayerScores(self) -> tuple[list[np.ndarray], np.ndarray]:
        """Per-layer category maxima and the borrower score.

        The borrower score is the sum of the borrower's stationary probability
        over its layer copies. The per-layer maximum is the largest stationary
        probability among the category nodes the borrower is attached to in
        that layer, zero when the field was never recorded.
        """
        if self.leading_eigenvector_norm is None:
            raise ValueError("Run PageRank first before getting the scores.")

        vec = np.asarray(self.leading_eigenvector_norm)
        if vec.ndim != 1:
            raise ValueError("The eigenvector is not a 1D array")
        if vec.shape[0] != self.N:
            raise ValueError(f"eigenvector has {vec.shape[0]} entries, expected {self.N}")

        index = self.index
        n_rows = self.n_rows
        layer_max = [np.zeros(n_rows, dtype=float) for _ in range(index.n_layers)]
        borrower_scores = self.aggregateBorrowers(vec)

        for layer in range(index.n_layers):
            categories = index.category_nodes(layer)
            first, last = categories.start, categories.stop
            for borrower_id in range(n_rows):
                node = index.borrower_node(layer, borrower_id)
                neighbors = self._source_neighbors(node, self.sparse)
                attached = neighbors[(neighbors >= first) & (neighbors < last)]
                if attached.size:
                    layer_max[layer][borrower_id] = float(np.max(vec[attached]))
        return layer_max, borrower_scores

    def getScores(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(job_max, reason_max, borrower_scores)`` -- the two-layer shape."""
        layer_max, borrower_scores = self.getLayerScores()
        if len(layer_max) != 2:
            raise ValueError("getScores is the two-layer form; use getLayerScores")
        return (layer_max[0], layer_max[1], borrower_scores)

    def addScores(
        self, job_scores: np.ndarray, reason_scores: np.ndarray, borrower_scores: np.ndarray
    ) -> pl.DataFrame:
        job_col = f"job_{self.influence_type}_max"
        reason_col = f"reason_{self.influence_type}_max"
        borrower_col = f"borrower_{self.influence_type}_sum"

        if (
            len(job_scores) != self.n_rows
            or len(reason_scores) != self.n_rows
            or len(borrower_scores) != self.n_rows
        ):
            raise ValueError("Scores must always match dataframe row count")

        self.df = self.df.with_columns(
            [
                pl.Series(job_col, np.asarray(job_scores)),
                pl.Series(reason_col, np.asarray(reason_scores)),
                pl.Series(borrower_col, np.asarray(borrower_scores)),
            ]
        )

        return self.df
