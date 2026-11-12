"""The Google/credit-diffusion operator ``G = alpha*T + (1-alpha)*P`` (paper Eq. 8), dense or implicit.

Its terms -- ``v`` from the influence matrix, the factored restart ``P``, and
``T`` from a raw adjacency -- come from ``quantum.inspired.operator_terms``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, issparse

from .operator_terms import (
    as_factored_restart,
    build_restart_matrix_from_influence,
    derive_personalization_vector,
)
from .validation import (
    DEFAULT_MAX_DENSE_BYTES,
    _to_dense_float64,
    check_dense_allocation,
    validate_column_stochastic_matrix,
    validate_probability_vector,
    validate_raw_adjacency,
)


# ---------------------------------------------------------------------------
# The Google/credit-diffusion operator (paper Eq. 8): dense or implicit
# ---------------------------------------------------------------------------


class ImplicitGoogleMatrix:
    """
    Implicit representation of ``G = alpha*T + (1-alpha)*P``.

    ``P`` is the factored restart matrix and is required: a missing restart
    would silently turn the operator into the rank-one ``v 1^T`` restart,
    which is a named control and never a default. Dense G is never built.
    """

    def __init__(
        self,
        T,
        v: np.ndarray,
        alpha: float = 0.85,
        restart_matrix: Any | None = None,
    ):
        if not issparse(T):
            T = csc_matrix(T)

        self.T = T.tocsc()
        self.v = np.asarray(v, dtype=np.float64)
        self.alpha = float(alpha)
        self.n = self.T.shape[0]

        if self.T.shape[0] != self.T.shape[1]:
            raise ValueError(f"T must be square. Got {self.T.shape}")
        if self.v.shape != (self.n,):
            raise ValueError(f"v must have shape ({self.n},). Got {self.v.shape}")
        if not np.isclose(self.v.sum(), 1.0, atol=1e-8):
            raise ValueError(f"v must sum to 1. Got {self.v.sum()}")
        if np.any(self.v < -1e-12):
            raise ValueError("v has negative entries")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")

        self.restart_matrix = as_factored_restart(restart_matrix, self.v)
        if self.restart_matrix is None:
            raise ValueError(
                "restart_matrix is required: without the factored restart the operator would fall "
                "back to the rank-one v 1^T restart, which is a named control, never a default"
            )
        if self.restart_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"restart_matrix must have shape {(self.n, self.n)}; got {self.restart_matrix.shape}"
            )

        # Column-construction profiling: counts and cumulative time for sqrt(G)
        # column access, so the full-graph matrix-free walk can report whether
        # repeated sparse-column materialization is the bottleneck.
        self._column_count = 0
        self._column_seconds = 0.0

    def column(self, j: int) -> np.ndarray:
        """Return one dense column G[:, j] -- the only dense column created."""
        from time import perf_counter

        t0 = perf_counter()
        t_col = self.T[:, j].toarray().ravel()
        g_col = self.alpha * t_col + (1.0 - self.alpha) * self.restart_matrix.column(j)

        g_col[g_col < 0.0] = 0.0
        s = g_col.sum()
        if s <= 0:
            raise ValueError(f"Column {j} of G has zero mass")

        out = g_col / s
        self._column_count += 1
        self._column_seconds += perf_counter() - t0
        return out

    def sqrt_column(self, j: int) -> np.ndarray:
        return np.sqrt(self.column(j))

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        return self.alpha * (self.T @ x) + (1.0 - self.alpha) * self.restart_matrix.matvec(x)

    def column_generation_stats(self) -> str:
        """Describe cumulative sqrt(G) column construction cost for logs."""
        if self._column_count <= 0:
            return "0 columns"
        avg_ms = (self._column_seconds / self._column_count) * 1000.0
        return (
            f"{self._column_count} columns, "
            f"{self._column_seconds:.3f}s total, {avg_ms:.3f}ms/column"
        )


@dataclass(slots=True)
class GoogleMatrixDiagnostics:
    """Trace metadata for a built quantum credit-diffusion (Google) matrix G."""

    shape: tuple[int, int]
    dense: bool
    alpha: float
    # Eagerly computed for dense builds. None for implicit builds: G is never
    # materialized, so ImplicitGoogleMatrix.column() normalizes on access instead.
    max_column_sum_error: float | None


class GoogleMatrixBuilder:
    """Build G from transition T and a column-stochastic restart term.

    Two representations of the same operator:

      - dense (``build_dense``): materializes G as an ``(n, n)`` ndarray. Used
        for the lumped operator, the reduced graphs, and Qiskit parity checks.
      - implicit (``build_implicit``): wraps T + factored P in an
        ``ImplicitGoogleMatrix`` that never materializes G. Used for the full
        MuLP graph by ``quantum.inspired.szegedy_walk.SzegedyWalk.implicit``.

    Both are consumed identically by ``SzegedyWalk`` and are numerically
    equivalent -- ``tests/test_circuits.py`` locks this in.
    """

    def __init__(self) -> None:
        self.diagnostics: GoogleMatrixDiagnostics | None = None

    def build(
        self,
        T: Any,
        v: np.ndarray,
        *,
        alpha: float = 0.85,
        dense: bool,
        atol: float = 1e-8,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
        restart_matrix: Any | None = None,
    ):
        if dense:
            return self.build_dense(
                T, v, alpha=alpha, atol=atol, max_dense_bytes=max_dense_bytes,
                restart_matrix=restart_matrix,
            )
        return self.build_implicit(T, v, alpha=alpha, atol=atol, restart_matrix=restart_matrix)

    def build_dense(
        self,
        T: np.ndarray,
        v: np.ndarray,
        *,
        alpha: float = 0.85,
        atol: float = 1e-8,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
        restart_matrix: Any | None = None,
    ) -> np.ndarray:
        """Materialize and validate a small dense G.

        The restart term is added column-block by column-block: the seed
        columns from the factored matrix, then ``(1-alpha) v`` broadcast into
        the empty columns in place. No separate ``v 1^T`` array is formed, and
        a missing restart is refused rather than replaced by one.
        """
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError(f"alpha must satisfy 0 <= alpha <= 1; got {alpha}")
        T = validate_column_stochastic_matrix(T, name="T", atol=atol)
        v = validate_probability_vector(v, name="v", n=T.shape[0], atol=atol)

        n = T.shape[0]
        check_dense_allocation("Google matrix G", (n, n), np.float64, max_dense_bytes)
        check_dense_allocation("sqrt(G)", (n, n), np.float64, max_dense_bytes)

        G = (float(alpha) * T).astype(np.float64, copy=True)
        beta = 1.0 - float(alpha)
        restart = as_factored_restart(restart_matrix, v)
        if restart is None:
            raise ValueError(
                "restart_matrix is required: without it the dense operator would carry the rank-one "
                "v 1^T restart, which is a named control, never a default"
            )
        if restart.shape != (n, n):
            raise ValueError(f"restart matrix has shape {restart.shape}; expected {(n, n)}")
        G += beta * restart.seed_columns.toarray()
        if restart.fill_mask.any():
            G[:, restart.fill_mask] += beta * v[:, np.newaxis]

        min_entry = float(G.min()) if G.size else 0.0
        if min_entry < -1e-15:
            raise ValueError(
                f"G contains negative entries; shape={G.shape}, alpha={alpha}, min={min_entry:.3e}"
            )
        if min_entry < 0.0:
            G[G < 0.0] = 0.0

        G = validate_column_stochastic_matrix(G, name="G", atol=atol)
        max_colsum_error = float(np.max(np.abs(G.sum(axis=0) - 1.0))) if G.size else 0.0
        self.diagnostics = GoogleMatrixDiagnostics(
            shape=tuple(G.shape), dense=True, alpha=float(alpha),
            max_column_sum_error=max_colsum_error,
        )
        return G

    def build_implicit(
        self,
        T: Any,
        v: np.ndarray,
        *,
        alpha: float = 0.85,
        atol: float = 1e-8,
        restart_matrix: Any | None = None,
    ) -> "ImplicitGoogleMatrix":
        """Wrap T and its restart term without materializing G."""
        v = validate_probability_vector(v, name="v", atol=atol)
        operator = ImplicitGoogleMatrix(T, v, alpha=alpha, restart_matrix=restart_matrix)
        self.diagnostics = GoogleMatrixDiagnostics(
            shape=(operator.n, operator.n), dense=False, alpha=float(alpha),
            max_column_sum_error=None,
        )
        return operator


def build_google_matrix(
    T: np.ndarray,
    v: np.ndarray,
    *,
    alpha: float = 0.85,
    atol: float = 1e-8,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    restart_matrix: Any | None = None,
) -> np.ndarray:
    """Build and validate a dense Google matrix."""
    return GoogleMatrixBuilder().build_dense(
        T, v, alpha=alpha, atol=atol, max_dense_bytes=max_dense_bytes,
        restart_matrix=restart_matrix,
    )


def build_G_from_mulp(
    ranker,
    alpha: float = 0.85,
    *,
    atol: float = 1e-8,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    return_transition: bool = False,
):
    """
    Extract the dense Google matrix G and personalization vector v from a
    fitted MultiLayerRanker. Small graphs only (tests, the lumped operator).
    """
    if ranker.adj_matrix is None:
        raise ValueError("Build adjacency matrix before building the quantum Google matrix")
    if ranker.personalization_matrix is None:
        raise ValueError("Build personalization matrix before building the quantum Google matrix")

    expected_shape = (ranker.N, ranker.N)
    validate_raw_adjacency(ranker.adj_matrix, expected_shape=expected_shape, atol=atol)
    v = derive_personalization_vector(ranker.personalization_matrix, n=ranker.N, atol=atol)

    T_raw = ranker._column_normalize_adjacency()
    T = _to_dense_float64(
        T_raw, name="MuLP normalized transition matrix T", max_dense_bytes=max_dense_bytes
    )
    col_sums = T.sum(axis=0)
    dangling = col_sums <= atol
    non_dangling = ~dangling
    if np.any(non_dangling) and not np.allclose(col_sums[non_dangling], 1.0, atol=atol):
        max_colsum_error = float(np.max(np.abs(col_sums[non_dangling] - 1.0)))
        raise ValueError(
            "MuLP normalized transition matrix has non-stochastic non-dangling columns; "
            f"max_colsum_error={max_colsum_error:.3e}, atol={atol}"
        )
    if np.any(dangling):
        T[:, dangling] = v[:, np.newaxis]
    T = validate_column_stochastic_matrix(T, name="T", atol=max(atol, 1e-10))

    restart = build_restart_matrix_from_influence(ranker.personalization_matrix, v, atol=atol)
    G = GoogleMatrixBuilder().build_dense(
        T, v, alpha=alpha, atol=atol, max_dense_bytes=max_dense_bytes, restart_matrix=restart
    )

    if return_transition:
        return G.astype(np.float64, copy=False), v.astype(np.float64, copy=False), T
    return G.astype(np.float64, copy=False), v.astype(np.float64, copy=False)
