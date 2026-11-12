"""The MuLP restart (influence) matrix, and its factored column-stochastic form.

A seed borrower -- one whose default is visible to the operator -- owns a
personalized restart column ``c_i``. Which of its layer copies that column
links is the influence type:

- ``intra``: self-loops only, each copy restarts to itself;
- ``inter``: cross-copy links only, each copy restarts to the borrower's other
  copy (undefined on a single-layer network, which has no other copy);
- ``combined``: both, so a seed's column is the full ``L x L`` block of ones
  over its copies and, once column-normalized, places mass ``1/L`` on each.

Every other column is empty here and is filled with ``v`` downstream, so a
non-seed restarts like any other node. That structural difference between seed
and non-seed columns is why a lumping key must contain the (masked) default
indicator: the two are not interchangeable states.

The restart matrix ``P`` is never materialised. ``P = R_norm + v 1_empty^T``
is rank-one outside the seed columns, and the dense outer product costs
``8 N^2`` bytes for nothing: :class:`FactoredRestart` keeps ``R_norm`` sparse,
the empty-column mask, and ``v``, and applies ``v (1_empty^T x)`` inside the
matvec.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csc_matrix, diags, issparse, lil_matrix
from scipy.sparse.linalg import LinearOperator

from .config import normalize_influence_type
from .indexing import MuLPGraphIndex


@dataclass(slots=True)
class PersonalizationDiagnostics:
    shape: tuple[int, int]
    sparse: bool
    influence_type: str
    bad_borrower_count: int
    nonzero_count: int
    n_layers: int


class PersonalizationMatrixBuilder:
    def __init__(self) -> None:
        self.diagnostics: PersonalizationDiagnostics | None = None

    def build(
        self,
        frame: pl.DataFrame,
        index: MuLPGraphIndex,
        *,
        sparse: bool,
        influence_type: str,
    ) -> Any:
        if "BAD" not in frame.columns:
            raise ValueError("Column BAD is not in dataframe!")

        influence_type = normalize_influence_type(influence_type)
        L = index.n_layers
        if L == 1 and influence_type == "inter":
            raise ValueError(
                "influence_type='inter' links a seed's layer copies to each other; a "
                "single-layer network has no other copy, so use 'intra' or 'combined'"
            )

        # Row position in `frame` is the borrower_id used throughout the index.
        bad_borrower_ids = (
            frame.with_row_index("borrower_id")
            .filter(pl.col("BAD").cast(pl.Int8) == 1)
            .get_column("borrower_id")
            .to_list()
        )
        if not bad_borrower_ids:
            raise ValueError("Invalid dataset: There are no BAD borrowers.")

        R: Any = (
            lil_matrix((index.N, index.N), dtype=np.int8)
            if sparse
            else np.zeros((index.N, index.N), dtype=np.int8)
        )

        for borrower_id in bad_borrower_ids:
            copies = [index.borrower_node(l, borrower_id) for l in range(L)]
            if influence_type in ("intra", "combined"):
                for node in copies:
                    R[node, node] = 1
            if influence_type in ("inter", "combined"):
                for a, b in permutations(copies, 2):
                    R[a, b] = 1

        nonzero_count = int(R.nnz) if sparse else int(np.count_nonzero(R))
        self.diagnostics = PersonalizationDiagnostics(
            shape=(index.N, index.N),
            sparse=sparse,
            influence_type=influence_type,
            bad_borrower_count=len(bad_borrower_ids),
            nonzero_count=nonzero_count,
            n_layers=L,
        )
        return R


def assert_personalization_vector(
    v: np.ndarray, index: MuLPGraphIndex, seed_borrower_ids, *, atol: float = 1e-12
) -> None:
    """``v`` must place exactly ``1/(L*|D|)`` on each seed copy and 0 elsewhere.

    This is the property the whole design rests on: mass enters the walk only
    through observed defaults, spread evenly over a seed's layer copies.
    """
    v = np.asarray(v, dtype=float)
    seeds = [int(i) for i in seed_borrower_ids]
    if v.shape != (index.N,):
        raise AssertionError(f"v has shape {v.shape}, expected ({index.N},)")
    if not seeds:
        raise AssertionError("no seed borrowers: v would have no support")

    L = index.n_layers
    expected_mass = 1.0 / (L * len(seeds))
    seed_nodes = {index.borrower_node(l, i) for l in range(L) for i in seeds}

    wrong = [
        (node, float(v[node]))
        for node in sorted(seed_nodes)
        if abs(float(v[node]) - expected_mass) > atol
    ]
    if wrong:
        raise AssertionError(
            f"v must place 1/(L*|D|) = {expected_mass:.6e} on each of the "
            f"{len(seed_nodes)} seed copies; {len(wrong)} differ (sample {wrong[:5]})"
        )

    leaked = float(v.sum()) - expected_mass * len(seed_nodes)
    if abs(leaked) > atol * max(1, len(seed_nodes)):
        raise AssertionError(
            f"v places {leaked:.3e} of its mass outside the seed copies; it must be zero"
        )


# ---------------------------------------------------------------------------
# The factored restart matrix
# ---------------------------------------------------------------------------


@dataclass
class FactoredRestart:
    """``P = seed_columns + v 1_fill^T`` without the outer product.

    ``seed_columns`` holds the column-normalised nonempty columns of ``R`` and
    is zero everywhere else; ``fill_mask`` marks the columns that were empty
    and restart to ``v``. ``P x = seed_columns x + v * sum(x[fill_mask])``.
    """

    seed_columns: csc_matrix
    fill_mask: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        self.seed_columns = csc_matrix(self.seed_columns, dtype=np.float64)
        self.fill_mask = np.asarray(self.fill_mask, dtype=bool)
        self.v = np.asarray(self.v, dtype=np.float64)
        n = self.seed_columns.shape[0]
        if self.seed_columns.shape != (n, n):
            raise ValueError(f"seed_columns must be square; got {self.seed_columns.shape}")
        if self.fill_mask.shape != (n,) or self.v.shape != (n,):
            raise ValueError("fill_mask and v must have one entry per node")

    @property
    def shape(self) -> tuple[int, int]:
        return self.seed_columns.shape

    @property
    def n(self) -> int:
        return self.seed_columns.shape[0]

    @property
    def n_seed_columns(self) -> int:
        return int((~self.fill_mask).sum())

    def column(self, j: int) -> np.ndarray:
        """Dense column ``P[:, j]`` -- the only dense object this class creates."""
        if self.fill_mask[j]:
            return self.v.copy()
        return self.seed_columns[:, j].toarray().ravel()

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return self.seed_columns @ x + self.v * x[self.fill_mask].sum()

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        return self.seed_columns.T @ y + self.fill_mask * (self.v @ y)

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(self.shape, matvec=self.matvec, rmatvec=self.rmatvec, dtype=float)

    def digest_parts(self) -> list[np.ndarray]:
        """Arrays whose bytes identify this matrix exactly."""
        csc = self.seed_columns.tocsc()
        csc.sort_indices()
        return [csc.data, csc.indices, csc.indptr, self.fill_mask, self.v, np.array(self.shape)]

    def assert_column_stochastic(self, atol: float = 1e-10) -> None:
        col_sums = np.asarray(self.seed_columns.sum(axis=0)).ravel() + self.fill_mask * self.v.sum()
        worst = float(np.max(np.abs(col_sums - 1.0))) if col_sums.size else 0.0
        if worst > atol:
            raise AssertionError(f"restart matrix is not column-stochastic (error {worst:.3e})")


def normalized_restart(R: Any, v: np.ndarray, *, atol: float = 0.0) -> FactoredRestart:
    """Normalize the nonempty columns of ``R``; mark the empty ones to take ``v``."""
    v = np.asarray(v, dtype=np.float64)
    matrix = (R if issparse(R) else csc_matrix(np.asarray(R))).tocsc().astype(np.float64)
    if matrix.shape != (v.size, v.size):
        raise ValueError(f"R has shape {matrix.shape}; expected ({v.size}, {v.size})")
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    nonzero = col_sums > atol
    scales = np.zeros(v.size, dtype=float)
    scales[nonzero] = 1.0 / col_sums[nonzero]
    seed_columns = (matrix @ diags(scales)).tocsc()
    return FactoredRestart(seed_columns=seed_columns, fill_mask=~nonzero, v=v)
