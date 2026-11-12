from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse


"""
Dataclass for tracing metadata when normalizing transition matrix, for diagnostics.
"""


@dataclass(slots=True)
class TransitionDiagnostics:

    shape: tuple[int, int]
    sparse: bool
    dangling_column_count: int
    max_column_sum_error: float


"""
Builds a column-stochastic transition matrix from raw MuLP supra-adjacency for PageRank.
"""


class TransitionMatrixNormalizer:
    def __init__(self) -> None:
        self.diagnostics: TransitionDiagnostics | None = None

    """
    Returns the probability vector dangling transition columns are patched with: the
    personalization matrix's normalised row sums, i.e. ``v`` (``d = v``). There is no
    fallback: a transition matrix built before the personalization matrix exists would
    silently patch dangling columns with ``1/N``, a different operator, so that is refused.
    """

    @staticmethod
    def dangling_patch_vector(personalization_matrix: Any | None, N: int) -> np.ndarray:
        if personalization_matrix is None:
            raise ValueError(
                "the transition matrix needs the personalization matrix first (d = v): build "
                "the personalization matrix before the transition matrix; there is no uniform fallback"
            )
        if issparse(personalization_matrix):
            raw = np.asarray(personalization_matrix.sum(axis=1), dtype=float).ravel()
        else:
            raw = np.asarray(personalization_matrix, dtype=float).sum(axis=1)
        if raw.shape != (int(N),):
            raise ValueError(f"personalization matrix has {raw.size} rows; expected {N}")
        total = float(raw.sum())
        if total <= 0.0:
            raise ValueError("the personalization matrix carries no restart mass; v is undefined")
        return raw / total

    """
    Build the column-stochastic transition matrix from raw supra-adjacency, and patch dangling
    nodes (no outgoing weight) with the personalization vector ``v``. Requires the
    personalization matrix: see ``dangling_patch_vector``.
    """

    def normalize(self, adj_matrix: Any, personalization_matrix: Any | None, N: int) -> Any:
        if adj_matrix is None:
            raise ValueError("Adjacency matrix has not been built yet")

        dangling_patch = self.dangling_patch_vector(
            personalization_matrix, N
        )  # applies it only to dangling columns of T to make them column-stochastic
        dangling_column_count = 0  # dangling node count set as 0 after done

        if issparse(
            adj_matrix
        ):  # CSR format gives contiguous per-row slices (indptr/indices/data) so each row (a node's outgoing edges) can be normalized independently
            A = adj_matrix.tocsr().astype(np.float64)
            data = A.data
            if np.isnan(data).any():
                raise ValueError("Adjacency matrix contains NaN entries")
            if np.isneginf(data).any():
                raise ValueError("Adjacency matrix contains negative infinity entries")
            finite_data = data[np.isfinite(data)]
            if finite_data.size and np.any(finite_data < 0.0):
                raise ValueError("Adjacency matrix contains negative entries")

            rows: list[int] = []
            cols: list[int] = []
            values: list[float] = []
            patch_targets = np.flatnonzero(dangling_patch > 0.0)
            for row in range(A.shape[0]):
                start, end = A.indptr[row], A.indptr[row + 1]
                row_cols = A.indices[start:end]
                row_data = A.data[start:end]
                inf_mask = np.isposinf(row_data)
                if np.any(inf_mask):
                    prob = 1.0 / float(
                        np.count_nonzero(inf_mask)
                    )  # ignore finite weights entirely and split all probability mass evenly across +inf entries
                    for col in row_cols[inf_mask]:
                        rows.append(row)
                        cols.append(int(col))
                        values.append(prob)
                    continue
                row_sum = float(row_data.sum())
                if (
                    row_sum <= 0.0
                ):  # dangling node case: fall back to the personalization-derived (or uniform) patch vector so its column of T still sums to 1
                    dangling_column_count += 1
                    for col in patch_targets:
                        rows.append(row)
                        cols.append(int(col))
                        values.append(float(dangling_patch[col]))
                    continue
                for col, value in zip(row_cols, row_data / row_sum, strict=True):
                    rows.append(row)
                    cols.append(int(col))
                    values.append(float(value))

            P = csr_matrix((values, (rows, cols)), shape=A.shape, dtype=np.float64)
            T = P.transpose().tocsr()
            col_sums = np.asarray(T.sum(axis=0)).ravel()
            max_colsum_error = float(np.max(np.abs(col_sums - 1.0))) if col_sums.size else 0.0
            self.diagnostics = TransitionDiagnostics(
                shape=tuple(T.shape),
                sparse=True,
                dangling_column_count=dangling_column_count,
                max_column_sum_error=max_colsum_error,
            )
            return T

        # NOTE: the dense implementation is the same, but handling edge cases are different from sparse:
        # precomputes inf_mask_all for the whole matrix to check for infinity entries, changes the whole row for a dangling node with zeroes included before continuing,
        # and uses a full slice for normal cases. The behavior and checks are identical otherwise.
        dense = np.asarray(adj_matrix, dtype=np.float64).copy()
        if np.isnan(dense).any():
            raise ValueError("Adjacency matrix contains NaN entries")
        if np.isneginf(dense).any():
            raise ValueError("Adjacency matrix contains negative infinity entries")
        finite_dense = dense[np.isfinite(dense)]
        if finite_dense.size and np.any(finite_dense < 0.0):
            raise ValueError("Adjacency matrix contains negative entries")

        P = np.zeros_like(dense, dtype=np.float64)
        inf_mask_all = np.isposinf(dense)
        finite = np.where(inf_mask_all, 0.0, dense)
        for row in range(dense.shape[0]):
            inf_cols = np.flatnonzero(inf_mask_all[row, :])
            if inf_cols.size:
                P[row, inf_cols] = 1.0 / float(inf_cols.size)
                continue
            row_data = finite[row, :]
            row_sum = float(row_data.sum())
            if row_sum <= 0.0:
                dangling_column_count += 1
                P[row, :] = dangling_patch
                continue
            P[row, :] = row_data / row_sum

        T = P.T.copy()
        col_sums = T.sum(axis=0)
        max_colsum_error = float(np.max(np.abs(col_sums - 1.0))) if col_sums.size else 0.0
        self.diagnostics = TransitionDiagnostics(
            shape=tuple(T.shape),
            sparse=False,
            dangling_column_count=dangling_column_count,
            max_column_sum_error=max_colsum_error,
        )
        return T
