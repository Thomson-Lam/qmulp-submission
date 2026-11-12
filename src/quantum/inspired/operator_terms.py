"""The terms of the Google operator: ``v`` from the influence matrix and the factored restart.

``quantum.inspired.google_matrix`` assembles ``G = alpha*T + (1-alpha)*P`` from
what is derived here. Nothing here forms a dense ``v 1^T``: the restart is the
factored ``classical.mulp.personalization.FactoredRestart``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse import csc_matrix, issparse

from classical.mulp.personalization import FactoredRestart, normalized_restart

from .validation import _matrix_data_for_fast_checks, _matrix_shape, validate_probability_vector


def derive_personalization_vector(
    R: Any,
    *,
    n: int | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """
    Derive v from a MuLP influence matrix R.

    MuLP uses R[i, j] as restart weight from node j to node i. The quantum path
    uses one global personalization vector v_i proportional to row-sums of R.
    """
    n_rows, n_cols = _matrix_shape(R)
    if n_rows != n_cols:
        raise ValueError(f"Personalization matrix R must be square; got shape={R.shape}")
    if n is not None and n_rows != n:
        raise ValueError(f"R shape={R.shape}, expected ({n}, {n})")

    data = _matrix_data_for_fast_checks(R)
    if data.size:
        if not np.isfinite(data).all():
            raise ValueError("Personalization matrix R contains non-finite entries")
        min_entry = float(data.min())
        if min_entry < -atol:
            raise ValueError(f"Personalization matrix R contains negatives; min={min_entry:.3e}")

    if issparse(R):
        v_unnorm = np.asarray(R.sum(axis=1), dtype=np.float64).ravel()
    else:
        v_unnorm = np.asarray(R, dtype=np.float64).sum(axis=1)

    total = float(v_unnorm.sum())
    if total <= atol:
        raise ValueError("Personalization matrix R is all zeros or numerically empty")
    v = v_unnorm / total
    return validate_probability_vector(v, name="v", n=n_rows, atol=atol)


def build_restart_matrix_from_influence(
    R: Any,
    v: np.ndarray,
    *,
    atol: float = 1e-10,
) -> FactoredRestart:
    """Normalize nonempty R columns; mark empty columns to take v.

    Returns the factored form (``classical.mulp.personalization.FactoredRestart``):
    the rank-one ``v 1^T`` block is applied inside the matvec and never stored.
    """
    n_rows, n_cols = _matrix_shape(R)
    if n_rows != n_cols:
        raise ValueError(f"Personalization matrix R must be square; got shape={R.shape}")
    v = validate_probability_vector(v, name="v", n=n_rows, atol=atol)
    data = _matrix_data_for_fast_checks(R)
    if data.size and (not np.isfinite(data).all() or float(data.min()) < -atol):
        raise ValueError("Personalization matrix R must contain finite nonnegative entries")
    restart = normalized_restart(R, v, atol=atol)
    restart.assert_column_stochastic(atol=max(atol, 1e-10))
    return restart


def as_factored_restart(restart: Any, v: np.ndarray) -> FactoredRestart | None:
    """Accept a FactoredRestart, a full column-stochastic P, or None."""
    if restart is None or isinstance(restart, FactoredRestart):
        return restart
    matrix = csc_matrix(restart, dtype=np.float64)
    col_sums = np.asarray(matrix.sum(axis=0)).ravel()
    if not np.allclose(col_sums, 1.0, atol=1e-8):
        raise ValueError("restart_matrix must be column-stochastic")
    return FactoredRestart(
        seed_columns=matrix, fill_mask=np.zeros(matrix.shape[0], dtype=bool), v=v
    )
