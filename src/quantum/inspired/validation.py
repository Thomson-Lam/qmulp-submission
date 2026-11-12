"""Matrix/vector validation and dense-allocation guards shared across quantum/inspired.

These are generic structural checks (square, finite, nonnegative, column-stochastic,
sums-to-one) plus a size guard that fails fast before a dense allocation would blow
memory -- not specific to any one walk representation, so they live in their own
module rather than inside the transition/G-matrix builders that consume them.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.sparse import issparse


def _default_max_dense_bytes() -> int | None:
    """Return the default dense-allocation cap, configurable via env var."""
    raw = os.environ.get("QWALK_MAX_DENSE_GIB", "16")
    if raw.strip().lower() in {"", "none", "unlimited", "0"}:
        return None
    try:
        return int(float(raw) * 1024**3)
    except ValueError:
        return 16 * 1024**3


DEFAULT_MAX_DENSE_BYTES = _default_max_dense_bytes()


def estimate_dense_bytes(shape: tuple[int, ...], dtype: np.dtype | type = np.float64) -> int:
    """Estimate bytes required by a dense ndarray with the supplied shape/dtype."""
    n_items = int(np.prod(shape, dtype=np.int64))
    return n_items * np.dtype(dtype).itemsize


def format_bytes(n_bytes: int | None) -> str:
    if n_bytes is None:
        return "unlimited"
    value = float(n_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def check_dense_allocation(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype | type = np.float64,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> int:
    """Fail early if a planned dense allocation exceeds the configured guard."""
    needed = estimate_dense_bytes(shape, dtype)
    if max_dense_bytes is not None and needed > max_dense_bytes:
        raise MemoryError(
            f"Dense allocation for {name} with shape={shape}, dtype={np.dtype(dtype)} "
            f"requires {format_bytes(needed)}, exceeding limit "
            f"{format_bytes(max_dense_bytes)}. Reduce graph size or raise "
            "--max-dense-gib / QWALK_MAX_DENSE_GIB."
        )
    return needed


# The exact Szegedy walk holds an N x N complex128 state matrix. Each recorded
# step applies W twice, and each application allocates a fresh state plus a
# transposed view, so live memory peaks at roughly three copies. A job that
# requests only one copy's worth is killed mid-walk.
STATE_COPIES_LIVE = 3


def state_bytes(n: int, copies: int = 1) -> int:
    """Bytes for ``copies`` N x N complex128 state matrices."""
    return int(n) * int(n) * np.dtype(np.complex128).itemsize * int(copies)


def state_allocation_report(n: int, available_bytes: int | None = None) -> dict[str, Any]:
    """What an exact walk on ``n`` supra-nodes needs, and whether it fits."""
    single = state_bytes(n, 1)
    working = state_bytes(n, STATE_COPIES_LIVE)
    return {
        "n": int(n),
        "state_bytes": single,
        "working_bytes": working,
        "state_human": format_bytes(single),
        "working_human": format_bytes(working),
        "available_bytes": available_bytes,
        "fits": None if available_bytes is None else working <= int(available_bytes),
    }


def check_state_allocation(n: int, available_bytes: int | None = None) -> dict[str, Any]:
    """Raise before allocating if the exact walk state cannot fit in memory.

    Called at walk construction rather than at the first allocation, so a batch
    job fails in the first second with the number it needed instead of being
    killed by the scheduler an hour in with no diagnosis. ``available_bytes``
    defaults to the Slurm allocation when the job declares one, then to the
    dense-allocation guard.
    """
    if available_bytes is None:
        available_bytes = _available_memory_bytes()
    report = state_allocation_report(n, available_bytes)
    if report["fits"] is False:
        raise MemoryError(
            f"Exact Szegedy walk on N={n} supra-nodes needs about "
            f"{report['working_human']} of live state "
            f"({report['state_human']} per copy x {STATE_COPIES_LIVE}), but only "
            f"{format_bytes(available_bytes)} is available. Raise the job's --mem "
            "using the matching row of resource_plan.csv, or reduce the network."
        )
    return report


def _available_memory_bytes() -> int | None:
    """Memory this process may use: the Slurm allocation, else the dense guard."""
    for key in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        raw = os.environ.get(key)
        if not raw:
            continue
        try:
            megabytes = float(raw)
        except ValueError:
            continue
        if key == "SLURM_MEM_PER_CPU":
            megabytes *= float(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        return int(megabytes * 1024**2)
    return DEFAULT_MAX_DENSE_BYTES


def _matrix_shape(matrix: Any) -> tuple[int, int]:
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError("matrix must be 2-dimensional")
    return int(matrix.shape[0]), int(matrix.shape[1])


def _matrix_data_for_fast_checks(matrix: Any) -> np.ndarray:
    if issparse(matrix):
        # LIL/DOK store object containers in .data; COO data is always numeric.
        return np.asarray(matrix.tocoo().data)
    return np.asarray(matrix)


def validate_raw_adjacency(
    A_raw: Any,
    *,
    expected_shape: tuple[int, int] | None = None,
    atol: float = 1e-12,
    allow_posinf: bool = True,
) -> None:
    """Validate raw MuLP supra-adjacency structural constraints.

    Positive infinity is accepted by default to represent the exact MuLP
    stickiness S→∞ limit; transition builders must collapse it to finite
    probabilities before stochastic validation.
    """
    n_rows, n_cols = _matrix_shape(A_raw)
    if n_rows != n_cols:
        raise ValueError(f"A_raw must be square; got shape={A_raw.shape}")
    if expected_shape is not None and tuple(A_raw.shape) != tuple(expected_shape):
        raise ValueError(f"A_raw shape={A_raw.shape}, expected {expected_shape}")

    data = _matrix_data_for_fast_checks(A_raw)
    if data.size:
        if np.isnan(data).any():
            raise ValueError("A_raw contains NaN entries")
        if np.isneginf(data).any():
            raise ValueError("A_raw contains negative infinity entries")
        if np.isposinf(data).any() and not allow_posinf:
            raise ValueError("A_raw contains positive infinity entries")
        finite_data = data[np.isfinite(data)]
        if finite_data.size:
            min_entry = float(finite_data.min())
            if min_entry < -atol:
                raise ValueError(f"A_raw contains negative entries; min={min_entry:.3e}")


def validate_probability_vector(
    v: np.ndarray,
    *,
    name: str = "v",
    n: int | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """Validate and return a float64 probability vector."""
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional; got shape={arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length={arr.shape[0]}, expected {n}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite entries")
    min_entry = float(arr.min()) if arr.size else 0.0
    if min_entry < -atol:
        raise ValueError(f"{name} contains negative entries; min={min_entry:.3e}")
    if min_entry < 0.0:
        arr = arr.copy()
        arr[arr < 0.0] = 0.0
    total = float(arr.sum())
    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(f"{name} must sum to 1; sum={total:.12g}, atol={atol}")
    return arr


def validate_column_stochastic_matrix(
    matrix: np.ndarray,
    *,
    name: str,
    atol: float = 1e-8,
) -> np.ndarray:
    """Validate a finite, nonnegative, square, column-stochastic matrix."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be square; got shape={M.shape}")
    if not np.isfinite(M).all():
        raise ValueError(f"{name} contains non-finite entries; shape={M.shape}")
    min_entry = float(M.min()) if M.size else 0.0
    max_entry = float(M.max()) if M.size else 0.0
    if min_entry < -1e-15:
        raise ValueError(
            f"{name} contains negative entries; shape={M.shape}, "
            f"min={min_entry:.3e}, max={max_entry:.3e}"
        )
    col_sums = M.sum(axis=0)
    max_colsum_error = float(np.max(np.abs(col_sums - 1.0))) if col_sums.size else 0.0
    if not np.allclose(col_sums, 1.0, atol=atol):
        raise ValueError(
            f"{name} is not column-stochastic; shape={M.shape}, "
            f"min={min_entry:.3e}, max={max_entry:.3e}, "
            f"max_colsum_error={max_colsum_error:.3e}, atol={atol}"
        )
    return M


def _to_dense_float64(
    matrix: Any,
    *,
    name: str,
    max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> np.ndarray:
    shape = _matrix_shape(matrix)
    check_dense_allocation(name, shape, np.float64, max_dense_bytes)
    if issparse(matrix):
        return matrix.toarray().astype(np.float64, copy=False)
    return np.asarray(matrix, dtype=np.float64).copy()
