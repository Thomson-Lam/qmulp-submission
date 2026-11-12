"""The running average of the measured distribution, by the coefficient recurrence on the busy subspace.

The state stays in ``span{A y, S A z}`` (``quantum.mechanism.limit``), and one
reflection--swap ``W`` acts on the coordinates as ``(y, z) -> (-z, y + 2 K z)``
with ``K`` the discriminant the router chose. One recorded step is ``W^2``.
The source marginal at every step is :func:`quantum.mechanism.limit.closed_row_mass`,
so a horizon of ``tau_max`` costs ``2 tau_max`` products with an ``R x R``
matrix instead of ``tau_max`` evolutions of an ``R x R`` state: 200,000 steps
on the 1,651-block Freddie Mac operator are minutes, not days.

Nothing here reports a score. The running averages feed the convergence
point ``tau*`` (``quantum.mechanism.convergence``), the full-versus-reduced
check and the classical finite-horizon control; the reported quantity is the
limit.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from quantum.mechanism.limit import closed_row_mass


def _check_square(K: np.ndarray, n: int) -> np.ndarray:
    K = np.asarray(K, dtype=float)
    if K.shape != (n, n):
        raise ValueError(f"K must have shape ({n}, {n}); got {K.shape}")
    return K


def step_marginals(G: np.ndarray, v: np.ndarray, K: np.ndarray, n_steps: int, *, register: str = "source") -> np.ndarray:
    """``p_t`` for ``t = 0 .. n_steps - 1`` over compact positions, shape ``(n_steps, n)``."""
    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    K = _check_square(K, n)
    if int(n_steps) < 1:
        raise ValueError("n_steps must be positive")
    y = np.sqrt(np.asarray(v, dtype=float)).astype(complex)
    z = np.zeros(n, dtype=complex)
    out = np.zeros((int(n_steps), n), dtype=float)
    for t in range(int(n_steps)):
        out[t] = closed_row_mass(y, z, G, K, register=register)
        for _ in range(2):
            y, z = -z, y + 2.0 * (K @ z)
    return out


def running_averages(
    G: np.ndarray, v: np.ndarray, K: np.ndarray, taus: Iterable[int], *, register: str = "source"
) -> dict[int, np.ndarray]:
    """``omega-bar_tau = (1/tau) sum_{t<tau} p_t`` at every ``tau`` in ``taus``, ``t = 0`` included.

    One pass to ``max(taus)`` with a running sum; only the requested horizons
    are kept, so following the average to ``tau_max = 200,000`` holds one
    vector of accumulators, not a history.
    """
    taus = sorted({int(t) for t in taus})
    if not taus or taus[0] < 1:
        raise ValueError("taus must be positive integers")
    G = np.asarray(G, dtype=float)
    n = G.shape[0]
    K = _check_square(K, n)
    y = np.sqrt(np.asarray(v, dtype=float)).astype(complex)
    z = np.zeros(n, dtype=complex)
    total = np.zeros(n, dtype=float)
    wanted = set(taus)
    out: dict[int, np.ndarray] = {}
    for t in range(taus[-1]):
        total += closed_row_mass(y, z, G, K, register=register)
        if t + 1 in wanted:
            out[t + 1] = total / float(t + 1)
        for _ in range(2):
            y, z = -z, y + 2.0 * (K @ z)
    return out


def classical_running_averages(G: np.ndarray, v: np.ndarray, taus: Iterable[int]) -> dict[int, np.ndarray]:
    """``(1/tau) sum_{t<tau} G^t v`` at every ``tau`` in ``taus``: the classical control on the same operator."""
    taus = sorted({int(t) for t in taus})
    if not taus or taus[0] < 1:
        raise ValueError("taus must be positive integers")
    G = np.asarray(G, dtype=float)
    x = np.asarray(v, dtype=float).copy()
    total = np.zeros_like(x)
    wanted = set(taus)
    out: dict[int, np.ndarray] = {}
    for t in range(taus[-1]):
        total += x
        if t + 1 in wanted:
            out[t + 1] = total / float(t + 1)
        x = G @ x
    return out
