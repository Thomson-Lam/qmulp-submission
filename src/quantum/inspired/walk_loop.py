"""The instrumented evolution loop behind ``SzegedyWalk.run`` and ``run_with_history``.

Exact N x N matrix-free Szegedy evolution: W^2 per recorded step by repeated
reflection and swap, never materializing the walk operator. The marginal is
recorded **before** each advance, so ``t = 0`` is the marginal of the v-seeded
initial state and the time average ``(1/tau) sum_{t<tau} q_t`` includes it.
Per-step timing and probability-mass diagnostics go through the
``quantum.inspired.szegedy`` logger so a full-graph run can be profiled without
a separate harness.
"""

from __future__ import annotations

import logging
from time import perf_counter

import numpy as np

_logger = logging.getLogger("quantum.inspired.szegedy")


def fmt_bytes(nbytes: int) -> str:
    """Render a byte count as a short human-readable string for logs."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _advance_timed(walk, Psi: np.ndarray) -> np.ndarray:
    """Apply one W² recorded step, timing each W application for DEBUG logs."""
    if _logger.isEnabledFor(logging.DEBUG):
        t1 = perf_counter()
        mid = walk._apply_W(Psi)
        t_w1 = perf_counter() - t1
        t2 = perf_counter()
        out = walk._apply_W(mid)
        t_w2 = perf_counter() - t2
        _logger.debug("  W^2 split: W1=%.3fs W2=%.3fs", t_w1, t_w2)
        return out
    return walk._advance(Psi)


def _column_stats(walk) -> str:
    """Summarize matrix-free sqrt(G) column construction, when applicable."""
    op = getattr(walk, "_operator", None)
    if op is None or not hasattr(op, "column_generation_stats"):
        return "n/a (dense)"
    return op.column_generation_stats()


def run_loop(
    walk,
    tau: int,
    *,
    validate_probabilities: bool,
    collect_history: bool,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``(omega, history)``: the time-averaged borrower score and, on request, every recorded step."""
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}")

    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
        )

    _logger.info(
        "starting walk: n=%d supranodes, tau=%d, step_power=2 (W^2 per recorded step), "
        "measure_register=%s, state_bytes~%s",
        walk.n,
        tau,
        walk.measure_register,
        fmt_bytes(walk.n * walk.n * np.complex128().itemsize),
    )

    t_start = perf_counter()
    t_build0 = perf_counter()
    Psi = walk._build_initial_state()
    build_seconds = perf_counter() - t_build0

    history = np.zeros((tau, walk.n_borrowers), dtype=float) if collect_history else None
    cumulative = np.zeros(walk.n_borrowers, dtype=float)

    marginal_total = 0.0
    advance_total = 0.0
    max_norm_error = 0.0

    for t in range(tau):
        if validate_probabilities:
            walk._assert_probability_normalized(Psi, context=f"step {t}")

        t_marg = perf_counter()
        q = walk._marginal_borrower_probs(Psi)
        marginal_total += perf_counter() - t_marg
        if collect_history:
            history[t] = q
        else:
            cumulative += q

        source_mass = float(walk._marginal_supra_node_probs(Psi, register="source").sum())
        max_norm_error = max(max_norm_error, abs(source_mass - 1.0))

        t_adv = perf_counter()
        Psi = _advance_timed(walk, Psi)
        advance_total += perf_counter() - t_adv

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "step %d/%d: borrower_mass=%.6f source_mass=%.6f advance=%.3fs",
                t + 1, tau, float(q.sum()), source_mass, perf_counter() - t_adv,
            )
        elif _logger.isEnabledFor(logging.INFO) and ((t + 1) % 10 == 0 or t + 1 == tau):
            elapsed = perf_counter() - t_start
            remaining = (elapsed / (t + 1)) * (tau - (t + 1)) if t + 1 else 0.0
            _logger.info(
                "step %d/%d: borrower_mass=%.6f source_mass=%.6f elapsed=%.2fs eta~%.1fs",
                t + 1, tau, float(q.sum()), source_mass, elapsed, remaining,
            )

    total_seconds = perf_counter() - t_start
    _logger.info(
        "walk complete: total=%.2fs (build=%.2fs marginal=%.2fs advance=%.2fs) "
        "max_norm_error=%.2e columns=%s",
        total_seconds, build_seconds, marginal_total, advance_total, max_norm_error, _column_stats(walk),
    )

    omega = history.mean(axis=0) if collect_history else cumulative / tau
    return omega, history
