"""The convergence point ``tau*`` of the running average towards the exact limit, per network, and its tables.

On the grid ``{1, ..., 1000} U {1010, 1020, ..., 10000} U {10100, 10200, ..., tau_max}``
the running average ``omega-bar_tau`` (coefficient recurrence with the
router's ``K``) is compared with the exact limit ``omega``:

- ``L1_tau = sum_supra |omega-bar_tau - omega|`` over all supra-nodes;
- ``rho_tau`` = Spearman(``omega-bar_tau``, ``omega``) over the test individuals;
- ``ov_tau`` = expected top-k overlap of the two over the test individuals,
  beside ``ov_self``, the self-overlap ceiling of the limit.

``tau*`` is the smallest grid point from which every later grid point up to
``tau_max`` satisfies ``L1_tau <= conv_tol_l1``, ``rho_tau >= conv_min_spearman``
and ``ov_tau >= ov_self - conv_overlap_slack``, **provided that tail holds at
least ``conv_min_tail_points`` grid points** (the point itself included). A
criterion that fails at every grid point but the last is therefore not a
convergence point: a running average whose Spearman with the limit flips
between two values until the second-last grid point would otherwise be
reported as converged at the cap. If the conditions are not met continuously,
or only on a shorter tail, the result is ``"> tau_max"`` with the last
violation recorded and the length of the satisfied tail; nothing falls back
silently. ``tau_star_l1`` uses the first condition alone
and ``tau_star_rank`` the last two. The constants ``c_exact`` and ``c_bound``
come from the spectral limit (``quantum.mechanism.limit``).

The convergence tables
----------------------
``T_convergence``, ``EC_convergence_sensitivity`` and ``EC_convergence_curves``.
Per network: the reduced size ``R``, the active eigenvalue groups ``r``, the
chord gap, ``c_exact`` and its bound, ``tau*`` and its two one-sided
variants, ``L1`` and Spearman at ``tau = 100``, the agreement of the
classical running average at ``tau*`` with the stationary score, and the
reduction facts the manuscript's Table 1 prints (``N_full`` after unused
slots are dropped, the reserved slots, classes, ``m``, qubits, angles, the
density of the lumped operator). The sensitivity table re-evaluates ``tau*``
over the tolerance grid and the curves file holds every grid point of every
network for Figure 2. Every file is written under the dataset's own scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from config import ExperimentConfig
from datasets import DatasetConfig
from evaluation.provenance import write_result
from evaluation.reporting import expected_top_k_overlap, l1_sum, spearman
from evaluation.ties import canonicalize
from quantum.lumped_operator import unfold_class_scores
from quantum.mechanism.limit import LimitingScores
from quantum.mechanism.running_average import running_averages


ABOVE_MAX = "> tau_max"
# Supra-node distributions are compared at full precision; borrower scores at the tie tolerance.
SUPRA_DECIMALS = 15


def convergence_grid(tau_max: int) -> np.ndarray:
    """``{1..1000} U {1010..10000 step 10} U {10100..tau_max step 100}``, clipped to ``tau_max``."""
    tau_max = int(tau_max)
    if tau_max < 1:
        raise ValueError("tau_max must be positive")
    grid = set(range(1, min(1000, tau_max) + 1))
    grid |= set(range(1010, min(10_000, tau_max) + 1, 10))
    grid |= set(range(10_100, tau_max + 1, 100))
    grid.add(tau_max)
    return np.array(sorted(grid), dtype=int)


def _test_scores(nodes, positions: np.ndarray, test_idx: np.ndarray, decimals: int) -> np.ndarray:
    ids, scores = unfold_class_scores(nodes, positions)
    lookup = np.full(int(ids.max()) + 1, np.nan)
    lookup[ids] = scores
    return canonicalize(lookup[np.asarray(test_idx, dtype=int)], decimals)


def convergence_curves(
    scores: LimitingScores, nodes, test_idx: np.ndarray, *, tau_max: int, k: int, decimals: int
) -> dict[str, Any]:
    """``L1_tau``, ``rho_tau`` and ``ov_tau`` at every grid horizon, and the self-overlap ceiling."""
    grid = convergence_grid(tau_max)
    averages = running_averages(scores.G, scores.v, scores.K, grid)
    omega = scores.limit.omega
    omega_test = _test_scores(nodes, omega, test_idx, decimals)
    ov_self = expected_top_k_overlap(omega_test, omega_test, k)
    rows = []
    for tau in grid:
        bar = averages[int(tau)]
        bar_test = _test_scores(nodes, bar, test_idx, decimals)
        rows.append({
            "tau": int(tau),
            "l1_tau": l1_sum(bar, omega, decimals=SUPRA_DECIMALS),
            "rho_tau": spearman(bar_test, omega_test, decimals=decimals),
            "ov_tau": expected_top_k_overlap(bar_test, omega_test, k),
            "ov_self": ov_self,
        })
    return {"grid": grid, "rows": rows, "ov_self": float(ov_self), "tau_max": int(tau_max), "k": int(k),
            "averages_at": {int(t): averages[int(t)] for t in grid}}


def _suffix_point(
    rows: Sequence[dict[str, Any]], satisfied, *, min_tail_points: int
) -> tuple[int | str, dict[str, Any] | None, int]:
    """The smallest grid point from which ``satisfied(row)`` holds for every later row, on a long enough tail.

    Returns ``(point, last_violation, tail_points)``. ``tail_points`` is the number of grid points in
    the satisfied suffix (zero when the last grid point fails), reported whether or not the suffix
    was accepted; the point is ``ABOVE_MAX`` when that suffix has fewer than ``min_tail_points``
    grid points. One shared function: stage 04 and the variability study both call it through
    :func:`tau_star`.
    """
    violations = [i for i, row in enumerate(rows) if not satisfied(row)]
    start = violations[-1] + 1 if violations else 0
    last = rows[violations[-1]] if violations else None
    tail = len(rows) - start
    if tail < int(min_tail_points):
        return ABOVE_MAX, last, tail
    return int(rows[start]["tau"]), last, tail


def tau_star(
    rows: Sequence[dict[str, Any]], *, tol_l1: float, min_spearman: float, overlap_slack: float, min_tail_points: int
) -> dict[str, Any]:
    """``tau*``, ``tau_star_l1`` and ``tau_star_rank`` with the last violation and the satisfied tail of each criterion.

    ``min_tail_points`` is ``config.conv_min_tail_points``; it is required, so no caller can skip the
    tail rule by omission. A grid shorter than the rule can never certify convergence and is an
    error, not a silent "> tau_max".
    """
    if int(min_tail_points) < 1:
        raise ValueError("min_tail_points must be at least one grid point")
    if len(rows) < int(min_tail_points):
        raise ValueError(
            f"the convergence grid has {len(rows)} points, fewer than the required tail of {int(min_tail_points)}; "
            "raise conv_tau_max or lower conv_min_tail_points"
        )
    ov_self = float(rows[0]["ov_self"])

    def l1_ok(row) -> bool:
        return bool(row["l1_tau"] <= float(tol_l1))

    def rank_ok(row) -> bool:
        rho = row["rho_tau"]
        return bool(np.isfinite(rho) and rho >= float(min_spearman)
                    and row["ov_tau"] >= ov_self - float(overlap_slack))

    star, last, tail = _suffix_point(rows, lambda r: l1_ok(r) and rank_ok(r), min_tail_points=min_tail_points)
    star_l1, last_l1, tail_l1 = _suffix_point(rows, l1_ok, min_tail_points=min_tail_points)
    star_rank, last_rank, tail_rank = _suffix_point(rows, rank_ok, min_tail_points=min_tail_points)

    def describe(v):
        return None if v is None else {"tau": int(v["tau"]), "l1_tau": float(v["l1_tau"]),
                                       "rho_tau": float(v["rho_tau"]), "ov_tau": float(v["ov_tau"])}

    return {
        "tau_star": star, "tau_star_l1": star_l1, "tau_star_rank": star_rank,
        "last_violation": describe(last), "last_violation_l1": describe(last_l1),
        "last_violation_rank": describe(last_rank),
        "tail_points": int(tail), "tail_points_l1": int(tail_l1), "tail_points_rank": int(tail_rank),
        "conv_tol_l1": float(tol_l1), "conv_min_spearman": float(min_spearman),
        "conv_overlap_slack": float(overlap_slack), "conv_min_tail_points": int(min_tail_points),
        "ov_self": ov_self, "tau_max": int(rows[-1]["tau"]),
        "reached": star != ABOVE_MAX,
    }


def tau_star_sensitivity(
    rows, *, l1_grid, spearman_grid, overlap_slack: float, min_tail_points: int
) -> list[dict[str, Any]]:
    """``tau*`` over the tolerance grid; one row per ``(tol_l1, min_spearman)``."""
    out = []
    for tol in l1_grid:
        for rho in spearman_grid:
            result = tau_star(rows, tol_l1=float(tol), min_spearman=float(rho), overlap_slack=overlap_slack,
                              min_tail_points=min_tail_points)
            out.append({"conv_tol_l1": float(tol), "conv_min_spearman": float(rho),
                        "tau_star": result["tau_star"], "tau_star_l1": result["tau_star_l1"],
                        "tau_star_rank": result["tau_star_rank"], "reached": result["reached"]})
    return out


def at_horizon(rows: Sequence[dict[str, Any]], tau: int) -> dict[str, float]:
    """``L1`` and ``rho`` at one grid horizon (``tau = 100`` is always on the grid)."""
    for row in rows:
        if int(row["tau"]) == int(tau):
            return {"l1": float(row["l1_tau"]), "rho": float(row["rho_tau"]), "ov": float(row["ov_tau"])}
    raise KeyError(f"tau={tau} is not on the convergence grid")


def tau_star_label(convergence: dict[str, Any]) -> str:
    """``tau*`` as a printed value: the integer, or ``> tau_max`` when it was not reached."""
    return str(convergence["tau_star"]) if convergence.get("reached") else f"{ABOVE_MAX} ({convergence['tau_max']})"


# ---------------------------------------------------------------------------
# The convergence tables
# ---------------------------------------------------------------------------


TABLE_NAME = "T_convergence"
SENSITIVITY_NAME = "EC_convergence_sensitivity"
CURVES_NAME = "EC_convergence_curves"


def _star(value) -> str:
    return str(value)


def convergence_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One table row per convergence record (``scores_stage.compute_network``)."""
    rows = []
    for rec in records:
        rows.append({
            "dataset": rec["dataset"], "network": rec["network"], "R": rec["R"], "route": rec["route"],
            "N_full": rec.get("N_full"), "n_slots": rec.get("n_slots"), "n_classes": rec.get("n_classes"),
            "reduction_factor": rec.get("reduction_factor"), "m": rec.get("m"), "qubits": rec.get("qubits"),
            "angles": rec.get("angles"), "gbar_density": rec.get("gbar_density"),
            "lumpability_residual": rec.get("lumpability_residual"), "compatibility_residual": rec.get("compatibility_residual"),
            "r_active": rec["r_active"], "delta_chord": rec["delta_chord"], "c_exact": rec["c_exact"],
            "c_bound": rec["c_bound"], "d_inner_max": rec.get("d_inner_max"), "d_gap_min": rec.get("d_gap_min"),
            "d_collapsed": rec.get("d_collapsed"), "d_collapse_dev_max": rec.get("d_collapse_dev_max"),
            "d_gap_warning": rec.get("d_gap_warning"),
            "tau_star": _star(rec["tau_star"]), "tau_star_l1": _star(rec["tau_star_l1"]),
            "tau_star_rank": _star(rec["tau_star_rank"]), "tau_star_reached": bool(rec["reached"]),
            "tail_points": rec.get("tail_points"), "conv_min_tail_points": rec.get("conv_min_tail_points"),
            "tau_star_printed": tau_star_label(rec), "tau_max": rec["tau_max"], "tau_eval": rec.get("tau_eval"),
            "l1_at_100": rec["l1_at_100"], "rho_100_vs_limit": rec["rho_100_vs_limit"],
            "rho_tau_star_vs_limit": rec["rho_tau_star_vs_limit"],
            "rho_classical_tau_star": rec["rho_classical_tau_star"],
            "decision_at_tau_star_agrees_3dp": bool(rec["decision_at_tau_star_agrees_3dp"]),
            "conv_tol_l1": rec["conv_tol_l1"], "conv_min_spearman": rec["conv_min_spearman"],
            "conv_overlap_slack": rec["conv_overlap_slack"],
            "last_violation_tau": None if rec["last_violation"] is None else rec["last_violation"]["tau"],
        })
    return rows


CONVERGENCE_COLUMNS = {
    "dataset": "dataset name", "network": "network name",
    "R": "supra-nodes of the lumped operator (blocks)", "route": "discriminant the router selected (D_bar or H)",
    "N_full": "supra-nodes of the unlumped operator after unused category slots are dropped (individual copies plus category nodes)",
    "n_slots": "category slots the supra-adjacency reserves (every category of every layer, per layer block); unused slots carry no mass",
    "n_classes": "lumping classes (individuals with the same category state per layer and the same masked label)",
    "reduction_factor": "N_full / R",
    "m": "register width, ceil(log2 R)", "qubits": "logical qubits of the Szegedy walk, 2 m",
    "angles": "multiplexed state-preparation angles of one reflection, 2^m (2^m - 1)",
    "gbar_density": "share of non-zero entries of the lumped operator G_bar",
    "lumpability_residual": "max |C G - G_bar C|", "compatibility_residual": "max |D B - B D_bar|",
    "r_active": "eigenvalue groups of the walk with non-zero initial mass",
    "delta_chord": "smallest chord between active eigenphase groups",
    "c_exact": "exact constant of the bound ||omega_bar_tau - omega||_1 <= c / tau",
    "c_bound": "2 (r - 1) / delta_chord",
    "d_inner_max": "largest |d| among the discriminant eigenvalues kept strictly inside the unit interval (1 - |d| >= d_tol); the others are folded into the eigenvalue-one block",
    "d_gap_min": "1 - d_inner_max: the smallest distance from the edge of an eigenvalue kept inside (small values mean slow convergence)",
    "d_collapsed": "discriminant eigenvalues folded into the eigenvalue-one block (1 - |d| < d_tol)",
    "d_collapse_dev_max": "largest 1 - |d| among the folded eigenvalues: how far from exactly +-1 the fold reached",
    "d_gap_warning": "whether d_gap_min is below 1e-6 (the limit then also warns)",
    "tail_points": "grid horizons in the satisfied suffix of the combined criterion (0 when the last grid point fails)",
    "conv_min_tail_points": "minimum grid points at or above tau_star required to accept a convergence point (config.conv_min_tail_points)",
    "tau_star": "convergence point: smallest grid horizon from which every criterion holds up to tau_max; '> tau_max' when not reached",
    "tau_star_l1": "convergence point of the L1 criterion alone", "tau_star_rank": "convergence point of the rank criteria alone",
    "tau_star_reached": "whether tau_star was reached before tau_max", "tau_star_printed": "tau_star as printed in the manuscript",
    "tau_max": "largest horizon the running average was followed to",
    "tau_eval": "horizon at which qppr_tau_star and classical_tau_star were evaluated: tau_star when reached, else tau_max "
                "(so qppr_tau_star is then the tau_max running average, not a converged value)",
    "l1_at_100": "L1 distance between the running average and the limit at tau = 100",
    "rho_100_vs_limit": "Spearman between the running average at tau = 100 and the limit over test individuals",
    "rho_tau_star_vs_limit": "Spearman between the running average at tau_eval and the limit over test individuals",
    "rho_classical_tau_star": "Spearman between the classical running average at tau_eval and the stationary score",
    "decision_at_tau_star_agrees_3dp": "whether PR-AUC, ROC-AUC and precision@k at tau_eval equal the limit's to three decimals",
    "conv_tol_l1": "L1 tolerance of the convergence criterion", "conv_min_spearman": "Spearman floor of the convergence criterion",
    "conv_overlap_slack": "top-k overlap slack of the convergence criterion",
    "last_violation_tau": "last grid horizon at which a criterion failed (null when tau_star is the first grid point)",
}


def write_convergence_tables(
    config: ExperimentConfig, dataset: DatasetConfig, records, curves, sensitivity, *, inputs, source_stage: str
) -> dict[str, Path]:
    """``T_convergence``, ``EC_convergence_sensitivity`` and ``EC_convergence_curves`` under the dataset's scope."""
    main = write_result(
        dataset.name, TABLE_NAME, pl.DataFrame(convergence_rows(records), strict=False, infer_schema_length=None),
        stage=source_stage, columns=CONVERGENCE_COLUMNS,
    )
    sens = write_result(
        dataset.name, SENSITIVITY_NAME, pl.DataFrame(list(sensitivity), strict=False, infer_schema_length=None),
        stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "conv_tol_l1": "L1 tolerance of this grid point",
                 "conv_min_spearman": "Spearman floor of this grid point", "tau_star": "convergence point under this grid point",
                 "tau_star_l1": "L1-only convergence point", "tau_star_rank": "rank-only convergence point",
                 "reached": "whether the convergence point was reached before tau_max"},
    )
    curves_out = write_result(
        dataset.name, CURVES_NAME, pl.DataFrame(list(curves), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "tau": "horizon (grid point)",
                 "l1_tau": "L1 distance between the running average and the limit over supra-nodes",
                 "rho_tau": "Spearman between the running average and the limit over test individuals",
                 "ov_tau": "expected top-k overlap between the running average and the limit over test individuals",
                 "ov_self": "self-overlap ceiling of the limit"},
    )
    return {"convergence": main, "sensitivity": sens, "curves": curves_out}
