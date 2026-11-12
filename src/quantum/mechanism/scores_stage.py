"""The scoring stage (notebook 04): the limit, the convergence point and the two arms' scores per network.

For every network of a dataset: route the primary lumped operator, compute
the limiting QPPR on both registers and the classical stationary score on the
same operator, follow the running average to ``tau_max`` and locate ``tau*``,
evaluate the running average at ``tau*`` (a check) and the classical running
average at ``tau*`` (a control), check the reduction against the unlumped
operator, and write one ``scores_<network>.parquet`` every later stage reads.
Nothing here is a fixed horizon: ``tau*`` is computed and reported.

The reduction table
-------------------
Residuals, route and the full-versus-reduced step check per operator. One row
per (dataset, network, variant): the sizes of the unlumped and the lumped
operator, the lumpability residual ``max |C G - G-bar C|``, the compatibility
residual ``max |D B - B D-bar|``, the route the router took and, where the
unlumped walk was run, ``max |individual score reduced - full|`` over the
recorded steps ``t = 0 .. reduction_check_steps - 1`` on the full operator.
The check is the coefficient recurrence on both sides
(``quantum.mechanism.running_average``), unfolded to individuals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from evaluation.context import DatasetContext, ScoredOperator
from evaluation.provenance import write_result
from evaluation.reporting import ranking_point_metrics, spearman
from evaluation.ties import distinct_levels
from fingerprint import TEST_OPERATOR_TAG
from paths import quantum_history_path
from quantum.lumped_operator import unfold_class_scores
from quantum.lumping import lumped_facts
from quantum.lumping_router import RoutedOperator, discriminant
from quantum.mechanism.convergence import at_horizon, convergence_curves, tau_star, tau_star_sensitivity
from quantum.mechanism.limit import INNER_GAP_WARN
from quantum.mechanism.running_average import classical_running_averages, step_marginals
from quantum.mechanism.variants import OperatorVariant


SCORE_COLUMNS = ("classical", "qppr_limit", "qppr_limit_destination", "qppr_tau_star", "classical_tau_star")
# What each score column of ``scores_<network>.parquet`` is; recorded in the stage manifest ("score_columns"),
# because the parquet is an intermediate, not a result table.
SCORE_COLUMN_DEFINITIONS = {
    "classical": "classical stationary score (the limit of the classical diffusion from v on the identical operator), summed over the individual's layer copies",
    "qppr_limit": "QPPR: the exact Cesaro limit of the source-register time average (Prop. 1), summed over the individual's layer copies",
    "qppr_limit_destination": "the same limit read off the destination register (a register check)",
    "qppr_tau_star": "the QPPR running average at tau_eval, summed over layer copies; tau_eval = tau_star when the convergence point "
                     "was reached and tau_max when it was not, so for a network reported as '> tau_max' this column holds the "
                     "running average at tau_max, not a converged value (T_convergence.tau_eval records which horizon it is)",
    "classical_tau_star": "the classical running average (finite-horizon diffusion from v) at the same tau_eval, a control",
}
FLAG_COLUMNS = ("blank_field", "blank_field_count", "structural_none", "unseen_level", "partially_observed")
HORIZON_100 = 100


@dataclass
class NetworkResult:
    """Everything the scoring stage computed on one network."""

    network: str
    scored: ScoredOperator
    scores: dict[str, np.ndarray]
    convergence: dict[str, Any]
    curves: list[dict[str, Any]]
    sensitivity: list[dict[str, Any]]
    reduction: dict[str, Any]
    facts: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


def _unfold(nodes, positions: np.ndarray, n: int) -> np.ndarray:
    ids, scores = unfold_class_scores(nodes, positions)
    if not np.array_equal(ids, np.arange(n)):
        raise AssertionError("the lumped operator does not cover every borrower exactly once")
    return scores


def ranker_history_error(ctx: DatasetContext, network: str, scored: ScoredOperator, n_steps: int) -> float | None:
    """``max |reduced - full|`` against the stage-04 checkpointed walk of the production ranker, if it exists."""
    path = quantum_history_path(network, TEST_OPERATOR_TAG, stage=ctx.dataset.stage("quantum"))
    if not path.exists():
        return None
    history = np.load(ctx._record(path))
    steps = min(int(n_steps), int(history.shape[0]))
    from quantum.mechanism.running_average import step_marginals

    reduced = step_marginals(scored.routed.G, scored.routed.v, scored.routed.K, steps)
    errors = [float(np.max(np.abs(_unfold(scored.nodes, reduced[t], ctx.n_borrowers) - history[t]))) for t in range(steps)]
    return float(max(errors))


def compute_network(ctx: DatasetContext, network: str, *, check_full: bool) -> NetworkResult:
    """Route, score, converge and check one network."""
    config = ctx.config
    scored = ctx.primary(network, keep_full=check_full)
    nodes, routed, limit = scored.nodes, scored.routed, scored.quantum
    n = ctx.n_borrowers

    curves = convergence_curves(limit, nodes, ctx.test_idx, tau_max=ctx.tau_max, k=ctx.k, decimals=ctx.decimals)
    star = tau_star(curves["rows"], tol_l1=config.conv_tol_l1, min_spearman=config.conv_min_spearman,
                    overlap_slack=config.conv_overlap_slack, min_tail_points=config.conv_min_tail_points)
    sensitivity = tau_star_sensitivity(curves["rows"], l1_grid=config.conv_sensitivity_l1,
                                       spearman_grid=config.conv_sensitivity_spearman, overlap_slack=config.conv_overlap_slack,
                                       min_tail_points=config.conv_min_tail_points)
    tau_eval = int(star["tau_star"]) if star["reached"] else int(ctx.tau_max)
    qppr_tau = _unfold(nodes, curves["averages_at"][tau_eval], n)
    classical_tau = _unfold(nodes, classical_running_averages(routed.G, routed.v, (tau_eval,))[tau_eval], n)
    scores = {
        "classical": scored.classical,
        "qppr_limit": limit.omega_source,
        "qppr_limit_destination": limit.omega_target,
        "qppr_tau_star": qppr_tau,
        "classical_tau_star": classical_tau,
    }
    q_lim, q_tau = ctx.evaluate(limit.omega_source), ctx.evaluate(qppr_tau)
    c_lim, c_tau = ctx.evaluate(scored.classical), ctx.evaluate(classical_tau)
    point_lim = ranking_point_metrics(q_lim, ctx.y_test, (ctx.k,), decimals=ctx.decimals)
    point_tau = ranking_point_metrics(q_tau, ctx.y_test, (ctx.k,), decimals=ctx.decimals)
    decision_keys = ("pr_auc", "roc_auc", f"precision_at_{ctx.k}")
    decision_agree = all(round(point_lim[key], 3) == round(point_tau[key], 3) for key in decision_keys)
    at100 = at_horizon(curves["rows"], HORIZON_100) if ctx.tau_max >= HORIZON_100 else {"l1": float("nan"), "rho": float("nan")}
    facts = lumped_facts(nodes, ctx.specs[network])
    l = limit.limit
    spec = ctx.specs[network]
    n_full = int(routed.residuals["n_full"])
    m = int(facts["register_width_m"])
    convergence = {
        "dataset": ctx.name, "network": network, "route": routed.route, **routed.record(),
        "R": int(routed.n),
        # Reduction facts, defined once here and read by Table 1, Table 5 and the
        # number map alike: N_full counts individual copies plus category nodes
        # after unused slots are dropped; n_slots is the reserved slot count.
        "N_full": n_full, "n_slots": int(spec.n_layers * spec.total_states), "n_classes": int(facts["n_classes"]),
        "reduction_factor": float(n_full) / float(routed.n), "m": m, "qubits": 2 * m, "angles": int((2**m) * (2**m - 1)),
        "gbar_density": float(np.count_nonzero(routed.G) / float(routed.G.size)), "n_groups_all": l.n_groups_all, "min_angle_gap_all": l.min_angle_gap_all,
        "r_active": l.r_active, "delta_chord": l.delta_chord_active, "delta_angle": l.delta_angle_active,
        "c_exact": l.c_exact, "c_bound": l.c_bound, "active_tol": l.active_tol,
        # The |d| = 1 collapse as it treated this operator (limit.collapse_diagnostics); d_gap_warning flags a
        # discriminant eigenvalue kept inside the unit interval yet closer to it than limit.INNER_GAP_WARN.
        "d_inner_max": l.d_inner_max, "d_gap_min": l.d_gap_min, "d_collapsed": l.d_collapsed,
        "d_collapse_dev_max": l.d_collapse_dev_max, "d_gap_warning": bool(l.d_gap_min < INNER_GAP_WARN),
        **star, "tau_eval": tau_eval, "tau_max": int(ctx.tau_max),
        "l1_at_100": at100["l1"], "rho_100_vs_limit": at100["rho"],
        "rho_tau_star_vs_limit": spearman(q_tau, q_lim, decimals=ctx.decimals),
        "rho_classical_tau_star": spearman(c_tau, c_lim, decimals=ctx.decimals),
        "decision_at_tau_star_agrees_3dp": bool(decision_agree),
        "decision_limit": {k: point_lim[k] for k in decision_keys},
        "decision_tau_star": {k: point_tau[k] for k in decision_keys},
        "levels_q": distinct_levels(q_lim, ctx.decimals), "levels_c": distinct_levels(c_lim, ctx.decimals),
    }
    reduction = reduction_row(ctx.name, network, scored.variant, routed,
                              check_steps=int(config.reduction_check_steps) if check_full else None)
    reduction["full_vs_reduced_error_ranker"] = ranker_history_error(ctx, network, scored, config.reduction_check_steps)
    if check_full:
        ctx.drop_full(network)
    return NetworkResult(network=network, scored=scored, scores=scores, convergence=convergence,
                         curves=[{"dataset": ctx.name, "network": network, **row} for row in curves["rows"]],
                         sensitivity=[{"dataset": ctx.name, "network": network, **row} for row in sensitivity],
                         reduction=reduction, facts=facts)


def scores_frame(ctx: DatasetContext, result: NetworkResult) -> pl.DataFrame:
    """One row per borrower: partition, outcome, every score column and every observation flag."""
    n = ctx.n_borrowers
    columns: dict[str, Any] = {
        "borrower_id": np.arange(n, dtype=int), "partition": ctx.partitions.partition, "BAD": ctx.partitions.y_true,
        **{name: np.asarray(result.scores[name], dtype=float) for name in SCORE_COLUMNS},
        **{name: np.asarray(ctx.flags[name]) for name in FLAG_COLUMNS},
    }
    return pl.DataFrame(columns)


def write_network_artifacts(ctx: DatasetContext, result: NetworkResult) -> list[Path]:
    """``scores_<network>.parquet``, ``convergence_<network>.json`` and ``convergence_curves_<network>.csv``."""
    out = ctx.output
    scores_path = out / f"scores_{result.network}.parquet"
    scores_frame(ctx, result).write_parquet(scores_path)
    convergence_path = out / f"convergence_{result.network}.json"
    convergence_path.write_text(json.dumps(
        {**result.convergence, "sensitivity": result.sensitivity, "reduction": result.reduction, "lumped": result.facts},
        indent=2, sort_keys=True, default=str))
    curves_path = out / f"convergence_curves_{result.network}.csv"
    pl.DataFrame(result.curves).write_csv(curves_path)
    return [scores_path, convergence_path, curves_path]


# ---------------------------------------------------------------------------
# The reduction table
# ---------------------------------------------------------------------------


TABLE_NAME = "EC_reduction_residuals"
NOT_RUN = "full walk not run on this operator"


def full_vs_reduced_error(routed: RoutedOperator, variant: OperatorVariant, n_steps: int) -> dict[str, float]:
    """``max_t max_i |reduced(i, t) - full(i, t)|`` over individuals for both routes, plus the route's own.

    ``routed`` must carry the unlumped operator (``keep_full=True``). The
    reduced walk is run with ``K = D-bar`` and with ``K = H`` so the table
    shows what each candidate would have cost, beside the route taken.
    """
    if routed.G_full is None or routed.v_full is None:
        raise ValueError("the routed operator carries no unlumped side; route it with keep_full=True")
    full = step_marginals(routed.G_full, routed.v_full, discriminant(routed.G_full), n_steps)
    ids_f = None
    full_ind = []
    for t in range(int(n_steps)):
        ids_f, scores = unfold_class_scores(variant.nodes_full, full[t])
        full_ind.append(scores)
    full_ind = np.stack(full_ind)
    out: dict[str, float] = {}
    for name, K in (("D_bar", routed.D_bar), ("H", routed.H)):
        reduced = step_marginals(routed.G, routed.v, K, n_steps)
        errors = []
        for t in range(int(n_steps)):
            ids_r, scores = unfold_class_scores(variant.nodes, reduced[t])
            if not np.array_equal(ids_r, ids_f):
                raise AssertionError("the lumped and unlumped graphs cover different individuals")
            errors.append(float(np.max(np.abs(scores - full_ind[t]))))
        out[f"full_vs_reduced_error_{name}"] = float(max(errors))
    out["full_vs_reduced_error"] = out[f"full_vs_reduced_error_{routed.route}"]
    out["full_vs_reduced_steps"] = int(n_steps)
    return out


def reduction_row(
    dataset: str, network: str, variant: OperatorVariant, routed: RoutedOperator, *, check_steps: int | None
) -> dict[str, Any]:
    """One table row; ``check_steps=None`` records the check as not run."""
    row: dict[str, Any] = {
        "dataset": dataset, "network": network, "variant": variant.label,
        "n_full": int(routed.residuals["n_full"]), "n_reduced": int(routed.residuals["n_reduced"]),
        "lumpability_residual": float(routed.residuals["lumpability_residual"]),
        "compatibility_residual": float(routed.residuals["compatibility_residual"]),
        "h_minus_dbar_max": float(routed.residuals["h_minus_dbar_max"]),
        "route": routed.route, "personalization_mode": variant.personalization_mode,
        "full_vs_reduced_error": float("nan"), "full_vs_reduced_error_D_bar": float("nan"),
        "full_vs_reduced_error_H": float("nan"), "full_vs_reduced_steps": None, "note": NOT_RUN,
    }
    if check_steps is not None:
        row.update(full_vs_reduced_error(routed, variant, int(check_steps)))
        row["note"] = ""
    return row


def write_reduction_table(rows, *, dataset: str, source_stage: str) -> Path:
    """``EC_reduction_residuals`` under the dataset's scope (each dataset writes its own file)."""
    frame = pl.DataFrame(list(rows), strict=False, infer_schema_length=None)
    return write_result(
        dataset, TABLE_NAME, frame, stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "variant": "operator variant",
                 "n_full": "supra-nodes of the unlumped operator (unused slots dropped)", "n_reduced": "supra-nodes of the lumped operator",
                 "lumpability_residual": "max |C G - G_bar C|", "compatibility_residual": "max |D B - B D_bar|",
                 "h_minus_dbar_max": "max |H - D_bar|", "route": "discriminant the router selected",
                 "personalization_mode": "restart form (combined or rank_one)",
                 "full_vs_reduced_error": "max over recorded steps and individuals of |reduced - full| step marginal under the selected route",
                 "full_vs_reduced_error_D_bar": "the same under K = D_bar", "full_vs_reduced_error_H": "the same under K = H",
                 "full_vs_reduced_steps": "recorded steps of the check", "note": f"'{NOT_RUN}' when the unlumped walk was not run",
                 "full_vs_reduced_error_ranker": "max |reduced - full| against the checkpointed production walk, when it exists"},
    )
