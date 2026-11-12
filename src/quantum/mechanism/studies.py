"""The mechanism studies: the interlayer-weight sweep (Figure 1), Table 3 (the control operators) and Table 4.

The interlayer-weight sweep: both arms on the two-layer operator at every ``eta``
--------------------------------------------------------------------------------
For every ``eta`` of the mechanism grid the two-layer lumped operator is
rebuilt, routed (the route chosen at each ``eta`` is recorded in ``route``) and both arms
scored. Recorded per ``eta``: Spearman(QPPR, classical) at that ``eta``, the
Spearman of each arm with its primary (``eta = 1``) counterpart, the expected
partially observed share of each arm's top-k, the route and both residuals.

Table 3: the limiting QPPR on every control operator against the primary scores
-------------------------------------------------------------------------------
Rows: ``own_copy_restart``, ``rank_one_restart``, ``flattened``, ``eta_0``,
``no_partial_copies`` and the best classical point of the ``(alpha, eta)``
grid by Spearman with the primary QPPR. Columns: Spearman and expected top-k
overlap against the primary QPPR, Spearman against the control's own
classical score, the route and the residuals. The whole classical grid goes
to ``EC_alpha_eta_grid`` with the smallest Spearman of a grid point's
classical score with the primary classical score.

Table 4: where the reordering lives -- the partially observed test individuals -- and the label-free control
------------------------------------------------------------------------------------------------------------
Per two-layer network and arm, on the test partition: the number of
partially observed individuals (no membership edge in at least one layer),
their default rate against the others, their expected share of the top-k,
their mean and median rank, the Spearman between the arms over everyone and
over the fully observed, the expected top-k Jaccard of each arm's ranking
with the same arm's ranking on the label-free operator and with the placebo
rankings (mean), and the placebo p-value for precision@k. The placebo columns are read from
stage 05's ``EC_placebo`` (:func:`placebo_columns`), which must exist: stage 05 runs first.

The three tables' names are ``ETA_TABLE_NAME``, ``VARIANTS_TABLE_NAME`` and ``PARTIAL_TABLE_NAME``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from config import MechanismConfig
from evaluation.context import DatasetContext, ScoredOperator, score_operator
from evaluation.provenance import write_result
from evaluation.reporting import expected_share, expected_top_k_overlap, spearman, top_k_jaccard
from quantum.mechanism.classical import alpha_eta_grid
from quantum.mechanism.variants import (
    OperatorVariant,
    eta_variant,
    flattened_variant,
    label_free_variant,
    no_partial_copies_variant,
    own_copy_restart_variant,
    rank_one_restart_variant,
)


# ---------------------------------------------------------------------------
# The interlayer-weight sweep
# ---------------------------------------------------------------------------


ETA_TABLE_NAME = "T_eta"


def eta_rows(ctx: DatasetContext, network: str, mech: MechanismConfig) -> tuple[list[dict[str, Any]], dict[float, ScoredOperator]]:
    """One row per ``eta`` and the scored variants, keyed by ``eta``."""
    spec = ctx.specs[network]
    primary = ctx.primary(network)
    ref_q, ref_c = ctx.evaluate(primary.quantum.omega_source), ctx.evaluate(primary.classical)
    partial = ctx.flag_test("partially_observed")
    rows: list[dict[str, Any]] = []
    scored: dict[float, ScoredOperator] = {}
    for eta in mech.eta_grid:
        variant = eta_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids, eta)
        result = score_operator(variant, ctx.config)
        q, c = ctx.evaluate(result.quantum.omega_source), ctx.evaluate(result.classical)
        rows.append(ctx.row(
            network, eta=float(eta), route=result.routed.route,
            lumpability_residual=result.routed.residuals["lumpability_residual"],
            compatibility_residual=result.routed.residuals["compatibility_residual"],
            n_dangling_columns=variant.n_dangling_columns,
            rho_qc=spearman(q, c, decimals=ctx.decimals),
            rho_c_vs_primary_c=spearman(c, ref_c, decimals=ctx.decimals),
            rho_q_vs_primary_q=spearman(q, ref_q, decimals=ctx.decimals),
            partial_share_topk_q=expected_share(q, partial, ctx.k),
            partial_share_topk_c=expected_share(c, partial, ctx.k),
            k=ctx.k,
        ))
        scored[float(eta)] = result
    return rows, scored


def write_eta_table(ctx: DatasetContext, rows, *, mech: MechanismConfig, source_stage: str) -> Path:
    """``T_eta`` under the dataset's scope: the data behind Figure 1."""
    k = ctx.k
    return write_result(
        ctx.name, ETA_TABLE_NAME, pl.DataFrame(list(rows), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "eta": "interlayer weight of the two-layer operator",
                 "route": "discriminant the router selected at this eta (D_bar or H)",
                 "lumpability_residual": "max |C G - G_bar C|", "compatibility_residual": "max |D B - B D_bar|",
                 "n_dangling_columns": "columns of the operator with no out-edge (patched with v)",
                 "rho_qc": "Spearman between qppr_limit and classical at this eta over test individuals",
                 "rho_c_vs_primary_c": "Spearman between classical at this eta and classical at the primary eta",
                 "rho_q_vs_primary_q": "Spearman between qppr_limit at this eta and qppr_limit at the primary eta",
                 "partial_share_topk_q": f"expected share of partially observed test individuals in the QPPR top {k}",
                 "partial_share_topk_c": f"expected share of partially observed test individuals in the classical top {k}",
                 "k": "k of the top-k quantities"},
    )


# ---------------------------------------------------------------------------
# Table 3: the control operators
# ---------------------------------------------------------------------------


VARIANTS_TABLE_NAME = "T3_variants"
GRID_NAME = "EC_alpha_eta_grid"


def _row(ctx: DatasetContext, network: str, label: str, scored: ScoredOperator, ref_q, ref_c) -> dict[str, Any]:
    q = ctx.test(scored.borrower_ids, scored.quantum.omega_source)
    c = ctx.test(scored.borrower_ids, scored.classical)
    covers_all = scored.borrower_ids.size == ctx.n_borrowers
    rho_vs_q = spearman(q, ref_q, decimals=ctx.decimals) if covers_all else float("nan")
    return ctx.row(
        network, variant=label, route=scored.routed.route,
        lumpability_residual=scored.routed.residuals["lumpability_residual"],
        compatibility_residual=scored.routed.residuals["compatibility_residual"],
        rho_vs_primary_q=rho_vs_q,
        ov_vs_primary_q=expected_top_k_overlap(q, ref_q, ctx.k) if covers_all else float("nan"),
        rho_vs_own_classical=spearman(q, c, decimals=ctx.decimals),
        rho_classical_vs_primary_c=spearman(c, ref_c, decimals=ctx.decimals) if covers_all else float("nan"),
        n_nodes=scored.variant.n_nodes, n_edges=scored.variant.n_edges, n_dangling_columns=scored.variant.n_dangling_columns,
        alpha=float(ctx.config.alpha), eta=None, k=ctx.k,
    )


def variant_rows(ctx: DatasetContext, network: str, mech: MechanismConfig, scored_eta: dict[float, ScoredOperator] | None = None):
    """The Table 3 rows of ``network`` and the full classical grid rows."""
    spec = ctx.specs[network]
    primary = ctx.primary(network)
    ref_q, ref_c = ctx.evaluate(primary.quantum.omega_source), ctx.evaluate(primary.classical)
    builders: list[tuple[str, OperatorVariant]] = [
        ("primary", primary.variant),
        ("own_copy_restart", own_copy_restart_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids)),
        ("rank_one_restart", rank_one_restart_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids)),
        ("flattened", flattened_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids)),
        ("eta_0", eta_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids, 0.0)),
        ("no_partial_copies", no_partial_copies_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids)),
    ]
    rows: list[dict[str, Any]] = []
    for label, variant in builders:
        if label == "primary":
            scored = primary
        elif label == "eta_0" and scored_eta is not None and 0.0 in scored_eta:
            scored = scored_eta[0.0]
        else:
            scored = score_operator(variant, ctx.config)
        row = _row(ctx, network, label, scored, ref_q, ref_c)
        # The interlayer weight each control ran at: the frozen stickiness for every control on the two-layer
        # graph, zero for the eta = 0 control, and none for the flattened operator (one layer, no interlayer edge).
        if label == "eta_0":
            row["eta"] = 0.0
        elif label != "flattened":
            row["eta"] = float(ctx.config.stickiness)
        rows.append(row)
    # alpha_eta_grid names the network itself; only the dataset is added here.
    grid = [{"dataset": ctx.name, **r} for r in alpha_eta_grid(
        ctx.partitions, spec, ctx.config, mech, ctx.hidden_ids, reference_q=primary.quantum.omega_source,
        reference_c=primary.classical, evaluation_idx=ctx.test_idx)]
    best = max(grid, key=lambda r: r["spearman_vs_primary_q"])
    rows.append(ctx.row(
        network, variant="best_classical_alpha_eta", route="classical", lumpability_residual=None,
        compatibility_residual=None, rho_vs_primary_q=best["spearman_vs_primary_q"],
        ov_vs_primary_q=best[f"top_{ctx.k}_overlap_vs_primary_q"], rho_vs_own_classical=None,
        rho_classical_vs_primary_c=best["spearman_vs_primary_c"], n_nodes=None, n_edges=None,
        n_dangling_columns=best["n_dangling_columns"], alpha=best["alpha"], eta=best["eta"], k=ctx.k,
    ))
    summary = {"best_alpha": best["alpha"], "best_eta": best["eta"], "rho_best_classical_grid": best["spearman_vs_primary_q"],
               "rho_classical_grid_min": float(min(r["spearman_vs_primary_c"] for r in grid))}
    return rows, grid, summary


def write_variant_tables(ctx: DatasetContext, rows, grid, *, mech: MechanismConfig, source_stage: str) -> dict[str, Path]:
    """``T3_variants`` and ``EC_alpha_eta_grid`` under the dataset's scope."""
    k = ctx.k
    main = write_result(
        ctx.name, VARIANTS_TABLE_NAME, pl.DataFrame(list(rows), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name",
                 "variant": "control operator (primary, own_copy_restart, rank_one_restart, flattened, eta_0, no_partial_copies, "
                            "best_classical_alpha_eta)",
                 "route": "discriminant the router selected (D_bar or H); 'classical' for the grid row",
                 "lumpability_residual": "max |C G - G_bar C|", "compatibility_residual": "max |D B - B D_bar|",
                 "rho_vs_primary_q": "Spearman of the variant's QPPR limit (classical score for the grid row) with the primary QPPR limit",
                 "ov_vs_primary_q": f"expected top-{k} overlap with the primary QPPR limit",
                 "rho_vs_own_classical": "Spearman of the variant's QPPR limit with the variant's own classical score",
                 "rho_classical_vs_primary_c": "Spearman of the variant's classical score with the primary classical score",
                 "n_nodes": "supra-nodes of the variant's lumped operator", "n_edges": "edges of the variant's lumped graph",
                 "n_dangling_columns": "columns without an out-edge", "alpha": "damping of the variant", "eta": "interlayer weight of the variant",
                 "k": "k of the top-k quantities"},
    )
    grid_table = write_result(
        ctx.name, GRID_NAME, pl.DataFrame(list(grid), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "alpha": "damping", "eta": "interlayer weight",
                 "n_dangling_columns": "columns without an out-edge",
                 "spearman_vs_primary_q": "Spearman of the classical score at (alpha, eta) with the primary QPPR limit",
                 f"top_{k}_overlap_vs_primary_q": f"expected top-{k} overlap with the primary QPPR limit",
                 "spearman_vs_primary_c": "Spearman of the classical score at (alpha, eta) with the primary classical score"},
    )
    return {"variants": main, "grid": grid_table}


# ---------------------------------------------------------------------------
# Table 4: the partially observed and the label-free control
# ---------------------------------------------------------------------------


PARTIAL_TABLE_NAME = "T4_partial"
ARMS = ("qppr", "classical")


def ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 for the largest score; tied levels share the mean rank."""
    return rankdata(-np.asarray(scores, dtype=float), method="average")


def placebo_columns(placebo_summary: list[dict[str, Any]], network: str, arm: str, k: int) -> dict[str, float]:
    """The two placebo columns of Table 4 for ``arm``, from stage 05's ``EC_placebo`` rows.

    Raises unless the summary holds exactly one row for the network and arm and both values are finite: a
    missing placebo is an error, never an empty column.
    """
    rows = [r for r in placebo_summary if r.get("network") == network and r.get("arm") == arm]
    if len(rows) != 1:
        raise ValueError(f"EC_placebo holds {len(rows)} rows for network {network!r} and arm {arm!r}; Table 4 needs exactly one")
    out: dict[str, float] = {}
    for column, source in (("jacc_placebo", f"mean_top_{k}_jaccard_with_true"), ("pval_p100", f"pval_precision_at_{k}")):
        value = rows[0].get(source)
        if value is None or not np.isfinite(float(value)):
            raise ValueError(f"EC_placebo column {source!r} is empty or not finite for network {network!r}, arm {arm!r}")
        out[column] = float(value)
    return out


def partial_rows(ctx: DatasetContext, network: str, placebo_summary: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The Table 4 rows of ``network`` (one per arm) and the label-free record."""
    k, y = ctx.k, ctx.y_test
    spec = ctx.specs[network]
    primary = ctx.primary(network)
    free = score_operator(label_free_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids), ctx.config)
    partial = ctx.flag_test("partially_observed")
    true = {"qppr": ctx.evaluate(primary.quantum.omega_source), "classical": ctx.evaluate(primary.classical)}
    label_free = {"qppr": ctx.evaluate(free.quantum.omega_source), "classical": ctx.evaluate(free.classical)}
    rho_all = spearman(true["qppr"], true["classical"], decimals=ctx.decimals)
    rho_excl = spearman(true["qppr"][~partial], true["classical"][~partial], decimals=ctx.decimals) if (~partial).sum() > 1 else float("nan")
    rows = []
    for arm in ARMS:
        r = ranks(true[arm])
        placebo = placebo_columns(placebo_summary, network, arm, k)
        rows.append(ctx.row(
            network, arm=arm, n_test=int(y.size), n_partial_test=int(partial.sum()),
            partial_defrate=float(y[partial].mean()) if partial.any() else float("nan"),
            other_defrate=float(y[~partial].mean()) if (~partial).any() else float("nan"),
            partial_share_topk=expected_share(true[arm], partial, k),
            partial_meanrank=float(r[partial].mean()) if partial.any() else float("nan"),
            partial_medianrank=float(np.median(r[partial])) if partial.any() else float("nan"),
            other_meanrank=float(r[~partial].mean()) if (~partial).any() else float("nan"),
            rho_qc=rho_all, rho_qc_excl_partial=rho_excl,
            jacc_labelfree=top_k_jaccard(true[arm], label_free[arm], k),
            jacc_placebo=placebo["jacc_placebo"],
            pval_p100=placebo["pval_p100"],
            route_labelfree=free.routed.route, k=k,
        ))
    record = {"label_free_route": free.routed.route, **{f"label_free_{key}": val for key, val in free.routed.residuals.items()
                                                       if isinstance(val, (int, float))}}
    return rows, record


def write_partial_table(ctx: DatasetContext, rows, *, source_stage: str) -> Path:
    """``T4_partial`` under the dataset's scope."""
    k = ctx.k
    return write_result(
        ctx.name, PARTIAL_TABLE_NAME, pl.DataFrame(list(rows), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "arm": "qppr or classical",
                 "n_test": "test individuals", "n_partial_test": "partially observed test individuals (no membership edge in at least one layer)",
                 "partial_defrate": "default rate of the partially observed test individuals",
                 "other_defrate": "default rate of the fully observed test individuals",
                 "partial_share_topk": f"expected share of partially observed individuals in the arm's top {k}",
                 "partial_meanrank": "mean rank (1 = largest score, average ranks on ties) of the partially observed",
                 "partial_medianrank": "median rank of the partially observed", "other_meanrank": "mean rank of the fully observed",
                 "rho_qc": "Spearman between the arms over all test individuals",
                 "rho_qc_excl_partial": "Spearman between the arms over the fully observed test individuals",
                 "jacc_labelfree": f"expected top-{k} Jaccard of the arm with the same arm on the label-free operator",
                 "jacc_placebo": f"mean expected top-{k} Jaccard of the arm with its placebo rankings",
                 "pval_p100": f"placebo p-value of the arm's precision at {k}",
                 "route_labelfree": "discriminant the router selected on the label-free operator", "k": "k of the top-k quantities"},
    )
