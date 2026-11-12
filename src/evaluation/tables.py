"""The evaluation tables of Table 2 and their e-companion companions, on the test partition.

Agreement: ``T2_agreement``, plus ``EC_registers`` and ``EC_classical_horizon``
--------------------------------------------------------------------------------
Per dataset and network, on the test partition: Spearman and Kendall tau-b
between the QPPR limit and the classical stationary score, the expected
top-k overlap with both self-overlap ceilings, the distinct levels of each
score, the Spearman of the running average at ``tau*`` with the limit, and
``tau*``. The decision and variability columns of Table 2 are produced by
their own stages and merged by the manuscript-table assembler.

Decision metrics: ``T2_decision`` (into Table 2) and ``EC_decision_full``
-------------------------------------------------------------------------
On ``qppr_limit`` and ``classical``: PR-AUC, ROC-AUC, precision@k and
lift@k with marginal 95% intervals from the shared bootstrap, the paired
QPPR-minus-classical differences with their intervals and sign probabilities,
and every comparator built from the development labels: the joint-cell
default rate, the one-hop reference count, the logit on the layer dummies,
the single-layer reference shares, the blank-field count (where the dataset
has blanks) and no skill. A comparator that wins is reported as such.

Strata: ``EC_strata``, both arms within the observation strata of each dataset, with the paired difference
------------------------------------------------------------------------------------------------------------
HMEQ: fully recorded (no blank field) against any blank field. Freddie Mac:
seller in the development vocabulary against an unseen seller. Within each
stratum the replicates are the shared resample indices restricted to the
stratum's members, so the pairing across arms is preserved and the stratum
size varies with the draw as it should.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from config import ExperimentConfig
from evaluation.context import (
    CLASSICAL,
    CLASSICAL_TAU_STAR,
    PAIRED,
    POINT_METRICS,
    QPPR,
    QPPR_DESTINATION,
    QPPR_TAU_STAR,
    Inputs,
    paired_difference_row,
    tie_columns,
)
from evaluation.controls import BASELINE_LABELS, CELL_RATE_LABEL, blank_field_count, build_baselines
from evaluation.provenance import ci_columns, write_result
from evaluation.reporting import (
    METRIC_FUNCTIONS,
    expected_top_k_overlap,
    interval,
    kendall_tau,
    no_skill_metrics,
    paired_difference,
    spearman,
)
from evaluation.ties import canonicalize, distinct_levels
from fingerprint import layer_columns_for
from quantum.mechanism.convergence import tau_star_label


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def agreement_row(data: Inputs, network: str) -> dict[str, Any]:
    """The agreement columns of one network."""
    k, d = data.k, data.decimals
    q, c = data.qppr(network), data.classical(network)
    q_tau = data.test(network, QPPR_TAU_STAR)
    conv = data.convergence(network)
    return {
        "dataset": data.dataset.name, "network": network,
        "rho_qc": spearman(q, c, decimals=d), "kendall_qc": kendall_tau(q, c, decimals=d),
        "ov_qc": expected_top_k_overlap(q, c, k), "selfov_q": expected_top_k_overlap(q, q, k),
        "selfov_c": expected_top_k_overlap(c, c, k), "levels_q": distinct_levels(q, d), "levels_c": distinct_levels(c, d),
        "rho_tau_star_vs_limit": spearman(q_tau, q, decimals=d), "tau_star": conv["tau_star"],
        "tau_star_reached": bool(conv["reached"]), "tau_star_printed": tau_star_label(conv), "route": conv["route"],
        "n_test": int(data.y_test.size), "k": k,
    }


def register_row(data: Inputs, network: str) -> dict[str, Any]:
    """Source against destination register limits, and each against classical."""
    k, d = data.k, data.decimals
    src, dst, c = data.qppr(network), data.test(network, QPPR_DESTINATION), data.classical(network)
    return {
        "dataset": data.dataset.name, "network": network,
        "rho_src_dst": spearman(src, dst, decimals=d), "kendall_src_dst": kendall_tau(src, dst, decimals=d),
        "ov_src_dst": expected_top_k_overlap(src, dst, k),
        "rho_src_c": spearman(src, c, decimals=d), "rho_dst_c": spearman(dst, c, decimals=d),
        "levels_src": distinct_levels(src, d), "levels_dst": distinct_levels(dst, d),
    }


def classical_horizon_row(data: Inputs, network: str) -> dict[str, Any]:
    """The classical running average at ``tau*`` against the classical stationary score."""
    c, c_tau = data.classical(network), data.test(network, CLASSICAL_TAU_STAR)
    conv = data.convergence(network)
    return {"dataset": data.dataset.name, "network": network, "tau_eval": conv["tau_eval"],
            "rho_classical_tau_star": spearman(c_tau, c, decimals=data.decimals),
            "kendall_classical_tau_star": kendall_tau(c_tau, c, decimals=data.decimals),
            "ov_classical_tau_star": expected_top_k_overlap(c_tau, c, data.k)}


def write_agreement_tables(config: ExperimentConfig, data: Inputs, *, source_stage: str) -> dict[str, Path]:
    """``T2_agreement``, ``EC_registers`` and ``EC_classical_horizon`` under the dataset's scope."""
    scope = data.dataset.name
    networks = list(data.dataset.ladder)
    k = data.k
    agreement = write_result(
        scope, "T2_agreement", pl.DataFrame([agreement_row(data, n) for n in networks], strict=False, infer_schema_length=None),
        stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name",
                 "rho_qc": "Spearman between qppr_limit and classical over test individuals",
                 "kendall_qc": "Kendall tau-b between qppr_limit and classical over test individuals",
                 "ov_qc": f"expected top-{k} overlap of the two rankings under uniform tie-breaking",
                 "selfov_q": f"self-overlap ceiling of qppr_limit at k = {k}", "selfov_c": f"self-overlap ceiling of classical at k = {k}",
                 "levels_q": "distinct test-score levels of qppr_limit at the tie tolerance",
                 "levels_c": "distinct test-score levels of classical at the tie tolerance",
                 "rho_tau_star_vs_limit": "Spearman between qppr_tau_star (running average at tau_eval) and qppr_limit",
                 "tau_star": "convergence point (grid horizon, or '> tau_max')", "tau_star_reached": "whether tau_star was reached",
                 "tau_star_printed": "tau_star as printed", "route": "discriminant the router selected",
                 "n_test": "test individuals", "k": "k of the top-k quantities"},
    )
    registers = write_result(
        scope, "EC_registers", pl.DataFrame([register_row(data, n) for n in networks], strict=False, infer_schema_length=None),
        stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name",
                 "rho_src_dst": "Spearman between the source-register and the destination-register limits",
                 "kendall_src_dst": "Kendall tau-b between the two register limits",
                 "ov_src_dst": f"expected top-{k} overlap between the two register limits",
                 "rho_src_c": "Spearman between the source-register limit and classical",
                 "rho_dst_c": "Spearman between the destination-register limit and classical",
                 "levels_src": "distinct levels of the source-register limit", "levels_dst": "distinct levels of the destination-register limit"},
    )
    horizon = write_result(
        scope, "EC_classical_horizon", pl.DataFrame([classical_horizon_row(data, n) for n in networks], strict=False, infer_schema_length=None),
        stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name",
                 "tau_eval": "horizon of the classical running average (tau_star when reached, else tau_max)",
                 "rho_classical_tau_star": "Spearman between the classical running average at tau_eval and the stationary score",
                 "kendall_classical_tau_star": "Kendall tau-b of the same pair",
                 "ov_classical_tau_star": f"expected top-{k} overlap of the same pair"},
    )
    return {"agreement": agreement, "registers": registers, "classical_horizon": horizon}


# ---------------------------------------------------------------------------
# Decision metrics
# ---------------------------------------------------------------------------


NO_SKILL = "no_skill"
BLANK_COUNT = "blank_field_count"
CELL_RATE = "cell_rate"


def comparator_scores(data: Inputs, spec, config: ExperimentConfig) -> dict[str, np.ndarray]:
    """``{name: test scores}`` for every comparator, fit on the development labels only."""
    frame = data.partitions.frame
    raw = build_baselines(frame, layer_columns_for(frame, spec), data.y, data.partitions.development_idx,
                          random_state=int(config.master_seed), include_single_layer=spec.n_layers > 1)
    out: dict[str, np.ndarray] = {}
    for name, values in raw.items():
        if name == NO_SKILL:
            continue
        key = CELL_RATE if name == "cell_seed_default_rate" else name
        out[key] = canonicalize(values[data.test_idx], data.decimals)
    if data.flags["blank_field_columns"]:
        out[BLANK_COUNT] = canonicalize(blank_field_count(data.flag_test("blank_field_count")), data.decimals)
    return out


def comparator_label(name: str, spec) -> str:
    if name == CELL_RATE:
        return f"{CELL_RATE_LABEL} ({', '.join(spec.layer_names)})"
    if name.startswith("single_layer_seed_default_rate_"):
        return f"single-layer reference share ({name.rsplit('_', 1)[-1]})"
    if name == BLANK_COUNT:
        return "blank-field count"
    return BASELINE_LABELS.get(name, name)


def decision_rows(data: Inputs, config: ExperimentConfig, network: str) -> list[dict[str, Any]]:
    """Every arm and comparator of one network with intervals, then the paired QPPR-minus-classical row."""
    spec = data.specs()[network]
    k = data.k
    q, c = data.qppr(network), data.classical(network)
    arms: list[tuple[str, str, np.ndarray]] = [(CLASSICAL, "classical stationary", c), (QPPR, "QPPR limit", q)]
    arms += [(name, comparator_label(name, spec), scores) for name, scores in comparator_scores(data, spec, config).items()]
    rows: list[dict[str, Any]] = []
    base = no_skill_metrics(data.y_test, (k,))
    row: dict[str, Any] = {"dataset": data.dataset.name, "network": network, "arm": NO_SKILL, "label": "no skill"}
    for metric in ("pr_auc", "roc_auc", f"precision_at_{k}", f"lift_at_{k}"):
        row[metric] = row[f"{metric}_ci_lo"] = row[f"{metric}_ci_hi"] = base[metric]
    row.update({"distinct_levels": 1, f"n_tied_at_{k}": int(data.y_test.size)})
    rows.append(row)
    for arm, label, scores in arms:
        row = {"dataset": data.dataset.name, "network": network, "arm": arm, "label": label}
        row.update(data.metrics(scores))
        row.update(tie_columns(scores, data.decimals, k))
        rows.append(row)
    rows.append(paired_difference_row(data, q, c, base={"dataset": data.dataset.name, "network": network,
                                                       "arm": PAIRED, "label": "QPPR minus classical (paired)"},
                                      overlap_key=f"top_{k}_overlap_qc"))
    for r in rows:
        r.setdefault("n_test", int(data.y_test.size))
        r.setdefault("n_test_pos", int(data.y_test.sum()))
    return rows


def table2_decision_rows(full_rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """The columns Table 2 carries: PR-AUC of both arms, the paired difference, and the cell-rate rows."""
    by = {(r["dataset"], r["network"], r["arm"]): r for r in full_rows}
    networks = sorted({(r["dataset"], r["network"]) for r in full_rows}, key=lambda x: str(x))
    out = []
    for dataset, network in networks:
        c, q, d = by[(dataset, network, CLASSICAL)], by[(dataset, network, QPPR)], by[(dataset, network, PAIRED)]
        cell = by.get((dataset, network, CELL_RATE))
        comparators = [r for key, r in by.items() if key[0] == dataset and key[1] == network
                       and key[2] not in (CLASSICAL, QPPR, PAIRED, NO_SKILL)]
        best = max(comparators, key=lambda r: r["pr_auc"]) if comparators else None
        out.append({
            "dataset": dataset, "network": network, "prauc_noskill": by[(dataset, network, NO_SKILL)]["pr_auc"],
            "prauc_c": c["pr_auc"], "prauc_q": q["pr_auc"], "d_prauc": d["pr_auc"],
            "d_prauc_lo": d["pr_auc_ci_lo"], "d_prauc_hi": d["pr_auc_ci_hi"],
            "p_delta_pos": d["pr_auc_p_delta_pos"], "p_delta_zero": d["pr_auc_p_delta_zero"], "p_delta_neg": d["pr_auc_p_delta_neg"],
            "roc_c": c["roc_auc"], "roc_q": q["roc_auc"], "d_roc": d["roc_auc"], "d_roc_lo": d["roc_auc_ci_lo"], "d_roc_hi": d["roc_auc_ci_hi"],
            "p100_c": c[f"precision_at_{k}"], "p100_q": q[f"precision_at_{k}"], "d_p100": d[f"precision_at_{k}"],
            "d_p100_lo": d[f"precision_at_{k}_ci_lo"], "d_p100_hi": d[f"precision_at_{k}_ci_hi"],
            "prauc_cellrate": None if cell is None else cell["pr_auc"],
            "best_comp_name": None if best is None else best["arm"], "best_comp_prauc": None if best is None else best["pr_auc"],
        })
    return out


def write_decision_tables(config: ExperimentConfig, data: Inputs, *, source_stage: str) -> dict[str, Any]:
    """``EC_decision_full`` (every arm, comparator and metric) and ``T2_decision`` under the dataset's scope."""
    scope = data.dataset.name
    k = data.k
    full = [row for network in data.dataset.ladder for row in decision_rows(data, config, network)]
    full_columns = {"dataset": "dataset name", "network": "network name",
                    "arm": "score: classical, qppr_limit, a comparator name, no_skill, or qppr_minus_classical (paired difference)",
                    "label": "printed name of the score",
                    **ci_columns("pr_auc", "PR-AUC on the test partition (paired difference on the paired row)"),
                    **ci_columns("roc_auc", "ROC-AUC on the test partition (paired difference on the paired row)"),
                    **ci_columns(f"precision_at_{k}", f"precision at {k} (expected over tie-breakings; paired difference on the paired row)"),
                    **ci_columns(f"lift_at_{k}", f"lift at {k} (paired difference on the paired row)"),
                    "distinct_levels": "distinct test-score levels at the tie tolerance",
                    f"n_tied_at_{k}": f"test individuals tied on the rank-{k} boundary",
                    f"top_{k}_overlap_qc": f"expected top-{k} overlap between qppr_limit and classical (paired row)",
                    f"top_{k}_overlap_qc_ci_lo": "lower end of its bootstrap interval", f"top_{k}_overlap_qc_ci_hi": "upper end of its bootstrap interval",
                    "n_test": "test individuals", "n_test_pos": "test positives"}
    for metric in ("pr_auc", "roc_auc", f"precision_at_{k}", f"lift_at_{k}"):
        for sign in ("pos", "zero", "neg"):
            full_columns[f"{metric}_p_delta_{sign}"] = f"share of shared replicates on which the paired {metric} difference is {sign}"
    full_table = write_result(scope, "EC_decision_full", pl.DataFrame(full, strict=False, infer_schema_length=None),
                              stage=source_stage, columns=full_columns)
    t2 = table2_decision_rows(full, k)
    t2_table = write_result(
        scope, "T2_decision", pl.DataFrame(t2, strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "prauc_noskill": "PR-AUC of no skill (test prevalence)",
                 "prauc_c": "PR-AUC of classical", "prauc_q": "PR-AUC of qppr_limit",
                 "d_prauc": "paired PR-AUC difference qppr_limit minus classical",
                 "d_prauc_lo": "lower end of its 95% paired bootstrap interval", "d_prauc_hi": "upper end of its 95% paired bootstrap interval",
                 "p_delta_pos": "share of replicates with a positive PR-AUC difference", "p_delta_zero": "share with a zero difference",
                 "p_delta_neg": "share with a negative difference",
                 "roc_c": "ROC-AUC of classical", "roc_q": "ROC-AUC of qppr_limit", "d_roc": "paired ROC-AUC difference",
                 "d_roc_lo": "lower end of its interval", "d_roc_hi": "upper end of its interval",
                 f"p100_c": f"precision at {k} of classical", "p100_q": f"precision at {k} of qppr_limit",
                 "d_p100": f"paired precision-at-{k} difference", "d_p100_lo": "lower end of its interval", "d_p100_hi": "upper end of its interval",
                 "prauc_cellrate": "PR-AUC of the joint-category development default rate (the cell rate)",
                 "best_comp_name": "comparator with the largest PR-AUC", "best_comp_prauc": "its PR-AUC"},
    )
    return {"full": full_table, "t2": t2_table, "rows": full, "t2_rows": t2}


# ---------------------------------------------------------------------------
# Observation strata
# ---------------------------------------------------------------------------


def strata_for(data: Inputs) -> dict[str, np.ndarray]:
    """The two strata of the dataset on the test partition, by what the dataset's absences mean."""
    if data.dataset.layer_absence_means == "blank_field":
        blank = data.flag_test("blank_field")
        return {"fully_recorded": ~blank, "any_blank_field": blank}
    unseen = np.asarray(data.flags["missing_per_layer"]["SellerName"])[data.test_idx] \
        if "SellerName" in data.flags["missing_per_layer"] else data.flag_test("unseen_level")
    return {"seller_known": ~unseen, "seller_unseen": unseen}


def stratum_replicates(scores, y, indices, mask, k) -> dict[str, np.ndarray]:
    """PR-AUC / ROC-AUC replicates within a stratum on the SAME shared resample indices."""
    reps = indices.shape[0]
    out = {name: np.full(reps, np.nan) for name in POINT_METRICS}
    for rep in range(reps):
        idx = indices[rep]
        idx = idx[mask[idx]]
        if idx.size == 0 or len(np.unique(y[idx])) < 2:
            continue
        for name in POINT_METRICS:
            out[name][rep] = METRIC_FUNCTIONS[name](scores[idx], y[idx], min(int(k), int(idx.size)))
    return out


def strata_rows(data: Inputs, network: str) -> list[dict[str, Any]]:
    """Per arm and stratum: size, default rate, PR-AUC and ROC-AUC with intervals; then the paired rows."""
    k = data.k
    arms = {CLASSICAL: data.classical(network), QPPR: data.qppr(network)}
    strata = strata_for(data)
    rows: list[dict[str, Any]] = []
    reps: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    points: dict[tuple[str, str], dict[str, float]] = {}
    for stratum, mask in strata.items():
        y_s = data.y_test[mask]
        for arm, scores in arms.items():
            row = {"dataset": data.dataset.name, "network": network, "stratum": stratum, "arm": arm,
                   "n": int(mask.sum()), "default_rate": float(y_s.mean()) if mask.any() else float("nan")}
            if mask.sum() > 0 and len(np.unique(y_s)) > 1:
                point = {m: METRIC_FUNCTIONS[m](scores[mask], y_s, min(k, int(mask.sum()))) for m in POINT_METRICS}
            else:
                point = {m: float("nan") for m in POINT_METRICS}
            r = stratum_replicates(scores, data.y_test, data.boot, mask, k)
            reps[(stratum, arm)], points[(stratum, arm)] = r, point
            for metric in POINT_METRICS:
                lo, hi = interval(r[metric])
                row[metric], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = point[metric], lo, hi
            rows.append(row)
        diff_row: dict[str, Any] = {"dataset": data.dataset.name, "network": network, "stratum": stratum,
                                    "arm": "qppr_minus_classical", "n": int(mask.sum()),
                                    "default_rate": float(y_s.mean()) if mask.any() else float("nan")}
        for metric in POINT_METRICS:
            diff = paired_difference(reps[(stratum, QPPR)][metric], reps[(stratum, CLASSICAL)][metric],
                                     points[(stratum, QPPR)][metric], points[(stratum, CLASSICAL)][metric])
            diff_row[metric], diff_row[f"{metric}_ci_lo"], diff_row[f"{metric}_ci_hi"] = diff["observed"], diff["ci_lo"], diff["ci_hi"]
            for sign in ("pos", "zero", "neg"):
                diff_row[f"{metric}_p_delta_{sign}"] = diff[f"p_delta_{sign}"]
        rows.append(diff_row)
    return rows


def write_strata_table(config: ExperimentConfig, data: Inputs, *, source_stage: str) -> Path:
    """``EC_strata`` on the dataset's primary network, under the dataset's scope."""
    rows = strata_rows(data, data.dataset.primary)
    columns = {"dataset": "dataset name", "network": "network name",
               "stratum": "observation stratum on the test partition (HMEQ: blank fields; Freddie Mac: seller vocabulary)",
               "arm": "classical, qppr_limit, or the paired difference qppr_minus_classical",
               "n": "test individuals in the stratum", "default_rate": "default rate within the stratum",
               **ci_columns("pr_auc", "PR-AUC within the stratum (paired difference on the difference row)"),
               **ci_columns("roc_auc", "ROC-AUC within the stratum (paired difference on the difference row)")}
    for metric in POINT_METRICS:
        for sign in ("pos", "zero", "neg"):
            columns[f"{metric}_p_delta_{sign}"] = f"share of shared replicates on which the paired {metric} difference is {sign}"
    return write_result(data.dataset.name, "EC_strata", pl.DataFrame(rows, strict=False, infer_schema_length=None),
                        stage=source_stage, columns=columns)
