"""The three synthetic experiments behind Figure 3, on the repository pipeline (network -> lump -> router -> both arms).

S1 (single layer, null control): dangling individuals are inert -- (a) as
non-reference test individuals they change no other score; (b) as a 10%
reference fraction they leave the ordering of the others unchanged.
S2 (single layer, size dispersion): two categories with equal reference
share and different sizes share one classical level and get two QPPR levels;
the effect grows with the dispersion of sizes and shrinks with the spread of
shares. S3 (two layers, partially observed test individuals): as their share
``m`` grows the arms diverge, QPPR's top 5% fills with them, and the paired
PR-AUC difference follows their relative risk ``kappa``. S3b is a sensitivity
run on S3's populations (same seeds, ``m`` of 0 and 0.3, ``kappa`` 1) that
places the development references per category instead of per joint cell
(``synthetic.generator``), to show whether the correlation between the arms
at ``m = 0`` depends on that choice; every row scores one population under
both placements and carries the paired difference. Every point is reported as
it comes out; nothing here is tuned to a direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.context import score_operator
from evaluation.reporting import expected_share, ranking_point_metrics, spearman
from evaluation.ties import canonicalize, distinct_levels
from quantum.mechanism.variants import primary_variant
from synthetic.generator import SyntheticInstance, generate_attribute_graph

S1_DANGLING = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3)
S2_LEVELS = (3, 6, 12, 24)
S2_DISPERSION = (0.0, 0.5, 1.0, 2.0)
S2_SPREAD = (0.0, 0.01, 0.03, 0.1)
S3_CONFIGS = {"hmeq_like": (6, 2), "fmac_like": (24, 19)}
S3_PARTIAL = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
S3_KAPPA = (0.5, 1.0, 2.0)
S3B_PARTIAL = (0.0, 0.3)
S3B_KAPPA = (1.0,)
PLACEMENTS = ("joint_cell", "category")
TOP_SHARE = 0.05
REF_SHARE_MEAN = 0.2
TEST_FRAC = 0.2


def score_instance(instance: SyntheticInstance, config) -> dict[str, Any]:
    """Both arms on the instance through the router: full-length score vectors and the route."""
    partitions = instance.partitions()
    variant = primary_variant(partitions, instance.spec, config, partitions.test_idx)
    scored = score_operator(variant, config)
    return {"classical": scored.classical, "qppr": scored.quantum.omega_source, "route": scored.routed.route,
            "residuals": scored.routed.residuals, "borrower_ids": scored.borrower_ids}


def _test_metrics(instance: SyntheticInstance, scores: dict[str, Any], mask: np.ndarray, config) -> dict[str, float]:
    """Spearman, levels and the paired PR-AUC difference of the two arms over ``mask`` within the test partition."""
    test = instance.test_idx
    keep = test[mask[test]]
    y = instance.frame["BAD"].to_numpy().astype(int)[keep]
    q = canonicalize(scores["qppr"][keep], config.tie_tolerance)
    c = canonicalize(scores["classical"][keep], config.tie_tolerance)
    out = {"n_eval": int(keep.size), "rho_qc": spearman(q, c, decimals=config.tie_tolerance),
           "levels_q": distinct_levels(q, config.tie_tolerance), "levels_c": distinct_levels(c, config.tie_tolerance)}
    if keep.size > 1 and len(np.unique(y)) > 1:
        k = max(1, int(round(TOP_SHARE * keep.size)))
        pq = ranking_point_metrics(q, y, (k,), decimals=config.tie_tolerance)
        pc = ranking_point_metrics(c, y, (k,), decimals=config.tie_tolerance)
        out.update(prauc_q=pq["pr_auc"], prauc_c=pc["pr_auc"], d_prauc=pq["pr_auc"] - pc["pr_auc"])
    else:
        out.update(prauc_q=float("nan"), prauc_c=float("nan"), d_prauc=float("nan"))
    return out


def s1_rows(config, *, n: int, replicates: int) -> list[dict[str, Any]]:
    rows = []
    for rep in range(int(replicates)):
        seed = config.seed_synthetic("S1", rep)
        for mode, ref_share in (("a_nonreference_test", 0.0), ("b_ten_percent_references", 0.1)):
            base_scores = None
            for share in S1_DANGLING:
                instance = generate_attribute_graph(n, 1, (12,), 0.5, REF_SHARE_MEAN, 0.03, TEST_FRAC, 0.0, 1.0, share, ref_share, seed)
                scores = score_instance(instance, config)
                n_base = instance.truth["n"]
                if share == 0.0:
                    base_scores = {k: scores[k][:n_base] for k in ("classical", "qppr")}
                change = {k: float(np.max(np.abs(scores[k][:n_base] - base_scores[k]))) for k in ("classical", "qppr")}
                order = {k: spearman(scores[k][:n_base], base_scores[k], decimals=config.tie_tolerance) for k in ("classical", "qppr")}
                metrics = _test_metrics(instance, scores, ~instance.dangling, config)
                rows.append({"experiment": "S1", "mode": mode, "replicate": rep, "seed": seed, "dangling_share": share,
                             "n_dangling": instance.truth["n_dangling"], "route": scores["route"],
                             "max_change_classical": change["classical"], "max_change_qppr": change["qppr"],
                             "rho_order_classical": order["classical"], "rho_order_qppr": order["qppr"], **metrics})
    return rows


def s2_rows(config, *, n: int, replicates: int) -> list[dict[str, Any]]:
    rows = []
    for rep in range(int(replicates)):
        for levels in S2_LEVELS:
            for dispersion in S2_DISPERSION:
                for spread in S2_SPREAD:
                    seed = config.seed_synthetic("S2", rep, levels, dispersion, spread)
                    instance = generate_attribute_graph(n, 1, (levels,), dispersion, REF_SHARE_MEAN, spread, TEST_FRAC, 0.0, 1.0, 0.0, 0.0, seed)
                    scores = score_instance(instance, config)
                    metrics = _test_metrics(instance, scores, np.ones(instance.n, dtype=bool), config)
                    rows.append({"experiment": "S2", "replicate": rep, "seed": seed, "levels": levels, "size_dispersion": dispersion,
                                 "ref_share_spread": spread, "route": scores["route"],
                                 "one_minus_rho": 1.0 - metrics["rho_qc"] if np.isfinite(metrics["rho_qc"]) else float("nan"),
                                 "abs_d_prauc": abs(metrics["d_prauc"]), **metrics})
    return rows


def s3_quantities(instance: SyntheticInstance, scores: dict[str, Any], config) -> dict[str, Any]:
    """The S3 quantities of one scored two-layer instance, shared by S3 and S3b."""
    test = instance.test_idx
    partial = instance.partially_observed
    all_m = _test_metrics(instance, scores, np.ones(instance.n, dtype=bool), config)
    full_m = _test_metrics(instance, scores, ~partial, config)
    k = max(1, int(round(TOP_SHARE * test.size)))
    q = canonicalize(scores["qppr"][test], config.tie_tolerance)
    c = canonicalize(scores["classical"][test], config.tie_tolerance)
    return {"n_partial_test": int(partial[test].sum()), "rho_qc_all": all_m["rho_qc"], "rho_qc_fully_observed": full_m["rho_qc"],
            "partial_share_top_q": expected_share(q, partial[test], k), "partial_share_top_c": expected_share(c, partial[test], k),
            "d_prauc": all_m["d_prauc"], "prauc_q": all_m["prauc_q"], "prauc_c": all_m["prauc_c"]}


def s3_rows(config, *, n: int, replicates: int) -> list[dict[str, Any]]:
    rows = []
    for rep in range(int(replicates)):
        for name, levels in S3_CONFIGS.items():
            for m in S3_PARTIAL:
                for kappa in S3_KAPPA:
                    seed = config.seed_synthetic("S3", rep, name, m, kappa)
                    instance = generate_attribute_graph(n, 2, levels, 0.5, REF_SHARE_MEAN, 0.03, TEST_FRAC, m, kappa, 0.0, 0.0, seed)
                    scores = score_instance(instance, config)
                    rows.append({"experiment": "S3", "config": name, "replicate": rep, "seed": seed, "partial_share": m, "kappa": kappa,
                                 "route": scores["route"], **s3_quantities(instance, scores, config)})
    return rows


def s3b_rows(config, *, n: int, replicates: int) -> list[dict[str, Any]]:
    """S3b: each S3 population (S3's seed) scored with the references placed per joint cell and per category.

    A row carries the per-category quantities under the S3 names, the same population's per-joint-cell correlation, the paired
    difference, and the largest gap between a category's realised reference share and its grid share under each placement.
    """
    rows = []
    for rep in range(int(replicates)):
        for name, levels in S3_CONFIGS.items():
            for m in S3B_PARTIAL:
                for kappa in S3B_KAPPA:
                    seed = config.seed_synthetic("S3", rep, name, m, kappa)
                    runs = {}
                    for placement in PLACEMENTS:
                        instance = generate_attribute_graph(n, 2, levels, 0.5, REF_SHARE_MEAN, 0.03, TEST_FRAC, m, kappa, 0.0, 0.0, seed,
                                                            reference_placement=placement)
                        scores = score_instance(instance, config)
                        runs[placement] = {"quantities": s3_quantities(instance, scores, config), "route": scores["route"],
                                           "gap": instance.truth["reference_share_gap"]}
                    cell, cat = runs["joint_cell"], runs["category"]
                    rho_cell, rho_cat = cell["quantities"]["rho_qc_all"], cat["quantities"]["rho_qc_all"]
                    paired = rho_cat - rho_cell if np.isfinite(rho_cat) and np.isfinite(rho_cell) else float("nan")
                    rows.append({"experiment": "S3b", "config": name, "replicate": rep, "seed": seed, "partial_share": m, "kappa": kappa,
                                 "route": cat["route"], "route_joint_cell": cell["route"], **cat["quantities"],
                                 "rho_qc_all_joint_cell": rho_cell, "d_rho_qc_all": paired,
                                 "ref_share_gap_category": cat["gap"], "ref_share_gap_joint_cell": cell["gap"]})
    return rows


def _band(values) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)), "n": int(v.size)}


def summarise(rows: list[dict[str, Any]], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    """Mean and 95% band of every metric per combination of ``keys`` over replicates."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        record = dict(zip(keys, key))
        record["n_replicates"] = len(members)
        for metric in metrics:
            band = _band([r.get(metric) for r in members])
            record[f"{metric}_mean"], record[f"{metric}_lo"], record[f"{metric}_hi"] = band["mean"], band["lo"], band["hi"]
        out.append(record)
    return out


def headline_keys(s1_summary, s2_summary, s3_summary, s3b_summary=None) -> dict[str, Any]:
    """The synthetic keys, read off the summaries; with S3b, also the correlation at ``m = 0`` under each placement."""
    def find(rows, **where):
        for r in rows:
            if all(abs(float(r[k]) - float(v)) < 1e-12 if isinstance(v, (int, float)) else r[k] == v for k, v in where.items()):
                return r
        return None

    s1a = [r for r in s1_summary if r["mode"] == "a_nonreference_test"]
    s2_sep = max((r["one_minus_rho_mean"] for r in s2_summary if np.isfinite(r["one_minus_rho_mean"])), default=float("nan"))
    keys = {"s1_max_change": max((r["max_change_qppr_mean"] for r in s1a), default=float("nan")),
            "s2_sep_max": s2_sep}
    for name in S3_CONFIGS:
        r0 = find(s3_summary, config=name, partial_share=0.0, kappa=1.0)
        r30 = find(s3_summary, config=name, partial_share=0.3, kappa=1.0)
        r20_05 = find(s3_summary, config=name, partial_share=0.2, kappa=0.5)
        r20_2 = find(s3_summary, config=name, partial_share=0.2, kappa=2.0)
        keys[f"{name}.s3_rho_m0"] = None if r0 is None else r0["rho_qc_all_mean"]
        keys[f"{name}.s3_rho_m30"] = None if r30 is None else r30["rho_qc_all_mean"]
        keys[f"{name}.s3_share_m30"] = None if r30 is None else r30["partial_share_top_q_mean"]
        keys[f"{name}.s3_dprauc_k05_m20"] = None if r20_05 is None else r20_05["d_prauc_mean"]
        keys[f"{name}.s3_dprauc_k2_m20"] = None if r20_2 is None else r20_2["d_prauc_mean"]
        if s3b_summary is not None:
            b0 = find(s3b_summary, config=name, partial_share=0.0, kappa=1.0)
            keys[f"{name}.s3b_rho_m0_category"] = None if b0 is None else b0["rho_qc_all_mean"]
            keys[f"{name}.s3b_rho_m0_joint_cell"] = None if b0 is None else b0["rho_qc_all_joint_cell_mean"]
            keys[f"{name}.s3b_d_rho_m0"] = None if b0 is None else b0["d_rho_qc_all_mean"]
    return keys


# Column definitions of the per-replicate files (``scripts/run_synthetic.py`` writes them).
_SHARED = {"experiment": "S1, S2, S3 or S3b", "replicate": "replicate index", "seed": "seed of the replicate (config.seed_synthetic)",
           "route": "discriminant the router selected", "n_eval": "test individuals the metrics were evaluated on",
           "rho_qc": "Spearman between qppr_limit and classical over the evaluated test individuals",
           "levels_q": "distinct levels of qppr_limit", "levels_c": "distinct levels of classical",
           "prauc_q": "PR-AUC of qppr_limit", "prauc_c": "PR-AUC of classical", "d_prauc": "PR-AUC of qppr_limit minus PR-AUC of classical"}
ROW_COLUMNS = {
    "S1": {**_SHARED, "mode": "a_nonreference_test (dangling non-reference test individuals) or b_ten_percent_references",
           "dangling_share": "share of individuals without a membership edge", "n_dangling": "dangling individuals",
           "max_change_classical": "max |classical - classical at share 0| over the non-dangling individuals",
           "max_change_qppr": "max |qppr_limit - qppr_limit at share 0| over the non-dangling individuals",
           "rho_order_classical": "Spearman of classical with its share-0 counterpart over the non-dangling",
           "rho_order_qppr": "Spearman of qppr_limit with its share-0 counterpart over the non-dangling"},
    "S2": {**_SHARED, "levels": "categories of the single layer", "size_dispersion": "Dirichlet-multinomial dispersion of category sizes",
           "ref_share_spread": "spread of the reference shares across categories", "one_minus_rho": "1 - rho_qc",
           "abs_d_prauc": "|d_prauc|"},
    "S3": {**_SHARED, "config": "hmeq_like (6, 2 levels) or fmac_like (24, 19 levels)",
           "partial_share": "share m of partially observed test individuals", "kappa": "relative default risk of the partially observed",
           "n_partial_test": "partially observed test individuals", "rho_qc_all": "Spearman between the arms over all test individuals",
           "rho_qc_fully_observed": "Spearman over the fully observed test individuals",
           "partial_share_top_q": "expected share of the partially observed in the QPPR top 5%",
           "partial_share_top_c": "the same for classical"},
}


_S3_QUANTITIES = ("n_partial_test", "rho_qc_all", "rho_qc_fully_observed", "partial_share_top_q", "partial_share_top_c", "d_prauc",
                  "prauc_q", "prauc_c")
ROW_COLUMNS["S3b"] = {
    "experiment": "S3b", "config": ROW_COLUMNS["S3"]["config"], "replicate": _SHARED["replicate"],
    "seed": "seed of the S3 population of the same configuration, m and kappa (config.seed_synthetic('S3', ...)): the population is the S3 one",
    "partial_share": ROW_COLUMNS["S3"]["partial_share"], "kappa": ROW_COLUMNS["S3"]["kappa"],
    "route": "discriminant the router selected with the references placed per category",
    "route_joint_cell": "discriminant the router selected with the references placed per joint cell (S3's placement)",
    **{q: f"{ROW_COLUMNS['S3'].get(q, _SHARED.get(q, q))} (references placed per category)" for q in _S3_QUANTITIES},
    "rho_qc_all_joint_cell": "Spearman between the arms over all test individuals, same population, references placed per joint cell (S3's value)",
    "d_rho_qc_all": "rho_qc_all (per category) minus rho_qc_all_joint_cell on the same population: the paired difference",
    "ref_share_gap_category": "largest gap between a category's realised reference share among the development individuals and its grid share, per-category placement",
    "ref_share_gap_joint_cell": "the same gap under the per-joint-cell placement",
}


def summary_columns(name: str, keys: tuple[str, ...], metrics: tuple[str, ...]) -> dict[str, str]:
    """Definitions of a summary file: the grouping keys, the replicate count and mean / 95% band per metric."""
    rows = ROW_COLUMNS[name]
    out = {key: rows.get(key, key) for key in keys}
    out["n_replicates"] = "replicates behind the row"
    for metric in metrics:
        what = rows.get(metric, metric)
        out[f"{metric}_mean"] = f"mean over replicates of {what}"
        out[f"{metric}_lo"] = f"2.5th percentile over replicates of {what}"
        out[f"{metric}_hi"] = f"97.5th percentile over replicates of {what}"
    return out
