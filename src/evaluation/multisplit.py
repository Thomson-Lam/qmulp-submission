"""The variability study behind Table 2's variability columns.

HMEQ: the processed frame is re-split with the distinct seeds
``{42} U {1..n_splits_hmeq}`` (a set, so seed 42 -- the frozen split, which
lies inside ``1..50`` -- is counted once: 50 distinct splits for n = 50; the
number is recorded as ``n_splits``), each split stratified 80/20 on the
outcome; both arms are rebuilt and routed on every split for every HMEQ
network, with the test labels of that split hidden. Nothing here writes
``frozen_split.json``. Freddie Mac has a calendar split and is not re-split:
its agreement measures get 95% intervals from the shared bootstrap matrix,
and ``n_reference_resamples_fmac`` random 80% subsets of the development
references are rebuilt and rescored (the dropped references become ordinary
development non-references). The two Freddie Mac summaries never share a
column name: ``boot_*`` is the bootstrap over test loans, ``resample_*`` the
summary over the reference resamples.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from config import ExperimentConfig
from datasets import DatasetConfig
from evaluation.context import Inputs, score_operator
from evaluation.masked_eval import DEVELOPMENT_PARTITION, TEST_PARTITION, Partitions
from evaluation.provenance import BOOT_PREFIX, RESAMPLE_PREFIX
from evaluation.reporting import expected_share, expected_top_k_overlap, interval, kendall_tau, ranking_point_metrics, spearman
from evaluation.ties import canonicalize, distinct_levels
from fingerprint import load_frozen_partitions, observation_flags
from hmeq.cleaning import stratified_split
from quantum.mechanism.convergence import convergence_curves, tau_star
from quantum.mechanism.variants import primary_variant

HMEQ_FROZEN_SEED = 42


def split_seeds(config: ExperimentConfig) -> list[int]:
    """The distinct seeds of the HMEQ study: ``{frozen seed} U {1..n_splits_hmeq}`` as a set, ascending.

    Seed 42 is the frozen split and also lies in ``1..50``; as a set it is counted once, so the study has
    ``n_splits_hmeq`` distinct splits when that is at least 42 (50 by default) and ``n_splits_hmeq + 1``
    below that (the smoke trim: 3). Set ``n_splits_hmeq = 51`` for 51 distinct splits.
    """
    return sorted({HMEQ_FROZEN_SEED} | set(range(1, int(config.n_splits_hmeq) + 1)))


def resplit(partitions: Partitions, seed: int, test_frac: float) -> Partitions:
    """A fresh stratified split of the same frame; the frozen split file is never touched."""
    y = partitions.y_true
    _dev, test = stratified_split(y, test_frac=test_frac, split_seed=int(seed))
    partition = np.full(y.size, DEVELOPMENT_PARTITION, dtype=object)
    partition[test] = TEST_PARTITION
    return Partitions(frame=partitions.frame, borrower_ids=partitions.borrower_ids, partition=partition.astype(str),
                      target_col=partitions.target_col, processed_path=None, split_path=None)


def _arms_on(partitions: Partitions, spec, config: ExperimentConfig) -> dict[str, Any]:
    scored = score_operator(primary_variant(partitions, spec, config, partitions.test_idx), config)
    return {"scored": scored, "q": scored.quantum.omega_source, "c": scored.classical}


def split_row(dataset: DatasetConfig, config: ExperimentConfig, partitions: Partitions, specs, flags, *,
              seed: int, with_tau_star: bool) -> list[dict[str, Any]]:
    """One row per network of one split: every agreement, decision and tie quantity of Table 2."""
    k, d = int(config.primary_k), int(config.tie_tolerance)
    test = partitions.test_idx
    y = partitions.y_true[test]
    partial = np.asarray(flags["partially_observed"], dtype=bool)[test]
    rows = []
    single_identical = {}
    for network in dataset.ladder:
        arms = _arms_on(partitions, specs[network], config)
        q, c = canonicalize(arms["q"][test], d), canonicalize(arms["c"][test], d)
        pq, pc = ranking_point_metrics(q, y, (k,), decimals=d), ranking_point_metrics(c, y, (k,), decimals=d)
        rho = spearman(q, c, decimals=d)
        row = {"dataset": dataset.name, "network": network, "seed": int(seed), "n_test": int(test.size),
               "rho_qc": rho, "kendall_qc": kendall_tau(q, c, decimals=d), "ov_qc": expected_top_k_overlap(q, c, k),
               "selfov_q": expected_top_k_overlap(q, q, k), "selfov_c": expected_top_k_overlap(c, c, k),
               "levels_q": distinct_levels(q, d), "levels_c": distinct_levels(c, d),
               "partial_share_top100_q": expected_share(q, partial, k), "partial_share_top100_c": expected_share(c, partial, k),
               "d_prauc": pq["pr_auc"] - pc["pr_auc"], "d_roc": pq["roc_auc"] - pc["roc_auc"],
               "d_p100": pq[f"precision_at_{k}"] - pc[f"precision_at_{k}"], "route": arms["scored"].routed.route,
               "tau_star": None, "tau_star_reached": None}
        if specs[network].n_layers == 1:
            single_identical[network] = bool(np.isfinite(rho) and abs(rho - 1.0) < 1e-12)
        if with_tau_star and network == dataset.primary:
            scored = arms["scored"]
            curves = convergence_curves(scored.quantum, scored.nodes, test, tau_max=config.tau_max_for(dataset.name), k=k, decimals=d)
            star = tau_star(curves["rows"], tol_l1=config.conv_tol_l1, min_spearman=config.conv_min_spearman,
                            overlap_slack=config.conv_overlap_slack, min_tail_points=config.conv_min_tail_points)
            row["tau_star"], row["tau_star_reached"] = star["tau_star"], star["reached"]
        rows.append(row)
    for row in rows:
        for network, identical in single_identical.items():
            row[f"single_identical_{network}"] = identical
    return rows


def hmeq_multisplit(dataset: DatasetConfig, config: ExperimentConfig, *, with_tau_star: bool = True) -> list[dict[str, Any]]:
    """Every HMEQ split (the distinct seeds of :func:`split_seeds`, the frozen one among them), every network."""
    base = load_frozen_partitions(dataset)
    flags = observation_flags(base.frame, dataset)
    specs = dataset.network_specs(base.frame)
    seeds = split_seeds(config)
    rows = []
    for seed in seeds:
        partitions = base if seed == HMEQ_FROZEN_SEED and base.split_path is not None else resplit(base, seed, config.test_frac)
        rows += split_row(dataset, config, partitions, specs, flags, seed=seed, with_tau_star=with_tau_star)
    return rows


def fmac_bootstrap_intervals(data: Inputs) -> list[dict[str, Any]]:
    """95% intervals of Spearman, Kendall and the top-k overlap from the shared resample matrix."""
    d, k = data.decimals, data.k
    rows = []
    for network in data.dataset.ladder:
        q, c = data.qppr(network), data.classical(network)
        rows.append({"dataset": data.dataset.name, "network": network, "rho_qc": spearman(q, c, decimals=d),
                     "kendall_qc": kendall_tau(q, c, decimals=d), "ov_qc": expected_top_k_overlap(q, c, k)})
        for name, measure in (("rho_qc", lambda a, b: spearman(a, b, decimals=d)),
                              ("kendall_qc", lambda a, b: kendall_tau(a, b, decimals=d)),
                              ("ov_qc", lambda a, b: expected_top_k_overlap(a, b, min(k, a.size)))):
            lo, hi = interval(data.agreement_replicates(q, c, measure))
            rows[-1][f"{name}_lo"], rows[-1][f"{name}_hi"] = lo, hi
    return rows


def fmac_reference_resamples(dataset: DatasetConfig, config: ExperimentConfig) -> list[dict[str, Any]]:
    """Random 80% subsets of the development references, rebuilt and rescored on every network."""
    base = load_frozen_partitions(dataset)
    flags = observation_flags(base.frame, dataset)
    specs = dataset.network_specs(base.frame)
    dev = base.development_idx
    references = dev[base.y_true[dev] == 1]
    rows = []
    for draw in range(int(config.n_reference_resamples_fmac)):
        rng = np.random.default_rng(config.seed_reference_resample(draw))
        keep = np.sort(rng.choice(references, size=int(round(0.8 * references.size)), replace=False))
        labels = base.y_true.copy()
        labels[np.setdiff1d(references, keep)] = 0
        partitions = Partitions(frame=base.frame_with_labels(labels), borrower_ids=base.borrower_ids, partition=base.partition,
                                target_col=base.target_col, processed_path=None, split_path=None)
        for row in split_row(dataset, config, partitions, specs, flags, seed=config.seed_reference_resample(draw), with_tau_star=False):
            row["draw"], row["n_references_kept"] = draw, int(keep.size)
            rows.append(row)
    return rows


def variability_summary(rows: list[dict[str, Any]], *, frozen_seed: int | None) -> list[dict[str, Any]]:
    """Median, IQR, min, max per network and quantity, plus the frozen split's percentile among the splits."""
    quantities = ("rho_qc", "kendall_qc", "ov_qc", "levels_q", "levels_c", "partial_share_top100_q", "d_prauc", "d_roc", "d_p100")
    out = []
    for network in sorted({r["network"] for r in rows}):
        mine = [r for r in rows if r["network"] == network]
        record: dict[str, Any] = {"dataset": mine[0]["dataset"], "network": network, "n_splits": len(mine)}
        for q in quantities:
            values = np.array([r[q] for r in mine if r.get(q) is not None and np.isfinite(r[q])], dtype=float)
            if values.size == 0:
                continue
            record[f"{q}_med"] = float(np.median(values))
            record[f"{q}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
            record[f"{q}_min"], record[f"{q}_max"] = float(values.min()), float(values.max())
            record[f"{q}_lo"], record[f"{q}_hi"] = float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
            if frozen_seed is not None:
                frozen = [r[q] for r in mine if r["seed"] == frozen_seed]
                if frozen:
                    record[f"{q}_frozen"] = float(frozen[0])
                    record[f"{q}_frozen_percentile"] = float((values <= frozen[0]).mean())
        for key in [k for k in mine[0] if k.startswith("single_identical_")]:
            record[f"n_{key}"] = int(sum(bool(r[key]) for r in mine))
        # tau* is evaluated on the primary network only. The count of splits that did not reach it is recorded
        # whenever it was evaluated, including when none reached it (the median then has nothing to summarise).
        evaluated = [r for r in mine if r.get("tau_star") is not None]
        if evaluated:
            record["n_tau_star_evaluated"] = len(evaluated)
            record["n_tau_star_not_reached"] = int(sum(1 for r in evaluated if not r["tau_star_reached"]))
            stars = [r["tau_star"] for r in evaluated if r["tau_star_reached"]]
            if stars:
                record["tau_star_med"], record["tau_star_min"], record["tau_star_max"] = float(np.median(stars)), min(stars), max(stars)
        out.append(record)
    return out


def resample_summary(summary: list[dict[str, Any]], intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Freddie Mac ``T2_variability`` rows: the reference-resample summary as ``resample_*`` columns and the
    bootstrap intervals over test loans as ``boot_*`` columns, never under the same name.

    ``n_splits`` of the generic summary becomes ``n_resamples`` (a calendar split is not re-split).
    """
    by_network = {r["network"]: r for r in intervals}
    out = []
    for record in summary:
        row: dict[str, Any] = {"dataset": record["dataset"], "network": record["network"], "n_resamples": record["n_splits"]}
        row.update({f"{RESAMPLE_PREFIX}{k}": v for k, v in record.items() if k not in ("dataset", "network", "n_splits")})
        row.update({f"{BOOT_PREFIX}{k}": v for k, v in by_network[record["network"]].items() if k not in ("dataset", "network")})
        out.append(row)
    return out


# Column definitions of the variability files (``scripts/run_multisplit.py`` writes them).
LONG_COLUMNS = {
    "dataset": "dataset name", "network": "network name",
    "seed": "split seed (HMEQ re-splits) or reference-resample seed (Freddie Mac)", "n_test": "test individuals of the split",
    "rho_qc": "Spearman between qppr_limit and classical over the split's test individuals",
    "kendall_qc": "Kendall tau-b of the same pair", "ov_qc": "expected top-k overlap of the same pair",
    "selfov_q": "self-overlap ceiling of qppr_limit", "selfov_c": "self-overlap ceiling of classical",
    "levels_q": "distinct test levels of qppr_limit", "levels_c": "distinct test levels of classical",
    "partial_share_top100_q": "expected share of partially observed test individuals in the QPPR top k",
    "partial_share_top100_c": "the same for classical",
    "d_prauc": "PR-AUC of qppr_limit minus PR-AUC of classical on the split", "d_roc": "ROC-AUC difference", "d_p100": "precision-at-k difference",
    "route": "discriminant the router selected", "tau_star": "convergence point of the primary network on this split (null elsewhere)",
    "tau_star_reached": "whether that convergence point was reached before tau_max",
    "draw": "reference-resample draw index (Freddie Mac)", "n_references_kept": "development references kept in the draw (Freddie Mac)",
}


def _bootstrap_columns(prefix: str = "") -> dict[str, str]:
    out = {}
    for q, what in (("rho_qc", "Spearman"), ("kendall_qc", "Kendall tau-b"), ("ov_qc", "expected top-k overlap")):
        out[f"{prefix}{q}"] = f"{what} between qppr_limit and classical on the frozen test partition"
        out[f"{prefix}{q}_lo"] = f"2.5th percentile of {what} over the shared bootstrap replicates of test loans"
        out[f"{prefix}{q}_hi"] = f"97.5th percentile of {what} over the shared bootstrap replicates of test loans"
    return out


def summary_columns(kind: str, single_layer: Sequence[str] = ()) -> dict[str, str]:
    """Definitions of one variability file's columns.

    ``summary`` is HMEQ's ``T2_variability`` (percentiles over the re-splits, ``n_splits`` of them);
    ``resample`` is Freddie Mac's ``T2_variability`` (``resample_*`` over the 80% reference resamples,
    ``boot_*`` the bootstrap over test loans, ``n_resamples`` of them); ``bootstrap`` is
    ``T2_variability_bootstrap``. ``single_layer`` names the single-layer networks whose
    ``n_single_identical_<network>`` count is defined.
    """
    if kind == "bootstrap":
        return {"dataset": "dataset name", "network": "network name", **_bootstrap_columns()}
    unit = "re-splits" if kind == "summary" else "reference resamples"
    core = {"tau_star_med": f"median convergence point over the {unit} where it was reached (primary network only)",
            "tau_star_min": "its minimum", "tau_star_max": "its maximum",
            "n_tau_star_evaluated": f"{unit} on which the convergence point was evaluated (primary network only)",
            "n_tau_star_not_reached": f"{unit} on which the convergence point was not reached (primary network only)"}
    for network in single_layer:
        core[f"n_single_identical_{network}"] = (f"{unit} on which the single-layer network {network} gives a QPPR limit with "
                                                 "Spearman exactly one against its classical score")
    for q in ("rho_qc", "kendall_qc", "ov_qc", "levels_q", "levels_c", "partial_share_top100_q", "d_prauc", "d_roc", "d_p100"):
        core[f"{q}_med"] = f"median of {q} over the {unit}"
        core[f"{q}_iqr"] = f"interquartile range of {q} over the {unit}"
        core[f"{q}_min"] = f"minimum of {q} over the {unit}"
        core[f"{q}_max"] = f"maximum of {q} over the {unit}"
        core[f"{q}_lo"] = f"2.5th percentile of {q} over the {unit} (not a bootstrap interval)"
        core[f"{q}_hi"] = f"97.5th percentile of {q} over the {unit}"
        if kind == "summary":
            core[f"{q}_frozen"] = f"{q} on the frozen split"
            core[f"{q}_frozen_percentile"] = f"share of the splits with {q} at or below the frozen split's value"
    if kind == "summary":
        return {"dataset": "dataset name", "network": "network name",
                "n_splits": "distinct re-split seeds: {frozen seed 42} U {1..n_splits_hmeq}, counted as a set", **core}
    if kind != "resample":
        raise ValueError(f"unknown variability file kind {kind!r}")
    out = {"dataset": "dataset name", "network": "network name",
           "n_resamples": "80% reference resamples (config.n_reference_resamples_fmac); the calendar split is not re-split"}
    out.update({f"{RESAMPLE_PREFIX}{name}": definition for name, definition in core.items()})
    out.update(_bootstrap_columns(BOOT_PREFIX))
    return out
