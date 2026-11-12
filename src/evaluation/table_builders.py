"""The composite builders behind the tables that are not one CSV: frames assembled from several result files.

``evaluation.table_specs`` names a builder in the :class:`TableSpec` of every table whose rows combine several
result CSVs or are laid out differently from them (T1 reduction facts, T2 agreement with decision and variability,
T5 and T6 with the instance facts in the group label, and the e-companion convergence, variability, synthetic,
noise-grid, statevector, circuit-resource and network-grouped tables). A builder takes the source frames keyed by
``(scope, name)`` and returns the frame the spec's columns read, with every cell that needs composing already
formatted (the ``RAW`` columns). Nothing here reads a file or writes one.
"""

from __future__ import annotations

import re
from typing import Any, Callable

import polars as pl

from config import DEFAULT_CONFIG
from datasets import DATASET_NAMES, get_dataset
from evaluation.formatting import (
    DATASET_LABELS, NETWORK_LABELS, SYNTHETIC_LABELS, category_label, ci_cell, coerce_value, format_cell, format_value, human_label,
    node_label,
)
from evaluation.provenance import BOOT_PREFIX, RESAMPLE_PREFIX
from quantum.hardware.noise_grading import NOT_EVALUATED, first_tau_without_signal, is_simulated

Builder = Callable[[dict[tuple[str, str], pl.DataFrame | None]], pl.DataFrame | None]


def _with_dataset(frame: pl.DataFrame, ds: str) -> pl.DataFrame:
    return frame if "dataset" in frame.columns else frame.with_columns(pl.lit(ds).alias("dataset"))


def stacked(frames: dict[tuple[str, str], pl.DataFrame | None], name: str) -> pl.DataFrame | None:
    """Every scope's copy of ``name`` stacked, datasets first (HMEQ, then Freddie Mac); ``None`` when no scope has it.

    A dataset scope's rows get a ``dataset`` column when the file carries none;
    a dataset-free scope (``synthetic``, ``runtime``) is stacked as it is.
    """
    order = {ds: i for i, ds in enumerate(DATASET_NAMES)}
    scopes = sorted((scope for (scope, n), frame in frames.items() if n == name and frame is not None),
                    key=lambda s: (order.get(s, len(order)), s))
    parts = [_with_dataset(frames[(scope, name)], scope) if scope in order else frames[(scope, name)] for scope in scopes]
    if not parts:
        return None
    return pl.concat([p.select([pl.col(c).cast(pl.Utf8) for c in p.columns]) for p in parts], how="diagonal")


def _rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    return [] if frame is None else frame.to_dicts()


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame | None:
    return pl.DataFrame(rows, strict=False, infer_schema_length=None) if rows else None


def ladder_order(ds: str) -> dict[str, int]:
    return {network: i for i, network in enumerate(get_dataset(ds).ladder)}


def variability_view(row: dict[str, Any], ds: str) -> dict[str, Any]:
    """One ``T2_variability`` row under the generic names the builders read.

    HMEQ's columns are the summary over its re-splits, as written. Freddie Mac's file carries the summary
    over the reference resamples as ``resample_*`` (here stripped of the prefix, ``n_resamples`` read as
    ``n_splits``) beside the bootstrap over test loans as ``boot_*`` (kept as they are).
    """
    if get_dataset(ds).split == "stratified":
        return dict(row)
    view = {k[len(RESAMPLE_PREFIX):]: v for k, v in row.items() if k.startswith(RESAMPLE_PREFIX)}
    view.update({k: v for k, v in row.items() if k.startswith(BOOT_PREFIX) or k in ("dataset", "network")})
    view["n_splits"] = row.get("n_resamples")
    return view


def _group_label(row: dict[str, Any], *parts: str) -> str:
    """The group label of a row: its dataset, then the labels of ``parts`` (``HMEQ: two-layer``)."""
    return f"{DATASET_LABELS.get(row['dataset'], row['dataset'])}: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Network-grouped tables: the network becomes a group title, so the row label is the score or the category
# ---------------------------------------------------------------------------

def with_network_groups(frame: pl.DataFrame | None) -> pl.DataFrame | None:
    """The frame with a ``group_label`` column naming each row's dataset and network (``HMEQ: two-layer``)."""
    if frame is None:
        return None
    rows = frame.to_dicts()
    for row in rows:
        row["group_label"] = _group_label(row, NETWORK_LABELS.get(row["network"], row["network"]))
    return _frame(rows)


def network_grouped(name: str) -> Builder:
    """A builder that stacks the datasets' copies of ``name`` and groups their rows by dataset and network."""

    def build(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
        return with_network_groups(stacked(frames, name))

    return build


def interval_cell(row: dict[str, Any], metric: str) -> str:
    """``estimate [lo, hi]`` of ``metric`` with its 95% interval; the paired-difference row (``qppr_minus_classical`` or the
    paired decision label) prints as a signed difference, every other row as a level."""
    paired = row.get("arm") == "qppr_minus_classical"
    return ci_cell(coerce_value(row.get(metric)), coerce_value(row.get(f"{metric}_ci_lo")), coerce_value(row.get(f"{metric}_ci_hi")), "diff" if paired else "corr")


def build_ec_decision_full(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The decision table grouped by dataset and network; each estimate is one cell with its interval, a difference on the paired row."""
    frame = with_network_groups(stacked(frames, "EC_decision_full"))
    if frame is None:
        return None
    k = int(DEFAULT_CONFIG.primary_k)
    rows = frame.to_dicts()
    for row in rows:
        paired = row.get("arm") == "qppr_minus_classical"
        row["pr_auc_cell"] = interval_cell(row, "pr_auc")
        row["roc_auc_cell"] = interval_cell(row, "roc_auc")
        row["precision_cell"] = interval_cell(row, f"precision_at_{k}")
        row["lift_cell"] = format_cell(coerce_value(row.get(f"lift_at_{k}")), "diff" if paired else "mean")
    return _frame(rows)


def build_ec_strata(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The strata table with each stratum, its size and its default rate as a group title: the rows are the three arms."""
    frame = stacked(frames, "EC_strata")
    if frame is None:
        return None
    rows = frame.to_dicts()
    for row in rows:
        facts = f"$n = {format_cell(coerce_value(row.get('n')), 'count')}$, default rate {format_cell(coerce_value(row.get('default_rate')), 'corr')}"
        row["group_label"] = _group_label(row, f"{human_label(row['stratum'])} ({facts})")
        row["pr_auc_cell"] = interval_cell(row, "pr_auc")
        row["roc_auc_cell"] = interval_cell(row, "roc_auc")
    return _frame(rows)


def build_ec_single_layer_share(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The single-layer check grouped by dataset and network, with the category named without its layer prefix."""
    frame = with_network_groups(stacked(frames, "EC_single_layer_share"))
    if frame is None:
        return None
    rows = frame.to_dicts()
    for row in rows:
        row["category_name"] = category_label(row["category"])
    return _frame(rows)


def build_ec_hardware_scores(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The retained instance's scores with each supra-node named by its layer and class (``employment, class 25``)."""
    frame = stacked(frames, "EC_hardware_scores")
    if frame is None:
        return None
    rows = frame.to_dicts()
    for row in rows:
        row["node_name"] = node_label(row["label"])
    return _frame(rows)


def build_ec_circuit_resources(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The circuit resources with the number of classes as the row label, ``(full)`` marking the unreduced instance."""
    frame = stacked(frames, "EC_circuit_resources")
    if frame is None:
        return None
    rows = frame.to_dicts()
    for row in rows:
        count = format_cell(coerce_value(row.get("classes")), "count")
        row["classes_label"] = f"{count} (full)" if row.get("instance") == "full" else count
    return _frame(rows)


# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------

T1_ROWS: tuple[tuple[str, str, str], ...] = (
    ("n_raw", "Records supplied", "count"), ("n_dropped", "Records dropped (no category)", "count"),
    ("n_borrowers", "Individuals retained", "count"), ("n_dev", "Development", "count"), ("n_test", "Test", "count"),
    ("n_ref", "Development references $|D|$", "count"), ("n_test_pos", "Test positives", "count"),
    ("prev_dev", "Prevalence, development", "corr"), ("prev_test", "Prevalence, test", "corr"), ("split", "Split", "text"),
    ("layer_l1", "Layer 1", "text"), ("levels_l1", "Levels, layer 1", "count"), ("n_missing_l1", "No category in layer 1", "count"),
    ("layer_l2", "Layer 2", "text"), ("levels_l2", "Levels, layer 2", "count"), ("n_missing_l2", "No category in layer 2", "count"),
    ("n_partial_test", "Partially observed (test)", "count"),
    ("n_slots", "Category slots reserved", "count"), ("N_full", "Supra-nodes $N$", "count"),
    ("n_classes", "Lumping classes", "count"), ("R", "Lumped supra-nodes $R$", "count"), ("m", "Register width $m$", "int"),
    ("qubits", "Logical qubits $2m$", "int"), ("angles", "State-preparation angles", "count"),
)
# The T1 quantities read off the primary network's convergence record rather than T1_data.
T1_FROM_CONVERGENCE = ("n_slots", "N_full", "n_classes", "R", "m", "qubits", "angles")


def short_split(ds: str, mine: dict[str, Any]) -> str:
    """The split in a few words (``stratified 80/20, seed 42``, ``temporal, 2023 -> 2024``); the full description stays in the CSV."""
    text = str(mine.get("split") or "")
    if get_dataset(ds).split == "stratified":
        seed = mine.get("split_seed")
        if seed is None:
            found = re.search(r"seed\s+(\d+)", text)
            seed = found.group(1) if found else None
        return "stratified 80/20" + (f", seed {int(float(seed))}" if seed is not None else "")
    years = re.findall(r"(?<!\d)(20\d\d)(?!\d)", text)
    return f"temporal, {years[0]} $\\to$ {years[-1]}" if len(years) >= 2 else (text or "temporal")


def build_t1(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """Quantities as rows, one column per dataset; the reduction facts come from the primary network's convergence row."""
    values: dict[str, dict[str, Any]] = {}
    for ds in DATASET_NAMES:
        data = frames.get((ds, "T1_data"))
        if data is None:
            continue
        mine = {r["key"]: coerce_value(r["value"]) for r in _rows(data)}
        conv = frames.get((ds, "T_convergence"))
        if conv is not None:
            primary = [r for r in _rows(conv) if r.get("network") == get_dataset(ds).primary]
            if primary:
                mine.update({q: coerce_value(primary[0].get(q)) for q in T1_FROM_CONVERGENCE})
        values[ds] = mine
    if not values:
        return None
    rows = []
    for key, label, rule in T1_ROWS:
        row = {"quantity": label}
        for ds in DATASET_NAMES:
            if ds not in values:
                row[ds] = "--"
            else:
                row[ds] = short_split(ds, values[ds]) if key == "split" else format_cell(values[ds].get(key), rule)
        rows.append(row)
    return pl.DataFrame(rows)


def build_t2(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """Agreement, decision and variability per network; the joint-cell score sits on the dataset's two-layer row."""
    out: list[dict[str, Any]] = []
    any_data = False
    for ds in DATASET_NAMES:
        agreement = frames.get((ds, "T2_agreement"))
        if agreement is None:
            continue
        any_data = True
        decision = {r["network"]: r for r in _rows(frames.get((ds, "T2_decision")))}
        variability = {r["network"]: variability_view(r, ds) for r in _rows(frames.get((ds, "T2_variability")))}
        order = ladder_order(ds)
        primary = get_dataset(ds).primary
        for row in sorted(_rows(agreement), key=lambda r: order.get(r["network"], 99)):
            network = row["network"]
            d, v = decision.get(network, {}), variability.get(network, {})
            out.append({
                "dataset": ds, "network": network, "rho_qc": row["rho_qc"], "kendall_qc": row["kendall_qc"], "ov_qc": row["ov_qc"],
                "selfov_q": row["selfov_q"], "selfov_c": row["selfov_c"], "levels_q": row["levels_q"], "levels_c": row["levels_c"],
                "tau_star_printed": row["tau_star_printed"], "prauc_c": d.get("prauc_c"), "prauc_q": d.get("prauc_q"),
                "d_prauc": d.get("d_prauc"), "d_prauc_lo": d.get("d_prauc_lo"), "d_prauc_hi": d.get("d_prauc_hi"),
                "prauc_score": d.get("prauc_cellrate") if network == primary else None,
                # HMEQ: the range over re-splits; Freddie Mac: the range over reference resamples, and the bootstrap interval over test loans.
                "rho_splits": ci_cell(v.get("rho_qc_med"), v.get("rho_qc_min"), v.get("rho_qc_max"), "corr") if v.get("rho_qc_med") is not None else "--",
                "rho_boot": (f"[{format_value(v['boot_rho_qc_lo'], 'corr')}, {format_value(v['boot_rho_qc_hi'], 'corr')}]"
                             if v.get("boot_rho_qc_lo") is not None and v.get("boot_rho_qc_hi") is not None else "--"),
            })
    return _frame(out) if any_data else None


def build_t5(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The runtime rows of each dataset; ``N`` and ``R`` are constants of the dataset and become its group title."""
    frame = stacked(frames, "T5_runtime")
    if frame is None:
        return None
    rows = [{k: coerce_value(v) for k, v in row.items()} for row in frame.to_dicts()]
    facts: dict[str, str] = {}
    for ds in DATASET_NAMES:
        mine = [r for r in rows if r.get("dataset") == ds]
        if mine:
            n = next((r["N_full"] for r in mine if r.get("N_full") is not None), None)
            reduced = next((r["R"] for r in mine if r.get("R") is not None), None)
            facts[ds] = (f"{DATASET_LABELS[ds]}: $N = {format_cell(n, 'count')}$ supra-nodes, "
                         f"$R = {format_cell(reduced, 'count')}$ lumped")
    for row in rows:
        row["group_label"] = facts[row["dataset"]]
    return _frame(rows)


# The rows of Table 6: the quantity, its source column and its formatting rule.
T6_QUANTITIES = (
    (r"$\rho$ (supra-nodes)", "rho", "corr"), (r"$\rho$ (masked classes)", "rho_masked_cell", None), ("TV", "tv", "corr"),
    (r"TV$_{\mathrm{uniform}}$", "tv_uniform", "corr"), ("Fidelity ratio", "fr", "corr"), ("Padding mass", "padding", "corr"),
    ("Depth (max)", "depth_max", "count"), ("Two-qubit gates (sum)", "two_qubit_sum", "count"), ("Shots", "shots", "count"),
    ("QPU seconds (est.)", "qpu_seconds", "mean"),
)


def build_t6(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The quantities of each dataset's retained instance as rows, one column per arm; the instance facts become the group title."""
    out = []
    for ds in DATASET_NAMES:
        table = frames.get((ds, "T6_hardware"))
        if table is None:
            continue
        header = (_rows(frames.get((ds, "T6_hardware_header"))) or [{}])[0]
        facts = (f"{DATASET_LABELS[ds]}: {format_cell(header.get('hw_k'), 'count')} classes, "
                 f"{format_cell(header.get('hw_borrowers'), 'count')} borrowers, $R = {format_cell(header.get('hw_R'), 'count')}$, "
                 f"$m = {format_cell(header.get('hw_m'), 'int')}$, $\\tau = {format_cell(header.get('hw_tau'), 'int')}$, "
                 f"route {format_cell(header.get('hw_route'), 'route')}")
        by_arm = {row["arm_key"]: row for row in _rows(table)}
        for label, key, rule in T6_QUANTITIES:
            cells = {}
            for arm_key in ("exact", "noiseless", "aernoise", "fake"):
                row = by_arm.get(arm_key)
                if row is None:
                    cells[arm_key] = "--"
                elif key == "rho_masked_cell":
                    cells[arm_key] = row.get("rho_masked_note") or format_cell(coerce_value(row.get("rho_masked")), "corr")
                else:
                    cells[arm_key] = format_cell(coerce_value(row.get(key)), rule)
            out.append({"dataset": ds, "group_label": facts, "quantity": label, **cells})
    return _frame(out)


def build_ec_convergence(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """``T_convergence`` per network with the smallest and largest ``tau*`` of the tolerance grid and the settings that did not reach it."""
    out = []
    for ds in DATASET_NAMES:
        conv = frames.get((ds, "T_convergence"))
        if conv is None:
            continue
        sens = _rows(frames.get((ds, "EC_convergence_sensitivity")))
        order = ladder_order(ds)
        for row in sorted(_rows(conv), key=lambda r: order.get(r["network"], 99)):
            mine = [str(s["tau_star"]) for s in sens if s.get("network") == row["network"]]
            reached = sorted(int(float(s)) for s in mine if not s.startswith(">"))
            out.append({**row, "dataset": ds, "grid_min": reached[0] if reached else None, "grid_max": reached[-1] if reached else None,
                        "grid_above": sum(1 for s in mine if s.startswith(">")) if mine else None})
    return _frame(out)


def build_ec_multisplit(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The variability summary per network in ladder order, both datasets under the generic names."""
    rows = []
    for ds in DATASET_NAMES:
        frame = frames.get((ds, "T2_variability"))
        if frame is None:
            continue
        order = ladder_order(ds)
        for raw in sorted(_rows(frame), key=lambda r: order.get(r["network"], 99)):
            r = {k: coerce_value(v) for k, v in variability_view(raw, ds).items()}
            r["dataset"] = ds
            r["tau_star_cell"] = ci_cell(r.get("tau_star_med"), r.get("tau_star_min"), r.get("tau_star_max"), "tau")
            r["rho_boot"] = (f"[{format_value(r['boot_rho_qc_lo'], 'corr')}, {format_value(r['boot_rho_qc_hi'], 'corr')}]"
                             if r.get("boot_rho_qc_lo") is not None and r.get("boot_rho_qc_hi") is not None else "--")
            rows.append(r)
    return _frame(rows)


def _find(rows: list[dict[str, Any]], **where: Any) -> dict[str, Any] | None:
    for r in rows:
        ok = True
        for k, v in where.items():
            got = coerce_value(r.get(k))
            if isinstance(v, (int, float)) and isinstance(got, (int, float)):
                ok &= abs(float(got) - float(v)) < 1e-12
            else:
                ok &= str(got) == str(v)
        if ok:
            return r
    return None


# The settings of the synthetic rows the table prints; the full grids stay in the CSVs. The panel titles of the
# spec are written from these, so a title and the rows under it cannot disagree.
S1_SHARE = 0.3
S2_SETTING = {"size_dispersion": 2.0, "ref_share_spread": 0.01}


def build_ec_synthetic(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """A long summary of S1--S3 and the S3b placement sensitivity run: per experiment, rows with the mean and its 95% band over replicates."""
    s1 = _rows(frames.get(("synthetic", "EC_synthetic_S1_summary")))
    s2 = _rows(frames.get(("synthetic", "EC_synthetic_S2_summary")))
    s3 = _rows(frames.get(("synthetic", "EC_synthetic_S3_summary")))
    s3b = _rows(frames.get(("synthetic", "EC_synthetic_S3b_summary")))
    if not (s1 or s2 or s3):
        return None
    out: list[dict[str, Any]] = []

    def add(exp: str, label: str, row: dict[str, Any] | None, metric: str, rule: str, *, config: str | None = None,
            m: float | None = None, kappa: float | None = None) -> None:
        if row is None:
            return
        out.append({"experiment": exp, "group_label": SYNTHETIC_LABELS.get(config, config) if config else None, "label": label,
                    "m": None if m is None else f"{m:g}", "kappa": None if kappa is None else f"{kappa:g}",
                    "mean": format_cell(coerce_value(row.get(f"{metric}_mean")), rule), "lo": format_cell(coerce_value(row.get(f"{metric}_lo")), rule),
                    "hi": format_cell(coerce_value(row.get(f"{metric}_hi")), rule), "n": format_cell(coerce_value(row.get("n_replicates")), "count")})

    a = _find(s1, mode="a_nonreference_test", dangling_share=S1_SHARE)
    add("S1", "(a) QPPR scores: max change", a, "max_change_qppr", "small")
    add("S1", "(a) classical scores: max change", a, "max_change_classical", "small")
    add("S1", r"(b) QPPR: $\rho$ with the share-0 order", _find(s1, mode="b_ten_percent_references", dangling_share=S1_SHARE),
        "rho_order_qppr", "corr")
    for levels in sorted({int(coerce_value(r["levels"])) for r in s2}):
        add("S2", f"$C = {levels}$", _find(s2, levels=levels, **S2_SETTING), "one_minus_rho", "corr")
    for cfg in sorted({str(r["config"]) for r in s3}):
        for quantity, m, kappa, metric, rule in ((r"$\rho(Q, C)$", 0.0, 1.0, "rho_qc_all", "corr"), (r"$\rho(Q, C)$", 0.3, 1.0, "rho_qc_all", "corr"),
                                                 ("partial share, QPPR top 5%", 0.3, 1.0, "partial_share_top_q", "corr"),
                                                 (r"$\Delta$PR-AUC", 0.2, 0.5, "d_prauc", "diff"), (r"$\Delta$PR-AUC", 0.2, 2.0, "d_prauc", "diff")):
            add("S3", quantity, _find(s3, config=cfg, partial_share=m, kappa=kappa), metric, rule, config=cfg, m=m, kappa=kappa)
    for cfg in sorted({str(r["config"]) for r in s3b}):
        for m in (0.0, 0.3):
            row = _find(s3b, config=cfg, partial_share=m, kappa=1.0)
            add("S3b", "per category", row, "rho_qc_all", "corr", config=cfg, m=m, kappa=1.0)
            add("S3b", "per category minus per joint cell", row, "d_rho_qc_all", "diff", config=cfg, m=m, kappa=1.0)
    return _frame(out)


def build_ec_noise_grid(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """Raw fidelity ratio per dataset, instance and arm, one column per recorded step, and the first step without signal."""
    grid = stacked(frames, "EC_noise_grid")
    if grid is None:
        return None
    steps = [int(t) for t in DEFAULT_CONFIG.hardware_steps]
    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in grid.to_dicts():
        r = {k: coerce_value(v) for k, v in row.items()}
        if not is_simulated(r) or str(r.get("mitigated")).lower() == "true" or r.get("arm") in (None, "exact"):
            continue
        key = (str(r["dataset"]), int(r["k_classes"]), str(r["arm"]))
        cell = cells.setdefault(key, {"dataset": key[0], "k_classes": key[1], "R": r.get("R"), "m": r.get("m"), "route": r.get("route"),
                                      "arm": key[2], "_raw": []})
        cell[f"fr_tau_{int(r['tau'])}"] = r.get("fidelity_ratio")
        cell["_raw"].append(r)
    arm_order = {arm: i for i, arm in enumerate(DEFAULT_CONFIG.hardware_noise_arms)}
    rows = sorted(cells.values(), key=lambda c: (DATASET_NAMES.index(c["dataset"]), c["k_classes"], arm_order.get(c["arm"], len(arm_order))))
    for row in rows:
        for t in steps:
            row.setdefault(f"fr_tau_{t}", None)
        # The one rule of noise_grading: the first tau >= hardware_min_tau at which the raw cell carries no signal.
        first = first_tau_without_signal(row.pop("_raw"), min_tau=int(DEFAULT_CONFIG.hardware_min_tau))
        row["first_tau_no_signal"] = None if first == NOT_EVALUATED else str(first)
    return _frame(rows)


def build_ec_statevector(frames: dict[tuple[str, str], pl.DataFrame | None]) -> pl.DataFrame | None:
    """The statevector rung transposed: the quantities as rows, one column per dataset."""
    by_dataset = {ds: (_rows(frames.get((ds, "EC_statevector"))) or [None])[0] for ds in DATASET_NAMES}
    if all(row is None for row in by_dataset.values()):
        return None
    quantities = (("status", "Status", "text"), ("R", "$R$", "count"), ("m", "$m$", "int"), ("logical_qubits", "Logical qubits", "int"),
                  ("T", "Steps $T$", "count"), ("wall_seconds", "Wall time (s)", "seconds"), ("max_abs_error", "Max abs. error", "small"),
                  ("reason", "Note", "text"))
    rows = []
    for key, label, rule in quantities:
        rows.append({"quantity": label, **{ds: "--" if row is None else format_cell(coerce_value(row.get(key)), rule) for ds, row in by_dataset.items()}})
    return _frame(rows)
