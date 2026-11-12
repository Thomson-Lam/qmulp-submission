"""Every number the manuscript prints: the key list, and its resolution into ``\\qppr@<key>`` macros.

The key list
------------
The key grammar is the manuscript's contract: ``<dataset>.<network>.<quantity>``
with ``dataset`` in ``{hmeq, fmac, synth, all}`` and ``network`` in
``{job, reason, postal, seller, 2l, -}`` (``-`` for a quantity that belongs to
no network: a dataset-level count, a synthetic headline, a reproducibility
fact). No alias of the grammar is emitted. Each :class:`KeySpec` names its
source -- a result CSV ``result:<scope>/<name>``, a stage JSON
``stage:<dir>/<file>.json`` or the stage manifests -- its column and its row
filter. A dataset's key may only read a row of that dataset. A key whose
source is ``keys`` is an aggregate of keys that already exist (``parts``): the
largest or smallest of them, or their one common value, which is an error when
they disagree.

Resolution
----------
:func:`resolve` reads one key from its source -- a result CSV, a stage JSON or
the stage manifests -- and prints it with the rule of its kind through
``evaluation.formatting``. A key whose source is missing is reported as
unresolved and **never** given a value: the manuscript prints it red. A
dataset's key may only read a row of that dataset: a result file under one
dataset's scope that holds rows of another is an error, not a value. An
aggregate of keys (source ``keys``) is resolved from the keys it names: a
part that is unresolved leaves the aggregate unresolved, and parts that
must agree (``same``) and do not are an error.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from datasets import DATASET_NAMES, get_dataset
from evaluation.formatting import coerce_value, format_value
from evaluation.provenance import BOOT_PREFIX, RESAMPLE_PREFIX


# ---------------------------------------------------------------------------
# Kinds, grammar and the key specification
# ---------------------------------------------------------------------------


KINDS = ("corr", "diff", "count", "real", "small", "tau", "text", "int", "route", "seconds", "mb")
# The unit of a kind that has one; ``number_map.csv`` (the key list) carries it in its ``unit`` column. A ``seconds`` key is
# wall-clock seconds, an ``mb`` key is peak traced memory in MiB (2^20 bytes), as the runtime benchmark records it.
KIND_UNITS = {"seconds": "s", "mb": "MiB"}
# Kinds whose source value is a string kept as written (``route`` prints as the discriminant's symbol).
TEXT_KINDS = ("text", "route")
NETWORK_SHORT = {"NET_JOB": "job", "NET_REASON": "reason", "NET_2L": "2l",
                 "NET_FMAC_POSTAL": "postal", "NET_FMAC_SELLER": "seller", "NET_FMAC_2L": "2l"}
DATASET_SHORT = {"hmeq": "hmeq", "fmac": "fmac", "synthetic": "synth", "all": "all"}
NO_NETWORK = "-"
KEY_GRAMMAR = re.compile(r"^(hmeq|fmac|synth|all)\.(job|reason|postal|seller|2l|-)\.[A-Za-z0-9_]+$")
MANIFESTS = "manifests"
FROM_KEYS = "keys"  # the source of an aggregate of other keys
COMBINES = ("min", "max", "same")
# The synthetic configurations as the manuscript names them.
SYNTHETIC_CONFIG_SHORT = {"hmeq_like": "hmeq", "fmac_like": "fmac"}


@dataclass(frozen=True)
class KeySpec:
    """Where one manuscript number comes from and how it is printed."""

    key: str
    source: str          # ``result:<scope>/<name>`` | ``stage:<dir>/<file>.json`` | ``manifests`` | ``keys``
    column: str          # CSV column, or a dotted path into the JSON, or a manifest field; empty for ``keys``
    kind: str
    row_filter: dict[str, Any] | None = None
    aggregate: str | None = None   # ``min`` / ``max`` over the filtered rows; ``argmin`` / ``argmax`` of ``column`` by ``by``
    by: str | None = None
    dataset: str | None = None     # the dataset this key belongs to; a source row of another dataset is an error
    parts: tuple[str, ...] = ()    # source ``keys``: the existing keys this one aggregates
    combine: str | None = None     # source ``keys``: ``min`` / ``max`` of the parts, or ``same`` (their one common value)

    def __post_init__(self) -> None:
        if not KEY_GRAMMAR.match(self.key):
            raise ValueError(f"key {self.key!r} is outside the manuscript grammar <dataset>.<network>.<quantity>")
        if self.kind not in KINDS:
            raise ValueError(f"key {self.key!r}: unknown kind {self.kind!r}")
        if self.aggregate in ("argmin", "argmax") and not self.by:
            raise ValueError(f"key {self.key!r}: an arg-aggregate needs the column it ranks on")
        if (self.source == FROM_KEYS) != bool(self.parts) or (self.source == FROM_KEYS) != (self.combine in COMBINES):
            raise ValueError(f"key {self.key!r}: an aggregate of keys needs parts and one of {COMBINES}, and only it has them")


def key_of(dataset: str, network: str | None, quantity: str) -> str:
    """``<dataset>.<network>.<quantity>`` in the manuscript grammar."""
    net = NO_NETWORK if network is None else NETWORK_SHORT[network]
    return f"{DATASET_SHORT[dataset]}.{net}.{quantity}"

# ---------------------------------------------------------------------------
# The key list
# ---------------------------------------------------------------------------


def _result(ds: str, name: str) -> str:
    return f"result:{ds}/{name}"


def _data_specs(ds: str) -> list[KeySpec]:
    """Dataset-level counts from ``T1_data``, filtered on the dataset **and** the quantity.

    ``split_seed`` exists only for a stratified split: a temporal split (Freddie Mac's calendar
    cohorts) draws nothing and has no seed, so that key is not emitted for it rather than left
    to print red forever.
    """
    kinds = {"n_raw": "count", "n_dropped": "count", "n_borrowers": "count", "n_dev": "count", "n_test": "count",
             "n_ref": "count", "n_test_pos": "count", "prev_dev": "corr", "prev_test": "corr", "levels_l1": "int",
             "levels_l2": "int", "n_missing_l1": "count", "n_missing_l2": "count", "n_partial_test": "count", "split_seed": "int"}
    if get_dataset(ds).split != "stratified":
        kinds.pop("split_seed")
    return [KeySpec(key_of(ds, None, q), _result(ds, "T1_data"), "value", kind, {"dataset": ds, "key": q}, dataset=ds)
            for q, kind in kinds.items()]


def _network_specs(ds: str, net: str) -> list[KeySpec]:
    where = {"dataset": ds, "network": net}
    conv = _result(ds, "T_convergence")
    specs = []
    for q, col, kind in (("N_full", "N_full", "count"), ("n_slots", "n_slots", "count"), ("n_classes", "n_classes", "count"),
                         ("R", "R", "count"), ("reduction_factor", "reduction_factor", "real"), ("m", "m", "int"), ("qubits", "qubits", "int"),
                         ("angles", "angles", "count"), ("gbar_density", "gbar_density", "corr"),
                         ("lump_resid_max", "lumpability_residual", "small"), ("compat_resid_max", "compatibility_residual", "small"),
                         ("tau_star", "tau_star_printed", "tau"), ("tau_star_l1", "tau_star_l1", "tau"), ("tau_star_rank", "tau_star_rank", "tau"),
                         ("tau_max", "tau_max", "count"), ("conv_tol_l1", "conv_tol_l1", "real"), ("conv_min_spearman", "conv_min_spearman", "corr"),
                         ("l1_at_100", "l1_at_100", "small"), ("rho_100_vs_limit", "rho_100_vs_limit", "corr"), ("r_active", "r_active", "count"),
                         ("delta_chord", "delta_chord", "small"), ("c_exact", "c_exact", "small"),
                         ("rho_classical_tau_star", "rho_classical_tau_star", "corr")):
        specs.append(KeySpec(key_of(ds, net, q), conv, col, kind, where, dataset=ds))
    for q, kind in (("rho_qc", "corr"), ("kendall_qc", "corr"), ("ov_qc", "corr"), ("selfov_q", "corr"), ("selfov_c", "corr"),
                    ("levels_q", "count"), ("levels_c", "count"), ("rho_tau_star_vs_limit", "corr")):
        specs.append(KeySpec(key_of(ds, net, q), _result(ds, "T2_agreement"), q, kind, where, dataset=ds))
    variability = _result(ds, "T2_variability")
    # HMEQ re-splits: unprefixed columns, n_splits of them. Freddie Mac has a calendar split: its range comes from the
    # reference resamples (resample_*, n_resamples of them) and its interval is the bootstrap over test loans (boot_*),
    # never the percentiles of the resamples.
    resampled = get_dataset(ds).split != "stratified"
    prefix = RESAMPLE_PREFIX if resampled else ""
    for q in ("rho_qc_med", "rho_qc_min", "rho_qc_max"):
        specs.append(KeySpec(key_of(ds, net, q), variability, f"{prefix}{q}", "corr", where, dataset=ds))
    specs.append(KeySpec(key_of(ds, net, "n_splits"), variability, "n_resamples" if resampled else "n_splits", "count", where, dataset=ds))
    interval_prefix = BOOT_PREFIX if resampled else ""
    for q in ("rho_qc_lo", "rho_qc_hi"):
        specs.append(KeySpec(key_of(ds, net, q), variability, f"{interval_prefix}{q}", "corr", where, dataset=ds))
    for q, kind in (("prauc_noskill", "corr"), ("prauc_c", "corr"), ("prauc_q", "corr"), ("d_prauc", "diff"), ("d_prauc_lo", "diff"),
                    ("d_prauc_hi", "diff"), ("roc_c", "corr"), ("roc_q", "corr"), ("d_roc", "diff"), ("d_roc_lo", "diff"), ("d_roc_hi", "diff"),
                    ("p100_c", "corr"), ("p100_q", "corr"), ("d_p100", "diff"), ("d_p100_lo", "diff"), ("d_p100_hi", "diff"),
                    ("prauc_cellrate", "corr"), ("best_comp_name", "text"), ("best_comp_prauc", "corr")):
        specs.append(KeySpec(key_of(ds, net, q), _result(ds, "T2_decision"), q, kind, where, dataset=ds))
    return specs


def _primary_specs(ds: str, net: str, single: tuple[str, ...], stage_mech: str) -> list[KeySpec]:
    mech = f"stage:{stage_mech}/mechanism_summary.json"
    where = {"dataset": ds, "network": net}
    specs = [
        KeySpec(key_of(ds, net, "rho_src_dst_min"), _result(ds, "EC_registers"), "rho_src_dst", "corr", {"dataset": ds}, aggregate="min", dataset=ds),
        KeySpec(key_of(ds, net, "rho_qc_eta_min"), _result(ds, "T_eta"), "rho_qc", "corr", where, aggregate="min", dataset=ds),
        KeySpec(key_of(ds, net, "rho_qc_eta_max"), _result(ds, "T_eta"), "rho_qc", "corr", where, aggregate="max", dataset=ds),
        KeySpec(key_of(ds, net, "eta_at_min"), _result(ds, "T_eta"), "eta", "real", where, aggregate="argmin", by="rho_qc", dataset=ds),
        KeySpec(key_of(ds, net, "rho_q_owncopy"), _result(ds, "T3_variants"), "rho_vs_primary_q", "corr", {**where, "variant": "own_copy_restart"}, dataset=ds),
        KeySpec(key_of(ds, net, "ov_q_owncopy"), _result(ds, "T3_variants"), "ov_vs_primary_q", "corr", {**where, "variant": "own_copy_restart"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_q_rankone"), _result(ds, "T3_variants"), "rho_vs_primary_q", "corr", {**where, "variant": "rank_one_restart"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_q_flat"), _result(ds, "T3_variants"), "rho_vs_primary_q", "corr", {**where, "variant": "flattened"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_q_eta0"), _result(ds, "T3_variants"), "rho_vs_primary_q", "corr", {**where, "variant": "eta_0"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_q_nopartial"), _result(ds, "T3_variants"), "rho_vs_primary_q", "corr", {**where, "variant": "no_partial_copies"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_qc_nopartial"), _result(ds, "T3_variants"), "rho_vs_own_classical", "corr", {**where, "variant": "no_partial_copies"}, dataset=ds),
        # The discriminant the router selected for each control operator, for the captions that name one: a macro, never typed.
        *[KeySpec(key_of(ds, net, f"route_{short}"), _result(ds, "T3_variants"), "route", "route", {**where, "variant": variant}, dataset=ds)
          for short, variant in (("primary", "primary"), ("owncopy", "own_copy_restart"), ("rankone", "rank_one_restart"), ("flat", "flattened"),
                                 ("eta0", "eta_0"), ("nopartial", "no_partial_copies"))],
        KeySpec(key_of(ds, net, "route_labelfree"), _result(ds, "T4_partial"), "route_labelfree", "route", {**where, "arm": "qppr"}, dataset=ds),
        KeySpec(key_of(ds, net, "rho_best_classical_grid"), mech, "grid_summary.rho_best_classical_grid", "corr", dataset=ds),
        KeySpec(key_of(ds, net, "best_alpha"), mech, "grid_summary.best_alpha", "real", dataset=ds),
        KeySpec(key_of(ds, net, "best_eta"), mech, "grid_summary.best_eta", "real", dataset=ds),
        KeySpec(key_of(ds, net, "rho_classical_grid_min"), mech, "grid_summary.rho_classical_grid_min", "corr", dataset=ds),
    ]
    partial = _result(ds, "T4_partial")
    for q, col, arm, kind in (("partial_share_top100_q", "partial_share_topk", "qppr", "corr"), ("partial_share_top100_c", "partial_share_topk", "classical", "corr"),
                              ("partial_meanrank_q", "partial_meanrank", "qppr", "real"), ("partial_meanrank_c", "partial_meanrank", "classical", "real"),
                              ("partial_defrate", "partial_defrate", "qppr", "corr"), ("other_defrate", "other_defrate", "qppr", "corr"),
                              ("rho_qc_excl_partial", "rho_qc_excl_partial", "qppr", "corr"), ("jacc_labelfree_q", "jacc_labelfree", "qppr", "corr"),
                              ("jacc_labelfree_c", "jacc_labelfree", "classical", "corr"), ("jacc_placebo_q", "jacc_placebo", "qppr", "corr"),
                              ("jacc_placebo_c", "jacc_placebo", "classical", "corr")):
        specs.append(KeySpec(key_of(ds, net, q), partial, col, kind, {**where, "arm": arm}, dataset=ds))
    placebo = _result(ds, "EC_placebo")
    for q, col, arm in (("n_perm", "n_perm", "qppr"), ("pval_prauc_q", "pval_pr_auc", "qppr"), ("pval_prauc_c", "pval_pr_auc", "classical"),
                        ("pval_p100_q", "pval_precision_at_100", "qppr"), ("pval_p100_c", "pval_precision_at_100", "classical")):
        specs.append(KeySpec(key_of(ds, net, q), placebo, col, "count" if q == "n_perm" else "corr", {**where, "arm": arm}, dataset=ds))
    importance = {"dataset": ds, "model": "xgboost", "method": "permutation"}
    specs += [
        KeySpec(key_of(ds, None, "dprauc_upper_max"), _result(ds, "downstream_bound"), "dprauc_upper_max", "diff", {"dataset": ds}, dataset=ds),
        KeySpec(key_of(ds, None, "layerdummy_importance_share"), _result(ds, "EC_feature_importance"), "layerdummy_importance_share", "corr", importance, dataset=ds),
        KeySpec(key_of(ds, None, "layerdummy_importance_rank"), _result(ds, "EC_feature_importance"), "layerdummy_importance_rank", "count", importance, dataset=ds),
        # "rank r of n_features": the denominator is covariates plus layer dummies, from the same row as the rank.
        KeySpec(key_of(ds, None, "layerdummy_importance_of"), _result(ds, "EC_feature_importance"), "n_features", "count", importance, dataset=ds),
        KeySpec(key_of(ds, None, "n_covariates"), _result(ds, "EC_feature_importance"), "n_covariates", "count", importance, dataset=ds),
    ]
    for s in single:
        # The list of equal-share pairs (raw postal codes and lender names) stays in the mechanism summary; the manuscript
        # needs how many there are. The re-split count belongs to the single-layer network, read from its own row.
        specs += [KeySpec(key_of(ds, s, "prop3_max_relerr"), mech, f"single_layer.{s}.prop3_max_relerr", "small", dataset=ds),
                  KeySpec(key_of(ds, s, "n_equal_share_pairs"), mech, f"single_layer.{s}.n_equal_share_pairs", "count", dataset=ds),
                  KeySpec(key_of(ds, s, "levels_q_minus_c"), mech, f"single_layer.{s}.levels_q_minus_c", "int", dataset=ds)]
        column = f"{RESAMPLE_PREFIX if get_dataset(ds).split != 'stratified' else ''}n_single_identical_{s}"
        specs.append(KeySpec(key_of(ds, s, "n_single_identical"), _result(ds, "T2_variability"), column, "count",
                             {"dataset": ds, "network": s}, dataset=ds))
    return specs


def _hardware_specs(ds: str) -> list[KeySpec]:
    header = _result(ds, "T6_hardware_header")
    specs = [KeySpec(key_of(ds, None, q), header, q, kind, {"dataset": ds}, dataset=ds)
             for q, kind in (("hw_k", "count"), ("hw_borrowers", "count"), ("hw_R", "count"), ("hw_m", "int"), ("hw_tau", "int"),
                             ("hw_depth", "count"), ("hw_2q", "count"), ("hw_survival", "small"), ("hw_qpu_s", "real"),
                             ("hw_tv_uniform", "corr"), ("hw_backend", "text"), ("hw_date", "text"),
                             # the first instance able to rank test borrowers (two masked classes), and the two noise entry points
                             ("hw_rankable_k", "count"), ("hw_rankable_m", "int"), ("hw_rankable_borrowers", "count"),
                             ("hw_rankable_fr", "corr"), ("noise_k_band", "text"), ("noise_tau_band", "text"))]
    # The simulated arms only; nothing in the repository describes an execution on hardware.
    for arm in ("noiseless", "aernoise", "fake"):
        for q in ("rho", "tv", "fr"):
            specs.append(KeySpec(key_of(ds, None, f"hw_{arm}_{q}"), _result(ds, "T6_hardware"), q, "corr", {"dataset": ds, "arm_key": arm}, dataset=ds))
    return specs


def _runtime_specs(ds: str) -> list[KeySpec]:
    specs = []
    # Seconds (wall-clock, median over the repeats) print to three significant figures and memory (peak traced, MiB) as an
    # integer; both units are stated in the column definitions of T5_runtime and in the key list of the README.
    for q, op, col, kind in (("t_step_full", "full_walk_per_step", "seconds", "seconds"), ("t_step_lumped", "lumped_walk_per_step", "seconds", "seconds"),
                             ("t_limit_full", "limit_full_eigendecomposition", "seconds", "seconds"),
                             ("t_limit_lumped", "limit_lumped_eigendecomposition", "seconds", "seconds"),
                             ("t_class_full", "full_operator_build", "seconds", "seconds"),
                             ("t_class_lumped", "class_construction_lumped_operator_residuals", "seconds", "seconds"),
                             ("mem_full", "limit_full_eigendecomposition", "peak_mb", "mb"),
                             ("mem_lumped", "limit_lumped_eigendecomposition", "peak_mb", "mb")):
        specs.append(KeySpec(key_of(ds, None, q), _result(ds, "T5_runtime"), col, kind, {"dataset": ds, "operation": op, "threads": 1}, dataset=ds))
    return specs


def _synthetic_specs() -> list[KeySpec]:
    source = "stage:08_synthetic/EC_synthetic_keys.json"
    specs = [KeySpec(key_of("synthetic", None, q), source, q, "small") for q in ("s1_max_change", "s2_sep_max")]
    for cfg, short in SYNTHETIC_CONFIG_SHORT.items():
        for q, kind in (("s3_rho_m0", "corr"), ("s3_rho_m30", "corr"), ("s3_share_m30", "corr"), ("s3_dprauc_k05_m20", "diff"), ("s3_dprauc_k2_m20", "diff")):
            specs.append(KeySpec(key_of("synthetic", None, f"{q}_{short}"), source, f"{cfg}.{q}", kind))
    return specs


def _reproducibility_specs() -> list[KeySpec]:
    specs = [KeySpec(key_of("all", None, "cpu"), "stage:10_runtime/machine.json", "cpu", "text")]
    for q, pkg in (("python", "python"), ("numpy", "numpy"), ("scipy", "scipy"), ("qiskit", "qiskit"), ("aer", "qiskit-aer"),
                   ("runtime_version", "qiskit-ibm-runtime")):
        specs.append(KeySpec(key_of("all", None, q), MANIFESTS, f"versions.{pkg}", "text"))
    specs.append(KeySpec(key_of("all", None, "commit"), MANIFESTS, "commit", "text"))
    return specs


def _aggregate_specs() -> list[KeySpec]:
    """Keys that aggregate keys which already exist: a worst case over networks, a fact shared by both datasets, a dataset-level alias."""
    networks = [(ds, net) for ds in DATASET_NAMES for net in get_dataset(ds).ladder]
    primaries = [(ds, get_dataset(ds).primary) for ds in DATASET_NAMES]
    singles = [(ds, net) for ds, net in networks if len(get_dataset(ds).networks[net]) == 1]

    def of(pairs: list[tuple[str, str]], quantity: str) -> tuple[str, ...]:
        return tuple(key_of(ds, net, quantity) for ds, net in pairs)

    specs = [
        # worst cases over the operators the paper reports
        KeySpec(key_of("all", None, "compat_resid_max"), FROM_KEYS, "", "small", parts=of(networks, "compat_resid_max"), combine="max"),
        KeySpec(key_of("all", None, "lump_resid_max"), FROM_KEYS, "", "small", parts=of(networks, "lump_resid_max"), combine="max"),
        KeySpec(key_of("all", None, "prop3_max_relerr"), FROM_KEYS, "", "small", parts=of(singles, "prop3_max_relerr"), combine="max"),
        KeySpec(key_of("all", None, "rho_src_dst_min"), FROM_KEYS, "", "corr", parts=of(primaries, "rho_src_dst_min"), combine="min"),
        # the calibration snapshot every simulated arm of both datasets reads: one fact, an error if the datasets disagree
        KeySpec(key_of("all", None, "hw_backend"), FROM_KEYS, "", "text", parts=tuple(key_of(ds, None, "hw_backend") for ds in DATASET_NAMES), combine="same"),
        KeySpec(key_of("all", None, "hw_date"), FROM_KEYS, "", "text", parts=tuple(key_of(ds, None, "hw_date") for ds in DATASET_NAMES), combine="same"),
    ]
    # the number of re-splits of HMEQ is one number, whichever network it is read from
    stratified = [(ds, net) for ds, net in networks if get_dataset(ds).split == "stratified"]
    for ds in sorted({ds for ds, _net in stratified}):
        specs.append(KeySpec(key_of(ds, None, "n_splits"), FROM_KEYS, "", "count",
                             parts=of([(d, n) for d, n in stratified if d == ds], "n_splits"), combine="same", dataset=ds))
    # the category slots of a dataset are those of its two-layer network
    for ds, net in primaries:
        specs.append(KeySpec(key_of(ds, None, "n_slots"), FROM_KEYS, "", "count", parts=of([(ds, net)], "n_slots"), combine="same", dataset=ds))
    return specs


def key_specs() -> list[KeySpec]:
    """Every key of the output contract, in the manuscript grammar."""
    specs: list[KeySpec] = []
    for ds in DATASET_NAMES:
        dataset = get_dataset(ds)
        specs += _data_specs(ds)
        for net in dataset.ladder:
            specs += _network_specs(ds, net)
        single = tuple(n for n, layers in dataset.networks.items() if len(layers) == 1)
        specs += _primary_specs(ds, dataset.primary, single, dataset.stage("mechanism"))
        specs += _hardware_specs(ds)
        specs += _runtime_specs(ds)
    specs += _synthetic_specs()
    specs += _reproducibility_specs()
    specs += _aggregate_specs()
    keys = [s.key for s in specs]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise ValueError(f"duplicate keys in the number map: {duplicates}")
    plain = {s.key for s in specs if s.source != FROM_KEYS}
    for spec in specs:
        absent = [part for part in spec.parts if part not in plain]
        if absent:
            raise ValueError(f"key {spec.key!r} aggregates keys that are not plain keys of the list: {absent}")
    return specs


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def fmt(value: Any, kind: str) -> str | None:
    """The printed form of a value under the kind's rule; ``None`` stays unresolved."""
    return format_value(value, kind)


def _match(row: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    for key, expected in where.items():
        got = row.get(key)
        try:
            if abs(float(got) - float(expected)) > 1e-12:
                return False
        except (TypeError, ValueError):
            if str(got) != str(expected):
                return False
    return True


def _text(value: Any) -> str | None:
    """A text key keeps its string as written (``none`` is a legitimate value); only a blank is unresolved."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _read_json_path(data: Any, column: str) -> tuple[bool, Any]:
    if isinstance(data, dict) and column in data:
        return True, data[column]  # a flat key that happens to contain dots
    node = data
    for part in column.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False, None
    return True, node


def read_manifests(pipeline_dir: Path) -> dict[str, Any]:
    """Every stage manifest under the pipeline directory, checked for one commit and one source identity.

    Returns ``{"manifests": [...], "commit": str | None, "dirty": bool, "source_sha256": str | None,
    "disagreement": str | None}``.
    """
    paths = sorted(Path(pipeline_dir).glob("*/manifest.json"))
    manifests = [json.loads(p.read_text()) for p in paths]
    commits = {str((m.get("git") or {}).get("commit")) for m in manifests}
    sources = {str(m.get("source_sha256")) for m in manifests if m.get("source_sha256") is not None}
    disagreement = None
    if len(commits) > 1:
        disagreement = f"manifests disagree on git.commit: {sorted(commits)}"
    elif len(sources) > 1:
        disagreement = f"manifests disagree on source_sha256: {sorted(s[:12] for s in sources)}"
    dirty = any(bool((m.get("git") or {}).get("dirty")) for m in manifests)
    return {"manifests": manifests, "paths": paths, "commit": next(iter(commits)) if len(commits) == 1 else None,
            "dirty": dirty, "source_sha256": next(iter(sources)) if len(sources) == 1 else None, "disagreement": disagreement}


def _resolve_manifests(spec: KeySpec, pipeline_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    info = read_manifests(pipeline_dir)
    if not info["manifests"]:
        record["status"] = "no manifest"
        return record
    if info["disagreement"]:
        raise AssertionError(f"{spec.key}: {info['disagreement']}; the results do not come from one run")
    if spec.column == "commit":
        commit = info["commit"]
        if commit in (None, "None"):
            record["status"] = "no commit recorded"
            return record
        if info["dirty"]:
            commit += f"+uncommitted:{info['source_sha256'][:12]}" if info["source_sha256"] else "+uncommitted"
        value: Any = commit
    else:
        value = None
        for manifest in info["manifests"]:
            found, node = _read_json_path(manifest, spec.column)
            if found and node is not None:
                value = node
                break
        if value is None:
            record["status"] = f"missing path {spec.column}"
            return record
    record["value"] = _text(value) if spec.kind in TEXT_KINDS else coerce_value(value)
    record["printed"] = fmt(record["value"], spec.kind)
    record["status"] = "ok" if record["printed"] is not None else "unresolved value"
    return record


@lru_cache(maxsize=1)
def _plain_specs() -> dict[str, KeySpec]:
    """The keys that are read from a source, by name: what an aggregate of keys is made of."""
    return {s.key: s for s in key_specs() if s.source != FROM_KEYS}


def _resolve_from_keys(spec: KeySpec, results_dir: Path, pipeline_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    """An aggregate of existing keys: ``max`` / ``min`` of the parts, or their one common value (``same``)."""
    record["column"] = spec.combine
    record["row_filter"] = json.dumps(list(spec.parts))
    parts = [resolve(_plain_specs()[key], results_dir, pipeline_dir) for key in spec.parts]
    unresolved = [p["key"] for p in parts if p["value"] is None]
    if unresolved:
        record["status"] = f"part {unresolved[0]} is unresolved"
        return record
    values = [p["value"] for p in parts]
    if spec.combine == "same":
        if any(v != values[0] for v in values):
            raise AssertionError(f"{spec.key}: its parts disagree: " + ", ".join(f"{p['key']} = {p['value']!r}" for p in parts))
        value = values[0]
    else:
        if not all(isinstance(v, (int, float)) for v in values):
            raise AssertionError(f"{spec.key}: {spec.combine} needs numbers, got " + ", ".join(f"{p['key']} = {p['value']!r}" for p in parts))
        value = min(values) if spec.combine == "min" else max(values)
    record["value"] = value
    record["printed"] = fmt(value, spec.kind)
    record["status"] = "ok" if record["printed"] is not None else "unresolved value"
    return record


def resolve(spec: KeySpec, results_dir: Path, pipeline_dir: Path) -> dict[str, Any]:
    """Read one key; the result carries ``value`` (``None`` when unresolved) and why."""
    record = {"key": spec.key, "source": spec.source, "column": spec.column,
              "row_filter": json.dumps(spec.row_filter or {}, sort_keys=True), "value": None, "printed": None, "status": ""}
    if spec.source == FROM_KEYS:
        return _resolve_from_keys(spec, Path(results_dir), Path(pipeline_dir), record)
    if spec.source == MANIFESTS:
        return _resolve_manifests(spec, Path(pipeline_dir), record)
    if spec.source.startswith("stage:"):
        path = Path(pipeline_dir) / spec.source[len("stage:"):]
        if not path.exists():
            record["status"] = "missing file"
            return record
        found, node = _read_json_path(json.loads(path.read_text()), spec.column)
        if not found:
            record["status"] = f"missing path {spec.column}"
            return record
        value = _text(node) if spec.kind in TEXT_KINDS else coerce_value(node)
    elif spec.source.startswith("result:"):
        scope, _, name = spec.source[len("result:"):].partition("/")
        path = Path(results_dir) / scope / f"{name}.csv"
        if not path.exists():
            record["status"] = "missing file"
            return record
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            all_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])
        if spec.dataset is not None and "dataset" in fieldnames and all_rows:
            if not any(str(r.get("dataset")) == spec.dataset for r in all_rows):
                raise AssertionError(
                    f"{spec.key}: {path} holds no row of dataset {spec.dataset!r} "
                    f"(found {sorted({str(r.get('dataset')) for r in all_rows})}); a result file under one scope "
                    "may not carry another dataset's rows")
        rows = [r for r in all_rows if _match(r, spec.row_filter)]
        if not rows:
            record["status"] = "no row matches the filter"
            return record
        if spec.column not in rows[0]:
            record["status"] = f"missing column {spec.column}"
            return record
        values = [_text(r[spec.column]) if spec.kind in TEXT_KINDS else coerce_value(r[spec.column]) for r in rows]
        if spec.aggregate in ("min", "max"):
            numeric = [v for v in values if isinstance(v, (int, float))]
            value = (min(numeric) if spec.aggregate == "min" else max(numeric)) if numeric else None
        elif spec.aggregate in ("argmin", "argmax"):
            if spec.by not in rows[0]:
                record["status"] = f"missing column {spec.by}"
                return record
            ranked = [(coerce_value(r[spec.by]), v) for r, v in zip(rows, values) if isinstance(coerce_value(r[spec.by]), (int, float))]
            if not ranked:
                value = None
            else:
                pick = min(ranked, key=lambda t: t[0]) if spec.aggregate == "argmin" else max(ranked, key=lambda t: t[0])
                value = pick[1]
        else:
            if len(rows) > 1:
                record["status"] = f"{len(rows)} rows match; the filter must select one"
                return record
            value = values[0]
    else:
        raise ValueError(f"{spec.key}: unknown source {spec.source!r}")
    record["value"] = value
    record["printed"] = fmt(value, spec.kind)
    record["status"] = "ok" if record["printed"] is not None else "unresolved value"
    return record


def write_number_map(results_dir: Path, pipeline_dir: Path, manuscript_dir: Path) -> dict[str, Any]:
    """``number_map.csv`` and ``numbers.tex`` under the manuscript directory."""
    specs = key_specs()
    records = [resolve(spec, Path(results_dir), Path(pipeline_dir)) for spec in specs]
    for spec, record in zip(specs, records):
        record["unit"] = KIND_UNITS.get(spec.kind, "")
    Path(manuscript_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(manuscript_dir) / "number_map.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "value", "printed", "unit", "source", "column", "row_filter", "status"])
        writer.writeheader()
        for r in records:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    lines = ["% generated by scripts/make_manuscript_tables.py; one macro per resolved key, nothing typed by hand"]
    for r in records:
        if r["printed"] is not None:
            lines.append(rf"\expandafter\def\csname qppr@{r['key']}\endcsname{{{r['printed']}}}")
    (Path(manuscript_dir) / "numbers.tex").write_text("\n".join(lines) + "\n")
    unresolved = [r for r in records if r["printed"] is None]
    return {"n_keys": len(records), "n_resolved": len(records) - len(unresolved), "unresolved": unresolved}
