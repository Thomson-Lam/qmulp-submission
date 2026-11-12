"""What every ``.tex`` table of the manuscript is made of: sources, panels, columns and formatting rules.

One :class:`TableSpec` per table the manuscript or the e-companion prints: the result CSVs it reads (scope and
name), how its rows are ordered and grouped, its :class:`Panel` s (the stacked blocks of one ``tabular``, each with
its own columns), the CSV column behind every printed column with its header, its formatting rule
(``evaluation.formatting``) and its **definition**, and, for a composite table, the function that builds its frame
from the source frames (``evaluation.table_builders``). Headers are short (symbols, ``$\\rho$``, ``$O_{100}$``); what
a header means, with its unit, is the column's ``definition``, which ``scripts/make_manuscript_tables.py`` writes to
``outputs/manuscript/table_columns.csv`` for the note under the table in the manuscript. That script is the only
consumer that writes ``.tex``; ``README.md``'s results map is generated from :func:`source_map` so the two cannot
drift.

A ``"<dataset>"`` scope expands to every dataset. Panels share the grid of columns of their table: a table is one
bare ``tabular`` of ``align`` (default: the first column left, the others right), a panel with fewer cells is
padded to the grid, and a wide cell spans several narrow columns of the other panels (``Column.span``). Long or raw
tables (convergence curves, noise-grid rows, multisplit rows, placebo draws, synthetic replicates, the alpha--eta
grid, ``T_eta``) have no spec: they stay CSV-only in the repository.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from config import DEFAULT_CONFIG
from datasets import DATASET_NAMES
from evaluation.formatting import DATASET_LABELS, ci_cell, format_cell, is_missing, spanned, tex_escape
from evaluation.table_builders import (  # noqa: F401 - stacked is re-exported for the assembler
    S1_SHARE, S2_SETTING, Builder, build_ec_circuit_resources, build_ec_convergence, build_ec_decision_full, build_ec_hardware_scores,
    build_ec_multisplit, build_ec_noise_grid, build_ec_single_layer_share, build_ec_statevector, build_ec_strata, build_ec_synthetic,
    build_t1, build_t2, build_t5, build_t6, network_grouped, stacked,
)

K = int(DEFAULT_CONFIG.primary_k)
STEPS = tuple(int(t) for t in DEFAULT_CONFIG.hardware_steps)
MIN_TAU = int(DEFAULT_CONFIG.hardware_min_tau)  # the first recorded step that is a walk (tau = 1 is the state preparation)
MAIN, EC = "main", "ec"
RAW = "raw"  # a cell already formatted by a composite builder; written verbatim


@dataclass(frozen=True)
class Column:
    """One printed column: the CSV column behind it, its header (math allowed), its rule and its definition.

    ``ci`` names the interval bounds printed after the estimate; ``span`` is the number of grid columns the cell
    covers and ``align`` the alignment it is set in (right by default when it spans or is aligned explicitly).
    """

    source: str
    header: str
    rule: str
    definition: str
    ci: tuple[str, str] | None = None
    span: int = 1
    align: str | None = None

    def cell(self, row: Mapping[str, Any]) -> str:
        value = row.get(self.source)
        if self.rule == RAW:
            text = "--" if is_missing(value) else str(value)
        elif self.ci is not None:
            text = ci_cell(value, row.get(self.ci[0]), row.get(self.ci[1]), self.rule)
        else:
            text = format_cell(value, self.rule)
        return spanned(text, self.span, self.align) if (self.span != 1 or self.align) else text

    def header_cell(self) -> str:
        text = tex_escape(self.header)
        return spanned(text, self.span, self.align) if (self.span != 1 or self.align) else text


@dataclass(frozen=True)
class Panel:
    """One stacked block of a table: an optional bold title and its columns; ``where`` keeps a subset of the rows, ``grouped=False`` omits the group rows."""

    title: str | None
    columns: tuple[Column, ...]
    where: Callable[[Mapping[str, Any]], bool] | None = None
    grouped: bool = True


@dataclass(frozen=True)
class TableSpec:
    name: str
    tier: str
    manuscript: str
    sources: tuple[tuple[str, str], ...]
    panels: tuple[Panel, ...]
    align: str | None = None
    build: Builder | None = None
    group_by: str | None = "dataset"
    sort: tuple[str, ...] = ()
    where: Callable[[Mapping[str, Any]], bool] | None = None
    # Sources every dataset must provide; the others may be missing (their columns print "--").
    optional: tuple[str, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> tuple[Column, ...]:
        return tuple(column for panel in self.panels for column in panel.columns)

    def expanded_sources(self) -> list[tuple[str, str]]:
        out = []
        for scope, name in self.sources:
            if scope == "<dataset>":
                out += [(ds, name) for ds in DATASET_NAMES]
            else:
                out.append((scope, name))
        return out


# ---------------------------------------------------------------------------
# The specs
# ---------------------------------------------------------------------------

def _ds(name: str) -> tuple[tuple[str, str], ...]:
    return (("<dataset>", name),)


def _single(*columns: Column) -> tuple[Panel, ...]:
    return (Panel(None, tuple(columns)),)


def _present(column: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda row: not is_missing(row.get(column))


def _is_not(column: str, value: str) -> Callable[[Mapping[str, Any]], bool]:
    return lambda row: row.get(column) != value


NETWORK = Column("network", "Network", "text", "Network: one layer of the dataset, or its two-layer network.")
ARM = Column("arm", "Arm", "text", "Score the row reports: the QPPR limit or the classical personalized PageRank score.")
ROUTE = "Discriminant the router selected for the lumped operator ($\\bar{D}$ or $H$)."
RHO = "Spearman correlation"
TAU_STAR = ("Convergence point of the running average: the smallest horizon from which every criterion of Stage 3 holds up to the cap "
            "$\\tau_{\\max}$; printed as $>\\tau_{\\max}$ with the cap when it is not reached.")
CI95 = "with its 95\\% bootstrap interval over the shared replicates of the test individuals"

TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        "T1_data_reduction", MAIN, "Table 1", _ds("T1_data") + _ds("T_convergence"), build=build_t1, group_by=None, align="lll",
        optional=("T_convergence",),
        panels=_single(Column("quantity", "Quantity", "text", "Quantity of the data stage, or of the reduction of the dataset's two-layer network."),
                       Column("hmeq", DATASET_LABELS["hmeq"], RAW, "Value for the HMEQ dataset."),
                       Column("fmac", DATASET_LABELS["fmac"], RAW, "Value for the Freddie Mac extract; the split row names the two first-payment cohorts."))),
    TableSpec(
        "T2_agreement", MAIN, "Table 2", _ds("T2_agreement") + _ds("T2_decision") + _ds("T2_variability"), build=build_t2, align="lrrrrrrrr",
        optional=("T2_decision", "T2_variability"),
        panels=(
            Panel("A. Agreement of QPPR (Q) with the classical score (C)", (
                NETWORK,
                Column("rho_qc", "$\\rho$", "corr", f"{RHO} between the QPPR limit and the classical score over the test individuals."),
                Column("kendall_qc", "$\\tau_b$", "corr", "Kendall $\\tau_b$ between the QPPR limit and the classical score over the test individuals."),
                Column("ov_qc", f"$O_{{{K}}}$", "corr", f"Expected overlap of the two top-{K} sets under uniform tie-breaking."),
                Column("selfov_q", f"$\\bar O^{{Q}}_{{{K}}}$", "corr",
                       f"Ceiling of the top-{K} overlap for the QPPR limit: the expected overlap of its top-{K} set with itself under uniform tie-breaking."),
                Column("selfov_c", f"$\\bar O^{{C}}_{{{K}}}$", "corr", "The same ceiling for the classical score."),
                Column("levels_q", "$L^{Q}$", "count", "Distinct score levels of the QPPR limit among the test individuals, at the tie tolerance."),
                Column("levels_c", "$L^{C}$", "count", "Distinct score levels of the classical score among the test individuals, at the tie tolerance."),
                Column("tau_star_printed", "$\\tau^*$", "tau", TAU_STAR))),
            Panel("B. PR-AUC: classical C, QPPR Q, $\\Delta$ = Q $-$ C, joint cell", (
                NETWORK,
                Column("prauc_c", "C", "corr", "PR-AUC of the classical score on the test partition."),
                Column("prauc_q", "Q", "corr", "PR-AUC of the QPPR limit on the test partition."),
                Column("d_prauc", "$\\Delta$ [95% CI]", "diff", f"Paired PR-AUC difference Q $-$ C {CI95}.",
                       ci=("d_prauc_lo", "d_prauc_hi"), span=3, align="r"),
                Column("prauc_score", "joint cell", "corr",
                       "PR-AUC of the joint-category development default rate, a score without a walk; reported on the two-layer row of each dataset.",
                       span=2, align="r"))),
            Panel("C. $\\rho$ over re-splits (HMEQ) and reference resamples (Freddie Mac)", (
                NETWORK,
                Column("rho_splits", "median [min, max]", RAW,
                       f"{RHO} between the QPPR limit and the classical score: median, minimum and maximum over the re-splits (HMEQ) "
                       "or over the reference resamples of the fixed calendar split (Freddie Mac).", span=3, align="r"),
                Column("rho_boot", "bootstrap [lo, hi]", RAW,
                       "2.5th and 97.5th percentiles of $\\rho$ over the shared bootstrap replicates of the test individuals, on the frozen test partition.",
                       span=2, align="r"))))),
    TableSpec(
        "T3_variants", MAIN, "Table 3", _ds("T3_variants"),
        panels=_single(
            Column("variant", "Variant", "text", "Operator variant the row evaluates: the primary operator and the controls of the variants table."),
            Column("alpha", "$\\alpha$", "real", "Damping $\\alpha$ of the variant."),
            Column("eta", "$\\eta$", "real", "Interlayer weight $\\eta$ of the variant."),
            Column("route", "Route", "route", ROUTE),
            Column("rho_vs_primary_q", "$\\rho_{\\mathrm{Q}}$", "corr",
                   f"{RHO} between the variant's QPPR limit and the primary QPPR limit over the test individuals (for the best-classical row, the classical score at that $(\\alpha, \\eta)$)."),
            Column("ov_vs_primary_q", f"$O_{{{K},\\mathrm{{Q}}}}$", "corr", f"Expected top-{K} overlap between the variant's QPPR limit and the primary QPPR limit."),
            Column("rho_vs_own_classical", "$\\rho_{\\mathrm{C}}$", "corr",
                   f"{RHO} between the variant's QPPR limit and the classical score on the same variant operator."))),
    TableSpec(
        "T4_partial", MAIN, "Table 4", _ds("T4_partial"), align="lrrrrrr",
        panels=(
            Panel("A. Partially observed test individuals", (
                ARM,
                Column("n_partial_test", "$n$", "count", "Test individuals with no membership edge in at least one layer (partially observed)."),
                Column("partial_defrate", "$d_{\\mathrm{p}}$", "corr", "Default rate of the partially observed test individuals."),
                Column("other_defrate", "$d_{\\mathrm{o}}$", "corr", "Default rate of the fully observed test individuals."),
                Column("partial_share_topk", f"$s_{{{K}}}$", "corr", f"Expected share of partially observed individuals in the score's top {K}."),
                Column("partial_meanrank", "$\\bar r$", "mean", "Mean rank of the partially observed individuals (1 = largest score, average ranks on ties)."),
                Column("partial_medianrank", "$\\tilde r$", "mean", "Median rank of the partially observed individuals."))),
            Panel("B. Agreement and controls", (
                ARM,
                Column("rho_qc", "$\\rho$", "corr", f"{RHO} between the two scores over all test individuals."),
                Column("rho_qc_excl_partial", "$\\rho_{\\mathrm{f}}$", "corr", f"{RHO} between the two scores over the fully observed test individuals."),
                Column("jacc_labelfree", f"$J_{{{K}}}$", "corr", f"Expected top-{K} Jaccard of the score with the same score on the label-free operator."),
                Column("jacc_placebo", f"$J^{{\\mathrm{{pl}}}}_{{{K}}}$", "corr", f"Mean expected top-{K} Jaccard of the score with its placebo rankings."),
                Column("pval_p100", "$p$", "corr", f"Placebo $p$-value of the score's precision at {K}."))))),
    TableSpec(
        "T5_runtime", MAIN, "Table 5", _ds("T5_runtime"), build=build_t5, group_by="group_label", sort=("dataset", "threads"), align="lrrrl",
        panels=_single(
            Column("operation", "Operation", "text", "Operation timed on the dataset's two-layer network; $N$ supra-nodes of the full operator and $R$ of the lumped one are in the group title."),
            Column("threads", "Threads", "int", "Threads available to the linear-algebra libraries."),
            Column("seconds", "Time (s)", "seconds",
                   "Median wall-clock time in seconds over the repeats (per step for the walk rows), to three significant figures."),
            Column("peak_mb", "Memory (MiB)", "mb", "Peak traced memory in MiB ($2^{20}$ bytes), rounded to an integer."),
            Column("status", "Note", "text", "Why a cell is empty, when it is."))),
    TableSpec(
        "T6_hardware", MAIN, "Table 6", _ds("T6_hardware") + _ds("T6_hardware_header"), build=build_t6, group_by="group_label", align="lrrrr",
        optional=("T6_hardware_header",),
        panels=_single(
            Column("quantity", "", "text", "Quantity of the retained instance, graded against the exact lumped average; the instance (classes, borrowers, $R$, $m$, $\\tau$, route) is in the group title."),
            Column("exact", "exact", RAW, "Exact lumped time average, the reference of every other column."),
            Column("noiseless", "noiseless shots", RAW, "Noiseless shot simulation of the circuits."),
            Column("aernoise", "Aer noise model", RAW, "Aer simulation with a noise model built from the calibration snapshot."),
            Column("fake", "FakeMarrakesh", RAW, "Simulation with the FakeMarrakesh calibration snapshot; no job ran on hardware."))),
    # --- e-companion ---
    TableSpec(
        "EC_counterexample", EC, "Table EC.1", (("synthetic", "EC_counterexample"),), group_by=None,
        panels=_single(
            Column("eta", "$\\eta$", "int", "Interlayer weight of the counterexample network."),
            Column("restart", "Restart", "text", "Restart of the walk: the combined multilayer restart or the rank-one restart (a control)."),
            Column("n_full", "$N$", "count", "Supra-nodes of the unlumped operator."),
            Column("n_reduced", "$R$", "count", "Supra-nodes of the lumped operator."),
            Column("compatibility_residual", "$\\max|DB-B\\bar{D}|$", "small", "Compatibility residual the router compares with its tolerance."),
            Column("route", "Route", "route", ROUTE),
            Column("e100_dbar", "$e_{100}$ ($\\bar{D}$)", "small", "Largest difference between the reduced and the full individual time average at $\\tau = 100$ under $K = \\bar{D}$."),
            Column("e100_h", "$e_{100}$ ($H$)", "small", "The same under $K = H$."))),
    TableSpec(
        "EC_reduction_residuals", EC, "e-companion, reduction", _ds("EC_reduction_residuals"),
        panels=_single(
            NETWORK,
            Column("n_full", "$N$", "count", "Supra-nodes of the unlumped operator (unused category slots dropped)."),
            Column("n_reduced", "$R$", "count", "Supra-nodes of the lumped operator."),
            Column("lumpability_residual", "$\\varepsilon_{\\mathrm{lump}}$", "small", "Lumpability residual $\\max|CG - \\bar{G}C|$."),
            Column("compatibility_residual", "$\\varepsilon_{\\mathrm{compat}}$", "small", "Compatibility residual $\\max|DB - B\\bar{D}|$."),
            Column("route", "Route", "route", ROUTE),
            Column("full_vs_reduced_error", "$e_{\\mathrm{full}}$", "small",
                   "Largest difference between the reduced and the full step marginal over the recorded steps and individuals, under the selected route."))),
    TableSpec(
        "EC_convergence", EC, "e-companion, convergence", _ds("T_convergence") + _ds("EC_convergence_sensitivity"), build=build_ec_convergence,
        optional=("EC_convergence_sensitivity",), align="lrrrrr",
        panels=(
            Panel("A. Convergence point $\\tau^*$ of the running average", (
                NETWORK,
                Column("R", "$R$", "count", "Supra-nodes of the lumped operator."),
                Column("route", "Route", "route", ROUTE),
                Column("tau_star_printed", "$\\tau^*$", "tau", TAU_STAR),
                Column("tau_star_l1", "$\\tau^*_{L_1}$", "tau", "Convergence point of the $L_1$ criterion alone."),
                Column("tau_star_rank", "$\\tau^*_{\\mathrm{rank}}$", "tau", "Convergence point of the rank criteria alone."))),
            Panel("B. $\\tau^*$ over the tolerance grid", (
                NETWORK,
                Column("grid_min", "min", "tau", "Smallest $\\tau^*$ over the grid of tolerances of the convergence criterion that reach a convergence point."),
                Column("grid_max", "max", "tau", "Largest $\\tau^*$ over the same grid."),
                Column("grid_above", "$n_{>}$", "count", "Grid settings that do not reach a convergence point before the cap $\\tau_{\\max}$."))),
            Panel("C. Constants of the bound", (
                NETWORK,
                Column("r_active", "$r$", "count", "Eigenvalue groups of the walk with non-zero initial mass."),
                Column("delta_chord", "$\\delta_{\\mathrm{chord}}$", "small", "Smallest chord between active eigenphase groups."),
                Column("c_exact", "$c_{\\mathrm{exact}}$", "small", "Exact constant of the bound $\\|\\bar\\omega_\\tau - \\omega\\|_1 \\le c/\\tau$."),
                Column("c_bound", "$2(r-1)/\\delta$", "small", "The analytic constant $2(r-1)/\\delta_{\\mathrm{chord}}$."))),
            Panel("D. Running average against the limit", (
                NETWORK,
                Column("l1_at_100", "$L_1(100)$", "small", "$L_1$ distance between the running average at $\\tau = 100$ and the limit."),
                Column("rho_100_vs_limit", "$\\rho_{100}$", "corr", f"{RHO} between the running average at $\\tau = 100$ and the limit over the test individuals."),
                Column("rho_classical_tau_star", "$\\rho_C(\\tau^*)$", "corr",
                       f"{RHO} between the classical running average at $\\tau^*$ and the classical stationary score."))))),
    TableSpec(
        "EC_registers", EC, "e-companion, registers", _ds("EC_registers"),
        panels=_single(
            NETWORK,
            Column("rho_src_dst", "$\\rho_{\\mathrm{sd}}$", "corr", f"{RHO} between the limits read from the source and the destination register."),
            Column("kendall_src_dst", "$\\tau_b$", "corr", "Kendall $\\tau_b$ between the two register limits."),
            Column("ov_src_dst", f"$O_{{{K}}}$", "corr", f"Expected top-{K} overlap between the two register limits."),
            Column("rho_src_c", "$\\rho_{\\mathrm{sC}}$", "corr", f"{RHO} between the source-register limit and the classical score."),
            Column("rho_dst_c", "$\\rho_{\\mathrm{dC}}$", "corr", f"{RHO} between the destination-register limit and the classical score."),
            Column("levels_src", "$L_{\\mathrm{s}}$", "count", "Distinct levels of the source-register limit."),
            Column("levels_dst", "$L_{\\mathrm{d}}$", "count", "Distinct levels of the destination-register limit."))),
    TableSpec(
        "EC_classical_horizon", EC, "e-companion, classical horizon", _ds("EC_classical_horizon"),
        panels=_single(
            NETWORK,
            Column("tau_eval", "$\\tau$", "tau", "Horizon of the classical running average: $\\tau^*$ when reached, else the cap $\\tau_{\\max}$."),
            Column("rho_classical_tau_star", "$\\rho$", "corr", f"{RHO} between the classical running average at that horizon and the stationary score."),
            Column("kendall_classical_tau_star", "$\\tau_b$", "corr", "Kendall $\\tau_b$ of the same pair."),
            Column("ov_classical_tau_star", f"$O_{{{K}}}$", "corr", f"Expected top-{K} overlap of the same pair."))),
    TableSpec(
        "EC_single_layer_share", EC, "e-companion, single layer", _ds("EC_single_layer_share"), build=build_ec_single_layer_share,
        group_by="group_label", align="p{1.5in}rrrrrr",
        panels=(
            Panel("A. Reference shares and the closed form", (
                Column("category_name", "Category", "text", "Category of the layer (``unrecorded'' is the family of individuals without a membership edge)."),
                Column("n_c", "$n_c$", "count", "Members of the category: development references, development non-references and masked test members."),
                Column("s_c", "$s_c$", "count", "Development references in the category."),
                Column("f_c", "$f_c$", "corr", "Reference share $s_c/n_c$."),
                Column("classical_test_member", "classical", "small", "Classical stationary score of one test member."),
                Column("formula", "formula", "small", "Value of the single-layer closed form $\\alpha f_c R / [(1-\\alpha)(1+\\alpha-\\alpha f_c)]$."),
                Column("rel_error", "rel. error", "small", "Relative error $|\\text{classical} - \\text{formula}|/\\text{formula}$."))),
            Panel("B. QPPR score and ranks", (
                Column("category_name", "Category", "text", "Category of the layer."),
                Column("rank_by_f", "$r_f$", "mean", "Rank of the category by reference share (1 = largest, average ranks on ties)."),
                Column("rank_by_classical", "$r_C$", "mean", "Rank of the category by the classical score."),
                Column("rank_by_qppr", "$r_Q$", "mean", "Rank of the category by the QPPR limit."),
                Column("qppr_test_member", "QPPR", "small", "QPPR limit of one test member."))))),
    TableSpec(
        "EC_multisplit", EC, "e-companion, variability", _ds("T2_variability"), build=build_ec_multisplit, align="lrrrrr",
        panels=(
            Panel("A. $\\rho$ over re-splits (HMEQ) and reference resamples (Freddie Mac)", (
                NETWORK,
                Column("n_splits", "$n$", "count", "Re-splits (HMEQ) or reference resamples (Freddie Mac) behind the row."),
                Column("rho_qc_med", "median [2.5%, 97.5%]", "corr",
                       f"Median of {RHO} between the QPPR limit and the classical score over the re-splits or resamples, with the 2.5th and 97.5th percentiles.",
                       ci=("rho_qc_lo", "rho_qc_hi")),
                Column("rho_qc_min", "min", "corr", "Minimum of $\\rho$ over the re-splits or resamples."),
                Column("rho_qc_max", "max", "corr", "Maximum of $\\rho$ over the re-splits or resamples."),
                Column("rho_boot", "bootstrap [lo, hi]", RAW,
                       "2.5th and 97.5th percentiles of $\\rho$ over the shared bootstrap replicates of the test individuals, on the frozen test partition."))),
            Panel("B. $\\Delta$PR-AUC and levels (medians)", (
                NETWORK,
                Column("d_prauc_med", "$\\Delta$ median [2.5%, 97.5%]", "diff",
                       "Median over the re-splits or resamples of the PR-AUC difference QPPR $-$ classical, with the 2.5th and 97.5th percentiles.",
                       ci=("d_prauc_lo", "d_prauc_hi"), span=3, align="r"),
                Column("levels_q_med", "$L^{Q}$", "mean", "Median over the re-splits or resamples of the distinct score levels of the QPPR limit."),
                Column("levels_c_med", "$L^{C}$", "mean", "Median over the re-splits or resamples of the distinct score levels of the classical score."))),
            Panel("C. $\\tau^*$ over re-splits (HMEQ)", (
                NETWORK,
                Column("tau_star_cell", "median [min, max]", RAW,
                       "Median, minimum and maximum of $\\tau^*$ over the re-splits on which it was reached (primary network only).", span=2, align="r"),
                Column("n_tau_star_not_reached", "$n_{>}$", "count", "Re-splits on which $\\tau^*$ was not reached before the cap $\\tau_{\\max}$.")),
                where=_present("n_tau_star_evaluated")))),
    TableSpec(
        "EC_decision_full", EC, "e-companion, decision", _ds("EC_decision_full"), build=build_ec_decision_full, group_by="group_label", align="p{1.9in}rrr",
        panels=(
            Panel("A. PR-AUC and ROC-AUC with 95% intervals", (
                Column("label", "Score", "text", "Score the row evaluates; the paired row is the difference QPPR $-$ classical."),
                Column("pr_auc_cell", "PR-AUC [95% CI]", RAW, f"PR-AUC on the test partition {CI95}; on the paired row, the paired difference."),
                Column("roc_auc_cell", "ROC-AUC [95% CI]", RAW, f"ROC-AUC on the test partition {CI95}; on the paired row, the paired difference."),
                Column("distinct_levels", "Levels", "count", "Distinct test-score levels at the tie tolerance."))),
            Panel(f"B. Precision at {K}", (
                Column("label", "Score", "text", "Score the row evaluates."),
                Column("precision_cell", f"P@{K} [95% CI]", RAW,
                       f"Precision at {K}, expected over tie-breakings, {CI95}; on the paired row, the paired difference."),
                Column("lift_cell", f"Lift@{K}", RAW, f"Lift at {K} (precision at {K} over the test prevalence); on the paired row, the difference."))))),
    TableSpec(
        "EC_placebo", EC, "e-companion, placebo", _ds("EC_placebo"), build=network_grouped("EC_placebo"), group_by="group_label", align="lrrrrrr",
        panels=(
            Panel("A. PR-AUC: true value, placebo mean, 5th and 95th percentile", (
                ARM,
                Column("n_perm", "$B$", "count", "Placebo permutations of the reference set."),
                Column("true_pr_auc", "true", "corr", "PR-AUC of the true reference set."),
                Column("placebo_mean_pr_auc", "mean", "corr", "Mean PR-AUC over the placebo draws."),
                Column("placebo_p05_pr_auc", "5th", "corr", "5th percentile of PR-AUC over the placebo draws."),
                Column("placebo_p95_pr_auc", "95th", "corr", "95th percentile of PR-AUC over the placebo draws."),
                Column("pval_pr_auc", "$p$", "corr", "Placebo $p$-value $(1 + \\#\\{\\text{placebo} \\ge \\text{true}\\})/(1 + B)$ of PR-AUC."))),
            Panel(f"B. Precision at {K} and the top-{K} Jaccard with the true ranking", (
                ARM,
                Column(f"true_precision_at_{K}", "true", "corr", f"Precision at {K} of the true reference set."),
                Column(f"placebo_mean_precision_at_{K}", "mean", "corr", f"Mean precision at {K} over the placebo draws."),
                Column(f"pval_precision_at_{K}", "$p$", "corr", f"Placebo $p$-value of the precision at {K}."),
                Column(f"mean_top_{K}_jaccard_with_true", f"$J_{{{K}}}$", "corr",
                       f"Mean expected top-{K} Jaccard of the placebo ranking with the true ranking."))))),
    TableSpec(
        "EC_strata", EC, "e-companion, strata", _ds("EC_strata"), build=build_ec_strata, group_by="group_label", align="lrr",
        panels=_single(
            ARM,
            Column("pr_auc_cell", "PR-AUC [95% CI]", RAW, f"PR-AUC within the stratum {CI95}; on the paired row, the paired difference."),
            Column("roc_auc_cell", "ROC-AUC [95% CI]", RAW, f"ROC-AUC within the stratum {CI95}; on the paired row, the paired difference."))),
    TableSpec(
        "EC_downstream", EC, "e-companion, downstream", _ds("EC_downstream"), align="llrr",
        panels=(
            Panel("A. Test-set performance with 95% intervals", (
                Column("model", "Model", "text", "Downstream model, fitted on the development partition."),
                Column("features", "Features", "text", "Feature set: the covariates alone, plus the classical score, or plus the QPPR limit."),
                Column("pr_auc", "PR-AUC [95% CI]", "corr", f"PR-AUC on the test partition {CI95}.", ci=("pr_auc_ci_lo", "pr_auc_ci_hi")),
                Column("roc_auc", "ROC-AUC [95% CI]", "corr", f"ROC-AUC on the test partition {CI95}.", ci=("roc_auc_ci_lo", "roc_auc_ci_hi")))),
            Panel("B. Change from the covariates-only model", (
                Column("model", "Model", "text", "Downstream model."),
                Column("features", "Features", "text", "Feature set."),
                Column("delta_pr_auc", "$\\Delta$PR-AUC [95% CI]", "diff", f"Paired PR-AUC difference against the covariates-only model {CI95}.",
                       ci=("delta_pr_auc_ci_lo", "delta_pr_auc_ci_hi")),
                Column("delta_roc_auc", "$\\Delta$ROC-AUC [95% CI]", "diff", f"Paired ROC-AUC difference against the covariates-only model {CI95}.",
                       ci=("delta_roc_auc_ci_lo", "delta_roc_auc_ci_hi"))),
                where=_is_not("features", "covariates_only")))),
    TableSpec(
        "EC_feature_importance", EC, "e-companion, feature importance", _ds("EC_feature_importance"), align="llrrrrr",
        panels=_single(
            Column("model", "Model", "text", "Downstream model."),
            Column("method", "Method", "text", "Importance method of the first share and rank columns."),
            Column("layerdummy_importance_share", "layer share", "corr", "Permutation importance of the layer indicators as a share of the total (average precision on the test partition)."),
            Column("layerdummy_importance_rank", "layer rank", "count", "Rank of the best layer indicator among all features by permutation importance."),
            Column("n_features", "of", "count", "Features the rank is out of: the covariates plus the layer indicators."),
            Column("shap_layerdummy_share", "SHAP share", "corr", "Mean absolute SHAP contribution of the layer indicators as a share of the total (gradient-boosted model only)."),
            Column("shap_layerdummy_rank", "SHAP rank", "count", "Rank of the best layer indicator among all features by SHAP share."))),
    TableSpec(
        "EC_synthetic", EC, "e-companion, synthetic (summary of Figure 3)",
        (("synthetic", "EC_synthetic_S1_summary"), ("synthetic", "EC_synthetic_S2_summary"),
         ("synthetic", "EC_synthetic_S3_summary"), ("synthetic", "EC_synthetic_S3b_summary")),
        build=build_ec_synthetic, group_by="group_label", align="lrrrrrr",
        optional=("EC_synthetic_S1_summary", "EC_synthetic_S2_summary", "EC_synthetic_S3_summary", "EC_synthetic_S3b_summary"),
        panels=(
            Panel(f"S1. Dangling share {S1_SHARE:g}: (a) non-reference, (b) 10% of references", (
                Column("label", "Setting", "text", "Quantity of the null control: individuals without a membership edge are inert.", span=3, align="l"),
                Column("mean", "mean", RAW, "Mean over the replicates."),
                Column("lo", "2.5%", RAW, "2.5th percentile over the replicates."),
                Column("hi", "97.5%", RAW, "97.5th percentile over the replicates."),
                Column("n", "$n$", RAW, "Replicates behind the row.")),
                where=lambda row: row.get("experiment") == "S1", grouped=False),
            Panel(f"S2. Single layer, dispersion {S2_SETTING['size_dispersion']:g}, spread {S2_SETTING['ref_share_spread']:g}: $1 - \\rho$", (
                Column("label", "Categories", "text", "Number $C$ of categories of the single layer.", span=3, align="l"),
                Column("mean", "mean", RAW, "Mean over the replicates of $1 - \\rho$ between the QPPR limit and the classical score."),
                Column("lo", "2.5%", RAW, "2.5th percentile over the replicates."),
                Column("hi", "97.5%", RAW, "97.5th percentile over the replicates."),
                Column("n", "$n$", RAW, "Replicates behind the row.")),
                where=lambda row: row.get("experiment") == "S2", grouped=False),
            Panel("S3. Partially observed share $m$ and weight ratio $\\kappa$", (
                Column("label", "Quantity", "text", "Quantity of the two-layer experiment."),
                Column("m", "$m$", RAW, "Share of partially observed test individuals."),
                Column("kappa", "$\\kappa$", RAW, "Relative default risk of the partially observed individuals."),
                Column("mean", "mean", RAW, "Mean over the replicates."),
                Column("lo", "2.5%", RAW, "2.5th percentile over the replicates."),
                Column("hi", "97.5%", RAW, "97.5th percentile over the replicates."),
                Column("n", "$n$", RAW, "Replicates behind the row.")),
                where=lambda row: row.get("experiment") == "S3"),
            Panel("S3b. References per category: $\\rho(Q, C)$ and its difference from S3", (
                Column("label", "Quantity", "text", "Quantity of the placement sensitivity run on the S3 populations."),
                Column("m", "$m$", RAW, "Share of partially observed test individuals."),
                Column("kappa", "$\\kappa$", RAW, "Relative default risk of the partially observed individuals."),
                Column("mean", "mean", RAW, "Mean over the replicates."),
                Column("lo", "2.5%", RAW, "2.5th percentile over the replicates."),
                Column("hi", "97.5%", RAW, "97.5th percentile over the replicates."),
                Column("n", "$n$", RAW, "Replicates behind the row.")),
                where=lambda row: row.get("experiment") == "S3b"))),
    TableSpec(
        "EC_scaling", EC, "e-companion, scaling (summary of the EC scaling figure)", (("runtime", "EC_scaling"),), group_by="config", align="rrrrrr",
        panels=(
            Panel("A. Sizes", (
                Column("n", "$n$", "count", "Individuals of the synthetic instance."),
                Column("N", "$N$", "count", "Supra-nodes of the unlumped operator."),
                Column("classes", "classes", "count", "Lumping classes."),
                Column("R", "$R$", "count", "Supra-nodes of the lumped operator."))),
            Panel("B. Time to compute (s)", (
                Column("n", "$n$", "count", "Individuals of the synthetic instance."),
                Column("t_lumped_limit", "lumped limit", "seconds", "Seconds of the lumped spectral limit, to three significant figures."),
                Column("t_full_limit", "full limit", "seconds", "Seconds of the full spectral limit, while $N$ allows it."),
                Column("t_lumped_step", "lumped step", "seconds", "Seconds per step of the lumped recurrence."),
                Column("t_full_step", "full step", "seconds", "Seconds per step of the full recurrence."),
                Column("full_status", "Note", "text", "Why the full computation was not run, when it was not."))))),
    TableSpec(
        "EC_circuit_resources", EC, "e-companion, circuit resources", _ds("EC_circuit_resources"), build=build_ec_circuit_resources, align="lrrrrrrrr",
        panels=(
            Panel("A. Register and operator", (
                Column("classes_label", "classes", "text", "Lumping classes the instance retains; ``full'' marks the unreduced instance."),
                Column("borrowers", "borrowers", "count", "Borrowers the instance covers."),
                Column("R", "$R$", "count", "Supra-nodes of the retained lumped operator."),
                Column("m", "$m$", "int", "Register width $\\lceil\\log_2 R\\rceil$."),
                Column("logical_qubits", "qubits", "int", "Logical qubits $2m$."),
                Column("route", "Route", "route", ROUTE),
                Column("gbar_density", "$\\delta_{\\bar{G}}$", "corr", "Share of non-zero entries of $\\bar{G}$."),
                Column("max_column_support", "$s_{\\max}$", "count", "Largest column support of $\\bar{G}$."),
                Column("fraction_columns_full_reference_support", "$f_{\\mathrm{ref}}$", "corr", "Share of columns with support on every reference copy."))),
            Panel("B. State-preparation angles and two-qubit gates", (
                Column("classes_label", "classes", "text", "Lumping classes the instance retains."),
                Column("state_prep_angles", "angles", "count", "Multiplexed state-preparation angles $2^m(2^m-1)$ of one reflection."),
                Column("two_qubit_logical_analytic", "analytic", "count", "Analytic two-qubit gate count of one reflection--swap step."),
                Column("two_qubit_logical_transpiled", "transpiled", "count", "Two-qubit gates of one logical step after a basis transpilation without routing, where $m$ allows it."))))),
    TableSpec(
        "EC_statevector", EC, "e-companion, statevector rung", _ds("EC_statevector"), build=build_ec_statevector, group_by=None, align="lrr",
        panels=_single(
            Column("quantity", "", "text", "Quantity of the statevector check of the lumped walk."),
            Column("hmeq", DATASET_LABELS["hmeq"], RAW, "Value for the HMEQ dataset."),
            Column("fmac", DATASET_LABELS["fmac"], RAW, "Value for the Freddie Mac extract."))),
    TableSpec(
        "EC_noise_grid", EC, "e-companion, noise grid (summary of Figure 4)", _ds("EC_noise_grid"), build=build_ec_noise_grid, align="rrrllrrrrrr",
        panels=(Panel(f"Raw fidelity ratio at each recorded step $\\tau$; $\\tau_0$: first $\\tau \\geq {MIN_TAU}$ without signal", (
            Column("k_classes", "$k$", "count", "Lumping classes the simulated instance retains."),
            Column("R", "$R$", "count", "Supra-nodes of the instance."),
            Column("m", "$m$", "int", "Register width $\\lceil\\log_2 R\\rceil$."),
            Column("route", "Route", "route", ROUTE),
            ARM,
            *(Column(f"fr_tau_{t}", f"$\\tau = {t}$", "real",
                     f"Fidelity ratio $1 - \\mathrm{{TV}}/\\mathrm{{TV}}_{{\\mathrm{{uniform}}}}$ of the raw output (no readout mitigation) at $\\tau = {t}$ recorded steps, "
                     "against the exact lumped average at that step; a dash where the cell was not simulated.") for t in STEPS),
            Column("first_tau_no_signal", "$\\tau_0$", RAW,
                   f"First $\\tau \\ge {MIN_TAU}$ at which the raw output carries no signal, that is, its fidelity ratio is not above the upper edge of the "
                   "pure-noise band of its own instance; $> t$ names the largest simulated $\\tau$ when every one carries signal, and a dash "
                   f"means no $\\tau \\ge {MIN_TAU}$ was simulated."))),)),
    TableSpec(
        "EC_hardware_scores", EC, "e-companion, hardware scores (selected instance)", _ds("EC_hardware_scores"), build=build_ec_hardware_scores, align="lrrr",
        panels=(
            Panel("A. Exact average and the noiseless and Aer arms", (
                Column("node_name", "Supra-node", "text", "Supra-node of the retained instance, named by its layer and class."),
                Column("exact", "exact", "real", "Exact time-averaged source probability at the selected $\\tau$."),
                Column("noiseless_shots", "noiseless shots", "real", "Time-averaged measured probability, noiseless shot simulation (raw)."),
                Column("aer_noise_model", "Aer noise model", "real", "Time-averaged measured probability, Aer noise model simulation (raw)."))),
            Panel("B. FakeMarrakesh arm", (
                Column("node_name", "Supra-node", "text", "Supra-node of the retained instance."),
                Column("exact", "exact", "real", "Exact time-averaged source probability."),
                Column("fake_marrakesh", "raw", "real", "Time-averaged measured probability, FakeMarrakesh simulation, raw."),
                Column("fake_marrakesh_mitigated", "mitigated", "real", "The same after per-circuit readout mitigation."))))),
)

SPECS_BY_NAME = {spec.name: spec for spec in TABLE_SPECS}


def source_map() -> list[dict[str, str]]:
    """``(table, tier, manuscript, sources)`` rows for the README's results map."""
    out = []
    for spec in TABLE_SPECS:
        sources = ", ".join(f"results/outputs/results/{scope}/{name}.csv" for scope, name in spec.expanded_sources())
        out.append({"table": f"results/outputs/manuscript/tables/{spec.name}.tex", "tier": spec.tier, "manuscript": spec.manuscript, "sources": sources})
    return out
