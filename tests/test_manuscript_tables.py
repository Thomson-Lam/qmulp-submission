"""The ``.tex`` assembler and its specs: every rendered tabular is well formed and reads only known sources.

No LaTeX is compiled (``tests/test_table_widths.py`` does that when ``pdflatex`` exists). On an in-memory frame the
renderer must produce a bare ``tabular`` whose every row spans exactly the grid of columns (a ``\\multicolumn{k}``
cell counting for ``k``, a ``p{..}`` column for one) and ends in ``\\\\``, with no ``\\textbackslash``, no unescaped
``_``, ``%`` or ``&`` outside math and no text hyphen in front of a number; the same scan runs over the tables the
specs build from the real result CSVs when ``outputs/results`` exists (rendered in memory by the code under test, never read
from a ``.tex`` file an earlier run left behind). Every spec names result tables under a known scope, every column
rule is one the formatter knows and every printed column has a definition, a route column prints through the one
``route`` rule, and no label prints a code identifier.
"""

from __future__ import annotations

import re
import unittest

import polars as pl

from evaluation.formatting import (
    RULES, TEXT_LABELS, Block, category_label, ci_cell, format_cell, grid_columns, human_label, node_label, spanned, tabular, tex_escape,
)
from evaluation.table_builders import interval_cell, short_split
from evaluation.table_specs import RAW, SPECS_BY_NAME, TABLE_SPECS, source_map
from paths import RESULT_SCOPES, RESULTS_DIR
from quantum.hardware.noise_grading import NOT_SIMULATED
from scripts.make_manuscript_tables import assemble, column_rows, grid_width, render

ALIGN = re.compile(r"\\begin\{tabular\}\{((?:p\{[^}]*\}|[lrc])+)\}")
SPAN = re.compile(r"\s*\\multicolumn\{(\d+)\}\{[lrc]\}\{")


def _outside_math(line: str) -> str:
    parts = line.split("$")
    return "".join(part for i, part in enumerate(parts) if i % 2 == 0)


def check_tabular(tex: str, context: str) -> None:
    lines = tex.strip().splitlines()
    assert lines[0].startswith(r"\begin{tabular}{") and lines[-1] == r"\end{tabular}", context
    n_cols = len(grid_columns(ALIGN.match(lines[0]).group(1)))
    assert r"\textbackslash" not in tex, context
    assert r"\begin{table}" not in tex and r"\caption" not in tex and r"\label" not in tex, context
    for line in lines[1:-1]:
        if line in (r"\toprule", r"\midrule", r"\bottomrule"):
            continue
        assert line.endswith(r"\\"), f"{context}: row does not end in \\\\: {line!r}"
        body = line[:-2]
        cells = re.split(r"(?<!\\)&", body)
        grid = sum(int(found.group(1)) if (found := SPAN.match(cell)) else 1 for cell in cells)
        assert grid == n_cols, f"{context}: {grid} grid cells for {n_cols} columns: {line!r}"
        outside = _outside_math(body)
        outside = re.sub(r"\\[&%#_]", "", outside)  # escaped characters are fine
        outside = outside.replace("&", "")           # the cell separators themselves
        assert not re.search(r"[%#_]", outside), f"{context}: unescaped character outside math: {line!r}"
        assert not re.search(r"(?<![\w-])-\d", outside), f"{context}: a text hyphen in front of a number (a minus is typeset as math): {line!r}"


class FormattingTests(unittest.TestCase):
    def test_escape_only_outside_math(self):
        self.assertEqual(tex_escape(r"$\Delta$PR-AUC"), r"$\Delta$PR-AUC")
        self.assertEqual(tex_escape(r"$\rho$ vs QPPR"), r"$\rho$ vs QPPR")
        self.assertEqual(tex_escape(r"$O_{100}$ ceiling"), r"$O_{100}$ ceiling")
        self.assertEqual(tex_escape("NET_2L & 5%"), r"NET\_2L \& 5\%")
        with self.assertRaises(ValueError):
            tex_escape("unbalanced $x")
        self.assertEqual(ci_cell(0.1234, 0.1, 0.2, "corr"), "0.123 [0.100, 0.200]")
        self.assertEqual(ci_cell(0.1234, None, 0.2, "corr"), "0.123")
        self.assertEqual(ci_cell(None, 0.1, 0.2, "corr"), "--")
        self.assertEqual(format_cell(None, "count"), "--")
        self.assertEqual((format_cell("D_bar", "route"), format_cell("H", "route")), (r"$\bar{D}$", "$H$"))
        self.assertEqual((format_cell("classical", "route"), format_cell(None, "route")), ("--", "--"))

    def test_numbers_carry_math_signs_and_thousands_separators(self):
        self.assertEqual((format_cell(-0.019, "diff"), format_cell(0.001, "diff")), ("$-$0.019", "$+$0.001"))
        self.assertEqual((format_cell(-0.277, "corr"), format_cell(0.062, "corr"), format_cell(-5, "int")), ("$-$0.277", "0.062", "$-$5"))
        self.assertEqual(ci_cell(-0.019, -0.052, 0.02, "diff"), "$-$0.019 [$-$0.052, $+$0.020]")
        self.assertEqual((format_cell(1651, "count"), format_cell(50000, "count"), format_cell(39800, "tau")), ("1{,}651", "50{,}000", "39{,}800"))
        self.assertEqual(format_cell("> tau_max (50000)", "tau"), r">\,$\tau_{\max}$ = 50{,}000")
        self.assertEqual(format_cell("> tau_max", "tau"), r">\,$\tau_{\max}$")

    def test_seconds_and_memory_have_their_own_rules(self):
        self.assertEqual([format_cell(v, "seconds") for v in (298.58, 2.306, 0.0312, 1234.4)], ["299", "2.31", "0.0312", "1{,}234"])
        self.assertEqual(format_cell(0.000799, "seconds"), r"$7.99\times10^{-4}$")
        self.assertEqual(format_cell(18851.8876, "mb"), "18{,}852")

    def test_identifiers_print_as_words_from_the_one_mapping(self):
        self.assertEqual(format_cell("blank_field_count", "text"), "blank-field count")
        self.assertEqual(format_cell("one_hop_seed_count", "text"), "one-hop reference count")
        self.assertEqual(format_cell("cell_rate", "text"), "joint-category default rate")
        self.assertEqual(format_cell("logit_on_layer_dummies", "text"), "logit on layer indicators")
        self.assertEqual(format_cell("single_layer_seed_default_rate_JOB", "text"), "single-layer reference share (employment)")
        self.assertEqual(format_cell("single-layer reference share (PostalCode)", "text"), "single-layer reference share (postal code)")
        self.assertEqual([format_cell(n, "text") for n in ("NET_JOB", "NET_REASON", "NET_FMAC_POSTAL", "NET_FMAC_SELLER", "NET_2L", "NET_FMAC_2L")],
                         ["employment", "loan purpose", "postal code", "seller", "two-layer", "two-layer"])
        self.assertEqual((human_label("hmeq"), human_label("fmac_like"), human_label("plus_classical")), ("HMEQ", "Freddie Mac-like", "plus classical"))
        self.assertEqual(category_label("JOB:JOB_Mgr"), "Mgr")
        self.assertEqual(category_label("SellerName:SellerName_FAIRWAY INDEPENDENT MORTGAGE CORPORATION"), "FAIRWAY INDEPENDENT MORTGAGE CORPORATION")
        self.assertEqual(category_label("unrecorded"), "unrecorded")
        self.assertEqual((node_label("JOB:group_23"), node_label("REASON:REASON_DebtCon")), ("employment, class 23", "loan purpose: DebtCon"))
        for source, printed in TEXT_LABELS.items():
            self.assertNotIn("_", printed, f"{source!r} would print a code identifier")
        self.assertNotIn(r"\_", format_cell("fully_recorded", "text"))

    def test_tabular_is_bare_and_grouped(self):
        tex = tabular(["Network", r"$\rho$"], [["JOB", "1.000"], ["Postal", "0.500"]], groups=[("HMEQ", 1), ("Freddie Mac", 1)])
        check_tabular(tex, "unit")
        self.assertIn(r"\multicolumn{2}{l}{\textit{HMEQ}} \\", tex)
        with self.assertRaises(ValueError):
            tabular(["a", "b"], [["only one"]])

    def test_panels_share_one_grid_and_a_wide_cell_spans_narrow_columns(self):
        wide = spanned(r"$-$0.019 [$-$0.052, $+$0.020]", 3)
        tex = tabular(align="lrrrr", panels=[
            Block("A. First", ["Network", "a", "b", "c", "d"], [["x", "1", "2", "3", "4"]], [("HMEQ", 1)]),
            Block("B. Second", ["Network", "C", spanned(r"$\Delta$ [95\% CI]", 3)], [["x", "0.1", wide]], [("HMEQ", 1)])])
        check_tabular(tex, "panels")
        self.assertIn(r"\multicolumn{5}{l}{\textbf{A. First}} \\", tex)
        self.assertIn(r"\multicolumn{3}{r}{$-$0.019 [$-$0.052, $+$0.020]}", tex)
        self.assertEqual(tex.count(r"\midrule"), 3, "a rule under each header and one between the panels")
        with self.assertRaises(ValueError):
            tabular(align="lrr", panels=[Block(None, ["a", "b", "c"], [["x", "y"]])])
        with self.assertRaises(ValueError):
            tabular(["a"], [["b"]], panels=[Block(None, ["a"], [["b"]])])


class SpecTests(unittest.TestCase):
    def test_specs_name_known_scopes_and_rules(self):
        names = [spec.name for spec in TABLE_SPECS]
        self.assertEqual(len(names), len(set(names)))
        for required in ("T1_data_reduction", "T2_agreement", "T3_variants", "T4_partial", "T5_runtime", "T6_hardware", "EC_counterexample",
                         "EC_reduction_residuals", "EC_convergence", "EC_registers", "EC_classical_horizon", "EC_single_layer_share",
                         "EC_multisplit", "EC_decision_full", "EC_placebo", "EC_strata", "EC_downstream", "EC_feature_importance",
                         "EC_synthetic", "EC_scaling", "EC_circuit_resources", "EC_statevector", "EC_noise_grid", "EC_hardware_scores"):
            self.assertIn(required, names)
        for spec in TABLE_SPECS:
            for scope, _name in spec.expanded_sources():
                self.assertIn(scope, RESULT_SCOPES, spec.name)
            for column in spec.columns:
                self.assertTrue(column.rule == RAW or column.rule in RULES, f"{spec.name}.{column.source}: {column.rule}")
                self.assertNotIn(r"\_", tex_escape(column.header), f"{spec.name}.{column.source}: a header must not print a code identifier")
                self.assertTrue(column.definition.strip(), f"{spec.name}.{column.source} has no definition for the note under the table")
            if spec.align:
                self.assertEqual(len(grid_columns(spec.align)), grid_width(spec), f"{spec.name}: the alignment is not the grid of its widest panel")
            self.assertEqual(len(column_rows(spec)), len(spec.columns), spec.name)
        rows = source_map()
        self.assertTrue(all(r["table"].startswith("results/outputs/manuscript/tables/") for r in rows))

    def test_every_route_column_prints_through_the_one_route_rule(self):
        with_route = {spec.name for spec in TABLE_SPECS if any(c.source == "route" for c in spec.columns)}
        self.assertLessEqual({"T3_variants", "EC_counterexample", "EC_reduction_residuals", "EC_convergence", "EC_circuit_resources",
                              "EC_noise_grid"}, with_route)
        for spec in TABLE_SPECS:
            for column in spec.columns:
                if column.source == "route":
                    self.assertEqual(column.rule, "route", f"{spec.name}: a route is printed by the route rule, not relabelled")

    def test_render_a_stacked_table(self):
        spec = SPECS_BY_NAME["T3_variants"]
        frame = pl.DataFrame({"dataset": ["hmeq", "hmeq"], "network": ["NET_2L", "NET_2L"], "variant": ["primary", "eta_0"],
                              "alpha": [0.85, 0.85], "eta": [1.0, 0.0], "route": ["D_bar", "H"], "rho_vs_primary_q": [1.0, 0.4321],
                              "ov_vs_primary_q": [1.0, 0.5], "rho_vs_own_classical": [0.3, float("nan")]})
        frames = {("hmeq", "T3_variants"): frame, ("fmac", "T3_variants"): None}
        tex = render(spec, frames)
        check_tabular(tex, "T3_variants")
        self.assertIn(r"$\bar{D}$", tex)
        self.assertIn("0.432", tex)
        self.assertIn("--", tex)
        self.assertIn(r"$\eta = 0$", tex, "a variant prints its label, not its identifier")
        self.assertNotIn("NET_2L", tex)
        self.assertNotIn("eta_0", tex)

    def test_the_noise_grid_summary_prints_the_first_step_without_signal_by_the_one_rule(self):
        def cell(k, tau, fr, status=None):
            return {"dataset": "hmeq", "k_classes": k, "instance": f"top_{k}_classes", "R": 10, "m": 4, "route": "D_bar", "arm": "fake_marrakesh",
                    "mitigated": False, "tau": tau, "status": status, "spearman_supra": 0.5, "fidelity_ratio": fr, "pure_noise_fidelity_hi": 0.02}

        rows = ([cell(3, t, 0.5) for t in range(1, 6)]                                    # signal at every step: no step without it
                + [cell(4, 1, 0.5), cell(4, 2, 0.5), cell(4, 3, 0.0), cell(4, 4, 0.5)]      # the first step without signal is 3
                + [cell(5, 1, 0.5), cell(5, 2, None, status=NOT_SIMULATED)])                # only the state preparation was simulated
        tex = render(SPECS_BY_NAME["EC_noise_grid"], {("hmeq", "EC_noise_grid"): pl.DataFrame(rows), ("fmac", "EC_noise_grid"): None})
        check_tabular(tex, "EC_noise_grid")
        printed = [line[:-2].split("&")[-1].strip() for line in tex.splitlines() if line.endswith(r"\\") and "FakeMarrakesh" in line]
        self.assertEqual(printed, ["> 5", "3", "--"], "a cell that was never evaluated prints a dash, not a step")
        self.assertIn("0.500", tex, "the table reads the raw fidelity ratio")

    def test_render_the_composite_t2_in_three_panels_with_the_joint_cell_on_the_two_layer_row(self):
        spec = SPECS_BY_NAME["T2_agreement"]
        networks = ["NET_JOB", "NET_2L"]
        agreement = pl.DataFrame({"dataset": ["hmeq"] * 2, "network": networks, "rho_qc": [0.9, 0.5], "kendall_qc": [0.8, 0.4], "ov_qc": [0.7, 0.3],
                                  "selfov_q": [0.9, 0.9], "selfov_c": [0.8, 0.8], "levels_q": [7, 20], "levels_c": [7, 10],
                                  "tau_star_printed": ["163", "12345"]})
        decision = pl.DataFrame({"dataset": ["hmeq"] * 2, "network": networks, "prauc_c": [0.31, 0.35], "prauc_q": [0.30, 0.34], "d_prauc": [-0.01, -0.01],
                                 "d_prauc_lo": [-0.03, -0.03], "d_prauc_hi": [0.01, 0.01], "prauc_cellrate": [0.28, 0.29]})
        variability = pl.DataFrame({"dataset": ["hmeq"] * 2, "network": networks, "rho_qc_med": [0.9, 0.5], "rho_qc_min": [0.8, 0.4], "rho_qc_max": [1.0, 0.6]})
        frames = {("hmeq", "T2_agreement"): agreement, ("hmeq", "T2_decision"): decision, ("hmeq", "T2_variability"): variability,
                  ("fmac", "T2_agreement"): None, ("fmac", "T2_decision"): None, ("fmac", "T2_variability"): None}
        tex = render(spec, frames)
        check_tabular(tex, "T2_agreement")
        self.assertEqual(tex.count(r"\textit{HMEQ}"), 3, "the three panels each group their rows by dataset")
        panel_b = {line.split(" &")[0]: line for line in tex.splitlines() if r"$-$0.010 [" in line}
        self.assertEqual(sorted(panel_b), ["employment", "two-layer"])
        self.assertIn(r"\multicolumn{2}{r}{0.290}", panel_b["two-layer"], "the joint-cell PR-AUC is a column of the two-layer row")
        self.assertIn(r"\multicolumn{2}{r}{--}", panel_b["employment"], "and a dash on the other rows")
        self.assertNotIn("0.280", tex, "the joint-cell score of a single-layer network is not printed")
        self.assertIn("0.500 [0.400, 0.600]", tex)
        self.assertIn("12{,}345", tex)

    def test_freddie_mac_variability_is_read_from_its_two_column_families(self):
        spec = SPECS_BY_NAME["T2_agreement"]
        agreement = pl.DataFrame({"dataset": ["fmac"], "network": ["NET_FMAC_2L"], "rho_qc": [0.39], "kendall_qc": [0.28], "ov_qc": [0.02],
                                  "selfov_q": [0.96], "selfov_c": [0.92], "levels_q": [361], "levels_c": [361], "tau_star_printed": ["> tau_max (200000)"]})
        variability = pl.DataFrame({"dataset": ["fmac"], "network": ["NET_FMAC_2L"], "n_resamples": [50], "resample_rho_qc_med": [0.49],
                                    "resample_rho_qc_min": [0.44], "resample_rho_qc_max": [0.55], "resample_rho_qc_lo": [0.45],
                                    "boot_rho_qc": [0.39], "boot_rho_qc_lo": [0.36], "boot_rho_qc_hi": [0.42]})
        frames = {(scope, name): None for scope in ("hmeq", "fmac") for name in ("T2_agreement", "T2_decision", "T2_variability")}
        frames[("fmac", "T2_agreement")], frames[("fmac", "T2_variability")] = agreement, variability
        tex = render(spec, frames)
        check_tabular(tex, "T2_agreement (fmac)")
        self.assertIn("0.490 [0.440, 0.550]", tex, "the reference-resample range, from resample_*")
        self.assertIn("[0.360, 0.420]", tex, "the bootstrap interval over test loans, from boot_*")
        self.assertNotIn("0.450", tex, "the resample percentile must not be mistaken for the bootstrap interval")
        self.assertIn(r">\,$\tau_{\max}$ = 200{,}000", tex, "the cap is printed with its separator")

    def test_the_multisplit_summary_follows_the_ladder_order_and_shows_unreached_tau_star(self):
        spec = SPECS_BY_NAME["EC_multisplit"]
        networks = ["NET_2L", "NET_JOB", "NET_REASON"]          # the order variability_summary writes (alphabetical)
        frame = pl.DataFrame({"dataset": ["hmeq"] * 3, "network": networks, "n_splits": [50] * 3, "rho_qc_med": [0.3, 0.9, 1.0],
                              "rho_qc_lo": [0.1, 0.8, 1.0], "rho_qc_hi": [0.5, 1.0, 1.0], "rho_qc_min": [0.1, 0.8, 1.0], "rho_qc_max": [0.5, 1.0, 1.0],
                              "n_tau_star_evaluated": [50, None, None], "n_tau_star_not_reached": [50, None, None]})
        tex = render(spec, {("hmeq", "T2_variability"): frame, ("fmac", "T2_variability"): None})
        check_tabular(tex, "EC_multisplit")
        first_panel = [line.split(" &")[0] for line in tex.splitlines() if line.startswith(("employment", "loan purpose", "two-layer"))][:3]
        self.assertEqual(first_panel, ["employment", "loan purpose", "two-layer"], "ladder order, not alphabetical")
        panel_c = tex.split(r"\textbf{C. ")[1]
        rows_c = [line for line in panel_c.splitlines() if line.startswith(("employment", "loan purpose", "two-layer"))]
        self.assertEqual(len(rows_c), 1, "the tau* panel keeps the network whose tau* was evaluated, and only it")
        self.assertTrue(rows_c[0].startswith("two-layer") and r"\multicolumn{2}{r}{--}" in rows_c[0] and rows_c[0].rstrip(r" \\").endswith("& 50 &  &"),
                        "the two-layer network missed tau* on all 50 splits: no median, a count of 50")

    def test_single_layer_tables_group_by_network_and_name_the_category_without_its_layer(self):
        spec = SPECS_BY_NAME["EC_single_layer_share"]
        frame = pl.DataFrame({"dataset": ["fmac"], "network": ["NET_FMAC_SELLER"], "category": ["SellerName:SellerName_FAIRWAY INDEPENDENT MORTGAGE CORPORATION"],
                              "n_c": [2388], "s_c": [450], "f_c": [0.188], "classical_test_member": [7.93e-5], "formula": [7.93e-5], "rel_error": [3.1e-15],
                              "rank_by_f": [14.5], "rank_by_classical": [14.5], "rank_by_qppr": [16.0], "qppr_test_member": [4.37e-5]})
        tex = render(spec, {("hmeq", "EC_single_layer_share"): None, ("fmac", "EC_single_layer_share"): frame})
        check_tabular(tex, "EC_single_layer_share")
        self.assertTrue(tex.startswith(r"\begin{tabular}{p{1.5in}rrrrrr}"), "a long category name wraps in a paragraph column")
        self.assertEqual(tex.count(r"\textit{Freddie Mac: seller}"), 2, "each panel repeats the dataset and network of its rows")
        self.assertIn("FAIRWAY INDEPENDENT MORTGAGE CORPORATION", tex)
        self.assertNotIn("SellerName", tex)

    def test_the_paired_row_prints_a_signed_difference_and_a_level_prints_unsigned(self):
        def row(arm, label, estimate, lo, hi):
            return {"dataset": "hmeq", "network": "NET_2L", "arm": arm, "label": label, "pr_auc": estimate, "pr_auc_ci_lo": lo, "pr_auc_ci_hi": hi,
                    "roc_auc": estimate, "roc_auc_ci_lo": lo, "roc_auc_ci_hi": hi, "precision_at_100": estimate, "precision_at_100_ci_lo": lo,
                    "precision_at_100_ci_hi": hi, "lift_at_100": 1.4, "distinct_levels": 361}

        frame = pl.DataFrame([row("classical", "classical stationary", 0.25, 0.2, 0.3), row("qppr_minus_classical", "QPPR minus classical (paired)", 0.019, -0.01, 0.04)])
        tex = render(SPECS_BY_NAME["EC_decision_full"], {("hmeq", "EC_decision_full"): frame, ("fmac", "EC_decision_full"): None})
        check_tabular(tex, "EC_decision_full")
        self.assertIn("0.250 [0.200, 0.300]", tex)
        self.assertIn("$+$0.019 [$-$0.010, $+$0.040]", tex)
        self.assertIn(r"QPPR $-$ classical (paired)", tex)
        self.assertEqual(interval_cell({"arm": "qppr", "x": 0.5, "x_ci_lo": 0.4, "x_ci_hi": 0.6}, "x"), "0.500 [0.400, 0.600]")

    def test_the_reduction_table_prints_short_labels(self):
        self.assertEqual(short_split("hmeq", {"split": "one stratified 80/20 split on the outcome, seed 42", "split_seed": 42}), "stratified 80/20, seed 42")
        self.assertEqual(short_split("fmac", {"split": "temporal: development = first payment 2023, test = first payment 2024"}), r"temporal, 2023 $\to$ 2024")

    @unittest.skipUnless(RESULTS_DIR.exists() and any(RESULTS_DIR.glob("*/*.csv")), "no result CSVs under outputs/results yet")
    def test_the_tables_built_from_the_real_results_are_well_formed(self):
        tables, _skipped = assemble()
        self.assertTrue(tables, "no table could be built from the result CSVs")
        for name, tex in sorted(tables.items()):
            check_tabular(tex, name)


if __name__ == "__main__":
    unittest.main()
