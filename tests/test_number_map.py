"""The number map: every key resolves against its named source, formats by its kind, and never invents a value.

On a temporary results tree holding a few hand-written CSV and JSON files: a
result key with a row filter picks the one matching row and column, an
aggregate key takes the minimum over the matching rows (an arg-aggregate the
value at the extremum of another column), a stage key follows its dotted
path, a missing file or column leaves the key unresolved, a result file
under one dataset's scope that holds another dataset's rows is an error, the
manifests must agree on one commit, and every key of the contract obeys the
manuscript grammar. An aggregate of keys takes the largest, the smallest or the one common value of keys that
already exist, raises when parts that must agree do not, and stays unresolved when a part is. When
``outputs/results`` holds a finished run of this very source (the stage manifests name the working tree's
``source_sha256``), every key must resolve; against the outputs of an older run that check is skipped.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path

from datasets import get_dataset
from evaluation.formatting import ROUTE_LABELS
from evaluation.number_map import KEY_GRAMMAR, KINDS, KeySpec, fmt, key_specs, read_manifests, resolve, write_number_map
from evaluation.provenance import source_sha256
from paths import PIPELINE_DIR, RESULTS_DIR
from quantum.lumping_router import ROUTE_DBAR, ROUTE_H

KNOWN_RESULTS = {
    "T1_data", "T_convergence", "T2_agreement", "T2_variability", "T2_decision", "EC_registers", "T_eta", "T3_variants",
    "T4_partial", "EC_placebo", "downstream_bound", "EC_feature_importance", "T6_hardware", "T6_hardware_header", "T5_runtime",
}
KNOWN_STAGE_FILES = ("mechanism_summary.json", "EC_synthetic_keys.json", "machine.json")


def outputs_not_of_this_source() -> str | None:
    """Why ``outputs/`` does not describe the code of the working tree (so that every key resolving is not to be expected), else ``None``.

    The key list grows with the code: a key may name a column that only a rerun of its stage writes. The check that every
    key resolves therefore applies to a finished run of this very source: no smoke tree, and every stage manifest names the
    ``source_sha256`` of the tree as it is now. Against the outputs of an older run it would fail on correct code.
    """
    paths = sorted(PIPELINE_DIR.glob("*/manifest.json"))
    if not paths:
        return "no stage manifests under outputs/ yet"
    manifests = [json.loads(p.read_text()) for p in paths]
    if any((m.get("extra") or {}).get("smoke") for m in manifests):
        return "a smoke tree is a wiring check: the keys its trimmed stages do not produce stay unresolved"
    current = source_sha256()
    if current is None:
        return "git cannot list the tree, so the source of the outputs cannot be compared with it"
    if {m.get("source_sha256") for m in manifests} != {current}:
        return "outputs/ was produced by another source tree than the working tree: every key resolves after the next run, not before"
    return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class FormattingTests(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(fmt(0.123456, "corr"), "0.123")
        self.assertEqual(fmt(-0.0042, "diff"), "$-$0.004")
        self.assertEqual(fmt(0.0042, "diff"), "$+$0.004")
        self.assertEqual(fmt(-0.277, "corr"), "$-$0.277")
        self.assertEqual(fmt(5853, "count"), "5{,}853")
        self.assertEqual(fmt(1171.0, "count"), "1{,}171")
        self.assertEqual(fmt(3.1e-3, "small"), r"$3.10\times10^{-3}$")
        self.assertEqual(fmt(0.0, "small"), "0")
        self.assertEqual(fmt(12345, "tau"), "12{,}345")
        self.assertEqual(fmt("> tau_max (50000)", "tau"), r">\,$\tau_{\max}$ = 50{,}000")
        self.assertEqual(fmt("NET_2L", "text"), "two-layer", "a network code prints its words, never an escaped identifier")
        self.assertEqual(fmt("blank_field_count", "text"), "blank-field count")
        self.assertEqual((fmt(298.58, "seconds"), fmt(0.0312, "seconds"), fmt(0.000799, "seconds")), ("299", "0.0312", r"$7.99\times10^{-4}$"))
        self.assertEqual(fmt(18851.8876, "mb"), "18{,}852")
        self.assertEqual(fmt("none", "text"), "none", "none is a legitimate text value (no equal-share pairs), not a missing one")
        self.assertIsNone(fmt("  ", "text"))
        self.assertEqual(fmt(r"$\Delta$PR-AUC", "text"), r"$\Delta$PR-AUC")
        self.assertIsNone(fmt(None, "corr"))
        self.assertIsNone(fmt(float("nan"), "corr"))
        self.assertEqual(fmt(0.5, "real"), "0.500")
        self.assertEqual(fmt(2.5e-5, "real"), r"$2.50\times10^{-5}$")
        self.assertEqual((fmt("D_bar", "route"), fmt("H", "route")), (r"$\bar{D}$", "$H$"))
        self.assertIsNone(fmt("  ", "route"), "a blank route is unresolved, never a guessed discriminant")


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results = Path(self._tmp.name) / "results"
        self.pipeline = Path(self._tmp.name) / "pipeline"
        _write_csv(self.results / "hmeq" / "T2_agreement.csv", [
            {"dataset": "hmeq", "network": "NET_JOB", "rho_qc": "1.0", "levels_q": "7"},
            {"dataset": "hmeq", "network": "NET_2L", "rho_qc": "0.4321", "levels_q": "20"},
        ])
        _write_csv(self.results / "hmeq" / "T_eta.csv", [
            {"dataset": "hmeq", "network": "NET_2L", "eta": "0.0", "rho_qc": "0.2"},
            {"dataset": "hmeq", "network": "NET_2L", "eta": "0.05", "rho_qc": "0.1"},
            {"dataset": "hmeq", "network": "NET_2L", "eta": "1.0", "rho_qc": "0.4"},
        ])
        _write_csv(self.results / "hmeq" / "T3_variants.csv", [
            {"dataset": "hmeq", "network": "NET_2L", "variant": "primary", "route": "D_bar"},
            {"dataset": "hmeq", "network": "NET_2L", "variant": "rank_one_restart", "route": "H"},
            {"dataset": "hmeq", "network": "NET_2L", "variant": "eta_0", "route": ""},
        ])
        # A file under the fmac scope that wrongly holds HMEQ rows: the A1 defect.
        _write_csv(self.results / "fmac" / "T1_data.csv", [{"dataset": "hmeq", "key": "n_borrowers", "value": "5853"}])
        (self.pipeline / "04_quantum").mkdir(parents=True)
        (self.pipeline / "04_quantum" / "convergence_NET_2L.json").write_text(json.dumps({"R": 116, "reduction": {"n_full": 11714}}))
        for stage, commit in (("01_data", "abc"), ("03_classical", "abc")):
            (self.pipeline / stage).mkdir(parents=True, exist_ok=True)
            (self.pipeline / stage / "manifest.json").write_text(json.dumps(
                {"git": {"commit": commit, "dirty": True}, "source_sha256": "f" * 64, "versions": {"numpy": "2.4.2"}}))

    def tearDown(self):
        self._tmp.cleanup()

    def test_csv_json_aggregate_and_missing(self):
        one = resolve(KeySpec("hmeq.2l.rho_qc", "result:hmeq/T2_agreement", "rho_qc", "corr", {"dataset": "hmeq", "network": "NET_2L"}, dataset="hmeq"),
                      self.results, self.pipeline)
        self.assertEqual((one["value"], one["printed"], one["status"]), (0.4321, "0.432", "ok"))
        agg = resolve(KeySpec("hmeq.-.rho_min", "result:hmeq/T2_agreement", "rho_qc", "corr", {"dataset": "hmeq"}, aggregate="min", dataset="hmeq"),
                      self.results, self.pipeline)
        self.assertEqual(agg["printed"], "0.432")
        at_min = resolve(KeySpec("hmeq.2l.eta_at_min", "result:hmeq/T_eta", "eta", "real", {"dataset": "hmeq"}, aggregate="argmin", by="rho_qc", dataset="hmeq"),
                         self.results, self.pipeline)
        self.assertEqual(at_min["printed"], "0.050")
        ambiguous = resolve(KeySpec("hmeq.-.x", "result:hmeq/T2_agreement", "rho_qc", "corr", None, dataset="hmeq"), self.results, self.pipeline)
        self.assertIsNone(ambiguous["printed"])
        self.assertIn("rows match", ambiguous["status"])
        js = resolve(KeySpec("hmeq.2l.N_full", "stage:04_quantum/convergence_NET_2L.json", "reduction.n_full", "count"), self.results, self.pipeline)
        self.assertEqual(js["printed"], "11{,}714")
        missing = resolve(KeySpec("hmeq.-.x", "result:hmeq/T_missing", "a", "corr", dataset="hmeq"), self.results, self.pipeline)
        self.assertEqual((missing["value"], missing["status"]), (None, "missing file"))
        no_col = resolve(KeySpec("hmeq.2l.x", "result:hmeq/T2_agreement", "nope", "corr", {"network": "NET_2L"}, dataset="hmeq"), self.results, self.pipeline)
        self.assertIsNone(no_col["printed"])

    def test_a_route_key_prints_the_discriminant_the_router_recorded_and_never_a_guess(self):
        def route(variant):
            return resolve(KeySpec(f"hmeq.2l.route_{variant}", "result:hmeq/T3_variants", "route", "route",
                                   {"dataset": "hmeq", "network": "NET_2L", "variant": variant}, dataset="hmeq"), self.results, self.pipeline)

        primary, rank_one = route("primary"), route("rank_one_restart")
        self.assertEqual((primary["value"], primary["printed"], primary["status"]), ("D_bar", r"$\bar{D}$", "ok"))
        self.assertEqual((rank_one["value"], rank_one["printed"]), ("H", "$H$"))
        blank, absent = route("eta_0"), route("flattened")
        self.assertIsNone(blank["printed"], "a row whose route is blank leaves the macro undefined")
        self.assertEqual((absent["printed"], absent["status"]), (None, "no row matches the filter"))

    def test_a_key_may_not_read_another_datasets_rows(self):
        with self.assertRaises(AssertionError):
            resolve(KeySpec("fmac.-.n_borrowers", "result:fmac/T1_data", "value", "count", {"dataset": "fmac", "key": "n_borrowers"}, dataset="fmac"),
                    self.results, self.pipeline)

    def test_manifests_agree_or_fail(self):
        info = read_manifests(self.pipeline)
        self.assertEqual(info["commit"], "abc")
        commit = resolve(KeySpec("all.-.commit", "manifests", "commit", "text"), self.results, self.pipeline)
        self.assertEqual(commit["printed"], "abc+uncommitted:" + "f" * 12)
        numpy = resolve(KeySpec("all.-.numpy", "manifests", "versions.numpy", "text"), self.results, self.pipeline)
        self.assertEqual(numpy["printed"], "2.4.2")
        (self.pipeline / "03_classical" / "manifest.json").write_text(json.dumps({"git": {"commit": "def", "dirty": False}, "versions": {}}))
        with self.assertRaises(AssertionError):
            resolve(KeySpec("all.-.commit", "manifests", "commit", "text"), self.results, self.pipeline)

    def test_write_number_map(self):
        manuscript = Path(self._tmp.name) / "manuscript"
        (self.results / "fmac" / "T1_data.csv").unlink()  # the deliberately wrong file would (rightly) raise
        report = write_number_map(self.results, self.pipeline, manuscript)
        tex = (manuscript / "numbers.tex").read_text()
        self.assertIn(r"\expandafter\def\csname qppr@hmeq.2l.rho_qc\endcsname{0.432}", tex)
        self.assertIn(r"\expandafter\def\csname qppr@hmeq.2l.route_primary\endcsname{$\bar{D}$}", tex)
        self.assertIn(r"\expandafter\def\csname qppr@hmeq.2l.route_rankone\endcsname{$H$}", tex)
        self.assertNotIn("qppr@hmeq.2l.route_eta0\\endcsname", tex, "a blank route defines no macro")
        self.assertGreater(len(report["unresolved"]), 0, "keys without a source stay unresolved rather than invented")
        self.assertTrue(all(r["printed"] is None for r in report["unresolved"]))
        self.assertTrue((manuscript / "number_map.csv").exists())
        with open(manuscript / "number_map.csv", newline="") as handle:
            units = {row["key"]: row["unit"] for row in csv.DictReader(handle)}
        self.assertEqual((units["hmeq.-.t_limit_full"], units["hmeq.-.mem_full"], units["hmeq.2l.rho_qc"]), ("s", "MiB", ""),
                         "the key list states the unit of a time and of a memory")


class AggregateTests(unittest.TestCase):
    """The keys that aggregate keys, resolved on a hand-written results tree (the six networks, the four single-layer ones, both datasets)."""

    NETWORKS = {"hmeq": ("NET_JOB", "NET_REASON", "NET_2L"), "fmac": ("NET_FMAC_POSTAL", "NET_FMAC_SELLER", "NET_FMAC_2L")}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results, self.pipeline = Path(self._tmp.name) / "results", Path(self._tmp.name) / "pipeline"
        self.keys = {s.key: s for s in key_specs()}
        compat = {"NET_JOB": "2e-16", "NET_REASON": "5e-16", "NET_2L": "1e-15", "NET_FMAC_POSTAL": "3e-16", "NET_FMAC_SELLER": "4e-16", "NET_FMAC_2L": "8e-16"}
        for ds, networks in self.NETWORKS.items():
            _write_csv(self.results / ds / "T_convergence.csv", [
                {"dataset": ds, "network": n, "compatibility_residual": compat[n], "lumpability_residual": "1e-14", "n_slots": "16" if ds == "hmeq" else "86"}
                for n in networks])
            _write_csv(self.results / ds / "EC_registers.csv", [
                {"dataset": ds, "network": n, "rho_src_dst": "0.99" if ds == "hmeq" else "0.97"} for n in networks])
            _write_csv(self.results / ds / "T2_variability.csv", [
                {"dataset": ds, "network": n, "n_splits": "50", "n_resamples": "50", "n_single_identical_" + networks[0]: "24",
                 "resample_n_single_identical_" + networks[0]: "24", "n_single_identical_" + networks[1]: "47",
                 "resample_n_single_identical_" + networks[1]: "47"} for n in networks])
            _write_csv(self.results / ds / "T6_hardware_header.csv", [{"dataset": ds, "hw_backend": "FakeMarrakesh", "hw_date": "2025-02-26"}])
            stage = self.pipeline / get_dataset(ds).stage("mechanism")
            stage.mkdir(parents=True)
            (stage / "mechanism_summary.json").write_text(json.dumps({"single_layer": {
                n: {"prop3_max_relerr": 1.5e-13 if n == networks[0] else 7.4e-15, "n_equal_share_pairs": 2 if n == networks[1] else 0, "levels_q_minus_c": 0}
                for n in networks[:2]}}))

    def tearDown(self):
        self._tmp.cleanup()

    def _resolve(self, key):
        return resolve(self.keys[key], self.results, self.pipeline)

    def test_the_largest_the_smallest_and_the_common_value(self):
        self.assertEqual(self._resolve("all.-.compat_resid_max")["printed"], r"$1.00\times10^{-15}$", "the largest of the six primary operators")
        self.assertEqual(self._resolve("all.-.lump_resid_max")["printed"], r"$1.00\times10^{-14}$")
        self.assertEqual(self._resolve("all.-.prop3_max_relerr")["printed"], r"$1.50\times10^{-13}$", "the largest over the four single-layer networks")
        self.assertEqual(self._resolve("all.-.rho_src_dst_min")["printed"], "0.970", "the smallest over the two datasets' two-layer networks")
        self.assertEqual((self._resolve("all.-.hw_backend")["printed"], self._resolve("all.-.hw_date")["printed"]), ("FakeMarrakesh", "2025-02-26"))
        self.assertEqual(self._resolve("hmeq.-.n_splits")["printed"], "50")
        self.assertEqual((self._resolve("hmeq.-.n_slots")["printed"], self._resolve("fmac.-.n_slots")["printed"]), ("16", "86"))

    def test_parts_that_must_agree_and_do_not_are_an_error_and_an_unresolved_part_is_not_guessed(self):
        _write_csv(self.results / "fmac" / "T6_hardware_header.csv", [{"dataset": "fmac", "hw_backend": "FakeBrisbane", "hw_date": "2025-02-26"}])
        with self.assertRaises(AssertionError):
            self._resolve("all.-.hw_backend")
        self.assertEqual(self._resolve("all.-.hw_date")["printed"], "2025-02-26", "the date still agrees")
        _write_csv(self.results / "hmeq" / "T2_variability.csv", [
            {"dataset": "hmeq", "network": n, "n_splits": s} for n, s in zip(self.NETWORKS["hmeq"], ("50", "50", "49"))])
        with self.assertRaises(AssertionError):
            self._resolve("hmeq.-.n_splits")
        (self.results / "fmac" / "EC_registers.csv").unlink()
        unresolved = self._resolve("all.-.rho_src_dst_min")
        self.assertEqual((unresolved["value"], unresolved["printed"]), (None, None))
        self.assertIn("is unresolved", unresolved["status"])

    def test_single_layer_counts_belong_to_their_network(self):
        self.assertEqual((self._resolve("hmeq.job.n_single_identical")["printed"], self._resolve("hmeq.reason.n_single_identical")["printed"]), ("24", "47"))
        self.assertEqual((self._resolve("fmac.postal.n_single_identical")["printed"], self._resolve("fmac.seller.n_single_identical")["printed"]), ("24", "47"))
        self.assertEqual((self._resolve("hmeq.job.n_equal_share_pairs")["printed"], self._resolve("hmeq.reason.n_equal_share_pairs")["printed"]), ("0", "2"))
        self.assertEqual(self._resolve("fmac.seller.n_equal_share_pairs")["printed"], "2")


class ContractTests(unittest.TestCase):
    def test_every_key_obeys_the_grammar_and_names_a_known_source(self):
        specs = key_specs()
        keys = [s.key for s in specs]
        self.assertEqual(len(keys), len(set(keys)))
        for spec in specs:
            self.assertRegex(spec.key, KEY_GRAMMAR, spec.key)
            if spec.source.startswith("stage:"):
                self.assertTrue(any(part in spec.source for part in KNOWN_STAGE_FILES), spec.source)
            elif spec.source.startswith("result:"):
                scope, _, name = spec.source[len("result:"):].partition("/")
                self.assertIn(scope, ("hmeq", "fmac"), spec.source)
                self.assertIn(name, KNOWN_RESULTS, spec.source)
                self.assertEqual(spec.dataset, scope, spec.key)
            else:
                self.assertIn(spec.source, ("manifests", "keys"), spec.source)
        self.assertFalse(any(re.search(r"ibm", k) for k in keys), "no IBM key may exist in the code")
        for required in ("hmeq.2l.rho_qc", "fmac.2l.tau_star", "hmeq.-.n_borrowers", "hmeq.2l.d_prauc_lo", "fmac.2l.pval_prauc_q",
                         "synth.-.s3_rho_m30_hmeq", "all.-.commit", "hmeq.-.hw_fake_rho", "hmeq.2l.reduction_factor", "hmeq.2l.n_slots",
                         "hmeq.2l.angles", "hmeq.2l.gbar_density", "hmeq.2l.eta_at_min", "hmeq.-.layerdummy_importance_of",
                         "fmac.2l.rho_qc_lo", "fmac.postal.n_single_identical", "hmeq.reason.n_single_identical", "all.-.aer", "all.-.runtime_version",
                         "hmeq.2l.route_primary", "hmeq.2l.route_rankone", "fmac.2l.route_flat", "fmac.2l.route_labelfree",
                         "hmeq.-.hw_rankable_k", "hmeq.-.hw_rankable_m", "fmac.-.hw_rankable_borrowers", "fmac.-.hw_rankable_fr",
                         "hmeq.-.noise_k_band", "fmac.-.noise_tau_band",
                         "all.-.compat_resid_max", "all.-.lump_resid_max", "all.-.prop3_max_relerr", "all.-.rho_src_dst_min", "all.-.hw_backend",
                         "all.-.hw_date", "hmeq.-.n_splits", "hmeq.-.n_slots", "fmac.-.n_slots", "hmeq.job.n_equal_share_pairs",
                         "hmeq.reason.n_equal_share_pairs", "fmac.postal.n_equal_share_pairs", "fmac.seller.n_equal_share_pairs"):
            self.assertIn(required, keys)
        self.assertEqual([k for k in keys if k.endswith(".equal_share_pairs")], [], "the list of pairs stays in the mechanism summary; the number map prints its count")
        self.assertEqual([k for k in keys if re.search(r"n_single_identical_\w+$", k)], [], "the re-split count belongs to the single-layer network")
        by_key = {s.key: s for s in specs}
        self.assertEqual(by_key["fmac.2l.rho_qc_lo"].column, "boot_rho_qc_lo", "Freddie Mac intervals are the bootstrap over test loans")
        self.assertEqual(by_key["hmeq.2l.rho_qc_lo"].column, "rho_qc_lo", "HMEQ percentiles are over its re-splits")
        self.assertEqual(by_key["fmac.2l.rho_qc_med"].column, "resample_rho_qc_med")
        self.assertEqual(by_key["fmac.2l.n_splits"].column, "n_resamples")
        self.assertEqual(by_key["hmeq.2l.n_splits"].column, "n_splits")
        self.assertEqual(by_key["fmac.postal.n_single_identical"].column, "resample_n_single_identical_NET_FMAC_POSTAL")
        self.assertEqual(by_key["fmac.postal.n_single_identical"].row_filter, {"dataset": "fmac", "network": "NET_FMAC_POSTAL"})
        self.assertEqual(by_key["hmeq.job.n_single_identical"].column, "n_single_identical_NET_JOB")
        self.assertEqual((by_key["hmeq.job.n_equal_share_pairs"].column, by_key["hmeq.job.n_equal_share_pairs"].kind),
                         ("single_layer.NET_JOB.n_equal_share_pairs", "count"))
        self.assertEqual({by_key[k].kind for k in by_key if k.rpartition(".")[2].startswith(("t_step", "t_limit", "t_class"))}, {"seconds"})
        self.assertEqual({by_key[k].kind for k in by_key if k.rpartition(".")[2].startswith("mem_")}, {"mb"})
        six = [f"{ds}.{net}.compat_resid_max" for ds in ("hmeq", "fmac") for net in (("job", "reason", "2l") if ds == "hmeq" else ("postal", "seller", "2l"))]
        for key, combine, parts in (("all.-.compat_resid_max", "max", six), ("all.-.lump_resid_max", "max", [p.replace("compat", "lump") for p in six]),
                                    ("all.-.prop3_max_relerr", "max", ["hmeq.job.prop3_max_relerr", "hmeq.reason.prop3_max_relerr",
                                                                       "fmac.postal.prop3_max_relerr", "fmac.seller.prop3_max_relerr"]),
                                    ("all.-.rho_src_dst_min", "min", ["hmeq.2l.rho_src_dst_min", "fmac.2l.rho_src_dst_min"]),
                                    ("all.-.hw_backend", "same", ["hmeq.-.hw_backend", "fmac.-.hw_backend"]),
                                    ("all.-.hw_date", "same", ["hmeq.-.hw_date", "fmac.-.hw_date"]),
                                    ("hmeq.-.n_splits", "same", ["hmeq.job.n_splits", "hmeq.reason.n_splits", "hmeq.2l.n_splits"]),
                                    ("hmeq.-.n_slots", "same", ["hmeq.2l.n_slots"]), ("fmac.-.n_slots", "same", ["fmac.2l.n_slots"])):
            self.assertEqual((by_key[key].source, by_key[key].combine, list(by_key[key].parts)), ("keys", combine, parts), key)
        self.assertNotIn("fmac.-.n_splits", keys, "only the number of HMEQ re-splits is a single number the manuscript needs")
        self.assertIn("hmeq.-.split_seed", keys)
        self.assertNotIn("fmac.-.split_seed", keys, "a calendar split draws nothing and has no seed")
        routes = [s for s in specs if s.key.rpartition(".")[2].startswith("route_")]
        self.assertEqual({s.kind for s in routes}, {"route"})
        self.assertEqual({s.source.rpartition("/")[2] for s in routes}, {"T3_variants", "T4_partial"})
        self.assertEqual(by_key["hmeq.2l.route_rankone"].row_filter["variant"], "rank_one_restart")
        self.assertEqual(by_key["fmac.2l.route_labelfree"].column, "route_labelfree")
        for key, kind in (("hw_rankable_k", "count"), ("hw_rankable_m", "int"), ("hw_rankable_borrowers", "count"), ("hw_rankable_fr", "corr"),
                          ("noise_k_band", "text"), ("noise_tau_band", "text")):
            spec = by_key[f"fmac.-.{key}"]
            self.assertEqual((spec.source, spec.column, spec.kind), ("result:fmac/T6_hardware_header", key, kind), key)
        self.assertNotIn("hmeq.-.hw_rankable_signal", keys, "a boolean is not a number the manuscript prints")
        self.assertLessEqual({s.kind for s in specs}, set(KINDS))
        self.assertLessEqual({ROUTE_DBAR, ROUTE_H}, set(ROUTE_LABELS), "every route the router can record has a printed form")

    @unittest.skipUnless(RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()), "outputs/results is absent; nothing to resolve")
    def test_every_key_resolves_on_real_outputs(self):
        reason = outputs_not_of_this_source()
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as td:
            report = write_number_map(RESULTS_DIR, PIPELINE_DIR, Path(td))
        unresolved = [f"{r['key']}: {r['status']} ({r['source']})" for r in report["unresolved"]]
        self.assertEqual(unresolved, [], "\n".join(unresolved))


if __name__ == "__main__":
    unittest.main()
