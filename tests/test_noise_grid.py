"""The circuit tier's pure functions, without a transpiler or Aer.

Pinned here, because a number the grid reports depends on each of them:

- the grading of a simulated arm against the exact distribution, and the masked-class
  Spearman, which ranks class probabilities and not unfolded per-member shares;
- the pure-noise band: per instance, reproducible, and wide enough that a fresh
  pure-noise output scored against the same exact average lands inside it;
- where the raw output of an arm stops carrying signal (``noise_k_band``, ``noise_tau_band``): one-sided and
  strict on the fidelity ratio, each instance judged at its longest simulated walk, ``tau = 1`` never used;
  the first instance able to rank test borrowers;
- the candidate selection (rankable instance above the fidelity floor, ``tau >= hardware_min_tau``,
  the QPU budget and the width cap) and the analytic two-qubit count;
- the simulation budget (memory is deterministic, time scales with gates and state size from an
  assumed throughput until an arm has run) and the shard identity (a stale shard is refused,
  every cell of the grid ends with a row);
- the per-circuit readout calibration: circuits are grouped by their measured-qubit tuple and each
  one is mitigated with the set of its own tuple.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from config import DEFAULT_CONFIG
from paths import PIPELINE_DIR, RESULTS_DIR
from quantum.hardware import shards
from quantum.hardware.aer import mitigated_omega_by_circuit
from quantum.hardware.mitigation import PerQubitCalibration, calibration_groups
from quantum.hardware.noise_grading import (
    NOT_SIMULATED,
    annotate_signal,
    carries_signal,
    first_rankable_instance,
    first_tau_without_signal,
    grade,
    inside_band,
    judged_cells,
    masked_class_spearman,
    noise_entry_points,
    pure_noise_band,
    pure_noise_outputs,
    signal_above_band,
    supra_metrics,
)
from quantum.hardware.noise_grid import (
    SimulationBudget,
    analytic_two_qubit_count,
    masked_groups,
    select_candidate,
    state_bytes,
)
from quantum.lumping import UNKNOWN_LABEL

CONFIG = DEFAULT_CONFIG
DECIMALS = CONFIG.tie_tolerance


def _nodes(levels, sizes=None):
    """One borrower copy per layer for each class (``levels`` are the masked labels, ``sizes`` the member counts), then a category node."""
    sizes = list(sizes) if sizes is not None else [2] * len(levels)
    rows, first = [], 0
    for g, (level, size) in enumerate(zip(levels, sizes)):
        members = ";".join(str(first + i) for i in range(size))
        first += size
        for layer_index, layer in enumerate(("JOB", "REASON")):
            rows.append({"node_id": layer_index * len(levels) + g, "node_type": "borrower_copy", "layer": layer, "layer_index": layer_index,
                         "member_borrower_ids": members, "label": f"{layer}:group_{g}", "BAD": 1.0 if level == 1 else 0.0,
                         "BAD_masked": level, "weight": float(size)})
    rows.append({"node_id": 2 * len(levels), "node_type": "category", "layer": "JOB", "layer_index": 0, "member_borrower_ids": None,
                 "label": "JOB:c", "BAD": 0.0, "BAD_masked": 0, "weight": None})
    return pd.DataFrame(rows)


class GradeTests(unittest.TestCase):
    def test_uniform_output_scores_zero_fidelity_and_a_nan_supra_spearman(self):
        nodes = _nodes([1, 0, UNKNOWN_LABEL, UNKNOWN_LABEL])
        R = len(nodes)
        exact = np.random.default_rng(0).random(R)
        exact /= exact.sum()
        uniform = np.full(R, 1.0 / R)
        g = grade(uniform, exact, nodes, DECIMALS)
        self.assertTrue(np.isnan(g["spearman_supra"]))
        self.assertAlmostEqual(g["fidelity_ratio"], 0.0, places=12)
        self.assertAlmostEqual(g["tv"], g["tv_uniform"], places=12)
        perfect = grade(exact, exact, nodes, DECIMALS)
        self.assertEqual(perfect["spearman_supra"], 1.0)
        self.assertAlmostEqual(perfect["fidelity_ratio"], 1.0, places=12)
        self.assertEqual(masked_groups(nodes), 2)

    def test_masked_class_spearman_ranks_class_probabilities_not_unfolded_shares(self):
        nodes = _nodes([1, UNKNOWN_LABEL, UNKNOWN_LABEL, UNKNOWN_LABEL, 0])      # positions: two copies per class, then the category
        exact = np.array([0.20, 0.10, 0.15, 0.05, 0.06, 0.04, 0.03, 0.02, 0.05, 0.05, 0.25])
        self.assertAlmostEqual(float(exact.sum()), 1.0, places=12)
        # the same class probabilities with the mass split differently between a class's two layer copies
        resplit = exact.copy()
        for first in (2, 4, 6):
            resplit[first], resplit[first + 1] = exact[first + 1], exact[first]
        self.assertAlmostEqual(masked_class_spearman(resplit, exact, nodes, DECIMALS), 1.0, places=12)
        self.assertLess(supra_metrics(resplit, exact)["spearman_supra"], 1.0)
        # the largest and the smallest masked class exchange their mass: the class order reverses
        reversed_order = exact.copy()
        reversed_order[2:4], reversed_order[6:8] = exact[6:8], exact[2:4]
        self.assertAlmostEqual(masked_class_spearman(reversed_order, exact, nodes, DECIMALS), -1.0, places=12)
        # fewer than two masked classes: nothing to rank
        self.assertTrue(np.isnan(masked_class_spearman(exact[:7], exact[:7], _nodes([1, UNKNOWN_LABEL, 0]), DECIMALS)))
        # classes of different sizes and an output with no structure: no spurious order (a per-member share would follow 1 / size)
        sized = _nodes([UNKNOWN_LABEL] * 3, sizes=[1, 4, 2])
        flat = np.full(len(sized), 1.0 / len(sized))
        self.assertTrue(np.isnan(masked_class_spearman(flat, np.random.default_rng(1).random(len(sized)), sized, DECIMALS)))


class PureNoiseBandTests(unittest.TestCase):
    SHOTS, DRAWS, TAUS = 4096, 400, (1, 2, 3)

    def _exact(self, R, seed):
        rng = np.random.default_rng(seed)
        out = {}
        for tau in self.TAUS:
            vector = rng.random(R) + 0.05
            out[tau] = vector / vector.sum()
        return out

    def _band(self, R, m, seed=11, draws=None):
        return pure_noise_band(self._exact(R, 0), m=m, shots=self.SHOTS, draws=draws or self.DRAWS, quantiles=(5.0, 95.0), seed=seed)

    def test_a_fresh_pure_noise_output_lands_inside_its_own_band(self):
        R, m = 20, 5
        exact = self._exact(R, 0)
        band = self._band(R, m)
        fresh = pure_noise_outputs(R, m, self.SHOTS, max(self.TAUS), self.DRAWS, np.random.default_rng(12))      # not the draws of the band
        for tau in self.TAUS:
            b = band[tau]
            self.assertLessEqual(b["spearman_lo"], b["spearman_median"])
            self.assertLessEqual(b["spearman_median"], b["spearman_hi"])
            self.assertLessEqual(b["fidelity_lo"], b["fidelity_hi"])
            graded = [supra_metrics(fresh[d, tau - 1], exact[tau]) for d in range(self.DRAWS)]
            rho = np.mean([inside_band(g["spearman_supra"], b["spearman_lo"], b["spearman_hi"]) for g in graded])
            fid = np.mean([inside_band(g["fidelity_ratio"], b["fidelity_lo"], b["fidelity_hi"]) for g in graded])
            # a 5-95 band holds about nine in ten fresh draws; the margin is many standard errors
            self.assertGreater(rho, 0.75)
            self.assertGreater(fid, 0.75)

    def test_the_band_is_per_instance_reproducible_and_wider_for_fewer_supra_nodes(self):
        small, large = self._band(6, 3), self._band(40, 6)
        for tau in self.TAUS:
            self.assertGreater(small[tau]["spearman_hi"] - small[tau]["spearman_lo"], large[tau]["spearman_hi"] - large[tau]["spearman_lo"])
        self.assertEqual(self._band(20, 5, seed=3, draws=60), self._band(20, 5, seed=3, draws=60))
        self.assertNotEqual(self._band(20, 5, seed=3, draws=60), self._band(20, 5, seed=4, draws=60))
        self.assertTrue(all(small[t]["draws"] == self.DRAWS for t in self.TAUS))

    def test_pure_noise_outputs_are_distributions_over_the_retained_support(self):
        out = pure_noise_outputs(5, 3, 256, 4, 7, np.random.default_rng(0))      # 5 supra-nodes in a 3-qubit register: three padding codes
        self.assertEqual(out.shape, (7, 4, 5))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-12)
        with self.assertRaises(ValueError):
            pure_noise_outputs(9, 3, 256, 1, 2, np.random.default_rng(0))
        self.assertFalse(inside_band(float("nan"), -1.0, 1.0))
        self.assertFalse(inside_band(None, -1.0, 1.0))
        self.assertTrue(inside_band(0.5, 0.0, 1.0))


def _cell(k, tau, fr, *, arm="fake_marrakesh", mitigated=False, status="", dataset="hmeq", lo=-0.02, hi=0.02, groups=1, m=3, borrowers=100):
    """A grid row with the columns the signal rule reads: the fidelity ratio and the edges of its own pure-noise band."""
    return {"dataset": dataset, "k_classes": k, "tau": tau, "arm": arm, "mitigated": mitigated, "status": status, "fidelity_ratio": fr,
            "pure_noise_fidelity_lo": lo, "pure_noise_fidelity_hi": hi, "n_masked_groups": groups, "m": m, "borrowers": borrowers}


class SignalRuleTests(unittest.TestCase):
    def test_signal_is_one_sided_and_strict(self):
        self.assertFalse(signal_above_band(0.02, 0.02), "a value on the upper edge is not signal")
        self.assertTrue(signal_above_band(0.0201, 0.02))
        self.assertFalse(signal_above_band(0.0, 0.02), "inside the band")
        self.assertFalse(signal_above_band(-0.5, 0.02), "far below the band is not signal either")
        for value, hi in ((float("nan"), 0.02), (None, 0.02), (0.5, float("nan")), (0.5, None), ("text", 0.02)):
            self.assertFalse(signal_above_band(value, hi), (value, hi))
        self.assertTrue(carries_signal(_cell(2, 5, 0.766, hi=0.042)))
        self.assertFalse(carries_signal(_cell(2, 5, 0.042, hi=0.042)))
        self.assertFalse(carries_signal(_cell(2, 5, -0.4, lo=-0.02, hi=0.042)), "a value below the lower edge is not signal")
        self.assertFalse(carries_signal(_cell(2, 5, None)))

    def test_annotate_signal_marks_raw_simulated_cells_only_and_drops_the_shard_s_spearman_flag(self):
        rows = [{**_cell(2, 2, 0.5), "inside_pure_noise_band": True}, _cell(2, 2, 0.5, mitigated=True),
                _cell(2, 3, None, status=NOT_SIMULATED), {**_cell(3, 2, 0.01), "inside_pure_noise_band": False}]
        out = annotate_signal(rows)
        self.assertEqual([r["signal_above_band"] for r in out], [True, None, None, False])
        self.assertTrue(all("inside_pure_noise_band" not in r for r in out))
        self.assertIn("inside_pure_noise_band", rows[0], "the input rows (the shards' own) are left as they are")
        self.assertNotIn("signal_above_band", rows[0])


class EntryPointTests(unittest.TestCase):
    @staticmethod
    def hmeq_shaped():
        """Signal at k = 2, 3, 4, 6 at the longest walk; none at 5 and 7..12; tau = 1 shows signal everywhere; k > 12 only at tau = 1."""
        rows = []
        for k in range(2, 13):
            for tau in range(1, 6):
                rows.append(_cell(k, tau, 0.3 if (k in (2, 3, 4, 6) or tau == 1) else 0.0))
        for k in (14, 16, 18):
            rows.append(_cell(k, 1, 0.9))
            rows += [_cell(k, tau, None, status=NOT_SIMULATED) for tau in (2, 3, 4, 5)]
        rows += [_cell(5, 5, 0.9, mitigated=True), _cell(5, 5, 0.9, arm="aer_noise_model"), _cell(5, 5, 0.9, arm="noiseless_shots")]
        return rows

    def test_the_hmeq_shaped_pattern_gives_noise_k_band_seven(self):
        found = noise_entry_points(self.hmeq_shaped(), min_tau=2, selected_k=2)
        self.assertEqual(found["noise_k_band"], 7)
        self.assertEqual(found["evaluated_k"], list(range(2, 13)), "k > 12 has no simulated cell at tau >= 2")
        self.assertEqual(found["k_with_signal"], [2, 3, 4, 6])
        self.assertEqual(set(found["judged_tau"].values()), {5}, "every k is judged at its longest walk, never at tau = 1")

    def test_a_noise_k_below_a_signal_k_does_not_count(self):
        below = [_cell(2, 5, 0.0), _cell(3, 5, 0.3), _cell(4, 5, 0.0)]
        self.assertEqual(noise_entry_points(below, min_tau=2)["noise_k_band"], 4)
        self.assertEqual(noise_entry_points(below[:2], min_tau=2)["noise_k_band"], "none", "the largest evaluated k carries signal")
        self.assertEqual(noise_entry_points([_cell(k, 5, 0.3) for k in (2, 3, 4)], min_tau=2)["noise_k_band"], "none", "every evaluated k carries signal")
        self.assertEqual(noise_entry_points([_cell(k, 5, 0.0) for k in (2, 3, 4)], min_tau=2)["noise_k_band"], 2, "no k carries signal")
        empty = noise_entry_points([], min_tau=2)
        self.assertEqual((empty["noise_k_band"], empty["noise_tau_band"], empty["evaluated_k"]), ("none", "not evaluated", []))

    def test_tau_one_is_never_judged_and_only_the_longest_walk_of_each_k_counts(self):
        only_preparation = [_cell(k, 1, 0.9) for k in (2, 3, 4)]
        found = noise_entry_points(only_preparation, min_tau=2)
        self.assertEqual((found["evaluated_k"], found["noise_k_band"]), ([], "none"))
        rows = [_cell(3, 2, 0.3), _cell(3, 3, 0.3), _cell(3, 4, 0.0),                      # signal early, none at the longest walk
                _cell(4, 2, 0.0), _cell(4, 5, 0.3)]                                       # none early, signal at the longest walk
        judged = judged_cells(rows, arm="fake_marrakesh", min_tau=2)
        self.assertEqual({k: int(r["tau"]) for k, r in judged.items()}, {3: 4, 4: 5})
        self.assertEqual(noise_entry_points(rows, min_tau=2)["k_with_signal"], [4])
        self.assertEqual(set(judged_cells(rows, arm="fake_marrakesh", min_tau=5)), {4}, "a larger min_tau drops the shorter instances")
        self.assertEqual(judged_cells(rows, arm="fake_marrakesh", min_tau=6), {})
        undefined = [_cell(2, 5, None), _cell(3, 5, float("nan")), _cell(4, 5, 0.3, hi=None)]
        self.assertEqual(noise_entry_points(undefined, min_tau=2)["evaluated_k"], [], "an undefined fidelity ratio or band is not evaluated")

    def test_noise_tau_band_is_the_first_step_without_signal_or_the_largest_simulated_step(self):
        always = [_cell(2, tau, 0.5) for tau in range(1, 6)]
        self.assertEqual(noise_entry_points(always, min_tau=2, selected_k=2)["noise_tau_band"], "> 5")
        self.assertEqual(first_tau_without_signal(always, min_tau=2), "> 5")
        mixed = [_cell(2, 1, 0.0), _cell(2, 2, 0.5), _cell(2, 3, 0.0), _cell(2, 4, 0.5)]
        self.assertEqual(first_tau_without_signal(mixed, min_tau=2), 3, "tau = 1 shows no signal here, but it is never used")
        self.assertEqual(first_tau_without_signal([_cell(2, 1, 0.0), _cell(2, 2, 0.5), _cell(2, 3, 0.5)], min_tau=2), "> 3",
                         "an instance simulated for fewer steps names the largest step it was simulated for")
        self.assertEqual(first_tau_without_signal([_cell(2, 1, 0.5), _cell(2, 2, None), _cell(2, 3, float("nan"))], min_tau=2), "not evaluated")
        self.assertEqual(first_tau_without_signal([_cell(2, 1, 0.5), _cell(2, 2, 0.5, status=NOT_SIMULATED)], min_tau=2), "not evaluated")
        rows = always + [_cell(3, 2, 0.0), _cell(2, 2, 0.0, arm="aer_noise_model"), _cell(2, 2, 0.0, mitigated=True)]
        self.assertEqual(noise_entry_points(rows, min_tau=2, selected_k=2)["noise_tau_band"], "> 5", "another arm and a mitigated row do not count")
        self.assertEqual(noise_entry_points(rows, arm="aer_noise_model", min_tau=2, selected_k=2)["noise_tau_band"], 2)
        self.assertEqual(noise_entry_points(rows, min_tau=2)["noise_tau_band"], "not evaluated", "without a selected instance")


class RankableInstanceTests(unittest.TestCase):
    def test_the_first_instance_with_two_masked_classes_and_its_best_raw_fidelity(self):
        rows = [_cell(2, 2, 0.9, groups=1), _cell(3, 2, 0.8, groups=1),
                _cell(4, 1, 0.9, groups=2, m=4, borrowers=300),                              # tau = 1 is the state preparation: ignored
                _cell(4, 2, 0.03, groups=2, m=4, borrowers=300), _cell(4, 3, 0.01, groups=2, m=4, borrowers=300),
                _cell(4, 2, 0.5, groups=2, m=4, borrowers=300, mitigated=True), _cell(4, 2, 0.6, groups=2, m=4, borrowers=300, arm="aer_noise_model"),
                _cell(5, 2, 0.4, groups=3, m=5, borrowers=400)]
        found = first_rankable_instance(rows, min_tau=2)
        self.assertEqual(found, {"hw_rankable_k": 4, "hw_rankable_m": 4, "hw_rankable_borrowers": 300, "hw_rankable_fr": 0.03,
                                 "hw_rankable_signal": True})
        below = first_rankable_instance([_cell(4, 2, 0.01, groups=2), _cell(4, 3, 0.005, groups=2)], min_tau=2)
        self.assertEqual((below["hw_rankable_fr"], below["hw_rankable_signal"]), (0.01, False))
        tie = first_rankable_instance([_cell(4, 2, 0.03, groups=2, hi=0.05), _cell(4, 4, 0.03, groups=2, hi=0.01)], min_tau=2)
        self.assertEqual((tie["hw_rankable_fr"], tie["hw_rankable_signal"]), (0.03, True), "equal fidelity ratios go to the longer walk")

    def test_nothing_is_invented_without_a_rankable_instance_or_a_simulated_cell(self):
        none = {"hw_rankable_k": None, "hw_rankable_m": None, "hw_rankable_borrowers": None, "hw_rankable_fr": None, "hw_rankable_signal": None}
        self.assertEqual(first_rankable_instance([_cell(2, 2, 0.9, groups=1), _cell(3, 2, 0.9, groups=0)], min_tau=2), none)
        self.assertEqual(first_rankable_instance([], min_tau=2), none)
        wide = [_cell(30, 2, None, groups=3, m=7, borrowers=900, status=NOT_SIMULATED)]
        found = first_rankable_instance(wide, min_tau=2)
        self.assertEqual((found["hw_rankable_k"], found["hw_rankable_m"], found["hw_rankable_borrowers"]), (30, 7, 900))
        self.assertEqual((found["hw_rankable_fr"], found["hw_rankable_signal"]), (None, None))


def _real_grid(dataset):
    """The rows of the 2026-10-19-style ``EC_noise_grid.csv`` of ``dataset``, or ``None`` when absent or from a smoke run."""
    path = RESULTS_DIR / dataset / "EC_noise_grid.csv"
    if not path.exists():
        return None
    if any((json.loads(p.read_text()).get("extra") or {}).get("smoke") for p in PIPELINE_DIR.glob("*/manifest.json")):
        return None
    return pl.read_csv(path, infer_schema_length=None).to_dicts()


class RealOutputsTests(unittest.TestCase):
    """The entry points of the first clean run (2026-10-19), read from the grid it wrote; skipped without that grid."""

    EXPECTED = {("hmeq", "fake_marrakesh"): 7, ("hmeq", "aer_noise_model"): 7, ("fmac", "fake_marrakesh"): 9, ("fmac", "aer_noise_model"): 7}

    def test_noise_k_band_of_the_first_clean_run(self):
        grids = {dataset: _real_grid(dataset) for dataset in ("hmeq", "fmac")}
        if any(rows is None for rows in grids.values()):
            self.skipTest("no real noise grid under outputs/results for both datasets (or a smoke tree)")
        for (dataset, arm), expected in self.EXPECTED.items():
            with self.subTest(dataset=dataset, arm=arm):
                self.assertEqual(noise_entry_points(grids[dataset], arm=arm, min_tau=int(CONFIG.hardware_min_tau))["noise_k_band"], expected)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def row(k, tau, fr, groups, qpu=10.0, m=6, borrowers=100, arm="fake_marrakesh", mitigated=False):
        return {"arm": arm, "mitigated": mitigated, "k_classes": k, "instance": f"top_{k}", "tau": tau, "fidelity_ratio": fr,
                "n_masked_groups": groups, "estimated_qpu_seconds": qpu, "m": m, "borrowers": borrowers, "R": 20, "depth_max": 1,
                "two_qubit_sum": 2, "predicted_survival_min": 0.1, "tv_uniform": 0.3}

    def test_prefers_a_rankable_instance_above_the_floor_and_flags_the_fallback(self):
        rows = [self.row(3, 2, 0.9, 1), self.row(4, 2, 0.6, 2), self.row(4, 3, 0.55, 2), self.row(6, 2, 0.8, 2, qpu=1e4),
                self.row(5, 2, 0.95, 3, m=9), self.row(4, 2, 0.99, 2, arm="aer_noise_model"), self.row(4, 2, 0.98, 2, mitigated=True),
                self.row(2, 1, 0.99, 2)]                                       # tau = 1 is the state preparation alone: never selected
        chosen = select_candidate(rows, CONFIG)
        self.assertEqual(chosen["flag"], "")
        self.assertEqual((chosen["selected"]["k_classes"], chosen["selected"]["tau"]), (4, 2))
        low = [self.row(3, 2, 0.9, 1), self.row(4, 2, 0.2, 2)]
        fallback = select_candidate(low, CONFIG)
        self.assertEqual(fallback["flag"], "cannot rank test borrowers")
        self.assertEqual(fallback["selected"]["k_classes"], 3)
        self.assertIsNone(select_candidate([self.row(3, 2, 0.9, 2, qpu=1e4)], CONFIG)["selected"])
        self.assertIsNone(select_candidate([self.row(3, 1, 0.9, 2)], CONFIG)["selected"])
        self.assertIsNone(select_candidate([{"arm": "fake_marrakesh", "status": NOT_SIMULATED}], CONFIG)["selected"])

    def test_analytic_two_qubit_count(self):
        self.assertEqual(analytic_two_qubit_count(1), 2 * 2 * 1 + 2 + 3)
        self.assertGreater(analytic_two_qubit_count(8), 2 * 256 * 255)
        self.assertLess(analytic_two_qubit_count(7), analytic_two_qubit_count(8) / 3)


class SimulationBudgetTests(unittest.TestCase):
    def test_memory_is_a_deterministic_limit_and_is_reported_with_its_estimate(self):
        self.assertEqual(state_bytes("fake_marrakesh", 10), 16 * 4**10)
        self.assertEqual(state_bytes("aer_noise_model", 10), 16 * 4**10)
        self.assertEqual(state_bytes("noiseless_shots", 10), 16 * 2**10)
        budget = SimulationBudget(total_seconds=1e12, memory_bytes=2**30, prior_bytes_per_second=1e9)
        run, info = budget.decide("fake_marrakesh", gates=100, bytes_needed=state_bytes("fake_marrakesh", 14))      # 4 GiB
        self.assertFalse(run)
        self.assertIn("GiB", info["reason"])
        self.assertEqual((info["estimated_state_bytes"], info["estimated_gates"]), (16 * 4**14, 100))
        run, info = budget.decide("fake_marrakesh", gates=100, bytes_needed=state_bytes("fake_marrakesh", 12))      # 256 MiB
        self.assertTrue(run)
        self.assertNotIn("reason", info)

    def test_time_scales_with_gates_and_state_size_and_starts_from_an_assumed_throughput(self):
        budget = SimulationBudget(total_seconds=100.0, memory_bytes=2**40, prior_bytes_per_second=1e6)
        self.assertAlmostEqual(budget.estimate("aer_noise_model", 10, 1000), 10 * 1000 / 1e6)       # no cell has run: gates x bytes / throughput
        run, info = budget.decide("aer_noise_model", gates=10**6, bytes_needed=10**6)                # 1e6 s: never started
        self.assertFalse(run)
        self.assertIn("budget", info["reason"])
        self.assertAlmostEqual(info["estimated_simulation_seconds"], 1e6)
        self.assertTrue(budget.decide("aer_noise_model", gates=1000, bytes_needed=10**4)[0])         # 10 s
        budget.record("aer_noise_model", gates=1000, bytes_needed=10**4, seconds=5.0)
        self.assertAlmostEqual(budget.spent_seconds, 5.0)
        self.assertAlmostEqual(budget.rate("aer_noise_model"), 5e-7)
        self.assertAlmostEqual(budget.rate("fake_marrakesh"), 5e-7)                                  # a density matrix as well
        self.assertAlmostEqual(budget.rate("noiseless_shots"), 1e-6)                                 # a statevector: still the assumed throughput
        self.assertAlmostEqual(budget.estimate("aer_noise_model", 1000, 2 * 10**4), 10.0)            # twice the state, twice the time
        self.assertTrue(budget.decide("fake_marrakesh", gates=1000, bytes_needed=10**5)[0])          # 5 s spent + 50 s estimated
        run, info = budget.decide("fake_marrakesh", gates=1000, bytes_needed=2 * 10**5)              # 5 s spent + 100 s estimated
        self.assertFalse(run)
        self.assertAlmostEqual(info["estimated_simulation_seconds"], 100.0)


def _instance(n=4, k=3):
    return {"k": k, "label": f"top_{k}", "G": np.full((n, n), 1.0 / n), "v": np.full(n, 1.0 / n), "R": n, "m": 2, "borrowers": 10,
            "n_masked_groups": 2}


class ShardTests(unittest.TestCase):
    def test_a_shard_is_refused_unless_it_was_written_under_exactly_this_identity(self):
        identity = shards.shard_identity(CONFIG, "hmeq", "06_hardware", _instance(), False)
        self.assertEqual(identity, shards.shard_identity(CONFIG, "hmeq", "06_hardware", _instance(), False))
        self.assertEqual(identity["shard_schema_version"], shards.SHARD_SCHEMA_VERSION)
        other_operator = _instance()
        other_operator["G"] = np.eye(4)
        others = [shards.shard_identity(dataclasses.replace(CONFIG, hardware_shots=2048), "hmeq", "06_hardware", _instance(), False),
                  shards.shard_identity(CONFIG, "hmeq", "06_hardware", other_operator, False),
                  shards.shard_identity(CONFIG, "fmac", "06_hardware", _instance(), False),
                  shards.shard_identity(CONFIG, "hmeq", "06_hardware", _instance(k=4), False),
                  shards.shard_identity(CONFIG, "hmeq", "06_hardware", _instance(), True)]
        for other in others:
            self.assertNotEqual(identity, other)
        with tempfile.TemporaryDirectory() as tmp:
            path = shards.shard_path(Path(tmp), 3)
            shards.write_shard(path, identity, {"rows": [{"arm": "noiseless_shots", "tau": 1}], "vectors": np.array([1.0, 2.0])}, complete=False)
            self.assertFalse(list(path.parent.glob("*.part")))
            record = shards.read_shard(path, identity)
            self.assertFalse(record["complete"])
            self.assertEqual(record["vectors"], [1.0, 2.0])
            for other in others:
                with self.assertRaisesRegex(AssertionError, "Delete the directory"):
                    shards.read_shard(path, other)
            path.write_text(json.dumps({"rows": []}))                                   # a shard from before the identity existed
            with self.assertRaisesRegex(AssertionError, "Delete the directory"):
                shards.read_shard(path, identity)

    def test_every_cell_of_the_grid_ends_with_a_row(self):
        instance, expected = _instance(), shards.expected_cells(CONFIG)
        rows, filled = shards.cells_of(None, instance, "hmeq", CONFIG)
        self.assertEqual((len(rows), filled), (len(expected), len(expected)))
        self.assertTrue(all(r["status"] == NOT_SIMULATED and r["reason"] for r in rows))
        partial = {"rows": [{"arm": "noiseless_shots", "tau": 1, "status": ""}, {"arm": "aer_noise_model", "tau": 1, "status": ""}]}
        rows, filled = shards.cells_of(partial, instance, "hmeq", CONFIG)
        self.assertEqual(filled, len(expected) - 2)
        self.assertEqual(sorted((r["arm"], int(r["tau"])) for r in rows), sorted(expected))


class CalibrationIndexingTests(unittest.TestCase):
    def test_circuits_are_grouped_by_their_ordered_measured_tuple(self):
        groups = calibration_groups([[0, 1], [2, 3], [0, 1], [1, 0], [2, 3]])
        self.assertEqual(groups.tuples, ((0, 1), (2, 3), (1, 0)))
        self.assertEqual(groups.index_of_circuit, (0, 1, 0, 2, 1))
        self.assertEqual([groups.n_sets_for(n) for n in range(1, 6)], [1, 2, 2, 3, 3])
        self.assertEqual(calibration_groups([]).index_of_circuit, ())

    def test_each_circuit_is_mitigated_with_the_set_of_its_own_tuple(self):
        shots = 1000
        noisy = PerQubitCalibration([np.array([[0.9, 0.3], [0.1, 0.7]])])           # A[measured, prepared], asymmetric readout error
        ideal = PerQubitCalibration([np.eye(2)])
        truth = np.array([0.5, 0.5])
        counts = [{"0": 600, "1": 400},                                              # the true state read through `noisy`
                  {"0": 500, "1": 500}]                                              # the true state read through `ideal`
        own, _ = mitigated_omega_by_circuit(counts, [noisy, ideal], [0, 1], n_red=2, m=1, shots=shots)
        np.testing.assert_allclose(own, truth, atol=1e-9)
        single, _ = mitigated_omega_by_circuit(counts, [noisy], [0, 0], n_red=2, m=1, shots=shots)      # one calibration for every circuit
        self.assertGreater(float(np.abs(single - truth).max()), 0.05)
        swapped, _ = mitigated_omega_by_circuit(counts, [noisy, ideal], [1, 0], n_red=2, m=1, shots=shots)
        self.assertGreater(float(np.abs(swapped - truth).max()), 0.03)     # (0.6, 0.4) and (1/3, 2/3) average to (0.4667, 0.5333)
        with self.assertRaises(ValueError):
            mitigated_omega_by_circuit(counts, [noisy, ideal], [0], n_red=2, m=1, shots=shots)


@pytest.mark.qiskit
class MeasuredQubitsTests(unittest.TestCase):
    def test_the_measured_tuple_of_a_routed_circuit_agrees_with_its_final_layout(self):
        from qiskit import QuantumCircuit
        from qiskit.transpiler import CouplingMap, generate_preset_pass_manager

        from quantum.hardware.mitigation import measured_physical_qubits

        m = 2
        qc = QuantumCircuit(2 * m, m)
        qc.h(0)
        qc.cx(0, 3)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.measure(range(m, 2 * m), range(m))
        line = CouplingMap.from_line(2 * m + 1)
        isa = generate_preset_pass_manager(coupling_map=line, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                                           seed_transpiler=3).run(qc)
        measured = measured_physical_qubits(isa)                                    # raises when the layout and the measures disagree
        self.assertEqual(len(measured), m)
        self.assertEqual(len(set(measured)), m)
        self.assertTrue(all(0 <= q < line.size() for q in measured))


if __name__ == "__main__":
    unittest.main()
