"""The one implementation of every reported quantity, and the uncertainty around it.

Invariants 8 and 9: tie-exact precision@k, lift@k and top-k overlap; the
normalized L1, normalized L2 and Kendall agreement measures; the shared
outcome-stratified bootstrap and the paired difference computed on it; the
baselines fit on visible labels only; the cross-fit cache that refuses another
configuration; and the CSV-only result writer (the manifest that records its tables is in
test_manifest).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from config import DEFAULT_CONFIG
from evaluation import crossfit as crossfit_module
from evaluation.controls import (
    blank_field_count,
    build_baselines,
    cell_seed_default_rate,
    one_hop_seed_count,
    state_matrix,
)
from evaluation.crossfit import (
    NO_FOLD,
    CrossfitResult,
    assert_crossfit_validity,
    assert_same_folds,
    crossfit_scores,
    fold_assignment,
)
from evaluation.masked_eval import Partitions, _ranking_report
from evaluation.provenance import write_result
from evaluation.reporting import (
    agreement_with,
    bootstrap_replicates,
    expected_precision_at_k,
    expected_top_k_overlap,
    inclusion_probability,
    kendall_tau,
    l1_sum,
    mean_abs_diff,
    metrics_with_ci,
    no_skill_metrics,
    paired_difference,
    placebo_p_value,
    spearman,
    top_k_jaccard,
    total_variation,
)
from evaluation.ties import boundary, canonicalize, distinct_levels
from fingerprint import bootstrap_indices
from hmeq.config import NET_2L

LAYER_COLUMNS = {
    "JOB": ["JOB_Mgr", "JOB_Office", "JOB_Other"],
    "REASON": ["REASON_DebtCon", "REASON_HomeImp"],
}


def _frame(n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(7)
    job = rng.integers(0, 4, size=n)  # 3 == unrecorded
    reason = rng.integers(0, 3, size=n)  # 2 == unrecorded
    bad = ((job == 0) | (reason == 0)).astype(int)
    bad[rng.random(n) < 0.15] ^= 1
    data = {"BAD": bad}
    for state, name in enumerate(LAYER_COLUMNS["JOB"]):
        data[name] = (job == state).astype(int)
    for state, name in enumerate(LAYER_COLUMNS["REASON"]):
        data[name] = (reason == state).astype(int)
    return pl.DataFrame(data)


def _partitions(frame: pl.DataFrame) -> Partitions:
    n = frame.height
    partition = np.array(["development"] * n, dtype=object)
    partition[np.arange(n) % 5 == 0] = "test"
    return Partitions(
        frame=frame,
        borrower_ids=np.arange(n),
        partition=partition.astype(str),
        target_col="BAD",
        processed_path=None,
        split_path=None,
    )


class TieExactMetricTests(unittest.TestCase):
    def test_inclusion_probability_splits_the_boundary_uniformly(self):
        np.testing.assert_allclose(
            inclusion_probability(np.array([3.0, 2.0, 2.0, 2.0, 1.0]), 3),
            [1.0, 2 / 3, 2 / 3, 2 / 3, 0.0],
        )

    def test_expected_precision_is_the_monte_carlo_limit_and_reduces_to_plain_precision_untied(
        self,
    ):
        rng = np.random.default_rng(0)
        scores = rng.integers(0, 4, size=200).astype(float)
        y = (rng.random(200) < 0.3).astype(int)
        precision, lift = expected_precision_at_k(scores, y, 25)
        # The Monte-Carlo reference, in the test only: random tie-breaks on the one boundary definition.
        above, tied, slots = boundary(canonicalize(scores, 12), 25)
        keys = np.random.default_rng(1).random((20000, tied.size))
        drawn = tied[np.argpartition(keys, slots - 1, axis=1)[:, :slots]]
        precision_mc = float((y[above].sum() + y[drawn].sum(axis=1)).mean()) / 25
        self.assertAlmostEqual(precision, precision_mc, delta=0.01)
        self.assertAlmostEqual(lift, precision_mc / float(y.mean()), delta=0.05)
        untied = np.arange(20, dtype=float)
        self.assertEqual(expected_precision_at_k(untied, (untied >= 15).astype(int), 5)[0], 1.0)
        self.assertEqual(expected_top_k_overlap(untied, untied.copy(), 10), 1.0)
        self.assertEqual(no_skill_metrics(np.array([1, 0, 0, 0]), (2,))["pr_auc"], 0.25)

    def test_the_quick_look_report_uses_the_same_precision_as_the_tables(self):
        scores = np.concatenate([np.full(10, 1.0), np.zeros(10)])
        y = np.concatenate([np.tile([1, 0], 5), np.zeros(10)]).astype(int)
        report = _ranking_report(
            scores,
            y,
            evaluation_idx=np.arange(20),
            seed_idx=np.array([], dtype=int),
            k_values=(5,),
            model_label="unit",
        )
        self.assertEqual(report["n_tied_at_boundary_5"], 10)
        self.assertEqual(report["distinct_levels"], 2)
        self.assertEqual(report["precision_at_5"], expected_precision_at_k(scores, y, 5)[0])

    def test_canonicalize_collapses_solver_noise_and_keeps_levels_apart(self):
        noisy = np.full(50, 1e-3) + np.random.default_rng(0).normal(0, 1e-17, size=50)
        self.assertGreater(np.unique(noisy).size, 1)
        self.assertEqual(distinct_levels(noisy, 12), 1)
        self.assertEqual(distinct_levels(np.array([1e-3, 2e-3, 3e-3]), 12), 3)
        with self.assertRaises(ValueError):
            canonicalize(np.array([1.0, np.nan]), 12)


class AgreementMeasureTests(unittest.TestCase):
    def test_the_three_l1_forms_are_one_sum_and_the_overlap_is_rounded_first(self):
        a = np.array([0.1, 0.2, 0.3, 0.4])
        b = np.array([0.1, 0.4, 0.3, 0.2])
        self.assertAlmostEqual(l1_sum(a, b), 0.4)
        self.assertAlmostEqual(mean_abs_diff(a, b), 0.4 / 4)
        self.assertAlmostEqual(total_variation(a, b), 0.2)
        # Duplicating the data leaves the mean absolute difference unchanged and doubles the sum.
        self.assertAlmostEqual(mean_abs_diff(np.tile(a, 2), np.tile(b, 2)), mean_abs_diff(a, b))
        self.assertAlmostEqual(l1_sum(np.tile(a, 2), np.tile(b, 2)), 2 * l1_sum(a, b))
        self.assertEqual(mean_abs_diff(a, a), 0.0)
        self.assertEqual(kendall_tau(a, a), 1.0)
        self.assertEqual(kendall_tau(a, a[::-1]), -1.0)
        self.assertTrue(np.isnan(kendall_tau(a, np.ones(4))))
        with self.assertRaises(ValueError):
            mean_abs_diff(a, a[:3])
        # A boundary tie is split uniformly, so a tied vector's overlap with itself is below one:
        # (1 + 3 * (2/3)^2) / 3 = 7/9 here. That ceiling is the self-overlap the tables report,
        # and solver noise below the tie tolerance is rounded away before it is computed.
        tied = np.array([3.0, 2.0, 2.0, 2.0, 1.0])
        noisy = tied + np.array([0.0, 1e-15, -1e-15, 0.0, 0.0])
        rounded = agreement_with(noisy, tied, 3, decimals=12, suffix="x")["top_3_overlap_x"]
        self.assertAlmostEqual(rounded, expected_top_k_overlap(tied, tied, 3))
        self.assertAlmostEqual(rounded, 7 / 9)
        self.assertAlmostEqual(top_k_jaccard(tied, tied, 3), (7 / 9) / (2 - 7 / 9))
        self.assertAlmostEqual(placebo_p_value(0.5, [0.1, 0.6, 0.5, 0.2]), (1 + 2) / (1 + 4))

    def test_spearman_is_canonicalized_and_a_self_comparison_is_exactly_one(self):
        a = np.array([0.1, 0.2, 0.2, 0.4])
        self.assertEqual(spearman(a, a), 1.0)
        # Solver noise below the tie tolerance must not break the tie on one side only.
        self.assertEqual(spearman(a, a + np.array([0.0, 1e-15, -1e-15, 0.0])), 1.0)
        self.assertTrue(np.isnan(spearman(a, np.ones(4))))


class SharedBootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = (rng.random(300) < 0.25).astype(int)
        signal = self.y + rng.normal(0, 1.0, size=300)
        self.strong, self.weak = signal, signal + rng.normal(0, 3.0, size=300)
        self.indices = bootstrap_indices(self.y, 200, seed=DEFAULT_CONFIG.master_seed)

    def test_indices_are_stratified_and_reproducible(self):
        self.assertEqual(self.indices.shape, (200, 300))
        for idx in self.indices[:10]:
            self.assertEqual(int(self.y[idx].sum()), int(self.y.sum()))
        np.testing.assert_array_equal(
            bootstrap_indices(self.y, 200, seed=DEFAULT_CONFIG.master_seed), self.indices
        )

    def test_marginal_intervals_bracket_the_point_and_the_paired_difference_is_exact_for_a_self_comparison(
        self,
    ):
        row = metrics_with_ci(self.strong, self.y, self.indices, k=25)
        self.assertLessEqual(row["pr_auc_ci_lo"], row["pr_auc"])
        self.assertGreaterEqual(row["pr_auc_ci_hi"], row["pr_auc"])
        self.assertIn("lift_at_25_ci_hi", row)
        reps = bootstrap_replicates(self.strong, self.y, self.indices, k=25)
        same = paired_difference(reps["roc_auc"], reps["roc_auc"].copy(), 1.0, 1.0)
        self.assertEqual((same["ci_lo"], same["ci_hi"], same["observed"]), (0.0, 0.0, 0.0))
        weak = bootstrap_replicates(self.weak, self.y, self.indices, k=25)
        diff = paired_difference(reps["roc_auc"], weak["roc_auc"], 1.0, 0.0)
        self.assertGreater(diff["p_delta_pos"], 0.9)
        self.assertAlmostEqual(diff["p_delta_pos"] + diff["p_delta_zero"] + diff["p_delta_neg"], 1.0)
        self.assertEqual(same["p_delta_zero"], 1.0)
        self.assertLess(diff["ci_lo"], diff["ci_hi"])


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = _frame()
        self.y = self.frame["BAD"].to_numpy().astype(int)
        self.fit_idx = np.arange(0, self.frame.height, 2)
        self.states = state_matrix(self.frame, LAYER_COLUMNS)

    def test_cell_rate_is_a_groupby_on_the_fit_set_and_one_hop_counts_each_seed_once(self):
        scores = cell_seed_default_rate(self.states, self.y, self.fit_idx)
        cell = np.array([i for i in self.fit_idx if tuple(self.states[i]) == tuple(self.states[3])])
        self.assertAlmostEqual(scores[3], float(self.y[cell].mean()))
        counts = one_hop_seed_count(
            np.array([[0, 0], [0, 0], [1, 1], [0, -1]]), np.array([1, 0, 0, 0]), np.arange(4)
        )
        np.testing.assert_array_equal(counts, [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(blank_field_count([0, 2, 1]), [0.0, 2.0, 1.0])

    def test_every_baseline_is_built_and_none_reads_a_label_outside_the_fit_set(self):
        left = build_baselines(self.frame, LAYER_COLUMNS, self.y, self.fit_idx)
        self.assertEqual(
            set(left),
            {
                "no_skill",
                "cell_seed_default_rate",
                "one_hop_seed_count",
                "logit_on_layer_dummies",
                "single_layer_seed_default_rate_JOB",
                "single_layer_seed_default_rate_REASON",
            },
        )
        flipped = self.y.copy()
        flipped[np.setdiff1d(np.arange(self.frame.height), self.fit_idx)] ^= 1
        right = build_baselines(self.frame, LAYER_COLUMNS, flipped, self.fit_idx)
        for name in left:
            np.testing.assert_allclose(left[name], right[name])


class CrossFitTests(unittest.TestCase):
    def setUp(self):
        self.parts = _partitions(_frame())
        self.y = self.parts.y_true

    @staticmethod
    def _structural_scorer(masked_frame: pl.DataFrame, tag: str) -> np.ndarray:
        y = masked_frame["BAD"].to_numpy().astype(int)
        return cell_seed_default_rate(
            state_matrix(masked_frame, LAYER_COLUMNS), y, np.arange(masked_frame.height)
        )

    def test_fold_index_travels_with_the_scores_and_a_leaking_feature_is_caught(self):
        result = crossfit_scores(self.parts, self._structural_scorer, folds=5, seed=42)
        self.assertEqual(result.fold_index.shape, result.scores.shape)
        assert_same_folds(result, result)
        with self.assertRaises(AssertionError):
            assert_same_folds(
                result,
                CrossfitResult(
                    scores=result.scores.copy(),
                    fold_index=np.roll(result.fold_index, 1),
                    n_folds=5,
                    seed=42,
                ),
            )
        self.assertLess(assert_crossfit_validity(result, self.y)["in_sample_auc"], 0.95)
        folds = fold_assignment(self.parts, 5, 42)
        leaked = CrossfitResult(
            scores=np.where(folds == NO_FOLD, np.nan, self.y.astype(float)),
            fold_index=folds,
            n_folds=5,
            seed=42,
        )
        with self.assertRaises(AssertionError) as caught:
            assert_crossfit_validity(leaked, self.y)
        self.assertIn("in-sample AUC", str(caught.exception))

    def test_a_cache_from_another_configuration_is_refused_and_the_same_one_is_reused(self):
        calls = []

        def counting_scorer(masked_frame, tag):
            calls.append(tag)
            return self._structural_scorer(masked_frame, tag)

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "crossfit_cache"
            first = crossfit_scores(
                self.parts,
                counting_scorer,
                folds=2,
                seed=42,
                cache_dir=cache,
                cache_key="net2l-2fold",
            )
            second = crossfit_scores(
                self.parts,
                counting_scorer,
                folds=2,
                seed=42,
                cache_dir=cache,
                cache_key="net2l-2fold",
            )
            self.assertEqual(len(calls), 2, "a matching cache must not re-solve")
            np.testing.assert_array_equal(
                first.scores[first.scored_idx], second.scores[second.scored_idx]
            )
            with self.assertRaises(ValueError) as ctx:
                crossfit_scores(
                    self.parts,
                    counting_scorer,
                    folds=2,
                    seed=42,
                    cache_dir=cache,
                    cache_key="other",
                )
            self.assertIn("net2l-2fold", str(ctx.exception))

    def test_no_function_pools_or_ranks_raw_scores_across_folds(self):
        offenders = [
            name
            for name, value in vars(crossfit_module).items()
            if callable(value)
            and getattr(value, "__module__", None) == crossfit_module.__name__
            and ("pool" in name or "within_fold" in name)
        ]
        self.assertEqual(offenders, [])


class ResultWriterTests(unittest.TestCase):
    def test_writes_one_csv_per_scope_and_registers_provenance(self):
        from evaluation import provenance

        frame = pl.DataFrame({"dataset": ["hmeq"], "network": ["NET_2L"], "pr_auc": [0.31234567], "pr_auc_ci_lo": [0.29], "pr_auc_ci_hi": [0.33]})
        with tempfile.TemporaryDirectory() as td:
            results = Path(td) / "results"
            with mock_results(results):
                path = write_result("hmeq", "unit_table", frame, stage="05_downstream",
                                    columns={"dataset": "dataset name", "network": "network", **provenance.ci_columns("pr_auc", "PR-AUC")})
            self.assertEqual(path, results / "hmeq" / "unit_table.csv")
            self.assertTrue(path.exists())
            self.assertFalse((results / "hmeq" / "unit_table.tex").exists())
            self.assertFalse((results / "hmeq" / "unit_table_meta.json").exists())
            self.assertAlmostEqual(pl.read_csv(path)["pr_auc"][0], 0.31234567, places=8)
            with self.assertRaises(ValueError):
                write_result("tables", "unit_table", frame, stage="05_downstream", columns={})
        record = provenance.registered_tables("05_downstream")["unit_table"]
        self.assertEqual(record["scope"], "hmeq")
        self.assertEqual(record["columns"]["pr_auc_ci_lo"], "lower end of the 95% bootstrap interval of pr_auc")
        self.assertEqual(record["undocumented"], [])
        self.assertTrue(record["path"].endswith("unit_table.csv"))
        provenance._REGISTRY.pop("05_downstream", None)


def mock_results(results: Path):
    """Patch the results root for one test: ``paths.results_dir`` and ``paths.result_path`` read ``paths.RESULTS_DIR`` at call time."""
    from unittest import mock

    return mock.patch("paths.RESULTS_DIR", results)


if __name__ == "__main__":
    unittest.main()
