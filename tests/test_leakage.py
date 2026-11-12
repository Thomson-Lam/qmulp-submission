"""Nothing a hidden label could reach: the split, the masking, the cleaner, the metric entry point.

Invariants 4-6 of the paper. The split is read from the frozen artifact and
never recomputed; test and held-out labels are masked before any operator is
built and a flipped hidden label changes nothing downstream; the absence marker
is never a category; and no reported metric is computed on a population that
contains its own seeds. Everything runs on synthetic frames; the count contract
against the real raw file is gated behind its presence.
"""

from __future__ import annotations

import ast
import inspect
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from evaluation.controls import cell_seed_default_rate, state_matrix
from evaluation.crossfit import NO_FOLD, crossfit_scores, fold_assignment, score_test_partition
from evaluation.masked_eval import (
    DEVELOPMENT_PARTITION,
    TEST_PARTITION,
    UNKNOWN_LABEL,
    Partitions,
    assert_out_of_influence,
    evaluate_ranking_on_subset,
    load_partitions,
)
from hmeq.cleaning import (
    MISSING_INDICATOR_SUFFIX,
    apply_imputation,
    clean_hmeq,
    clean_hmeq_frame,
    drop_unconnectable,
    fit_medians,
)
from hmeq.config import RAW_CSV
from preprocessing.missing_levels import (
    MISSING_SENTINEL,
    assert_absence_consistent,
    is_missing_column,
    split_missing_columns,
)

RAW_COLUMNS = [
    "BAD",
    "LOAN",
    "MORTDUE",
    "VALUE",
    "REASON",
    "JOB",
    "YOJ",
    "DEROG",
    "DELINQ",
    "CLAGE",
    "NINQ",
    "CLNO",
    "DEBTINC",
]
JOBS = ["Mgr", "Office", "Other", "ProfExe", "Sales", "Self"]
REASONS = ["DebtCon", "HomeImp"]


def synthetic_raw(n: int = 160) -> pd.DataFrame:
    """A raw HMEQ-shaped frame: every level in development, some blanks, a few unconnectable rows."""
    rows = []
    for i in range(n):
        both_missing = i % 40 == 39
        rows.append(
            {
                "BAD": 1 if i % 5 == 0 else 0,
                "LOAN": 1000.0 + i,
                "MORTDUE": None if i % 9 == 0 else 100.0 + 3.0 * i,
                "VALUE": 200.0 + i,
                "REASON": None if (both_missing or i % 17 == 0) else REASONS[i % 2],
                "JOB": None if (both_missing or (i % 13 == 0 and i % 17 != 0)) else JOBS[i % 6],
                "YOJ": float(i % 20),
                "DEROG": 0.0,
                "DELINQ": 0.0,
                "CLAGE": 100.0 + i,
                "NINQ": 1.0,
                "CLNO": 10.0,
                "DEBTINC": None if i % 7 == 0 else 30.0 + (i % 11),
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def clean_to(root: Path, *, seed: int = 5, stem: str = "a"):
    """Clean the synthetic raw file into ``root``; returns ``(processed, split, result)``."""
    csv = root / "hmeq.csv"
    if not csv.exists():
        synthetic_raw().to_csv(csv, index=False)
    processed, split = root / f"{stem}_processed.parquet", root / f"{stem}_frozen_split.json"
    return processed, split, clean_hmeq(csv, processed, split, test_frac=0.25, split_seed=seed)


# ---------------------------------------------------------------------------
# Invariant 4: the frozen split
# ---------------------------------------------------------------------------


class FrozenSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.processed, cls.split, cls.result = clean_to(Path(cls._tmp.name))
        cls.parts = load_partitions(cls.processed, cls.split)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_split_is_written_once_with_indices_counts_seed_and_checksum(self):
        record = json.loads(self.split.read_text())
        for key in (
            "raw_sha256",
            "processed_sha256",
            "n_rows",
            "development_idx",
            "test_idx",
            "n_development_defaults",
            "n_test_defaults",
            "split_seed",
            "test_frac",
            "dropped_raw_indices",
        ):
            self.assertIn(key, record)
        self.assertEqual(record["split_seed"], 5)
        self.assertEqual(len(record["development_idx"]) + len(record["test_idx"]), record["n_rows"])
        combined = np.sort(np.concatenate([self.parts.development_idx, self.parts.test_idx]))
        np.testing.assert_array_equal(combined, np.arange(self.parts.frame.height))
        np.testing.assert_array_equal(self.parts.borrower_ids, np.arange(self.parts.frame.height))
        self.assertEqual(set(self.parts.partition), {DEVELOPMENT_PARTITION, TEST_PARTITION})

    def test_load_partitions_never_resplits(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(load_partitions)))
        called = {
            getattr(n.func, "id", getattr(n.func, "attr", ""))
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        self.assertNotIn("train_test_split", called)

    def test_a_frame_the_split_was_not_made_for_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            other_processed, _split, other = clean_to(Path(td), seed=99, stem="b")
            self.assertNotEqual(other.processed_sha256, self.result.processed_sha256)
            with self.assertRaises(ValueError) as ctx:
                load_partitions(other_processed, self.split)
            self.assertIn("sha256", str(ctx.exception))
            record = json.loads(self.split.read_text())
            record["n_test_defaults"] += 1
            tampered = Path(td) / "frozen_split.json"
            tampered.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                load_partitions(self.processed, tampered)


# ---------------------------------------------------------------------------
# Invariants 5 and 6: masking, the unknown level, the fold that never sees itself
# ---------------------------------------------------------------------------


class MaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        processed, split, _r = clean_to(Path(cls._tmp.name))
        cls.parts = load_partitions(processed, split)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_mask_labels_zeroes_exactly_the_hidden_ids_and_masked_levels_use_the_unknown_level(
        self,
    ):
        y = self.parts.y_true
        hide = self.parts.test_idx
        masked = self.parts.mask_labels(hide)[self.parts.target_col].to_numpy().astype(int)
        np.testing.assert_array_equal(masked[hide], 0)
        keep = self.parts.development_idx
        np.testing.assert_array_equal(masked[keep], y[keep])
        levels = self.parts.masked_labels(hide)
        np.testing.assert_array_equal(levels[hide], UNKNOWN_LABEL)
        np.testing.assert_array_equal(levels[keep], y[keep])
        np.testing.assert_array_equal(self.parts.y_true, y)  # the source frame is untouched
        seeds = self.parts.seed_ids(hide)
        self.assertEqual(np.intersect1d(seeds, hide).size, 0)
        with self.assertRaises(ValueError):
            self.parts.mask_labels([self.parts.frame.height])

    @staticmethod
    def _structural_scorer(masked_frame: pl.DataFrame, tag: str) -> np.ndarray:
        """A score built only from other borrowers' visible labels, as the diffusion is."""
        columns = {
            "JOB": [
                c
                for c in masked_frame.columns
                if c.startswith("JOB_")
                and not is_missing_column(c)
                and not c.endswith(MISSING_INDICATOR_SUFFIX)
            ],
            "REASON": [
                c
                for c in masked_frame.columns
                if c.startswith("REASON_")
                and not is_missing_column(c)
                and not c.endswith(MISSING_INDICATOR_SUFFIX)
            ],
        }
        y = masked_frame["BAD"].to_numpy().astype(int)
        return cell_seed_default_rate(
            state_matrix(masked_frame, columns), y, np.arange(masked_frame.height)
        )

    def test_flipping_a_held_out_folds_labels_does_not_change_that_folds_scores(self):
        folds = 3
        assignment = fold_assignment(self.parts, folds, 42)
        np.testing.assert_array_equal(np.flatnonzero(assignment == NO_FOLD), self.parts.test_idx)
        baseline = crossfit_scores(
            self.parts, self._structural_scorer, folds=folds, seed=42, fold_index=assignment
        )
        self.assertTrue(np.isnan(baseline.scores[self.parts.test_idx]).all())
        for fold in range(folds):
            members = baseline.fold_members(fold)
            flipped_y = self.parts.y_true.copy()
            flipped_y[members] ^= 1
            flipped_y[self.parts.test_idx] ^= 1
            flipped = Partitions(
                frame=self.parts.frame_with_labels(flipped_y),
                borrower_ids=self.parts.borrower_ids,
                partition=self.parts.partition,
                target_col="BAD",
                processed_path=None,
                split_path=None,
            )
            # The recorded folds are held fixed: the question is whether a hidden label reaches the operator.
            other = crossfit_scores(
                flipped, self._structural_scorer, folds=folds, seed=42, fold_index=assignment
            )
            np.testing.assert_array_equal(
                other.scores[members], baseline.scores[members], err_msg=f"fold {fold}"
            )
            self.assertFalse(
                np.array_equal(
                    other.scores[baseline.fold_members((fold + 1) % folds)],
                    baseline.scores[baseline.fold_members((fold + 1) % folds)],
                ),
                "the flipped labels are visible to the other folds, so their scores must move",
            )
        seen = {}

        def scorer(masked_frame, tag):
            seen["visible"] = masked_frame["BAD"].to_numpy().astype(int)
            return np.zeros(masked_frame.height)

        score_test_partition(self.parts, scorer)
        self.assertEqual(int(seen["visible"][self.parts.test_idx].sum()), 0)

    def test_no_metric_is_computed_on_a_population_containing_its_seeds(self):
        y = self.parts.y_true
        seeds = np.flatnonzero((y == 1) & (self.parts.partition == DEVELOPMENT_PARTITION))
        rng = np.random.default_rng(1)
        scores = np.where(
            np.isin(np.arange(y.size), seeds),
            rng.uniform(0.9, 1.0, y.size),
            rng.uniform(0.0, 0.1, y.size),
        )
        report = evaluate_ranking_on_subset(
            scores, y, self.parts.test_idx, seeds, k_values=(5,), model_label="ok"
        )
        self.assertIn("roc_auc", report)
        for population in (
            seeds,
            np.arange(y.size),
            np.concatenate([self.parts.test_idx, seeds[:1]]),
        ):
            with self.assertRaises(AssertionError):
                evaluate_ranking_on_subset(
                    scores, y, population, seeds, k_values=(5,), model_label="leak"
                )
        with self.assertRaises(AssertionError) as ctx:
            assert_out_of_influence([7, 8], [8, 9])
        self.assertIn("8", str(ctx.exception))


# ---------------------------------------------------------------------------
# The cleaner's order of operations and the absence rule, on synthetic data
# ---------------------------------------------------------------------------


class CleanerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = synthetic_raw()
        self.result = clean_hmeq_frame(self.raw, test_frac=0.25, split_seed=5)

    def test_counts_drop_split_and_default_totals_are_consistent(self):
        kept, n_dropped, dropped = drop_unconnectable(self.raw)
        self.assertEqual((n_dropped, dropped), (4, [39, 79, 119, 159]))
        self.assertEqual(self.result.n_rows, len(self.raw) - 4)
        self.assertEqual(
            self.result.development_idx.size + self.result.test_idx.size, self.result.n_rows
        )
        bad = self.result.frame["BAD"].to_numpy().astype(int)
        counts = self.result.default_counts()
        self.assertEqual(counts["n_defaults"], int(bad.sum()))
        self.assertEqual(
            counts["n_development_defaults"] + counts["n_test_defaults"], counts["n_defaults"]
        )
        self.assertAlmostEqual(self.result.test_idx.size / self.result.n_rows, 0.25, delta=0.01)

    def test_medians_are_fitted_on_development_only_and_flags_are_retained(self):
        kept, _n, _d = drop_unconnectable(self.raw)
        medians = fit_medians(kept, self.result.development_idx)
        self.assertEqual(medians, self.result.imputation_medians)
        kept2 = kept.copy()
        kept2.loc[int(self.result.test_idx[0]), "MORTDUE"] = 1e9
        self.assertEqual(
            fit_medians(kept2, self.result.development_idx)["MORTDUE"], medians["MORTDUE"]
        )
        _filled, flags, n_missing = apply_imputation(kept, medians)
        self.assertEqual(sorted(flags), ["DEBTINC_was_missing", "MORTDUE_was_missing"])
        for flag in flags:
            self.assertEqual(
                int(self.result.frame[flag].sum()),
                n_missing[flag.replace(MISSING_INDICATOR_SUFFIX, "")],
            )
        self.assertEqual(self.result.frame.isna().sum().sum(), 0)

    def test_a_blank_layer_variable_is_a_marker_never_a_category(self):
        frame = self.result.frame
        job = [
            c
            for c in frame.columns
            if c.startswith("JOB_") and not c.endswith(MISSING_INDICATOR_SUFFIX)
        ]
        genuine, markers = split_missing_columns(job)
        self.assertEqual((len(genuine), markers), (6, ["JOB_Missing"]))
        self.assertTrue((frame[job].sum(axis=1) == 1).all())
        self.assertEqual(int((frame["JOB_Missing"] & frame["REASON_Missing"]).sum()), 0)
        assert_absence_consistent(frame, genuine, markers, "JOB")
        self.assertFalse(is_missing_column("JOB_MissingPersonsBureau"))
        with self.assertRaises(ValueError):
            assert_absence_consistent(
                pd.DataFrame({"JOB_A": [1, 0], "JOB_Missing": [0, 0]}),
                ["JOB_A"],
                ["JOB_Missing"],
                "JOB",
            )
        collision = self.raw.copy()
        collision.loc[1, "JOB"] = MISSING_SENTINEL
        with self.assertRaises(AssertionError):
            clean_hmeq_frame(collision)


@unittest.skipUnless(RAW_CSV.exists(), f"{RAW_CSV} is not present")
class RealDataCountTests(unittest.TestCase):
    """The recorded HMEQ counts. Never adjust an expected value; the discrepancy is the finding."""

    def test_the_recorded_counts(self):
        raw = pd.read_csv(RAW_CSV)
        result = clean_hmeq_frame(raw, test_frac=0.2, split_seed=42)
        self.assertEqual(len(raw), 5_960)
        self.assertEqual(result.n_unconnectable_dropped, 107)
        self.assertEqual(result.n_rows, 5_853)
        self.assertEqual(
            result.default_counts(),
            {"n_defaults": 1_179, "n_development_defaults": 943, "n_test_defaults": 236},
        )
        self.assertEqual((result.development_idx.size, result.test_idx.size), (4_682, 1_171))
        genuine_job, _m = split_missing_columns(
            [
                c
                for c in result.frame.columns
                if c.startswith("JOB_") and not c.endswith(MISSING_INDICATOR_SUFFIX)
            ]
        )
        genuine_reason, _m = split_missing_columns(
            [
                c
                for c in result.frame.columns
                if c.startswith("REASON_") and not c.endswith(MISSING_INDICATOR_SUFFIX)
            ]
        )
        self.assertEqual((len(genuine_job), len(genuine_reason)), (6, 2))
        self.assertEqual(2 * (result.n_rows + 6 + 2), 11_722)


if __name__ == "__main__":
    unittest.main()
