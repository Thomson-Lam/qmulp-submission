"""The single-layer proposition on random bipartite graphs.

(1) The classical score of every non-reference member of a category equals
``alpha f_c R / [(1-alpha)(1+alpha-alpha f_c)]`` to 1e-10 relative; (2) adding
non-reference individuals without a category changes no other individual's
score in either arm (they are inert); (3) two categories with equal reference
share and different sizes share one classical level and get two QPPR limits.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import polars as pl

from config import DEFAULT_CONFIG, LayerSpec, NetworkSpec
from evaluation.context import score_operator
from evaluation.masked_eval import DEVELOPMENT_PARTITION, TEST_PARTITION, Partitions
from quantum.lumping import UNKNOWN_LABEL
from quantum.mechanism.single_layer_share import category_table, equal_share_pairs
from quantum.mechanism.variants import primary_variant

CONFIG = DEFAULT_CONFIG
ALPHA = CONFIG.resolved_alpha()


def _frame(sizes: list[int], ref_counts: list[int], test_counts: list[int], n_dangling_test: int = 0, n_dangling_ref: int = 0):
    """A single-layer frame: category c has ``sizes[c]`` members, ``ref_counts[c]`` references, ``test_counts[c]`` test members.

    ``n_dangling_test`` non-reference test individuals and ``n_dangling_ref`` development *references* carry no
    category (the absent state only): their columns are dangling.
    """
    columns = [f"CAT_c{c}" for c in range(len(sizes))] + ["CAT_Missing"]
    rows, partition = [], []
    for c, (n, s, t) in enumerate(zip(sizes, ref_counts, test_counts)):
        for i in range(n):
            row = {col: 0 for col in columns}
            row[f"CAT_c{c}"] = 1
            row["BAD"] = 1 if i < s else 0
            rows.append(row)
            partition.append(TEST_PARTITION if i >= n - t else DEVELOPMENT_PARTITION)
    for _ in range(n_dangling_test):
        row = {col: 0 for col in columns}
        row["CAT_Missing"] = 1
        row["BAD"] = 0
        rows.append(row)
        partition.append(TEST_PARTITION)
    for _ in range(n_dangling_ref):
        row = {col: 0 for col in columns}
        row["CAT_Missing"] = 1
        row["BAD"] = 1
        rows.append(row)
        partition.append(DEVELOPMENT_PARTITION)
    frame = pl.DataFrame(pd.DataFrame(rows, columns=[*columns, "BAD"]).to_dict(orient="list"))
    parts = Partitions(frame=frame, borrower_ids=np.arange(frame.height), partition=np.array(partition, dtype=str),
                       target_col="BAD", processed_path=None, split_path=None)
    spec = NetworkSpec(name="NET_ONE", layers=(LayerSpec("CAT", tuple(columns)),))
    return parts, spec


def _scored(parts, spec):
    variant = primary_variant(parts, spec, CONFIG, parts.test_idx)
    return score_operator(variant, CONFIG)


class SingleLayerShareTests(unittest.TestCase):
    def test_the_formula_holds_on_random_categories(self):
        rng = np.random.default_rng(3)
        for trial in range(3):
            sizes = rng.integers(4, 30, size=6).tolist()
            refs = [int(rng.integers(1, max(2, n // 2))) for n in sizes]
            tests = [int(rng.integers(1, max(2, n - s))) for n, s in zip(sizes, refs)]
            parts, spec = _frame(sizes, refs, tests)
            records = category_table(_scored(parts, spec), ALPHA, CONFIG.tie_tolerance)
            for r in records:
                if r["is_dangling_family"]:
                    continue
                self.assertLess(r["rel_error"], 1e-10, (trial, r["category"]))
            # The classical ranks follow the share ranks exactly.
            by_f = [r["rank_by_f"] for r in records if not r["is_dangling_family"]]
            by_c = [r["rank_by_classical"] for r in records if not r["is_dangling_family"]]
            self.assertEqual(by_f, by_c)

    def test_the_formula_holds_with_dangling_development_references(self):
        """A reference without a category adds ``alpha pi_dangling / |D|`` to the common constant (Prop. 3b).

        The constant without that term (the earlier one) is wrong by a visible margin on this fixture, so the
        fixture does exercise the term; with it every category matches to 1e-10 relative.
        """
        rng = np.random.default_rng(5)
        for trial, (n_ref_dangling, n_test_dangling) in enumerate(((3, 0), (6, 4), (1, 7))):
            sizes = rng.integers(6, 30, size=5).tolist()
            refs = [int(rng.integers(1, max(2, n // 2))) for n in sizes]
            tests = [int(rng.integers(1, max(2, n - s))) for n, s in zip(sizes, refs)]
            parts, spec = _frame(sizes, refs, tests, n_dangling_test=n_test_dangling, n_dangling_ref=n_ref_dangling)
            records = category_table(_scored(parts, spec), ALPHA, CONFIG.tie_tolerance)
            genuine = [r for r in records if not r["is_dangling_family"]]
            self.assertGreater(genuine[0]["pi_dangling"], 0.0, trial)
            self.assertEqual(genuine[0]["n_ref"], sum(refs) + n_ref_dangling)
            old_constant = (1.0 - ALPHA) * (1.0 - genuine[0]["seed_mass"]) / genuine[0]["n_ref"]
            self.assertGreater(abs(genuine[0]["R"] - old_constant) / genuine[0]["R"], 1e-3, trial)
            for r in genuine:
                self.assertLess(r["rel_error"], 1e-10, (trial, r["category"]))
            by_f = [r["rank_by_f"] for r in genuine]
            by_c = [r["rank_by_classical"] for r in genuine]
            self.assertEqual(by_f, by_c)

    def test_dangling_non_reference_individuals_are_inert(self):
        sizes, refs, tests = [10, 7, 12], [2, 1, 3], [3, 2, 4]
        base_parts, spec = _frame(sizes, refs, tests)
        more_parts, _ = _frame(sizes, refs, tests, n_dangling_test=5)
        base, more = _scored(base_parts, spec), _scored(more_parts, spec)
        n = base_parts.frame.height
        np.testing.assert_allclose(more.classical[:n], base.classical, atol=1e-14)
        np.testing.assert_allclose(more.quantum.omega_source[:n], base.quantum.omega_source, atol=1e-14)
        np.testing.assert_allclose(more.classical[n:], 0.0, atol=1e-14)
        np.testing.assert_allclose(more.quantum.omega_source[n:], 0.0, atol=1e-14)

    def test_equal_share_different_size_categories_share_one_classical_level_and_two_qppr_levels(self):
        parts, spec = _frame([10, 20, 15], [2, 4, 1], [3, 6, 4])
        records = category_table(_scored(parts, spec), ALPHA, CONFIG.tie_tolerance)
        pairs = equal_share_pairs(records)
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertAlmostEqual(pair["classical_a"], pair["classical_b"], places=12)
        self.assertGreater(abs(pair["qppr_a"] - pair["qppr_b"]), 1e-9)
        self.assertEqual({pair["n_a"], pair["n_b"]}, {10, 20})
        # The masked test members count towards n_c, so the share is s_c / n_c on the full membership.
        self.assertEqual([r["n_c"] for r in records if not r["is_dangling_family"]], [10, 20, 15])
        self.assertTrue(all(r["n_test"] > 0 for r in records))
        self.assertTrue(all(int(r["s_c"]) >= 1 for r in records))
        self.assertEqual(UNKNOWN_LABEL, -1)


if __name__ == "__main__":
    unittest.main()
