"""The synthetic generator and the S1(a) inertness claim on a small instance.

The generator is reproducible by seed and its frame has the schema the
network builders read; the dangling non-reference test individuals of S1(a)
change no other individual's score in either arm (exactly, not statistically);
and the lumped operator reproduces the unlumped step marginals on a generated
two-layer instance. The development references are placed per joint cell by
default; the per-category placement of the S3b sensitivity run changes only the
development labels and realises every category's share, and S3b scores the S3
population itself under both placements.
"""

from __future__ import annotations

import unittest

import numpy as np

from config import DEFAULT_CONFIG
from evaluation.masked_eval import DEVELOPMENT_PARTITION, TEST_PARTITION
from quantum.mechanism.scores_stage import full_vs_reduced_error
from quantum.mechanism.variants import primary_variant, route_variant
from synthetic.experiments import (
    REF_SHARE_MEAN,
    S3_CONFIGS,
    S3B_KAPPA,
    S3B_PARTIAL,
    TEST_FRAC,
    s3_quantities,
    s3b_rows,
    score_instance,
)
from synthetic.generator import generate_attribute_graph

CONFIG = DEFAULT_CONFIG


class GeneratorTests(unittest.TestCase):
    def test_reproducible_by_seed_with_the_expected_schema(self):
        a = generate_attribute_graph(120, 2, (4, 3), 0.5, 0.2, 0.03, 0.25, 0.2, 2.0, 0.05, 0.1, seed=7)
        b = generate_attribute_graph(120, 2, (4, 3), 0.5, 0.2, 0.03, 0.25, 0.2, 2.0, 0.05, 0.1, seed=7)
        self.assertTrue(a.frame.equals(b.frame))
        np.testing.assert_array_equal(a.partition, b.partition)
        self.assertEqual(set(a.frame.columns), {*[f"L1_c{k}" for k in range(4)], "L1_Missing", *[f"L2_c{k}" for k in range(3)],
                                                "L2_Missing", "BAD"})
        self.assertEqual(a.n, 126)
        self.assertEqual(int(a.dangling.sum()), 6)
        # One active column per layer family; partially observed individuals are test-only.
        for layer, levels in (("L1", 4), ("L2", 3)):
            block = a.frame.select([f"{layer}_c{k}" for k in range(levels)] + [f"{layer}_Missing"]).to_numpy()
            self.assertTrue((block.sum(axis=1) == 1).all())
        self.assertTrue((a.partition[a.partially_observed] == "test").all())
        self.assertTrue(a.partially_observed[:120].any())
        c = generate_attribute_graph(120, 2, (4, 3), 0.5, 0.2, 0.03, 0.25, 0.2, 2.0, 0.0, 0.1, seed=7)
        self.assertTrue(c.frame.equals(a.frame.head(120)), "the dangling individuals are appended, the base is unchanged")

    def test_s1a_dangling_non_reference_test_individuals_are_exactly_inert(self):
        base = generate_attribute_graph(150, 1, (5,), 0.5, 0.2, 0.03, 0.2, 0.0, 1.0, 0.0, 0.0, seed=3)
        more = generate_attribute_graph(150, 1, (5,), 0.5, 0.2, 0.03, 0.2, 0.0, 1.0, 0.1, 0.0, seed=3)
        s_base, s_more = score_instance(base, CONFIG), score_instance(more, CONFIG)
        n = base.n
        np.testing.assert_allclose(s_more["classical"][:n], s_base["classical"], atol=1e-12)
        np.testing.assert_allclose(s_more["qppr"][:n], s_base["qppr"], atol=1e-12)
        np.testing.assert_allclose(s_more["classical"][n:], 0.0, atol=1e-12)
        np.testing.assert_allclose(s_more["qppr"][n:], 0.0, atol=1e-12)

    def test_lumped_equals_full_on_a_generated_two_layer_instance(self):
        instance = generate_attribute_graph(80, 2, (3, 2), 0.5, 0.2, 0.03, 0.25, 0.2, 1.0, 0.0, 0.0, seed=11)
        partitions = instance.partitions()
        variant = primary_variant(partitions, instance.spec, CONFIG, partitions.test_idx)
        routed = route_variant(variant, CONFIG, keep_full=True)
        error = full_vs_reduced_error(routed, variant, 8)
        self.assertLess(error["full_vs_reduced_error"], 1e-12)


class PlacementTests(unittest.TestCase):
    def test_category_placement_changes_only_the_development_labels_and_realises_category_shares(self):
        args = (2000, 2, (6, 5), 0.0, 0.2, 0.1, TEST_FRAC, 0.0, 1.0, 0.0, 0.0)
        cell = generate_attribute_graph(*args, seed=5)
        explicit = generate_attribute_graph(*args, seed=5, reference_placement="joint_cell")
        category = generate_attribute_graph(*args, seed=5, reference_placement="category")
        self.assertTrue(cell.frame.equals(explicit.frame), "the default placement is per joint cell")
        self.assertEqual((cell.truth["reference_placement"], category.truth["reference_placement"]), ("joint_cell", "category"))
        np.testing.assert_array_equal(cell.partition, category.partition)
        self.assertTrue(cell.frame.drop("BAD").equals(category.frame.drop("BAD")))
        dev, test = cell.partition == DEVELOPMENT_PARTITION, cell.partition == TEST_PARTITION
        bad_cell, bad_category = cell.frame["BAD"].to_numpy(), category.frame["BAD"].to_numpy()
        np.testing.assert_array_equal(bad_cell[test], bad_category[test])
        self.assertTrue((bad_cell[dev] != bad_category[dev]).any())
        # with two layers the per-cell placement pulls a category's realised share towards the middle of the grid
        self.assertGreater(cell.truth["reference_share_gap"], 0.03)
        self.assertLess(category.truth["reference_share_gap"], 0.03)
        with self.assertRaises(ValueError):
            generate_attribute_graph(*args, seed=5, reference_placement="marginal")

    def test_one_layer_category_placement_realises_each_category_share(self):
        one = generate_attribute_graph(1500, 1, (6,), 0.0, 0.2, 0.1, TEST_FRAC, 0.0, 1.0, 0.0, 0.0, seed=2, reference_placement="category")
        self.assertLess(one.truth["reference_share_gap"], 0.02)

    def test_s3b_scores_the_s3_population_under_both_placements(self):
        rows = s3b_rows(CONFIG, n=300, replicates=1)
        self.assertEqual(len(rows), len(S3_CONFIGS) * len(S3B_PARTIAL) * len(S3B_KAPPA))
        row = next(r for r in rows if r["config"] == "hmeq_like" and r["partial_share"] == 0.0)
        seed = CONFIG.seed_synthetic("S3", 0, "hmeq_like", 0.0, 1.0)
        self.assertEqual((row["experiment"], row["seed"]), ("S3b", seed))
        s3 = generate_attribute_graph(300, 2, S3_CONFIGS["hmeq_like"], 0.5, REF_SHARE_MEAN, 0.03, TEST_FRAC, 0.0, 1.0, 0.0, 0.0, seed)
        expected = s3_quantities(s3, score_instance(s3, CONFIG), CONFIG)["rho_qc_all"]
        np.testing.assert_allclose(row["rho_qc_all_joint_cell"], expected, rtol=0.0, atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(row["d_rho_qc_all"], row["rho_qc_all"] - row["rho_qc_all_joint_cell"], rtol=0.0, atol=1e-12, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
