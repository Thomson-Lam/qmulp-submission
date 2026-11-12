"""Exact lumpability of the reduced operator, and the key it is lumped on.

The reduced arm exists to run a graph that no longer fits on a device by
merging borrowers the operator cannot tell apart. That is only a legitimate
reduction if it is *exact*: the lumped node's score must equal the sum of its
members' scores in the unlumped graph. Seeds and non-seeds are not
interchangeable -- a seed owns a personalized restart column, a non-seed
restarts to ``v`` -- so the key is the joint triple ``(JOB, REASON, BAD_masked)``
and never a marginal or the true label.

Asserted here: (1) the operator built from ``(nodes, edges)`` reproduces the
production ``MultiLayerRanker`` operator; (2) ``lumping_level="none"``
reproduces the unlumped node and edge sets exactly; (3) lumped stationary
probability aggregates over members; (4) the tau-averaged Szegedy score
aggregates the same way and unfolds to every borrower; (5) the reduced
personalization is size-weighted; (6) the key is the masked triple, so a
hidden true label changes nothing; (7) an absent value is never a shared
attribute; (8) the pre-fix forward membership weight breaks exactness.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import pandas as pd
import pytest

from quantum.lumped_operator import (
    borrower_layer_positions,
    build_lumped_matrices,
    personalization_vector,
    unfold_class_scores,
)
from quantum.lumping import build_lumped_subgraph, build_topk_lumped_subgraph
from tests.fixtures import (
    ALPHA,
    ATOL,
    JOB_COLS,
    REASON_COLS,
    LumpingFixture,
    category_positions,
    member_lists,
    stationary_eig,
    tau_average,
)


@pytest.mark.slow
class ExactLumpabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = LumpingFixture(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_unlumped_operator_matches_multilayerranker(self):
        """Without this, a mismatch later could not be blamed on the lumping.

        The placeholder segments receive no edge and no restart mass, so their
        stationary probability is exactly zero and the reduced operator over
        real nodes is comparable with the full one.
        """
        fx = self.fx
        placeholders = [i for i, label in enumerate(fx.ranker.labels) if "unused_" in label]
        self.assertTrue(placeholders, "the fixture must exercise the placeholder segments")
        np.testing.assert_allclose(fx.pi_full[placeholders], 0.0, atol=ATOL)
        G_none, _v = build_lumped_matrices(fx.nodes_none, fx.edges_none, alpha=ALPHA)
        for members in member_lists(fx.nodes_none).values():
            self.assertEqual(len(members), 1, "lumping_level='none' must not merge")
        fx.assert_aggregates(self, fx.nodes_none, stationary_eig(G_none), fx.pi_full)

    def test_no_lumping_reproduces_the_unlumped_edge_set_with_weights(self):
        fx = self.fx
        self.assertEqual(len(category_positions(fx.nodes_none)), fx.ranker.n_job + fx.ranker.n_reason)
        to_full: dict[int, int] = {}
        node_ids = [int(x) for x in fx.nodes_none["node_id"].tolist()]
        for position, members in member_lists(fx.nodes_none).items():
            to_full[node_ids[position]] = fx.node_of(fx.nodes_none.iloc[position]["layer"])(members[0])
        for label, position in category_positions(fx.nodes_none).items():
            to_full[node_ids[position]] = fx.category_node(label)
        reduced_edges = {(to_full[int(r.source)], to_full[int(r.target)], round(float(r.weight), 12))
                         for r in fx.edges_none.itertuples(index=False)}
        adjacency = np.asarray(fx.ranker.adj_matrix, dtype=float)
        full_edges = {(int(i), int(j), round(float(adjacency[i, j]), 12)) for i, j in zip(*np.nonzero(adjacency))}
        self.assertEqual(reduced_edges, full_edges)

    def test_lumped_stationary_probability_aggregates_over_members(self):
        fx = self.fx
        self.assertTrue([m for m in member_lists(fx.nodes_lump).values() if len(m) > 1], "the fixture must actually merge")
        G_lump, _v = build_lumped_matrices(fx.nodes_lump, fx.edges_lump, alpha=ALPHA)
        fx.assert_aggregates(self, fx.nodes_lump, stationary_eig(G_lump), fx.pi_full)

    def test_tau_averaged_quantum_score_aggregates_and_unfolds_to_every_borrower(self):
        """``sqrt`` does not commute with summation, so this is measured, not inferred.

        Members of a group are related by a graph automorphism and the
        personalized initial state is symmetric under it, which keeps the
        state in the symmetric subspace where the walk is unitarily equivalent
        to the lumped walk.
        """
        fx = self.fx
        omega_full = tau_average(fx.G_full, fx.v_full, fx.bl1_full, fx.bl2_full)
        G_lump, v_lump = build_lumped_matrices(fx.nodes_lump, fx.edges_lump, alpha=ALPHA)
        bl1, bl2 = borrower_layer_positions(fx.nodes_lump)
        omega_lump = tau_average(G_lump, v_lump, bl1, bl2)
        fx.assert_aggregates(self, fx.nodes_lump, omega_lump, omega_full)

        ids, unfolded_q = unfold_class_scores(fx.nodes_lump, omega_lump)
        np.testing.assert_array_equal(ids, np.arange(len(fx.df)))
        np.testing.assert_allclose(unfolded_q, omega_full[fx.bl1_full] + omega_full[fx.bl2_full], atol=ATOL)
        _ids, unfolded_pi = unfold_class_scores(fx.nodes_lump, stationary_eig(G_lump))
        np.testing.assert_allclose(unfolded_pi, np.asarray(fx.ranker.aggregateBorrowers(fx.pi_full)), atol=ATOL)

    def test_reduced_personalization_is_the_member_summed_full_vector(self):
        fx = self.fx
        v_lump = personalization_vector(fx.nodes_lump)
        expected = np.zeros(len(fx.nodes_lump), dtype=float)
        for position, members in member_lists(fx.nodes_lump).items():
            node_of = fx.node_of(fx.nodes_lump.iloc[position]["layer"])
            expected[position] = float(sum(fx.v_full[node_of(m)] for m in members))
        np.testing.assert_allclose(v_lump, expected / expected.sum(), atol=ATOL)
        sizes = {p: len(m) for p, m in member_lists(fx.nodes_lump).items() if v_lump[p] > 0}
        biggest, smallest = max(sizes, key=sizes.get), min(sizes, key=sizes.get)
        self.assertGreater(sizes[biggest], sizes[smallest], "fixture needs uneven seed groups")
        self.assertAlmostEqual(v_lump[biggest] / v_lump[smallest], sizes[biggest] / sizes[smallest], delta=1e-12)

    def test_the_key_is_the_masked_triple_so_a_hidden_true_label_changes_nothing(self):
        fx = self.fx
        bad = fx.df["BAD"].to_numpy().astype(int)
        for members in member_lists(fx.nodes_lump).values():
            self.assertEqual(len({int(bad[m]) for m in members}), 1, "seed and non-seed never share a class")
        hidden = np.array([0, 5, 12, 20, 33, 39])
        levels = bad.copy()
        levels[hidden] = -1
        nodes_a, edges_a = build_topk_lumped_subgraph(fx.df, JOB_COLS, REASON_COLS, fx.borrower_ids,
                                                      lumping_level="attr_bad", bad_masked=levels)
        flipped = fx.df.copy()
        flipped.loc[hidden, "BAD"] = 1 - flipped.loc[hidden, "BAD"]
        nodes_b, edges_b = build_topk_lumped_subgraph(flipped, JOB_COLS, REASON_COLS, fx.borrower_ids,
                                                      lumping_level="attr_bad", bad_masked=levels)
        pd.testing.assert_frame_equal(nodes_a, nodes_b)
        pd.testing.assert_frame_equal(edges_a, edges_b)
        for members in member_lists(nodes_a).values():
            self.assertEqual(len({int(levels[m]) for m in members}), 1)
        seed_nodes = nodes_a[(nodes_a["node_type"] == "borrower_copy") & (nodes_a["BAD"] == 1)]
        for members in seed_nodes["member_borrower_ids"]:
            self.assertFalse({int(m) for m in members.split(";")} & set(hidden.tolist()))
        with self.assertRaises(ValueError):
            build_topk_lumped_subgraph(fx.df, JOB_COLS, REASON_COLS, [0, 1], bad_masked=np.full(40, 2))

    def test_the_old_forward_membership_weight_breaks_exactness(self):
        """Injects weight ``m`` on group -> category, as the code used to write. If this passes, the bug is back."""
        fx = self.fx
        sizes = {int(fx.nodes_lump.iloc[p]["node_id"]): len(m) for p, m in member_lists(fx.nodes_lump).items()}
        broken = fx.edges_lump.copy()
        forward = broken["relation"].isin(["job_membership", "reason_membership"])
        broken.loc[forward, "weight"] = [float(sizes[int(s)]) for s in broken.loc[forward, "source"]]
        self.assertGreater(float(broken.loc[forward, "weight"].max()), 1.0)
        pi_broken = stationary_eig(build_lumped_matrices(fx.nodes_lump, broken, alpha=ALPHA)[0])
        mismatches = 0
        for position, members in member_lists(fx.nodes_lump).items():
            node_of = fx.node_of(fx.nodes_lump.iloc[position]["layer"])
            if abs(pi_broken[position] - float(sum(fx.pi_full[node_of(m)] for m in members))) > 1e-6:
                mismatches += 1
        self.assertGreater(mismatches, 0)


class ClassKeyingTests(unittest.TestCase):
    """The absence rule and the layer rule of the class builder, on a four-borrower frame."""

    JOB = ["JOB_A", "JOB_Missing"]
    REASON = ["REASON_X", "REASON_Missing"]

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({"JOB_A": [1, 0, 1, 0], "JOB_Missing": [0, 1, 0, 1],
                             "REASON_X": [1, 1, 0, 0], "REASON_Missing": [0, 0, 1, 1], "BAD": [1, 1, 1, 1]})

    @staticmethod
    def _incident(edges: pd.DataFrame, node_id: int) -> set[str]:
        return set(edges.loc[(edges["source"] == node_id) | (edges["target"] == node_id), "relation"])

    def test_an_absent_value_gets_no_category_node_and_no_edge(self):
        nodes, edges = build_topk_lumped_subgraph(self._frame(), self.JOB, self.REASON, [0, 1, 2, 3], lumping_level="none")
        labels = set(nodes.loc[nodes["node_type"] == "category", "label"])
        self.assertEqual(labels, {"JOB:JOB_A", "REASON:REASON_X"})
        # node ids: 0-3 JOB copies, 4-7 REASON copies.
        self.assertIn("job_membership", self._incident(edges, 0))
        self.assertNotIn("job_membership", self._incident(edges, 1))
        self.assertNotIn("reason_membership", self._incident(edges, 6))
        neighbours = {n: set(edges.loc[edges["source"] == n, "target"]) | set(edges.loc[edges["target"] == n, "source"]) for n in (1, 3)}
        self.assertEqual(neighbours[1] & neighbours[3], set(), "borrowers sharing only a blank are not connected")
        valid = set(nodes["node_id"].tolist())
        self.assertTrue(set(edges["source"]).issubset(valid) and set(edges["target"]).issubset(valid))

    def test_one_or_two_layers_only(self):
        nodes, edges = build_lumped_subgraph(self._frame(), [("JOB", self.JOB)], [0, 1, 2, 3], lumping_level="attr_bad")
        self.assertEqual(set(nodes["layer"]), {"JOB"})
        self.assertNotIn("stickiness", set(edges["relation"]))
        with self.assertRaises(ValueError):
            build_lumped_subgraph(self._frame().assign(BAND_Q1=[1, 0, 1, 0]),
                                  [("JOB", self.JOB), ("REASON", self.REASON), ("BAND", ["BAND_Q1"])], [0, 1, 2, 3])
        for level in ("job_bad", "bad_only", "bogus"):
            with self.assertRaises(ValueError, msg=level):
                build_topk_lumped_subgraph(self._frame(), self.JOB, self.REASON, [0, 1], lumping_level=level)


if __name__ == "__main__":
    unittest.main()
