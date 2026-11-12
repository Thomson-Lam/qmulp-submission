"""The operator both arms are built on, and the walk that runs on it.

Invariants 1-3 and 10 of the paper: ``G = alpha*T + (1-alpha)*P`` with the
factored MuLP restart (seed columns are the borrower's ``c_i``, non-seed
columns are ``v``); ``v`` carries ``1/(L*|D|)`` on every seed layer copy and
nothing elsewhere; the walk is ``W = S_swap (2 Pi - I)``, ``W^2`` per recorded
step, measured on the source register, time-averaged over ``t = 0 .. tau-1``
with ``t = 0`` included; the supra-graph layout is the paper's block form and
an absent value is never a shared attribute. Synthetic fixtures only.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import issparse

from classical.mulp import MultiLayerRanker
from classical.mulp.personalization import FactoredRestart
from quantum.inspired.google_matrix import build_G_from_mulp
from quantum.inspired.mulp import QuantumWalkRanker
from quantum.inspired.szegedy_walk import SzegedyWalk
from tests.oracles import dense_szegedy_projector, dense_walk_unitary, state_matrix_to_vector


def _fixture_df() -> pl.DataFrame:
    """6 borrowers, 4 JOB and 2 REASON states, borrower 3 with JOB unrecorded."""
    return pl.DataFrame(
        {
            "BAD": [1, 0, 1, 0, 0, 1],
            "JOB_A": [1, 0, 0, 0, 1, 0],
            "JOB_B": [0, 1, 0, 0, 0, 1],
            "JOB_C": [0, 0, 1, 0, 0, 0],
            "JOB_D": [0, 0, 0, 0, 0, 0],
            "JOB_Missing": [0, 0, 0, 1, 0, 0],
            "REASON_X": [1, 0, 0, 1, 1, 0],
            "REASON_Y": [0, 1, 1, 0, 0, 1],
        }
    )


SEEDS = [0, 2, 5]


class _FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls._tmp.name) / "fixture.parquet"
        _fixture_df().write_parquet(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _ranker(self, **kwargs) -> MultiLayerRanker:
        ranker = MultiLayerRanker(str(self.path), **{"sparse": True, **kwargs})
        ranker.buildAdjMatrix()
        ranker.buildPersonalizationMatrix()
        return ranker


def _dense_operator(ranker: MultiLayerRanker) -> np.ndarray:
    return np.asarray(ranker.pagerank_operator @ np.eye(ranker.N))


# ---------------------------------------------------------------------------
# Invariants 1 and 2: the operator formula, the restart columns, v
# ---------------------------------------------------------------------------


class OperatorFormulaTests(_FixtureCase):
    def test_operator_is_alpha_t_plus_one_minus_alpha_factored_p(self):
        ranker = self._ranker(influence_type="combined")
        ranker.pageRank(alpha=0.7)
        self.assertIsInstance(ranker.restart_matrix, FactoredRestart)
        self.assertTrue(issparse(ranker.restart_matrix.seed_columns))
        T = ranker.transition_matrix.toarray()
        P = np.column_stack([ranker.restart_matrix.column(j) for j in range(ranker.N)])
        np.testing.assert_allclose(_dense_operator(ranker), 0.7 * T + 0.3 * P, atol=1e-13)
        np.testing.assert_allclose(_dense_operator(ranker).sum(axis=0), 1.0, atol=1e-12)

    def test_v_is_one_over_l_d_on_every_seed_copy_for_every_influence_type(self):
        for influence_type in ("combined", "intra", "inter"):
            with self.subTest(influence_type=influence_type):
                ranker = self._ranker(influence_type=influence_type)
                ranker.buildGoogleOperator(0.85)
                v = ranker._dangling_patch_vector()
                L = ranker.n_layers
                expected = np.zeros(ranker.N)
                for layer in range(L):
                    for seed in SEEDS:
                        expected[ranker.index.borrower_node(layer, seed)] = 1.0 / (L * len(SEEDS))
                np.testing.assert_allclose(v, expected, atol=1e-15)

    def test_seed_columns_are_the_borrowers_own_c_i_and_non_seed_columns_are_v(self):
        """The three influence types differ only in c_i; swapping intra and inter fails here."""
        expected_c = {
            "combined": lambda own, other: {own: 0.5, other: 0.5},
            "intra": lambda own, other: {own: 1.0},
            "inter": lambda own, other: {other: 1.0},
        }
        for influence_type, c_i in expected_c.items():
            with self.subTest(influence_type=influence_type):
                ranker = self._ranker(influence_type=influence_type)
                ranker.buildGoogleOperator(0.85)
                v = ranker._dangling_patch_vector()
                restart = ranker.restart_matrix
                seed_nodes = set()
                for seed in SEEDS:
                    job, reason = ranker.index.borrower_node(0, seed), ranker.index.borrower_node(1, seed)
                    seed_nodes |= {job, reason}
                    for own, other in ((job, reason), (reason, job)):
                        column = np.zeros(ranker.N)
                        for node, mass in c_i(own, other).items():
                            column[node] = mass
                        np.testing.assert_allclose(restart.column(own), column, atol=1e-15)
                for j in range(ranker.N):
                    if j not in seed_nodes:
                        np.testing.assert_allclose(restart.column(j), v, atol=1e-15)
                self.assertEqual(restart.n_seed_columns, 2 * len(SEEDS))

    def test_influence_types_give_distinct_operators_shared_by_both_arms(self):
        operators = {}
        for influence_type in ("inter", "intra", "combined"):
            ranker = self._ranker(influence_type=influence_type)
            ranker.pageRank(alpha=0.85)
            G, _v = build_G_from_mulp(ranker, alpha=0.85)
            np.testing.assert_allclose(_dense_operator(ranker), G, atol=1e-12)
            operators[influence_type] = G
        self.assertFalse(np.allclose(operators["inter"], operators["intra"]))
        self.assertFalse(np.allclose(operators["intra"], operators["combined"]))


class StationaryDistributionTests(_FixtureCase):
    def test_pagerank_solves_g_pi_equals_pi_and_is_a_distribution(self):
        ranker = self._ranker()
        pi = ranker.pageRank(alpha=0.85)
        self.assertLess(float(np.linalg.norm(ranker.pagerank_operator @ pi - pi, 1)), 1e-9)
        self.assertGreaterEqual(float(pi.min()), 0.0)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=12)
        _job, _reason, borrower = ranker.getScores()
        expected = pi[: ranker.n_rows] + pi[ranker.reason_offset : ranker.reason_offset + ranker.n_rows]
        np.testing.assert_allclose(borrower, expected, atol=1e-12)

    def test_finite_horizon_control_averages_powers_of_g_from_v(self):
        ranker = self._ranker()
        ranker.pageRank(alpha=0.85)
        history = ranker.finiteHorizon(4)
        v = ranker._dangling_patch_vector()
        G = _dense_operator(ranker)
        expected = np.stack([np.linalg.matrix_power(G, t) @ v for t in range(4)])
        np.testing.assert_allclose(history, expected, atol=1e-12)

    def test_sparse_and_dense_solutions_agree(self):
        pi_sparse = self._ranker(sparse=True).pageRank(alpha=0.85)
        pi_dense = self._ranker(sparse=False).pageRank(alpha=0.85)
        self.assertLess(float(np.max(np.abs(pi_sparse - pi_dense))), 1e-6)


# ---------------------------------------------------------------------------
# Invariant 10 and the absence rule: the supra-graph layout
# ---------------------------------------------------------------------------


class SupraGraphTests(_FixtureCase):
    def test_block_layout_and_the_four_eq3_blocks(self):
        S = 2.0
        ranker = MultiLayerRanker(str(self.path), sparse=True, stickiness=S)
        A = ranker.buildAdjMatrix().toarray()
        n_rows, size = ranker.n_rows, ranker.index.layer_size
        self.assertEqual((ranker.n_job, ranker.n_reason), (4, 2))
        self.assertEqual(size, n_rows + 4 + 2)
        self.assertEqual(ranker.N, 2 * size)
        A_1, I_12, I_21, A_2 = A[:size, :size], A[:size, size:], A[size:, :size], A[size:, size:]
        for block in (A_1, A_2):
            self.assertTrue(np.allclose(block, block.T))
            self.assertTrue(np.all(block[:n_rows, :n_rows] == 0))
            self.assertTrue(np.all(block[n_rows:, n_rows:] == 0))
        expected = np.zeros((size, size))
        expected[:n_rows, :n_rows] = S * np.eye(n_rows)
        self.assertTrue(np.allclose(I_12, expected) and np.allclose(I_21, expected))

    def test_edges_are_stored_source_row_dest_column(self):
        ranker = MultiLayerRanker(str(self.path), bidirectional=False, sparse=True)
        A = ranker.buildAdjMatrix().toarray()
        job_cat = ranker.index.job_category_node(ranker.job_cols.index("JOB_A"))
        self.assertEqual(A[0, job_cat], 1.0)
        self.assertEqual(A[job_cat, 0], 0.0)
        # The interlayer edge joins the same entity, so it is inserted both ways regardless.
        self.assertEqual(A[0, ranker.reason_offset], 1.0)
        self.assertEqual(A[ranker.reason_offset, 0], 1.0)

    def test_placeholder_columns_are_dangling_and_patched_with_v(self):
        ranker = self._ranker()
        T = ranker.buildTransitionMatrix().toarray()
        n_rows, n_job, size, off = ranker.n_rows, ranker.n_job, ranker.index.layer_size, ranker.reason_offset
        placeholders = list(range(n_rows + n_job, size)) + list(range(off + n_rows, off + n_rows + n_job))
        self.assertEqual(placeholders, ranker.index.all_placeholder_nodes)
        self.assertTrue(np.all(ranker.adj_matrix.toarray()[placeholders, :] == 0))
        v = ranker._dangling_patch_vector()
        for column in placeholders:
            np.testing.assert_allclose(T[:, column], v, atol=1e-12)
        self.assertTrue(np.allclose(T.sum(axis=0), 1.0, atol=1e-12))

    def test_an_unrecorded_field_costs_a_membership_edge_and_nothing_more(self):
        ranker = MultiLayerRanker(str(self.path), sparse=False)
        A = np.asarray(ranker.buildAdjMatrix(), dtype=float)
        self.assertEqual(ranker.job_cols, ["JOB_A", "JOB_B", "JOB_C", "JOB_D"])
        self.assertEqual(ranker.job_missing_cols, ["JOB_Missing"])
        job_nodes = list(ranker.index.category_nodes(0))
        self.assertEqual(float(A[3, job_nodes].sum()), 0.0)  # borrower 3: no JOB edge
        self.assertEqual(float(A[3, ranker.reason_offset + 3]), 1.0)  # still reaches its other copy
        for borrower in (0, 1, 2, 4, 5):
            self.assertEqual(float(A[borrower, job_nodes].sum()), 1.0)
        self.assertEqual(ranker.n_unrecorded_job, 1)

    def test_sparse_and_dense_representations_agree(self):
        sparse, dense_r = self._ranker(sparse=True), self._ranker(sparse=False)
        np.testing.assert_array_equal(sparse.adj_matrix.toarray(), dense_r.adj_matrix)
        np.testing.assert_array_equal(sparse.personalization_matrix.toarray(), dense_r.personalization_matrix)
        np.testing.assert_allclose(sparse.buildTransitionMatrix().toarray(), dense_r.buildTransitionMatrix(), atol=1e-12)


class SingleLayerTests(_FixtureCase):
    def test_job_only_network_has_one_block_and_v_has_1_over_d(self):
        ranker = MultiLayerRanker(str(self.path), sparse=True, layer_names=("JOB",))
        self.assertEqual((ranker.n_layers, ranker.N, ranker.reason_offset), (1, 6 + 4, None))
        ranker.buildAdjMatrix()
        self.assertEqual(ranker.index.all_placeholder_nodes, [])
        ranker.buildPersonalizationMatrix()
        pi = ranker.pageRank(alpha=0.85)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=12)
        v = ranker._dangling_patch_vector()
        for seed in SEEDS:
            self.assertAlmostEqual(float(v[seed]), 1.0 / 3.0)
        with self.assertRaises(ValueError):
            ranker.getScores()

    def test_inter_influence_and_bad_layer_lists_are_refused(self):
        ranker = MultiLayerRanker(str(self.path), sparse=True, layer_names=("JOB",), influence_type="inter")
        ranker.buildAdjMatrix()
        with self.assertRaises(ValueError):
            ranker.buildPersonalizationMatrix()
        for names in (("REASON", "JOB"), ("JOB", "INCOME_BAND"), ()):
            with self.assertRaises(ValueError):
                MultiLayerRanker(str(self.path), layer_names=names)

    def test_the_quantum_walk_runs_on_a_single_layer(self):
        ranker = self._ranker(layer_names=("JOB",))
        walker = QuantumWalkRanker(ranker=ranker, alpha=0.85)
        walker.buildSzegedyWalk()
        self.assertIsNone(walker.szegedy_walk.bl2)
        omega = walker.quantumPageRank(tau=4, validate_probabilities=True)
        self.assertEqual(omega.shape, (6,))
        self.assertLessEqual(float(omega.sum()), 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# Invariant 3: the walk
# ---------------------------------------------------------------------------

G3 = np.array([[0.2, 0.5, 0.1], [0.3, 0.2, 0.6], [0.5, 0.3, 0.3]])
V3 = np.array([0.6, 0.3, 0.1])


class WalkMechanicsTests(unittest.TestCase):
    def test_matrix_free_operations_match_the_dense_reference(self):
        n = 3
        Pi = dense_szegedy_projector(G3)
        R = 2 * Pi - np.eye(n * n)
        W = dense_walk_unitary(G3)
        self.assertTrue(np.allclose(Pi @ Pi, Pi, atol=1e-12))
        self.assertTrue(np.allclose(W.conj().T @ W, np.eye(n * n), atol=1e-12))
        walk = SzegedyWalk(G3, V3, np.array([0]), np.array([1]))
        Psi = walk._build_initial_state()
        self.assertAlmostEqual(np.linalg.norm(Psi), 1.0, places=12)
        vec = state_matrix_to_vector
        self.assertTrue(np.allclose(vec(walk._apply_R_Pi(Psi)), R @ vec(Psi), atol=1e-12))
        self.assertTrue(np.allclose(vec(walk._apply_W(Psi)), W @ vec(Psi), atol=1e-12))
        rng = np.random.default_rng(1)
        random_state = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        self.assertAlmostEqual(np.linalg.norm(walk._advance(random_state)), np.linalg.norm(random_state), places=12)

    def test_default_measure_register_is_source_the_axis_1_marginal(self):
        walk = SzegedyWalk(G3, V3, np.array([0]), np.array([1]))
        self.assertEqual(walk.measure_register, "source")
        Psi = np.array([[0.6, 0.8, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=complex)
        marginal = walk._marginal_supra_node_probs(Psi)
        np.testing.assert_allclose(marginal, (np.abs(Psi) ** 2).sum(axis=1))
        self.assertFalse(np.allclose(marginal, (np.abs(Psi) ** 2).sum(axis=0)))
        np.testing.assert_allclose(walk._marginal_supra_node_probs(Psi, register="target"), [0.36, 0.64, 0.0])

    def test_time_average_includes_t_zero_and_divides_by_tau(self):
        """Hand-computed: q_0 is v summed over layer copies; omega is the plain mean of tau steps."""
        walk = SzegedyWalk(G3, V3, np.array([0]), np.array([1]))
        tau = 5
        omega, history = walk.run_with_history(tau=tau, validate_probabilities=True)
        self.assertEqual(history.shape, (tau, 1))
        np.testing.assert_allclose(history[0], [V3[0] + V3[1]], atol=1e-12)
        np.testing.assert_allclose(omega, (history[0] + history[1:].sum(axis=0)) / tau, atol=1e-14)
        np.testing.assert_allclose(walk.run(tau=tau), omega, atol=1e-14)
        # The reported score is not the endpoint and not an average over t = 1 .. tau.
        self.assertFalse(np.allclose(omega, history[-1]))
        self.assertFalse(np.allclose(omega, history[1:].mean(axis=0)))
        self.assertFalse(np.allclose(walk.run(tau=tau + 1), omega))


if __name__ == "__main__":
    unittest.main()
