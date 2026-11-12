"""The mechanism round: the router, the limiting distribution, its spectrum, the control operators, the classical scores.

On the 40-borrower fixture of ``tests/fixtures.py``: (1) the primary operator
routes to the lumped discriminant with both residuals at machine precision,
and the limit on the routed lumped operator equals the limit on the unlumped
operator after unfolding; (2) the interlayer weight zero and the rank-one
control route to the compressed discriminant, under which the lumped limit
still equals the unlumped one while the lumped discriminant's does not;
(3) the Cesàro limit equals the long-horizon running average on both
registers and the gap shrinks with the horizon; (4) the limit is a
distribution, the eigenvalue groups carry the whole initial state, the
closed-form and the direct row masses agree, and every grouped phase is one
the discriminant allows; (5) on a tiny operator the grouped phases are the
eigenphases of the explicit ``N^2 x N^2`` unitary; (6) the control operators
are what their docstrings say; (7) the classical scores on the lumped
operator aggregate the production ranker's.
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np
import polars as pl

from config import DEFAULT_CONFIG, DEFAULT_MECHANISM, LayerSpec, NetworkSpec
from evaluation.masked_eval import DEVELOPMENT_PARTITION, TEST_PARTITION, Partitions
from evaluation.reporting import mean_abs_diff
from quantum.lumped_operator import borrower_layer_positions, build_lumped_matrices, unfold_class_scores
from quantum.lumping import full_lumped_subgraph
from quantum.lumping_router import ROUTE_DBAR, ROUTE_H, discriminant
from quantum.mechanism.classical import stationary
from quantum.mechanism.limit import (
    TWO_PI,
    _components,
    _direct_row_mass,
    _group,
    closed_row_mass,
    limiting_distribution,
    limiting_scores,
)
from quantum.mechanism.running_average import classical_running_averages, running_averages
from quantum.mechanism.variants import (
    eta_variant,
    flattened_variant,
    label_free_variant,
    primary_variant,
    rank_one_restart_variant,
    route_variant,
    single_layer_variants,
    uniform_over_copies,
)
from tests.fixtures import ALPHA, ATOL, JOB_COLS, REASON_COLS, SPEC, LumpingFixture, member_lists
from tests.oracles import dense_walk_unitary

MECH = DEFAULT_MECHANISM
CONFIG = DEFAULT_CONFIG
HIDDEN = np.array([0, 5, 12, 20, 33, 39])
SPEC_2L = NetworkSpec(name="NET_2L", layers=(LayerSpec("JOB", tuple(JOB_COLS)), LayerSpec("REASON", tuple(REASON_COLS))))
SPEC_JOB = NetworkSpec(name="NET_JOB", layers=(LayerSpec("JOB", tuple(JOB_COLS)),))


def _limit(G, v, **overrides):
    return limiting_distribution(
        G, v, d_tol=CONFIG.d_tol, phase_tol=CONFIG.phase_tol, mass_tol=CONFIG.mass_tol, **overrides
    )


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    diff = np.abs((a - b) % TWO_PI)
    return np.minimum(diff, TWO_PI - diff)


def _partitions(frame_pd) -> Partitions:
    """The fixture frame as a ``Partitions`` with ``HIDDEN`` as the test partition."""
    frame = pl.DataFrame(frame_pd.to_dict(orient="list"))
    partition = np.full(frame.height, DEVELOPMENT_PARTITION, dtype=object)
    partition[HIDDEN] = TEST_PARTITION
    return Partitions(
        frame=frame, borrower_ids=np.arange(frame.height), partition=partition.astype(str),
        target_col="BAD", processed_path=None, split_path=None,
    )


def _full_limit_unfolded(variant, routed) -> np.ndarray:
    """The limit on the unlumped operator (``K = D``), unfolded to borrowers."""
    limit = _limit(routed.G_full, routed.v_full)
    _ids, scores = unfold_class_scores(variant.nodes_full, limit.omega)
    return scores


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = LumpingFixture(cls._tmp.name)
        cls.partitions = _partitions(cls.fx.df)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_the_primary_operator_routes_to_the_lumped_discriminant_exactly(self):
        variant = primary_variant(self.partitions, SPEC_2L, CONFIG, HIDDEN)
        routed = route_variant(variant, CONFIG, keep_full=True)
        self.assertEqual(routed.route, ROUTE_DBAR)
        self.assertLess(routed.residuals["lumpability_residual"], CONFIG.router_tol)
        self.assertLess(routed.residuals["compatibility_residual"], CONFIG.router_tol)
        self.assertLess(routed.residuals["h_minus_dbar_max"], 1e-11)
        lumped = limiting_scores(variant.nodes, routed, CONFIG)
        np.testing.assert_allclose(lumped.omega_source, _full_limit_unfolded(variant, routed), atol=1e-10)
        self.assertEqual(lumped.route, ROUTE_DBAR)

    def test_eta_zero_and_the_rank_one_control_need_the_compressed_discriminant(self):
        # Both controls break compatibility only through a reference class of two or more with
        # a dangling copy (unrecorded field, column patched with v). At eta = 0 the class's own
        # restart columns then differ inside one block pair; under the rank-one restart the
        # copies' out-degrees differ across layers (a dangling copy has none), and with equal
        # degrees (one category edge, one interlayer edge per copy) the restart stays
        # compatible. The fixture has no such class, so the two development borrowers with an
        # unrecorded REASON are made references for this case.
        frame = self.fx.df.copy()
        frame.loc[[37, 38], "BAD"] = 1
        incompatible = _partitions(frame)
        for variant in (eta_variant(incompatible, SPEC_2L, CONFIG, HIDDEN, 0.0),
                        rank_one_restart_variant(incompatible, SPEC_2L, CONFIG, HIDDEN)):
            routed = route_variant(variant, CONFIG, keep_full=True)
            self.assertEqual(routed.route, ROUTE_H, variant.label)
            self.assertGreater(routed.residuals["compatibility_residual"], CONFIG.router_tol, variant.label)
            self.assertLess(routed.residuals["lumpability_residual"], CONFIG.router_tol, variant.label)
            full = _full_limit_unfolded(variant, routed)
            with_h = limiting_scores(variant.nodes, routed, CONFIG).omega_source
            np.testing.assert_allclose(with_h, full, atol=1e-10, err_msg=variant.label)
            wrong = _limit(routed.G, routed.v, K=routed.D_bar)
            _ids, with_dbar = unfold_class_scores(variant.nodes, wrong.omega)
            self.assertGreater(float(np.max(np.abs(with_dbar - full))), 1e-8, variant.label)
        # On the fixture as given every partially observed reference is a class of one, so
        # both controls stay compatible and the router keeps the lumped discriminant.
        for variant in (eta_variant(self.partitions, SPEC_2L, CONFIG, HIDDEN, 0.0),
                        rank_one_restart_variant(self.partitions, SPEC_2L, CONFIG, HIDDEN)):
            as_given = route_variant(variant, CONFIG, keep_full=True)
            self.assertEqual(as_given.route, ROUTE_DBAR, variant.label)
            self.assertLess(as_given.residuals["h_minus_dbar_max"], 1e-11, variant.label)

    def test_the_label_free_operator_routes_and_carries_no_label(self):
        variant = label_free_variant(self.partitions, SPEC_2L, CONFIG, HIDDEN)
        routed = route_variant(variant, CONFIG, keep_full=True)
        self.assertIn(routed.route, (ROUTE_DBAR, ROUTE_H))
        u = uniform_over_copies(variant.nodes)
        np.testing.assert_allclose(routed.v, u)
        # G[:, j] = alpha*T[:, j] + (1-alpha)*u for every column j, reference classes included.
        position = {int(n): p for p, n in enumerate(variant.nodes["node_id"].astype(int))}
        A = np.zeros((len(variant.nodes), len(variant.nodes)))
        for row in variant.edges.itertuples(index=False):
            A[position[int(row.source)], position[int(row.target)]] = float(row.weight)
        T = np.array([row / row.sum() if row.sum() > 0 else u for row in A]).T
        np.testing.assert_allclose(routed.G, ALPHA * T + (1.0 - ALPHA) * u[:, np.newaxis], atol=ATOL)
        np.testing.assert_allclose(limiting_scores(variant.nodes, routed, CONFIG).omega_source,
                                   _full_limit_unfolded(variant, routed), atol=1e-10)


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fx = LumpingFixture(cls._tmp.name)
        cls.G, cls.v = build_lumped_matrices(cls.fx.nodes_lump, cls.fx.edges_lump, alpha=ALPHA)
        cls.K = discriminant(cls.G)
        cls.bl1, cls.bl2 = borrower_layer_positions(cls.fx.nodes_lump)
        cls.limit = _limit(cls.G, cls.v)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_limit_equals_the_long_horizon_average_on_both_registers(self):
        for register, omega in (("source", self.limit.omega), ("target", self.limit.omega_target)):
            pbar = running_averages(self.G, self.v, self.K, (2_000, 20_000), register=register)
            short = mean_abs_diff(pbar[2_000], omega, decimals=15)
            long = mean_abs_diff(pbar[20_000], omega, decimals=15)
            self.assertLess(long, MECH.fixture_l1_tol, register)
            self.assertGreater(short, long, register)

    def test_mass_grouping_and_the_two_row_mass_forms(self):
        G, v, K = self.G, self.v, self.K
        self.assertAlmostEqual(float(self.limit.omega.sum()), 1.0, delta=CONFIG.mass_tol)
        self.assertAlmostEqual(float(self.limit.omega_target.sum()), 1.0, delta=CONFIG.mass_tol)
        self.assertAlmostEqual(float(self.limit.group_masses.sum()), 1.0, delta=CONFIG.mass_tol)
        M = np.sqrt(G)
        phases, Ys, Zs, d = _components(K, np.sqrt(v), d_tol=CONFIG.d_tol)
        group_phases, Yg, Zg = _group(phases, Ys, Zs, phase_tol=CONFIG.phase_tol)
        np.testing.assert_allclose(group_phases, self.limit.phases)
        for Y, Z in zip(Yg, Zg):
            for register in ("source", "target"):
                np.testing.assert_allclose(
                    _direct_row_mass(Y, Z, M, register=register),
                    closed_row_mass(Y, Z, G, K, register=register), atol=1e-12,
                )
        theta = np.arccos(np.clip(d, -1.0, 1.0))
        allowed = np.concatenate([[0.0], (2.0 * theta) % TWO_PI, (-2.0 * theta) % TWO_PI])
        for phase in group_phases:
            self.assertLess(float(_circular_distance(allowed, phase).min()), CONFIG.phase_tol)
        self.assertLessEqual(self.limit.r_active, self.limit.n_groups_all)
        self.assertGreater(self.limit.c_exact, 0.0)
        self.assertLessEqual(self.limit.c_exact, self.limit.c_bound * (1.0 + 1e-9))

    def test_grouped_phases_are_the_eigenphases_of_the_explicit_unitary(self):
        rng = np.random.default_rng(7)
        n = 5
        G = rng.random((n, n)) + 0.05
        G /= G.sum(axis=0, keepdims=True)
        v = rng.random(n)
        v /= v.sum()
        W = dense_walk_unitary(G)
        explicit = np.angle(np.linalg.eigvals(W @ W)) % TWO_PI
        explicit = explicit[_circular_distance(explicit, 0.0) > 1e-9]
        grouped = _limit(G, v).phases
        grouped = grouped[_circular_distance(grouped, 0.0) > 1e-9]
        self.assertGreater(grouped.size, 0)
        for phase in explicit:
            self.assertLess(float(_circular_distance(grouped, phase).min()), 1e-9)
        for phase in grouped:
            self.assertLess(float(_circular_distance(explicit, phase).min()), 1e-9)

    def test_the_control_operators_are_what_they_say(self):
        fx = self.fx
        partitions = _partitions(fx.df)
        # eta = 0: no interlayer edge, dangling copies, still column-stochastic.
        zero = eta_variant(partitions, SPEC_2L, CONFIG, HIDDEN, 0.0)
        self.assertFalse(zero.edges["relation"].str.startswith("stickiness").any())
        self.assertGreater(zero.n_dangling_columns, 0)
        G0, _v0 = build_lumped_matrices(zero.nodes, zero.edges, alpha=ALPHA)
        np.testing.assert_allclose(G0.sum(axis=0), 1.0, atol=ATOL)
        # eta at the frozen stickiness is the primary graph, frame for frame.
        one = eta_variant(partitions, SPEC_2L, CONFIG, HIDDEN, CONFIG.stickiness)
        nodes, edges = full_lumped_subgraph(partitions, SPEC_2L, CONFIG, HIDDEN)
        self.assertTrue(one.nodes.equals(nodes) and one.edges.equals(edges))
        # The unlumped graph covers every borrower once per layer.
        self.assertEqual(int((one.nodes_full["node_type"] == "borrower_copy").sum()), 2 * len(fx.df))
        # The flattened monoplex: one node per class, both families, one layer; lumped and unlumped alike.
        flat = flattened_variant(partitions, SPEC_2L, CONFIG, HIDDEN)
        n_classes = int((nodes["node_type"] == "borrower_copy").sum()) // 2
        self.assertEqual(int((flat.nodes["node_type"] == "borrower_copy").sum()), n_classes)
        self.assertEqual(int((flat.nodes_full["node_type"] == "borrower_copy").sum()), len(fx.df))
        families = {str(label).split(":")[0] for label in flat.nodes.loc[flat.nodes["node_type"] == "category", "label"]}
        self.assertEqual(families, {"JOB", "REASON"})
        self.assertEqual(len(borrower_layer_positions(flat.nodes)), 1)
        routed = route_variant(flat, CONFIG, keep_full=True)
        ids, _scores = unfold_class_scores(flat.nodes, stationary(routed.G))
        np.testing.assert_array_equal(ids, np.arange(len(fx.df)))
        np.testing.assert_allclose(limiting_scores(flat.nodes, routed, CONFIG).omega_source,
                                   _full_limit_unfolded(flat, routed), atol=1e-10)
        # recorded_only drops exactly the borrowers unrecorded in the layer.
        job = single_layer_variants(partitions, SPEC_JOB, CONFIG, HIDDEN)
        unrecorded = [i for i, (job_state, _reason, _bad) in enumerate(SPEC) if job_state == "-"]
        self.assertEqual(job["recorded_only"].n_borrowers_dropped, len(unrecorded))
        retained = {m for ms in member_lists(job["recorded_only"].nodes).values() for m in ms}
        self.assertEqual(retained, set(range(len(fx.df))) - set(unrecorded))
        self.assertEqual(job["recorded_only"].n_dangling_columns, 0)
        self.assertGreater(job["as_is"].n_dangling_columns, 0)
        self.assertTrue(job["uniform_dangling"].uniform_dangling)
        routed_u = route_variant(job["uniform_dangling"], CONFIG, keep_full=True)
        np.testing.assert_allclose(limiting_scores(job["uniform_dangling"].nodes, routed_u, CONFIG).omega_source,
                                   _full_limit_unfolded(job["uniform_dangling"], routed_u), atol=1e-10)

    def test_classical_scores_on_the_lumped_operator_aggregate_the_ranker(self):
        fx = self.fx
        fx.assert_aggregates(self, fx.nodes_lump, stationary(self.G), fx.pi_full)
        tau = 12
        reduced = classical_running_averages(self.G, self.v, (tau,))[tau]
        full = np.asarray(fx.ranker.finiteHorizon(tau), dtype=float).mean(axis=0)
        fx.assert_aggregates(self, fx.nodes_lump, reduced, full)


if __name__ == "__main__":
    unittest.main()
