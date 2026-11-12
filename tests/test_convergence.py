"""The spectral limit, the running average and the convergence point on a small random operator.

On a 20-node random column-stochastic operator (a full operator, so ``K = D``):
the running average by the coefficient recurrence equals the exact walk's
time average step for step; the spectral limit is what the running average
approaches, and ``tau * L1_tau <= c_exact`` at every grid horizon (the bound
of the limit module); ``tau*`` is monotone in both tolerances and reports
``"> tau_max"`` rather than a number when the criterion is never met or is met
only on a tail shorter than ``conv_min_tail_points`` (a Spearman that flips
until the second-last grid point is not a convergence point at the cap); the
limit reports how its ``|d| = 1`` collapse treated the spectrum.
"""

from __future__ import annotations

import unittest
import warnings
from unittest import mock

import numpy as np

from config import DEFAULT_CONFIG
from evaluation.reporting import l1_sum
from quantum.inspired.szegedy_walk import SzegedyWalk
from quantum.lumping_router import discriminant
from quantum.mechanism.convergence import ABOVE_MAX, convergence_grid, tau_star, tau_star_sensitivity
from quantum.mechanism.limit import INNER_GAP_WARN, collapse_diagnostics, limiting_distribution
from quantum.mechanism.running_average import classical_running_averages, running_averages, step_marginals

N = 20
TAU_MAX = 200_000


def _operator(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    G = rng.random((N, N)) ** 3 + 1e-3
    G /= G.sum(axis=0, keepdims=True)
    v = rng.random(N)
    return G, v / v.sum()


class RunningAverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.G, cls.v = _operator()
        cls.K = discriminant(cls.G)
        cls.limit = limiting_distribution(cls.G, cls.v, K=cls.K, d_tol=DEFAULT_CONFIG.d_tol,
                                          phase_tol=DEFAULT_CONFIG.phase_tol, mass_tol=DEFAULT_CONFIG.mass_tol)

    def test_the_recurrence_is_the_exact_walk_step_for_step(self):
        walk = SzegedyWalk(self.G, self.v, np.arange(N))
        exact = walk.supra_node_history(25)
        np.testing.assert_allclose(step_marginals(self.G, self.v, self.K, 25), exact, atol=1e-12)
        averages = running_averages(self.G, self.v, self.K, (1, 7, 25))
        np.testing.assert_allclose(averages[25], exact.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(averages[1], self.v, atol=1e-12)
        classical = classical_running_averages(self.G, self.v, (1, 3))
        np.testing.assert_allclose(classical[3], (self.v + self.G @ self.v + self.G @ (self.G @ self.v)) / 3.0, atol=1e-14)

    def test_the_running_average_approaches_the_limit_within_the_spectral_bound(self):
        grid = [10, 100, 1_000, 10_000, 100_000, TAU_MAX]
        averages = running_averages(self.G, self.v, self.K, grid)
        omega = self.limit.omega
        errors = {tau: l1_sum(averages[tau], omega, decimals=15) for tau in grid}
        for tau, error in errors.items():
            self.assertLessEqual(tau * error, self.limit.c_exact * (1.0 + 1e-9), tau)
        self.assertLessEqual(self.limit.c_exact, self.limit.c_bound * (1.0 + 1e-9))
        self.assertLess(errors[TAU_MAX], errors[1_000])
        self.assertLess(errors[TAU_MAX], 1e-3)
        self.assertGreaterEqual(self.limit.r_active, 2)
        self.assertGreater(self.limit.delta_chord_active, 0.0)
        self.assertEqual(self.limit.r_active, int((self.limit.group_masses > self.limit.active_tol).sum()))


class TauStarTests(unittest.TestCase):
    @staticmethod
    def _rows(l1, rho, ov, ov_self=1.0):
        return [{"tau": t, "l1_tau": a, "rho_tau": b, "ov_tau": c, "ov_self": ov_self}
                for t, a, b, c in zip(range(1, len(l1) + 1), l1, rho, ov)]

    def _star(self, rows, *, tol_l1=0.01, min_spearman=0.99, min_tail_points=1):
        return tau_star(rows, tol_l1=tol_l1, min_spearman=min_spearman, overlap_slack=0.01, min_tail_points=min_tail_points)

    def test_grid_shape(self):
        grid = convergence_grid(12_345)
        self.assertEqual(grid[0], 1)
        self.assertEqual(grid[-1], 12_345)
        self.assertIn(100, grid)
        self.assertIn(1_010, grid)
        self.assertIn(10_100, grid)
        self.assertNotIn(1_005, grid)
        self.assertNotIn(10_050, grid)
        self.assertEqual(len(convergence_grid(50)), 50)

    def test_tau_star_is_a_suffix_point_and_monotone_in_the_tolerances(self):
        l1 = [0.5, 0.05, 0.005, 0.02, 0.004, 0.003, 0.002]
        rho = [0.5, 0.9, 0.99, 0.995, 0.999, 0.999, 0.999]
        ov = [0.3, 0.8, 0.99, 0.995, 1.0, 1.0, 1.0]
        rows = self._rows(l1, rho, ov)
        result = self._star(rows)
        self.assertEqual(result["tau_star"], 5)          # the L1 violation at tau = 4 ends the earlier run
        self.assertEqual(result["tau_star_l1"], 5)
        self.assertEqual(result["tau_star_rank"], 3)
        self.assertEqual(result["last_violation"]["tau"], 4)
        self.assertEqual((result["tail_points"], result["tail_points_rank"]), (3, 5))
        self.assertTrue(result["reached"])
        loose = self._star(rows, tol_l1=0.03)
        self.assertLessEqual(loose["tau_star"], result["tau_star"])
        strict = self._star(rows, tol_l1=0.001)
        self.assertEqual(strict["tau_star"], ABOVE_MAX)
        self.assertFalse(strict["reached"])
        self.assertEqual(strict["last_violation"]["tau"], 7)
        self.assertEqual(strict["tail_points"], 0)
        sensitivity = tau_star_sensitivity(rows, l1_grid=(0.001, 0.01, 0.03), spearman_grid=(0.98, 0.99, 0.995),
                                           overlap_slack=0.01, min_tail_points=1)
        self.assertEqual(len(sensitivity), 9)
        numeric = [r["tau_star"] for r in sensitivity if r["reached"]]
        self.assertTrue(all(isinstance(t, int) for t in numeric))
        # A NaN Spearman (a constant running average) can never satisfy the rank criterion.
        nan_rows = self._rows([0.0, 0.0], [float("nan"), float("nan")], [1.0, 1.0])
        self.assertEqual(self._star(nan_rows)["tau_star"], ABOVE_MAX)

    def test_a_criterion_met_only_at_the_end_of_the_grid_is_not_a_convergence_point(self):
        """The HMEQ two-layer artefact: Spearman flips between 1.000 and 0.987 until the second-last grid point."""
        n = 50
        flipping = [0.987 if i % 2 == 0 else 0.999 for i in range(n)]   # fails at tau = 49 (the second-last point)
        rows = self._rows([0.001] * n, flipping, [1.0] * n)
        required = DEFAULT_CONFIG.conv_min_tail_points
        result = self._star(rows, min_tail_points=required)
        self.assertEqual(result["tau_star"], ABOVE_MAX)
        self.assertFalse(result["reached"])
        self.assertEqual(result["last_violation"]["tau"], n - 1)
        self.assertEqual((result["tail_points"], result["conv_min_tail_points"]), (1, required))
        # The earlier rule (a one-point tail) accepted the cap itself: tau* = tau_max.
        self.assertEqual(self._star(rows, min_tail_points=1)["tau_star"], n)
        # The same holds for the one-sided variants and the sensitivity grid.
        self.assertEqual(result["tau_star_rank"], ABOVE_MAX)
        self.assertEqual(result["tau_star_l1"], 1)
        grid = tau_star_sensitivity(rows, l1_grid=(0.01,), spearman_grid=(0.99,), overlap_slack=0.01, min_tail_points=required)
        self.assertFalse(grid[0]["reached"])

    def test_the_tail_must_hold_at_least_the_required_number_of_grid_points(self):
        n, required = 50, 20
        for tail, expected in ((20, n - 19), (19, ABOVE_MAX), (35, n - 34)):
            rows = self._rows([0.5] * (n - tail) + [0.001] * tail, [0.999] * n, [1.0] * n)
            result = self._star(rows, min_tail_points=required)
            self.assertEqual(result["tau_star"], expected, tail)
            self.assertEqual(result["tail_points"], tail)
            self.assertEqual(result["reached"], tail >= required)
        rows = self._rows([0.001] * n, [0.999] * n, [1.0] * n)
        self.assertEqual(self._star(rows, min_tail_points=required)["tau_star"], 1)      # converged from the first point
        with self.assertRaises(ValueError):
            self._star(rows, min_tail_points=0)
        with self.assertRaises(ValueError):
            self._star(rows[:5], min_tail_points=required)                              # a grid shorter than the rule

    def test_the_tail_rule_is_a_validated_configuration_field(self):
        self.assertEqual(DEFAULT_CONFIG.conv_min_tail_points, 20)
        self.assertIn("conv_min_tail_points", DEFAULT_CONFIG.as_dict())
        with self.assertRaises(ValueError):
            DEFAULT_CONFIG.with_overrides(conv_min_tail_points=0)


class CollapseDiagnosticTests(unittest.TestCase):
    def test_the_inner_spectrum_and_the_fold_are_reported(self):
        d = np.array([-0.5, 0.2, 1.0 - 5e-7, 1.0 - 1e-12, -1.0])
        report = collapse_diagnostics(d, 1e-9)
        self.assertAlmostEqual(report["d_inner_max"], 1.0 - 5e-7, places=12)
        self.assertAlmostEqual(report["d_gap_min"], 5e-7, places=12)
        self.assertEqual(report["d_collapsed"], 2)
        self.assertAlmostEqual(report["d_collapse_dev_max"], 1e-12, places=15)
        self.assertLess(report["d_gap_min"], INNER_GAP_WARN)
        nothing = collapse_diagnostics(np.array([1.0, -1.0]), 1e-9)
        self.assertTrue(np.isnan(nothing["d_inner_max"]) and np.isnan(nothing["d_gap_min"]))
        self.assertEqual(nothing["d_collapsed"], 2)

    def test_the_limit_carries_them_and_warns_when_an_inner_eigenvalue_is_too_close_to_the_edge(self):
        G, v = _operator()
        K = discriminant(G)
        kwargs = dict(K=K, d_tol=DEFAULT_CONFIG.d_tol, phase_tol=DEFAULT_CONFIG.phase_tol, mass_tol=DEFAULT_CONFIG.mass_tol)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # a random dense operator is nowhere near the edge: no warning
            limit = limiting_distribution(G, v, **kwargs)
        gap = 1.0 - np.abs(np.clip(limit.d, -1.0, 1.0))
        inner = gap >= DEFAULT_CONFIG.d_tol
        self.assertAlmostEqual(limit.d_inner_max, float(np.abs(limit.d[inner]).max()), places=15)
        self.assertAlmostEqual(limit.d_gap_min, 1.0 - limit.d_inner_max, places=15)
        self.assertEqual(limit.d_collapsed + int(inner.sum()), N)
        with mock.patch("quantum.mechanism.limit.INNER_GAP_WARN", 0.999):      # exercise the warning path itself
            with self.assertWarns(RuntimeWarning):
                limiting_distribution(G, v, **kwargs)


if __name__ == "__main__":
    unittest.main()
