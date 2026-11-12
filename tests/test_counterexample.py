"""Table EC.1 on the eight-individual counterexample: the router's residuals, routes and the exactness of H.

The multilayer restart at ``eta = 1`` and the rank-one restart at ``eta = 0``
are compatible (residual at machine precision, route ``D-bar``, the reduced
time average equals the unlumped one); the rank-one restart at ``eta = 1``
and the multilayer restart at ``eta = 0`` are not (route ``H``), the lumped
discriminant reproduces the unlumped time average only to the recorded error,
and the compressed discriminant reproduces it to machine precision in every
case. The expected magnitudes are the manuscript's.
"""

from __future__ import annotations

import unittest

from quantum.mechanism.variants import COMBINED_RESTART, RANK_ONE_RESTART
from scripts.ec_counterexample import EXPECTED, compute_rows

EXACT = 1e-12


class CounterexampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = {(row["eta"], row["restart"]): row for row in compute_rows()}

    def row(self, eta: float, mode: str) -> dict:
        return self.rows[(eta, "multilayer" if mode == COMBINED_RESTART else "rank-one")]

    def test_sizes(self):
        for row in self.rows.values():
            self.assertEqual((row["n_full"], row["n_reduced"]), (18, 10))
            self.assertLess(row["lumpability_residual"], EXACT)

    def test_compatible_cases_route_to_the_lumped_discriminant(self):
        for eta, mode in ((1.0, COMBINED_RESTART), (0.0, RANK_ONE_RESTART)):
            row = self.row(eta, mode)
            self.assertEqual(row["route"], "D_bar", (eta, mode))
            self.assertLess(row["compatibility_residual"], EXACT, (eta, mode))
            self.assertLess(row["e100_dbar"], EXACT, (eta, mode))

    def test_incompatible_cases_route_to_h_with_the_recorded_magnitudes(self):
        for eta, mode in ((1.0, RANK_ONE_RESTART), (0.0, COMBINED_RESTART)):
            row = self.row(eta, mode)
            expected_e, expected_r = EXPECTED[(eta, mode)]
            self.assertEqual(row["route"], "H", (eta, mode))
            self.assertAlmostEqual(row["compatibility_residual"], expected_r, delta=0.25 * expected_r, msg=(eta, mode))
            self.assertAlmostEqual(row["e100_dbar"], expected_e, delta=0.25 * expected_e, msg=(eta, mode))

    def test_h_reproduces_the_unlumped_time_average_in_every_case(self):
        for key, row in self.rows.items():
            self.assertLess(row["e100_h"], EXACT, key)


if __name__ == "__main__":
    unittest.main()
