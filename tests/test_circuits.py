"""Every circuit-level implementation against the reference walk, and the shot-to-score conversion.

- Qiskit bit order and padded-state decoding: a shot-based score is a
  distribution over the retained nodes and the padding mass is reported apart;
- ``StructuredSzegedyCircuit`` state preparation, reflection and W^2
  reproducing the dense reference exactly;
- the matrix-free and materialized ``SzegedyWalk`` computing the same scores;
- the Aer layer contract and the deterministic readout-mitigation math.
The dense references the structured circuit and the Aer layers are checked against live in
``tests/oracles.py``. No hardware-feasibility claim is made anywhere here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from classical.mulp.mulp import MultiLayerRanker
from quantum.hardware import aer as qhw
from quantum.hardware.mitigation import calibration_from_counts, mitigate_counts
from quantum.inspired.google_matrix import build_G_from_mulp
from quantum.inspired.mulp import QuantumWalkRanker
from quantum.inspired.szegedy_walk import SzegedyWalk
from tests.oracles import SzegedyCircuit, dense_walk_unitary, run_szegedy_aer_layers


def _random_column_stochastic(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 1.0, size=(dim, dim))
    return raw / raw.sum(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Counts -> node probabilities: renormalized over the retained support
# ---------------------------------------------------------------------------


class CountDecodingTests(unittest.TestCase):
    def test_padding_mass_is_reported_apart_and_the_rest_renormalized(self):
        probs, padding = qhw.counts_to_probs_with_padding_mass({"00": 2, "01": 1, "11": 1}, n_red=3, m=2, total_shots=4)
        self.assertAlmostEqual(padding, 0.25)
        np.testing.assert_allclose(probs, [2 / 3, 1 / 3, 0.0])
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=12)

    def test_statevector_marginal_goes_through_the_same_conversion(self):
        # Amplitude on padding state 3 of a 4-state register with n_red = 3: removed, rest renormalized.
        sv = np.zeros(16, dtype=complex)
        sv[0 * 4 + 0] = np.sqrt(0.5)
        sv[3 * 4 + 0] = np.sqrt(0.5)
        probs = qhw.statevector_register_probs(sv, n_red=3, dim=4, register="source")
        np.testing.assert_allclose(probs, [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            qhw.statevector_register_probs(sv, n_red=3, dim=4, register="bad")


# ---------------------------------------------------------------------------
# Structured circuit vs dense reference
# ---------------------------------------------------------------------------


@pytest.mark.slow
@unittest.skipUnless(qhw._QISKIT_OK, "Qiskit is not installed")
class StructuredCircuitParityTests(unittest.TestCase):
    def test_reflection_about_zero_and_multiplexed_preparation(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Statevector

        from quantum.hardware.structured_circuit import build_multiplexed_state_prep, build_reflection_about_zero

        for m in (1, 2, 3):
            mat = Operator(build_reflection_about_zero(m)).data
            expected = -np.eye(2**m)
            expected[0, 0] = 1.0
            self.assertTrue(np.allclose(mat, expected, atol=1e-9), msg=f"m={m}")

        dim, m = 4, 2
        columns = [np.sqrt(_random_column_stochastic(dim, seed=1)[:, j]) for j in range(dim)]
        B = build_multiplexed_state_prep(columns, m=m)
        for source in range(dim):
            init = QuantumCircuit(2 * m)
            for bit in range(m):
                if (source >> bit) & 1:
                    init.x(m + bit)
            sv = Statevector(init.compose(B)).data
            expected = np.zeros(dim * dim, dtype=complex)
            expected[source * dim : (source + 1) * dim] = columns[source]
            self.assertTrue(np.allclose(sv, expected, atol=1e-9), msg=f"source={source}")

    def test_structured_walk_step_matches_the_dense_unitary_including_padding(self):
        from qiskit.quantum_info import Operator

        from quantum.hardware.structured_circuit import StructuredSzegedyCircuit

        for dim, seed in ((2, 0), (4, 1)):
            G = _random_column_stochastic(dim, seed=seed)
            struct = StructuredSzegedyCircuit(G, np.ones(dim) / dim)
            self.assertTrue(np.allclose(Operator(struct._walk_step_circuit()).data, dense_walk_unitary(G), atol=1e-8))
        G = _random_column_stochastic(3, seed=7)
        v = np.array([0.5, 0.3, 0.2])
        struct = StructuredSzegedyCircuit(G, v)
        reference = SzegedyCircuit(G, v, max_unitary_state_dim=16)
        self.assertEqual(struct.dim, 4)
        self.assertTrue(np.allclose(Operator(struct._walk_step_circuit()).data, reference._walk_unitary(), atol=1e-8))
        with self.assertRaisesRegex(ValueError, "fixed at 2"):
            struct.build_circuits_per_step(2, step_power=1)


@pytest.mark.slow
@unittest.skipUnless(qhw._QISKIT_OK, "Qiskit is not installed")
class AerLayerTests(unittest.TestCase):
    def test_layers_are_distributions_over_retained_nodes_with_padding_reported(self):
        G = _random_column_stochastic(3, seed=1)
        v = np.array([0.5, 0.3, 0.2])
        res = run_szegedy_aer_layers(G, v, tau=3, shots=512, seed=7)
        self.assertEqual((res["n_red"], res["padded_dim"], res["measured_register"], res["step_power"]), (3, 4, "source", 2))
        self.assertIn("shots", res["sampling"])
        self.assertTrue(np.allclose(res["noiseless_aer_statevector"], res["exact"], atol=1e-10))
        for score_key, padding_key in (("noiseless_aer", "padding_mass_noiseless"), ("noisy_aer", "padding_mass_noisy"),
                                       ("readout_mitigated", "padding_mass_mitigated")):
            scores = np.array(res[score_key], dtype=float)
            padding = np.array(res[padding_key], dtype=float)
            self.assertAlmostEqual(float(scores.sum()), 1.0, places=9, msg=score_key)
            self.assertGreaterEqual(scores.min(), -1e-12, msg=score_key)
            self.assertEqual(padding.shape, (3,), msg=padding_key)
            self.assertTrue(((padding >= 0.0) & (padding < 1.0)).all(), msg=padding_key)
            self.assertAlmostEqual(res[f"{padding_key}_mean"], float(padding.mean()), places=12)


# ---------------------------------------------------------------------------
# Matrix-free vs materialized SzegedyWalk on the same MuLP graph
# ---------------------------------------------------------------------------


def _tiny_fixture_path(tmpdir: str) -> Path:
    data = {
        "BAD": [1, 0, 1, 0, 0],
        "JOB_A": [1, 0, 0, 0, 1], "JOB_B": [0, 1, 0, 0, 0], "JOB_C": [0, 0, 1, 0, 0], "JOB_D": [0, 0, 0, 1, 0],
        "REASON_X": [1, 0, 0, 1, 0], "REASON_Y": [0, 1, 0, 0, 1], "REASON_Z": [0, 0, 1, 0, 0],
    }
    path = Path(tmpdir) / "fixture.parquet"
    pl.DataFrame(data).write_parquet(path)
    return path


@pytest.mark.slow
class WalkImplementationParityTests(unittest.TestCase):
    def _run_both(self, path: Path, *, stickiness: float, alpha: float, tau: int) -> tuple[np.ndarray, np.ndarray]:
        ranker = MultiLayerRanker(str(path), bidirectional=False, sparse=True, stickiness=stickiness)
        ranker.buildAdjMatrix()
        ranker.buildPersonalizationMatrix()
        qranker = QuantumWalkRanker(ranker=ranker, alpha=alpha)
        qranker.buildSzegedyWalk()
        matrix_free = qranker.quantumPageRank(tau=tau, validate_probabilities=True)
        G, v = build_G_from_mulp(ranker, alpha=alpha)
        bl1 = np.arange(ranker.n_rows, dtype=int)
        materialized = SzegedyWalk(G, v, bl1, bl1 + ranker.reason_offset).run(tau=tau, validate_probabilities=True)
        return matrix_free, materialized

    def test_matrix_free_and_materialized_walks_agree(self):
        with tempfile.TemporaryDirectory() as td:
            path = _tiny_fixture_path(td)
            for stickiness, alpha, tau in ((1.0, 0.85, 6), (4.0, 0.7, 4)):
                matrix_free, materialized = self._run_both(path, stickiness=stickiness, alpha=alpha, tau=tau)
                self.assertTrue(np.allclose(matrix_free, materialized, atol=1e-9), msg=f"S={stickiness}")


# ---------------------------------------------------------------------------
# Readout mitigation: deterministic math
# ---------------------------------------------------------------------------


class MitigationMathTests(unittest.TestCase):
    def test_readout_mitigation_inverts_a_known_assignment_error(self):
        calibration = calibration_from_counts({"0": 90, "1": 10}, [{"0": 10, "1": 90}], n_qubits=1, shots=100)
        mitigated = mitigate_counts({"0": 70, "1": 30}, calibration, shots=100)
        self.assertAlmostEqual(mitigated["0"], 0.75, places=12)
        self.assertAlmostEqual(mitigated["1"], 0.25, places=12)
        # The constrained solve never returns a negative probability, even when the plain inverse would.
        calibration = calibration_from_counts({"0": 95, "1": 5}, [{"0": 5, "1": 95}], n_qubits=1, shots=100)
        mitigated = mitigate_counts({"0": 99, "1": 1}, calibration, shots=100)
        self.assertGreaterEqual(min(mitigated.values()), 0.0)
        self.assertAlmostEqual(sum(mitigated.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
