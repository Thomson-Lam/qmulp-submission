"""
Matrix-free (M3-style) readout mitigation.

Why not a naive full calibration matrix: with an 11-12 qubit position register,
the full assignment matrix is 2^11 x 2^11 (2048x2048) or worse -- expensive to
build and invert, and mostly wasted since any single run only ever observes a
handful of distinct bitstrings. M3's actual trick (per IBM's own documentation)
is to restrict the correction to the small subspace of *observed* bitstrings,
using the fact that the full assignment matrix is a tensor product of small
per-qubit (2x2) assignment matrices -- so we never materialize the full matrix,
only the tiny sub-block needed for whichever bitstrings actually showed up.

This module is self-contained (no dependency on a specific backend's Job/Result
API) so it works the same way whatever produced the counts: you give it raw
calibration counts and raw experiment counts (plain dicts of {bitstring:
count}), and it gives you back mitigated probabilities.

Calibration is per circuit. Routing places the measured register on different
physical qubits in different circuits, so :func:`calibration_groups` groups the
circuits of a set by their measured-qubit tuple, one calibration set per distinct
tuple (:func:`calibration_circuits_on` builds its circuits on exactly those
qubits), and each circuit is mitigated with the set of its own tuple
(``aer.mitigated_omega_by_circuit``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PerQubitCalibration:
    """Per-qubit 2x2 assignment matrix: A[i][measured, prepared]."""

    matrices: list  # list of 2x2 np.ndarray, one per measured qubit


def build_calibration_circuits(n_qubits: int):
    """Return (baseline_circuit, [flip_i_circuit for i in range(n_qubits)]).

    baseline: all qubits prepared |0>.
    flip_i:   qubit i prepared |1>, rest |0>.
    Only n_qubits + 1 circuits total (not 2^n) -- assumes per-qubit-independent
    readout error (no crosstalk), the same assumption Qiskit's own
    LocalReadoutMitigator and M3's default calibration make.
    """
    from qiskit import QuantumCircuit

    baseline = QuantumCircuit(n_qubits, n_qubits, name="cal_baseline")
    baseline.measure(range(n_qubits), range(n_qubits))

    flips = []
    for i in range(n_qubits):
        qc = QuantumCircuit(n_qubits, n_qubits, name=f"cal_flip_{i}")
        qc.x(i)
        qc.measure(range(n_qubits), range(n_qubits))
        flips.append(qc)

    return baseline, flips


def calibration_from_counts(
    baseline_counts: dict, flip_counts: list, n_qubits: int, shots: int
) -> PerQubitCalibration:
    """Build per-qubit 2x2 assignment matrices from raw calibration counts.

    baseline_counts: counts dict from running the all-|0> circuit.
    flip_counts: list of counts dicts, one per qubit, from the bit-i-flipped circuit,
    in the order ``build_calibration_circuits`` emits them (baseline first, then
    flip_0 .. flip_{n-1}); ``n_qubits + 1`` count dictionaries in all.
    """
    if len(flip_counts) != int(n_qubits):
        raise ValueError(f"expected {int(n_qubits) + 1} calibration count dictionaries (baseline + one flip per qubit); "
                         f"got {len(flip_counts) + 1}")
    matrices = []
    for i in range(n_qubits):
        p0_given_0 = _bit_zero_fraction(baseline_counts, i, shots)
        p0_given_1 = _bit_zero_fraction(flip_counts[i], i, shots)
        # A[measured, prepared]
        A = np.array(
            [
                [p0_given_0, p0_given_1],
                [1 - p0_given_0, 1 - p0_given_1],
            ]
        )
        matrices.append(A)
    return PerQubitCalibration(matrices=matrices)


def _bit_zero_fraction(counts: dict, bit_index: int, shots: int) -> float:
    total_zero = 0
    for bitstring, n in counts.items():
        # Qiskit convention: rightmost character is qubit 0.
        bit = bitstring[::-1][bit_index]
        if bit == "0":
            total_zero += n
    return total_zero / max(shots, 1)


def mitigate_counts(raw_counts: dict, calibration: PerQubitCalibration, shots: int) -> dict:
    """Correct raw counts using the reduced-subspace (M3-style) method.

    Restricts the correction to the small set of *observed* bitstrings instead
    of the full 2^n space: builds the |observed| x |observed| sub-block of the
    tensor-product assignment matrix and solves the nonnegative least-squares
    problem in that subspace, then renormalizes.
    """
    observed = list(raw_counts.keys())
    k = len(observed)
    n_qubits = len(calibration.matrices)

    p_noisy = np.array([raw_counts[b] / shots for b in observed])

    A_reduced = np.empty((k, k))
    for row, b_measured in enumerate(observed):
        for col, b_prepared in enumerate(observed):
            entry = 1.0
            bm = b_measured[::-1]
            bp = b_prepared[::-1]
            for q in range(n_qubits):
                mi = int(bm[q])
                pi = int(bp[q])
                entry *= calibration.matrices[q][mi, pi]
            A_reduced[row, col] = entry

    # Constrained least squares: the corrected vector is nonnegative by
    # construction (NNLS), then renormalised over the observed subspace; no
    # quasi-probability is clipped after the fact.
    from scipy.optimize import nnls

    p_true, _residual = nnls(A_reduced, p_noisy)
    total = p_true.sum()
    if total > 0:
        p_true = p_true / total

    return {b: float(p) for b, p in zip(observed, p_true)}


# ---------------------------------------------------------------------------
# Which physical qubits a circuit measures, and one calibration set per distinct tuple
# ---------------------------------------------------------------------------


def measured_physical_qubits(isa_circuit) -> list[int]:
    """Physical qubits of the measured (source) register, in clbit order, from the ISA circuit's final layout."""
    m = int(isa_circuit.num_clbits)
    measured = sorted((isa_circuit.find_bit(inst.clbits[0]).index, isa_circuit.find_bit(inst.qubits[0]).index)
                      for inst in isa_circuit.data if inst.operation.name == "measure")
    from_measures = [q for _c, q in measured]
    if isa_circuit.layout is None:
        return from_measures
    final = list(isa_circuit.layout.final_index_layout())
    from_layout = [int(final[q]) for q in range(m, 2 * m)]
    if from_layout != from_measures:
        raise AssertionError(f"final layout {from_layout} disagrees with the measure instructions {from_measures}")
    return from_layout


def calibration_circuits_on(physical_qubits: Sequence[int], backend, seed: int) -> list:
    """The readout-calibration circuits transpiled at level 0 onto ``physical_qubits`` so the layout is preserved.

    The baseline circuit first, then one single-flip circuit per qubit, in the order
    :func:`calibration_from_counts` expects.
    """
    from qiskit.transpiler import generate_preset_pass_manager

    baseline, flips = build_calibration_circuits(len(physical_qubits))
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=0,
                                      initial_layout=list(physical_qubits), seed_transpiler=int(seed))
    return list(pm.run([baseline, *flips]))


@dataclass(frozen=True)
class CalibrationGroups:
    """Circuits grouped by the ordered tuple of physical qubits they measure.

    ``tuples[g]`` is the measured-qubit tuple of calibration set ``g`` (sets are numbered by first appearance)
    and ``index_of_circuit[t]`` the set circuit ``t`` is mitigated with.
    """

    tuples: tuple[tuple[int, ...], ...]
    index_of_circuit: tuple[int, ...]

    def n_sets_for(self, n_circuits: int) -> int:
        """Calibration sets the first ``n_circuits`` circuits use (numbering by first appearance makes this a prefix)."""
        return max(self.index_of_circuit[: int(n_circuits)], default=-1) + 1


def calibration_groups(measured_qubits: Sequence[Sequence[int]]) -> CalibrationGroups:
    """Group per-circuit measured-qubit tuples: one calibration set per distinct *ordered* tuple.

    Two circuits measuring the same qubits into different classical bits need different sets, because a
    calibration circuit measures qubit ``i`` of its tuple into bit ``i``.
    """
    tuples: list[tuple[int, ...]] = []
    index: dict[tuple[int, ...], int] = {}
    of_circuit: list[int] = []
    for measured in measured_qubits:
        key = tuple(int(q) for q in measured)
        if key not in index:
            index[key] = len(tuples)
            tuples.append(key)
        of_circuit.append(index[key])
    return CalibrationGroups(tuples=tuple(tuples), index_of_circuit=tuple(of_circuit))
