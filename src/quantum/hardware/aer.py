"""Aer execution of the lumped Szegedy circuits: the statevector snapshots, counts to supra-node vectors, readout mitigation.

Shared Qiskit utilities
-----------------------
The statevector snapshot runner and the shot-to-score conversion. The dense
tiny-graph oracle the structured circuit is checked against is
``tests/oracles.py``. Nothing here or elsewhere in the repository submits to a
device. Every conversion of measured mass into node probabilities goes through
:func:`retained_distribution`.

The Aer layers
--------------
Every shot-based vector is a distribution over the ``n_red`` retained
supra-nodes: the mass that landed on padding codes is removed and the rest
renormalized at the single conversion point
(:func:`counts_to_probs_with_padding_mass`), and the padding mass is reported
beside every distance, never folded into a score. Readout mitigation is
calibrated on the qubits each circuit measures
(``mitigation.calibration_circuits_on``), one calibration set per distinct
measured-qubit tuple and each circuit mitigated with the set of its own tuple
(:func:`mitigated_omega_by_circuit`). Zero-noise extrapolation is not performed
anywhere in the repository. The runner that puts every layer through one
operator, checked against the exact walk, is a test oracle
(``tests/oracles.py``).
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from quantum.hardware.structured_circuit import StructuredSzegedyCircuit


try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    _QISKIT_OK = True
    _QISKIT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _QISKIT_OK = False
    _QISKIT_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Register marginals and the retained-support renormalization
# ---------------------------------------------------------------------------


def statevector_register_probs(
    sv: np.ndarray,
    *,
    n_red: int,
    dim: int,
    register: str = "source",
) -> np.ndarray:
    """Marginalize a row-major |source>|target> statevector."""
    target_aliases = {"target", "second", "destination"}
    source_aliases = {"source", "first", "position"}
    if register not in target_aliases | source_aliases:
        raise ValueError(
            "register must be 'target'/'second'/'destination' or 'source'/'first'/'position'"
        )

    register_probs = np.zeros(dim, dtype=float)
    for flat_index, amp in enumerate(np.asarray(sv, dtype=complex)):
        source = flat_index // dim
        target = flat_index % dim
        idx = target if register in target_aliases else source
        register_probs[idx] += float(abs(amp) ** 2)
    return retained_distribution(register_probs, n_red)[0]


def run_statevector_snapshots(
    sc: Any,
    tau: int,
    *,
    measured_register: str = "source",
    step_power: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Exact per-step marginals from an Aer statevector evolution carried in the state.

    Building ``tau`` circuits with ``0..tau-1`` recorded steps each is an
    ``O(tau^2)`` cost that is prohibitive on the lumped operator at
    ``tau = 100``. Holding all ``tau`` steps in one circuit is not the way out
    of it: the structured W unrolls to tens of thousands of gates, so
    ``tau`` applications of W² flatten into millions of instructions and the
    job is killed for memory before any result is saved. The evolution is
    therefore carried in the state rather than in the circuit -- one W² block,
    transpiled once, applied per recorded step, each run seeded with the
    previous step's statevector (Aer's ``set_statevector``). The work is still
    ``tau`` applications of W², the same the NumPy reference pays, at the
    memory of a single step.

    Transpilation is level 0 throughout, as in
    :func:`_run_counts`: the block is only re-expressed in the
    simulator's basis, never re-optimised.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}")
    if step_power != 2:
        raise ValueError(f"step_power is fixed at 2 (W²); got {step_power}")
    from qiskit.quantum_info import Statevector

    simulator = AerSimulator(method="statevector", seed_simulator=seed)
    n_qubits = 2 * sc.m

    initial = sc.build_statevector_circuit(0, step_power=step_power)
    if not hasattr(initial, "save_statevector") or not hasattr(initial, "set_statevector"):
        raise RuntimeError("Aer statevector save/set instructions are unavailable")

    def run_saving_state(circuit) -> np.ndarray:
        """Execute one circuit that ends in ``save_statevector`` and return that state."""
        return np.asarray(
            simulator.run(circuit, seed_simulator=seed).result().data(0)["sv"], dtype=complex
        )

    initial.save_statevector(label="sv")
    sv = run_saving_state(transpile(initial, simulator, optimization_level=0, seed_transpiler=seed))

    probs = np.zeros((tau, sc.n_red), dtype=float)
    probs[0] = statevector_register_probs(
        sv, n_red=sc.n_red, dim=sc.dim, register=measured_register
    )
    if tau == 1:
        return probs

    if hasattr(sc, "_walk_step_circuit"):
        W_gate = sc._walk_step_circuit().to_gate(label="W")
    else:
        from qiskit.circuit.library import UnitaryGate

        W_gate = UnitaryGate(sc._walk_unitary(), label="W")
    block = QuantumCircuit(n_qubits)
    for _ in range(step_power):
        block.append(W_gate, list(range(n_qubits)))
    block = transpile(block, simulator, optimization_level=0, seed_transpiler=seed)
    block.save_statevector(label="sv")

    for t in range(1, tau):
        # The seeded state is the only thing that changes between steps, so the
        # already-transpiled block is composed onto it rather than rebuilt.
        step = QuantumCircuit(n_qubits)
        step.set_statevector(Statevector(sv))
        step.compose(block, inplace=True)
        sv = run_saving_state(step)
        probs[t] = statevector_register_probs(
            sv, n_red=sc.n_red, dim=sc.dim, register=measured_register
        )
    return probs


def retained_distribution(register_probs: np.ndarray, n_red: int) -> tuple[np.ndarray, float]:
    """Split a register marginal into the retained-node distribution and the padding mass.

    The single point at which measured mass becomes node probabilities. The
    first ``n_red`` basis states are the retained nodes; everything above is a
    padding code the circuit's ``2**m`` register carries but the operator has no
    node for. The retained part is **renormalized over its own sum** so the
    returned vector is a distribution over the ``n_red`` nodes and comparable
    with the exact reference, which sums to one. The discarded ``padding_mass``
    is returned beside it as a diagnostic, reported and never folded in.
    """
    register_probs = np.asarray(register_probs, dtype=float)
    probs = register_probs[:n_red].copy()
    padding_mass = float(register_probs[n_red:].sum())
    total = float(probs.sum())
    if total > 0.0:
        probs /= total
    return probs, padding_mass


def counts_to_probs_with_padding_mass(
    counts: dict[str, int],
    n_red: int,
    m: int,
    total_shots: int,
) -> tuple[np.ndarray, float]:
    """Convert counts to a distribution over the ``n_red`` retained nodes plus the padding mass.

    Returns ``(probs, padding_mass)``: ``probs`` sums to one over the retained
    support (see :func:`retained_distribution`) and ``padding_mass`` is the
    fraction of shots that landed on padding codes, reported separately.
    """
    if total_shots <= 0:
        raise ValueError(f"total_shots must be positive; got {total_shots}")
    register_probs = np.zeros(2**m, dtype=float)
    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        # Qiskit displays classical bits most-significant first (c[m-1]...c[0]).
        # Since we measure qubit k into classical bit k, int(clean, 2) is the
        # intended little-register basis index.
        idx = int(clean, 2) if clean else 0
        register_probs[idx] += count / total_shots
    return retained_distribution(register_probs, n_red)


# ---------------------------------------------------------------------------
# The Aer layers
# ---------------------------------------------------------------------------


def counts_history_to_omega(counts_list: list[dict[str, int]], *, n_red: int, m: int, shots: int) -> tuple[np.ndarray, np.ndarray]:
    """Time-average per-step counts into one supra-node vector, with the per-step padding mass."""
    tau = len(counts_list)
    omega = np.zeros(n_red, dtype=float)
    padding = np.zeros(tau, dtype=float)
    for t, counts in enumerate(counts_list):
        probs, pad = counts_to_probs_with_padding_mass(counts, n_red, m, shots)
        omega += probs
        padding[t] = pad
    return omega / max(tau, 1), padding


def probs_dict_to_omega(probs_dict: dict[str, float], *, n_red: int, m: int, shots: int) -> tuple[np.ndarray, float]:
    """A ``{bitstring: probability}`` layer through the same padding conversion as counts."""
    pseudo_counts = {b: max(0.0, float(p)) * shots for b, p in probs_dict.items()}
    probs, pad = counts_to_probs_with_padding_mass(pseudo_counts, n_red, m, shots)
    return probs, float(pad)


def run_counts(circuits: Iterable, simulator, shots: int, seeds: Sequence[int]) -> list[dict[str, int]]:
    """Execute already-compiled circuits one by one, each with its own simulator seed.

    Level-0 transpilation only re-expresses an ISA circuit in the simulator's
    basis; it never re-optimises the gate count.
    """
    from qiskit import transpile

    out: list[dict[str, int]] = []
    for qc, seed in zip(circuits, seeds):
        t_qc = transpile(qc, simulator, optimization_level=0, seed_transpiler=int(seed))
        result = simulator.run(t_qc, shots=int(shots), seed_simulator=int(seed)).result()
        out.append(result.get_counts(t_qc))
    return out


def mitigated_omega_by_circuit(
    counts_list, calibrations, circuit_to_calibration, *, n_red: int, m: int, shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Readout-mitigated time average of per-step counts, each circuit under the calibration set of its own measured qubits.

    ``calibrations[g]`` is the set of measured-qubit tuple ``g`` and ``circuit_to_calibration[t]`` the set circuit
    ``t`` is mitigated with (``mitigation.CalibrationGroups.index_of_circuit``); both come from the same grouping.
    A single calibration for every circuit is the case ``calibrations = [c]``, ``circuit_to_calibration = [0] * tau``.
    """
    from quantum.hardware.mitigation import mitigate_counts

    if len(circuit_to_calibration) < len(counts_list):
        raise ValueError(f"{len(counts_list)} circuits but {len(circuit_to_calibration)} calibration indices")
    omega = np.zeros(n_red, dtype=float)
    padding = np.zeros(len(counts_list), dtype=float)
    for t, counts in enumerate(counts_list):
        calibration = calibrations[int(circuit_to_calibration[t])]
        probs, pad = probs_dict_to_omega(mitigate_counts(counts, calibration, shots), n_red=n_red, m=m, shots=shots)
        omega += probs
        padding[t] = pad
    return omega / max(len(counts_list), 1), padding


def lumped_statevector_history(G_red: np.ndarray, v_red: np.ndarray, *, tau: int, seed: int) -> tuple[np.ndarray, float]:
    """Exact Aer statevector marginals of the structured circuit built from the lumped operator, and the wall time."""
    struct = StructuredSzegedyCircuit(G_red, v_red)
    t0 = perf_counter()
    history = run_statevector_snapshots(struct, int(tau), measured_register="source", seed=int(seed))
    return history, perf_counter() - t0
