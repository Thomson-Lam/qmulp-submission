"""Independent oracles the tests check the pipeline against; nothing the pipeline runs imports this module.

Each oracle is built differently from the production code it checks, so that agreement is evidence:

- the dense reference operators of the Szegedy walk: ``Pi``, ``S_swap`` and ``W = S_swap (2 Pi - I)`` as explicit
  ``(n^2, n^2)`` matrices, for tiny graphs only (the matrix-free walk is compared with them);
- :class:`SzegedyCircuit`, the dense tiny-graph circuit (``W`` as one ``UnitaryGate``) the structured circuit is
  checked against; it shares the structured circuit's padding and per-step builders and differs in what is dense;
- :func:`run_szegedy_aer_layers`, the exact reference, the Aer statevector layer and the shot-based layers on one lumped
  operator, compared with the exact walk.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Sequence

import numpy as np

from evaluation.reporting import l1_sum
from quantum.hardware.aer import (
    _QISKIT_OK,
    counts_history_to_omega,
    mitigated_omega_by_circuit,
    run_counts,
    run_statevector_snapshots,
)
from quantum.hardware.structured_circuit import StructuredSzegedyCircuit
from quantum.inspired.szegedy_walk import SzegedyWalk
from quantum.inspired.validation import DEFAULT_MAX_DENSE_BYTES, check_dense_allocation, validate_column_stochastic_matrix

if _QISKIT_OK:  # pragma: no cover - environment dependent
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation


# ---------------------------------------------------------------------------
# Dense reference operators for the Szegedy walk (tiny graphs only)
# ---------------------------------------------------------------------------

MAX_DENSE_SUPRA_NODES = 64


def state_matrix_to_vector(Psi: np.ndarray) -> np.ndarray:
    """Row-major vectorization: index = source*n + target."""
    return np.asarray(Psi, dtype=complex).reshape(-1)


def _assert_small_enough_for_dense(n: int, name: str) -> None:
    if int(n) > MAX_DENSE_SUPRA_NODES:
        bytes_needed = 16 * int(n) ** 4
        raise ValueError(
            f"{name} would allocate an ({n}^2 x {n}^2) complex128 array "
            f"({bytes_needed / 2**30:.2f} GiB) for {n} supra-nodes; the dense reference is "
            f"limited to {MAX_DENSE_SUPRA_NODES} supra-nodes. Use the matrix-free walk instead: "
            "SzegedyWalk(G, v, ...) applies R_Pi and the swap without forming the projector, "
            "and SzegedyWalk.implicit(T, v, ...) never forms G either."
        )


def dense_szegedy_projector(G: np.ndarray) -> np.ndarray:
    G = validate_column_stochastic_matrix(G, name="G", atol=1e-10)
    n = G.shape[0]
    _assert_small_enough_for_dense(n, "dense_szegedy_projector")
    d2 = n * n
    G_sqrt = np.sqrt(G)
    Pi = np.zeros((d2, d2), dtype=complex)
    for j in range(n):
        psi_j = np.zeros(d2, dtype=complex)
        psi_j[j * n : (j + 1) * n] = G_sqrt[:, j]
        Pi += np.outer(psi_j, psi_j.conj())
    return Pi


def dense_swap_operator(n: int) -> np.ndarray:
    _assert_small_enough_for_dense(n, "dense_swap_operator")
    S = np.zeros((n * n, n * n), dtype=complex)
    for source in range(n):
        for target in range(n):
            S[target * n + source, source * n + target] = 1.0
    return S


def dense_walk_unitary(G: np.ndarray) -> np.ndarray:
    n = G.shape[0]
    _assert_small_enough_for_dense(n, "dense_walk_unitary")
    Pi = dense_szegedy_projector(G)
    R_Pi = 2.0 * Pi - np.eye(n * n, dtype=complex)
    return dense_swap_operator(n) @ R_Pi


# ---------------------------------------------------------------------------
# The dense tiny-graph Szegedy circuit: a correctness oracle for the structured circuit
# ---------------------------------------------------------------------------


class SzegedyCircuit(StructuredSzegedyCircuit):
    """
    Qiskit circuit builder for W = S_swap * R_Pi on a reduced graph.

    Vector convention is row-major |source>|target> with flat index
    source*dim + target. In Qiskit's little-endian qubit indexing this means
    the low m qubits encode target and the high m qubits encode source.
    The paper's "position" register is ``source`` (the first factor of
    H = C^n (x) C^n; see ``SzegedyWalk._marginal_supra_node_probs``), so
    paper-style PageRank measures qubits range(m, 2*m), not range(0, m).

    The padding of ``G`` and ``v``, the initial state vector, the measured qubits and the per-step circuits are
    the structured circuit's (``quantum.hardware.structured_circuit``), defined once there. What is dense here, and
    is what the structured circuit is checked against: ``W`` as one explicit unitary and the initial state as one
    ``StatePreparation`` on both registers.
    """

    def __init__(
        self,
        G_red: np.ndarray,
        v_red: np.ndarray,
        *,
        max_unitary_state_dim: int = 256,
        validate_unitary: bool = True,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        super().__init__(G_red, v_red, max_dense_bytes=max_dense_bytes)
        self.max_unitary_state_dim = int(max_unitary_state_dim)
        self.validate_unitary = bool(validate_unitary)
        self.max_dense_bytes = max_dense_bytes
        self._W_cache: np.ndarray | None = None

        if self.state_dim > self.max_unitary_state_dim:
            raise ValueError(
                f"Dense Qiskit unitary would act on state_dim={self.state_dim}; "
                f"limit is {self.max_unitary_state_dim}. Reduce n_red or raise "
                "max_unitary_state_dim."
            )
        check_dense_allocation(
            "Qiskit dense W unitary",
            (self.state_dim, self.state_dim),
            np.complex128,
            max_dense_bytes,
        )
        self.g_vecs = np.sqrt(np.maximum(self.G_pad, 0.0))

    def _walk_unitary(self) -> np.ndarray:
        """Return dense W = S_swap * R_Pi for the padded reduced graph."""
        if self._W_cache is not None:
            return self._W_cache

        d2 = self.state_dim
        Pi = np.zeros((d2, d2), dtype=complex)
        for source in range(self.dim):
            psi_j = np.zeros(d2, dtype=complex)
            psi_j[source * self.dim : (source + 1) * self.dim] = self.g_vecs[:, source]
            Pi += np.outer(psi_j, psi_j.conj())

        R_Pi = 2.0 * Pi - np.eye(d2, dtype=complex)
        S_swap = np.zeros((d2, d2), dtype=complex)
        for source in range(self.dim):
            for target in range(self.dim):
                S_swap[target * self.dim + source, source * self.dim + target] = 1.0

        W = S_swap @ R_Pi
        if self.validate_unitary:
            ident = np.eye(d2, dtype=complex)
            if not np.allclose(W.conj().T @ W, ident, atol=1e-10):
                raise ValueError("Constructed Szegedy W failed W^†W = I unitarity check")
            if not np.allclose(W @ W.conj().T, ident, atol=1e-10):
                raise ValueError("Constructed Szegedy W failed WW^† = I unitarity check")
        self._W_cache = W
        return W

    def _initialization_circuit(self) -> QuantumCircuit:
        """The dense preparation of |Psi_0>: one ``StatePreparation`` of the whole initial state on both registers."""
        qc = QuantumCircuit(2 * self.m, name="Psi0_dense")
        qc.append(StatePreparation(self._initial_state_vector(), label="Psi0"), list(range(2 * self.m)))
        return qc

    def _walk_step_circuit(self) -> QuantumCircuit:
        """One application of the dense ``W`` as a ``2m``-qubit circuit."""
        from qiskit.circuit.library import UnitaryGate

        qc = QuantumCircuit(2 * self.m, name="W_dense")
        qc.append(UnitaryGate(self._walk_unitary(), label="W"), list(range(2 * self.m)))
        return qc


# ---------------------------------------------------------------------------
# The Aer layers on one lumped operator, against the exact walk
# ---------------------------------------------------------------------------

# Supra-node probability vectors are compared at full precision; the tie
# tolerance applies to borrower scores, not to the register marginals.
L1_DECIMALS = 15
SUPPORTED_LAYERS = ("exact", "noiseless_aer_statevector", "noiseless_aer", "noisy_aer", "readout_mitigated")


def exact_reference(G_red: np.ndarray, v_red: np.ndarray, tau: int) -> np.ndarray:
    """The exact per-step source marginals of the lumped walk, shape ``(tau, n_red)``."""
    n_red = int(G_red.shape[0])
    walk = SzegedyWalk(G_red, v_red, np.arange(n_red), np.arange(n_red), measure_register="source")
    return walk.supra_node_history(int(tau), register="source")


def run_szegedy_aer_layers(
    G_red: np.ndarray, v_red: np.ndarray, *, tau: int, shots: int = 4096, seed: int = 42, layers: Sequence[str] | None = None,
    backend_name: str | None = None, isa_circuits: Sequence | None = None, noise_model=None,
    one_qubit_prob: float = 0.001, two_qubit_prob: float = 0.01, readout_prob: float = 0.01,
    optimization_level: int = 3, max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
) -> dict[str, Any]:
    """The requested layers on one lumped operator; never submits to hardware.

    ``backend_name`` selects the IBM-calibrated noise (``AerSimulator.from_backend``
    on the fake backend) and ``isa_circuits`` lets a caller hand in the per-step
    circuits already transpiled against that backend; otherwise the circuits are
    transpiled here. ``noise_model`` replaces the fake backend's own noise by an
    explicit Aer noise model. Without a backend a generic depolarizing model is
    used. Seeds: circuit ``t`` runs under ``seed + t``; calibration set ``g`` under ``seed + g``.
    """
    if tau <= 0 or shots <= 0:
        raise ValueError("tau and shots must be positive")
    requested = set(SUPPORTED_LAYERS) if layers is None else set(map(str, layers))
    unknown = requested - set(SUPPORTED_LAYERS)
    if unknown:
        raise ValueError(f"unknown layers {sorted(unknown)}; choose from {SUPPORTED_LAYERS}")
    need_sv = "noiseless_aer_statevector" in requested
    need_noiseless = "noiseless_aer" in requested
    need_noisy = bool(requested & {"noisy_aer", "readout_mitigated"})
    need_mitigation = "readout_mitigated" in requested
    need_qiskit = need_sv or need_noiseless or need_noisy
    if need_qiskit and not _QISKIT_OK:
        raise ImportError("Qiskit/Aer is not installed; cannot run requested Aer layers")

    t_start = perf_counter()
    exact_history = exact_reference(G_red, v_red, tau)
    omega_exact = exact_history.mean(axis=0)
    n_red = int(G_red.shape[0])
    m = int(np.ceil(np.log2(max(n_red, 2))))
    results: dict[str, Any] = {"n_red": n_red, "m_qubits_per_register": m, "total_logical_qubits": 2 * m, "padded_dim": 2**m,
                               "tau": int(tau), "shots": int(shots), "seed": int(seed), "measured_register": "source",
                               "step_power": 2, "sampling": f"Aer counts, {int(shots)} shots per circuit, one seed per circuit",
                               "exact": omega_exact.tolist(), "exact_history": exact_history.tolist()}
    if not need_qiskit:
        results["elapsed_seconds"] = perf_counter() - t_start
        return results

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    struct = StructuredSzegedyCircuit(G_red, v_red, max_dense_bytes=max_dense_bytes)
    m, n_red = struct.m, struct.n_red
    seeds = [int(seed) + t for t in range(int(tau))]
    if need_sv:
        sv_history = run_statevector_snapshots(struct, tau, measured_register="source", step_power=2, seed=seed)
        results.update(noiseless_aer_statevector=sv_history.mean(axis=0).tolist(),
                       l1_noiseless_sv_vs_exact=l1_sum(sv_history.mean(axis=0), omega_exact, decimals=L1_DECIMALS))
    fake_backend = None
    if need_noiseless or need_noisy:
        circuits = struct.build_circuits_per_step(tau, measured_register="source", step_power=2)
        if backend_name is not None:
            from quantum.hardware.execution import get_fake_backend

            fake_backend = get_fake_backend(backend_name)
            transpiled = list(isa_circuits) if isa_circuits is not None else transpile(
                circuits, backend=fake_backend, optimization_level=optimization_level, seed_transpiler=seed)
            if len(transpiled) != tau:
                raise ValueError(f"isa_circuits must hold one circuit per recorded step ({tau})")
            results["basis_gates"] = sorted(str(op) for op in fake_backend.target.operation_names)
        else:
            transpiled = transpile(circuits, basis_gates=["rz", "sx", "x", "cx"], optimization_level=optimization_level, seed_transpiler=seed)
            results["basis_gates"] = ["rz", "sx", "x", "cx"]
        results["noise"] = ({"model": "ibm_calibrated", "backend_name": backend_name} if backend_name is not None and noise_model is None
                            else {"model": "explicit_noise_model"} if noise_model is not None
                            else {"model": "depolarizing_readout", "one_qubit_prob": one_qubit_prob, "two_qubit_prob": two_qubit_prob,
                                  "readout_prob": readout_prob})
    if need_noiseless:
        sim = AerSimulator.from_backend(fake_backend, noise_model=None) if fake_backend is not None else AerSimulator()
        counts = run_counts(transpiled, sim, shots, seeds)
        omega, padding = counts_history_to_omega(counts, n_red=n_red, m=m, shots=shots)
        results.update(noiseless_aer=omega.tolist(), padding_mass_noiseless=padding.tolist(),
                       padding_mass_noiseless_mean=float(padding.mean()), l1_noiseless_sampled_vs_exact=l1_sum(omega, omega_exact, decimals=L1_DECIMALS))
    if need_noisy:
        if noise_model is not None:
            noisy_sim = AerSimulator(noise_model=noise_model)
        elif fake_backend is not None:
            noisy_sim = AerSimulator.from_backend(fake_backend)
        else:
            from quantum.hardware.execution import build_depolarizing_readout_noise_model

            noisy_sim = AerSimulator(noise_model=build_depolarizing_readout_noise_model(
                one_qubit_prob=one_qubit_prob, two_qubit_prob=two_qubit_prob, readout_prob=readout_prob))
        noisy_counts = run_counts(transpiled, noisy_sim, shots, seeds)
        omega, padding = counts_history_to_omega(noisy_counts, n_red=n_red, m=m, shots=shots)
        results.update(noisy_aer=omega.tolist(), padding_mass_noisy=padding.tolist(), padding_mass_noisy_mean=float(padding.mean()),
                       l1_noisy_vs_exact=l1_sum(omega, omega_exact, decimals=L1_DECIMALS))
        if need_mitigation:
            from quantum.hardware.mitigation import (
                build_calibration_circuits, calibration_circuits_on, calibration_from_counts, calibration_groups,
                measured_physical_qubits,
            )

            # One calibration set per distinct measured-qubit tuple; every circuit uses the set of its own tuple.
            # Without a backend the circuits are unrouted and all measure the same register.
            measured = ([measured_physical_qubits(qc) for qc in transpiled] if fake_backend is not None
                        else [list(range(m))] * int(tau))
            groups = calibration_groups(measured)
            calibrations = []
            for g, qubits in enumerate(groups.tuples):
                if fake_backend is not None:
                    cal_circuits = calibration_circuits_on(qubits, fake_backend, int(seed) + g)
                else:
                    baseline, flips = build_calibration_circuits(m)
                    cal_circuits = [baseline, *flips]
                cal_counts = run_counts(cal_circuits, noisy_sim, shots, [int(seed) + g] * len(cal_circuits))
                calibrations.append(calibration_from_counts(cal_counts[0], cal_counts[1:], m, shots))
            results["calibration_qubit_tuples"] = [list(t) for t in groups.tuples]
            results["circuit_to_calibration"] = list(groups.index_of_circuit)
            omega_m, padding_m = mitigated_omega_by_circuit(noisy_counts, calibrations, groups.index_of_circuit, n_red=n_red, m=m, shots=shots)
            results.update(readout_mitigated=omega_m.tolist(), padding_mass_mitigated=padding_m.tolist(),
                           padding_mass_mitigated_mean=float(padding_m.mean()), l1_mitigated_vs_exact=l1_sum(omega_m, omega_exact, decimals=L1_DECIMALS))
    results["requested_layers"] = sorted(requested)
    results["elapsed_seconds"] = perf_counter() - t_start
    return results
