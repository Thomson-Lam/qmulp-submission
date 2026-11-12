"""Gate-decomposed Szegedy circuit used only for reduced Qiskit/Aer simulation.

The circuit implements the fixed W² evolution through explicit swap,
multiplexed state preparation, and reflection gates:

    W = S_swap * R_Pi
    R_Pi = 2*Pi - I = B (I_source (x) Z0_target) B^dagger

Aer needs this decomposition so depolarizing noise can attach to actual one-
and two-qubit gates instead of one opaque dense ``UnitaryGate(W)``. This module
is not a supported Szegedy hardware-execution path. The dense
``SzegedyCircuit`` (``tests/oracles.py``) remains a tiny-graph correctness oracle:
it inherits the padding, the initial state, the measured qubits and the per-step
builders defined here and replaces only the state preparation and ``W`` by
their dense forms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import StatePreparation, UCRYGate

    _QISKIT_OK = True
    _QISKIT_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - environment dependent
    _QISKIT_OK = False
    _QISKIT_IMPORT_ERROR = exc

from quantum.inspired.validation import (
    DEFAULT_MAX_DENSE_BYTES,
    check_dense_allocation,
    validate_column_stochastic_matrix,
    validate_probability_vector,
)


def build_reflection_about_zero(m: int) -> "QuantumCircuit":
    """Return an ``m``-qubit circuit implementing ``Z0 = 2|0...0><0...0| - I``."""
    if m <= 0:
        raise ValueError(f"m must be positive; got {m}")
    qc = QuantumCircuit(m, name="Z0")
    if m == 1:
        qc.z(0)
        return qc
    qc.x(range(m))
    qc.h(m - 1)
    qc.mcx(list(range(m - 1)), m - 1)
    qc.h(m - 1)
    qc.x(range(m))
    # X*;H;MCX;H;X* implements a phase flip on |0...0> only (the complement of
    # Z0); an overall -1 global phase corrects it to Z0 = diag(+1,-1,...,-1).
    qc.global_phase += np.pi
    return qc


def build_multiplexed_state_prep(columns: list[np.ndarray], *, m: int) -> "QuantumCircuit":
    """Return the exact multiplexed operator ``B`` using a UCRY probability tree.

    The required action is ``B|source=j>|target=0> = |source=j>|g_j>``.
    ``columns[j]`` must be a normalized, finite, nonnegative real vector of
    length ``dim=2**m``. These are precisely the ``sqrt(G_pad[:, j])`` states
    used by :class:`StructuredSzegedyCircuit`.

    Target bits are prepared from most to least significant. At each level a
    uniformly controlled ``RY`` selects an angle from the source value and the
    target prefix already prepared. Qiskit's ``UCRYGate`` expects its target
    qubit first; its remaining qubits and angle indices use little-endian
    order. Register layout remains target first, source second, so the flat
    statevector index is ``source * dim + target``.
    """
    if m <= 0:
        raise ValueError(f"m must be positive; got {m}")
    dim = 2**m
    if len(columns) != dim:
        raise ValueError(f"expected {dim} columns for m={m}; got {len(columns)}")

    checked_columns: list[np.ndarray] = []
    for j, col in enumerate(columns):
        try:
            numeric = np.asarray(col, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {j} must contain numeric amplitudes") from exc
        if numeric.shape != (dim,):
            raise ValueError(f"column {j} has shape {numeric.shape}; expected ({dim},)")
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"column {j} must contain only finite amplitudes")
        if np.any(np.abs(numeric.imag) > 1e-12):
            raise ValueError(f"column {j} must contain real amplitudes")
        real = np.asarray(numeric.real, dtype=float)
        if np.any(real < -1e-12):
            raise ValueError(f"column {j} must contain nonnegative amplitudes")
        real = np.maximum(real, 0.0)
        norm = float(np.linalg.norm(real))
        if not np.isclose(norm, 1.0, atol=1e-10, rtol=1e-10):
            raise ValueError(f"column {j} must be normalized; norm={norm:.12g}")
        checked_columns.append(real)

    reg_t = QuantumRegister(m, "t")
    reg_s = QuantumRegister(m, "s")
    qc = QuantumCircuit(reg_t, reg_s, name="B")

    # At target bit k, source bits are the low-order UCR controls and the
    # already-prepared higher target bits encode the prefix. Consequently,
    # angle_index = source + dim * prefix.
    for k in reversed(range(m)):
        angles: list[float] = []
        prefix_count = 2 ** (m - k - 1)
        block_size = 2 ** (k + 1)
        split_size = 2**k
        for prefix in range(prefix_count):
            start = prefix * block_size
            split = start + split_size
            end = start + block_size
            for source in range(dim):
                col = checked_columns[source]
                left = float(np.linalg.norm(col[start:split]))
                right = float(np.linalg.norm(col[split:end]))
                angle = 0.0 if left == 0.0 and right == 0.0 else 2.0 * np.arctan2(right, left)
                angles.append(float(angle))

        controls = list(reg_s) + list(reg_t[k + 1 :])
        qc.append(UCRYGate(angles), [reg_t[k]] + controls)

    return qc


class StructuredSzegedyCircuit:
    """Reduced Qiskit/Aer Szegedy builder using explicit gate decomposition."""

    def __init__(
        self,
        G_red: np.ndarray,
        v_red: np.ndarray,
        *,
        max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES,
    ) -> None:
        if not _QISKIT_OK:
            raise ImportError(f"Qiskit is not installed/importable: {_QISKIT_IMPORT_ERROR}")

        self.G = validate_column_stochastic_matrix(
            np.asarray(G_red, dtype=float), name="G_red", atol=1e-8
        )
        self.v = validate_probability_vector(
            np.asarray(v_red, dtype=float), name="v_red", n=self.G.shape[0], atol=1e-8
        )
        self.n_red = self.G.shape[0]
        self.m = int(np.ceil(np.log2(max(self.n_red, 2))))
        self.dim = 2**self.m
        self.state_dim = self.dim**2

        # Qiskit retains each angle as a Python float plus a list reference;
        # budget four float64 words per angle as a conservative lower bound.
        check_dense_allocation(
            "structured UCRY angles",
            (4, self.dim, self.dim - 1),
            np.float64,
            max_dense_bytes,
        )
        check_dense_allocation(
            "structured initial statevector",
            (self.state_dim,),
            np.complex128,
            max_dense_bytes,
        )

        self.G_pad = np.zeros((self.dim, self.dim), dtype=float)
        self.G_pad[: self.n_red, : self.n_red] = self.G
        for j in range(self.dim):
            if self.G_pad[:, j].sum() < 1e-12:
                self.G_pad[:, j] = 1.0 / self.dim
        self.G_pad = validate_column_stochastic_matrix(self.G_pad, name="G_pad", atol=1e-8)

        self.v_pad = np.zeros(self.dim, dtype=float)
        self.v_pad[: self.n_red] = self.v
        if self.v_pad.sum() > 1e-12:
            self.v_pad /= self.v_pad.sum()
        else:
            self.v_pad[:] = 1.0 / self.dim
        self.v_pad = validate_probability_vector(self.v_pad, name="v_pad", n=self.dim, atol=1e-8)

        self._columns = [self._column_state(j) for j in range(self.dim)]
        self._W_step_cache: "QuantumCircuit" | None = None

    def _column_state(self, j: int) -> np.ndarray:
        return np.sqrt(np.maximum(self.G_pad[:, j], 0.0))

    def _initial_state_vector(self) -> np.ndarray:
        """Return |Psi_0> with index = source*dim + target (same convention as SzegedyCircuit)."""
        sv = np.zeros(self.state_dim, dtype=complex)
        for source in range(self.dim):
            if self.v_pad[source] < 1e-15:
                continue
            start = source * self.dim
            sv[start : start + self.dim] = np.sqrt(self.v_pad[source]) * self._columns[source]
        norm = np.linalg.norm(sv)
        if norm <= 1e-12:
            raise ValueError("Initial state vector has zero norm")
        return sv / norm

    def _initialization_circuit(self) -> "QuantumCircuit":
        """Prepare |Psi_0> without a dense state preparation on both registers.

        The initial state factorizes as ``B(|0>_target (x) sqrt(v)_source)``.
        Preparing only the ``m``-qubit source state avoids asking Qiskit to
        synthesize a generic ``2m``-qubit StatePreparation, which becomes
        impractical for the larger reduced graphs this builder targets.
        """
        m = self.m
        qr = QuantumRegister(2 * m, name="q")
        qc = QuantumCircuit(qr, name="Psi0_structured")
        target = qr[:m]
        source = qr[m:]

        qc.append(StatePreparation(np.sqrt(self.v_pad), label="sqrt(v)"), source)
        B = build_multiplexed_state_prep(self._columns, m=m)
        qc.append(B.to_gate(label="B_init"), list(target) + list(source))
        return qc

    def _measurement_qubits(self, measured_register: str) -> list[int]:
        if measured_register in {"target", "second", "destination"}:
            return list(range(self.m))
        if measured_register in {"source", "first", "position"}:
            return list(range(self.m, 2 * self.m))
        raise ValueError(
            "measured_register must be 'target'/'second'/'destination' or 'source'/'first'/'position'"
        )

    def _walk_step_circuit(self) -> "QuantumCircuit":
        """One structured application of W = S_swap * R_Pi, as a 2m-qubit gate."""
        if self._W_step_cache is not None:
            return self._W_step_cache

        m = self.m
        qr = QuantumRegister(2 * m, name="q")
        qc = QuantumCircuit(qr, name="W_structured")
        target = qr[:m]
        source = qr[m:]

        # R_Pi = B (I (x) Z0) B^dagger: applied to a state (time order left to
        # right), that means B^dagger first, then Z0, then B.
        B = build_multiplexed_state_prep(self._columns, m=m)
        qc.append(B.inverse().to_gate(label="B_dag"), list(target) + list(source))

        Z0 = build_reflection_about_zero(m)
        qc.append(Z0.to_gate(label="Z0"), list(target))

        qc.append(B.to_gate(label="B"), list(target) + list(source))

        for i in range(m):
            qc.swap(target[i], source[i])

        self._W_step_cache = qc
        return qc

    def build_statevector_circuit(self, t: int, *, step_power: int = 2) -> "QuantumCircuit":
        """State preparation + t recorded steps, with no measurement.

        Mirrors ``SzegedyCircuit.build_statevector_circuit`` so the structured
        circuit can be run through Aer's exact statevector method for parity
        against the exact reduced NumPy reference.
        """
        if step_power != 2:
            raise ValueError(f"step_power is fixed at 2 (W²); got {step_power}")
        qc = QuantumCircuit(2 * self.m, name=f"state_t{t}")
        qc.compose(self._initialization_circuit(), inplace=True)
        W_gate = self._walk_step_circuit().to_gate(label="W")
        for _ in range(t * step_power):
            qc.append(W_gate, list(range(2 * self.m)))
        return qc

    def build_circuits_per_step(
        self, tau: int, *, measured_register: str = "source", step_power: int = 2
    ) -> list["QuantumCircuit"]:
        """Build circuits for t=0..tau-1, matching SzegedyCircuit's interface."""
        if tau <= 0:
            raise ValueError(f"tau must be positive; got {tau}")
        if step_power != 2:
            raise ValueError(f"step_power is fixed at 2 (W²); got {step_power}")

        circuits: list[QuantumCircuit] = []
        init_gate = self._initialization_circuit().to_gate(label="Psi0")
        W_gate = self._walk_step_circuit().to_gate(label="W")
        measure_qubits = self._measurement_qubits(measured_register)

        for t in range(tau):
            qr = QuantumRegister(2 * self.m, name="q")
            cr = ClassicalRegister(self.m, name="c")
            qc = QuantumCircuit(qr, cr, name=f"step_{t}")
            qc.append(init_gate, qr)
            for _ in range(t * step_power):
                qc.append(W_gate, list(range(2 * self.m)))
            qc.measure(measure_qubits, list(range(self.m)))
            circuits.append(qc)
        return circuits
