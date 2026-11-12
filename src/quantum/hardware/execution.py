"""The simulated devices: the one IBM fake backend (its calibration snapshot, device model and QPU-time estimate) and the generic Aer noise model.

Every consumer reads the target from ``config.IBM_FAKE_BACKEND`` and
always transpiles against the backend's real ``Target`` (basis gates plus
coupling map), never a generic basis-gate list. Fake backends ship as static
calibration snapshots inside ``qiskit_ibm_runtime``: no network access and no
IBM account is involved, and nothing here submits to real hardware.

:func:`build_depolarizing_readout_noise_model` is the simple depolarizing and
readout Aer noise model for reduced hardware-style circuits (the
``aer_noise_model`` arm).
"""

from __future__ import annotations

import statistics
from typing import Any

from config import IBM_FAKE_BACKEND


# ---------------------------------------------------------------------------
# The fake backend
# ---------------------------------------------------------------------------


DEFAULT_FAKE_BACKEND = IBM_FAKE_BACKEND
TWO_QUBIT_OP_NAMES = {"cx", "cz", "ecr", "rzz", "swap"}


def get_fake_backend(name: str = DEFAULT_FAKE_BACKEND):
    """Instantiate a local IBM fake-backend calibration snapshot."""
    import qiskit_ibm_runtime.fake_provider as fake_provider

    try:
        cls = getattr(fake_provider, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown fake backend {name!r} in qiskit_ibm_runtime.fake_provider") from exc
    return cls()


def two_qubit_count(isa_circuit: Any) -> int:
    return int(sum(c for name, c in isa_circuit.count_ops().items() if name in TWO_QUBIT_OP_NAMES))


def circuit_cost(isa_circuit: Any) -> dict[str, int]:
    """Width, depth, one-/two-qubit gates (one-qubit: all but 2q, measure, barrier, delay, reset), measurements, total."""
    ops = dict(isa_circuit.count_ops())
    non_gate = {"measure", "barrier", "delay", "reset"}
    two_qubit = two_qubit_count(isa_circuit)
    total = int(sum(c for name, c in ops.items() if name not in non_gate))
    return {
        "width": int(isa_circuit.num_qubits),
        "depth": int(isa_circuit.depth()),
        "two_qubit_gates": two_qubit,
        "one_qubit_gates": int(total - two_qubit),
        "measurements": int(ops.get("measure", 0)),
        "total_gates": total,
        "total_ops": int(sum(ops.values())),
    }


# ---------------------------------------------------------------------------
# Device model read off the backend's own calibration snapshot
# ---------------------------------------------------------------------------

_REP_DELAY_FALLBACK_S = 250e-6
_MAX_CIRCUITS_FALLBACK = 300
_MAX_SHOTS_FALLBACK = 100_000
# A snapshot whose median two-qubit error is at or above this is not a usable
# calibration (FakeTorino's, for one, reports 1.0 on its cz entries).
BROKEN_CALIBRATION_ERROR = 0.1


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def ibm_device_model(name: str = DEFAULT_FAKE_BACKEND) -> dict[str, Any]:
    """Read an IBM device's limits and calibration medians off its snapshot.

    Every constant the runtime budget model needs comes from the backend rather
    than a guess: register width, circuits per job, shots per circuit, the
    reset wait that dominates per-shot time, and the median two-qubit/readout
    error and duration.
    """
    backend = get_fake_backend(name)
    target = backend.target

    two_qubit_errors: list[float] = []
    two_qubit_durations: list[float] = []
    native_two_qubit_gates: list[str] = []
    one_qubit_errors: list[float] = []
    one_qubit_durations: list[float] = []
    for operation_name in target.operation_names:
        operation = target.operation_from_name(operation_name)
        n_q = getattr(operation, "num_qubits", None)
        if n_q not in (1, 2) or operation_name in ("measure", "reset", "delay"):
            continue
        for props in (target[operation_name] or {}).values():
            if props is None:
                continue
            if n_q == 2:
                if getattr(props, "error", None) is not None:
                    two_qubit_errors.append(float(props.error))
                if getattr(props, "duration", None) is not None:
                    two_qubit_durations.append(float(props.duration))
            else:
                if getattr(props, "error", None) is not None:
                    one_qubit_errors.append(float(props.error))
                if getattr(props, "duration", None) is not None:
                    one_qubit_durations.append(float(props.duration))
        if n_q == 2:
            native_two_qubit_gates.append(str(operation_name))

    readout_errors: list[float] = []
    readout_durations: list[float] = []
    if "measure" in target.operation_names:
        for props in (target["measure"] or {}).values():
            if props is None:
                continue
            if getattr(props, "error", None) is not None:
                readout_errors.append(float(props.error))
            if getattr(props, "duration", None) is not None:
                readout_durations.append(float(props.duration))

    conf_dict: dict[str, Any] = getattr(backend, "_conf_dict", None) or {}
    rep_delay_us = conf_dict.get("default_rep_delay")
    max_circuits = conf_dict.get("max_experiments") or getattr(backend, "max_circuits", None)
    max_shots = conf_dict.get("max_shots")
    processor = conf_dict.get("processor_type") or {}

    two_qubit_error = _median_or_none(two_qubit_errors)
    calibration_warning = None
    if two_qubit_error is not None and two_qubit_error >= BROKEN_CALIBRATION_ERROR:
        calibration_warning = (
            f"{name} reports a median two-qubit error of {two_qubit_error:.3g}, at or above "
            f"{BROKEN_CALIBRATION_ERROR}; this calibration snapshot is not usable"
        )

    return {
        "backend_name": name,
        "num_qubits": int(backend.num_qubits),
        "processor_family": processor.get("family"),
        "processor_revision": processor.get("revision"),
        "basis_gates": sorted(str(op) for op in target.operation_names),
        "native_two_qubit_gates": sorted(set(native_two_qubit_gates)),
        "max_circuits_per_job": int(max_circuits or _MAX_CIRCUITS_FALLBACK),
        "max_shots_per_circuit": int(max_shots or _MAX_SHOTS_FALLBACK),
        "default_rep_delay_s": (
            float(rep_delay_us) * 1e-6 if rep_delay_us is not None else _REP_DELAY_FALLBACK_S
        ),
        "two_qubit_error": two_qubit_error,
        "two_qubit_duration_s": _median_or_none(two_qubit_durations),
        "one_qubit_error": _median_or_none(one_qubit_errors),
        "one_qubit_duration_s": _median_or_none(one_qubit_durations),
        "readout_error": _median_or_none(readout_errors),
        "readout_duration_s": _median_or_none(readout_durations),
        "calibration_warning": calibration_warning,
    }


def estimate_runtime_seconds(isa_circuits, shots: int, model: dict[str, Any]) -> dict[str, float]:
    """Estimated QPU time for a batch: ``shots x (circuit duration + rep delay)``.

    Circuit duration is the ISA depth times the median two-qubit gate duration
    plus one readout -- an upper-bound-flavoured estimate, since not every
    layer of the depth is a two-qubit gate. Labelled an estimate wherever it is
    reported.

    The estimate covers only the circuits passed in (the noise grid passes the walk
    circuits and one readout-calibration set per distinct measured-qubit tuple). It
    excludes per-job overhead (queueing, compilation on the service, calibration
    loading), and it prices one execution of each circuit.
    """
    two_q = model.get("two_qubit_duration_s") or 60e-9
    readout = model.get("readout_duration_s") or 1e-6
    rep_delay = model.get("default_rep_delay_s") or _REP_DELAY_FALLBACK_S
    per_circuit = []
    for circuit in isa_circuits:
        duration = int(circuit.depth()) * float(two_q) + float(readout)
        per_circuit.append(float(shots) * (duration + float(rep_delay)))
    return {
        "estimated_seconds": float(sum(per_circuit)),
        "estimated_seconds_per_circuit_max": float(max(per_circuit)) if per_circuit else 0.0,
        "n_circuits": len(per_circuit),
        "shots": int(shots),
    }


# ---------------------------------------------------------------------------
# The generic Aer noise model
# ---------------------------------------------------------------------------


def build_depolarizing_readout_noise_model(
    *,
    one_qubit_prob: float = 0.001,
    two_qubit_prob: float = 0.01,
    readout_prob: float = 0.01,
):
    """Simple Aer noise model for reduced hardware-style circuits."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

    for name, value in {
        "one_qubit_prob": one_qubit_prob,
        "two_qubit_prob": two_qubit_prob,
        "readout_prob": readout_prob,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1], got {value}")

    noise_model = NoiseModel()
    one = depolarizing_error(float(one_qubit_prob), 1)
    two = depolarizing_error(float(two_qubit_prob), 2)

    for gate in ["id", "x", "sx", "h", "rz", "rx", "ry", "u", "u1", "u2", "u3"]:
        try:
            noise_model.add_all_qubit_quantum_error(one, [gate])
        except Exception:
            pass
    for gate in ["cx", "cz", "ecr"]:
        try:
            noise_model.add_all_qubit_quantum_error(two, [gate])
        except Exception:
            pass

    ro = ReadoutError([[1.0 - readout_prob, readout_prob], [readout_prob, 1.0 - readout_prob]])
    try:
        noise_model.add_all_qubit_readout_error(ro)
    except Exception:
        pass
    return noise_model
