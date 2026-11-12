"""The circuit tier: resources, the statevector rung, the noise grid, the candidate selection, its budget and its route.

Lumping sets the register width (``R`` supra-nodes, ``m = ceil(log2 R)``
qubits per register). The restart makes ``G-bar`` dense on the reference
blocks, so sparse-access constructions do not apply and the multiplexed state
preparation needs ``2^m (2^m - 1)`` angles; one reflection--swap then costs
thousands of two-qubit gates even on a tiny retained instance. The noise grid
measures, for nested retained instances and recorded steps ``tau = 1..5``, how
far three simulated arms stay from the exact lumped walk at the supra-node
level, against the pure-noise reference of the same instance, and selects one
instance per dataset for a hand-run submission outside the repository. Every
number is reported as it comes out; the exact and the noisy time averages are
compared at the supra-node level because the masked test borrowers of a small
instance may all sit in one class, where a borrower-level rank correlation has
nothing to rank (the masked-class Spearman is reported where two classes exist).

Grading, the pure-noise band and the column and arm definitions live in
``noise_grading``, the shard identity in ``shards``.

The simulation budget
---------------------
Which cells of one retained instance are run and which are recorded as not
simulated. Two limits, both recorded with their estimate when they decline a cell:

- **memory** (deterministic): a noisy arm runs as a density matrix of ``16 * 4^n`` bytes over the ``n``
  physical qubits the routed circuit touches (routing spreads a ``2m``-qubit circuit over more), the ideal
  arm as a statevector of ``16 * 2^n`` bytes; a cell whose state exceeds ``hardware_density_matrix_max_bytes``
  is not run;
- **time**: the cells of an instance run in recorded-step order, all arms at each step, and share
  ``hardware_instance_budget_seconds``. A gate of a simulated circuit sweeps the whole state, so a cell costs
  about ``seconds_per_gate_byte * gates * state_bytes``. Before an arm has run, the rate is the one an arm of the
  same kind (density matrix or statevector) measured, and failing that the assumed throughput
  ``hardware_prior_bytes_per_second`` (a deliberately pessimistic one, so a first cell that could not end inside
  the allocation is declined, not started). After a cell the arm's own measured rate replaces it. A cell whose
  estimate would end past the budget is not run.

The time limit depends on the speed of the node, so *which* borderline cells exist can differ between machines;
every simulated cell is deterministic (seeded) and every declined cell says why, with the numbers.

The router's verdict on a retained instance
-------------------------------------------
A retained instance is a connected set of whole lumped classes with its operator
rebuilt from its own graph (``class_candidates``). Its member-level counterpart --
the same classes' individuals as one node per individual per layer, built by the
same builder -- is the operator the lumped one must be the quotient of. The
route functions below measure both residuals of that pair and record the route,
so a retained instance no longer bypasses the router (the full instance is the
stage's primary operator, already routed in stage 04, and must route to
``D_bar``).

The circuits implement the walk on the retained lumped operator itself, whatever the
route: the route says whether that walk is also the exact quotient of the member-level
walk (``D_bar``) or only of its compression (``H``). It is recorded, never used to
change the circuit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from quantum.hardware.class_candidates import build_candidate, sweep_sizes
from quantum.hardware.execution import circuit_cost, estimate_runtime_seconds, two_qubit_count
from quantum.hardware.mitigation import (
    calibration_circuits_on,
    calibration_from_counts,
    calibration_groups,
    measured_physical_qubits,
)
from quantum.hardware.noise_grading import (
    ARM_DEFINITIONS,
    NOT_SIMULATED,
    READOUT_CALIBRATION_NOTE,
    grade,
    inside_band,
    is_simulated,
    pure_noise_band,
)
from quantum.lumped_operator import build_lumped_matrices, seed_mask
from quantum.lumping import UNKNOWN_LABEL
from quantum.lumping_router import (
    ROUTE_DBAR,
    RoutedOperator,
    block_structure,
    discriminant,
    lumpability_residual,
    route_operator,
)
from quantum.mechanism.running_average import step_marginals
from quantum.mechanism.variants import full_graph


# ---------------------------------------------------------------------------
# The simulation budget
# ---------------------------------------------------------------------------


DENSITY_ARMS = ("aer_noise_model", "fake_marrakesh")
GIB = 2**30


def state_bytes(arm: str, n_active: int) -> int:
    """Bytes of the simulated state of ``arm`` at ``n_active`` touched qubits."""
    return int(16 * (4 ** int(n_active) if arm in DENSITY_ARMS else 2 ** int(n_active)))


def kind_of(arm: str) -> str:
    return "density_matrix" if arm in DENSITY_ARMS else "statevector"


@dataclass
class SimulationBudget:
    """Shared by every arm and recorded step of one instance; ``decide`` before a cell, ``record`` after it."""

    total_seconds: float
    memory_bytes: int
    prior_bytes_per_second: float
    spent_seconds: float = 0.0
    seconds_per_gate_byte: dict[str, float] = field(default_factory=dict)

    def rate(self, arm: str) -> float:
        """Seconds per (gate x state byte): the arm's own, else the mean of its kind's, else the assumed throughput."""
        own = self.seconds_per_gate_byte.get(arm)
        if own is not None:
            return float(own)
        same_kind = [r for a, r in self.seconds_per_gate_byte.items() if kind_of(a) == kind_of(arm)]
        return sum(same_kind) / len(same_kind) if same_kind else 1.0 / float(self.prior_bytes_per_second)

    def estimate(self, arm: str, gates: int, bytes_needed: int) -> float:
        """Estimated seconds of a cell of ``gates`` ISA gates on a state of ``bytes_needed`` bytes."""
        return self.rate(arm) * int(gates) * int(bytes_needed)

    def decide(self, arm: str, *, gates: int, bytes_needed: int) -> tuple[bool, dict[str, Any]]:
        """``(run, info)``; ``info`` carries the estimates and, when the cell is declined, the ``reason``."""
        estimate = self.estimate(arm, gates, bytes_needed)
        info: dict[str, Any] = {"estimated_gates": int(gates), "estimated_state_bytes": int(bytes_needed),
                                "estimated_simulation_seconds": estimate, "budget_seconds": float(self.total_seconds),
                                "spent_seconds": float(self.spent_seconds)}
        if bytes_needed > self.memory_bytes:
            info["reason"] = (f"the {kind_of(arm).replace('_', ' ')} of the routed circuit needs {bytes_needed / GIB:.1f} GiB, "
                              f"above the {self.memory_bytes / GIB:.0f} GiB budget (estimate)")
            return False, info
        if self.spent_seconds + estimate > self.total_seconds:
            info["reason"] = (f"estimated {estimate / 3600.0:.2f} h of simulation ({int(gates):,} gates on {bytes_needed / GIB:.2f} GiB) after "
                              f"{self.spent_seconds / 3600.0:.2f} h already spent exceeds the {self.total_seconds / 3600.0:.2f} h instance "
                              "budget (estimate)")
            return False, info
        return True, info

    def record(self, arm: str, *, gates: int, bytes_needed: int, seconds: float) -> None:
        """Account a simulated cell: its seconds count against the budget and set the arm's rate."""
        self.spent_seconds += float(seconds)
        self.seconds_per_gate_byte[arm] = float(seconds) / (max(int(gates), 1) * max(int(bytes_needed), 1))


# ---------------------------------------------------------------------------
# The router's verdict on a retained instance
# ---------------------------------------------------------------------------


def member_ids(nodes: pd.DataFrame) -> list[int]:
    """Every borrower id of the classes of a lumped frame, sorted."""
    copies = nodes[nodes["node_type"].astype(str) == "borrower_copy"]
    ids: set[int] = set()
    for members in copies["member_borrower_ids"].astype(str):
        ids.update(int(m) for m in members.split(";") if m)
    return sorted(ids)


def routed_record(routed: RoutedOperator) -> dict[str, Any]:
    """The route and residuals of a routed operator as scalars."""
    r = routed.residuals
    return {"route": routed.route, "lumpability_residual": float(r["lumpability_residual"]),
            "compatibility_residual": float(r["compatibility_residual"]), "n_full": int(r.get("n_full", 0)),
            "n_reduced": int(r.get("n_reduced", routed.n)), "note": ""}


def primary_route_record(routed: RoutedOperator) -> dict[str, Any]:
    """The full instance's record, from the primary operator's own routing; it must route to ``D_bar``."""
    if routed.route != ROUTE_DBAR:
        raise AssertionError(f"the full instance must route to {ROUTE_DBAR}; the router chose {routed.route} "
                             f"(compatibility residual {routed.residuals['compatibility_residual']:.3e})")
    return {**routed_record(routed), "note": "the stage's primary operator"}


def not_routed_record(note: str) -> dict[str, Any]:
    return {"route": None, "lumpability_residual": None, "compatibility_residual": None, "n_full": None, "n_reduced": None, "note": note}


def retained_route_record(instance: dict[str, Any], partitions, spec, config, hidden_ids) -> dict[str, Any]:
    """Route a retained instance against the member-level operator of its own classes.

    A retained operator that is not the quotient of its member-level operator is recorded with the
    lumpability residual that says so (route null), not raised: the circuits are what they are.
    """
    ids = member_ids(instance["nodes"])
    nodes_f, edges_f = full_graph(partitions, spec, config, hidden_ids, stickiness=config.stickiness, borrower_ids=ids)
    G_full, _v_full = build_lumped_matrices(nodes_f, edges_f, alpha=config.resolved_alpha(), influence_type=config.influence_type)
    blocks = block_structure(instance["nodes"], nodes_f)
    lump = lumpability_residual(G_full, instance["G"], blocks)
    if lump > float(config.router_tol):
        return {**not_routed_record("the retained operator is not the quotient of the member-level operator of its classes"),
                "lumpability_residual": float(lump), "n_full": int(blocks.n_full), "n_reduced": int(blocks.n_blocks)}
    routed = route_operator(instance["G"], instance["v"], G_full=G_full, blocks=blocks, router_tol=config.router_tol)
    return {**routed_record(routed), "n_full": int(blocks.n_full), "n_reduced": int(blocks.n_blocks),
            "note": "the retained operator against the member-level operator of the same classes"}


# ---------------------------------------------------------------------------
# The noise grid and the candidate selection
# ---------------------------------------------------------------------------


NOISE_ARMS = ("noiseless_shots", "aer_noise_model", "fake_marrakesh")
GROWTH_RULE = ("nested connected growth from the reference class with the most restart mass, adding at each step the class "
               "sharing the most category nodes with the retained set (ties: larger restart mass, then key order); "
               "quantum.hardware.class_candidates.grow_connected_classes")
PADDING_CONVENTION = ("padding codes carry uniform columns in G_pad (quantum.hardware.structured_circuit); measured mass on "
                      "padding codes is removed and the retained support renormalised")
MEASURED_QUBITS_NOTE = ("the per-step circuits are routed independently on one shared initial layout; routing swaps move the "
                        "measured (source) register, so the circuits do not all measure the same physical qubits and their readout "
                        "calibrations are not interchangeable")


def active_qubits(isa_circuit) -> int:
    """Physical qubits touched by at least one gate or measurement of an ISA circuit."""
    touched = set()
    for instruction in isa_circuit.data:
        if instruction.operation.name in ("barrier", "delay"):
            continue
        touched.update(isa_circuit.find_bit(q).index for q in instruction.qubits)
    return len(touched)


def masked_groups(nodes: pd.DataFrame) -> int:
    """Classes (layer-0 copies) whose masked label is unknown: the test groups a borrower ranking can order."""
    copies = nodes[(nodes["node_type"].astype(str) == "borrower_copy") & (nodes["layer_index"].astype(int) == 0)]
    return int((copies["BAD_masked"].astype(int) == UNKNOWN_LABEL).sum())


def analytic_two_qubit_count(m: int) -> int:
    """Logical CNOTs of one reflection--swap: two multiplexed preparations, the reflection about zero and ``m`` swaps.

    A uniformly controlled rotation with ``c`` controls costs ``2^c`` CNOTs
    (Mottonen et al.); level ``k`` of the preparation has ``2m - k - 1``
    controls, so ``B`` costs ``2^m (2^m - 1)`` CNOTs. The reflection about
    zero is an ``(m-1)``-controlled Toffoli (``O(m^2)`` CNOTs without
    ancillas, counted as ``2 m^2``) and each swap three CNOTs.
    """
    m = int(m)
    return int(2 * (2**m) * (2**m - 1) + 2 * m * m + 3 * m)


def resources_row(dataset: str, nodes: pd.DataFrame, G: np.ndarray, v: np.ndarray, config, *, label: str, borrowers: int,
                  route: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classes, ``R``, ``m``, qubits, density and support facts, angles, the logical two-qubit count and the route."""
    R = int(G.shape[0])
    m = int(np.ceil(np.log2(max(R, 2))))
    refs = seed_mask(nodes)
    support = G > 0
    full_ref_support = float(np.mean(support[refs, :].all(axis=0))) if refs.any() else float("nan")
    copies = nodes[(nodes["node_type"].astype(str) == "borrower_copy") & (nodes["layer_index"].astype(int) == 0)]
    route = route or {}
    row = {"dataset": dataset, "instance": label, "classes": int(len(copies)), "R": R, "m": m, "logical_qubits": 2 * m,
           "borrowers": int(borrowers), "gbar_density": float(support.sum() / (R * R)), "max_column_support": int(support.sum(axis=0).max()),
           "fraction_columns_full_reference_support": full_ref_support, "state_prep_angles": int((2**m) * (2**m - 1)),
           "two_qubit_logical_analytic": analytic_two_qubit_count(m), "two_qubit_logical_transpiled": None,
           "n_masked_groups": masked_groups(nodes), "route": route.get("route"),
           "lumpability_residual": route.get("lumpability_residual"), "compatibility_residual": route.get("compatibility_residual")}
    if m <= int(config.hardware_max_register_width):
        from qiskit import transpile

        from quantum.hardware.structured_circuit import StructuredSzegedyCircuit

        struct = StructuredSzegedyCircuit(G, v)
        step = transpile(struct._walk_step_circuit(), basis_gates=["rz", "sx", "x", "cx"],
                         optimization_level=int(config.hardware_optimization_level),
                         seed_transpiler=config.seed_transpiler(dataset, label, "logical"))
        row["two_qubit_logical_transpiled"] = two_qubit_count(step)
    return row


def retained_instances(nodes: pd.DataFrame, edges: pd.DataFrame, config) -> list[dict[str, Any]]:
    """The nested retained instances (full first), each with its rebuilt operator and masked-group count."""
    from quantum.hardware.class_candidates import class_keys

    n_full = len(class_keys(nodes))
    out = []
    for k in sweep_sizes(config, n_full):
        candidate = build_candidate(nodes, edges, k, config)
        op = candidate["operator"]
        out.append({"k": int(k), "label": candidate["label"], "nodes": candidate["nodes"], "edges": candidate["edges"],
                    "G": op["G"], "v": op["v"], "R": int(op["n_supra_nodes"]), "m": int(op["register_width_m"]),
                    "borrowers": int(op["borrowers_covered"]), "test_borrowers": int(op["test_borrowers_covered"]),
                    "n_masked_groups": masked_groups(candidate["nodes"]), "simulated": int(op["register_width_m"]) <= int(config.hardware_max_register_width)})
    return out


def transpile_shared_layout(struct, n_steps: int, backend, seed: int, *, optimization_level: int) -> dict[str, Any]:
    """Per-step ISA circuits for ``t = 0 .. n_steps - 1`` on one shared initial layout.

    The deepest circuit is transpiled first (Sabre layout and routing, at the
    preset level); its initial layout is then imposed on every circuit, so all of them
    start on the same physical qubits. Routing then places the measured register on
    different physical qubits in different circuits (:data:`MEASURED_QUBITS_NOTE`); the
    measured tuple of every circuit is recorded and the readout calibration is built
    per distinct tuple.
    """
    from qiskit.transpiler import generate_preset_pass_manager

    circuits = struct.build_circuits_per_step(int(n_steps), measured_register="source", step_power=2)
    began = perf_counter()
    level = int(optimization_level)
    first = generate_preset_pass_manager(target=backend.target, optimization_level=level, seed_transpiler=int(seed),
                                         layout_method="sabre", routing_method="sabre").run(circuits[-1])
    layout = [int(q) for q in first.layout.initial_index_layout(filter_ancillas=True)]
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=level, seed_transpiler=int(seed),
                                      initial_layout=layout, routing_method="sabre")
    isa = [pm.run(qc) for qc in circuits]
    measured = [measured_physical_qubits(qc) for qc in isa]
    active = [active_qubits(qc) for qc in isa]
    return {"isa": isa, "initial_layout": layout, "measured_qubits": measured, "transpile_seconds": perf_counter() - began,
            "costs": [circuit_cost(qc) for qc in isa], "active_qubits": active, "active_qubits_max": int(max(active))}


def predicted_survival_conservative(costs: list[dict[str, int]], model: dict[str, Any]) -> list[float]:
    """``prod (1 - e)`` over the gates of each circuit, every one-qubit gate (RZ included) at the one-qubit median."""
    e1, e2, r = (float(model[k]) for k in ("one_qubit_error", "two_qubit_error", "readout_error"))
    return [float((1.0 - e1) ** int(c["one_qubit_gates"]) * (1.0 - e2) ** int(c["two_qubit_gates"]) * (1.0 - r) ** int(c["measurements"]))
            for c in costs]


def _simulator(arm: str, backend, model: dict[str, Any]):
    """The Aer simulator of one arm, sized like the backend so the ISA circuits run unchanged, and its description.

    The arms are defined in ``noise_grading.ARM_DEFINITIONS``. A noisy arm runs as a density matrix over the qubits
    the circuit touches; whether a cell fits the budget is decided per cell (:class:`SimulationBudget`), not here.
    """
    from qiskit_aer import AerSimulator

    from quantum.hardware.execution import build_depolarizing_readout_noise_model

    if arm == "noiseless_shots":
        return AerSimulator.from_backend(backend, noise_model=None), "no noise"
    if arm == "aer_noise_model":
        generic = build_depolarizing_readout_noise_model(one_qubit_prob=float(model["one_qubit_error"]),
                                                         two_qubit_prob=float(model["two_qubit_error"]),
                                                         readout_prob=float(model["readout_error"]))
        return AerSimulator.from_backend(backend, noise_model=generic, method="density_matrix"), \
            "depolarizing 1q/2q + symmetric readout at the device medians, no thermal relaxation (density matrix)"
    if arm == "fake_marrakesh":
        return AerSimulator.from_backend(backend, method="density_matrix"), \
            "NoiseModel.from_backend: depolarizing + thermal relaxation + readout from the calibration (density matrix)"
    raise ValueError(f"unknown arm {arm!r}")


def _calibrate(arm: str, simulator, cal_circuits, dataset: str, k: int, m: int, config) -> tuple[list, list[dict[str, Any]]]:
    """One calibration set per distinct measured-qubit tuple, run on the arm's own simulator, and its record."""
    from quantum.hardware.aer import run_counts

    sets, record = [], []
    for g, circuits in enumerate(cal_circuits):
        seeds = [config.seed_aer(dataset, k, arm, "calibration", g, i) for i in range(len(circuits))]
        counts = run_counts(circuits, simulator, config.hardware_shots, seeds)
        calibration = calibration_from_counts(counts[0], counts[1:], m, config.hardware_shots)
        sets.append(calibration)
        record.append({"set": g, "p_meas0_prep0": [float(a[0, 0]) for a in calibration.matrices],
                       "p_meas0_prep1": [float(a[0, 1]) for a in calibration.matrices]})
    return sets, record


def simulate_instance(dataset: str, instance: dict[str, Any], layout: dict[str, Any], config, backend, model: dict[str, Any], *,
                      route: dict[str, Any], checkpoint: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Every cell of one retained simulated instance: rows, supra-node vectors, the pure-noise band and the calibration record.

    Cells run in recorded-step order, all arms at each step, under one :class:`SimulationBudget`; a cell that
    does not fit is recorded as not simulated with its estimate. ``checkpoint`` receives the partial record after
    every cell, so a task that is killed leaves the cells it finished.
    """
    from quantum.hardware.aer import counts_history_to_omega, mitigated_omega_by_circuit, run_counts

    G, v, nodes = instance["G"], instance["v"], instance["nodes"]
    R, m, k = instance["R"], instance["m"], instance["k"]
    taus = [int(t) for t in config.hardware_steps]
    steps = step_marginals(G, v, discriminant(G), max(taus))
    exact = {tau: steps[:tau].mean(axis=0) for tau in taus}
    band = pure_noise_band(exact, m=m, shots=config.hardware_shots, draws=config.hardware_noise_band_draws,
                           quantiles=config.hardware_noise_band_quantiles, seed=config.seed_noise_band(dataset, k))
    groups = calibration_groups(layout["measured_qubits"])
    cal_circuits = [calibration_circuits_on(qubits, backend, config.seed_transpiler(dataset, k, "calibration", g))
                    for g, qubits in enumerate(groups.tuples)]
    budget = SimulationBudget(total_seconds=float(config.hardware_instance_budget_seconds), memory_bytes=int(config.hardware_density_matrix_max_bytes),
                              prior_bytes_per_second=float(config.hardware_prior_bytes_per_second))
    arms = {arm: _simulator(arm, backend, model) for arm in config.hardware_noise_arms}
    calibrations: dict[str, list] = {}
    rows: list[dict[str, Any]] = []
    vectors: dict[str, Any] = {"exact": {str(t): exact[t].tolist() for t in taus}}
    record: dict[str, Any] = {
        "k": k, "label": instance["label"], "status": "simulated", "rows": rows, "vectors": vectors,
        "noise_band": {str(t): band[t] for t in taus}, "route": route, "arms": {a: ARM_DEFINITIONS[a] for a in config.hardware_noise_arms},
        "layout": {"initial_layout": layout["initial_layout"], "measured_qubits": layout["measured_qubits"],
                   "measured_qubits_note": MEASURED_QUBITS_NOTE, "active_qubits": layout["active_qubits"],
                   "active_qubits_max": layout["active_qubits_max"], "transpile_seconds": layout["transpile_seconds"], "costs": layout["costs"]},
        "calibration": {"note": READOUT_CALIBRATION_NOTE, "measured_qubit_tuples": [list(t) for t in groups.tuples],
                        "circuit_to_calibration": list(groups.index_of_circuit), "sets": {}},
    }

    def flush() -> None:
        """Account the budget in the record and hand the partial record to the checkpoint: after every cell, so a task killed
        part-way through an expensive cell still leaves every cell it finished."""
        record["budget"] = {"spent_seconds": budget.spent_seconds, "seconds_per_gate_byte": dict(budget.seconds_per_gate_byte),
                            "limit_seconds": budget.total_seconds, "prior_bytes_per_second": budget.prior_bytes_per_second}
        if checkpoint is not None:
            checkpoint(record)

    for tau in taus:
        isa, costs, actives = layout["isa"][:tau], layout["costs"][:tau], layout["active_qubits"][:tau]
        gates = int(sum(c["total_gates"] for c in costs))
        needed = groups.n_sets_for(tau)
        n_active = int(max(actives))
        b = band[tau]
        base_common = {"dataset": dataset, "k_classes": k, "instance": instance["label"], "R": R, "m": m, "borrowers": instance["borrowers"],
                       "n_masked_groups": instance["n_masked_groups"], "tau": tau, "shots": int(config.hardware_shots),
                       "depth_max": int(max(c["depth"] for c in costs)), "two_qubit_sum": int(sum(c["two_qubit_gates"] for c in costs)),
                       "predicted_survival_min": float(min(predicted_survival_conservative(costs, model))),
                       "estimated_qpu_seconds": estimate_runtime_seconds([*isa, *(c for g in range(needed) for c in cal_circuits[g])],
                                                                       config.hardware_shots, model)["estimated_seconds"],
                       "active_qubits": n_active, "n_calibration_sets": needed, "optimization_level": int(config.hardware_optimization_level),
                       "transpiler_seed": config.seed_transpiler(dataset, k), "basis": ",".join(model["native_two_qubit_gates"]),
                       "route": route.get("route"), "lumpability_residual": route.get("lumpability_residual"),
                       "compatibility_residual": route.get("compatibility_residual"),
                       "pure_noise_spearman_lo": b["spearman_lo"], "pure_noise_spearman_hi": b["spearman_hi"],
                       "pure_noise_fidelity_lo": b["fidelity_lo"], "pure_noise_fidelity_hi": b["fidelity_hi"]}
        for arm in config.hardware_noise_arms:
            simulator, label = arms[arm]
            base = {**base_common, "arm": arm, "noise": label}
            run, info = budget.decide(arm, gates=gates, bytes_needed=state_bytes(arm, n_active))
            if not run:
                rows.append({**base, "mitigated": False, "status": NOT_SIMULATED, **info})
                flush()
                continue
            began = perf_counter()
            counts = run_counts(isa, simulator, config.hardware_shots, [config.seed_aer(dataset, k, tau, arm, t) for t in range(tau)])
            omega, padding = counts_history_to_omega(counts, n_red=R, m=m, shots=config.hardware_shots)
            graded = grade(omega, exact[tau], nodes, config.tie_tolerance)
            seconds = perf_counter() - began
            budget.record(arm, gates=gates, bytes_needed=info["estimated_state_bytes"], seconds=seconds)
            # "inside_pure_noise_band" (the Spearman inside its band) stays in the shard row exactly as every earlier run wrote it, so
            # that no shard needs re-simulation; the assembled grid drops it and carries "signal_above_band" instead
            # (noise_grading.annotate_signal, computed from the fidelity ratio and its band edge).
            rows.append({**base, "mitigated": False, "padding_mass": float(padding.mean()), **graded, "wall_seconds": seconds,
                         "estimated_gates": gates, "estimated_state_bytes": info["estimated_state_bytes"],
                         "estimated_simulation_seconds": info["estimated_simulation_seconds"],
                         "inside_pure_noise_band": inside_band(graded["spearman_supra"], b["spearman_lo"], b["spearman_hi"])})
            vectors.setdefault(arm, {"raw": {}, "mitigated": {}})["raw"][str(tau)] = omega.tolist()
            if arm == "noiseless_shots":
                flush()
                continue
            if arm not in calibrations:
                calibrations[arm], record["calibration"]["sets"][arm] = _calibrate(arm, simulator, cal_circuits, dataset, k, m, config)
            omega_m, padding_m = mitigated_omega_by_circuit(counts, calibrations[arm], groups.index_of_circuit[:tau],
                                                            n_red=R, m=m, shots=config.hardware_shots)
            rows.append({**base, "mitigated": True, "padding_mass": float(padding_m.mean()), **grade(omega_m, exact[tau], nodes, config.tie_tolerance)})
            vectors[arm]["mitigated"][str(tau)] = omega_m.tolist()
            flush()
    return record


def not_simulated_rows(dataset: str, instance: dict[str, Any], config, route: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every cell of an instance wider than the register cap, with the estimate of what it would have needed."""
    route = route or {}
    reason = (f"register width m={instance['m']} exceeds hardware_max_register_width={config.hardware_max_register_width}; one "
              f"reflection--swap alone needs about {analytic_two_qubit_count(instance['m']):,} logical two-qubit gates (estimate)")
    return [{"dataset": dataset, "k_classes": instance["k"], "instance": instance["label"], "R": instance["R"], "m": instance["m"],
             "borrowers": instance["borrowers"], "n_masked_groups": instance["n_masked_groups"], "tau": int(tau), "arm": arm,
             "mitigated": False, "status": NOT_SIMULATED, "reason": reason, "route": route.get("route"),
             "lumpability_residual": route.get("lumpability_residual"), "compatibility_residual": route.get("compatibility_residual"),
             "estimated_two_qubit_logical": analytic_two_qubit_count(instance["m"])}
            for tau in config.hardware_steps for arm in config.hardware_noise_arms]


def statevector_rung(G: np.ndarray, v: np.ndarray, config, *, budget_seconds: float, dataset: str) -> dict[str, Any]:
    """Aer statevector of the logical lumped circuit for ``t = 0..T``, ``T`` as fits the wall budget, against the NumPy walk."""
    R = int(G.shape[0])
    m = int(np.ceil(np.log2(max(R, 2))))
    if m > int(config.hardware_statevector_max_register_width):
        angles = (2**m) * (2**m - 1)
        return {"dataset": dataset, "R": R, "m": m, "logical_qubits": 2 * m, "status": "not attempted",
                "reason": f"m={m} exceeds the statevector register cap {config.hardware_statevector_max_register_width}",
                "estimated_angle_bytes": angles * 32, "estimated_statevector_bytes": 16 * (4**m), "T": 0}
    from quantum.hardware.aer import lumped_statevector_history

    probe, probe_seconds = lumped_statevector_history(G, v, tau=2, seed=config.seed_aer(dataset, "statevector", "probe"))
    per_step = max(probe_seconds / 2.0, 1e-6)
    T = int(max(1, min(int(max(config.hardware_steps)) * 10, (float(budget_seconds) - probe_seconds) // per_step)))
    history, seconds = lumped_statevector_history(G, v, tau=T, seed=config.seed_aer(dataset, "statevector"))
    exact = step_marginals(G, v, discriminant(G), T)
    errors = np.abs(history - exact).max(axis=1)
    return {"dataset": dataset, "R": R, "m": m, "logical_qubits": 2 * m, "status": "ran", "T": T, "wall_seconds": seconds,
            "probe_seconds": probe_seconds, "max_abs_error": float(errors.max()), "max_abs_error_per_step": errors.tolist(),
            "budget_seconds": float(budget_seconds)}


def select_candidate(rows: list[dict[str, Any]], config) -> dict[str, Any]:
    """Maximise the fidelity ratio under ``fake_marrakesh`` subject to rankable test groups, the QPU budget and the width cap.

    Only cells at ``tau >= hardware_min_tau`` qualify: at ``tau = 1`` the circuit holds the state preparation alone
    (``p_0 = v``), which is not a walk result. An instance with fewer than two masked classes or a fidelity ratio
    below the floor is only chosen, flagged "cannot rank test borrowers", when no other qualifies.
    """
    fake = [r for r in rows if r.get("arm") == "fake_marrakesh" and not r.get("mitigated") and is_simulated(r)
            and int(r.get("tau", 0)) >= int(config.hardware_min_tau) and np.isfinite(r.get("fidelity_ratio", float("nan")))]
    budget = [r for r in fake if r["estimated_qpu_seconds"] <= float(config.hardware_max_runtime_seconds)
              and r["m"] <= int(config.hardware_max_register_width)]
    rankable = [r for r in budget if r["n_masked_groups"] >= 2 and r["fidelity_ratio"] >= float(config.hardware_min_fidelity_ratio)]
    pool, flag = (rankable, "") if rankable else (budget, "cannot rank test borrowers")
    if not pool:
        return {"selected": None, "flag": "no instance within the budget", "n_candidates": 0}
    best = max(pool, key=lambda r: (r["fidelity_ratio"], r["borrowers"], r["tau"]))
    keys = ("k_classes", "instance", "R", "m", "borrowers", "tau", "fidelity_ratio", "n_masked_groups", "estimated_qpu_seconds",
            "depth_max", "two_qubit_sum", "predicted_survival_min", "tv_uniform", "route", "n_calibration_sets")
    return {"selected": {k: best.get(k) for k in keys}, "flag": flag, "n_candidates": len(pool),
            "rule": f"max fidelity_ratio under fake_marrakesh at tau >= {config.hardware_min_tau}; n_masked_groups >= 2 if any such "
                    f"instance reaches {config.hardware_min_fidelity_ratio}; estimated QPU <= {config.hardware_max_runtime_seconds} s; "
                    f"m <= {config.hardware_max_register_width}"}
