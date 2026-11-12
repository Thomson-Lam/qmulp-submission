"""Export the selected circuit instance of a dataset as data, for a run made by hand outside the repository.

    python -m scripts.export_hardware_instance --dataset hmeq
    python -m scripts.export_hardware_instance --dataset fmac
    python -m scripts.export_hardware_instance --dataset hmeq --k 3     # another retained instance, the selected walk length

A data export, not a submission path: it imports no runtime account or credential code and sends nothing
anywhere. It reads the selection of stage 06 (``hardware_candidates.json``), rebuilds the retained operator of
the selected instance (or, with ``--k``, of the instance with that many classes, at the selected walk length),
checks its exact average against the shard the stage wrote, and writes under
``outputs/results/<dataset>/hardware_export/`` (``hardware_export_k<k>/`` for an instance other than the selected one):

- ``walk_circuits_logical.qpy``      the walk circuits ``t = 0 .. tau - 1`` as built, before any transpilation
- ``walk_circuits_basis_gates.qpy``  the same circuits expressed in ``rz, sx, x, cx`` without routing (level 0):
                                     still logical, with no coupling map and no layout
- ``exact_average.csv``              the exact lumped time average over the supra-nodes
- ``supra_node_map.csv``             supra-node index -> node, register code, class size, masked label; padding codes
- ``instance.json``                  shots, tau, sizes, seeds, the calibration snapshot facts, the padding and calibration
                                     conventions, and the SHA-256 of every file
- ``HOWTO.txt``                      how to run exactly what was simulated and compare it with the exact vector
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG  # noqa: E402
from datasets import DATASET_NAMES, get_dataset  # noqa: E402
from evaluation.context import DatasetContext  # noqa: E402
from evaluation.provenance import read_result  # noqa: E402
from fingerprint import resolve_config, sha256_path  # noqa: E402
from paths import relative_to_repo, results_dir  # noqa: E402
from quantum.hardware import noise_grid as grid, shards  # noqa: E402
from quantum.hardware.structured_circuit import StructuredSzegedyCircuit  # noqa: E402
from quantum.lumping_router import discriminant  # noqa: E402
from quantum.mechanism.running_average import step_marginals  # noqa: E402

BASIS_GATES = ["rz", "sx", "x", "cx"]


def dump_qpy(path: Path, circuits: list) -> str | None:
    """Write ``circuits`` as QPY and read them back; ``None`` on success, else the error (and no file is left)."""
    from qiskit import qpy

    try:
        with open(path, "wb") as handle:
            qpy.dump(circuits, handle)
        with open(path, "rb") as handle:
            loaded = list(qpy.load(handle))
        if len(loaded) != len(circuits) or any(a.num_qubits != b.num_qubits for a, b in zip(loaded, circuits)):
            raise ValueError("the circuits read back from the QPY file do not match the ones written")
        return None
    except Exception as exc:
        path.unlink(missing_ok=True)
        return f"{type(exc).__name__}: {exc}"


def howto(facts: dict, files: dict) -> str:
    m, tau, shots = facts["m"], facts["tau"], facts["shots"]
    walk = "walk_circuits_logical.qpy" if "walk_circuits_logical.qpy" in files else "walk_circuits_basis_gates.qpy"
    which = "SELECTED" if facts["is_selected_instance"] else "REQUESTED"
    estimate = "not recorded" if facts["estimated_qpu_seconds"] is None else f"{facts['estimated_qpu_seconds']:.1f} s"
    return f"""HOW TO RUN THE {which} INSTANCE ({facts['dataset']}, {facts['k_classes']} classes, R = {facts['R']}, m = {m}, tau = {tau})
==============================================================================================================
This export is data. Nothing in the repository submits to a device; the run below is made by hand.

1. Load the walk circuits: qiskit.qpy.load on {walk}. Circuit t (t = 0 .. {tau - 1}) holds the state preparation
   and 2t applications of W (W^2 per recorded step); it measures the source register (qubits {m} .. {2 * m - 1})
   into classical bits 0 .. {m - 1}. {"walk_circuits_basis_gates.qpy holds the same circuits in rz, sx, x, cx without routing." if "walk_circuits_logical.qpy" in files else "The as-built logical file could not be written (see instance.json); the basis-gate file is the logical circuit."}

2. Transpile for the target backend with ONE shared initial layout, as the simulation did: transpile the deepest
   circuit (t = {tau - 1}) first with generate_preset_pass_manager(target=backend.target, optimization_level=1,
   layout_method="sabre", routing_method="sabre", seed_transpiler=<seed>), read
   first.layout.initial_index_layout(filter_ancillas=True), and transpile every circuit with that initial_layout
   (routing_method="sabre", same level and seed). The seed used in the simulation is in instance.json.

3. Read, for each transpiled circuit, the physical qubits of its measured register in classical-bit order (the
   qubits of its measure instructions). Routing moves the register, so the circuits do not all measure the same
   qubits and readout calibrations are NOT interchangeable between them.

4. Readout calibration, per distinct ordered tuple of measured qubits: the baseline circuit (all qubits |0>,
   measured) and one single-flip circuit per qubit (X on that qubit, then measure all), transpiled at level 0 with
   initial_layout = the tuple. Run them in the same job.

5. Run all circuits at {shots} shots each. The estimated QPU time (walk circuits plus one calibration set per
   tuple, from the snapshot's medians) is {estimate} in the simulation record; it is an
   estimate, not a measurement.

6. Convert counts to supra-node probabilities: bit string -> integer with classical bit 0 least significant (the
   register code in supra_node_map.csv); codes >= R = {facts['R']} are padding: drop their mass and renormalise the
   rest. Mitigate each circuit with the calibration of its own tuple (per-qubit assignment matrices from the
   baseline and flip counts; nonnegative least squares over the observed bit strings; see
   quantum/hardware/mitigation.py). Average the {tau} per-circuit distributions: the time average over
   t = 0 .. {tau - 1}, t = 0 included.

7. Compare with exact_average.csv over the {facts['R']} supra-nodes: Spearman, total variation TV, TV to the uniform
   distribution, and the fidelity ratio 1 - TV / TV_uniform (quantum/hardware/noise_grading.py). Read the result
   against the pure-noise band of this instance at tau = {tau} (EC_noise_grid: pure_noise_fidelity_lo / _hi): the
   output carries signal only when its fidelity ratio is strictly above pure_noise_fidelity_hi, the upper edge of
   the band. The Spearman over the supra-nodes (pure_noise_spearman_lo / _hi) is a secondary column.

8. A failed job, or an output inside the pure-noise band, is the result. Do not resubmit to improve it. Report the
   backend, the job identifier, the date and the calibration timestamp of the device with the numbers.
"""


def _grid_cell(dataset: str, k: int, tau: int) -> dict:
    """The raw FakeMarrakesh cell of instance ``k`` at ``tau`` in the assembled noise grid, or an empty record when the grid has none."""
    frame = read_result(dataset, "EC_noise_grid")
    if frame is None:
        return {}
    rows = frame.filter((pl.col("k_classes") == k) & (pl.col("tau") == tau) & (pl.col("arm") == "fake_marrakesh") & ~pl.col("mitigated")).to_dicts()
    return rows[0] if rows else {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True)
    parser.add_argument("--k", type=int, default=None,
                        help="classes of the instance to export (default: the instance stage 06 selected); the walk length is the selected one")
    args = parser.parse_args(argv)
    config, smoke = resolve_config(DEFAULT_CONFIG)
    dataset = get_dataset(args.dataset)
    stage_output = dataset.stage_dir("hardware")
    record = json.loads((stage_output / "hardware_candidates.json").read_text())
    sel = record.get("selected")
    if sel is None:
        raise SystemExit(f"no instance was selected for {dataset.name}: {record.get('flag')}")
    ctx = DatasetContext.load(dataset, config, stage_key="hardware")
    primary = ctx.primary(dataset.primary)
    selected_k = int(sel["k_classes"])
    wanted = selected_k if args.k is None else int(args.k)
    instance = next((i for i in grid.retained_instances(primary.variant.nodes, primary.variant.edges, config) if i["k"] == wanted), None)
    if instance is None:
        raise SystemExit(f"{dataset.name} has no retained instance with {wanted} classes")
    tau, k = int(sel["tau"]), int(instance["k"])
    is_selected = k == selected_k
    # The QPU estimate and the route of the exported instance: the selection's own for the selected one, else its raw FakeMarrakesh cell of the grid.
    cell = dict(sel) if is_selected else _grid_cell(dataset.name, k, tau)
    G, v = instance["G"], instance["v"]
    exact = step_marginals(G, v, discriminant(G), tau).mean(axis=0)
    stored = shards.read_shard(shards.shard_path(stage_output, k),
                               shards.shard_identity(config, ctx.name, dataset.stage("hardware"), instance, smoke))["vectors"]["exact"][str(tau)]
    if not np.allclose(exact, np.asarray(stored), atol=1e-12, rtol=0.0):
        raise AssertionError("the exact average rebuilt for the export differs from the one in the shard the stage wrote")
    out = results_dir(dataset.name) / ("hardware_export" if is_selected else f"hardware_export_k{k}")
    out.mkdir(parents=True, exist_ok=True)
    struct = StructuredSzegedyCircuit(G, v)
    circuits = struct.build_circuits_per_step(tau, measured_register="source", step_power=2)
    from qiskit import transpile

    basis = transpile(circuits, basis_gates=BASIS_GATES, optimization_level=0, seed_transpiler=config.seed_transpiler(ctx.name, k, "export"))
    errors = {"walk_circuits_basis_gates.qpy": dump_qpy(out / "walk_circuits_basis_gates.qpy", list(basis))}
    errors["walk_circuits_logical.qpy"] = dump_qpy(out / "walk_circuits_logical.qpy", circuits)
    nodes, m, R = instance["nodes"], int(instance["m"]), int(instance["R"])
    pl.DataFrame({"supra_node": list(range(R)), "exact": [float(x) for x in exact]}).write_csv(out / "exact_average.csv")
    layer = nodes["layer_index"].tolist()
    rows = [{"supra_node": i, "register_code": format(i, f"0{m}b"), "node_type": str(nodes["node_type"].iloc[i]),
             "layer_index": None if layer[i] != layer[i] else int(layer[i]), "label": str(nodes["label"].iloc[i]),
             "class_size": None if nodes["weight"].iloc[i] != nodes["weight"].iloc[i] else float(nodes["weight"].iloc[i]),
             "masked_label": int(nodes["BAD_masked"].iloc[i]), "members": nodes["member_borrower_ids"].iloc[i]} for i in range(R)]
    rows += [{"supra_node": i, "register_code": format(i, f"0{m}b"), "node_type": "padding", "layer_index": None, "label": None,
              "class_size": None, "masked_label": None, "members": None} for i in range(R, 2**m)]
    pl.DataFrame(rows, strict=False, infer_schema_length=None).write_csv(out / "supra_node_map.csv")
    files = {name: sha256_path(out / name) for name in ("walk_circuits_basis_gates.qpy", "walk_circuits_logical.qpy", "exact_average.csv", "supra_node_map.csv")
             if (out / name).exists()}
    facts = {"dataset": dataset.name, "k_classes": k, "label": instance["label"], "R": R, "m": m, "logical_qubits": 2 * m, "tau": tau,
             "n_circuits": tau, "shots": int(config.hardware_shots), "measured_register": "source", "step_power": 2,
             "is_selected_instance": is_selected,
             "estimated_qpu_seconds": None if cell.get("estimated_qpu_seconds") is None else float(cell["estimated_qpu_seconds"]),
             "optimization_level_in_simulation": int(config.hardware_optimization_level),
             "transpiler_seed_in_simulation": int(config.seed_transpiler(ctx.name, k)), "backend_snapshot": record["backend_snapshot"],
             "padding_convention": record["padding_convention"], "readout_calibration": record["readout_calibration"],
             "calibration_tuples_in_simulation": (record.get("selected_instance") or {}).get("calibration", {}).get("measured_qubit_tuples"),
             "route": cell.get("route"), "flag": record.get("flag"), "dumps_not_written": {n: e for n, e in errors.items() if e},
             "smoke": bool(smoke)}
    (out / "instance.json").write_text(json.dumps({**facts, "files_sha256": files}, indent=2, default=str))
    (out / "HOWTO.txt").write_text(howto(facts, files))
    for name in (*files, "instance.json", "HOWTO.txt"):
        print(relative_to_repo(out / name))
    for name, error in errors.items():
        if error:
            print(f"NOT WRITTEN {name}: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
