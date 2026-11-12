"""Stage 06: circuit resources, the statevector rung, the noise grid, the selected instance and Table 6.

Per dataset: the lumped primary operator and its nested retained instances
(``quantum.hardware.noise_grid``); ``EC_circuit_resources``; the statevector
rung on the full lumped operator (``EC_statevector``); for every instance, recorded
step and arm the simulated cells, raw and readout-mitigated per circuit
(``EC_noise_grid``), each read against the pure-noise band of its own instance;
the selected instance (``hardware_candidates.json``, with the ISA circuits as QPY)
and, for it, ``T6_hardware`` and the per-supra-node vectors of every arm
(``EC_hardware_scores``). Every arm is an Aer simulation; nothing is submitted to
a device, and the hand-run submission of the selected instance starts from
``scripts/export_hardware_instance.py``.

An array task (``scripts/warm.py --stage 06``) simulates one instance into a shard
(``quantum.hardware.shards``): identity-checked, written after every recorded step.
The stage assembles whatever shards exist; a cell no complete shard covers is
recorded as ``not simulated: resource budget`` with the reason, so the table has a
row for every (instance, arm, step) even when a task ran out of allocation.

Table 6
-------
One retained instance per dataset, every simulated arm against the exact lumped
walk. Rows per dataset: ``exact``, ``noiseless_shots``, ``aer_noise_model``,
``fake_marrakesh``. Columns: Spearman against the exact time average at the
supra-node level, the Spearman over the masked classes (or ``single
group``), TV, TV to uniform, the fidelity ratio, the padding mass, the maximal
depth, the summed two-qubit count, the shots and the estimated QPU seconds.
A header file carries the instance facts (classes, borrowers, ``R``, ``m``,
``tau``, the FakeMarrakesh snapshot and its calibration date), the first instance
able to rank test borrowers and the two noise entry points. Nothing here
says more than the numbers do; no row describes an execution on hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from config import ExperimentConfig
from datasets import DatasetConfig, get_dataset
from evaluation.context import DatasetContext
from evaluation.provenance import RunTimer, write_manifest, write_result
from fingerprint import resolve_config
from quantum.hardware import noise_grid as grid, shards
from quantum.hardware.execution import get_fake_backend, ibm_device_model
from quantum.hardware.noise_grading import (
    ARM_DEFINITIONS,
    NOISE_GRID_COLUMNS,
    READOUT_CALIBRATION_NOTE,
    RESOURCES_COLUMNS,
    STATEVECTOR_COLUMNS,
    annotate_signal,
    first_rankable_instance,
    is_simulated,
    noise_entry_points,
)
from quantum.hardware.structured_circuit import StructuredSzegedyCircuit


STAGE_KEY = "hardware"
STATEVECTOR_BUDGET_S = 1800.0
NOISY_ARMS = ("aer_noise_model", "fake_marrakesh")


def backend_snapshot(backend, model: dict[str, Any]) -> dict[str, Any]:
    """What the fake backend is, as recorded: name, version, calibration date, runtime version."""
    from importlib.metadata import version

    props = None
    try:
        props = backend.properties()
    except Exception:
        props = None
    stamp = getattr(props, "last_update_date", None) if props is not None else None
    # ``calibration_date`` is the calendar day of the snapshot (what the manuscript prints as hw_date);
    # ``calibration_timestamp`` keeps the full stamp with its time zone.
    day = None if stamp is None else (stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp)[:10])
    return {"backend_name": model["backend_name"], "backend_version": getattr(backend, "backend_version", None),
            "calibration_date": day, "calibration_timestamp": None if stamp is None else str(stamp),
            "qiskit_ibm_runtime": version("qiskit-ibm-runtime"), "native_two_qubit_gates": model["native_two_qubit_gates"],
            "basis_gates": model["basis_gates"], "num_qubits": model["num_qubits"]}


def route_of(ctx: DatasetContext, instance: dict[str, Any], primary) -> dict[str, Any]:
    """The route record of an instance: the primary operator's own for the full instance, a rebuilt member-level check otherwise."""
    if instance["label"] == "full":
        return grid.primary_route_record(primary.routed)
    if not instance["simulated"]:
        return grid.not_routed_record("not simulated (wider than the register cap), so not routed")
    return grid.retained_route_record(instance, ctx.partitions, ctx.specs[ctx.dataset.primary], ctx.config, ctx.hidden_ids)


def instance_rows(ctx: DatasetContext, instance: dict[str, Any], primary, backend, model, output: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Simulate one retained instance and write its shard (partial after each step, complete at the end)."""
    config = ctx.config
    path = shards.shard_path(output, instance["k"])
    route = route_of(ctx, instance, primary)
    if not instance["simulated"]:
        record = {"k": instance["k"], "label": instance["label"], "status": grid.NOT_SIMULATED, "route": route, "layout": None,
                  "rows": grid.not_simulated_rows(ctx.name, instance, config, route)}
        shards.write_shard(path, identity, record, complete=True)
        return record
    struct = StructuredSzegedyCircuit(instance["G"], instance["v"])
    layout = grid.transpile_shared_layout(struct, int(max(config.hardware_steps)), backend, config.seed_transpiler(ctx.name, instance["k"]),
                                          optimization_level=int(config.hardware_optimization_level))
    # Flushed so a task that later runs out of time still leaves the circuit size in its log.
    print(f"instance {instance['label']}: m={instance['m']}, touched physical qubits={layout['active_qubits_max']}, "
          f"two-qubit gates at tau={max(config.hardware_steps)}: {layout['costs'][-1]['two_qubit_gates']}, "
          f"transpile {layout['transpile_seconds']:.0f} s, route {route['route']}", flush=True)
    from qiskit import qpy

    qpy_path = output / f"isa_circuits_{instance['label']}.qpy"
    with open(qpy_path, "wb") as handle:
        qpy.dump(list(layout["isa"]), handle)
    record = grid.simulate_instance(ctx.name, instance, layout, config, backend, model, route=route,
                                    checkpoint=lambda partial: shards.write_shard(path, identity, {**partial, "qpy": qpy_path.name}, complete=False))
    record["qpy"] = qpy_path.name
    shards.write_shard(path, identity, record, complete=True)
    return record


def rows_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """One frame from heterogeneous cell rows: every key of any row becomes a column (the defined ones first)."""
    columns = list(dict.fromkeys([*NOISE_GRID_COLUMNS, *(key for row in rows for key in row)]))
    return pl.DataFrame([{key: row.get(key) for key in columns} for row in rows], strict=False, infer_schema_length=None)


def score_rows(dataset: str, instance: dict[str, Any], vectors: dict[str, Any], tau: int, arms) -> list[dict[str, Any]]:
    """Per supra-node vectors of the selected instance at ``tau``: the exact average and every arm, raw and mitigated."""
    nodes = instance["nodes"]
    layer = nodes["layer_index"].tolist()
    exact = vectors["exact"][str(tau)]
    out = []
    for i in range(int(instance["R"])):
        row: dict[str, Any] = {"dataset": dataset, "supra_node": i, "node_type": str(nodes["node_type"].iloc[i]),
                               "layer_index": None if layer[i] != layer[i] else int(layer[i]), "label": str(nodes["label"].iloc[i]),
                               "exact": float(exact[i])}
        for arm in arms:
            raw = (vectors.get(arm) or {}).get("raw", {}).get(str(tau))
            row[arm] = None if raw is None else float(raw[i])
            if arm != "noiseless_shots":
                mitigated = (vectors.get(arm) or {}).get("mitigated", {}).get(str(tau))
                row[f"{arm}_mitigated"] = None if mitigated is None else float(mitigated[i])
        out.append(row)
    return out


def run(dataset_name: str, config: ExperimentConfig | None = None, *, instances: list[int] | None = None, assemble: bool = True) -> dict[str, Any]:
    """Run the circuit tier of one dataset; ``instances`` restricts the retained sizes (an array task runs one)."""
    from config import DEFAULT_CONFIG

    config, smoke = resolve_config(config or DEFAULT_CONFIG)
    dataset = get_dataset(dataset_name)
    timer = RunTimer()
    ctx = DatasetContext.load(dataset, config, stage_key=STAGE_KEY)
    output, source = ctx.output, dataset.stage(STAGE_KEY)
    primary = ctx.primary(dataset.primary)
    backend = get_fake_backend(config.ibm_backend)
    model = ibm_device_model(config.ibm_backend)
    snapshot = backend_snapshot(backend, model)
    retained = grid.retained_instances(primary.variant.nodes, primary.variant.edges, config)
    identity = {i["k"]: shards.shard_identity(config, ctx.name, source, i, smoke) for i in retained}
    compute_missing = instances is not None or smoke
    for instance in (i for i in retained if instances is None or i["k"] in set(instances)):
        path = shards.shard_path(output, instance["k"])
        if path.exists():
            record = shards.read_shard(path, identity[instance["k"]])      # a stale identity raises here
            if record.get("complete") or not compute_missing:
                continue
            print(f"instance {instance['label']}: the shard is partial (its task ended early); simulating the instance again", flush=True)
        elif not compute_missing:
            continue
        instance_rows(ctx, instance, primary, backend, model, output, identity[instance["k"]])
    if not assemble:
        return {"instances": [i["k"] for i in retained if instances is None or i["k"] in set(instances)],
                "artifacts": [shards.shard_path(output, i["k"]) for i in retained if instances is None or i["k"] in set(instances)]}
    return assemble_stage(ctx, config, smoke, source, primary, retained, identity, snapshot, model, timer)


def assemble_stage(ctx: DatasetContext, config: ExperimentConfig, smoke: bool, source: str, primary, retained, identity, snapshot, model,
                   timer: RunTimer) -> dict[str, Any]:
    """The tables, the selection and the manifest from the shards that exist; every missing cell is recorded, not omitted."""
    output, dataset = ctx.output, ctx.dataset
    rows: list[dict[str, Any]] = []
    records: dict[int, dict[str, Any] | None] = {}
    routes: dict[int, dict[str, Any]] = {}
    incomplete: list[int] = []
    filled_total = 0
    for instance in retained:
        k = instance["k"]
        path = shards.shard_path(output, k)
        record = shards.read_shard(path, identity[k]) if path.exists() else None
        if record is None or not record.get("complete"):
            incomplete.append(k)
        cell_rows, filled = shards.cells_of(record, instance, ctx.name, config)
        rows += cell_rows
        filled_total += filled
        records[k] = record
        routes[k] = (record or {}).get("route") or (grid.primary_route_record(primary.routed) if instance["label"] == "full" else None)
    rows = annotate_signal(rows)        # signal_above_band from the fidelity ratio and its band edge; shards need no re-simulation
    if incomplete:
        print(f"WARNING: {len(incomplete)} instance(s) have no complete shard ({incomplete}); {filled_total} cell(s) recorded as "
              "'not simulated: resource budget'", flush=True)
    resources = [grid.resources_row(ctx.name, primary.variant.nodes, primary.routed.G, primary.routed.v, config, label="full",
                                    borrowers=ctx.n_borrowers, route=routes[retained[0]["k"]])]
    resources += [grid.resources_row(ctx.name, i["nodes"], i["G"], i["v"], config, label=i["label"], borrowers=i["borrowers"],
                                     route=routes[i["k"]]) for i in retained[1:]]
    res_path = write_result(ctx.name, "EC_circuit_resources", pl.DataFrame(resources, strict=False, infer_schema_length=None),
                            stage=source, columns=RESOURCES_COLUMNS)
    sv = grid.statevector_rung(primary.routed.G, primary.routed.v, config, budget_seconds=STATEVECTOR_BUDGET_S if not smoke else 60.0, dataset=ctx.name)
    sv_path = write_result(ctx.name, "EC_statevector",
                           pl.DataFrame([{k: v for k, v in sv.items() if k != "max_abs_error_per_step"}], strict=False, infer_schema_length=None),
                           stage=source, columns=STATEVECTOR_COLUMNS)
    grid_path = write_result(ctx.name, "EC_noise_grid", rows_frame(rows), stage=source, columns=NOISE_GRID_COLUMNS)
    selection = grid.select_candidate(rows, config)
    selected_k = selection["selected"]["k_classes"] if selection["selected"] is not None else None
    min_tau = int(config.hardware_min_tau)
    entry = {arm: noise_entry_points(rows, arm=arm, min_tau=min_tau, selected_k=selected_k) for arm in NOISY_ARMS if arm in config.hardware_noise_arms}
    rankable = first_rankable_instance(rows, min_tau=min_tau)
    outputs: list[Path] = [res_path, sv_path, grid_path]
    tables: dict[str, Any] = {}
    header: dict[str, Any] = {}
    chosen_record: dict[str, Any] = {}
    if selection["selected"] is not None:
        sel = selection["selected"]
        instance = next(i for i in retained if i["k"] == sel["k_classes"])
        chosen = records[instance["k"]]
        chosen_record = {"k": instance["k"], "label": instance["label"], "tau": sel["tau"], "route": routes[instance["k"]],
                         "calibration": chosen["calibration"], "qpy": chosen.get("qpy")}
        fake = entry.get("fake_marrakesh", {})
        header = {"dataset": ctx.name, "hw_k": sel["k_classes"], "hw_borrowers": sel["borrowers"], "hw_R": sel["R"], "hw_m": sel["m"],
                  "hw_tau": sel["tau"], "hw_depth": sel["depth_max"], "hw_2q": sel["two_qubit_sum"], "hw_survival": sel["predicted_survival_min"],
                  "hw_qpu_s": sel["estimated_qpu_seconds"], "hw_tv_uniform": sel["tv_uniform"],
                  # The snapshot every simulated arm reads and its calibration date; nothing ran on a device.
                  "hw_backend": snapshot["backend_name"], "hw_date": snapshot["calibration_date"],
                  "hw_route": sel.get("route"), "hw_calibration_sets": sel.get("n_calibration_sets"),
                  "n_masked_groups": sel["n_masked_groups"], "flag": selection["flag"], **rankable,
                  "noise_k_band": fake.get("noise_k_band", "none"), "noise_tau_band": fake.get("noise_tau_band", "none"),
                  **{f"{key}_{arm}": value for arm, e in entry.items() for key, value in e.items() if key in ("noise_k_band", "noise_tau_band")}}
        scores = score_rows(ctx.name, instance, chosen["vectors"], int(sel["tau"]), config.hardware_noise_arms)
        tables = write_hardware_tables(config, dataset, table6_rows(ctx.name, sel, rows), header, scores, inputs=ctx.inputs, source_stage=source)
        outputs += list(tables.values())
    candidates_path = output / "hardware_candidates.json"
    candidates_path.write_text(json.dumps(
        {**selection, "dataset": ctx.name, "backend_snapshot": snapshot, "growth_rule": grid.GROWTH_RULE, "padding_convention": grid.PADDING_CONVENTION,
         "optimization_level": int(config.hardware_optimization_level), "arms": ARM_DEFINITIONS, "readout_calibration": READOUT_CALIBRATION_NOTE,
         "noise_entry_points": entry, "rankable_instance": rankable, "selected_instance": chosen_record, "incomplete_shards": incomplete, "n_cells_filled": filled_total,
         "shard_schema_version": shards.SHARD_SCHEMA_VERSION}, indent=2, default=str))
    outputs.append(candidates_path)
    manifest = write_manifest(output, arm_name=source, config=config, inputs=ctx.inputs, outputs=outputs,
                              seeds={"master_seed": config.master_seed, "transpiler": "config.seed_transpiler(dataset, k)",
                                     "aer": "config.seed_aer(dataset, k, tau, arm, t)", "noise_band": "config.seed_noise_band(dataset, k)"},
                              timer=timer, execution_mode="fake_backend", dataset=dataset,
                              extra={"dataset": ctx.name, "backend_snapshot": snapshot, "device_model": model, "selection": selection,
                                     "noise_entry_points": entry, "rankable_instance": rankable, "arms": ARM_DEFINITIONS, "readout_calibration": READOUT_CALIBRATION_NOTE,
                                     "statevector": {k: v for k, v in sv.items() if k != "max_abs_error_per_step"},
                                     "growth_rule": grid.GROWTH_RULE, "padding_convention": grid.PADDING_CONVENTION, "smoke": smoke,
                                     "shard_schema_version": shards.SHARD_SCHEMA_VERSION, "incomplete_shards": incomplete,
                                     "n_cells_filled": filled_total,
                                     "instances": [{**{k: i[k] for k in ("k", "label", "R", "m", "borrowers", "n_masked_groups", "simulated")},
                                                    "route": (routes[i["k"]] or {}).get("route")} for i in retained]})
    return {"resources": resources, "statevector": sv, "grid_rows": rows, "selection": selection, "header": header, "tables": tables,
            "artifacts": [*outputs, manifest]}


# ---------------------------------------------------------------------------
# Table 6
# ---------------------------------------------------------------------------


ARM_KEYS = {"noiseless_shots": "noiseless", "aer_noise_model": "aernoise", "fake_marrakesh": "fake", "exact": "exact"}
SINGLE_GROUP = "single group"


def table6_rows(dataset: str, selected: dict[str, Any], grid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows of one dataset from the noise grid at the selected instance and ``tau``."""
    k, tau = selected["k_classes"], selected["tau"]
    cells = [r for r in grid_rows if r.get("k_classes") == k and r.get("tau") == tau and not r.get("mitigated") and is_simulated(r)]
    rankable = selected["n_masked_groups"] >= 2
    rows = [{"dataset": dataset, "arm": "exact", "arm_key": "exact", "rho": 1.0, "rho_masked": 1.0 if rankable else None,
             "rho_masked_note": "" if rankable else SINGLE_GROUP,
             "tv": 0.0, "tv_uniform": selected["tv_uniform"], "fr": 1.0, "padding": 0.0, "depth_max": None, "two_qubit_sum": None,
             "shots": None, "qpu_seconds": None, "qpu_kind": "none"}]
    for cell in sorted(cells, key=lambda r: list(ARM_KEYS).index(r["arm"])):
        rows.append({"dataset": dataset, "arm": cell["arm"], "arm_key": ARM_KEYS[cell["arm"]], "rho": cell["spearman_supra"],
                     "rho_masked": cell.get("spearman_masked_classes") if rankable else None,
                     "rho_masked_note": "" if rankable else SINGLE_GROUP, "tv": cell["tv"],
                     "tv_uniform": cell["tv_uniform"], "fr": cell["fidelity_ratio"], "padding": cell["padding_mass"],
                     "depth_max": cell["depth_max"], "two_qubit_sum": cell["two_qubit_sum"], "shots": cell["shots"],
                     "qpu_seconds": cell["estimated_qpu_seconds"], "qpu_kind": "estimated"})
    return rows


HEADER_COLUMNS = {
    "dataset": "dataset name", "hw_k": "classes of the retained instance", "hw_borrowers": "borrowers the instance covers",
    "hw_R": "supra-nodes of the instance", "hw_m": "register width", "hw_tau": "recorded steps of the selected cell",
    "hw_depth": "maximal transpiled depth over the per-time circuits", "hw_2q": "summed two-qubit gates over the per-time circuits",
    "hw_survival": "conservative predicted survival probability, min over the circuits",
    "hw_qpu_s": "estimated QPU seconds of the selected cell (shots x (depth x gate time + measurement + repetition delay))",
    "hw_tv_uniform": "total variation between the uniform distribution and the exact time average",
    "hw_backend": "the calibration snapshot every simulated arm reads (a fake backend; nothing ran on hardware)",
    "hw_date": "calibration date of that snapshot",
    "n_masked_groups": "distinct masked classes the instance retains", "flag": "selection flag (empty, or 'cannot rank test borrowers')",
    "hw_route": "discriminant the router selects for the retained operator against its member-level operator (D_bar or H)",
    "hw_calibration_sets": "distinct measured-qubit tuples among the selected cell's circuits: readout calibration sets used (per circuit)",
    "hw_rankable_k": "classes of the first instance able to rank test borrowers: the smallest k that retains two or more masked classes "
                     "(null when no instance does)",
    "hw_rankable_m": "register width of that instance", "hw_rankable_borrowers": "borrowers that instance covers",
    "hw_rankable_fr": "largest raw fake_marrakesh fidelity ratio of that instance over its simulated cells at tau >= hardware_min_tau "
                      "(null when none was simulated)",
    "hw_rankable_signal": "whether that cell carries signal: its fidelity ratio is strictly above the upper edge of its own pure-noise band",
    "noise_k_band": "entry point over instances: the smallest number of classes k such that neither it nor any larger evaluated k carries "
                    "signal in the raw fake_marrakesh output; each k is judged at its longest simulated walk (the largest tau >= "
                    "hardware_min_tau) and signal means a fidelity ratio strictly above the upper edge of its own pure-noise band; "
                    "'none' when no evaluated k satisfies it",
    "noise_tau_band": "entry point over steps, for the selected instance: the first tau >= hardware_min_tau at which its raw fake_marrakesh "
                      "cell carries no signal, or '> <tau>' (its largest simulated tau) when every one does; 'not evaluated' without a "
                      "simulated cell",
    "noise_k_band_aer_noise_model": "the same entry point for the aer_noise_model arm",
    "noise_tau_band_aer_noise_model": "the same first tau for the aer_noise_model arm",
    "noise_k_band_fake_marrakesh": "the same as noise_k_band, named by arm",
    "noise_tau_band_fake_marrakesh": "the same as noise_tau_band, named by arm",
}


def write_hardware_tables(config: ExperimentConfig, dataset: DatasetConfig, rows: list[dict[str, Any]], header: dict[str, Any],
                          scores: list[dict[str, Any]], *, inputs, source_stage: str) -> dict[str, Path]:
    """``T6_hardware``, ``T6_hardware_header`` and ``EC_hardware_scores`` under the dataset's scope."""
    scope = dataset.name
    main = write_result(
        scope, "T6_hardware", pl.DataFrame(rows, strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "arm": "exact, noiseless_shots, aer_noise_model or fake_marrakesh",
                 "arm_key": "short arm key used by the number map", "rho": "Spearman against the exact time average over the R supra-nodes",
                 "rho_masked": "Spearman over the class probabilities of the masked classes (null when fewer than two)",
                 "rho_masked_note": f"'{SINGLE_GROUP}' when the masked-class Spearman is undefined",
                 "tv": "total variation to the exact time average", "tv_uniform": "total variation between uniform and exact",
                 "fr": "fidelity ratio 1 - TV / TV_uniform", "padding": "mean measured mass on padding codes",
                 "depth_max": "maximal transpiled depth", "two_qubit_sum": "summed two-qubit gates", "shots": "shots per circuit",
                 "qpu_seconds": "estimated QPU seconds", "qpu_kind": "'estimated' for every simulated arm; 'none' for the exact row"},
    )
    header_path = write_result(scope, "T6_hardware_header", pl.DataFrame([header], strict=False, infer_schema_length=None),
                               stage=source_stage, columns=HEADER_COLUMNS)
    score_columns = {"dataset": "dataset name", "supra_node": "supra-node index of the retained instance (the row of its lumped node table)",
                     "node_type": "borrower_copy or category", "layer_index": "layer of a borrower copy (null for a category node)",
                     "label": "the node's label in the lumped graph", "exact": "exact time-averaged source probability at the selected tau"}
    for arm in ("noiseless_shots", "aer_noise_model", "fake_marrakesh"):
        score_columns[arm] = f"time-averaged measured probability under the {arm} arm (raw); null when the cell was not simulated"
        if arm != "noiseless_shots":
            score_columns[f"{arm}_mitigated"] = f"the same after per-circuit readout mitigation ({arm})"
    scores_table = write_result(scope, "EC_hardware_scores", pl.DataFrame(scores, strict=False, infer_schema_length=None),
                                stage=source_stage, columns=score_columns)
    return {"table": main, "header": header_path, "scores": scores_table}
