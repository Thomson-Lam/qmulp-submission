"""Stage 02's named steps: what each network is, before any walk runs.

Builds every network of the dataset's ladder and records node counts, edge
counts, degree distributions, the level ceiling, and the lumped equivalence
classes keyed on ``(<layer states>, BAD_masked)`` with the register width they
imply. Then the exact-lumpability check on the primary network: the
stationary distribution of the lumped operator, unfolded to borrowers, must
equal the unlumped stationary distribution -- zero reduction error, measured
rather than asserted.

Which networks, which frame and which directory come from the
:class:`datasets.DatasetConfig` every function takes; nothing here names a dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from classical.network_diagnostics import (
    RESOURCE_PLAN_STATUS,
    calibrate_step_time,
    describe_network,
    write_network_facts,
    write_resource_plan,
)
from config import ExperimentConfig, NetworkSpec
from datasets import DatasetConfig
from evaluation.masked_eval import Partitions
from evaluation.provenance import write_result
from fingerprint import (
    TEST_OPERATOR_TAG,
    build_ranker,
    layer_columns_for,
    layer_marker_columns_for,
    solve_classical,
)
from paths import masked_frame_path
from quantum.lumped_operator import build_lumped_matrices, class_members, unfold_class_scores
from quantum.lumping import full_lumped_subgraph, lumped_facts

LUMPABILITY_TOL = 1e-8
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


def stationary_distribution(G: np.ndarray) -> np.ndarray:
    """Leading eigenvector of a small dense column-stochastic G, unit sum."""
    values, vectors = np.linalg.eig(G)
    leading = int(np.argmin(np.abs(values - 1.0)))
    pi = np.real(vectors[:, leading])
    pi = pi / pi.sum()
    residual = float(np.linalg.norm(G @ pi - pi, 1))
    if residual > 1e-10 or pi.min() < -1e-10:
        raise AssertionError(
            f"lumped stationary vector invalid: residual={residual:.3e}, min={pi.min():.3e}"
        )
    return np.maximum(pi, 0.0) / np.maximum(pi, 0.0).sum()


def lumped_key_label(dataset: DatasetConfig) -> str:
    """The joint key the classes are built on, spelled out for records and captions."""
    return "(" + ", ".join(f"{name} state" for name in dataset.layer_order) + ", BAD_masked)"


def describe_one_network(
    partitions: Partitions,
    spec: NetworkSpec,
    config: ExperimentConfig,
    masked_levels: np.ndarray,
    dataset: DatasetConfig,
) -> dict[str, Any]:
    """Structural facts, the adjacency, and the lumped classes of one network."""
    frame = partitions.frame
    columns = layer_columns_for(frame, spec)
    markers = layer_marker_columns_for(frame, spec)
    ranker = build_ranker(dataset.processed_path(), spec, config)
    adjacency = ranker.buildAdjMatrix()
    facts = describe_network(
        frame,
        spec,
        columns,
        index=ranker.index,
        adjacency=adjacency,
        marker_columns=markers,
        bad_masked=masked_levels,
    )
    nodes, edges = full_lumped_subgraph(partitions, spec, config, partitions.test_idx)
    lumped = lumped_facts(nodes, spec)
    facts["lumped"] = lumped
    facts["lumped_key"] = f"{lumped_key_label(dataset)} restricted to the network's layers"
    return {
        "facts": facts,
        "ranker": ranker,
        "adjacency": adjacency,
        "nodes": nodes,
        "edges": edges,
        "lumped": lumped,
    }


def network_summary_row(
    spec: NetworkSpec, facts: dict[str, Any], n_borrowers: int
) -> dict[str, Any]:
    lumped = facts["lumped"]
    return {
        "network": spec.name,
        "n_layers": spec.n_layers,
        "n_borrowers": n_borrowers,
        "category_nodes": spec.total_states,
        "N": facts["N"],
        "n_edges": facts["n_edges"],
        "n_dangling": facts["n_dangling"],
        "mean_degree": facts["degree_distribution"]["mean_degree"],
        "max_degree": facts["degree_distribution"]["max_degree"],
        "max_distinct_levels": facts["max_distinct_levels"],
        "induced_distinct_levels": facts["induced_distinct_levels"],
        "lumped_classes": lumped["n_classes"],
        "lumped_seed_classes": lumped["n_seed_classes"],
        "lumped_supra_nodes": lumped["n_supra_nodes_lumped"],
        "register_width_m": lumped["register_width_m"],
        "logical_qubits_2m": lumped["logical_qubits_2m"],
        "statevector_kib": lumped["statevector_bytes"] / 1024,
        "borrowers_covered": lumped["borrowers_covered"],
    }


def write_network_files(
    output: Path, spec_name: str, facts: dict[str, Any], nodes, edges
) -> list[Path]:
    """Degree distribution and the lumped class graph, per network."""
    facts["degree_distribution_file"] = f"degree_distribution_{spec_name}.csv"
    degree_path = output / f"degree_distribution_{spec_name}.csv"
    pl.DataFrame(
        [
            {"degree": float(k), "n_nodes": int(v)}
            for k, v in facts["degree_distribution"]["histogram"].items()
        ]
    ).write_csv(degree_path)
    nodes_path = output / f"lumped_nodes_{spec_name}.csv"
    edges_path = output / f"lumped_edges_{spec_name}.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    return [degree_path, nodes_path, edges_path]


def write_edge_coverage(output: Path, built: dict[str, dict[str, Any]], ladder) -> Path:
    """Per network and layer, the share of borrowers carrying a category in that layer.

    A level fitted only on the development partition cannot reach a test
    borrower that never showed it, and such a borrower earns no edge in that
    layer: the gap is reported rather than smoothed over.
    """
    path = output / "layer_edge_coverage.csv"
    pl.DataFrame(
        [
            {"network": name, "layer": layer, "edge_coverage": value}
            for name in ladder
            for layer, value in built[name]["facts"]["layer_edge_coverage"].items()
        ]
    ).write_csv(path)
    return path


def lumpability_check(
    partitions: Partitions, spec: NetworkSpec, config: ExperimentConfig, nodes, edges
) -> dict[str, Any]:
    """Unlumped vs lumped stationary distribution on the test-operator masking.

    Raises if the unfolded lumped distribution differs from the full one by
    more than floating-point noise: the lumping is exact or it is nothing.
    """
    masked = partitions.mask_labels(partitions.test_idx)
    path = masked_frame_path(spec.name, TEST_OPERATOR_TAG)
    masked.write_parquet(path)
    bundle = solve_classical(path, spec, config)
    omega_full = bundle.omega

    G_red, v_red = build_lumped_matrices(
        nodes, edges, alpha=config.resolved_alpha(), influence_type=config.influence_type
    )
    pi_red = stationary_distribution(G_red)
    borrower_ids, unfolded = unfold_class_scores(nodes, pi_red)
    if borrower_ids.size != omega_full.size or not np.array_equal(
        borrower_ids, np.arange(omega_full.size)
    ):
        raise AssertionError("the lumped classes do not cover every borrower exactly once")
    worst_unfold = float(np.max(np.abs(unfolded - omega_full)))

    worst_class = 0.0
    for position, members in class_members(nodes).items():
        layer = int(nodes.iloc[position]["layer_index"])
        start = bundle.ranker.index.offset(layer)
        expected = float(bundle.pi[[start + m for m in members]].sum())
        worst_class = max(worst_class, abs(float(pi_red[position]) - expected))

    passed = worst_unfold < LUMPABILITY_TOL and worst_class < LUMPABILITY_TOL
    if not passed:
        raise AssertionError(
            f"exact lumpability failed on {spec.name}: max |unfolded - full| = {worst_unfold:.3e}, "
            f"max class error = {worst_class:.3e} (tolerance {LUMPABILITY_TOL})"
        )
    return {
        "spec": spec.name,
        "masking": "test operator (test labels hidden)",
        "max_abs_error_unfolded_vs_full": worst_unfold,
        "max_abs_error_class_vs_member_sum": worst_class,
        "tolerance": LUMPABILITY_TOL,
        "passed": passed,
        "n_supra_nodes_full": int(bundle.ranker.N),
        "n_supra_nodes_lumped": int(len(nodes)),
    }


def write_network_summary_table(
    config: ExperimentConfig, dataset: DatasetConfig, summary: pl.DataFrame, inputs: list[Path]
) -> Path:
    """``network_summary`` under the dataset's scope: one row per network of the ladder."""
    return write_result(
        dataset.name, "network_summary", summary, stage=dataset.stage("network"),
        columns={"network": "network name", "n_layers": "layers L", "n_borrowers": "individuals",
                 "category_nodes": "category nodes over the layers", "N": "supra-nodes of the full operator (slots included)",
                 "n_edges": "edges", "n_dangling": "dangling columns", "mean_degree": "mean degree", "max_degree": "maximal degree",
                 "max_distinct_levels": "level ceiling: distinct category states the layers can induce",
                 "induced_distinct_levels": "distinct category states occupied", "lumped_classes": "lumping classes",
                 "lumped_seed_classes": "reference classes", "lumped_supra_nodes": "supra-nodes of the lumped operator",
                 "register_width_m": "register width m", "logical_qubits_2m": "logical qubits 2 m",
                 "statevector_kib": "statevector size in KiB", "borrowers_covered": "individuals the lumped operator covers"},
    )


def write_network_artifacts(
    output: Path,
    spec_facts: dict[str, Any],
    check: dict[str, Any],
    summary: pl.DataFrame,
    specs,
    n_borrowers: int,
) -> list[Path]:
    """network_facts.json, network_summary.csv, lumpability_check.json, resource_plan.csv."""
    (output / "lumpability_check.json").write_text(json.dumps(check, indent=2, sort_keys=True))
    write_network_facts(
        output / "network_facts.json", spec_facts, extra={"lumpability_check": check}
    )
    summary.write_csv(output / "network_summary.csv")
    write_resource_plan(output / "resource_plan.csv", specs, n_borrowers)
    return [
        output / "network_facts.json",
        output / "network_summary.csv",
        output / "lumpability_check.json",
        output / "resource_plan.csv",
    ]


def calibrate_step_constant(
    config: ExperimentConfig, dataset: DatasetConfig, specs, n_steps: int = 3
) -> dict[str, Any]:
    """Measured-versus-planned seconds per recorded step on the smallest network.

    Builds the quantum walker on the processed frame (timing does not depend on
    the mask), times three recorded steps and reports the ratio to the resource
    plan. Optional and cheap (about 25 s); a failure is returned as data so it
    never gates the notebook.
    """
    from quantum.inspired.stage import build_walker

    processed = dataset.processed_path()
    frame_height = pl.read_parquet(processed).height
    spec = min(specs, key=lambda s: s.n_supra_nodes(frame_height))
    try:
        walker = build_walker(processed, spec, config)
        walker.buildSzegedyWalk()
        return {
            "status": "measured",
            **calibrate_step_time(walker.szegedy_walk, spec, frame_height, n_steps),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "network": spec.name,
            "error": f"{type(exc).__name__}: {exc}",
            "plan_constant": RESOURCE_PLAN_STATUS,
        }
