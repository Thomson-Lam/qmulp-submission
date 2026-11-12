"""Structural facts about a network specification, the ceiling they imply, and the job budget they set.

The authoritative source for every graph integer the manuscript states.

A borrower's structural identity is the tuple of category nodes it touches --
one state per layer, plus an "unrecorded" state where the field is blank.
Borrowers sharing a tuple are related by a graph automorphism, so *any*
structural score assigns them exactly equal values. The number of distinct
scores a network can produce is therefore bounded by

    levels <= prod over layers (states_l + 1) - 1

independently of how many borrowers are scored. The subtraction removes the
borrower with nothing recorded anywhere, who is dropped upstream -- but that
drop is judged across *every* layer of the dataset, so in a single-layer
ablation a borrower unrecorded in that one layer can survive and occupy the
unrecorded identity. ``max_distinct_levels_in_population`` is the ceiling that
applies once that is accounted for, and the occupied count is asserted against
it. This module measures the ceiling and the
equivalence classes that realize it (keyed on the attribute tuple and the
*masked* default indicator, as the lumped operator is), so the limit is
reported rather than discovered by a referee. The tie handling of a score is
``evaluation.ties``.

The resource plan
-----------------
The memory and time budget every batch job's ``--mem`` must cite. The exact
Szegedy walk stores an ``N x N`` complex128 state and keeps about three copies
live per step, so the requirement grows as ``N^2``. A job sized by habit rather
than by arithmetic is killed partway through a walk that costs hours. The time
constant is the N = 5,859 measurement scaled as N^2; the memory arithmetic is
exact. Notebook 02 re-times three steps against it.

``write_resource_plan`` emits ``resource_plan.csv``, one row per network
specification, and every sbatch script cites the row it was sized from.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from fingerprint import UNRECORDED, layer_state_ids
from paths import relative_to_repo
from preprocessing.missing_levels import absence_hub_sizes
from quantum.inspired.validation import STATE_COPIES_LIVE, format_bytes, state_bytes


def attribute_tuples(frame: pl.DataFrame, layer_columns: dict[str, list[str]]) -> list[tuple]:
    """The connected-attribute tuple per borrower, one state per layer."""
    per_layer = [layer_state_ids(frame, cols) for cols in layer_columns.values()]
    return list(zip(*(ids.tolist() for ids in per_layer)))


def automorphism_classes(
    frame: pl.DataFrame,
    layer_columns: dict[str, list[str]],
    bad_masked: np.ndarray | None = None,
) -> dict[tuple, list[int]]:
    """Group borrowers by (attribute tuple, masked default indicator).

    The indicator is part of the key because a seed owns a personalized restart
    column and a non-seed does not: the two are not interchangeable states even
    when their attributes match. It is the *masked* indicator (1 / 0 / unknown)
    so that no hidden borrower's true label shapes the classes.
    """
    tuples = attribute_tuples(frame, layer_columns)
    if bad_masked is None:
        bad_masked = np.zeros(len(tuples), dtype=int)
    bad_masked = np.asarray(bad_masked, dtype=int)
    classes: dict[tuple, list[int]] = {}
    for borrower, attributes in enumerate(tuples):
        classes.setdefault((*attributes, int(bad_masked[borrower])), []).append(borrower)
    return classes


def lumped_register_facts(n_classes: int, total_states: int, n_layers: int) -> dict[str, int]:
    """Supra-node count of the lumped operator and the Szegedy register it needs.

    The walk indexes supra-nodes, not classes: ``n_red = L * classes + category
    nodes``, so the register width is ``m = ceil(log2 n_red)`` and the circuit
    uses ``2m`` logical qubits. ``m_classes_only`` is the width the naive
    ``ceil(log2 classes)`` formula gives, recorded so the two are not confused.
    """
    n_red = int(n_layers) * int(n_classes) + int(total_states)
    m = int(np.ceil(np.log2(max(n_red, 2))))
    return {
        "n_classes": int(n_classes),
        "n_supra_nodes_lumped": n_red,
        "register_width_m": m,
        "logical_qubits_2m": 2 * m,
        "m_classes_only": int(np.ceil(np.log2(max(int(n_classes), 2)))),
    }


def degree_distribution(adjacency) -> dict[str, Any]:
    from scipy.sparse import issparse

    degrees = (
        np.asarray(adjacency.sum(axis=1)).ravel()
        if issparse(adjacency)
        else np.asarray(adjacency).sum(axis=1)
    )
    nonzero = degrees[degrees > 0]
    histogram = Counter(np.round(degrees, 6).tolist())
    return {
        "n_nodes": int(degrees.size),
        "n_dangling": int((degrees == 0).sum()),
        "mean_degree": float(nonzero.mean()) if nonzero.size else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "histogram": {str(k): int(v) for k, v in sorted(histogram.items())[:50]},
    }


def bad_assortativity(
    frame: pl.DataFrame, columns: list[str], target: np.ndarray
) -> dict[str, Any]:
    """Per-state default rate in one layer, and the spread across its states.

    A layer whose states all carry the same default rate contributes structure
    but no signal; the spread says which layers are worth having.
    """
    states = layer_state_ids(frame, columns)
    target = np.asarray(target, dtype=int)
    rates: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for state in sorted(set(states.tolist())):
        mask = states == state
        name = "unrecorded" if state == UNRECORDED else columns[state]
        rates[name] = float(target[mask].mean())
        sizes[name] = int(mask.sum())
    connected = [v for k, v in rates.items() if k != "unrecorded"]
    return {
        "state_default_rate": rates,
        "state_size": sizes,
        "spread": float(max(connected) - min(connected)) if connected else 0.0,
        "overall_default_rate": float(target.mean()),
    }


@dataclass
class NetworkFacts:
    payload: dict[str, Any]

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.payload, indent=2, sort_keys=True, default=str))
        return output


def describe_network(
    frame: pl.DataFrame,
    spec,
    layer_columns: dict[str, list[str]],
    *,
    index=None,
    adjacency=None,
    marker_columns: dict[str, list[str]] | None = None,
    target_col: str = "BAD",
    bad_masked: np.ndarray | None = None,
) -> dict[str, Any]:
    """Every structural fact for one network specification.

    ``adjacency`` is optional: the schema-level facts (the level ceiling, the
    automorphism classes) need no graph, so the ceiling can be reported before
    any expensive graph is built.
    """
    n_borrowers = frame.height
    target = frame[target_col].to_numpy().astype(int) if target_col in frame.columns else None
    if bad_masked is None and target is not None:
        bad_masked = target

    classes = automorphism_classes(frame, layer_columns, bad_masked)
    class_sizes = sorted((len(v) for v in classes.values()), reverse=True)
    tuples = attribute_tuples(frame, layer_columns)

    # Does anyone in this population have nothing recorded in any of this
    # network's layers? That is what decides which ceiling applies.
    all_unrecorded_present = tuple(UNRECORDED for _ in layer_columns) in set(tuples)
    population_ceiling = spec.max_distinct_levels() + int(all_unrecorded_present)
    if len(set(tuples)) > population_ceiling:
        raise AssertionError(
            f"{spec.name}: {len(set(tuples))} occupied structural levels exceed the ceiling "
            f"{population_ceiling}. The ceiling is derived from the layer schema, so this is "
            "a broken encoding, not a property of the data."
        )

    facts: dict[str, Any] = {
        "spec": spec.name,
        "description": spec.description,
        "n_layers": spec.n_layers,
        "n_borrowers": n_borrowers,
        "total_states": spec.total_states,
        "states_per_layer": {layer.name: layer.n_states for layer in spec.layers},
        "layer_size": spec.layer_size(n_borrowers),
        "N": spec.n_supra_nodes(n_borrowers),
        # The schema ceiling, prod(states_l + 1) - 1. The subtraction assumes
        # the borrower with nothing recorded anywhere was dropped upstream,
        # which holds for a network spanning every layer of its dataset.
        "max_distinct_levels": spec.max_distinct_levels(),
        # The ceiling that actually applies to this population. A single-layer
        # ablation is judged unconnectable across *all* the dataset's layers,
        # so a borrower with nothing recorded in this network's one layer can
        # survive and occupy the unrecorded identity; where it does, the
        # subtraction above does not apply and the ceiling is one higher.
        "all_unrecorded_present": all_unrecorded_present,
        "max_distinct_levels_in_population": population_ceiling,
        # Occupied attribute tuples, ignoring seed status: the number of
        # distinct values any structural score can take on this population,
        # known before a single walk is run. This is the gate quantity.
        "induced_distinct_levels": len(set(tuples)),
        "n_automorphism_classes": len(classes),
        "lumped_register": lumped_register_facts(len(classes), spec.total_states, spec.n_layers),
        "largest_class_sizes": class_sizes[:10],
        "n_singleton_classes": sum(1 for size in class_sizes if size == 1),
        "layer_edge_coverage": {
            name: float((layer_state_ids(frame, cols) != UNRECORDED).mean())
            for name, cols in layer_columns.items()
        },
        # layer_columns holds genuine categories only, so the markers have to
        # be named explicitly: passing genuine columns alone would report an
        # empty set of absence hubs and look like there were none.
        "absence_hub_sizes": absence_hub_sizes(
            frame,
            *layer_columns.values(),
            markers=[c for cols in (marker_columns or {}).values() for c in cols],
        ),
    }

    if target is not None:
        facts["bad_assortativity"] = {
            name: bad_assortativity(frame, cols, target) for name, cols in layer_columns.items()
        }

    if index is not None:
        facts["n_placeholder_nodes"] = len(index.all_placeholder_nodes)

    if adjacency is not None:
        degrees = degree_distribution(adjacency)
        facts["n_edges"] = int(
            adjacency.nnz if hasattr(adjacency, "nnz") else np.count_nonzero(adjacency)
        )
        facts["degree_distribution"] = degrees
        facts["n_dangling"] = degrees["n_dangling"]
        if index is not None:
            expected = len(index.all_placeholder_nodes)
            facts["dangling_matches_placeholders"] = degrees["n_dangling"] >= expected
    else:
        facts["n_edges"] = None
        facts["n_dangling"] = None

    return facts


def write_network_facts(
    path: str | Path, spec_facts: dict[str, dict[str, Any]], extra: dict[str, Any] | None = None
) -> NetworkFacts:
    payload = {"specs": spec_facts}
    if extra:
        payload.update(extra)
    facts = NetworkFacts(payload)
    facts.write(path)
    payload["path"] = relative_to_repo(path)
    return facts


# ---------------------------------------------------------------------------
# The resource plan
# ---------------------------------------------------------------------------


# One recorded step (two applications of W) per supra-node squared, from a
# measurement on the real masked NET_JOB test operator: 7.91 s per recorded
# step at N = 5,859, single-threaded (author's measurement, September 2026).
# Scaled as N^2 this gives about 31.6 s/step at the two-layer N = 11,722. The
# previous constant (5.4e-8, scaled from an N = 23,484 run) was 4.3x optimistic
# and put notebook 04 under its wall clock. Used only to size a job's --time.
MEASURED_STEP_SECONDS = 7.91
MEASURED_STEP_N = 5_859
MEASURED_STEP_DATE = "2026-09-03"
SECONDS_PER_STEP_PER_N2 = MEASURED_STEP_SECONDS / (MEASURED_STEP_N**2)
RESOURCE_PLAN_STATUS = (
    f"measured: {MEASURED_STEP_SECONDS} s per recorded step on the NET_JOB masked test operator at "
    f"N = {MEASURED_STEP_N:,}, single-threaded, {MEASURED_STEP_DATE}; scaled as N^2 to every row"
)

# Slurm rejects a job whose memory request it cannot satisfy, and a walk that
# fits exactly will still die on transient allocations, so the plan asks for
# headroom over the computed working set.
MEMORY_HEADROOM = 1.25


@dataclass(frozen=True)
class ResourceRow:
    spec_name: str
    n_layers: int
    n_borrowers: int
    total_states: int
    n_supra_nodes: int
    state_bytes: int
    working_bytes: int
    recommended_mem_gb: int
    seconds_per_step: float
    max_distinct_levels: int

    def as_dict(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "n_layers": self.n_layers,
            "n_borrowers": self.n_borrowers,
            "total_states": self.total_states,
            "N": self.n_supra_nodes,
            "state_bytes": self.state_bytes,
            "state_human": format_bytes(self.state_bytes),
            "working_bytes": self.working_bytes,
            "working_human": format_bytes(self.working_bytes),
            "recommended_mem_gb": self.recommended_mem_gb,
            "estimated_seconds_per_recorded_step": round(self.seconds_per_step, 3),
            "max_distinct_levels": self.max_distinct_levels,
            "state_copies_assumed": STATE_COPIES_LIVE,
            "status": RESOURCE_PLAN_STATUS,
        }


def plan_row(spec, n_borrowers: int) -> ResourceRow:
    n = spec.n_supra_nodes(n_borrowers)
    working = state_bytes(n, STATE_COPIES_LIVE)
    return ResourceRow(
        spec_name=spec.name,
        n_layers=spec.n_layers,
        n_borrowers=int(n_borrowers),
        total_states=spec.total_states,
        n_supra_nodes=n,
        state_bytes=state_bytes(n, 1),
        working_bytes=working,
        recommended_mem_gb=max(4, int(working * MEMORY_HEADROOM / 1024**3) + 1),
        seconds_per_step=SECONDS_PER_STEP_PER_N2 * n * n,
        max_distinct_levels=spec.max_distinct_levels(),
    )


def build_resource_plan(specs, n_borrowers: int) -> pl.DataFrame:
    return pl.DataFrame([plan_row(spec, n_borrowers).as_dict() for spec in specs])


def write_resource_plan(path: str | Path, specs, n_borrowers: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    build_resource_plan(specs, n_borrowers).write_csv(output)
    return output


def recommended_mem_gb(spec, n_borrowers: int) -> int:
    """The ``--mem`` value a job running this specification should request."""
    return plan_row(spec, n_borrowers).recommended_mem_gb


def calibrate_step_time(walk, spec, n_borrowers: int, n_steps: int = 3) -> dict:
    """Time ``n_steps`` recorded steps of a built walk and compare with the plan.

    Cheap by construction (single network, three steps) and never a gate: the
    result is printed into the executed notebook so a stale constant is visible
    there rather than only in a blown wall clock.
    """
    from quantum.inspired.walk_cache import time_steps

    planned = plan_row(spec, n_borrowers)
    seconds = time_steps(walk, int(n_steps))
    measured = float(sum(seconds) / len(seconds))
    return {
        "network": spec.name,
        "N": planned.n_supra_nodes,
        "n_steps_timed": int(n_steps),
        "step_seconds": [round(float(x), 3) for x in seconds],
        "measured_seconds_per_step": measured,
        "planned_seconds_per_step": planned.seconds_per_step,
        "measured_over_planned": (
            measured / planned.seconds_per_step if planned.seconds_per_step else float("nan")
        ),
        "plan_constant": RESOURCE_PLAN_STATUS,
    }
