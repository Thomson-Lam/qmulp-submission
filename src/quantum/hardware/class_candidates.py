"""Connected subsets of equivalence classes: the candidates of the hardware sweep.

The reduction for hardware is over **classes**, not borrowers: every candidate
is a connected set of lumped classes, grown breadth-first from the seed class
carrying the most restart mass and preferring, at each step, the class with the
most edges into what is already retained. Density matters more than count: a
nearly edgeless subgraph gives the walk nothing to interfere across.

Every candidate's operator is **rebuilt** from its own ``(nodes, edges)`` by
``quantum.lumped_operator.build_lumped_matrices`` -- transitions renormalised,
dangling columns patched with the retained ``v``, restart columns rebuilt -- and
asserted column-stochastic. A principal submatrix of the full ``G`` is not the
reduced operator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantum.lumped_operator import build_lumped_matrices, personalization_vector


def class_keys(nodes: pd.DataFrame) -> list[str]:
    """One key per class, in layer-0 copy order."""
    copies = nodes[(nodes["node_type"] == "borrower_copy") & (nodes["layer_index"] == 0)]
    return [str(k) for k in copies["member_borrower_ids"].tolist()]


def class_adjacency(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Classes are adjacent when a membership edge joins them to the same category node."""
    key_of_node = dict(zip(nodes["node_id"].astype(int), nodes["member_borrower_ids"].astype(str)))
    is_category = set(nodes.loc[nodes["node_type"] == "category", "node_id"].astype(int))
    touching: dict[int, set[str]] = {}
    for row in edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        if t in is_category and s not in is_category:
            touching.setdefault(t, set()).add(key_of_node[s])
    adjacency: dict[str, dict[str, int]] = {k: {} for k in class_keys(nodes)}
    for members in touching.values():
        for a in members:
            for b in members:
                if a != b:
                    adjacency[a][b] = adjacency[a].get(b, 0) + 1
    return adjacency


def class_mass(nodes: pd.DataFrame) -> dict[str, float]:
    """Restart mass per class (its share of ``v``), zero for non-seed classes."""
    v = personalization_vector(nodes)
    mass: dict[str, float] = {}
    for position, row in enumerate(nodes.itertuples(index=False)):
        if row.node_type == "borrower_copy":
            key = str(row.member_borrower_ids)
            mass[key] = mass.get(key, 0.0) + float(v[position])
    return mass


def grow_connected_classes(nodes: pd.DataFrame, edges: pd.DataFrame, n_classes: int) -> list[str]:
    """Breadth-first, densest-first growth to ``n_classes`` classes.

    Starts from the seed class with the largest restart mass, so every
    candidate retains at least one observed-default seed with positive
    personalisation mass. Ties go to the heavier class, then the lexically
    smaller key, so the sweep is deterministic.
    """
    adjacency = class_adjacency(nodes, edges)
    mass = class_mass(nodes)
    keys = class_keys(nodes)
    if n_classes <= 0 or n_classes > len(keys):
        raise ValueError(f"n_classes must be in [1, {len(keys)}]; got {n_classes}")
    start = max(keys, key=lambda k: (mass.get(k, 0.0), k))
    if mass.get(start, 0.0) <= 0.0:
        raise ValueError("no seed class carries restart mass; the sweep cannot start")

    retained = [start]
    retained_set = {start}
    while len(retained) < n_classes:
        frontier: dict[str, int] = {}
        for k in retained:
            for neighbour, count in adjacency[k].items():
                if neighbour not in retained_set:
                    frontier[neighbour] = frontier.get(neighbour, 0) + count
        if not frontier:
            break  # the component is exhausted; the candidate is what is connected
        pick = max(frontier, key=lambda k: (frontier[k], mass.get(k, 0.0), -keys.index(k)))
        retained.append(pick)
        retained_set.add(pick)
    return retained


def subgraph_for_classes(nodes: pd.DataFrame, edges: pd.DataFrame, keys) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain the given classes' copies, the categories they touch, and the edges between them."""
    keep_keys = set(str(k) for k in keys)
    copies = nodes[(nodes["node_type"] == "borrower_copy") & nodes["member_borrower_ids"].astype(str).isin(keep_keys)]
    copy_ids = set(copies["node_id"].astype(int))
    touched = set()
    for row in edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        if s in copy_ids and t not in copy_ids:
            touched.add(t)
        if t in copy_ids and s not in copy_ids:
            touched.add(s)
    keep_ids = copy_ids | touched
    nodes_sub = nodes[nodes["node_id"].astype(int).isin(keep_ids)].reset_index(drop=True)
    edges_sub = edges[
        edges["source"].astype(int).isin(keep_ids) & edges["target"].astype(int).isin(keep_ids)
    ].reset_index(drop=True)
    return nodes_sub, edges_sub


def is_connected(nodes: pd.DataFrame, edges: pd.DataFrame) -> bool:
    ids = [int(x) for x in nodes["node_id"].tolist()]
    if not ids:
        return False
    neighbours: dict[int, set[int]] = {i: set() for i in ids}
    for row in edges.itertuples(index=False):
        s, t = int(row.source), int(row.target)
        neighbours[s].add(t)
        neighbours[t].add(s)
    seen = {ids[0]}
    stack = [ids[0]]
    while stack:
        node = stack.pop()
        for nxt in neighbours[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(ids)


def coverage_counts(weights: np.ndarray, bad_masked: np.ndarray) -> dict[str, int]:
    """Borrowers covered by a set of classes, split by the masked label of each class.

    ``weights`` are the member counts of the layer-0 class copies and
    ``bad_masked`` their lumping-key level: ``1`` is a seed class, ``-1``
    (``quantum.lumping.UNKNOWN_LABEL``) a test or held-out class. The one
    definition of ``borrowers_covered`` / ``test_borrowers_covered`` /
    ``seed_borrowers_covered``, shared by the sweep rows and the coverage bound.
    """
    weights = np.asarray(weights, dtype=float)
    level = np.asarray(bad_masked, dtype=int)
    return {
        "borrowers_covered": int(round(weights.sum())),
        "test_borrowers_covered": int(round(weights[level == -1].sum())),
        "seed_borrowers_covered": int(round(weights[level == 1].sum())),
    }


def candidate_operator(nodes_sub: pd.DataFrame, edges_sub: pd.DataFrame, *, alpha: float, influence_type: str) -> dict[str, Any]:
    """Rebuild and validate the operator on a retained subgraph."""
    G, v = build_lumped_matrices(nodes_sub, edges_sub, alpha=alpha, influence_type=influence_type)
    col_err = float(np.max(np.abs(G.sum(axis=0) - 1.0)))
    if col_err > 1e-10:
        raise AssertionError(f"candidate operator is not column-stochastic (error {col_err:.3e})")
    copies = nodes_sub[nodes_sub["node_type"] == "borrower_copy"]
    layer0 = copies[copies["layer_index"] == 0]
    seeds = int((layer0["BAD"] == 1).sum())
    if seeds < 1 or float(v.sum()) <= 0.0:
        raise AssertionError("candidate retains no observed-default seed with positive mass")
    n_red = int(len(nodes_sub))
    m = int(np.ceil(np.log2(max(n_red, 2))))
    return {
        "G": G,
        "v": v,
        "n_classes": int(len(layer0)),
        "n_seed_classes": seeds,
        **coverage_counts(layer0["weight"].to_numpy(), layer0["BAD_masked"].to_numpy()),
        "n_supra_nodes": n_red,
        "register_width_m": m,
        "logical_qubits": 2 * m,
        "connected": is_connected(nodes_sub, edges_sub),
        "column_sum_error": col_err,
    }


def sweep_sizes(config, n_full: int) -> list[int]:
    """The descending class counts to try: the full count first, then the configured
    sweep clamped to it, without duplicates."""
    sizes = {int(n_full)}
    for n in config.hardware_class_sweep:
        n = min(int(n), int(n_full))
        if n >= 1:
            sizes.add(n)
    return sorted(sizes, reverse=True)


def candidate_label(n_classes: int, n_full: int) -> str:
    return "full" if n_classes == n_full else f"top_{n_classes}_classes"


def build_candidate(nodes: pd.DataFrame, edges: pd.DataFrame, n_classes: int, config) -> dict[str, Any]:
    """Grow (or take all) classes, rebuild the operator, and describe the candidate."""
    n_full = len(class_keys(nodes))
    keys = class_keys(nodes) if n_classes == n_full else grow_connected_classes(nodes, edges, n_classes)
    nodes_sub, edges_sub = subgraph_for_classes(nodes, edges, keys)
    operator = candidate_operator(nodes_sub, edges_sub, alpha=config.resolved_alpha(),
                                  influence_type=config.influence_type)
    return {"label": candidate_label(n_classes, n_full), "keys": keys, "nodes": nodes_sub, "edges": edges_sub,
            "operator": operator}
