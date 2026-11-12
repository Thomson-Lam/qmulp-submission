"""The operators the mechanism study compares, and their routing. None of them is a reported score.

Every variant is a pair of graphs in the format
``quantum.lumped_operator.build_lumped_matrices`` reads: the lumped
``(nodes, edges)`` and the unlumped ``(nodes_full, edges_full)`` built by the
same builder on the same population with the test labels hidden exactly as
the primary operator is, plus the influence type and the personalization mode
the operators are to be built with.

- :func:`eta_variant`: the two-layer graph at an arbitrary interlayer weight,
  ``0`` included (no interlayer edge; the copies stay coupled through the
  combined restart and through the dangling patches to ``v``).
- :func:`own_copy_restart_variant`: the primary graph with a reference
  individual restarting to its own copy only (``influence_type="intra"``).
- :func:`rank_one_restart_variant`: the primary graph with every column
  restarting at ``v`` -- the named rank-one control, never a default.
- :func:`flattened_variant`: a single-layer monoplex with one node per class
  and membership edges into every attribute family.
- :func:`label_free_variant`: the restart, the dangling patch and the initial
  state all uniform over individual copies; no label enters the operator.
- :func:`placebo_variant`: the primary graph with a placebo reference set.
- :func:`single_layer_variants`: one layer as built, restricted to recorded
  borrowers, or with its dangling columns patched by the uniform vector.

Routing
-------
:func:`route_variant` is the one place a variant turns into operators: the
router (``quantum.lumping_router``) builds the lumped and the unlumped operator
and chooses the discriminant. The personalization mode decides the restart on
both sides at once, so the lumped and the unlumped operator always share a
convention:

- ``combined_restart`` -- the primary MuLP restart (reference columns restart
  to the individual's own copies, every other column to ``v``);
- ``rank_one_restart`` -- every column restarts to ``v``, the named control;
- ``label_free`` -- restart, dangling patch and initial state all uniform
  over individual copies.

``uniform_dangling`` patches the dangling columns with the uniform vector
over supra-nodes instead of the restart vector (single-layer control).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fingerprint import UNRECORDED, layer_columns_for, layer_state_ids
from quantum.lumped_operator import group_weights, personalization_vector
from quantum.lumping import UNKNOWN_LABEL, build_lumped_subgraph, frame_to_pandas
from quantum.lumping_router import RoutedOperator, route_graphs


FLAT_LAYER = "FLAT"
MEMBERSHIP = "_membership"
UNRECORDED_FAMILY = "unrecorded"
BORROWER_COPY = "borrower_copy"
COMBINED_RESTART = "combined_restart"
RANK_ONE_RESTART = "rank_one_restart"
LABEL_FREE = "label_free"
PERSONALIZATION_MODES = (COMBINED_RESTART, RANK_ONE_RESTART, LABEL_FREE)


@dataclass(frozen=True)
class OperatorVariant:
    """One control operator: its lumped and unlumped graphs and how the router is to build them."""

    label: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    nodes_full: pd.DataFrame
    edges_full: pd.DataFrame
    influence_type: str = "combined"
    personalization_mode: str = COMBINED_RESTART
    # A dangling patch other than the restart vector, on the lumped side; the
    # full side's patch is derived from it by the router through ``patch_full``.
    dangling_patch: np.ndarray | None = None
    uniform_dangling: bool = False
    n_dangling_columns: int = 0
    n_borrowers_dropped: int = 0

    def __post_init__(self) -> None:
        if self.personalization_mode not in PERSONALIZATION_MODES:
            raise ValueError(f"personalization_mode must be one of {PERSONALIZATION_MODES}")

    @property
    def n_nodes(self) -> int:
        return int(len(self.nodes))

    @property
    def n_edges(self) -> int:
        return int(len(self.edges))


def layer_columns(spec) -> list[tuple[str, list[str]]]:
    """``[(layer, one-hot columns)]`` in block order, markers included, as the class builder expects."""
    return [(layer.name, list(layer.source_columns)) for layer in spec.layers]


def _graph(partitions, spec, hidden_ids, *, level: str, stickiness: float, borrower_ids=None, bad_masked=None):
    frame = partitions.frame
    ids = list(range(frame.height)) if borrower_ids is None else [int(b) for b in borrower_ids]
    levels = partitions.masked_labels(hidden_ids) if bad_masked is None else bad_masked
    return build_lumped_subgraph(
        frame_to_pandas(frame), layer_columns(spec), ids, lumping_level=level,
        stickiness=float(stickiness), bad_masked=np.asarray(levels, dtype=int),
    )


def lumped_graph(partitions, spec, config, hidden_ids, *, stickiness: float, borrower_ids=None, bad_masked=None):
    """The population lumped on the joint key, at an arbitrary interlayer weight.

    Identical to ``quantum.lumping.full_lumped_subgraph`` at
    ``stickiness == config.stickiness`` on the whole population. ``borrower_ids``
    restricts the population and ``bad_masked`` replaces the masked indicator of
    ``hidden_ids``.
    """
    del config
    return _graph(partitions, spec, hidden_ids, level="attr_bad", stickiness=stickiness,
                  borrower_ids=borrower_ids, bad_masked=bad_masked)


def full_graph(partitions, spec, config, hidden_ids, *, stickiness: float, borrower_ids=None, bad_masked=None):
    """The same population unlumped (one node per individual per layer), by the same builder."""
    del config
    return _graph(partitions, spec, hidden_ids, level="none", stickiness=stickiness,
                  borrower_ids=borrower_ids, bad_masked=bad_masked)


def both_graphs(partitions, spec, config, hidden_ids, *, stickiness: float, borrower_ids=None, bad_masked=None):
    """``(nodes, edges, nodes_full, edges_full)`` on one population, one masking, one interlayer weight."""
    nodes, edges = lumped_graph(partitions, spec, config, hidden_ids, stickiness=stickiness,
                                borrower_ids=borrower_ids, bad_masked=bad_masked)
    nodes_f, edges_f = full_graph(partitions, spec, config, hidden_ids, stickiness=stickiness,
                                  borrower_ids=borrower_ids, bad_masked=bad_masked)
    return nodes, edges, nodes_f, edges_f


def _copies(nodes: pd.DataFrame) -> pd.DataFrame:
    return nodes[nodes["node_type"].astype(str) == BORROWER_COPY]


def _membership_sources(edges: pd.DataFrame) -> set[int]:
    return {int(s) for s, rel in zip(edges["source"], edges["relation"]) if str(rel).endswith(MEMBERSHIP)}


def dangling_columns(nodes: pd.DataFrame, edges: pd.DataFrame) -> int:
    """Class copies with no edge at all; ``build_lumped_matrices`` patches their columns."""
    sources = {int(s) for s in edges["source"].tolist()}
    return int(sum(1 for node in _copies(nodes)["node_id"].astype(int) if node not in sources))


def _variant(label: str, graphs, **kwargs) -> OperatorVariant:
    nodes, edges, nodes_f, edges_f = graphs
    return OperatorVariant(label, nodes, edges, nodes_f, edges_f,
                           n_dangling_columns=dangling_columns(nodes, edges), **kwargs)


def primary_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """The primary operator itself, as a variant, so it routes like every control."""
    return _variant("primary", both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness))


def eta_variant(partitions, spec, config, hidden_ids, eta: float) -> OperatorVariant:
    """The two-layer graph at interlayer weight ``eta``; ``eta = 0`` emits no interlayer edge."""
    if spec.n_layers != 2:
        raise ValueError(f"the eta sweep needs a two-layer network; {spec.name} has {spec.n_layers}")
    return _variant(f"eta_{float(eta):g}", both_graphs(partitions, spec, config, hidden_ids, stickiness=float(eta)))


def own_copy_restart_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """The primary graph; a reference individual's columns restart to the individual's own copy only."""
    return _variant("own_copy_restart", both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness),
                    influence_type="intra")


def rank_one_restart_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """The primary graph with every column restarting at ``v``: the named rank-one control."""
    return _variant(RANK_ONE_RESTART, both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness),
                    personalization_mode=RANK_ONE_RESTART)


def label_free_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """``G = alpha*T + (1-alpha)*u 1^T`` with ``u`` uniform over individual copies, no label anywhere."""
    return _variant(LABEL_FREE, both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness),
                    personalization_mode=LABEL_FREE)


def flatten(nodes2: pd.DataFrame, edges2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One node per class joined to every category it has a state in; no copies, no interlayer edge.

    Works on the lumped and on the unlumped two-layer graph alike (an unlumped
    "class" is one borrower). Weights are the lumped family's: class ->
    category ``1.0``, category -> class ``m``. A reference class restarts to
    itself, which ``build_lumped_matrices`` applies because every node sits in
    one layer.
    """
    copies = _copies(nodes2)
    classes = copies[copies["layer_index"].astype(int) == 0].reset_index(drop=True)
    class_of_key = {str(k): g for g, k in enumerate(classes["member_borrower_ids"].astype(str))}
    class_of_node = {int(n): class_of_key[str(k)]
                     for n, k in zip(copies["node_id"].astype(int), copies["member_borrower_ids"].astype(str))}
    n_groups = len(classes)
    categories = nodes2[nodes2["node_type"].astype(str) == "category"].sort_values("node_id")
    flat_id = {int(n): n_groups + i for i, n in enumerate(categories["node_id"].astype(int))}
    state_columns = [c for c in nodes2.columns if c.startswith("state_")]
    node_rows: list[dict[str, Any]] = []
    for g, row in classes.iterrows():
        node_rows.append({"node_id": int(g), "node_type": BORROWER_COPY, "layer": FLAT_LAYER, "layer_index": 0,
                          "borrower_id": row["borrower_id"], "member_borrower_ids": row["member_borrower_ids"],
                          "label": f"{FLAT_LAYER}:group_{int(g)}", "BAD": float(row["BAD"]),
                          "BAD_masked": int(row["BAD_masked"]), "weight": float(row["weight"]),
                          **{c: row[c] for c in state_columns}})
    for _, row in categories.iterrows():
        node_rows.append({"node_id": flat_id[int(row["node_id"])], "node_type": "category", "layer": FLAT_LAYER,
                          "layer_index": 0, "borrower_id": None, "member_borrower_ids": None,
                          "label": str(row["label"]), "BAD": 0.0, "BAD_masked": 0, "weight": None,
                          **{c: None for c in state_columns}})
    edge_rows: list[dict[str, Any]] = []
    for row in edges2.itertuples(index=False):
        relation = str(row.relation)
        if relation.endswith(MEMBERSHIP):
            source, target = class_of_node[int(row.source)], flat_id[int(row.target)]
        elif relation.endswith(f"{MEMBERSHIP}_rev"):
            source, target = flat_id[int(row.source)], class_of_node[int(row.target)]
        else:
            continue
        edge_rows.append({"source": source, "target": target, "relation": relation, "weight": float(row.weight)})
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def flattened_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """The single-layer monoplex of the two-layer classes, lumped and unlumped."""
    nodes2, edges2, nodes2_f, edges2_f = both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness)
    nodes, edges = flatten(nodes2, edges2)
    nodes_f, edges_f = flatten(nodes2_f, edges2_f)
    return _variant("flattened", (nodes, edges, nodes_f, edges_f))


def drop_partial_copies(nodes: pd.DataFrame, edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove every class copy with no membership edge, and the interlayer edges that joined it.

    An individual with no category in layer ``l`` then has no copy in ``l``;
    its restart (``build_lumped_matrices`` through ``copy_groups``) is
    uniform over the copies that exist and interlayer edges join existing
    copies only. Works on the lumped and the unlumped graph alike.
    """
    attached = _membership_sources(edges)
    copies = _copies(nodes)
    drop = {int(n) for n in copies["node_id"].astype(int) if int(n) not in attached}
    kept_nodes = nodes[~nodes["node_id"].astype(int).isin(drop)].reset_index(drop=True)
    kept_edges = edges[~edges["source"].astype(int).isin(drop) & ~edges["target"].astype(int).isin(drop)]
    return kept_nodes, kept_edges.reset_index(drop=True)


def no_partial_copies_variant(partitions, spec, config, hidden_ids) -> OperatorVariant:
    """The primary graph with the copies that carry no membership edge removed."""
    nodes2, edges2, nodes2_f, edges2_f = both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness)
    nodes, edges = drop_partial_copies(nodes2, edges2)
    nodes_f, edges_f = drop_partial_copies(nodes2_f, edges2_f)
    return _variant("no_partial_copies", (nodes, edges, nodes_f, edges_f))


def placebo_variant(partitions, spec, config, hidden_ids, seed_ids, *, label: str) -> OperatorVariant:
    """The primary graph with the reference set reassigned to ``seed_ids``; hidden labels stay hidden."""
    hidden = np.asarray(list(hidden_ids), dtype=int)
    seeds = np.asarray(list(seed_ids), dtype=int)
    if np.intersect1d(hidden, seeds).size:
        raise ValueError("a placebo reference set must not contain a hidden borrower")
    levels = partitions.placebo_labels(seeds)
    levels[hidden] = UNKNOWN_LABEL
    return _variant(label, both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness,
                                       bad_masked=levels))


def uniform_over_copies(nodes: pd.DataFrame) -> np.ndarray:
    """Uniform over individuals, split equally across copies: class size on every copy, zero elsewhere."""
    copy = (nodes["node_type"].astype(str) == BORROWER_COPY).to_numpy()
    u = np.where(copy, group_weights(nodes), 0.0)
    return u / float(u.sum())


def uniform_over_supra_nodes(nodes: pd.DataFrame) -> np.ndarray:
    """The image of the uniform vector over unlumped supra-nodes: class size per copy, one per category."""
    u = group_weights(nodes)
    return u / float(u.sum())


def single_layer_variants(partitions, spec, config, hidden_ids) -> dict[str, OperatorVariant]:
    """``as_is``, ``recorded_only`` and ``uniform_dangling`` for one single-layer network."""
    if spec.n_layers != 1:
        raise ValueError(f"single-layer variants need one layer; {spec.name} has {spec.n_layers}")
    graphs = both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness)
    frame = partitions.frame
    genuine = layer_columns_for(frame, spec)[spec.layers[0].name]
    recorded = np.flatnonzero(layer_state_ids(frame, genuine) != UNRECORDED)
    graphs_r = both_graphs(partitions, spec, config, hidden_ids, stickiness=config.stickiness, borrower_ids=recorded)
    return {
        "as_is": _variant("as_is", graphs),
        "recorded_only": _variant("recorded_only", graphs_r, n_borrowers_dropped=int(frame.height - recorded.size)),
        "uniform_dangling": _variant("uniform_dangling", graphs, uniform_dangling=True),
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _vectors(variant: OperatorVariant) -> dict[str, np.ndarray | None]:
    mode = variant.personalization_mode
    out: dict[str, np.ndarray | None] = {
        "personalization_lumped": None, "personalization_full": None,
        "dangling_patch_lumped": None, "dangling_patch_full": None,
    }
    if mode == RANK_ONE_RESTART:
        out["personalization_lumped"] = personalization_vector(variant.nodes)
        out["personalization_full"] = personalization_vector(variant.nodes_full)
    elif mode == LABEL_FREE:
        u_l, u_f = uniform_over_copies(variant.nodes), uniform_over_copies(variant.nodes_full)
        out.update(personalization_lumped=u_l, personalization_full=u_f,
                   dangling_patch_lumped=u_l, dangling_patch_full=u_f)
    elif mode != COMBINED_RESTART:
        raise ValueError(f"unknown personalization mode {mode!r}")
    if variant.uniform_dangling:
        out["dangling_patch_lumped"] = uniform_over_supra_nodes(variant.nodes)
        out["dangling_patch_full"] = uniform_over_supra_nodes(variant.nodes_full)
    if variant.dangling_patch is not None:
        raise ValueError("pass uniform_dangling=True; an explicit lumped patch has no unlumped counterpart")
    return out


def route_variant(variant: OperatorVariant, config, *, keep_full: bool = False) -> RoutedOperator:
    """Both operators of ``variant`` at the frozen ``alpha`` and the discriminant the router chose."""
    routed = route_graphs(
        variant.nodes, variant.edges, variant.nodes_full, variant.edges_full,
        alpha=config.resolved_alpha(), influence_type=variant.influence_type,
        router_tol=config.router_tol, personalization_mode=variant.personalization_mode,
        keep_full=keep_full, **_vectors(variant),
    )
    routed.residuals["variant"] = variant.label
    return routed
