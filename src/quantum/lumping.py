"""Lumping: the equivalence classes of borrowers and their keyed ``(nodes, edges)`` subgraph.

Shared by both halves of the study. The classical arm is compared against the
lumped operator to assert zero reduction error (notebook 02), the exact
quantum-inspired arm runs on it (notebook 04), and the hardware sweep reduces
over its classes (notebook 06). It therefore lives at ``quantum/lumping.py``
rather than inside either ``quantum/inspired`` or ``quantum/hardware``. The
operator rebuilt on the classes, and the unfolding back to borrowers, are in
``quantum.lumped_operator``.

Classes
-------
``build_lumped_subgraph`` merges borrowers that share the key
``(JOB state, REASON state, BAD_masked)`` into one class node per layer.
**The key is the joint triple, and the default indicator in it is the masked
one.** ``bad_masked`` is what the operator can see: ``1`` for a development seed
with an observed default, ``0`` for a visible non-default, and
``UNKNOWN_LABEL`` for every test borrower and every cross-fit-held-out
development borrower. The true ``BAD`` of a hidden borrower never enters the
key, so no test borrower's class can depend on its outcome.

Node id layout, for ``n_groups`` classes across ``L`` layers:

    [l*n_groups, (l+1)*n_groups)   layer l's copy of class g
    [L*n_groups, ...)              category nodes, layer by layer

Only category states that actually receive a membership edge get a node, so
callers must read ids off the returned frame rather than recomputing offsets.
Absence is not a shared attribute: a class whose state in a layer is
unrecorded gets no membership edge and no category node there.

Edge weights: class -> category ``1.0`` (each member has one unit membership
edge), category -> class ``m`` (the category node had ``m`` unit edges into the
class's members), class copy <-> class copy ``S``. These are what make the
lumping exact; ``quantum.lumped_operator`` documents why and
``tests/test_lumping.py`` asserts it.

Only classes keyed on the masked default indicator can be exact: a seed owns a
personalized restart column while a non-seed restarts to ``v``, so the two are
not interchangeable states and must never share a class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from preprocessing.missing_levels import MISSING_SUFFIX, missing_category_indices

# "attr_bad" keys on the full connected-attribute tuple plus the masked default
# indicator. "job_reason_bad" is the two-layer name for the same thing and is
# kept as an alias. Coarser levels were removed: each dropped an edge family.
LUMPING_LEVELS: tuple[str, ...] = ("none", "attr_bad", "job_reason_bad")
LUMPING_ALIASES = {"job_reason_bad": "attr_bad"}

# Category id for a row with no active one-hot column: a field that was never
# recorded. Distinct from any real state, per layer.
UNRECORDED_CATEGORY = -1
# The masked default indicator's third level; must match evaluation.masked_eval.
UNKNOWN_LABEL = -1

DEFAULT_LAYER_NAMES = ("JOB", "REASON")


def _category_ids(df: Any, cols: Sequence[str]) -> np.ndarray:
    """argmax over one-hot columns -> state id per row, ``-1`` where all zero."""
    values = np.asarray(df[list(cols)])
    ids = values.argmax(axis=1).astype(int)
    ids[~values.astype(bool).any(axis=1)] = UNRECORDED_CATEGORY
    return ids


def _bad_column(df: Any) -> np.ndarray:
    if "BAD" not in list(df.columns):
        raise ValueError("df must contain a BAD column")
    return np.asarray(df["BAD"]).astype(int)


def _normalize_level(lumping_level: str) -> str:
    if lumping_level not in LUMPING_LEVELS:
        raise ValueError(f"lumping_level must be one of {LUMPING_LEVELS}; got {lumping_level!r}")
    return LUMPING_ALIASES.get(lumping_level, lumping_level)


def missing_category_report(
    df: Any,
    job_cols: Sequence[str],
    reason_cols: Sequence[str],
    borrower_ids: Sequence[int],
    *,
    missing_suffix: str = MISSING_SUFFIX,
) -> dict[str, Any]:
    """Count how many selected borrowers have an unrecorded JOB/REASON level."""
    borrower_ids = [int(b) for b in borrower_ids]
    job_cols, reason_cols = list(job_cols), list(reason_cols)
    missing_job = missing_category_indices(job_cols, missing_suffix)
    missing_reason = missing_category_indices(reason_cols, missing_suffix)

    job_id_all = _category_ids(df, job_cols)
    reason_id_all = _category_ids(df, reason_cols)
    unrecorded_job = missing_job | {UNRECORDED_CATEGORY}
    unrecorded_reason = missing_reason | {UNRECORDED_CATEGORY}
    job_missing = [b for b in borrower_ids if int(job_id_all[b]) in unrecorded_job]
    reason_missing = [b for b in borrower_ids if int(reason_id_all[b]) in unrecorded_reason]
    both = sorted(set(job_missing) & set(reason_missing))
    return {
        "missing_job_columns": [job_cols[i] for i in sorted(missing_job)],
        "missing_reason_columns": [reason_cols[i] for i in sorted(missing_reason)],
        "n_selected": len(borrower_ids),
        "n_job_missing": len(job_missing),
        "n_reason_missing": len(reason_missing),
        "n_both_missing": len(both),
        "n_either_missing": len(set(job_missing) | set(reason_missing)),
    }


def build_lumped_subgraph(
    df: Any,
    layer_columns: "Sequence[tuple[str, Sequence[str]]]",
    borrower_ids: Sequence[int],
    *,
    lumping_level: str = "none",
    stickiness: float = 1.0,
    connect_missing_categories: bool = False,
    missing_suffix: str = MISSING_SUFFIX,
    bad_masked: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``(nodes_df, edges_df)`` for a one- or two-layer selection.

    Parameters
    ----------
    df : frame carrying every layer's one-hot columns and ``BAD``.
    layer_columns : ``[(layer_name, one_hot_columns), ...]`` in layer order,
        absence markers included -- this function excludes them itself.
    borrower_ids : row indices into ``df``, already selected.
    lumping_level : ``"none"`` keeps one node per borrower per layer;
        ``"attr_bad"`` merges borrowers sharing the attribute tuple and the
        masked default indicator.
    stickiness : the MuLP interlayer weight ``S``. At ``S == 0`` no interlayer
        edge is emitted (a zero-weight edge and an absent edge are the same
        operator, and an absent edge keeps the edge frame honest); the copies
        then stay coupled only through the combined restart, which sends a seed
        to both of its copies, and through the dangling patches to ``v``.
    bad_masked : the masked indicator per row of ``df`` (``1``, ``0``,
        ``UNKNOWN_LABEL``). When omitted the frame's own ``BAD`` column is the
        indicator -- which is only right when that column is already masked, as
        it is for a frame produced by ``Partitions.mask_labels``.

    The emitted ``BAD`` column on class nodes is the *seed indicator* the
    operator reads (1 for a seed class, 0 otherwise); ``BAD_masked`` carries
    the full three-level key.
    """
    level = _normalize_level(lumping_level)
    borrower_ids = [int(b) for b in borrower_ids]
    if not borrower_ids:
        raise ValueError("borrower_ids must not be empty")
    layers = [(str(name), list(cols)) for name, cols in layer_columns]
    if not 1 <= len(layers) <= 2:
        raise ValueError("the network has one or two layers")

    stickiness = float(stickiness)
    if not stickiness >= 0.0:
        raise ValueError(f"stickiness must be non-negative; got {stickiness!r}")

    n_layers = len(layers)
    state_ids = [_category_ids(df, cols) for _name, cols in layers]
    if bad_masked is None:
        levels_all = _bad_column(df)
    else:
        levels_all = np.asarray(bad_masked, dtype=int)
        if levels_all.shape != (len(df),):
            raise ValueError("bad_masked must have one entry per row of df")
    unknown_levels = set(np.unique(levels_all).tolist()) - {0, 1, UNKNOWN_LABEL}
    if unknown_levels:
        raise ValueError(
            f"bad_masked takes values 0, 1, {UNKNOWN_LABEL}; got {sorted(unknown_levels)}"
        )

    if level == "none":
        groups: list[list[int]] = [[b] for b in borrower_ids]
        group_states = [[int(ids[b]) for b in borrower_ids] for ids in state_ids]
        group_level = [int(levels_all[b]) for b in borrower_ids]
    else:
        buckets: dict[tuple, list[int]] = {}
        for b in borrower_ids:
            key = (*(int(ids[b]) for ids in state_ids), int(levels_all[b]))
            buckets.setdefault(key, []).append(b)
        ordered_keys = sorted(buckets.keys())
        groups = [buckets[k] for k in ordered_keys]
        group_states = [[k[l] for k in ordered_keys] for l in range(n_layers)]
        group_level = [k[-1] for k in ordered_keys]

    n_groups = len(groups)
    marker_ids = [missing_category_indices(cols, missing_suffix) for _name, cols in layers]

    def connects(state: int, markers: set[int]) -> bool:
        if int(state) == UNRECORDED_CATEGORY:
            return False
        return connect_missing_categories or int(state) not in markers

    edge_groups: list[list[tuple[int, int]]] = []
    for layer in range(n_layers):
        edge_groups.append(
            [
                (g, int(group_states[layer][g]))
                for g in range(n_groups)
                if connects(group_states[layer][g], marker_ids[layer])
            ]
        )

    category_node: list[dict[int, int]] = []
    next_id = n_layers * n_groups
    used_states: list[list[int]] = []
    for layer in range(n_layers):
        states = sorted({s for _g, s in edge_groups[layer]})
        used_states.append(states)
        category_node.append({s: next_id + i for i, s in enumerate(states)})
        next_id += len(states)

    node_rows: list[dict[str, Any]] = []
    for layer, (layer_name, _cols) in enumerate(layers):
        for g, members in enumerate(groups):
            node_rows.append(
                {
                    "node_id": layer * n_groups + g,
                    "node_type": "borrower_copy",
                    "layer": layer_name,
                    "layer_index": layer,
                    "borrower_id": float(members[0]),
                    "member_borrower_ids": ";".join(str(m) for m in members),
                    "label": f"{layer_name}:group_{g}",
                    "BAD": 1.0 if group_level[g] == 1 else 0.0,
                    "BAD_masked": int(group_level[g]),
                    "weight": float(len(members)),
                    **{
                        f"state_{name}": int(group_states[l][g])
                        for l, (name, _c) in enumerate(layers)
                    },
                }
            )
    for layer, (layer_name, cols) in enumerate(layers):
        for state in used_states[layer]:
            node_rows.append(
                {
                    "node_id": category_node[layer][state],
                    "node_type": "category",
                    "layer": layer_name,
                    "layer_index": layer,
                    "borrower_id": None,
                    "member_borrower_ids": None,
                    "label": f"{layer_name}:{cols[state]}",
                    "BAD": 0.0,
                    "BAD_masked": 0,
                    "weight": None,
                    **{f"state_{name}": None for name, _c in layers},
                }
            )
    nodes = pd.DataFrame(node_rows)

    edge_rows: list[dict[str, Any]] = []
    for layer, (layer_name, _cols) in enumerate(layers):
        relation = f"{layer_name.lower()}_membership"
        for g, state in edge_groups[layer]:
            copy_node = layer * n_groups + g
            cat_node = category_node[layer][state]
            edge_rows.append(
                {"source": copy_node, "target": cat_node, "relation": relation, "weight": 1.0}
            )
            edge_rows.append(
                {
                    "source": cat_node,
                    "target": copy_node,
                    "relation": f"{relation}_rev",
                    "weight": float(len(groups[g])),
                }
            )

    if n_layers == 2 and stickiness > 0.0:
        for g in range(n_groups):
            node_a, node_b = g, n_groups + g
            edge_rows.append(
                {"source": node_a, "target": node_b, "relation": "stickiness", "weight": stickiness}
            )
            edge_rows.append(
                {
                    "source": node_b,
                    "target": node_a,
                    "relation": "stickiness_rev",
                    "weight": stickiness,
                }
            )

    return nodes, pd.DataFrame(edge_rows)


def build_topk_lumped_subgraph(
    df: Any,
    job_cols: Sequence[str],
    reason_cols: Sequence[str],
    borrower_ids: Sequence[int],
    *,
    lumping_level: str = "none",
    stickiness: float = 1.0,
    connect_missing_categories: bool = False,
    missing_suffix: str = MISSING_SUFFIX,
    bad_masked: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-layer entry point, delegating to :func:`build_lumped_subgraph`."""
    return build_lumped_subgraph(
        df,
        [(DEFAULT_LAYER_NAMES[0], job_cols), (DEFAULT_LAYER_NAMES[1], reason_cols)],
        borrower_ids,
        lumping_level=lumping_level,
        stickiness=stickiness,
        connect_missing_categories=connect_missing_categories,
        missing_suffix=missing_suffix,
        bad_masked=bad_masked,
    )


def write_subgraph_csvs(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: str | Path,
    *,
    prefix: str = "hardware_subgraph",
) -> tuple[Path, Path]:
    """Persist (nodes, edges) to a CSV pair."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes_path = out / f"{prefix}_nodes.csv"
    edges_path = out / f"{prefix}_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    return nodes_path, edges_path


# ---------------------------------------------------------------------------
# The whole population, lumped -- what notebooks 02, 04 and 06 all start from
# ---------------------------------------------------------------------------


def frame_to_pandas(frame) -> pd.DataFrame:
    """Polars -> pandas without the optional pyarrow engine."""
    return pd.DataFrame(frame.to_dict(as_series=False))


def full_lumped_subgraph(partitions, spec, config, hidden_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ``(nodes, edges)`` of the whole population lumped on ``(JOB, REASON, BAD_masked)``.

    ``hidden_ids`` are the borrowers whose labels the operator may not see (the
    test partition for the test operator). Their key carries the unknown level,
    so no hidden true label shapes a class. ``spec`` is a ``config.NetworkSpec``
    and ``partitions`` an ``evaluation.masked_eval.Partitions``.
    """
    from preprocessing.missing_levels import MISSING_INDICATOR_SUFFIX

    frame = partitions.frame
    layer_columns = [
        (
            layer.name,
            sorted(
                c
                for c in frame.columns
                if c.startswith(layer.prefix) and not c.endswith(MISSING_INDICATOR_SUFFIX)
            ),
        )
        for layer in spec.layers
    ]
    return build_lumped_subgraph(
        frame_to_pandas(frame),
        layer_columns,
        list(range(frame.height)),
        lumping_level="attr_bad",
        stickiness=config.stickiness,
        bad_masked=partitions.masked_labels(hidden_ids),
    )


def lumped_facts(nodes: pd.DataFrame, spec) -> dict[str, Any]:
    """Class count, supra-node count and register width of a lumped subgraph."""
    copies = nodes[nodes["node_type"] == "borrower_copy"]
    n_classes = int(len(copies)) // spec.n_layers
    n_red = int(len(nodes))
    m = int(np.ceil(np.log2(max(n_red, 2))))
    covered = int(round(copies[copies["layer_index"] == 0]["weight"].sum()))
    return {
        "n_classes": n_classes,
        "n_seed_classes": int((copies[copies["layer_index"] == 0]["BAD"] == 1).sum()),
        "n_supra_nodes_lumped": n_red,
        "register_width_m": m,
        "logical_qubits_2m": 2 * m,
        "statevector_bytes": int(16 * (2 ** (2 * m))),
        "borrowers_covered": covered,
    }
