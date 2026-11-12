"""The lumped operator: ``(G, v)`` on the equivalence classes, and unfolding back to borrowers.

``quantum.lumping`` builds the classes and their ``(nodes, edges)`` subgraph;
this module turns that subgraph into the operator the walk runs on and recovers
borrower scores from class scores.

Edge weights under lumping
--------------------------
Merging ``m`` structurally identical borrowers must preserve the transition
*probabilities* out of every node, so the two directions differ:

===========================  ==========  ======================================
Edge                         Weight      Why
===========================  ==========  ======================================
class -> category            ``1.0``     each member has one unit membership edge
category -> class            ``m``       the category node had ``m`` unit edges
                                         into that class's members
class copy <-> class copy    ``S``       the per-member interlayer weight
===========================  ==========  ======================================

With these weights the lumped chain is exactly lumpable: a class's stationary
probability equals the sum of its members' unlumped probabilities, and likewise
for the tau-averaged Szegedy score. ``tests/test_lumping.py`` asserts both and
carries a regression case injecting the wrong weight.

The operator
------------
``build_lumped_matrices`` returns the reduced ``(G, v)`` implied by a
``(nodes, edges)`` subgraph, built with the same conventions as
``classical.mulp``: ``A[source, target]`` row-normalized then transposed into
the column-stochastic ``T``, a restart whose seed columns restart onto that
class's own layer copies and whose remaining columns take ``v``, and
``G = alpha*T + (1-alpha)*P``. The ``v 1^T`` block is added into ``G`` column
block by column block, never formed as a separate matrix.

``personalization_vector`` puts mass proportional to member count on every
layer copy of every seed class -- the member-summed image of ``1/(L*|D|)``.

``unfold_class_scores`` recovers individual borrower scores from class scores:
members of a class are graph automorphs, so each holds ``score / m``, summed
over the class's layer copies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INFLUENCE_TYPES = ("combined", "intra", "inter")


def _borrower_copy_mask(nodes: pd.DataFrame) -> np.ndarray:
    return (nodes["node_type"].astype(str) == "borrower_copy").to_numpy()


def group_weights(nodes: pd.DataFrame) -> np.ndarray:
    """Per-position member count, 1.0 where none is recorded.

    Indexed by *compact position* (row position in ``nodes``), never by the
    ``node_id`` value.
    """
    n = len(nodes)
    weights = np.ones(n, dtype=float)
    if "weight" not in nodes.columns:
        return weights
    raw = pd.to_numeric(nodes["weight"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw) & (raw > 0.0)
    weights[valid] = raw[valid]
    return weights


def seed_mask(nodes: pd.DataFrame) -> np.ndarray:
    """Borrower-copy positions whose class is a seed class (``BAD == 1``)."""
    bad = pd.to_numeric(nodes["BAD"], errors="coerce").fillna(0).to_numpy(dtype=float)
    return _borrower_copy_mask(nodes) & (bad.astype(int) == 1)


def personalization_vector(nodes: pd.DataFrame) -> np.ndarray:
    """Reduced ``v`` over every compact position, summing to 1."""
    if len(nodes) == 0:
        raise ValueError("nodes must not be empty")
    seed = seed_mask(nodes)
    if not seed.any():
        raise ValueError("subgraph contains no seed borrower-copy node to restart from")
    v = np.where(seed, group_weights(nodes), 0.0)
    return v / float(v.sum())


def _group_key(nodes: pd.DataFrame) -> np.ndarray:
    """Identifier shared by all layer copies of one class."""
    if "member_borrower_ids" in nodes.columns:
        return nodes["member_borrower_ids"].astype(str).to_numpy()
    return np.array([str(i) for i in range(len(nodes))])


def borrower_layer_positions(nodes: pd.DataFrame) -> "list[np.ndarray]":
    """One position array per layer, aligned class by class, over the classes present in every layer."""
    is_copy = _borrower_copy_mask(nodes)
    layers = nodes["layer"].astype(str).to_numpy()
    keys = _group_key(nodes)
    order = list(dict.fromkeys(layers[p] for p in range(len(nodes)) if is_copy[p]))

    by_layer = {
        name: {keys[p]: p for p in range(len(nodes)) if is_copy[p] and layers[p] == name}
        for name in order
    }
    shared = [k for k in by_layer[order[0]] if all(k in by_layer[n] for n in order)]
    return [np.array([by_layer[name][k] for k in shared], dtype=int) for name in order]


def copy_groups(nodes: pd.DataFrame) -> "list[list[int]]":
    """The compact positions of every class's copies, one list per class, however many copies it has.

    On the primary graph every class has one copy per layer and this is
    ``zip(*borrower_layer_positions(nodes))``; on the no-partial-copies
    control a class may have a single copy, and its restart column must still
    be the class's own (``xi_i`` uniform over the copies that exist).
    """
    is_copy = _borrower_copy_mask(nodes)
    keys = _group_key(nodes)
    groups: dict[str, list[int]] = {}
    for position in range(len(nodes)):
        if is_copy[position]:
            groups.setdefault(keys[position], []).append(position)
    return list(groups.values())


def _probability_vector(vector, n: int, name: str) -> np.ndarray:
    """A caller-supplied vector over compact positions: nonnegative, unit sum."""
    out = np.asarray(vector, dtype=float)
    if out.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {out.shape}")
    if out.min() < 0.0 or not np.isclose(out.sum(), 1.0, atol=1e-10):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    return out


def build_lumped_matrices(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    alpha: float = 0.85,
    influence_type: str = "combined",
    personalization: np.ndarray | None = None,
    dangling_patch: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense ``(G, v)`` in compact positions; ``G`` is column-stochastic.

    Rebuilt from the retained ``(nodes, edges)`` every time: ``T`` is
    re-normalised on the retained edges, dangling columns are patched with the
    retained ``v``, and the restart columns are rebuilt on the retained seed
    classes. A principal submatrix of the full ``G`` would be none of these.

    Both hooks default to ``None``, which leaves the operator exactly as
    described above. ``personalization`` replaces the seed-based ``v`` with the
    given vector and forms **no** seed column, so that
    ``G = alpha*T + (1-alpha)*u 1^T``: this is the unpersonalized structural
    control of ``quantum.mechanism.variants`` and the one permitted exception
    to the factored restart; it is reported only as a control. ``dangling_patch``
    replaces ``v`` in the dangling columns of ``T`` alone.
    """
    if influence_type not in INFLUENCE_TYPES:
        raise ValueError(f"influence_type must be one of {INFLUENCE_TYPES}; got {influence_type!r}")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    n = len(nodes)
    node_ids = [int(x) for x in nodes["node_id"].tolist()]
    node_to_pos = {node: i for i, node in enumerate(node_ids)}
    if len(node_to_pos) != n:
        raise ValueError("node_id must be unique")

    if personalization is None:
        v = personalization_vector(nodes)
    else:
        v = _probability_vector(personalization, n, "personalization")
    patch = v if dangling_patch is None else _probability_vector(dangling_patch, n, "dangling_patch")

    A = np.zeros((n, n), dtype=float)
    has_weight = "weight" in edges.columns
    for row in edges.itertuples(index=False):
        A[node_to_pos[int(row.source)], node_to_pos[int(row.target)]] = (
            float(row.weight) if has_weight else 1.0
        )

    P = np.zeros_like(A)
    for row in range(n):
        row_sum = float(A[row, :].sum())
        P[row, :] = patch if row_sum <= 0.0 else A[row, :] / row_sum
    T = P.T.copy()

    bad = pd.to_numeric(nodes["BAD"], errors="coerce").fillna(0).to_numpy(dtype=float).astype(int)

    # Seed columns of the restart matrix, already normalised. With a supplied
    # ``personalization`` no class is a seed and every column takes ``u``. A
    # seed class restarts over the copies it has (all layers on the primary
    # graph; the existing copies on the no-partial-copies control).
    R = np.zeros((n, n), dtype=float)
    for group in copy_groups(nodes):
        if personalization is not None or bad[group[0]] != 1:
            continue
        for column in group:
            if influence_type in ("intra", "combined"):
                R[column, column] = 1.0
            if influence_type in ("inter", "combined"):
                for other in group:
                    if other != column:
                        R[other, column] = 1.0
    col_sums = R.sum(axis=0)
    seed_cols = col_sums > 0.0
    R[:, seed_cols] /= col_sums[seed_cols]

    a = float(alpha)
    G = a * T
    G[:, seed_cols] += (1.0 - a) * R[:, seed_cols]
    if (~seed_cols).any():
        # The rank-one fill, applied in place into the empty columns.
        G[:, ~seed_cols] += (1.0 - a) * v[:, np.newaxis]
    if not np.allclose(G.sum(axis=0), 1.0, atol=1e-10):
        raise AssertionError("lumped Google matrix is not column-stochastic")
    return G, v


def class_sizes(nodes: pd.DataFrame) -> dict[int, int]:
    """Compact position -> member count, for borrower-copy positions."""
    weights = group_weights(nodes)
    return {int(p): int(round(weights[p])) for p in np.flatnonzero(_borrower_copy_mask(nodes))}


def class_members(nodes: pd.DataFrame) -> dict[int, list[int]]:
    """Compact position -> member borrower ids, for borrower-copy positions."""
    out: dict[int, list[int]] = {}
    is_copy = _borrower_copy_mask(nodes)
    members = nodes["member_borrower_ids"].astype(str).to_numpy()
    for position in np.flatnonzero(is_copy):
        out[int(position)] = [int(m) for m in members[position].split(";") if m != ""]
    return out


def unfold_class_scores(
    nodes: pd.DataFrame, class_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Borrower scores from class scores: ``(borrower_ids, scores)``.

    A class score is the sum over its members (exact lumpability), and the
    members are automorphic, so each holds ``score / m``; a borrower's score
    is that share summed over the class's layer copies.
    """
    scores = np.asarray(class_scores, dtype=float)
    if scores.shape != (len(nodes),):
        raise ValueError(f"class_scores must have one entry per node; got {scores.shape}")
    totals: dict[int, float] = {}
    for position, members in class_members(nodes).items():
        share = scores[position] / len(members)
        for borrower in members:
            totals[borrower] = totals.get(borrower, 0.0) + share
    borrower_ids = np.array(sorted(totals), dtype=int)
    return borrower_ids, np.array([totals[b] for b in borrower_ids], dtype=float)
