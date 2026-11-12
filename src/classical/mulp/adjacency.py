"""The raw MuLP supra-adjacency, for one or two layers.

Two edge families, and nothing else:

- **membership**, within a layer: borrower copy <-> its own category node,
  weight 1. Omitted entirely when the field was never recorded -- an absent
  value is not an attribute borrowers can share.
- **interlayer**, when there are two layers: a borrower's two copies joined to
  each other with weight ``S`` (stickiness), in both directions. A single-layer
  network has no interlayer edges and ``S`` plays no role.

Category nodes are never coupled across layers: a category is a property of one
relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, lil_matrix

from .dataset import MuLPDatasetLoader
from .indexing import MuLPGraphIndex


@dataclass(slots=True)
class AdjacencyDiagnostics:
    shape: tuple[int, int]
    sparse: bool
    bidirectional: bool
    stickiness: float
    support_count: int
    n_rows: int
    n_layers: int
    states_per_layer: tuple[int, ...]
    membership_edges_per_layer: tuple[int, ...]
    # Fraction of borrowers earning a membership edge in each layer.
    layer_edge_coverage: tuple[float, ...]

    @property
    def n_job(self) -> int:
        return self.states_per_layer[0]

    @property
    def n_reason(self) -> int:
        return self.states_per_layer[1] if len(self.states_per_layer) > 1 else 0


class SupraAdjacencyBuilder:
    def __init__(self) -> None:
        self.diagnostics: AdjacencyDiagnostics | None = None

    def build(
        self,
        frame: pl.DataFrame,
        index: MuLPGraphIndex,
        *,
        sparse: bool,
        bidirectional: bool,
        stickiness: float,
    ) -> Any:
        adj: Any = (
            lil_matrix((index.N, index.N), dtype=np.float64)
            if sparse
            else np.zeros((index.N, index.N), dtype=np.float64)
        )

        def add_edge(source: int, dest: int, weight: float, *, both_ways: bool) -> None:
            adj[source, dest] = weight
            if both_ways:
                adj[dest, source] = weight

        membership_counts: list[int] = []
        for layer in index.layers:
            state_of = {col: s for s, col in enumerate(layer.columns)}
            pairs_active = MuLPDatasetLoader.active_pairs(frame, list(layer.columns))
            for borrower_id, column in pairs_active:
                add_edge(
                    index.borrower_node(layer.position, borrower_id),
                    index.category_node(layer.position, state_of[column]),
                    1.0,
                    both_ways=bidirectional,
                )
            membership_counts.append(len(pairs_active))

        # A borrower's copies are the same entity, so the interlayer edge is
        # always inserted in both directions regardless of `bidirectional`.
        S = float(stickiness)
        if index.n_layers == 2:
            for borrower_id in range(index.n_rows):
                add_edge(
                    index.borrower_node(0, borrower_id),
                    index.borrower_node(1, borrower_id),
                    S,
                    both_ways=True,
                )

        if sparse:
            adj = adj.tocsr()
            support = int(adj.nnz)
        else:
            support = int(np.count_nonzero(adj))

        n_rows = max(index.n_rows, 1)
        self.diagnostics = AdjacencyDiagnostics(
            shape=(index.N, index.N),
            sparse=bool(sparse),
            bidirectional=bool(bidirectional),
            stickiness=S,
            support_count=support,
            n_rows=index.n_rows,
            n_layers=index.n_layers,
            states_per_layer=tuple(layer.n_states for layer in index.layers),
            membership_edges_per_layer=tuple(membership_counts),
            layer_edge_coverage=tuple(c / n_rows for c in membership_counts),
        )
        self._assert_symmetric(adj, sparse=sparse, bidirectional=bidirectional)
        return adj

    @staticmethod
    def _assert_symmetric(adj: Any, *, sparse: bool, bidirectional: bool) -> None:
        """With bidirectional membership the whole supra-adjacency is symmetric."""
        if not bidirectional:
            return
        if sparse:
            difference = abs(adj - adj.transpose())
            worst = float(difference.max()) if difference.nnz else 0.0
        else:
            worst = float(np.abs(adj - adj.T).max()) if adj.size else 0.0
        if worst > 1e-12:
            raise AssertionError(
                f"supra-adjacency is not symmetric under bidirectional=True "
                f"(max |A - A^T| = {worst:.3e})"
            )

    # Kept so callers that select a representation explicitly keep working.
    def build_sparse(self, frame, index, **kwargs) -> csr_matrix:
        return self.build(frame, index, sparse=True, **kwargs)

    def build_dense(self, frame, index, **kwargs) -> np.ndarray:
        return self.build(frame, index, sparse=False, **kwargs)
