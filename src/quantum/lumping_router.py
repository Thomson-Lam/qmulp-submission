"""The lumping router: which discriminant the lumped walk runs on, and the two residuals that justify it.

Lumping merges automorphic supra-nodes into blocks. With ``C`` the block
indicator (``R x N``), ``M = diag(m_a)`` the block sizes and
``B = C^T M^{-1/2}`` (orthonormal columns), the full Google operator ``G`` is
lumpable when ``C G = G-bar C``; the lumpability residual ``max |C G - G-bar C|``
over the covered supra-nodes measures how far that holds. The Szegedy walk on
``G`` is driven by the discriminant ``D = sqrt(G o G^T)`` (``o`` elementwise).
Two candidates describe its action on the block-constant subspace:

- ``D-bar = sqrt(G-bar o G-bar^T)``, the discriminant of the lumped operator;
- ``H = B^T D B``, the compressed discriminant, which is exact whenever the
  full operator is invariant under the automorphisms that define the blocks.

The two agree exactly when ``D B = B D-bar`` (the compatibility condition);
``sqrt`` does not commute with the member sum, so they differ, for instance,
on dangling copies patched with ``v`` (interlayer weight zero) and under the
label-free restart. :func:`route_operator` computes both residuals, chooses
``K = D-bar`` when ``max |D B - B D-bar| <= router_tol`` and ``K = H``
otherwise, and records the choice. Every lumped quantum computation in the
repository goes through it; nothing else builds a discriminant for a walk.

The full side is the unlumped graph built by the same builder as the lumped
one (``quantum.lumping.build_lumped_subgraph`` at ``lumping_level="none"``)
and the same dense operator builder, so the two sides share every
convention. The dense full operator costs ``8 N^2`` bytes (about 1.2 GB on
either dataset's two-layer network); the residuals are a handful of
``N x N`` passes and one ``N x N x R`` product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from quantum.lumped_operator import build_lumped_matrices, group_weights

ROUTE_DBAR = "D_bar"
ROUTE_H = "H"


@dataclass(frozen=True)
class BlockStructure:
    """Which block (compact lumped position) every full supra-node belongs to.

    ``block_of`` is ``-1`` for a full node no block covers (a placeholder slot
    or an unused category, which carries no mass); ``sizes`` are the member
    counts the lumped frame records, checked against the map.
    """

    block_of: np.ndarray
    sizes: np.ndarray

    @property
    def n_full(self) -> int:
        return int(self.block_of.size)

    @property
    def n_blocks(self) -> int:
        return int(self.sizes.size)

    @property
    def covered(self) -> np.ndarray:
        return np.flatnonzero(self.block_of >= 0)

    def indicator(self) -> csr_matrix:
        """``C`` (``R x N``): one per covered node in its block's row."""
        cov = self.covered
        return csr_matrix(
            (np.ones(cov.size), (self.block_of[cov], cov)), shape=(self.n_blocks, self.n_full)
        )

    def isometry(self) -> np.ndarray:
        """``B = C^T M^{-1/2}`` (``N x R``), zero on uncovered rows."""
        B = np.zeros((self.n_full, self.n_blocks), dtype=float)
        cov = self.covered
        B[cov, self.block_of[cov]] = 1.0 / np.sqrt(self.sizes[self.block_of[cov]])
        return B


def _node_keys(nodes: pd.DataFrame) -> list[tuple[str, Any]]:
    """One hashable key per node: ``("copy", layer, borrower)`` per member, or ``("cat", label)``."""
    keys: list[tuple[str, Any]] = []
    for row in nodes.itertuples(index=False):
        if str(row.node_type) == "borrower_copy":
            keys.append(("copy", str(row.layer), str(row.member_borrower_ids)))
        else:
            keys.append(("cat", str(row.label)))
    return keys


def block_structure(nodes_lumped: pd.DataFrame, nodes_full: pd.DataFrame) -> BlockStructure:
    """Map the unlumped graph's nodes onto the lumped graph's blocks, member by member."""
    position_of: dict[tuple[str, Any], int] = {}
    for position, row in enumerate(nodes_lumped.itertuples(index=False)):
        if str(row.node_type) == "borrower_copy":
            for member in str(row.member_borrower_ids).split(";"):
                if member:
                    position_of[("copy", str(row.layer), member)] = position
        else:
            position_of[("cat", str(row.label))] = position
    block_of = np.array([position_of.get(key, -1) for key in _node_keys(nodes_full)], dtype=int)
    sizes = group_weights(nodes_lumped)
    counts = np.bincount(block_of[block_of >= 0], minlength=len(nodes_lumped))
    if not np.array_equal(counts, np.rint(sizes).astype(int)):
        raise AssertionError(
            "the lumped frame's member counts do not match the members found in the full graph"
        )
    return BlockStructure(block_of=block_of, sizes=sizes)


def discriminant(G: np.ndarray) -> np.ndarray:
    """``D = sqrt(G o G^T)``, real symmetric, spectrum in ``[-1, 1]``; one ``N x N`` temporary."""
    G = np.asarray(G, dtype=float)
    return np.sqrt(G * G.T)


def lumpability_residual(G_full: np.ndarray, G_bar: np.ndarray, blocks: BlockStructure) -> float:
    """``max |C G - G-bar C|`` over the covered columns of the full operator."""
    C = blocks.indicator()
    cov = blocks.covered
    left = np.asarray(C @ G_full)[:, cov]
    right = (np.asarray(G_bar, dtype=float) @ C.toarray())[:, cov]
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def compatibility_residual(DB: np.ndarray, D_bar: np.ndarray, blocks: BlockStructure) -> float:
    """``max |D B - B D-bar|`` over the covered rows, with ``D B`` already formed."""
    cov = blocks.covered
    right = blocks.isometry() @ np.asarray(D_bar, dtype=float)
    return float(np.max(np.abs(DB[cov] - right[cov]))) if cov.size else 0.0


def compressed_discriminant(DB: np.ndarray, blocks: BlockStructure) -> np.ndarray:
    """``H = B^T D B`` from the product ``D B`` already formed; symmetrised against rounding."""
    H = blocks.isometry().T @ DB
    return 0.5 * (H + H.T)


@dataclass(frozen=True)
class RoutedOperator:
    """A lumped operator with the discriminant its walk runs on, and why."""

    G: np.ndarray
    v: np.ndarray
    K: np.ndarray
    route: str
    D_bar: np.ndarray
    H: np.ndarray
    residuals: dict[str, Any] = field(default_factory=dict)
    # The unlumped operator the residuals were measured against, kept only on
    # request (it is 8 N^2 bytes) for the full-versus-reduced step check.
    G_full: np.ndarray | None = None
    v_full: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.G.shape[0])

    def record(self) -> dict[str, Any]:
        """What every artifact built on this operator records."""
        return {"route": self.route, **dict(self.residuals)}


def route_operator(
    G_bar: np.ndarray,
    v_bar: np.ndarray,
    *,
    G_full: np.ndarray,
    blocks: BlockStructure,
    router_tol: float,
) -> RoutedOperator:
    """Both residuals, both candidates, and the one discriminant the lumped walk uses.

    ``K = D-bar`` when ``max |D B - B D-bar| <= router_tol``, else ``K = H``.
    The lumpability residual is recorded alongside; a lumped operator that is
    not lumpable (residual above ``router_tol``) is refused, because neither
    candidate describes a walk on the full operator then.
    """
    G_bar = np.asarray(G_bar, dtype=float)
    G_full = np.asarray(G_full, dtype=float)
    if G_bar.shape != (blocks.n_blocks, blocks.n_blocks):
        raise ValueError(f"G_bar has shape {G_bar.shape}; the block map has {blocks.n_blocks} blocks")
    if G_full.shape != (blocks.n_full, blocks.n_full):
        raise ValueError(f"G_full has shape {G_full.shape}; the block map covers {blocks.n_full} nodes")
    lump = lumpability_residual(G_full, G_bar, blocks)
    if lump > float(router_tol):
        raise AssertionError(
            f"the lumped operator is not lumpable: max |C G - G-bar C| = {lump:.3e} > {router_tol}"
        )
    D_bar = discriminant(G_bar)
    DB = discriminant(G_full) @ blocks.isometry()
    compat = compatibility_residual(DB, D_bar, blocks)
    H = compressed_discriminant(DB, blocks)
    route = ROUTE_DBAR if compat <= float(router_tol) else ROUTE_H
    K = D_bar if route == ROUTE_DBAR else H
    return RoutedOperator(
        G=G_bar, v=np.asarray(v_bar, dtype=float), K=K, route=route, D_bar=D_bar, H=H,
        residuals={
            "lumpability_residual": lump,
            "compatibility_residual": compat,
            "h_minus_dbar_max": float(np.max(np.abs(H - D_bar))),
            "router_tol": float(router_tol),
        },
    )


def route_graphs(
    nodes_lumped: pd.DataFrame,
    edges_lumped: pd.DataFrame,
    nodes_full: pd.DataFrame,
    edges_full: pd.DataFrame,
    *,
    alpha: float,
    influence_type: str,
    router_tol: float,
    personalization_mode: str = "combined_restart",
    dangling_patch_lumped: np.ndarray | None = None,
    dangling_patch_full: np.ndarray | None = None,
    personalization_lumped: np.ndarray | None = None,
    personalization_full: np.ndarray | None = None,
    keep_full: bool = False,
) -> RoutedOperator:
    """Build both operators with one builder and route the lumped one.

    ``personalization_mode`` is recorded only; the operators follow the
    ``personalization_*`` and ``dangling_patch_*`` hooks of
    :func:`quantum.lumped_operator.build_lumped_matrices` (``None`` on both
    sides is the primary combined restart; a vector on both sides is a
    rank-one control such as ``rank_one_restart`` or the label-free operator).
    ``keep_full`` retains the unlumped ``(G, v)`` on the result.
    """
    G_bar, v_bar = build_lumped_matrices(
        nodes_lumped, edges_lumped, alpha=alpha, influence_type=influence_type,
        personalization=personalization_lumped, dangling_patch=dangling_patch_lumped,
    )
    G_full, v_full = build_lumped_matrices(
        nodes_full, edges_full, alpha=alpha, influence_type=influence_type,
        personalization=personalization_full, dangling_patch=dangling_patch_full,
    )
    blocks = block_structure(nodes_lumped, nodes_full)
    routed = route_operator(G_bar, v_bar, G_full=G_full, blocks=blocks, router_tol=router_tol)
    routed.residuals["personalization_mode"] = str(personalization_mode)
    routed.residuals["n_full"] = int(blocks.n_full)
    routed.residuals["n_reduced"] = int(blocks.n_blocks)
    if keep_full:
        return RoutedOperator(G=routed.G, v=routed.v, K=routed.K, route=routed.route, D_bar=routed.D_bar,
                              H=routed.H, residuals=routed.residuals, G_full=G_full, v_full=v_full)
    return routed
