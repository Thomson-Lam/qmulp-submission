"""Classical scores on the same dense lumped operators the limit is computed on.

The stationary score is the classical arm of every mechanism comparison. The
``(alpha, eta)`` grid answers the referee's question whether *any* classical
damping and interlayer weight reproduces the limiting QPPR ordering: the answer
is the maximum Spearman over the grid and the point where it is attained, which
the tables report and nothing here interprets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from classical.network import stationary_distribution
from evaluation.reporting import agreement_with
from evaluation.ties import canonicalize
from quantum.lumped_operator import build_lumped_matrices, unfold_class_scores
from quantum.mechanism.variants import eta_variant


def stationary(G: np.ndarray) -> np.ndarray:
    """``pi`` with ``G pi = pi``, nonnegative, unit sum, on a dense column-stochastic ``G``.

    ``classical.network.stationary_distribution`` is the repository's one dense
    solver: the eigenvector of eigenvalue one with the residual, sign and sum
    assertions ``MultiLayerRanker.pageRank`` makes. This name exists so the
    mechanism modules read as the classical arm; it adds nothing.
    """
    return stationary_distribution(np.asarray(G, dtype=float))


def alpha_eta_grid(
    partitions,
    spec,
    config,
    mech,
    hidden_ids,
    *,
    reference_q: np.ndarray,
    reference_c: np.ndarray,
    evaluation_idx,
) -> list[dict[str, Any]]:
    """The classical stationary score at every ``(alpha, eta)`` of the mechanism grid.

    Each row records the agreement, on ``evaluation_idx``, with the primary
    limiting QPPR (``reference_q``) and with the primary classical score
    (``reference_c``), both full-length borrower vectors, so the table shows
    how far each grid point moved from its own baseline as well.
    """
    k = int(config.primary_k)
    decimals = int(config.tie_tolerance)
    idx = np.asarray(evaluation_idx, dtype=int)
    # Canonicalized once here: the top-k overlap reads ties off exact equality.
    ref_q = canonicalize(np.asarray(reference_q, dtype=float)[idx], decimals)
    ref_c = canonicalize(np.asarray(reference_c, dtype=float)[idx], decimals)
    rows: list[dict[str, Any]] = []
    for eta in mech.eta_grid:
        variant = eta_variant(partitions, spec, config, hidden_ids, eta)
        for alpha in mech.alpha_grid:
            G, _v = build_lumped_matrices(
                variant.nodes, variant.edges, alpha=float(alpha), influence_type=config.influence_type
            )
            ids, scores = unfold_class_scores(variant.nodes, stationary(G))
            if not np.array_equal(ids, np.arange(partitions.frame.height)):
                raise AssertionError("the grid operator does not cover every borrower")
            evaluated = canonicalize(scores[idx], decimals)
            rows.append(
                {
                    "network": spec.name,
                    "alpha": float(alpha),
                    "eta": float(eta),
                    "n_dangling_columns": variant.n_dangling_columns,
                    **agreement_with(evaluated, ref_q, k, decimals=decimals, suffix="vs_primary_q"),
                    **agreement_with(evaluated, ref_c, k, decimals=decimals, suffix="vs_primary_c"),
                }
            )
    return rows
