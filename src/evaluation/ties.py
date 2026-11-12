"""Tie handling, applied before every ranking metric.

Two borrowers connected to the same category nodes in every layer are related
by a graph automorphism, so any structural score gives them exactly equal
values -- up to floating-point noise in the eigensolver or the walk. Sorting
such a vector orders them by whatever the solver's last bits happened to be,
and a top-k metric then reports an artifact of the sort.

The effect is not small. Measured on the two-layer artifacts, the within-class
score range was 4.4e-17 (classical) and 3.3e-18 (quantum) against a score span
of about 2e-3, and 15 borrowers sat exactly on the rank-100 boundary. Restoring
the ties moved quantum PR-AUC from 0.29084 to 0.27590 against a classical
0.27347: most of the reported gap was rounding in the fourteenth digit.

:func:`canonicalize` rounds to a fixed number of decimals so equal scores
compare equal; every tie-exact top-k expectation is then built on the one
:func:`boundary` (``evaluation.reporting``), never on a random tie-break.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def canonicalize(scores, decimals: int) -> np.ndarray:
    """Round a score vector so automorphic borrowers compare exactly equal.

    ``decimals`` is ``ExperimentConfig.tie_tolerance``. It must be coarse
    enough to swallow solver noise and fine enough to keep genuinely different
    levels apart; the gap between those two is four orders of magnitude here,
    so the choice is not delicate.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"scores must be 1-dimensional; got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("scores contain NaN or infinite values")
    return np.round(arr, int(decimals))


def distinct_levels(scores, decimals: int) -> int:
    """How many different values the score actually takes.

    The single number that says whether a schema can discriminate at all: it is
    bounded above by the network's automorphism structure, not by the sample
    size.
    """
    return int(np.unique(canonicalize(scores, decimals)).size)


def boundary(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Indices strictly above the rank-k cut, indices tied on it, and slots left.

    The one definition of the rank-k boundary; ``evaluation.reporting`` builds
    every tie-exact top-k expectation on it.
    """
    order = np.sort(scores)[::-1]
    threshold = order[k - 1]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    return above, tied, int(k - above.size)



def tie_report(scores, decimals: int, top_k) -> dict[str, Any]:
    """Distinct levels plus the boundary composition at each reported ``k``."""
    canonical = canonicalize(scores, decimals)
    report: dict[str, Any] = {
        "n_scored": int(canonical.size),
        "distinct_levels": int(np.unique(canonical).size),
        "decimals": int(decimals),
    }
    for k in top_k:
        k_eff = min(int(k), canonical.size)
        above, tied, slots = boundary(canonical, k_eff)
        report[f"k_{int(k)}"] = {
            "n_strictly_above": int(above.size),
            "n_tied_at_boundary": int(tied.size),
            "n_contested_slots": max(0, slots),
        }
    return report
