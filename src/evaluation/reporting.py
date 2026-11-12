"""The one implementation of every reported quantity, and the uncertainty around it.

Every ranking-agreement and distance measure the paper reports is defined
here exactly once: Spearman (average ranks), Kendall tau-b, the expected
top-k overlap and its Jaccard, precision@k and lift@k, PR-AUC and ROC-AUC,
the summed L1 (``l1_sum``), the mean absolute difference (``mean_abs_diff``)
and the total variation (``total_variation = l1_sum / 2``). Every other module
imports these; a second implementation of any of them anywhere is a defect.
Every input is rounded at the tie tolerance before a rank is formed, so
automorphic borrowers compare equal on both sides of every measure.

Every uncertainty comes from the one shared resample matrix written by stage
01 (:func:`fingerprint.bootstrap_indices`), so intervals are paired across arms
by construction. Comparative claims use :func:`paired_difference` on those
common replicates, never two marginal intervals.

Top-k metrics are exact expectations under uniform tie-breaking of the
borrowers sitting on the rank-k boundary (``evaluation.ties.boundary``). The
result writer (``write_result``, CSV only) is ``evaluation.provenance.write_result``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.ties import boundary, canonicalize

# ---------------------------------------------------------------------------
# Tie-exact ranking metrics
# ---------------------------------------------------------------------------


def inclusion_probability(scores: np.ndarray, k: int) -> np.ndarray:
    """P(borrower is in the top k) under uniform tie-breaking on the boundary.

    Borrowers strictly above the rank-k score are in with probability one;
    the ``h`` slots left on the boundary are shared uniformly by the ``g``
    borrowers tied there, ``(k - h) / g`` each.
    """
    scores = np.asarray(scores, dtype=float)
    k = min(int(k), scores.size)
    if k <= 0:
        raise ValueError("k must be positive")
    above, tied, slots = boundary(scores, k)
    prob = np.zeros(scores.size, dtype=float)
    prob[above] = 1.0
    if slots > 0 and tied.size:
        prob[tied] = slots / float(tied.size)
    return prob


def expected_precision_at_k(scores, y, k: int) -> tuple[float, float]:
    """``(precision@k, lift@k)`` as exact expectations over tie-breakings."""
    y = np.asarray(y, dtype=float)
    prob = inclusion_probability(scores, k)
    k_eff = min(int(k), y.size)
    precision = float((prob * y).sum() / k_eff)
    base_rate = float(y.mean())
    lift = precision / base_rate if base_rate > 0 else float("nan")
    return precision, lift


def expected_top_k_overlap(a, b, k: int) -> float:
    """Expected top-k overlap ``(1/k) sum_i r_i(a) r_i(b)`` under independent tie-breaking."""
    pa = inclusion_probability(a, k)
    pb = inclusion_probability(b, k)
    k_eff = min(int(k), pa.size)
    return float((pa * pb).sum() / k_eff)


def top_k_jaccard(a, b, k: int) -> float:
    """Jaccard of the expected top-k sets, ``n / (2k - n)`` with ``n = k * overlap``."""
    overlap = expected_top_k_overlap(a, b, k)
    return float(overlap / (2.0 - overlap)) if overlap < 2.0 else float("nan")


def expected_share(scores, flag, k: int) -> float:
    """Share of the expected top-k carrying ``flag`` under the same tie rule."""
    scores = np.asarray(scores, dtype=float)
    k_eff = min(int(k), scores.size)
    return float((inclusion_probability(scores, k_eff) * np.asarray(flag, dtype=float)).sum() / k_eff)


def _paired(a, b, decimals: int) -> tuple[np.ndarray, np.ndarray]:
    a = canonicalize(a, decimals)
    b = canonicalize(b, decimals)
    if a.shape != b.shape:
        raise ValueError(f"score vectors differ in length: {a.shape} vs {b.shape}")
    return a, b


def spearman(a, b, *, decimals: int = 12) -> float:
    """Spearman's rho between two canonicalized score vectors; ties share the mean rank.

    Both inputs pass through the same rounding as every other agreement
    measure, so scores that differ only below ``decimals`` are ties on both
    sides and a self-comparison is exactly one. A constant vector has no
    ranking and returns NaN.
    """
    a, b = _paired(a, b, decimals)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    if np.array_equal(a, b):
        return 1.0
    return float(spearmanr(a, b).statistic)


def kendall_tau(a, b, *, decimals: int = 12) -> float:
    """Kendall's tau-b between two canonicalized score vectors; ties are handled by tau-b."""
    a, b = _paired(a, b, decimals)
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(kendalltau(a, b).statistic)


def l1_sum(a, b, *, decimals: int = 12) -> float:
    """``sum |a - b|`` over the compared elements: the L1 distance between two distributions."""
    a, b = _paired(a, b, decimals)
    return float(np.abs(a - b).sum())


def mean_abs_diff(a, b, *, decimals: int = 12) -> float:
    """``sum |a - b| / n``: the summed L1 per compared element, comparable across sizes."""
    a, b = _paired(a, b, decimals)
    return float(np.abs(a - b).sum() / a.size)


def total_variation(a, b, *, decimals: int = 12) -> float:
    """``sum |a - b| / 2``: the total variation distance between two distributions."""
    return 0.5 * l1_sum(a, b, decimals=decimals)


def agreement_with(
    scores, reference, k: int, *, decimals: int = 12, suffix: str
) -> dict[str, float]:
    """Every ranking-agreement measure of ``scores`` against ``reference``, keyed ``<name>_<suffix>``.

    Both vectors are rounded at ``decimals`` before any measure, the top-k
    overlap included, so a boundary tie is a tie on both sides.
    """
    a, b = _paired(scores, reference, decimals)
    return {
        f"spearman_{suffix}": spearman(a, b, decimals=decimals),
        f"kendall_{suffix}": kendall_tau(a, b, decimals=decimals),
        f"normalized_l1_{suffix}": mean_abs_diff(a, b, decimals=decimals),
        f"top_{int(k)}_overlap_{suffix}": expected_top_k_overlap(a, b, k),
    }


def ranking_point_metrics(
    scores, y, k_values: Sequence[int], *, decimals: int = 12
) -> dict[str, float]:
    """PR-AUC, ROC-AUC, and P@k / Lift@k for every k, on canonicalized scores."""
    y = np.asarray(y, dtype=int)
    s = canonicalize(scores, decimals)
    if len(np.unique(y)) < 2:
        raise ValueError("ranking metrics need both outcome classes")
    out = {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "distinct_levels": int(np.unique(s).size),
    }
    for k in k_values:
        precision, lift = expected_precision_at_k(s, y, int(k))
        out[f"precision_at_{int(k)}"] = precision
        out[f"lift_at_{int(k)}"] = lift
    return out


def no_skill_metrics(y, k_values: Sequence[int]) -> dict[str, float]:
    """The metrics a constant score earns: base rate, one half, base rate, one."""
    y = np.asarray(y, dtype=int)
    base = float(y.mean())
    out = {"pr_auc": base, "roc_auc": 0.5, "distinct_levels": 1}
    for k in k_values:
        out[f"precision_at_{int(k)}"] = base
        out[f"lift_at_{int(k)}"] = 1.0
    return out


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray, int], float]] = {
    "pr_auc": lambda s, y, k: float(average_precision_score(y, s)),
    "roc_auc": lambda s, y, k: float(roc_auc_score(y, s)),
    "precision_at_k": lambda s, y, k: expected_precision_at_k(s, y, k)[0],
    "lift_at_k": lambda s, y, k: expected_precision_at_k(s, y, k)[1],
}


def bootstrap_replicates(
    scores, y, indices: np.ndarray, *, k: int, decimals: int = 12
) -> dict[str, np.ndarray]:
    """Per-metric replicate arrays on the shared resample matrix."""
    y = np.asarray(y, dtype=int)
    s = canonicalize(scores, decimals)
    indices = np.asarray(indices, dtype=int)
    reps = indices.shape[0]
    out = {name: np.full(reps, np.nan) for name in METRIC_FUNCTIONS}
    for rep in range(reps):
        idx = indices[rep]
        y_rep = y[idx]
        if len(np.unique(y_rep)) < 2:
            continue
        s_rep = s[idx]
        for name, function in METRIC_FUNCTIONS.items():
            out[name][rep] = function(s_rep, y_rep, int(k))
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles of the finite replicates."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def metrics_with_ci(
    scores, y, indices: np.ndarray, *, k: int, decimals: int = 12
) -> dict[str, Any]:
    """Point estimates plus shared-bootstrap 95% intervals, flat for a CSV row."""
    point = ranking_point_metrics(scores, y, (k,), decimals=decimals)
    reps = bootstrap_replicates(scores, y, indices, k=k, decimals=decimals)
    row: dict[str, Any] = {
        "pr_auc": point["pr_auc"],
        "roc_auc": point["roc_auc"],
        f"precision_at_{k}": point[f"precision_at_{k}"],
        f"lift_at_{k}": point[f"lift_at_{k}"],
        "distinct_levels": point["distinct_levels"],
    }
    for name, values in reps.items():
        lo, hi = interval(values)
        label = name.replace("_k", f"_{k}")
        row[f"{label}_ci_lo"] = lo
        row[f"{label}_ci_hi"] = hi
    return row


def paired_difference(
    rep_a: np.ndarray, rep_b: np.ndarray, obs_a: float, obs_b: float
) -> dict[str, float]:
    """Interval and the sign probabilities of ``a - b`` on common replicates.

    ``p_delta_pos``, ``p_delta_zero`` and ``p_delta_neg`` are the shares of
    finite replicates with a positive, exactly zero and negative difference;
    they sum to one, so a tie-heavy comparison is visible as mass at zero
    rather than hidden in a strict inequality.
    """
    diff = np.asarray(rep_a, dtype=float) - np.asarray(rep_b, dtype=float)
    finite = diff[np.isfinite(diff)]
    lo, hi = interval(diff)
    n = float(finite.size)
    return {
        "observed": float(obs_a - obs_b),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_delta_pos": float((finite > 0).sum() / n) if n else float("nan"),
        "p_delta_zero": float((finite == 0).sum() / n) if n else float("nan"),
        "p_delta_neg": float((finite < 0).sum() / n) if n else float("nan"),
        "n_replicates": int(n),
    }


def placebo_p_value(true_value: float, placebo_values) -> float:
    """``(1 + #{placebo >= true}) / (1 + B)``: the permutation p-value with the +1 correction."""
    values = np.asarray(placebo_values, dtype=float)
    values = values[np.isfinite(values)]
    return float((1 + int((values >= float(true_value)).sum())) / (1 + values.size))
