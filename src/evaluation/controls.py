"""The controls every ranking is reported against: the reference baselines and the reference-permutation placebo.

Baselines
---------
A diffusion score on this network is a function of the borrower's connected
attributes and of which borrowers defaulted, and nothing else. So the question
a referee asks first is whether the walk beats a groupby on those same
attributes, or a count of the blank fields in the record. Every baseline
:func:`build_baselines` produces appears in Table 1, and the blank-field count
in the missingness table, so that comparison is unavoidable and a baseline that
wins is reported as such.

Every baseline is built from the *fit* labels only -- the development
borrowers whose defaults the arm is allowed to see. Under cross-fitting that is
development minus the held-out fold; for the test-partition report it is the
whole development partition. The same ``fit_idx`` is handed to the diffusion
operator, so a baseline never sees a label the walk did not.

Placebo
-------
``B`` reference sets drawn independently of the outcome. Each draw takes
``|D_dev|`` individuals uniformly from the development partition, label
ignored, rebuilds the classes and both operators on that reference set (test
labels stay hidden), routes the lumped operator and scores both arms through
the limit and the stationary distribution. Reported per arm: the true value,
the placebo mean, the 5th and 95th percentiles and
``p = (1 + #(placebo >= true)) / (1 + B)`` for PR-AUC and precision@k, and
the expected top-k Jaccard of each placebo ranking with the true ranking.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from evaluation.context import DatasetContext, score_operator
from evaluation.provenance import write_result
from evaluation.reporting import placebo_p_value, ranking_point_metrics, top_k_jaccard
from fingerprint import UNRECORDED, layer_state_ids
from quantum.mechanism.variants import placebo_variant


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


BASELINE_NAMES = (
    "no_skill",
    "cell_seed_default_rate",
    "single_layer_seed_default_rate",
    "one_hop_seed_count",
    "logit_on_layer_dummies",
)


def state_matrix(frame: pl.DataFrame, layer_columns: dict[str, list[str]]) -> np.ndarray:
    """``(n_borrowers, L)`` matrix of per-layer state ids."""
    return np.column_stack([layer_state_ids(frame, cols) for cols in layer_columns.values()])


def _fit_arrays(y, fit_idx) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    fit_idx = np.asarray(fit_idx, dtype=int)
    if fit_idx.size == 0:
        raise ValueError("fit_idx is empty; a baseline needs visible labels")
    return y, fit_idx


def no_skill(n_borrowers: int) -> np.ndarray:
    """A constant score. Its PR-AUC is the base rate, by construction."""
    return np.zeros(int(n_borrowers), dtype=float)


def cell_seed_default_rate(states: np.ndarray, y, fit_idx) -> np.ndarray:
    """Default rate among fit borrowers sharing the full attribute tuple.

    An unseen tuple falls back to the global fit default rate rather than to
    zero: zero would rank a borrower the fit set never saw below every observed
    cell, which is a statement about coverage, not about risk.
    """
    y, fit_idx = _fit_arrays(y, fit_idx)
    keys = [tuple(row) for row in states.tolist()]
    totals: dict[tuple, list[int]] = {}
    for i in fit_idx:
        cell = totals.setdefault(keys[i], [0, 0])
        cell[0] += int(y[i])
        cell[1] += 1
    fallback = float(y[fit_idx].mean())
    return np.array(
        [totals[key][0] / totals[key][1] if key in totals else fallback for key in keys],
        dtype=float,
    )


def single_layer_seed_default_rate(
    states: np.ndarray, y, fit_idx, layer_names
) -> dict[str, np.ndarray]:
    """One score per layer: the default rate of the borrower's own state there.

    A borrower with no recorded value in a layer takes the global fit rate,
    for the same reason an unseen cell does.
    """
    y, fit_idx = _fit_arrays(y, fit_idx)
    fallback = float(y[fit_idx].mean())
    scores: dict[str, np.ndarray] = {}
    for position, name in enumerate(layer_names):
        column = states[:, position]
        totals: dict[int, list[int]] = {}
        for i in fit_idx:
            state = int(column[i])
            if state == UNRECORDED:
                continue
            cell = totals.setdefault(state, [0, 0])
            cell[0] += int(y[i])
            cell[1] += 1
        scores[name] = np.array(
            [
                totals[int(s)][0] / totals[int(s)][1] if int(s) in totals else fallback
                for s in column
            ],
            dtype=float,
        )
    return scores


def one_hop_seed_count(states: np.ndarray, y, fit_idx) -> np.ndarray:
    """Distinct fit-set defaulters sharing at least one category node.

    The crudest possible diffusion: one hop out to the borrower's category
    nodes and one hop back, with no damping and no iteration. Distinct rather
    than path-counted, by inclusion-exclusion over the layer subsets, so a
    defaulter matching in three layers still counts once. Layers where the
    borrower has no recorded value contribute nothing, which is the same
    no-edge rule the graph uses.
    """
    y, fit_idx = _fit_arrays(y, fit_idx)
    n_layers = states.shape[1]
    seeds = fit_idx[y[fit_idx] == 1]
    if seeds.size == 0:
        return np.zeros(states.shape[0], dtype=float)

    # One count table per non-empty layer subset: joint state tuple -> seeds.
    tables: dict[tuple[int, ...], dict[tuple[int, ...], int]] = {}
    for size in range(1, n_layers + 1):
        for subset in combinations(range(n_layers), size):
            table: dict[tuple[int, ...], int] = {}
            for i in seeds:
                key = tuple(int(states[i, position]) for position in subset)
                if UNRECORDED in key:
                    continue
                table[key] = table.get(key, 0) + 1
            tables[subset] = table

    counts = np.zeros(states.shape[0], dtype=float)
    for borrower in range(states.shape[0]):
        recorded = [p for p in range(n_layers) if int(states[borrower, p]) != UNRECORDED]
        total = 0
        for size in range(1, len(recorded) + 1):
            sign = 1 if size % 2 == 1 else -1
            for subset in combinations(recorded, size):
                key = tuple(int(states[borrower, position]) for position in subset)
                total += sign * tables[subset].get(key, 0)
        counts[borrower] = float(total)
    return counts


def logit_on_layer_dummies(
    frame: pl.DataFrame,
    layer_columns: dict[str, list[str]],
    y,
    fit_idx,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """Logistic regression on the layer one-hot columns alone, fit on the fit set.

    The reference that decides whether the network contributes anything: the
    diffusion score is a function of exactly these dummies, so a walk that does
    not beat this has added structure without adding information.
    """
    y, fit_idx = _fit_arrays(y, fit_idx)
    columns = [c for cols in layer_columns.values() for c in cols]
    design = frame.select(columns).to_numpy().astype(float)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    model.fit(design[fit_idx], y[fit_idx])
    return model.predict_proba(design)[:, 1]


def build_baselines(
    frame: pl.DataFrame,
    layer_columns: dict[str, list[str]],
    y,
    fit_idx,
    *,
    random_state: int = 42,
    include_single_layer: bool = True,
) -> dict[str, np.ndarray]:
    """All five baselines, keyed by name, each scoring every borrower."""
    states = state_matrix(frame, layer_columns)
    scores: dict[str, np.ndarray] = {
        "no_skill": no_skill(frame.height),
        "cell_seed_default_rate": cell_seed_default_rate(states, y, fit_idx),
        "one_hop_seed_count": one_hop_seed_count(states, y, fit_idx),
        "logit_on_layer_dummies": logit_on_layer_dummies(
            frame, layer_columns, y, fit_idx, random_state=random_state
        ),
    }
    if include_single_layer:
        for name, values in single_layer_seed_default_rate(
            states, y, fit_idx, list(layer_columns)
        ).items():
            scores[f"single_layer_seed_default_rate_{name}"] = values
    return scores


CELL_RATE_LABEL = "cell default rate"
BLANK_COUNT_LABEL = "blank-field count"
# Table labels of the other build_baselines outputs (single-layer rates are
# labelled per layer at the call site).
BASELINE_LABELS = {
    "one_hop_seed_count": "one-hop seed count",
    "logit_on_layer_dummies": "logit on layer dummies",
}


def blank_field_count(n_missing_fields) -> np.ndarray:
    """The trivial missingness baseline: the number of blank fields in the record, as a score.

    Uses no label at all, so it needs no fit set; it is the referee's first
    objection and is reported beside the diffusion scores wherever missingness
    is stratified.
    """
    return np.asarray(n_missing_fields, dtype=float)


# ---------------------------------------------------------------------------
# The reference-permutation placebo
# ---------------------------------------------------------------------------


ARMS = ("qppr", "classical")


def placebo_seed_sets(partitions, config) -> list[np.ndarray]:
    """``n_placebo_permutations`` reference sets of size ``|D_dev|`` from the development partition.

    Drawn uniformly, independently of the outcome: "unrelated to default"
    means the selection carries no information about who defaulted, not that
    defaulters are excluded. Draw ``p`` uses ``config.seed_placebo(p)``.
    """
    development = partitions.development_idx
    n_seeds = int(partitions.y_true[development].sum())
    sets = []
    for p in range(int(config.n_placebo_permutations)):
        rng = np.random.default_rng(config.seed_placebo(p))
        sets.append(np.sort(rng.choice(development, size=n_seeds, replace=False)))
    return sets


def _arm_scores(ctx: DatasetContext, scored) -> dict[str, np.ndarray]:
    return {"qppr": ctx.evaluate(scored.quantum.omega_source), "classical": ctx.evaluate(scored.classical)}


def run_placebo(ctx: DatasetContext, network: str) -> dict[str, Any]:
    """Every placebo draw on ``network``: per-draw rows, per-arm summaries and the p-values."""
    k, y = ctx.k, ctx.y_test
    spec = ctx.specs[network]
    true = _arm_scores(ctx, ctx.primary(network))
    true_point = {arm: ranking_point_metrics(true[arm], y, (k,), decimals=ctx.decimals) for arm in ARMS}
    draws: list[dict[str, Any]] = []
    seed_sets = placebo_seed_sets(ctx.partitions, ctx.config)
    for p, seeds in enumerate(seed_sets):
        variant = placebo_variant(ctx.partitions, spec, ctx.config, ctx.hidden_ids, seeds, label=f"placebo_{p}")
        scored = score_operator(variant, ctx.config)
        scores = _arm_scores(ctx, scored)
        for arm in ARMS:
            point = ranking_point_metrics(scores[arm], y, (k,), decimals=ctx.decimals)
            draws.append({"dataset": ctx.name, "network": network, "permutation": p, "arm": arm,
                          "seed": ctx.config.seed_placebo(p), "route": scored.routed.route,
                          "pr_auc": point["pr_auc"], "roc_auc": point["roc_auc"],
                          f"precision_at_{k}": point[f"precision_at_{k}"],
                          f"top_{k}_jaccard_with_true": top_k_jaccard(scores[arm], true[arm], k)})
    summary = []
    for arm in ARMS:
        mine = [d for d in draws if d["arm"] == arm]
        row: dict[str, Any] = {"dataset": ctx.name, "network": network, "arm": arm, "n_perm": len(mine)}
        for metric in ("pr_auc", f"precision_at_{k}"):
            values = np.array([d[metric] for d in mine], dtype=float)
            row[f"true_{metric}"] = true_point[arm][metric]
            row[f"placebo_mean_{metric}"] = float(values.mean()) if values.size else float("nan")
            row[f"placebo_p05_{metric}"] = float(np.percentile(values, 5)) if values.size else float("nan")
            row[f"placebo_p95_{metric}"] = float(np.percentile(values, 95)) if values.size else float("nan")
            row[f"pval_{metric}"] = placebo_p_value(true_point[arm][metric], values)
            row[f"n_placebo_ge_true_{metric}"] = int((values >= true_point[arm][metric]).sum())
        jacc = np.array([d[f"top_{k}_jaccard_with_true"] for d in mine], dtype=float)
        row[f"mean_top_{k}_jaccard_with_true"] = float(jacc.mean()) if jacc.size else float("nan")
        row["routes"] = ",".join(sorted({d["route"] for d in mine}))
        summary.append(row)
    return {"draws": draws, "summary": summary, "seed_sets": {f"placebo_{p}": s.tolist() for p, s in enumerate(seed_sets)}}


def write_placebo_tables(ctx: DatasetContext, result: dict[str, Any], *, source_stage: str) -> dict[str, Path]:
    """``EC_placebo`` (summary) and ``EC_placebo_draws`` (every draw) under the dataset's scope."""
    k = ctx.k
    summary_columns = {"dataset": "dataset name", "network": "network name", "arm": "qppr or classical",
                       "n_perm": "placebo permutations B",
                       f"mean_top_{k}_jaccard_with_true": f"mean expected top-{k} Jaccard of the placebo ranking with the true ranking",
                       "routes": "discriminants the router selected over the draws"}
    for metric in ("pr_auc", f"precision_at_{k}"):
        summary_columns.update({f"true_{metric}": f"{metric} of the true reference set",
                                f"placebo_mean_{metric}": f"mean {metric} over the placebo draws",
                                f"placebo_p05_{metric}": f"5th percentile of {metric} over the draws",
                                f"placebo_p95_{metric}": f"95th percentile of {metric} over the draws",
                                f"pval_{metric}": f"(1 + number of draws with placebo >= true) / (1 + B) for {metric}",
                                f"n_placebo_ge_true_{metric}": f"draws with {metric} at or above the true value"})
    summary = write_result(ctx.name, "EC_placebo", pl.DataFrame(result["summary"], strict=False, infer_schema_length=None),
                           stage=source_stage, columns=summary_columns)
    draws = write_result(
        ctx.name, "EC_placebo_draws", pl.DataFrame(result["draws"], strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "permutation": "draw index (0-based)", "arm": "qppr or classical",
                 "seed": "seed of the draw (config.seed_placebo)", "route": "discriminant the router selected on the draw",
                 "pr_auc": "PR-AUC of the arm on the placebo operator", "roc_auc": "ROC-AUC of the arm on the placebo operator",
                 f"precision_at_{k}": f"precision at {k} of the arm on the placebo operator",
                 f"top_{k}_jaccard_with_true": f"expected top-{k} Jaccard of the placebo ranking with the true ranking"},
    )
    return {"summary": summary, "draws": draws}
