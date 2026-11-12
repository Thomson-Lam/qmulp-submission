"""The downstream robustness check (e-companion only): does either score add anything a credit model lacks?

Logistic regression and XGBoost on the borrower-level covariates (every
field except the network's layer variables), with the cross-fitted score
added: ``covariates_only``, ``plus_classical``, ``plus_qppr_limit``. Training
rows are the development partition scored by K fold operators (held-out fold
and test labels masked, the limit and the stationary score of each fold's
lumped operator through the router); test rows are scored by the test
operator. Every score is transformed to its **within-operator percentile**
among the individuals that operator scored out of influence (the held-out
fold, or the test partition) before modelling, so the train and test columns
share one scale. ``corr(omega_C, omega_Q)`` on the training and on the test
rows is reported as a diagnostic. Deltas are paired bootstrap differences
against ``covariates_only`` on the shared resample matrix.

Feature importance: on a model that also carries the layer dummies, the
permutation importance (and, for XGBoost, the SHAP mean absolute
contribution) of the layer dummies as a share of the total, with the best
layer dummy's rank among all features. None of this is a headline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.context import DatasetContext, Inputs, score_operator
from evaluation.crossfit import fold_assignment
from evaluation.provenance import ci_columns, write_result
from evaluation.reporting import bootstrap_replicates, metrics_with_ci, paired_difference, spearman
from preprocessing.missing_levels import MISSING_INDICATOR_SUFFIX
from quantum.mechanism.variants import primary_variant

MODELS = ("logistic", "xgboost")
FEATURE_SETS = ("covariates_only", "plus_classical", "plus_qppr_limit")
ADDED = {"covariates_only": (), "plus_classical": ("classical",), "plus_qppr_limit": ("qppr_limit",)}
IMPORTANCE_REPEATS = 10
CROSSFIT_IDENTITY_TOL = 1e-8


def covariate_columns(frame: pl.DataFrame, spec, dataset) -> list[str]:
    """Every borrower-level covariate the frame carries, the layer variables excluded."""
    layer_block = [c for layer in spec.layers for c in layer.source_columns if c in frame.columns]
    numeric_present = [c for c in dataset.numeric_covariates if c in frame.columns]
    flags = [c for c in frame.columns if c.endswith(MISSING_INDICATOR_SUFFIX)]
    taken = {*layer_block, *numeric_present, *flags, dataset.label_col}
    other = [c for c in frame.columns if c not in taken]
    return numeric_present + flags + other


def layer_dummy_columns(frame: pl.DataFrame, spec) -> list[str]:
    """The one-hot columns of the network's layers present in the frame (markers included)."""
    return [c for layer in spec.layers for c in layer.source_columns if c in frame.columns]


def percentile_within(scores: np.ndarray, members: np.ndarray) -> np.ndarray:
    """``(rank - 0.5) / n`` of each member among ``members`` (average ranks on ties); NaN elsewhere."""
    out = np.full(scores.size, np.nan)
    values = np.asarray(scores, dtype=float)[members]
    out[members] = (rankdata(values, method="average") - 0.5) / float(values.size)
    return out


def _estimator(model: str, seed: int):
    if model == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))
    from xgboost import XGBClassifier

    return XGBClassifier(random_state=seed, n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
                         colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss", n_jobs=1)


def crossfit_scores(ctx: DatasetContext, data: Inputs, network: str) -> dict[str, Any]:
    """Both arms' cross-fitted development scores through the router, plus the test-operator scores."""
    config, partitions = ctx.config, ctx.partitions
    spec = ctx.specs[network]
    folds = fold_assignment(partitions, config.crossfit_folds, config.seed_folds())
    n = ctx.n_borrowers
    crossfit = {"classical": np.full(n, np.nan), "qppr_limit": np.full(n, np.nan)}
    routes = []
    for fold in range(int(config.crossfit_folds)):
        members = np.flatnonzero(folds == fold)
        hidden = np.concatenate([members, partitions.test_idx])
        scored = score_operator(primary_variant(partitions, spec, config, hidden), config)
        crossfit["classical"][members] = scored.classical[members]
        crossfit["qppr_limit"][members] = scored.quantum.omega_source[members]
        routes.append(scored.routed.route)
    stage03, stage03_folds = data.classical_crossfit(network)
    if not np.array_equal(stage03_folds, folds):
        raise AssertionError("the fold assignment differs from stage 03's; the two arms would not share folds")
    dev = partitions.development_idx
    identity = float(np.max(np.abs(crossfit["classical"][dev] - stage03[dev])))
    if identity > CROSSFIT_IDENTITY_TOL:
        raise AssertionError(f"the lumped fold scores differ from stage 03's cross-fitted classical scores by {identity:.3e}")
    test = {"classical": data.full(network, "classical"), "qppr_limit": data.full(network, "qppr_limit")}
    return {"crossfit": crossfit, "test": test, "fold_index": folds, "routes": routes, "crossfit_identity_max_abs": identity}


def feature_frame(ctx: DatasetContext, data: Inputs, network: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Percentile-transformed features on the development (train) and test rows, and the diagnostics."""
    partitions = ctx.partitions
    dev, test, folds = partitions.development_idx, partitions.test_idx, raw["fold_index"]
    features: dict[str, dict[str, np.ndarray]] = {}
    for name in ("classical", "qppr_limit"):
        train = np.full(ctx.n_borrowers, np.nan)
        for fold in np.unique(folds[dev]):
            members = np.flatnonzero(folds == fold)
            train[members] = percentile_within(raw["crossfit"][name], members)[members]
        test_feature = percentile_within(raw["test"][name], test)
        features[name] = {"train": train[dev], "test": test_feature[test]}
    diagnostics = {
        "corr_train_c_q": spearman(raw["crossfit"]["classical"][dev], raw["crossfit"]["qppr_limit"][dev], decimals=ctx.decimals),
        "corr_test_c_q": spearman(raw["test"]["classical"][test], raw["test"]["qppr_limit"][test], decimals=ctx.decimals),
        "corr_train_c_q_percentile": spearman(features["classical"]["train"], features["qppr_limit"]["train"], decimals=ctx.decimals),
        "fold_routes": list(raw["routes"]), "crossfit_identity_max_abs": raw["crossfit_identity_max_abs"],
    }
    return {"features": features, "diagnostics": diagnostics}


def evaluate_feature_sets(ctx: DatasetContext, data: Inputs, network: str, features: dict[str, dict[str, np.ndarray]],
                          models=MODELS) -> list[dict[str, Any]]:
    """Every (model, feature set): metrics with intervals and paired deltas against covariates-only."""
    config, partitions, frame = ctx.config, ctx.partitions, ctx.partitions.frame
    spec = ctx.specs[network]
    dev, test = partitions.development_idx, partitions.test_idx
    y_train, y_test = partitions.y_true[dev], partitions.y_true[test]
    columns = covariate_columns(frame, spec, ctx.dataset)
    matrix = frame.select(columns).to_numpy().astype(float)
    base_train, base_test = matrix[dev], matrix[test]
    k = ctx.k
    rows: list[dict[str, Any]] = []
    for model in models:
        reference_reps = reference_point = None
        for feature_set in FEATURE_SETS:
            extra_train = [features[name]["train"] for name in ADDED[feature_set]]
            extra_test = [features[name]["test"] for name in ADDED[feature_set]]
            x_train = np.column_stack([base_train, *extra_train]) if extra_train else base_train
            x_test = np.column_stack([base_test, *extra_test]) if extra_test else base_test
            estimator = _estimator(model, config.seed_downstream(model))
            estimator.fit(x_train, y_train)
            probability = estimator.predict_proba(x_test)[:, 1]
            row: dict[str, Any] = {"dataset": ctx.name, "network": network, "model": model, "features": feature_set,
                                   "n_covariates": len(columns), "n_features": int(x_train.shape[1]),
                                   "crossfit_folds": int(config.crossfit_folds)}
            row.update(metrics_with_ci(probability, y_test, data.boot, k=k, decimals=config.tie_tolerance))
            reps = bootstrap_replicates(probability, y_test, data.boot, k=k, decimals=config.tie_tolerance)
            if feature_set == "covariates_only":
                reference_reps, reference_point = reps, {"pr_auc": row["pr_auc"], "roc_auc": row["roc_auc"]}
            for metric in ("pr_auc", "roc_auc"):
                diff = paired_difference(reps[metric], reference_reps[metric], row[metric], reference_point[metric])
                row[f"delta_{metric}"] = diff["observed"]
                row[f"delta_{metric}_ci_lo"], row[f"delta_{metric}_ci_hi"] = diff["ci_lo"], diff["ci_hi"]
                for sign in ("pos", "zero", "neg"):
                    row[f"delta_{metric}_p_delta_{sign}"] = diff[f"p_delta_{sign}"]
            rows.append(row)
    return rows


def feature_importance(ctx: DatasetContext, network: str, models=MODELS) -> list[dict[str, Any]]:
    """Permutation importance (and SHAP share for XGBoost) of the layer dummies among all covariates."""
    config, partitions, frame = ctx.config, ctx.partitions, ctx.partitions.frame
    spec = ctx.specs[network]
    dev, test = partitions.development_idx, partitions.test_idx
    columns = covariate_columns(frame, spec, ctx.dataset)
    dummies = layer_dummy_columns(frame, spec)
    names = columns + dummies
    matrix = frame.select(names).to_numpy().astype(float)
    y_train, y_test = partitions.y_true[dev], partitions.y_true[test]
    is_dummy = np.array([c in dummies for c in names])
    rows = []
    for model in models:
        estimator = _estimator(model, config.seed_downstream(model))
        estimator.fit(matrix[dev], y_train)
        perm = permutation_importance(estimator, matrix[test], y_test, scoring="average_precision",
                                      n_repeats=IMPORTANCE_REPEATS, random_state=config.seed_downstream(f"{model}_perm"))
        importance = np.maximum(perm.importances_mean, 0.0)
        total = float(importance.sum())
        share = float(importance[is_dummy].sum() / total) if total > 0 else float("nan")
        order = rankdata(-importance, method="min")
        row = {"dataset": ctx.name, "network": network, "model": model, "method": "permutation",
               "layerdummy_importance_share": share, "layerdummy_importance_rank": int(order[is_dummy].min()),
               "n_covariates": len(columns), "n_layer_dummies": len(dummies), "n_features": len(names),
               "best_layer_dummy": names[int(np.flatnonzero(is_dummy)[np.argmax(importance[is_dummy])])]}
        if model == "xgboost":
            import xgboost

            contributions = estimator.get_booster().predict(xgboost.DMatrix(matrix[test]), pred_contribs=True)
            shap = np.abs(contributions[:, :-1]).mean(axis=0)
            total_shap = float(shap.sum())
            row["shap_layerdummy_share"] = float(shap[is_dummy].sum() / total_shap) if total_shap > 0 else float("nan")
            row["shap_layerdummy_rank"] = int(rankdata(-shap, method="min")[is_dummy].min())
        rows.append(row)
    return rows


def downstream_bound(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """``dprauc_upper_max``: the largest upper interval end of the PR-AUC delta over models and score columns."""
    deltas = [r for r in rows if r["features"] != "covariates_only"]
    return {"dataset": rows[0]["dataset"] if rows else None, "network": rows[0]["network"] if rows else None,
            "dprauc_upper_max": float(max(r["delta_pr_auc_ci_hi"] for r in deltas)) if deltas else float("nan"),
            "droc_upper_max": float(max(r["delta_roc_auc_ci_hi"] for r in deltas)) if deltas else float("nan")}


def write_downstream_tables(ctx: DatasetContext, rows, importance, diagnostics, *, inputs, source_stage: str) -> dict[str, Any]:
    """``EC_downstream``, ``EC_feature_importance`` and ``downstream_bound`` under the dataset's scope."""
    scope = ctx.name
    layers = "/".join(ctx.dataset.layer_order)
    main_columns = {"dataset": "dataset name", "network": "network name", "model": "logistic or xgboost",
                    "features": "covariates_only, plus_classical or plus_qppr_limit",
                    "n_covariates": f"borrower-level covariates (every field except the {layers} layer variables)",
                    "n_features": "columns of the feature matrix of this feature set",
                    **ci_columns("pr_auc", "PR-AUC on the test partition"), **ci_columns("roc_auc", "ROC-AUC on the test partition"),
                    **ci_columns("delta_pr_auc", "paired PR-AUC difference against covariates_only on the shared replicates"),
                    **ci_columns("delta_roc_auc", "paired ROC-AUC difference against covariates_only on the shared replicates")}
    for metric in ("pr_auc", "roc_auc"):
        for sign in ("pos", "zero", "neg"):
            main_columns[f"delta_{metric}_p_delta_{sign}"] = f"share of shared replicates on which the paired {metric} difference is {sign}"
    main = write_result(scope, "EC_downstream", pl.DataFrame(list(rows), strict=False, infer_schema_length=None),
                        stage=source_stage, columns=main_columns)
    importance_table = write_result(
        scope, "EC_feature_importance", pl.DataFrame(list(importance), strict=False, infer_schema_length=None), stage=source_stage,
        columns={"dataset": "dataset name", "network": "network name", "model": "logistic or xgboost", "method": "permutation",
                 "layerdummy_importance_share": f"permutation importance of the {layers} dummies as a share of the total "
                                                f"(average precision on the test partition, {IMPORTANCE_REPEATS} repeats)",
                 "layerdummy_importance_rank": "rank of the best layer dummy among all features (covariates plus layer dummies)",
                 "n_covariates": "borrower-level covariates in the covariates-only model",
                 "n_layer_dummies": "layer one-hot columns (markers included)",
                 "n_features": "covariates plus layer dummies: the denominator of layerdummy_importance_rank",
                 "best_layer_dummy": "the layer dummy with the largest permutation importance",
                 "shap_layerdummy_share": "XGBoost SHAP mean absolute contribution of the layer dummies as a share of the total",
                 "shap_layerdummy_rank": "rank of the best layer dummy by SHAP share among all features"},
    )
    bound = downstream_bound(list(rows))
    bound_path = write_result(scope, "downstream_bound", pl.DataFrame([bound]), stage=source_stage,
                              columns={"dataset": "dataset name", "network": "network name",
                                       "dprauc_upper_max": "largest upper interval end of the paired PR-AUC delta over models and score columns",
                                       "droc_upper_max": "largest upper interval end of the paired ROC-AUC delta"})
    return {"downstream": main, "importance": importance_table, "bound": bound_path, "bound_row": bound, "diagnostics": diagnostics}
