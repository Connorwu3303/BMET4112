"""
Aim 2 model 3: tuned XGBoost.

Gradient boosted trees, with the same folds and seed as the other two models.
Stage 1 picks the imbalance arm at a fixed config (per-fold scale_pos_weight,
SMOTETomek or nothing), stage 2 runs a two-step search inside each training
fold (TwoStageSearch in ml_common), and stage 3 fits the three feature
specifications.

Features aren't standardised since tree splits only depend on value order.
XGBoost handles missing values itself, but median imputation is still done in
the pipeline so all three models get identical inputs. Permutation importance
on held-out folds is the main ranking; gain importance is computed on training
data and is shown only for contrast.

Outputs (processed/aim2/results/, with the --tag suffix if given):
    xgb_summary.txt, xgb_fold_metrics.csv, xgb_importance.csv,
    xgb_performance.png, xgb_specifications.png, xgb_importance.png,
    xgb_results.json, xgb_oof_scores.npz

Usage:
    python3 aim2/train_xgboost.py
    python3 aim2/train_xgboost.py --quick
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_common import (
    AIM2_DIR,
    N_INNER_FOLDS,
    N_OUTER_FOLDS,
    N_OUTER_REPEATS,
    PRIMARY_OUTCOME,
    RANDOM_STATE,
    RESULTS_DIR,
    SPECIFICATIONS,
    TwoStageSearch,
    bootstrap_auc_ci,
    build_pipeline,
    clean_params,
    delong_test,
    ensure_results_dir,
    evaluate_predictions,
    load_dataset,
    nested_cv,
    plot_model_performance,
    plot_specification_comparison,
    summarise_nested_cv,
)
from xgb_support import AutoWeightXGBClassifier

ensure_results_dir()

PRIMARY_SPEC = "obm_plus_confounders"

# imbalance arms: (label, resampling strategy, weighting)
IMBALANCE_ARMS = [
    ("scale_pos_weight_auto", "none", "auto"),
    ("smotetomek", "smotetomek", None),
    ("none", "none", None),
]

# settings shared by every XGBoost estimator here
#   tree_method "hist"  histogram split finding, fast and the recommended
#                       default at this data size
#   n_jobs 1            the search already runs candidates in parallel, so each
#                       fit is single-threaded to avoid oversubscribing the
#                       8 cores
#   eval_metric         set so XGBoost doesn't print its default-metric notice
BASE_XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbosity=0,
)

# fixed config for the stage 1 imbalance comparison: moderate depth, modest
# learning rate, row and column subsampling. Standard starting values, picked
# before seeing any results
STAGE1_CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def broad_search_space(quick=False):
    """
    Random search space over nine parameters.

    learning_rate is log-uniform since its effect is multiplicative, and the
    L1/L2 leaf penalties are too since their ranges cover orders of magnitude.
    Depth stays at 2 to 6 since boosting works best with shallow trees, and
    depth 2 already gives pairwise interactions the linear model can't
    represent. min_child_weight (minimum summed Hessian per leaf) is the main
    guard against leaves built on a few participants.
    """
    if quick:
        return {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 4],
            "model__min_child_weight": [1, 5],
        }
    return {
        "model__n_estimators": [100, 200, 300, 500, 800],
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": [2, 3, 4, 5, 6],
        "model__min_child_weight": [1, 3, 5, 10, 20],
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.4, 0.6),
        "model__gamma": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        "model__reg_alpha": loguniform(1e-3, 10.0),
        "model__reg_lambda": loguniform(0.1, 20.0),
    }


def focused_grid(centre):
    """
    Small grid around one training fold's broad-search winner.

    Learning rate and number of rounds are tried at half, equal and double the
    winning value, depth at minus one, equal and plus one, and the other six
    stay fixed. The centre isn't rounded, so the grid always contains the broad
    winner, and with fixed seeds and single-threaded fits the focused step
    can't score worse than the broad one on the inner folds.
    """
    lr = float(centre.get("model__learning_rate", 0.05))
    n = int(centre.get("model__n_estimators", 300))
    d = int(centre.get("model__max_depth", 3))
    grid = {
        "model__learning_rate": sorted({min(lr * 0.5, 0.5), lr, min(lr * 2.0, 0.5)}),
        "model__n_estimators": sorted({max(50, n // 2), n, min(1600, n * 2)}),
        "model__max_depth": sorted({max(1, d - 1), d, d + 1}),
    }
    for key in ("model__min_child_weight", "model__subsample",
                "model__colsample_bytree", "model__gamma",
                "model__reg_alpha", "model__reg_lambda"):
        if key in centre:
            grid[key] = [centre[key]]
    return grid


def make_estimator(weighting):
    """Plain XGBoost, or the variant that sets scale_pos_weight per fold."""
    if weighting == "auto":
        return AutoWeightXGBClassifier(**BASE_XGB_PARAMS)
    return XGBClassifier(**BASE_XGB_PARAMS)


def _inner_cv():
    return StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                           random_state=RANDOM_STATE)


def make_fixed_factory(imbalance, weighting, config):
    """Factory for a pipeline at one fixed configuration (stage 1)."""
    def factory():
        estimator = make_estimator(weighting)
        estimator.set_params(**config)
        pipeline = build_pipeline(estimator, imbalance=imbalance, scale=False,
                                  random_state=RANDOM_STATE)
        # one-candidate grid so the interface matches the real searches and
        # nested_cv doesn't need a special case
        return GridSearchCV(
            pipeline, param_grid={"model__n_estimators": [config["n_estimators"]]},
            scoring="roc_auc", cv=_inner_cv(), n_jobs=-1, refit=True,
            error_score=np.nan,
        )
    return factory


def make_random_search_factory(imbalance, weighting, space, n_iter):
    def factory():
        pipeline = build_pipeline(make_estimator(weighting), imbalance=imbalance,
                                  scale=False, random_state=RANDOM_STATE)
        return RandomizedSearchCV(
            pipeline, param_distributions=space, n_iter=n_iter,
            scoring="roc_auc", cv=_inner_cv(), n_jobs=-1, refit=True,
            random_state=RANDOM_STATE, error_score=np.nan,
        )
    return factory


def make_grid_factory(imbalance, weighting, grid):
    def factory():
        pipeline = build_pipeline(make_estimator(weighting), imbalance=imbalance,
                                  scale=False, random_state=RANDOM_STATE)
        return GridSearchCV(
            pipeline, param_grid=grid, scoring="roc_auc", cv=_inner_cv(),
            n_jobs=-1, refit=True, error_score=np.nan,
        )
    return factory


def outer_fold_permutation_importance(X, y, factory, features,
                                      n_repeats_perm=5, n_splits=5,
                                      random_state=RANDOM_STATE):
    """
    Permutation importance averaged over held-out folds.

    In each fold the full search is fitted on the training part, then each
    feature is shuffled in the test part and the drop in AUC recorded. Because
    it's measured on unseen data, it shows how much the model relies on a
    feature to predict, without crediting anything it memorised.
    """
    X_values = X.to_numpy(dtype=float)
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    importances = []
    for train_idx, test_idx in skf.split(X_values, y):
        search = factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_values[train_idx], y[train_idx])
        result = permutation_importance(
            search.best_estimator_, X_values[test_idx], y[test_idx],
            scoring="roc_auc", n_repeats=n_repeats_perm,
            random_state=random_state, n_jobs=-1,
        )
        importances.append(result.importances_mean)
    matrix = np.vstack(importances)
    return pd.DataFrame({
        "feature": features,
        "perm_importance_mean": matrix.mean(axis=0),
        "perm_importance_sd": matrix.std(axis=0),
        "perm_folds_positive": (matrix > 0).sum(axis=0),
        "n_folds": matrix.shape[0],
    })


def representative_value(series):
    """
    One representative value of a hyperparameter across the outer folds.

    Discrete settings (number of trees, depth, gamma) use the most common
    value. Continuous ones sampled from a distribution (learning rate,
    subsample fractions, penalties) almost never repeat exactly between folds,
    so the mode would just be some arbitrary fold's value. Those use the median.
    """
    values = series.dropna()
    if values.empty:
        return None
    floats = [float(v) for v in values if isinstance(v, (float, np.floating))]
    continuous = (len(floats) == len(values)
                  and not all(v.is_integer() for v in floats)
                  and values.nunique() > len(values) // 2)
    if continuous:
        return float(np.median(np.asarray(floats)))
    return values.mode().iloc[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smaller search and one repeat, for a fast check")
    parser.add_argument("--outcome", default=PRIMARY_OUTCOME)
    parser.add_argument(
        "--specifications", default=None,
        help="comma separated specification names from feature_manifest.json",
    )
    parser.add_argument("--tag", default="",
                        help="suffix for every output filename")
    parser.add_argument("--n-iter", type=int, default=None,
                        help="broad randomised search candidates per fold")
    args = parser.parse_args()

    n_repeats = 1 if args.quick else N_OUTER_REPEATS
    specs = (tuple(x.strip() for x in args.specifications.split(","))
             if args.specifications else SPECIFICATIONS)
    tag = f"_{args.tag}" if args.tag else ""
    primary_spec = PRIMARY_SPEC if PRIMARY_SPEC in specs else specs[-1]
    n_iter = args.n_iter or (6 if args.quick else 30)
    space = broad_search_space(quick=args.quick)

    report = []

    def emit(text=""):
        print(text)
        report.append(text)

    emit("=" * 78)
    emit("AIM 2 MODEL 3 - OPTIMISED XGBOOST")
    emit("=" * 78)
    started = time.time()

    X, y, features, meta = load_dataset(primary_spec, outcome=args.outcome)
    emit(f"\nOutcome                : {args.outcome}")
    emit(f"Participants           : {meta['n_participants']}")
    emit(f"  excluded, outcome undefined (prevalent CVD at baseline): "
         f"{meta['n_dropped_outcome_undefined']}")
    emit(f"Events                 : {meta['n_events']} "
         f"({meta['event_rate']:.2%})")
    emit(f"Features ({primary_spec}): {meta['n_features']}")
    emit(f"Events per variable    : "
         f"{meta['n_events'] / max(meta['n_features'], 1):.2f}")
    emit(f"\nCross validation       : {n_repeats} repeat(s) x "
         f"{N_OUTER_FOLDS} outer folds, {N_INNER_FOLDS} inner folds")
    emit(f"Broad search           : {n_iter} candidates per outer fold")
    emit(f"Random seed            : {RANDOM_STATE}")
    emit("\nFeatures are not standardised (tree splits depend only on value")
    emit("ordering). Median imputation is applied inside the pipeline so all")
    emit("three models receive identical inputs.")

    results_json = {"outcome": args.outcome,
                    "meta": {k: v for k, v in meta.items() if k != "pid"}}

    # --- stage 1: imbalance strategy at a fixed config ---
    emit("\n" + "=" * 78)
    emit("STAGE 1 - CLASS IMBALANCE STRATEGY")
    emit("=" * 78)
    emit(f"\n  Compared at a fixed configuration ({STAGE1_CONFIG}) so the")
    emit("  imbalance question is isolated from hyperparameter search noise.")

    arm_metrics = {}
    for label, imbalance, weighting in IMBALANCE_ARMS:
        emit(f"\n  --- arm: {label} ---")
        factory = make_fixed_factory(imbalance, weighting, STAGE1_CONFIG)
        result = nested_cv(X, y, factory, n_outer=N_OUTER_FOLDS,
                           n_repeats=n_repeats, random_state=RANDOM_STATE,
                           verbose=True)
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        metrics = evaluate_predictions(y[ok], scores[ok])
        metrics["mean_inner_score"] = float(result["folds"]["inner_best_score"].mean())
        arm_metrics[label] = metrics
        emit(f"    pooled AUC {metrics['auc_roc']:.4f}  "
             f"AUPRC {metrics['auprc']:.4f}  Brier {metrics['brier']:.4f}")

    emit("\n  Arm comparison:")
    emit(f"    {'arm':<26s} {'AUC':>8s} {'AUPRC':>8s} {'Brier':>8s} "
         f"{'inner':>8s}")
    for label, metrics in arm_metrics.items():
        emit(f"    {label:<26s} {metrics['auc_roc']:>8.4f} "
             f"{metrics['auprc']:>8.4f} {metrics['brier']:>8.4f} "
             f"{metrics['mean_inner_score']:>8.4f}")

    best_arm = max(arm_metrics, key=lambda k: arm_metrics[k]["auc_roc"])
    spread = (max(m["auc_roc"] for m in arm_metrics.values())
              - min(m["auc_roc"] for m in arm_metrics.values()))
    emit(f"\n  Best arm by pooled out-of-fold AUC: {best_arm}")
    emit(f"  Spread in AUC across arms: {spread:.4f}. The arm is chosen on")
    emit("  out-of-fold AUC, as for the other two models, so the headline")
    emit("  carries selection optimism of at most roughly this spread.")
    best_brier = min(arm_metrics, key=lambda k: arm_metrics[k]["brier"])
    emit(f"  Best arm by calibration (lowest Brier): {best_brier}")
    results_json["stage1_imbalance_arms"] = arm_metrics
    results_json["stage1_best_arm"] = best_arm
    results_json["stage1_auc_spread"] = spread

    _, chosen_imbalance, chosen_weighting = next(
        a for a in IMBALANCE_ARMS if a[0] == best_arm
    )

    # --- stage 2: two-step search inside every outer training fold ---
    emit("\n" + "=" * 78)
    emit("STAGE 2 - HYPERPARAMETER OPTIMISATION")
    emit("=" * 78)
    emit(f"\n  Inside every outer training fold: a broad randomised search of")
    emit(f"  {n_iter} candidates over nine parameters, then a focused grid on")
    emit("  learning rate, number of trees and depth around THAT fold's")
    emit("  winner. Both stages and the choice see training data only.")

    def two_stage_factory():
        return TwoStageSearch(
            broad_factory=make_random_search_factory(
                chosen_imbalance, chosen_weighting, space, n_iter),
            grid_builder=focused_grid,
            grid_factory=lambda grid: make_grid_factory(
                chosen_imbalance, chosen_weighting, grid)(),
        )

    t0 = time.time()
    best_result = nested_cv(X, y, two_stage_factory, n_outer=N_OUTER_FOLDS,
                            n_repeats=n_repeats, random_state=RANDOM_STATE,
                            verbose=True)
    emit(f"    elapsed: {(time.time() - t0) / 60:.1f} min")
    report.extend(summarise_nested_cv(best_result, label="two-stage search"))

    params = best_result["params"]
    n_focused = int((params["search_stage"] == "focused").sum())
    gain = params["inner_focused_score"] - params["inner_broad_score"]
    emit(f"\n    focused grid chosen in {n_focused}/{len(params)} folds")
    emit(f"    inner refinement gain      : mean {gain.mean():+.4f} "
         f"(range {gain.min():+.4f} to {gain.max():+.4f})")

    emit("\n    Hyperparameters selected across the outer folds:")
    emit(f"      {'parameter':<28s} {'representative':>15s}   spread across folds")
    selected_summary = {}
    for col in [c for c in params.columns if c.startswith("model__")]:
        rep = representative_value(params[col])
        selected_summary[col] = rep
        values = params[col].dropna()
        try:
            numeric = values.astype(float)
            spread_txt = (f"min {numeric.min():.4g}, median {numeric.median():.4g}, "
                          f"max {numeric.max():.4g}")
        except (TypeError, ValueError):
            spread_txt = ", ".join(f"{v} x{c}" for v, c in
                                   values.value_counts().head(4).items())
        rep_txt = f"{rep:.4g}" if isinstance(rep, (float, np.floating)) else str(rep)
        emit(f"      {col.replace('model__', ''):<28s} {rep_txt:>15s}   {spread_txt}")

    results_json["stage2"] = {
        "procedure": "broad randomised search then focused grid, both inside "
                     "each outer training fold",
        "n_iter_broad": n_iter,
        "folds_focused_won": n_focused,
        "n_folds": int(len(params)),
        "mean_inner_refinement_gain": float(gain.mean()),
        "representative_params": {k: (float(v) if isinstance(v, (float, np.floating))
                                      else (int(v) if isinstance(v, (int, np.integer))
                                            else str(v)))
                                  for k, v in selected_summary.items()},
        "selected_params_by_fold": params.to_dict(orient="records"),
    }
    best_result["folds"].to_csv(
        os.path.join(RESULTS_DIR, f"xgb_fold_metrics{tag}.csv"), index=False
    )

    # --- stage 3: feature specifications ---
    emit("\n" + "=" * 78)
    emit("STAGE 3 - FEATURE SPECIFICATION COMPARISON")
    emit("=" * 78)

    spec_curves, spec_metrics = {}, {}
    for spec in specs:
        Xs, ys, feats, m = load_dataset(spec, outcome=args.outcome)
        emit(f"\n  --- {spec} ({m['n_features']} features) ---")
        if spec == primary_spec:
            result = best_result
            emit("    (reusing the stage 2 run: identical data, folds and search)")
        else:
            result = nested_cv(Xs, ys, two_stage_factory,
                               n_outer=N_OUTER_FOLDS, n_repeats=n_repeats,
                               random_state=RANDOM_STATE, verbose=True)
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        spec_curves[spec] = (ys[ok], scores[ok])
        metrics = evaluate_predictions(ys[ok], scores[ok])
        metrics.update(bootstrap_auc_ci(ys[ok], scores[ok],
                                       n_boot=500 if args.quick else 2000))
        metrics["n_features"] = m["n_features"]
        spec_metrics[spec] = metrics
        report.extend(summarise_nested_cv(result, label=spec))

    emit("\n  Specification summary:")
    emit(f"    {'specification':<24s} {'n_feat':>7s} {'AUC':>8s} "
         f"{'95% CI':>18s} {'AUPRC':>8s}")
    for spec, metrics in spec_metrics.items():
        emit(f"    {spec:<24s} {metrics['n_features']:>7d} "
             f"{metrics['auc_roc']:>8.4f} "
             f"  [{metrics['auc_ci_lo']:.3f}, {metrics['auc_ci_hi']:.3f}] "
             f"{metrics['auprc']:>8.4f}")
    results_json["stage3_specifications"] = spec_metrics

    emit("\n  Paired DeLong tests:")
    delong_results = {}
    for a, b in [("confounders_only", "obm_plus_confounders"),
                 ("confounders_only", "obm_only"),
                 ("obm_only", "obm_plus_confounders")]:
        if a not in spec_curves or b not in spec_curves:
            continue
        ya, sa = spec_curves[a]
        yb, sb = spec_curves[b]
        if not np.array_equal(ya, yb):
            continue
        test = delong_test(ya, sa, sb)
        if test:
            delong_results[f"{a}_vs_{b}"] = test
            emit(f"    {a:<22s} -> {b:<22s} dAUC {test['difference']:+.4f} "
                 f"(se {test['se']:.4f}, p {test['p_value']:.4g})")
    results_json["stage3_delong"] = delong_results

    # --- feature importance ---
    emit("\n" + "=" * 78)
    emit("FEATURE IMPORTANCE")
    emit("=" * 78)
    emit("\n  Permutation importance on held-out folds is the primary ranking.")
    emit("  Gain importance from the final fit is shown for contrast.")

    t0 = time.time()
    perm_df = outer_fold_permutation_importance(
        X, y, two_stage_factory, features,
        n_repeats_perm=3 if args.quick else 5,
        n_splits=3 if args.quick else N_OUTER_FOLDS,
    )
    emit(f"\n  permutation importance elapsed: {(time.time() - t0) / 60:.1f} min")

    # final fit for gain importance, at the representative config of the folds
    # whose performance is reported
    final_config = clean_params(selected_summary)
    final_estimator = make_estimator(chosen_weighting)
    valid = final_estimator.get_params()
    final_config = {k: v for k, v in final_config.items() if k in valid}
    final_estimator.set_params(**final_config)
    final_pipeline = build_pipeline(final_estimator, imbalance=chosen_imbalance,
                                    scale=False, random_state=RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X.to_numpy(dtype=float), y)
    # fitted on a plain numpy array, so the booster names features f0, f1, ...
    # and any feature never used in a split is missing from the score dict
    gain_scores = final_pipeline.named_steps["model"].get_booster().get_score(
        importance_type="gain")
    gain = np.array([gain_scores.get(f"f{i}", 0.0) for i in range(len(features))])

    importance = perm_df.merge(
        pd.DataFrame({"feature": features, "gain_importance": gain}), on="feature"
    ).sort_values("perm_importance_mean", ascending=False)

    with open(os.path.join(AIM2_DIR, "feature_manifest.json")) as handle:
        manifest = json.load(handle)
    confounder_set = set(manifest["confounders"])
    importance["is_confounder"] = importance["feature"].isin(confounder_set)
    importance.to_csv(os.path.join(RESULTS_DIR, f"xgb_importance{tag}.csv"),
                      index=False)

    emit(f"\n  Top 25 by permutation importance:")
    emit(f"    {'feature':<42s} {'perm':>9s} {'sd':>8s} {'+folds':>7s} "
         f"{'gain':>9s}")
    for _, row in importance.head(25).iterrows():
        marker = "C" if row["is_confounder"] else " "
        emit(f"  {marker} {row['feature']:<42s} "
             f"{row['perm_importance_mean']:>+9.5f} "
             f"{row['perm_importance_sd']:>8.5f} "
             f"{int(row['perm_folds_positive'])}/{int(row['n_folds'])}    "
             f"{row['gain_importance']:>9.4f}")
    emit("\n  (C marks a clinical confounder rather than an oximetry biomarker)")

    n_helpful = int((importance["perm_importance_mean"] > 0).sum())
    consistent = importance[(importance["perm_importance_mean"] > 0)
                            & (importance["perm_folds_positive"]
                               == importance["n_folds"])]
    emit(f"\n  features with positive mean permutation importance: "
         f"{n_helpful}/{len(importance)}")
    emit(f"  features helping in EVERY fold                    : "
         f"{len(consistent)}")

    emit("\n  Top 12 oximetry biomarkers:")
    for _, row in importance[~importance["is_confounder"]].head(12).iterrows():
        emit(f"    {row['feature']:<42s} {row['perm_importance_mean']:>+9.5f}")

    results_json["importance_top30"] = importance.head(30)[
        ["feature", "perm_importance_mean", "perm_importance_sd",
         "gain_importance"]].to_dict(orient="records")
    results_json["importance_consistent_all_folds"] = int(len(consistent))
    results_json["final_model"] = {"config": {k: str(v) for k, v in final_config.items()},
                                   "imbalance": chosen_imbalance,
                                   "weighting": str(chosen_weighting)}

    # --- plots ---
    y_primary, s_primary = spec_curves[primary_spec]
    extra = []
    if "confounders_only" in spec_curves:
        yc, sc = spec_curves["confounders_only"]
        if np.array_equal(yc, y_primary):
            extra.append(("confounders only", sc))
    plot_model_performance(
        y_primary, s_primary,
        title=f"XGBoost, {primary_spec}, out-of-fold predictions ({args.outcome})",
        path=os.path.join(RESULTS_DIR, f"xgb_performance{tag}.png"),
        extra_curves=extra,
    )
    plot_specification_comparison(
        spec_curves, y_primary,
        title="XGBoost by feature specification (nested CV out-of-fold)",
        path=os.path.join(RESULTS_DIR, f"xgb_specifications{tag}.png"),
    )
    top = importance.head(25).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.34 * len(top))))
    colours = ["tab:orange" if c else "tab:blue" for c in top["is_confounder"]]
    axes[0].barh(range(len(top)), top["perm_importance_mean"],
                 xerr=top["perm_importance_sd"], color=colours, capsize=2)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["feature"], fontsize=7)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set_xlabel("drop in held-out AUC when shuffled")
    axes[0].set_title("Permutation importance (primary)", fontsize=10)
    axes[0].grid(alpha=0.2, axis="x")
    axes[1].barh(range(len(top)), top["gain_importance"], color=colours)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("average gain per split")
    axes[1].set_title("Gain importance (training data, for contrast)", fontsize=10)
    axes[1].grid(alpha=0.2, axis="x")
    fig.suptitle("XGBoost feature importance\n"
                 "blue = oximetry biomarker, orange = clinical confounder",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"xgb_importance{tag}.png"), dpi=140)
    plt.close(fig)

    # --- head to head vs LR and RF, if their results exist ---
    others = {}
    for name, filename in (("LR", f"logreg_results{tag}.json"),
                           ("RF", f"rf_results{tag}.json")):
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path) as handle:
                others[name] = json.load(handle).get("stage3_specifications", {})
    if others:
        emit("\n" + "=" * 78)
        emit("HEAD TO HEAD - AUC BY SPECIFICATION")
        emit("=" * 78)
        header = "".join(f"{n + ' AUC':>10s}" for n in others)
        emit(f"\n    {'specification':<24s}{header}{'XGB AUC':>10s}")
        for spec in specs:
            row = "".join(
                f"{others[n][spec]['auc_roc']:>10.4f}" if spec in others[n] else f"{'-':>10s}"
                for n in others)
            emit(f"    {spec:<24s}{row}{spec_metrics[spec]['auc_roc']:>10.4f}")
        emit("\n  Point estimates only. Run aim2/compare_models.py for paired")
        emit("  DeLong tests between the models.")

    # --- headline result ---
    emit("\n" + "=" * 78)
    emit("HEADLINE RESULT")
    emit("=" * 78)
    primary = spec_metrics[primary_spec]
    emit(f"\n  XGBoost, {primary_spec}, outcome {args.outcome}")
    emit(f"    AUC-ROC  {primary['auc_roc']:.4f} "
         f"[{primary['auc_ci_lo']:.3f}, {primary['auc_ci_hi']:.3f}]")
    emit(f"    AUPRC    {primary['auprc']:.4f} "
         f"[{primary['auprc_ci_lo']:.3f}, {primary['auprc_ci_hi']:.3f}] "
         f"(no skill {primary['auprc_baseline']:.4f}, "
         f"lift {primary['auprc_lift']:.2f}x)")
    emit(f"    sens {primary['sensitivity']:.3f} / spec {primary['specificity']:.3f} "
         f"at Youden threshold {primary['threshold']:.4f}")
    emit(f"    balanced accuracy {primary['balanced_accuracy']:.4f}")
    emit(f"    Brier {primary['brier']:.4f}")
    emit(f"\n  imbalance strategy : {best_arm}")
    emit(f"  configuration      : {final_config}")
    emit(f"\n  Elapsed: {(time.time() - started) / 60:.1f} min")

    np.savez(os.path.join(RESULTS_DIR, f"xgb_oof_scores{tag}.npz"),
             **{spec: np.vstack([ys, ss]) for spec, (ys, ss) in spec_curves.items()})
    with open(os.path.join(RESULTS_DIR, f"xgb_summary{tag}.txt"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(RESULTS_DIR, f"xgb_results{tag}.json"), "w") as handle:
        json.dump(results_json, handle, indent=2, default=str)

    print(f"\n  report : {os.path.join(RESULTS_DIR, f'xgb_summary{tag}.txt')}")
    print(f"  plots  : {RESULTS_DIR}/xgb_*{tag}.png")
    print("\nNext: python3 aim2/compare_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
