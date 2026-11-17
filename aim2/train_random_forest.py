"""
Aim 2 model 2: tuned random forest.

The first model here that can learn threshold effects and interactions by
itself. Same folds and seed as logistic regression, so the two can be compared
with paired DeLong. Stage 1 picks the imbalance arm at a fixed config, stage 2
runs a two-step search inside each training fold (TwoStageSearch in
ml_common), and stage 3 fits the three feature specifications.

Features aren't standardised since splits only depend on value order.
Permutation importance on held-out folds is the main ranking; impurity
importance is computed on training data and favours features with many split
points, so it's only shown for contrast. Both split credit between correlated
biomarkers, so a near-zero score may just mean redundancy.

Outputs (processed/aim2/results/, with the --tag suffix if given):
    rf_summary.txt, rf_fold_metrics.csv, rf_importance.csv,
    rf_performance.png, rf_specifications.png, rf_importance.png,
    rf_results.json, rf_oof_scores.npz (the summary gets a head to head
    section if logreg_results.json exists)

Usage:
    python3 aim2/train_random_forest.py
    python3 aim2/train_random_forest.py --quick
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from scipy.stats import randint

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
    clean_param_value,
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

ensure_results_dir()

PRIMARY_SPEC = "obm_plus_confounders"

# imbalance arms. Both class-weight variants are tried since they behave
# differently with bagging: "balanced" uses weights from the whole training
# set, "balanced_subsample" recomputes them in each bootstrap sample
IMBALANCE_ARMS = [
    ("class_weight_balanced_subsample", "none", "balanced_subsample"),
    ("class_weight_balanced", "none", "balanced"),
    ("smotetomek", "smotetomek", None),
    ("none", "none", None),
]

# fixed config used only for the stage 1 arm comparison, so the imbalance
# choice is cheap and doesn't get mixed up with search noise
STAGE1_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}


def broad_search_space(quick=False):
    """
    Random search space for the forest.

    More trees can't overfit, so n_estimators mainly costs runtime. max_depth
    and min_samples_leaf control complexity: with a few hundred events a leaf
    of two participants is fitting individuals. A small max_features
    decorrelates the trees across the correlated biomarkers, and bootstrap
    tests whether bagging helps at this sample size.
    """
    if quick:
        return {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 6, 12],
            "model__min_samples_leaf": [1, 5, 15],
            "model__max_features": ["sqrt", 0.1],
        }
    return {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 4, 6, 8, 12, 20],
        "model__min_samples_split": randint(2, 40),
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": ["sqrt", "log2", 0.05, 0.1, 0.2, 0.4],
        "model__bootstrap": [True, False],
    }


def focused_grid(best_params):
    """
    Small grid around the random search winner.

    Only refines the two settings that matter most for generalisation at this
    events-per-variable ratio (leaf size and feature subsampling). Tree count
    is pinned high since more trees can't hurt.
    """
    leaf = int(best_params.get("model__min_samples_leaf", 5))
    max_features = best_params.get("model__max_features", "sqrt")

    leaf_candidates = sorted({
        max(1, leaf - 5), max(1, leaf - 2), leaf, leaf + 2, leaf + 5, leaf + 10
    })
    feature_candidates = ["sqrt", 0.05, 0.1, 0.2]
    if max_features not in feature_candidates:
        feature_candidates.append(max_features)

    return {
        "model__n_estimators": [800],
        "model__max_depth": [best_params.get("model__max_depth", None)],
        "model__min_samples_leaf": leaf_candidates,
        "model__max_features": feature_candidates,
        "model__min_samples_split": [
            int(best_params.get("model__min_samples_split", 2))
        ],
        "model__bootstrap": [best_params.get("model__bootstrap", True)],
    }


def make_estimator(class_weight):
    return RandomForestClassifier(
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,          # the search is parallel, so one thread per fit
    )


def make_fixed_factory(imbalance, class_weight, config):
    """Factory for the pipeline at one fixed configuration (used in stage 1)."""
    def factory():
        estimator = make_estimator(class_weight)
        estimator.set_params(**config)
        pipeline = build_pipeline(
            estimator, imbalance=imbalance, scale=False,
            random_state=RANDOM_STATE,
        )
        # a one-candidate GridSearchCV has the same interface as the real
        # searches, so nested_cv doesn't need a special case
        return GridSearchCV(
            pipeline, param_grid={"model__n_estimators": [config["n_estimators"]]},
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE),
            n_jobs=-1, refit=True, error_score=np.nan,
        )
    return factory


def make_random_search_factory(imbalance, class_weight, space, n_iter):
    def factory():
        pipeline = build_pipeline(
            make_estimator(class_weight), imbalance=imbalance, scale=False,
            random_state=RANDOM_STATE,
        )
        return RandomizedSearchCV(
            pipeline, param_distributions=space, n_iter=n_iter,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE),
            n_jobs=-1, refit=True, random_state=RANDOM_STATE,
            error_score=np.nan,
        )
    return factory


def make_grid_factory(imbalance, class_weight, grid):
    def factory():
        pipeline = build_pipeline(
            make_estimator(class_weight), imbalance=imbalance, scale=False,
            random_state=RANDOM_STATE,
        )
        return GridSearchCV(
            pipeline, param_grid=grid, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE),
            n_jobs=-1, refit=True, error_score=np.nan,
        )
    return factory


def outer_fold_permutation_importance(X, y, factory, features,
                                      n_repeats_perm=5, n_splits=5,
                                      random_state=RANDOM_STATE):
    """
    Permutation importance averaged over held-out outer folds.

    In each fold the pipeline is fitted on the training part, then each feature
    is shuffled in the test part and the drop in AUC recorded. Because it's
    measured on data the model never saw, it shows how much the model relies on
    the feature to predict, without crediting anything it memorised.
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
        best = search.best_estimator_
        result = permutation_importance(
            best, X_values[test_idx], y[test_idx],
            scoring="roc_auc", n_repeats=n_repeats_perm,
            random_state=random_state, n_jobs=-1,
        )
        importances.append(result.importances_mean)

    matrix = np.vstack(importances)
    return pd.DataFrame({
        "feature": features,
        "perm_importance_mean": matrix.mean(axis=0),
        "perm_importance_sd": matrix.std(axis=0),
        # how many folds the feature helped in. Helping in one fold and hurting
        # in another is noise, whatever the mean looks like
        "perm_folds_positive": (matrix > 0).sum(axis=0),
        "n_folds": matrix.shape[0],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--outcome", default=PRIMARY_OUTCOME)
    parser.add_argument(
        "--specifications", default=None,
        help="comma separated specification names from feature_manifest.json. "
             "Defaults to the three primary arms. Use this to run the "
             "over-parameterised sensitivity arms, for example "
             "confounders_only,obm_full,obm_full_plus_confounders",
    )
    parser.add_argument(
        "--tag", default="",
        help="suffix appended to every output filename, so a secondary run "
             "does not overwrite the primary results",
    )
    parser.add_argument("--n-iter", type=int, default=None,
                        help="randomised search iterations per outer fold")
    args = parser.parse_args()

    # same number of repeats as the other two models. On the half cohort the
    # forest used 2 and logistic regression used 3, so their out-of-fold
    # scores were averaged over different numbers of repeats
    n_repeats = 1 if args.quick else N_OUTER_REPEATS

    # specs to fit, the three primary arms by default. The sensitivity arms
    # have hundreds of features and take much longer, so they're opt-in
    specs = (tuple(x.strip() for x in args.specifications.split(","))
             if args.specifications else SPECIFICATIONS)
    # filename suffix so a secondary outcome or sensitivity run doesn't
    # overwrite the primary results
    tag = f"_{args.tag}" if args.tag else ""
    # spec used for the headline result and plots. If a sensitivity run leaves
    # out the primary arm, the last (richest) requested arm is used instead
    primary_spec = PRIMARY_SPEC if PRIMARY_SPEC in specs else specs[-1]
    n_iter = args.n_iter or (8 if args.quick else 25)
    space = broad_search_space(quick=args.quick)

    report = []

    def emit(text=""):
        print(text)
        report.append(text)

    emit("=" * 78)
    emit("AIM 2 MODEL 2 - OPTIMISED RANDOM FOREST")
    emit("=" * 78)

    started = time.time()

    X, y, features, meta = load_dataset(primary_spec, outcome=args.outcome)
    emit(f"\nOutcome                : {args.outcome}")
    emit(f"Participants           : {meta['n_participants']}")
    emit(f"Events                 : {meta['n_events']} "
         f"({meta['event_rate']:.2%})")
    emit(f"Features               : {meta['n_features']}")
    emit(f"Events per variable    : "
         f"{meta['n_events'] / max(meta['n_features'], 1):.2f}")
    emit(f"\nCross validation       : {n_repeats} repeat(s) x "
         f"{N_OUTER_FOLDS} outer folds, {N_INNER_FOLDS} inner folds")
    emit(f"Randomised search      : {n_iter} candidates per outer fold")
    emit(f"Random seed            : {RANDOM_STATE}")
    emit(f"\nNote: features are NOT standardised for the forest. Tree splits")
    emit(f"depend only on the ordering of values, so any monotonic rescaling")
    emit(f"leaves the model unchanged. Median imputation is still applied")
    emit(f"inside the pipeline because scikit-learn's forest cannot consume")
    emit(f"NaN directly.")

    results_json = {"outcome": args.outcome,
                    "meta": {k: v for k, v in meta.items() if k != "pid"}}

    # --- stage 1: imbalance strategy at a fixed config ---
    emit("\n" + "=" * 78)
    emit("STAGE 1 - CLASS IMBALANCE STRATEGY")
    emit("=" * 78)
    emit(f"\n  Compared at a fixed configuration ({STAGE1_CONFIG}) so the")
    emit(f"  imbalance question is isolated from hyperparameter search noise.")

    arm_metrics = {}
    for label, imbalance, class_weight in IMBALANCE_ARMS:
        emit(f"\n  --- arm: {label} ---")
        factory = make_fixed_factory(imbalance, class_weight, STAGE1_CONFIG)
        result = nested_cv(X, y, factory, n_outer=N_OUTER_FOLDS,
                           n_repeats=n_repeats, random_state=RANDOM_STATE,
                           verbose=True)
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        metrics = evaluate_predictions(y[ok], scores[ok])
        arm_metrics[label] = metrics
        emit(f"    pooled AUC {metrics['auc_roc']:.4f}  "
             f"AUPRC {metrics['auprc']:.4f}  Brier {metrics['brier']:.4f}")

    emit("\n  Arm comparison:")
    emit(f"    {'arm':<34s} {'AUC':>8s} {'AUPRC':>8s} {'Brier':>8s}")
    for label, metrics in arm_metrics.items():
        emit(f"    {label:<34s} {metrics['auc_roc']:>8.4f} "
             f"{metrics['auprc']:>8.4f} {metrics['brier']:>8.4f}")

    best_arm = max(arm_metrics, key=lambda k: arm_metrics[k]["auc_roc"])
    emit(f"\n  Best arm by pooled out-of-fold AUC: {best_arm}")
    results_json["stage1_imbalance_arms"] = arm_metrics
    results_json["stage1_best_arm"] = best_arm

    chosen = next(a for a in IMBALANCE_ARMS if a[0] == best_arm)
    _, chosen_imbalance, chosen_class_weight = chosen

    # --- stage 2: two-step search inside every outer training fold ---
    emit("\n" + "=" * 78)
    emit("STAGE 2 - HYPERPARAMETER OPTIMISATION")
    emit("=" * 78)
    emit(f"\n  Inside every outer training fold: a broad randomised search of")
    emit(f"  {n_iter} candidates, then a focused grid around THAT fold's winner,")
    emit(f"  keeping whichever scores higher on the fold's own inner CV. Both")
    emit(f"  stages and the choice see training data only.")

    def two_stage_factory():
        return TwoStageSearch(
            broad_factory=make_random_search_factory(
                chosen_imbalance, chosen_class_weight, space, n_iter),
            grid_builder=focused_grid,
            grid_factory=lambda grid: make_grid_factory(
                chosen_imbalance, chosen_class_weight, grid)(),
        )

    t0 = time.time()
    best_result = nested_cv(X, y, two_stage_factory, n_outer=N_OUTER_FOLDS,
                            n_repeats=n_repeats, random_state=RANDOM_STATE,
                            verbose=True)
    best_factory = two_stage_factory
    emit(f"    elapsed: {(time.time() - t0) / 60:.1f} min")
    report.extend(summarise_nested_cv(best_result, label="two-stage search"))

    params = best_result["params"]
    n_focused = int((params["search_stage"] == "focused").sum())
    gain = params["inner_focused_score"] - params["inner_broad_score"]
    emit(f"\n    focused grid beat the broad search in {n_focused}/{len(params)} folds")
    emit(f"    inner refinement gain      : mean {gain.mean():+.4f} "
         f"(range {gain.min():+.4f} to {gain.max():+.4f})")
    emit("\n    Hyperparameters selected across outer folds:")
    for col in [c for c in params.columns if c.startswith("model__")]:
        counts = params[col].value_counts()
        shown = ", ".join(f"{v} x{c}" for v, c in counts.head(4).items())
        emit(f"      {col:<32s} {shown}")

    results_json["stage2"] = {
        "procedure": "broad randomised search then focused grid, both inside "
                     "each outer training fold",
        "n_iter_broad": n_iter,
        "folds_focused_won": n_focused,
        "n_folds": int(len(params)),
        "mean_inner_refinement_gain": float(gain.mean()),
        "selected_params_by_fold": params.to_dict(orient="records"),
    }

    best_result["folds"].to_csv(
        os.path.join(RESULTS_DIR, f"rf_fold_metrics{tag}.csv"), index=False
    )

    # --- stage 3: feature specifications ---
    emit("\n" + "=" * 78)
    emit("STAGE 3 - FEATURE SPECIFICATION COMPARISON")
    emit("=" * 78)

    spec_curves = {}
    spec_metrics = {}
    for spec in specs:
        Xs, ys, feats, m = load_dataset(spec, outcome=args.outcome)
        emit(f"\n  --- {spec} ({m['n_features']} features) ---")
        if spec == primary_spec:
            # same data, folds and search as stage 2, so just reuse that run
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
                 f"(p {test['p_value']:.4g})")
    results_json["stage3_delong"] = delong_results

    # --- feature importance ---
    emit("\n" + "=" * 78)
    emit("FEATURE IMPORTANCE")
    emit("=" * 78)
    emit("\n  Permutation importance on held-out folds is the primary ranking.")
    emit("  Impurity importance from the final fit is reported alongside so")
    emit("  the disagreement between the two is visible.")

    t0 = time.time()
    perm_df = outer_fold_permutation_importance(
        X, y, best_factory, features,
        n_repeats_perm=3 if args.quick else 5,
        n_splits=3 if args.quick else N_OUTER_FOLDS,
    )
    emit(f"\n  permutation importance elapsed: {(time.time() - t0) / 60:.1f} min")

    # final fit for impurity importance. Uses the most common value of each
    # hyperparameter across the winning nested CV folds. Something like the
    # median of the grid values could be a config the search never chose in
    # any fold, and then the importances wouldn't match the model whose
    # performance is reported.
    winning_params = best_result["params"]
    final_config = clean_params({
        col: winning_params[col].mode().iloc[0]
        for col in winning_params.columns if col.startswith("model__")
    })
    emit(f"\n  final model configuration, modal across the winning folds:")
    emit(f"    {final_config}")

    final_estimator = make_estimator(chosen_class_weight)
    valid = final_estimator.get_params()
    final_config = {k: v for k, v in final_config.items() if k in valid}
    final_estimator.set_params(**final_config)
    final_pipeline = build_pipeline(
        final_estimator, imbalance=chosen_imbalance, scale=False,
        random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X.to_numpy(dtype=float), y)
    impurity = final_pipeline.named_steps["model"].feature_importances_

    importance = perm_df.merge(
        pd.DataFrame({"feature": features, "impurity_importance": impurity}),
        on="feature",
    )
    importance = importance.sort_values("perm_importance_mean",
                                        ascending=False)
    importance.to_csv(os.path.join(RESULTS_DIR, f"rf_importance{tag}.csv"),
                      index=False)

    with open(os.path.join(AIM2_DIR, "feature_manifest.json")) as handle:
        manifest = json.load(handle)
    confounder_set = set(manifest["confounders"])
    importance["is_confounder"] = importance["feature"].isin(confounder_set)

    emit(f"\n  Top 25 by permutation importance:")
    emit(f"    {'feature':<42s} {'perm':>9s} {'sd':>8s} {'+folds':>7s} "
         f"{'impurity':>9s}")
    for _, row in importance.head(25).iterrows():
        # called "marker" because "tag" is the filename suffix, and reusing
        # that name here renamed every file written after this loop
        marker = "C" if row["is_confounder"] else " "
        emit(f"  {marker} {row['feature']:<42s} "
             f"{row['perm_importance_mean']:>+9.5f} "
             f"{row['perm_importance_sd']:>8.5f} "
             f"{int(row['perm_folds_positive'])}/{int(row['n_folds'])}    "
             f"{row['impurity_importance']:>9.5f}")
    emit("\n  (C marks a clinical confounder rather than an oximetry biomarker)")

    n_helpful = int((importance["perm_importance_mean"] > 0).sum())
    consistent = importance[
        (importance["perm_importance_mean"] > 0)
        & (importance["perm_folds_positive"] == importance["n_folds"])
    ]
    emit(f"\n  features with positive mean permutation importance: "
         f"{n_helpful}/{len(importance)}")
    emit(f"  features helping in EVERY fold                    : "
         f"{len(consistent)}")
    emit("  The second number is the one to trust. A feature with a large mean")
    emit("  that helps in only some folds is being credited for fold-specific")
    emit("  noise.")

    top_obm = importance[~importance["is_confounder"]].head(12)
    emit(f"\n  Top 12 oximetry biomarkers:")
    for _, row in top_obm.iterrows():
        emit(f"    {row['feature']:<42s} "
             f"{row['perm_importance_mean']:>+9.5f}")

    results_json["importance_top30"] = importance.head(30)[
        ["feature", "perm_importance_mean", "perm_importance_sd",
         "impurity_importance"]
    ].to_dict(orient="records")

    # --- plots ---
    y_primary, s_primary = spec_curves[primary_spec]
    extra = []
    if "confounders_only" in spec_curves:
        yc, sc = spec_curves["confounders_only"]
        if np.array_equal(yc, y_primary):
            extra.append(("confounders only", sc))

    plot_model_performance(
        y_primary, s_primary,
        title=f"Random forest, {primary_spec}, out-of-fold predictions "
              f"({args.outcome})",
        path=os.path.join(RESULTS_DIR, f"rf_performance{tag}.png"),
        extra_curves=extra,
    )
    plot_specification_comparison(
        spec_curves, y_primary,
        title="Random forest by feature specification (nested CV out-of-fold)",
        path=os.path.join(RESULTS_DIR, f"rf_specifications{tag}.png"),
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

    axes[1].barh(range(len(top)), top["impurity_importance"], color=colours)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("mean decrease in impurity")
    axes[1].set_title("Impurity importance (biased, for contrast)", fontsize=10)
    axes[1].grid(alpha=0.2, axis="x")
    fig.suptitle("Random forest feature importance\n"
                 "blue = oximetry biomarker, orange = clinical confounder",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"rf_importance{tag}.png"), dpi=140)
    plt.close(fig)

    # --- head to head vs logistic regression ---
    logreg_json = os.path.join(RESULTS_DIR, f"logreg_results{tag}.json")
    if os.path.exists(logreg_json):
        emit("\n" + "=" * 78)
        emit("HEAD TO HEAD - RANDOM FOREST VERSUS LOGISTIC REGRESSION")
        emit("=" * 78)
        with open(logreg_json) as handle:
            logreg = json.load(handle)
        lr_specs = logreg.get("stage3_specifications", {})
        emit(f"\n    {'specification':<24s} {'LR AUC':>9s} {'RF AUC':>9s} "
             f"{'delta':>8s}")
        for spec in specs:
            if spec in lr_specs and spec in spec_metrics:
                lr_auc = lr_specs[spec]["auc_roc"]
                rf_auc = spec_metrics[spec]["auc_roc"]
                emit(f"    {spec:<24s} {lr_auc:>9.4f} {rf_auc:>9.4f} "
                     f"{rf_auc - lr_auc:>+8.4f}")
        emit("\n  Both models used identical folds, seed and preprocessing, so")
        emit("  these differences are attributable to the model class.")
        emit("  A negligible or negative delta would indicate the")
        emit("  cardiovascular signal available in these biomarkers is")
        emit("  essentially linear and additive, in which case the forest's")
        emit("  extra capacity is only fitting noise. A clear positive delta")
        emit("  indicates real threshold or interaction structure, which is")
        emit("  the hypothesis motivating the deep learning arm in Aim 3.")
        emit("\n  A paired DeLong test between the two models requires their")
        emit("  out-of-fold score vectors on identical participants; run")
        emit("  aim2/compare_models.py once both have been fitted.")

    # --- headline result ---
    emit("\n" + "=" * 78)
    emit("HEADLINE RESULT")
    emit("=" * 78)
    primary = spec_metrics[primary_spec]
    emit(f"\n  Random forest, {primary_spec}, outcome {args.outcome}")
    emit(f"    AUC-ROC  {primary['auc_roc']:.4f} "
         f"[{primary['auc_ci_lo']:.3f}, {primary['auc_ci_hi']:.3f}]")
    emit(f"    AUPRC    {primary['auprc']:.4f} "
         f"[{primary['auprc_ci_lo']:.3f}, {primary['auprc_ci_hi']:.3f}] "
         f"(no skill {primary['auprc_baseline']:.4f}, "
         f"lift {primary['auprc_lift']:.2f}x)")
    emit(f"    sens {primary['sensitivity']:.3f} / "
         f"spec {primary['specificity']:.3f}")
    emit(f"    balanced accuracy {primary['balanced_accuracy']:.4f}")
    emit(f"    Brier {primary['brier']:.4f}")
    emit(f"\n  imbalance strategy : {best_arm}")
    emit(f"  configuration      : {final_config}")
    emit(f"\n  Elapsed: {(time.time() - started) / 60:.1f} min")

    # out-of-fold scores for the paired tests in compare_models.py
    np.savez(
        os.path.join(RESULTS_DIR, f"rf_oof_scores{tag}.npz"),
        **{spec: np.vstack([ys, ss])
           for spec, (ys, ss) in spec_curves.items()},
    )

    with open(os.path.join(RESULTS_DIR, f"rf_summary{tag}.txt"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(RESULTS_DIR, f"rf_results{tag}.json"), "w") as handle:
        json.dump(results_json, handle, indent=2, default=str)

    print(f"\n  report : {os.path.join(RESULTS_DIR, f'rf_summary{tag}.txt')}")
    print(f"  plots  : {RESULTS_DIR}/rf_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
