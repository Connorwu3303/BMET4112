"""
Aim 2 model 1: tuned logistic regression baseline.

The model to beat, since clinical CVD risk scores are built from it. Tuning
runs in three stages so each result traces back to one choice:
  1. class imbalance: SMOTETomek, class weights or nothing
  2. penalty type and strength (L1, L2, elastic net) by nested CV search
  3. feature specification: confounders only, OBM only, OBM plus confounders
Resampling is compared rather than assumed because it's known to hurt
calibration. All reported performance comes from the outer CV loop. The
coefficients come from one final fit on everyone, for interpretation only.

Outputs (processed/aim2/results/, with the --tag suffix if given):
    logreg_summary.txt, logreg_fold_metrics.csv,
    logreg_selected_params_by_fold.csv, logreg_coefficients.csv,
    logreg_performance.png, logreg_specifications.png,
    logreg_coefficients.png, logreg_results.json, logreg_oof_scores.npz

Usage:
    python3 aim2/train_logistic_regression.py
    python3 aim2/train_logistic_regression.py --quick   # smaller grids
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
    SPECIFICATIONS,
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

RESULTS_DIR = ensure_results_dir()

# stage 1 arms: (label, imbalance strategy, class_weight for the estimator)
IMBALANCE_ARMS = [
    ("smotetomek", "smotetomek", None),
    ("class_weight_balanced", "none", "balanced"),
    ("none", "none", None),
]

PRIMARY_SPEC = "obm_plus_confounders"


def penalty_grid(quick=False):
    """
    Grid over penalty type and strength.

    L2 keeps every biomarker with a small weight, which suits signal spread
    over many correlated features. L1 zeroes most coefficients, but picks one
    feature from a correlated group more or less at random, so the choice
    changes between folds. Elastic net mixes the two and is more stable. C is
    the inverse strength (small C means heavy regularisation), log spaced well
    into heavy penalties.
    """
    if quick:
        c_values = np.logspace(-3, 1, 5)
        return [
            {"model__penalty": ["l2"], "model__C": c_values,
             "model__solver": ["liblinear"]},
            {"model__penalty": ["l1"], "model__C": c_values,
             "model__solver": ["liblinear"]},
        ]

    c_values = np.logspace(-4, 1.5, 12)
    return [
        # liblinear handles both L1 and L2 fine at this size and doesn't need
        # scaling to converge
        {"model__penalty": ["l2"], "model__C": c_values,
         "model__solver": ["liblinear"]},
        {"model__penalty": ["l1"], "model__C": c_values,
         "model__solver": ["liblinear"]},
        # saga is the only solver with elastic net. It converges slowly on
        # correlated standardised features, hence the higher max_iter
        {"model__penalty": ["elasticnet"], "model__C": np.logspace(-3, 1, 5),
         "model__l1_ratio": [0.15, 0.5, 0.85], "model__solver": ["saga"],
         "model__max_iter": [3000]},
    ]


def make_search_factory(grid, imbalance, class_weight, scoring="roc_auc"):
    """
    Returns a factory that builds a fresh GridSearchCV over the whole pipeline,
    so no state carries over between outer folds. Grid keys use the "model__"
    prefix, so imputation, scaling and resampling are refitted for every
    candidate.
    """
    def factory():
        estimator = LogisticRegression(
            class_weight=class_weight,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        pipeline = build_pipeline(
            estimator, imbalance=imbalance, scale=True,
            random_state=RANDOM_STATE,
        )
        return GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=StratifiedKFold(
                n_splits=N_INNER_FOLDS, shuffle=True,
                random_state=RANDOM_STATE,
            ),
            n_jobs=-1,
            refit=True,
            error_score=np.nan,
        )
    return factory


def most_frequent_joint_config(params):
    """
    The hyperparameter combination picked together in the most outer folds.

    Taking each parameter's mode separately can give a combination no fold
    chose, so the interpretation model wouldn't match any selected model.
    Values that don't apply to a fold's penalty (l1_ratio for L2) are dropped
    so the defaults are used, and ties go to the first configuration seen.
    Returns (config_without_prefix, n_folds_chosen, n_folds_total).
    """
    model_cols = [c for c in params.columns if c.startswith("model__")]
    joint_key = params[model_cols].astype(str).agg(" | ".join, axis=1)
    counts = joint_key.value_counts(sort=False)
    best_key = max(counts.index, key=lambda k: (counts[k], -list(joint_key).index(k)))
    winner = params.loc[joint_key == best_key].iloc[0]
    config = clean_params({
        col: winner[col] for col in model_cols if not pd.isna(winner[col])
    })
    return config, int(counts[best_key]), int(len(params))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="smaller grids and fewer repeats for a fast check")
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
    args = parser.parse_args()

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
    grid = penalty_grid(quick=args.quick)

    report = []

    def emit(text=""):
        print(text)
        report.append(text)

    emit("=" * 78)
    emit("AIM 2 MODEL 1 - OPTIMISED LOGISTIC REGRESSION")
    emit("=" * 78)

    started = time.time()

    # --- data ---
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
    emit(f"Missing cells          : {X.isna().mean().mean():.2%} "
         f"(median-imputed inside each training fold)")

    emit(f"\nCross validation       : {n_repeats} repeat(s) x "
         f"{N_OUTER_FOLDS} outer folds, {N_INNER_FOLDS} inner folds")
    emit(f"Random seed            : {RANDOM_STATE}")

    results_json = {"outcome": args.outcome, "meta": {
        k: v for k, v in meta.items() if k != "pid"
    }}

    # --- stage 1: imbalance strategy ---
    emit("\n" + "=" * 78)
    emit("STAGE 1 - CLASS IMBALANCE STRATEGY")
    emit("=" * 78)
    emit("\n  All three arms use the identical pipeline, grid, folds and seed,")
    emit("  so the only difference is how the class imbalance is handled.")
    emit("  Resampling and imputation both sit inside the pipeline, so they")
    emit("  are refitted on each training fold and never see the test fold.")

    arm_results = {}
    for label, imbalance, class_weight in IMBALANCE_ARMS:
        emit(f"\n  --- arm: {label} "
             f"(resampling={imbalance}, class_weight={class_weight}) ---")
        factory = make_search_factory(grid, imbalance, class_weight)
        t0 = time.time()
        result = nested_cv(
            X, y, factory, n_outer=N_OUTER_FOLDS, n_repeats=n_repeats,
            random_state=RANDOM_STATE, verbose=True,
        )
        arm_results[label] = result
        emit(f"    elapsed: {(time.time() - t0) / 60:.1f} min")
        report.extend(summarise_nested_cv(result, label=label))

    emit("\n  Arm comparison on pooled out-of-fold predictions:")
    emit(f"    {'arm':<24s} {'AUC':>8s} {'AUPRC':>8s} {'Brier':>8s} "
         f"{'sens':>7s} {'spec':>7s}")
    arm_scores = {}
    for label, result in arm_results.items():
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        metrics = evaluate_predictions(y[ok], scores[ok])
        arm_scores[label] = metrics
        emit(f"    {label:<24s} {metrics['auc_roc']:>8.4f} "
             f"{metrics['auprc']:>8.4f} {metrics['brier']:>8.4f} "
             f"{metrics['sensitivity']:>7.3f} {metrics['specificity']:>7.3f}")

    best_arm = max(arm_scores, key=lambda k: arm_scores[k]["auc_roc"])
    emit(f"\n  Best arm by pooled out-of-fold AUC: {best_arm}")

    # also report the best arm by Brier, since resampling often loses on
    # calibration even when it wins on AUC
    best_brier = min(arm_scores, key=lambda k: arm_scores[k]["brier"])
    emit(f"  Best arm by calibration (lowest Brier): {best_brier}")
    if best_brier != best_arm:
        emit("  These disagree. AUC measures ranking only, so a resampled model")
        emit("  can rank well while producing probabilities that systematically")
        emit("  overstate risk, because it was trained on a cohort whose")
        emit("  apparent event rate was inflated to 50%. The arm chosen below")
        emit("  optimises discrimination; if the thesis needs calibrated")
        emit("  absolute risk rather than ranking, prefer the better-calibrated")
        emit("  arm or apply post hoc calibration on a held-out fold.")

    results_json["stage1_imbalance_arms"] = {
        label: {k: v for k, v in m.items()}
        for label, m in arm_scores.items()
    }
    results_json["stage1_best_arm"] = best_arm

    chosen = next(a for a in IMBALANCE_ARMS if a[0] == best_arm)
    _, chosen_imbalance, chosen_class_weight = chosen

    # --- stage 2: which hyperparameters the search picked ---
    emit("\n" + "=" * 78)
    emit("STAGE 2 - SELECTED HYPERPARAMETERS")
    emit("=" * 78)
    params = arm_results[best_arm]["params"]
    emit(f"\n  Hyperparameters chosen across the {len(params)} outer folds")
    emit("  of the winning arm. Stability here matters: a penalty that varies")
    emit("  wildly between folds means the search is chasing noise and the")
    emit("  selected value should not be quoted as 'the' optimum.")
    for col in [c for c in params.columns if c.startswith("model__")]:
        counts = params[col].value_counts()
        emit(f"\n    {col}:")
        for value, count in counts.items():
            emit(f"      {value}  chosen in {count}/{len(params)} folds")

    folds = arm_results[best_arm]["folds"]
    folds.to_csv(os.path.join(RESULTS_DIR, f"logreg_fold_metrics{tag}.csv"),
                 index=False)
    # save the full per-fold selections (the forest and XGBoost scripts do the
    # same) so the joint config picked below can be checked
    params.to_csv(
        os.path.join(RESULTS_DIR, f"logreg_selected_params_by_fold{tag}.csv"),
        index=False,
    )
    results_json["stage2_selected_params_by_fold"] = params.to_dict(orient="records")
    results_json["stage2_selected_params"] = {
        col: {str(k): int(v) for k, v in params[col].value_counts().items()}
        for col in params.columns if col.startswith("model__")
    }

    # --- stage 3: feature specifications ---
    emit("\n" + "=" * 78)
    emit("STAGE 3 - FEATURE SPECIFICATION COMPARISON")
    emit("=" * 78)
    emit("\n  The thesis question. If OBM plus confounders does not beat")
    emit("  confounders alone, the oximetry biomarkers add nothing once age")
    emit("  and BMI are known, regardless of how good the combined AUC looks")
    emit("  in isolation.")

    spec_results = {}
    spec_curves = {}
    for spec in specs:
        Xs, ys, feats, m = load_dataset(spec, outcome=args.outcome)
        emit(f"\n  --- specification: {spec} ({m['n_features']} features) ---")
        factory = make_search_factory(grid, chosen_imbalance,
                                      chosen_class_weight)
        result = nested_cv(
            Xs, ys, factory, n_outer=N_OUTER_FOLDS, n_repeats=n_repeats,
            random_state=RANDOM_STATE, verbose=True,
        )
        spec_results[spec] = (result, ys, m)
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        spec_curves[spec] = (ys[ok], scores[ok])
        report.extend(summarise_nested_cv(result, label=spec))

    emit("\n  Specification summary on pooled out-of-fold predictions:")
    emit(f"    {'specification':<24s} {'n_feat':>7s} {'AUC':>8s} "
         f"{'95% CI':>18s} {'AUPRC':>8s}")
    spec_metrics = {}
    for spec, (result, ys, m) in spec_results.items():
        scores = result["mean_oof"]
        ok = ~np.isnan(scores)
        metrics = evaluate_predictions(ys[ok], scores[ok])
        ci = bootstrap_auc_ci(ys[ok], scores[ok],
                              n_boot=500 if args.quick else 2000)
        metrics.update(ci)
        spec_metrics[spec] = metrics
        emit(f"    {spec:<24s} {m['n_features']:>7d} "
             f"{metrics['auc_roc']:>8.4f} "
             f"  [{ci['auc_ci_lo']:.3f}, {ci['auc_ci_hi']:.3f}] "
             f"{metrics['auprc']:>8.4f}")

    results_json["stage3_specifications"] = spec_metrics

    # paired DeLong, since every spec is scored on the same participants
    emit("\n  Paired DeLong tests for correlated ROC curves:")
    emit("    (same participants in every arm, so the AUC estimates are")
    emit("     correlated and an unpaired test would understate the standard")
    emit("     error of the difference)")
    comparisons = [
        ("confounders_only", "obm_plus_confounders"),
        ("confounders_only", "obm_only"),
        ("obm_only", "obm_plus_confounders"),
    ]
    delong_results = {}
    for a, b in comparisons:
        if a not in spec_curves or b not in spec_curves:
            continue
        ya, sa = spec_curves[a]
        yb, sb = spec_curves[b]
        if len(ya) != len(yb) or not np.array_equal(ya, yb):
            emit(f"    {a} vs {b}: SKIP, differing participant sets")
            continue
        test = delong_test(ya, sa, sb)
        if test is None:
            continue
        delong_results[f"{a}_vs_{b}"] = test
        stars = ""
        if test["p_value"] < 0.001:
            stars = " ***"
        elif test["p_value"] < 0.01:
            stars = " **"
        elif test["p_value"] < 0.05:
            stars = " *"
        emit(f"    {a:<22s} -> {b:<22s} "
             f"dAUC {test['difference']:+.4f} "
             f"(se {test['se']:.4f}, z {test['z']:+.2f}, "
             f"p {test['p_value']:.4g}){stars}")
    results_json["stage3_delong"] = delong_results

    incremental = delong_results.get(
        "confounders_only_vs_obm_plus_confounders"
    )
    if incremental:
        emit("\n  INTERPRETATION of the incremental value of oximetry:")
        if incremental["p_value"] < 0.05 and incremental["difference"] > 0:
            emit(f"    Adding the oximetry biomarkers to the clinical")
            emit(f"    confounders improves AUC by "
                 f"{incremental['difference']:+.4f} "
                 f"(p = {incremental['p_value']:.4g}).")
            emit(f"    The oximetry waveform carries cardiovascular risk")
            emit(f"    information beyond age, BMI, blood pressure, lipids,")
            emit(f"    smoking and medication status.")
        else:
            emit(f"    Adding the oximetry biomarkers changes AUC by "
                 f"{incremental['difference']:+.4f} "
                 f"(p = {incremental['p_value']:.4g}), which is not")
            emit(f"    statistically significant at this sample size.")
            emit(f"    On this half-cohort the oximetry biomarkers do not")
            emit(f"    demonstrably add to the clinical confounders for a")
            emit(f"    linear model. This is a real and reportable result, and")
            emit(f"    it is exactly why the random forest and the 1D-CNN are")
            emit(f"    worth running: a linear model can only use monotonic,")
            emit(f"    additive effects, so a null here does not rule out")
            emit(f"    nonlinear or interaction structure in the waveform.")
            emit(f"    Worth rechecking once the full cohort is available,")
            emit(f"    since the confidence interval is wide at this n.")

    # --- final model for interpretation ---
    emit("\n" + "=" * 78)
    emit("FINAL MODEL - COEFFICIENTS AND ODDS RATIOS")
    emit("=" * 78)
    emit("\n  Fitted on all participants at the settings selected above. This")
    emit("  model is for INTERPRETATION ONLY: its apparent performance would")
    emit("  be optimistic because every participant contributed to the fit.")
    emit("  Every performance number quoted in this report comes from the")
    emit("  nested cross validation instead.")

    modal_params, n_joint, n_total = most_frequent_joint_config(params)
    emit(f"\n  most frequent joint configuration, chosen together in "
         f"{n_joint}/{n_total} folds")
    results_json["final_config_joint_folds"] = n_joint
    # max_iter is only in the grid for elastic net, so it may or may not be
    # here. Drop it, otherwise it clashes with the max_iter passed below
    # (duplicate keyword). This fit only runs once, so it gets a high ceiling
    # either way: convergence matters more than speed here
    modal_params.pop("max_iter", None)
    emit(f"\n  settings: {modal_params}")

    final_estimator = LogisticRegression(
        class_weight=chosen_class_weight,
        max_iter=10000,
        random_state=RANDOM_STATE,
        **modal_params,
    )
    final_pipeline = build_pipeline(
        final_estimator, imbalance=chosen_imbalance, scale=True,
        random_state=RANDOM_STATE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X.to_numpy(dtype=float), y)

    coefficients = final_pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": coefficients,
        # features are standardised in the pipeline, so each coefficient is
        # the change in log odds per 1 SD. exp() gives an odds ratio per SD,
        # which can be compared across features on different scales
        "odds_ratio_per_sd": np.exp(coefficients),
        "abs_coefficient": np.abs(coefficients),
    })
    coef_df["selected"] = coef_df["coefficient"] != 0
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(os.path.join(RESULTS_DIR, f"logreg_coefficients{tag}.csv"),
                   index=False)

    n_selected = int(coef_df["selected"].sum())
    emit(f"\n  non-zero coefficients: {n_selected}/{len(features)}")
    if modal_params.get("penalty") in ("l1", "elasticnet"):
        emit(f"  The penalty zeroed {len(features) - n_selected} features,")
        emit(f"  which is the selection doing its job at this events-per-")
        emit(f"  variable ratio.")

    emit(f"\n  Largest 25 standardised effects:")
    emit(f"    {'feature':<44s} {'coef':>9s} {'OR/SD':>8s}")
    for _, row in coef_df.head(25).iterrows():
        emit(f"    {row['feature']:<44s} {row['coefficient']:>+9.4f} "
             f"{row['odds_ratio_per_sd']:>8.3f}")

    # split the surviving OBMs from the confounders (the SHAP analysis later
    # looks at this more carefully)
    with open(os.path.join(AIM2_DIR, "feature_manifest.json")) as handle:
        manifest = json.load(handle)
    confounder_set = set(manifest["confounders"])
    obm_top = coef_df[
        (~coef_df["feature"].isin(confounder_set)) & coef_df["selected"]
    ]
    conf_top = coef_df[
        coef_df["feature"].isin(confounder_set) & coef_df["selected"]
    ]
    emit(f"\n  surviving oximetry biomarkers : {len(obm_top)}")
    emit(f"  surviving confounders          : {len(conf_top)}")
    emit(f"\n  Top 12 oximetry biomarkers by absolute standardised effect:")
    for _, row in obm_top.head(12).iterrows():
        emit(f"    {row['feature']:<44s} {row['coefficient']:>+9.4f} "
             f"{row['odds_ratio_per_sd']:>8.3f}")
    emit(f"\n  Top confounders for reference:")
    for _, row in conf_top.head(10).iterrows():
        emit(f"    {row['feature']:<44s} {row['coefficient']:>+9.4f} "
             f"{row['odds_ratio_per_sd']:>8.3f}")

    results_json["final_model"] = {
        "settings": {k: (float(v) if isinstance(v, (int, float, np.floating))
                         else str(v))
                     for k, v in modal_params.items()},
        "imbalance": chosen_imbalance,
        "class_weight": str(chosen_class_weight),
        "n_features": len(features),
        "n_selected": n_selected,
        "top_features": coef_df.head(30)[
            ["feature", "coefficient", "odds_ratio_per_sd"]
        ].to_dict(orient="records"),
    }

    # --- plots ---
    primary_result, primary_y, _ = spec_results[primary_spec]
    scores = primary_result["mean_oof"]
    ok = ~np.isnan(scores)
    extra = []
    if "confounders_only" in spec_curves:
        yc, sc = spec_curves["confounders_only"]
        if np.array_equal(yc, primary_y[ok]):
            extra.append(("confounders only", sc))

    plot_model_performance(
        primary_y[ok], scores[ok],
        title=f"Logistic regression, {primary_spec}, out-of-fold predictions "
              f"({args.outcome})",
        path=os.path.join(RESULTS_DIR, f"logreg_performance{tag}.png"),
        extra_curves=extra,
    )
    plot_specification_comparison(
        spec_curves, primary_y[ok],
        title="Logistic regression by feature specification "
              "(nested CV out-of-fold)",
        path=os.path.join(RESULTS_DIR, f"logreg_specifications{tag}.png"),
    )

    # coefficient plot
    top = coef_df[coef_df["selected"]].head(25).iloc[::-1]
    if len(top):
        fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
        colours = [
            "tab:orange" if f in confounder_set else "tab:blue"
            for f in top["feature"]
        ]
        ax.barh(range(len(top)), top["coefficient"], color=colours)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"], fontsize=7)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("standardised coefficient (log odds per SD)")
        ax.set_title("Logistic regression, largest standardised effects\n"
                     "blue = oximetry biomarker, orange = clinical confounder",
                     fontsize=10)
        ax.grid(alpha=0.2, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"logreg_coefficients{tag}.png"),
                    dpi=140)
        plt.close(fig)

    # --- wrap up ---
    primary_metrics = spec_metrics[primary_spec]
    emit("\n" + "=" * 78)
    emit("HEADLINE RESULT")
    emit("=" * 78)
    emit(f"\n  Logistic regression, {primary_spec}, outcome {args.outcome}")
    emit(f"    AUC-ROC  {primary_metrics['auc_roc']:.4f} "
         f"[{primary_metrics['auc_ci_lo']:.3f}, "
         f"{primary_metrics['auc_ci_hi']:.3f}]")
    emit(f"    AUPRC    {primary_metrics['auprc']:.4f} "
         f"[{primary_metrics['auprc_ci_lo']:.3f}, "
         f"{primary_metrics['auprc_ci_hi']:.3f}] "
         f"(no skill {primary_metrics['auprc_baseline']:.4f}, "
         f"lift {primary_metrics['auprc_lift']:.2f}x)")
    emit(f"    sens {primary_metrics['sensitivity']:.3f} / "
         f"spec {primary_metrics['specificity']:.3f} "
         f"at Youden threshold {primary_metrics['threshold']:.4f}")
    emit(f"    balanced accuracy {primary_metrics['balanced_accuracy']:.4f}")
    emit(f"    Brier {primary_metrics['brier']:.4f}")
    emit(f"\n  imbalance strategy selected: {best_arm}")
    emit(f"  penalty selected           : {modal_params}")
    emit(f"\n  Elapsed: {(time.time() - started) / 60:.1f} min")

    # save out-of-fold scores so compare_models.py can run paired DeLong tests
    # against the other models on the same participants
    np.savez(
        os.path.join(RESULTS_DIR, f"logreg_oof_scores{tag}.npz"),
        **{spec: np.vstack([ys, ss]) for spec, (ys, ss) in spec_curves.items()},
    )

    with open(os.path.join(RESULTS_DIR, f"logreg_summary{tag}.txt"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(RESULTS_DIR, f"logreg_results{tag}.json"), "w") as handle:
        json.dump(results_json, handle, indent=2, default=str)

    print(f"\n  report  : {os.path.join(RESULTS_DIR, f'logreg_summary{tag}.txt')}")
    print(f"  metrics : {os.path.join(RESULTS_DIR, f'logreg_results{tag}.json')}")
    print(f"  plots   : {RESULTS_DIR}/logreg_*.png")
    print("\nNext: python3 aim2/train_random_forest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
