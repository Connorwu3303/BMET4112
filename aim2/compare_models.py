"""
Paired head-to-head comparison of the Aim 2 classical models.

Loads the out-of-fold predictions saved by each model script and compares them
properly. Reading two AUCs off two reports and picking the bigger one isn't a
real comparison: with a few hundred events one AUC has a standard error of
several points, so a 0.02 gap can be noise.

All models were scored on the same participants with the same folds (every
model script takes its folds from RANDOM_STATE in ml_common), so the test is
paired. Sharing participants makes the two AUCs correlated, and DeLong's test
handles this by estimating their covariance from the placement values. An
unpaired test would inflate the standard error of the difference and miss real
effects.

Also reports decision-curve net benefit, since a model can win on AUC and still
be useless at every threshold a clinician would pick.

Outputs (processed/aim2/results/): model_comparison.csv, .txt, .json, .png

Usage:
    python3 aim2/compare_models.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ml_common import (
    RESULTS_DIR,
    SPECIFICATIONS,
    bootstrap_auc_ci,
    delong_test,
    ensure_results_dir,
    evaluate_predictions,
)

ensure_results_dir()

MODEL_FILES = {
    "logistic_regression": "logreg_oof_scores.npz",
    "random_forest": "rf_oof_scores.npz",
    "xgboost": "xgb_oof_scores.npz",
}

PRIMARY_SPEC = "obm_plus_confounders"

# thresholds for the net benefit curve. With a 25 to 30% event rate nobody
# would screen at more than about 40% predicted risk, so stop there
THRESHOLD_RANGE = np.linspace(0.01, 0.40, 60)


def net_benefit(y_true, y_score, threshold):
    """
    Decision-curve net benefit at one threshold probability.

        NB = TP/n - (FP/n) * (pt / (1 - pt))

    pt/(1-pt) is the odds at the threshold. Acting at 10% risk means accepting
    nine false positives per true positive. NB is true positives gained per
    participant after paying for the false positives, which is easier to read
    clinically than AUC.
    """
    y_true = np.asarray(y_true)
    predicted = np.asarray(y_score) >= threshold
    n = len(y_true)
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    odds = threshold / (1.0 - threshold)
    return tp / n - (fp / n) * odds


def main():
    report = []

    def emit(text=""):
        print(text)
        report.append(text)

    emit("=" * 78)
    emit("AIM 2 - PAIRED MODEL COMPARISON")
    emit("=" * 78)

    loaded = {}
    for model, filename in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            loaded[model] = np.load(path)
            emit(f"\n  loaded {model:<22s} from {filename}")
        else:
            emit(f"\n  MISSING {model:<21s} ({filename} not found)")

    if len(loaded) < 2:
        emit("\n  Need at least two fitted models to compare. Run")
        emit("  train_logistic_regression.py and train_random_forest.py first.")
        return 1

    models = list(loaded)

    # --- per specification, per model summary ---
    emit("\n" + "=" * 78)
    emit("PERFORMANCE BY MODEL AND SPECIFICATION")
    emit("=" * 78)
    emit(f"\n  {'specification':<24s} {'model':<22s} {'AUC':>8s} "
         f"{'95% CI':>18s} {'AUPRC':>8s} {'Brier':>8s}")

    table = []
    for spec in SPECIFICATIONS:
        for model in models:
            if spec not in loaded[model]:
                continue
            arr = loaded[model][spec]
            y_true, y_score = arr[0].astype(int), arr[1]
            metrics = evaluate_predictions(y_true, y_score)
            ci = bootstrap_auc_ci(y_true, y_score, n_boot=2000)
            emit(f"  {spec:<24s} {model:<22s} {metrics['auc_roc']:>8.4f} "
                 f"  [{ci['auc_ci_lo']:.3f}, {ci['auc_ci_hi']:.3f}] "
                 f"{metrics['auprc']:>8.4f} {metrics['brier']:>8.4f}")
            table.append({
                "specification": spec, "model": model,
                **{k: v for k, v in metrics.items()}, **ci,
            })
    pd.DataFrame(table).to_csv(
        os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False
    )

    # --- paired DeLong between models, within each specification ---
    emit("\n" + "=" * 78)
    emit("PAIRED DELONG TESTS BETWEEN MODELS")
    emit("=" * 78)
    emit("\n  Within each specification, both models scored the same")
    emit("  participants using the same folds, so this is a paired comparison.")

    comparisons = {}
    for spec in SPECIFICATIONS:
        available = [m for m in models if spec in loaded[m]]
        if len(available) < 2:
            continue
        emit(f"\n  {spec}:")
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                a, b = available[i], available[j]
                arr_a, arr_b = loaded[a][spec], loaded[b][spec]
                y_a, s_a = arr_a[0].astype(int), arr_a[1]
                y_b, s_b = arr_b[0].astype(int), arr_b[1]
                if len(y_a) != len(y_b) or not np.array_equal(y_a, y_b):
                    emit(f"    {a} vs {b}: SKIP, participant sets differ "
                         f"({len(y_a)} vs {len(y_b)})")
                    continue
                test = delong_test(y_a, s_a, s_b)
                if not test:
                    continue
                comparisons[f"{spec}:{a}_vs_{b}"] = test
                verdict = "significant" if test["p_value"] < 0.05 \
                    else "not significant"
                emit(f"    {a} -> {b}")
                emit(f"      AUC {test['auc_a']:.4f} -> {test['auc_b']:.4f}, "
                     f"difference {test['difference']:+.4f} "
                     f"(se {test['se']:.4f})")
                emit(f"      z = {test['z']:+.3f}, p = {test['p_value']:.4g}  "
                     f"({verdict})")

    # --- net benefit ---
    emit("\n" + "=" * 78)
    emit("DECISION CURVE ANALYSIS")
    emit("=" * 78)
    emit("\n  Net benefit against two reference strategies:")
    emit("    treat all  intervene on everyone, benefit falls as the")
    emit("               threshold rises because the false positive cost grows")
    emit("    treat none intervene on nobody, net benefit exactly zero")
    emit("\n  A model is only clinically useful where its curve sits above")
    emit("  BOTH references. A model can have a respectable AUC and still fail")
    emit("  this test across every plausible threshold.")

    nb_rows = []
    for model in models:
        if PRIMARY_SPEC not in loaded[model]:
            continue
        arr = loaded[model][PRIMARY_SPEC]
        y_true, y_score = arr[0].astype(int), arr[1]
        prevalence = float(np.mean(y_true))
        for threshold in THRESHOLD_RANGE:
            nb_rows.append({
                "model": model, "threshold": threshold,
                "net_benefit": net_benefit(y_true, y_score, threshold),
                "treat_all": prevalence - (1 - prevalence) * (
                    threshold / (1 - threshold)
                ),
            })
    nb = pd.DataFrame(nb_rows)

    if not nb.empty:
        emit(f"\n  Net benefit at selected thresholds (spec {PRIMARY_SPEC}):")
        emit(f"    {'threshold':>10s} " +
             " ".join(f"{m[:16]:>17s}" for m in models) +
             f" {'treat all':>11s}")
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.30):
            row = nb.iloc[(nb["threshold"] - threshold).abs().argsort()[:1]]
            actual = float(row["threshold"].iloc[0])
            values = []
            for model in models:
                sel = nb[(nb["model"] == model)
                         & (np.isclose(nb["threshold"], actual))]
                values.append(float(sel["net_benefit"].iloc[0])
                              if len(sel) else np.nan)
            treat_all = float(row["treat_all"].iloc[0])
            emit(f"    {actual:>10.3f} " +
                 " ".join(f"{v:>17.5f}" for v in values) +
                 f" {treat_all:>11.5f}")

        # threshold range where each model beats treat all and treat none
        emit("\n  Threshold range where each model beats both references:")
        for model in models:
            sel = nb[nb["model"] == model]
            useful = sel[(sel["net_benefit"] > sel["treat_all"])
                         & (sel["net_benefit"] > 0)]
            if len(useful):
                emit(f"    {model:<22s} {useful['threshold'].min():.3f} "
                     f"to {useful['threshold'].max():.3f}")
            else:
                emit(f"    {model:<22s} never beats both references in "
                     f"{THRESHOLD_RANGE.min():.2f}-{THRESHOLD_RANGE.max():.2f}")

    # --- figure: ROC, precision-recall, decision curve ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for model in models:
        if PRIMARY_SPEC not in loaded[model]:
            continue
        arr = loaded[model][PRIMARY_SPEC]
        y_true, y_score = arr[0].astype(int), arr[1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        axes[0].plot(fpr, tpr, lw=1.8,
                     label=f"{model} ({roc_auc_score(y_true, y_score):.3f})")
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        axes[1].plot(recall, precision, lw=1.8,
                     label=f"{model} "
                           f"({average_precision_score(y_true, y_score):.3f})")

    axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"ROC, {PRIMARY_SPEC}")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.2)

    first = next(m for m in models if PRIMARY_SPEC in loaded[m])
    baseline = float(np.mean(loaded[first][PRIMARY_SPEC][0]))
    axes[1].axhline(baseline, color="k", ls=":", lw=1,
                    label=f"no skill ({baseline:.3f})")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"Precision-recall, {PRIMARY_SPEC}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    if not nb.empty:
        for model in models:
            sel = nb[nb["model"] == model]
            if len(sel):
                axes[2].plot(sel["threshold"], sel["net_benefit"], lw=1.8,
                             label=model)
        ref = nb[nb["model"] == models[0]]
        axes[2].plot(ref["threshold"], ref["treat_all"], "k--", lw=1,
                     label="treat all")
        axes[2].axhline(0, color="k", ls=":", lw=1, label="treat none")
        axes[2].set_xlabel("threshold probability")
        axes[2].set_ylabel("net benefit")
        axes[2].set_title("Decision curve")
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.2)
        axes[2].set_ylim(bottom=min(-0.02, nb["net_benefit"].min()))

    fig.suptitle("Aim 2 classical models, nested CV out-of-fold predictions",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=140)
    plt.close(fig)

    # --- summary ---
    emit("\n" + "=" * 78)
    emit("SUMMARY")
    emit("=" * 78)
    best = max(
        (r for r in table if r["specification"] == PRIMARY_SPEC),
        key=lambda r: r["auc_roc"], default=None,
    )
    if best:
        emit(f"\n  Best model on {PRIMARY_SPEC}: {best['model']}")
        emit(f"    AUC-ROC {best['auc_roc']:.4f} "
             f"[{best['auc_ci_lo']:.3f}, {best['auc_ci_hi']:.3f}]")
        emit(f"    AUPRC   {best['auprc']:.4f} "
             f"(no skill {best['auprc_baseline']:.4f})")
    emit("\n  Read the DeLong p values above before claiming one model beats")
    emit("  another. Overlapping confidence intervals with a non-significant")
    emit("  paired test means the two models are indistinguishable on this")
    emit("  cohort, which is itself a reportable finding.")

    with open(os.path.join(RESULTS_DIR, "model_comparison.txt"), "w") as handle:
        handle.write("\n".join(report) + "\n")
    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w") as handle:
        json.dump({"delong": comparisons, "table": table}, handle,
                  indent=2, default=str)

    print(f"\n  report : {os.path.join(RESULTS_DIR, 'model_comparison.txt')}")
    print(f"  plot   : {os.path.join(RESULTS_DIR, 'model_comparison.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
