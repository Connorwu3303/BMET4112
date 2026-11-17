"""
Builds the Aim 2 results tables straight from the saved model outputs.

Every number in the write-up comes from the saved JSON and CSV files, so nothing
gets copied across by hand.

Reads the full-cohort results in processed/aim2/results/ and, for the half vs
full comparison, the archived half-cohort results in
processed/archive_half_cohort_2591/aim2/results/.

Output: processed/aim2/results/results_tables.md

Usage:
    python3 aim2/summarise_results.py
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

PROCESSED = "/Volumes/ThesisHD/Data/processed"
RESULTS = os.path.join(PROCESSED, "aim2", "results")
HALF = os.path.join(PROCESSED, "archive_half_cohort_2591", "aim2", "results")
OUT = os.path.join(RESULTS, "results_tables.md")

MODELS = [("Logistic regression", "logreg"), ("Random forest", "rf"), ("XGBoost", "xgb")]
SPECS = [("confounders_only", "Confounders only", 18),
         ("obm_only", "Biomarkers only", 34),
         ("obm_plus_confounders", "Biomarkers + confounders", 52)]
PRIMARY = "obm_plus_confounders"


def load(folder, prefix):
    path = os.path.join(folder, f"{prefix}_results.json")
    return json.load(open(path)) if os.path.exists(path) else None


def p_fmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def elapsed(folder, prefix):
    path = os.path.join(folder, f"{prefix}_summary.txt")
    if not os.path.exists(path):
        return None
    found = re.findall(r"Elapsed:\s*([\d.]+)\s*min", open(path).read())
    return float(found[-1]) if found else None


def mode_value(values):
    s = pd.Series(values)
    counts = s.value_counts(dropna=False)
    return counts.index[0], int(counts.iloc[0]), len(s)


def main():
    full = {p: load(RESULTS, p) for _, p in MODELS}
    half = {p: load(HALF, p) for _, p in MODELS}
    missing = [p for p, r in full.items() if r is None]
    if missing:
        print(f"Missing full-cohort results for: {missing}")
        return 1
    comparison = json.load(open(os.path.join(RESULTS, "model_comparison.json")))
    out = []
    w = out.append

    w("# Results tables (generated, do not edit by hand)\n")
    w("Every number below is read directly from the saved model outputs by "
      "`aim2/summarise_results.py`.\n")

    # cohort
    fm, hm = full["logreg"]["meta"], half["logreg"]["meta"]
    w("## Cohort\n")
    w("| | Half cohort | Full cohort |")
    w("|---|---|---|")
    w(f"| Participants in incident analysis | {hm['n_participants']:,} | {fm['n_participants']:,} |")
    w(f"| Incident CVD events | {hm['n_events']:,} | {fm['n_events']:,} |")
    w(f"| Event rate | {hm['event_rate']:.1%} | {fm['event_rate']:.1%} |")
    w(f"| Events per variable (52 features) | {hm['n_events'] / 52:.1f} | {fm['n_events'] / 52:.1f} |")
    w(f"| Excluded, prevalent CVD at baseline | {hm['n_dropped_outcome_undefined']:,} | {fm['n_dropped_outcome_undefined']:,} |\n")

    # headline
    w("## Headline, primary specification (biomarkers + confounders, 52 features)\n")
    w("| Model | AUC-ROC [95% CI] | AUPRC [95% CI] | Lift | Sensitivity | Specificity | Balanced acc. | Brier |")
    w("|---|---|---|---|---|---|---|---|")
    for name, p in MODELS:
        m = full[p]["stage3_specifications"][PRIMARY]
        w(f"| {name} | {m['auc_roc']:.4f} [{m['auc_ci_lo']:.3f}, {m['auc_ci_hi']:.3f}] "
          f"| {m['auprc']:.4f} [{m['auprc_ci_lo']:.3f}, {m['auprc_ci_hi']:.3f}] "
          f"| {m['auprc_lift']:.2f}x | {m['sensitivity']:.3f} | {m['specificity']:.3f} "
          f"| {m['balanced_accuracy']:.4f} | {m['brier']:.4f} |")
    base = full["logreg"]["stage3_specifications"][PRIMARY]["auprc_baseline"]
    w(f"\nAUPRC no-skill baseline = event rate = {base:.4f}. Sensitivity and specificity at "
      "each model's Youden threshold.\n")

    # by spec
    w("## AUC-ROC by feature specification\n")
    w("| Specification | Features | Logistic regression | Random forest | XGBoost |")
    w("|---|---|---|---|---|")
    for key, label, n in SPECS:
        cells = []
        for _, p in MODELS:
            m = full[p]["stage3_specifications"][key]
            cells.append(f"{m['auc_roc']:.4f} [{m['auc_ci_lo']:.3f}, {m['auc_ci_hi']:.3f}]")
        w(f"| {label} | {n} | " + " | ".join(cells) + " |")
    w("")

    # incremental value
    w("## Do the biomarkers add to the confounders? (paired DeLong, within each model)\n")
    w("| Model | Confounders AUC | + Biomarkers AUC | Difference | p |")
    w("|---|---|---|---|---|")
    for name, p in MODELS:
        t = full[p]["stage3_delong"]["confounders_only_vs_obm_plus_confounders"]
        w(f"| {name} | {t['auc_a']:.4f} | {t['auc_b']:.4f} | {t['difference']:+.4f} | {p_fmt(t['p_value'])} |")
    w("")

    # between models
    w("## Model versus model, primary specification (paired DeLong)\n")
    w("| Comparison | AUC A | AUC B | Difference (B minus A) | p |")
    w("|---|---|---|---|---|")
    for key, t in comparison["delong"].items():
        spec, pair = key.split(":")
        if spec != PRIMARY:
            continue
        a, b = pair.split("_vs_")
        w(f"| {a.replace('_', ' ')} vs {b.replace('_', ' ')} | {t['auc_a']:.4f} | {t['auc_b']:.4f} "
          f"| {t['difference']:+.4f} | {p_fmt(t['p_value'])} |")
    w("")

    # imbalance
    w("## Stage 1: class imbalance strategy (full cohort)\n")
    w("| Model | Strategy | AUC | AUPRC | Brier | Chosen |")
    w("|---|---|---|---|---|---|")
    for name, p in MODELS:
        best = full[p]["stage1_best_arm"]
        for arm, m in full[p]["stage1_imbalance_arms"].items():
            w(f"| {name} | {arm} | {m['auc_roc']:.4f} | {m['auprc']:.4f} | {m['brier']:.4f} | {'yes' if arm == best else ''} |")
    w("")

    # hyperparameters
    w("## Stage 2: selected hyperparameters (full cohort, across 15 outer folds)\n")
    lr = full["logreg"]
    w("**Logistic regression** (grid search, 49 candidates per inner search)\n")
    w("| Parameter | Selected (folds) |")
    w("|---|---|")
    for col, counts in lr["stage2_selected_params"].items():
        shown = ", ".join(f"{v} ({c}/15)" for v, c in
                          sorted(counts.items(), key=lambda kv: -kv[1]))
        w(f"| {col.replace('model__', '')} | {shown} |")
    fmod = lr["final_model"]
    w(f"\nFinal interpretation model: {fmod['n_selected']} of {fmod['n_features']} "
      "coefficients non-zero.\n")

    for name, p in (("Random forest", "rf"), ("XGBoost", "xgb")):
        st = full[p]["stage2"]
        rows = pd.DataFrame(st["selected_params_by_fold"])
        w(f"**{name}** (broad randomised search of {st['n_iter_broad']} candidates, then focused "
          f"grid, both nested inside each outer fold; focused grid chosen in "
          f"{st['folds_focused_won']}/{st['n_folds']} folds, mean inner gain "
          f"{st['mean_inner_refinement_gain']:+.4f})\n")
        w("| Parameter | Representative | Across folds |")
        w("|---|---|---|")
        reps = st.get("representative_params", {})
        for col in [c for c in rows.columns if c.startswith("model__")]:
            vals = rows[col]
            numeric = pd.to_numeric(vals, errors="coerce")
            key = col.replace("model__", "")
            if col in reps:
                rep = reps[col]
                rep_txt = f"{rep:.4g}" if isinstance(rep, float) else str(rep)
            else:
                v, c, n = mode_value(vals)
                rep_txt = f"{v} ({c}/{n})"
            if numeric.notna().all() and vals.nunique() > 3:
                spread = f"min {numeric.min():.4g}, median {numeric.median():.4g}, max {numeric.max():.4g}"
            else:
                spread = ", ".join(f"{v} x{c}" for v, c in vals.value_counts(dropna=False).items())
            w(f"| {key} | {rep_txt} | {spread} |")
        w("")

    # half vs full
    w("## Half cohort versus full cohort\n")
    w("| Model | Specification | Half AUC [95% CI] | Full AUC [95% CI] | CI width half | CI width full |")
    w("|---|---|---|---|---|---|")
    for name, p in MODELS[:2]:
        for key, label, _ in SPECS:
            h = half[p]["stage3_specifications"][key]
            f = full[p]["stage3_specifications"][key]
            w(f"| {name} | {label} | {h['auc_roc']:.4f} [{h['auc_ci_lo']:.3f}, {h['auc_ci_hi']:.3f}] "
              f"| {f['auc_roc']:.4f} [{f['auc_ci_lo']:.3f}, {f['auc_ci_hi']:.3f}] "
              f"| {h['auc_ci_hi'] - h['auc_ci_lo']:.3f} | {f['auc_ci_hi'] - f['auc_ci_lo']:.3f} |")
    w("")
    w("| Model | Half: biomarkers add? | Full: biomarkers add? |")
    w("|---|---|---|")
    for name, p in MODELS[:2]:
        h = half[p]["stage3_delong"]["confounders_only_vs_obm_plus_confounders"]
        f = full[p]["stage3_delong"]["confounders_only_vs_obm_plus_confounders"]
        w(f"| {name} | {h['difference']:+.4f} (p {p_fmt(h['p_value'])}) "
          f"| {f['difference']:+.4f} (p {p_fmt(f['p_value'])}) |")
    hc = json.load(open(os.path.join(HALF, "model_comparison.json")))["delong"]
    hkey = f"{PRIMARY}:logistic_regression_vs_random_forest"
    fkey = hkey
    if hkey in hc and fkey in comparison["delong"]:
        h, f = hc[hkey], comparison["delong"][fkey]
        w(f"\nForest minus logistic regression, primary specification: half {h['difference']:+.4f} "
          f"(p {p_fmt(h['p_value'])}), full {f['difference']:+.4f} (p {p_fmt(f['p_value'])}).\n")

    # features
    w("## Features\n")
    coef = pd.read_csv(os.path.join(RESULTS, "logreg_coefficients.csv"))
    manifest = json.load(open(os.path.join(PROCESSED, "aim2", "feature_manifest.json")))
    conf = set(manifest["confounders"])
    sel = coef[coef["selected"]]
    obm_sel = sel[~sel["feature"].isin(conf)]
    w(f"Logistic regression kept {len(sel)} of {len(coef)} features "
      f"({len(obm_sel)} biomarkers, {len(sel) - len(obm_sel)} confounders).\n")
    w("| Rank | LR biomarker (OR per SD) | RF biomarker (permutation) | XGBoost biomarker (permutation) |")
    w("|---|---|---|---|")
    rf_imp = pd.read_csv(os.path.join(RESULTS, "rf_importance.csv"))
    xg_imp = pd.read_csv(os.path.join(RESULTS, "xgb_importance.csv"))
    rf_o = rf_imp[~rf_imp["feature"].isin(conf)].head(6).reset_index(drop=True)
    xg_o = xg_imp[~xg_imp["feature"].isin(conf)].head(6).reset_index(drop=True)
    lr_o = obm_sel.reindex(obm_sel["coefficient"].abs().sort_values(ascending=False).index).head(6).reset_index(drop=True)
    for i in range(6):
        a = f"`{lr_o.loc[i, 'feature']}` ({lr_o.loc[i, 'odds_ratio_per_sd']:.3f})" if i < len(lr_o) else ""
        b = f"`{rf_o.loc[i, 'feature']}` ({rf_o.loc[i, 'perm_importance_mean']:+.4f})" if i < len(rf_o) else ""
        c = f"`{xg_o.loc[i, 'feature']}` ({xg_o.loc[i, 'perm_importance_mean']:+.4f})" if i < len(xg_o) else ""
        w(f"| {i + 1} | {a} | {b} | {c} |")
    for name, imp in (("Random forest", rf_imp), ("XGBoost", xg_imp)):
        pos = int((imp["perm_importance_mean"] > 0).sum())
        cons = int(((imp["perm_importance_mean"] > 0) & (imp["perm_folds_positive"] == imp["n_folds"])).sum())
        w(f"\n{name}: {pos} of {len(imp)} features positive on average, {cons} helped in every fold.")
    age = coef[coef["feature"] == "age_s1"].iloc[0]
    w(f"\nAge odds ratio per SD (logistic regression): {age['odds_ratio_per_sd']:.3f}.\n")

    # runtime
    w("## Runtime\n")
    w("| Model | Half cohort (min) | Full cohort (min) |")
    w("|---|---|---|")
    for name, p in MODELS:
        h, f = elapsed(HALF, p), elapsed(RESULTS, p)
        w(f"| {name} | {h if h is not None else '-'} | {f if f is not None else '-'} |")

    text = "\n".join(out) + "\n"
    assert "\u2014" not in text
    with open(OUT, "w") as handle:
        handle.write(text)
    print(text)
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
