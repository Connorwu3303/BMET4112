"""
Aim 3: combines the deep score with the clinical variables and tests what it adds.

C1 asks whether the deep score alone beats the Aim 2 biomarker model alone
(paired DeLong). C2, the main question, asks whether it adds to the clinical
model. DeLong is unreliable for nested models, so C2's primary test is a
likelihood ratio test on the deep score's coefficient. It's fitted on the outer
test predictions, not the training rows the epoch was picked on, with the
clinical logit as a fixed offset. The change in AUC is the effect size, judged
against a pre-registered equivalence margin.

The stacker reuses the hyperparameters the Aim 2 confounders only model picked
in each fold (from aim2_comparators.py) instead of searching its own, so the
two sides of C2 differ by exactly one column.

Outputs (processed/aim3/results/): stack_eval_<arm>.json, fold_detail_<arm>.csv,
oof_<arm>.npz (with --tag appended to each name if given)

Usage:
    python3 aim3/stack_eval.py --arm A
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.metrics import roc_auc_score             # noqa: E402

import ml_common as MC                                # noqa: E402
import data as D                                      # noqa: E402
from folds import get_verified_folds                  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
SELECT_DIR = os.path.join(RESULTS, "selected")

# fixed before any run that used the CVD labels
EQUIVALENCE_MARGIN = 0.005
N_BOOT = 2000
RANDOM_STATE = 42


def frozen_params(repeat, fold, spec="confounders_only"):
    """
    Hyperparameters the Aim 2 model picked for this fold.

    Read from disk instead of searched again, so the stacked model and its
    comparator differ by exactly one column.
    """
    path = os.path.join(RESULTS, "aim2_frozen_params_by_fold.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run aim2_comparators.py first: the stacker must "
            f"reuse the Aim 2 hyperparameters rather than search its own.")
    t = pd.read_csv(path)
    row = t[(t["specification"] == spec) & (t["repeat"] == repeat) &
            (t["fold"] == fold)]
    if len(row) != 1:
        raise ValueError(f"expected one row for {spec} repeat {repeat} "
                         f"fold {fold}, found {len(row)}")
    row = row.iloc[0]
    out = {"C": float(row["model__C"]), "penalty": str(row["model__penalty"]),
           "solver": str(row["model__solver"])}
    if out["penalty"] == "elasticnet":
        out["l1_ratio"] = float(row["model__l1_ratio"])
        out["max_iter"] = int(row["model__max_iter"])
    return out


def fit_frozen_lr(x_train, y_train, x_test, params, class_weight="balanced"):
    """
    Fits the Aim 2 logistic regression with fixed hyperparameters.

    Median imputation and standardisation are fitted on the training rows only,
    same as Aim 2 did by keeping them inside the cross validated pipeline, so
    nothing leaks in from the test rows.
    """
    tf = D.fit_confounder_transform(x_train)
    xt = D.apply_confounder_transform(x_train, tf)
    xe = D.apply_confounder_transform(x_test, tf)
    kwargs = {"C": params["C"], "penalty": params["penalty"],
              "solver": params["solver"], "class_weight": class_weight,
              "random_state": RANDOM_STATE,
              "max_iter": params.get("max_iter", 2000)}
    if params["penalty"] == "elasticnet":
        kwargs["l1_ratio"] = params["l1_ratio"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(**kwargs)
        lr.fit(xt, y_train)
    return (lr.predict_proba(xe)[:, 1], lr.decision_function(xe), lr, tf)


def platt(train_score, train_y, test_score):
    """
    Platt recalibration: a one variable logistic regression on the score.

    Fitted on the cross fitted scores of the outer training rows and applied to
    the test scores, so no test information goes into calibrating the test
    predictions.
    """
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(train_score.reshape(-1, 1), train_y)
    return lr.predict_proba(test_score.reshape(-1, 1))[:, 1]


def held_out_coefficient_test(y, offset_logit, z):
    """
    Tests whether the deep score adds to the clinical model, on held out rows.

    Fits y ~ offset(clinical logit) + beta * z and runs a likelihood ratio test
    of beta = 0. The offset fixes the clinical coefficient at one, so the
    regression can't rescale the clinical model to fit these rows and the test
    only asks what the deep score adds to the clinical prediction as it is.
    """
    import statsmodels.api as sm
    ok = np.isfinite(offset_logit) & np.isfinite(z) & np.isfinite(y)
    y, off, zz = y[ok], offset_logit[ok], z[ok]
    # standardised so beta is per SD and comparable with the odds ratios
    # Aim 2 reports for its biomarkers
    zs = (zz - zz.mean()) / (zz.std() if zz.std() > 1e-12 else 1.0)
    try:
        null = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(),
                      offset=off).fit()
        full = sm.GLM(y, sm.add_constant(zs), family=sm.families.Binomial(),
                      offset=off).fit()
        from scipy import stats
        lr_stat = float(null.deviance - full.deviance)
        p = float(stats.chi2.sf(lr_stat, df=1))
        beta = float(full.params[1])
        se = float(full.bse[1])
        return {"n": int(ok.sum()), "beta": beta, "se": se,
                "or_per_sd": float(np.exp(beta)),
                "or_ci_lo": float(np.exp(beta - 1.96 * se)),
                "or_ci_hi": float(np.exp(beta + 1.96 * se)),
                "lr_chi2": lr_stat, "p_value": p,
                "wald_z": beta / se if se > 0 else float("nan")}
    except Exception as exc:
        return {"n": int(ok.sum()), "error": str(exc)}


def paired_bootstrap_delta_auc(y, score_a, score_b, n_boot=N_BOOT, seed=42):
    """
    Bootstraps the difference in AUC, resampling participants.

    Paired because both scores are for the same people. Resampling participants
    keeps the correlation between the two models, which is where the comparison
    gets its power.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], score_b[idx])
                      - roc_auc_score(y[idx], score_a[idx]))
    deltas = np.array(deltas)
    return {"delta": float(roc_auc_score(y, score_b)
                           - roc_auc_score(y, score_a)),
            "se_boot": float(deltas.std()),
            "ci_lo": float(np.percentile(deltas, 2.5)),
            "ci_hi": float(np.percentile(deltas, 97.5)),
            "ci90_lo": float(np.percentile(deltas, 5.0)),
            "ci90_hi": float(np.percentile(deltas, 95.0)),
            "n_boot_used": int(len(deltas))}


def available(arm, tag=""):
    """Which (repeat, fold) selections exist on disk for this arm."""
    out = []
    if not os.path.isdir(SELECT_DIR):
        return out
    for f in sorted(os.listdir(SELECT_DIR)):
        if not f.endswith(".npz") or not f.startswith(f"{arm}_r"):
            continue
        if tag and not f.endswith(f"{tag}.npz"):
            continue
        if not tag and any(f.endswith(f"{t}.npz")
                           for t in ("_perm", "_pos", "_smoke")):
            continue
        stem = f[:-4]
        parts = stem.split("_")
        out.append((int(parts[1][1:]), int(parts[2][1:]), f))
    return out


def assemble(arm, data, tag=""):
    """
    Builds the per repeat out of fold matrices for the deep score, the matched
    clinical comparator and the stacked model.

    Returns a dict of (n_repeats, n_participants) arrays in cohort row order.
    They're filled by participant ID since the evaluation loaders are length
    sorted, so position doesn't line up with the cohort.
    """
    y = data["y"]
    pid = data["pid"]
    conf = data["confounders"]
    pid_to_row = {int(p): i for i, p in enumerate(pid)}
    n = len(y)
    repeats = sorted({r for r, _, _ in available(arm, tag)})
    if not repeats:
        raise SystemExit(f"no selected outputs found for arm {arm}; "
                         f"run run_unit.py then select_epoch.py")

    keys = ["deep_marg", "deep_inc", "clinical", "stacked", "deep_platt"]
    mats = {k: np.full((len(repeats), n), np.nan) for k in keys}
    clinical_logit = np.full((len(repeats), n), np.nan)
    fold_rows = []

    for ri, repeat in enumerate(repeats):
        for r, fold, fname in available(arm, tag):
            if r != repeat:
                continue
            z = np.load(os.path.join(SELECT_DIR, fname), allow_pickle=True)
            test_pid = z["test_pid"].astype(int)
            test_rows = np.array([pid_to_row[int(p)] for p in test_pid])
            train_rows = np.array([pid_to_row[int(p)]
                                   for p in z["outer_train_pid"].astype(int)])

            mats["deep_marg"][ri, test_rows] = z["test_z_marg"]
            mats["deep_inc"][ri, test_rows] = z["test_z_inc"]

            params = frozen_params(repeat, fold)

            # matched clinical comparator: the Aim 2 model on the same fold with
            # the same frozen hyperparameters, fitted here so the stacked model
            # differs from it by one column
            p_clin, l_clin, _, _ = fit_frozen_lr(
                conf[train_rows], y[train_rows], conf[test_rows], params)
            mats["clinical"][ri, test_rows] = p_clin
            clinical_logit[ri, test_rows] = l_clin

            # stacked model: same thing plus the cross fitted deep score
            train_z = z["train_z_inc"][train_rows]
            if not np.all(np.isfinite(train_z)):
                raise RuntimeError(
                    f"{fname}: cross fitted deep scores missing for "
                    f"{int((~np.isfinite(train_z)).sum())} outer training rows")
            x_train = np.column_stack([conf[train_rows], train_z])
            x_test = np.column_stack([conf[test_rows], z["test_z_inc"]])
            p_stack, _, lr_stack, _ = fit_frozen_lr(
                x_train, y[train_rows], x_test, params)
            mats["stacked"][ri, test_rows] = p_stack

            # calibrated standalone deep score, for the head to head (C1)
            tr_marg = z["train_z_marg"][train_rows]
            mats["deep_platt"][ri, test_rows] = platt(
                tr_marg, y[train_rows], z["test_z_marg"])

            fold_rows.append({
                "arm": arm, "repeat": repeat, "fold": fold,
                "e_marg": int(z["e_marg"]), "e_inc": int(z["e_inc"]),
                "n_test": len(test_rows),
                "n_test_events": int(y[test_rows].sum()),
                "auc_deep_marg": float(roc_auc_score(
                    y[test_rows], z["test_z_marg"]))
                if len(np.unique(y[test_rows])) > 1 else np.nan,
                "auc_clinical": float(roc_auc_score(y[test_rows], p_clin)),
                "auc_stacked": float(roc_auc_score(y[test_rows], p_stack)),
                "deep_coef_in_stack": float(lr_stack.coef_[0][-1]),
                # spread across the three inner models on this test fold, i.e.
                # how much a single model actually varies
                "inner_auc_spread": float(np.nanmax([
                    roc_auc_score(y[test_rows], zm)
                    for zm in z["test_z_marg_per_model"]])
                    - np.nanmin([roc_auc_score(y[test_rows], zm)
                                 for zm in z["test_z_marg_per_model"]]))
                if len(np.unique(y[test_rows])) > 1 else np.nan,
            })

    return {"repeats": repeats, "mats": mats,
            "clinical_logit": clinical_logit,
            "folds": pd.DataFrame(fold_rows)}


def summarise(name, y, score):
    """Headline metrics for one prediction vector."""
    ok = np.isfinite(score)
    m = MC.evaluate_predictions(y[ok], score[ok])
    ci = MC.bootstrap_auc_ci(y[ok], score[ok], n_boot=N_BOOT)
    out = {"model": name, "n": int(ok.sum()), "n_events": int(y[ok].sum())}
    for k in ("auc_roc", "auprc", "auprc_baseline", "auprc_lift", "brier"):
        out[k] = m.get(k)
    # bootstrap_auc_ci returns a dict. An earlier version read it like a tuple,
    # which the chain validation caught before the long run.
    if ci:
        for k in ("auc_ci_lo", "auc_ci_hi", "auprc_ci_lo", "auprc_ci_hi",
                  "n_boot_used"):
            if k in ci:
                out[k] = ci[k]
    # calibration slope and intercept on the logit scale
    p = np.clip(score[ok], 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        import statsmodels.api as sm
        fit = sm.GLM(y[ok], sm.add_constant(lp),
                     family=sm.families.Binomial()).fit()
        out["calibration_intercept"] = float(fit.params[0])
        out["calibration_slope"] = float(fit.params[1])
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B", "NC", "NCA", "D"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 76)
    print(f"AIM 3 FUSION AND EVALUATION: arm {args.arm}")
    print("=" * 76)

    data = get_verified_folds(quiet=True)
    y = data["y"]
    asm = assemble(args.arm, data, args.tag)
    repeats = asm["repeats"]
    print(f"\nrepeats available: {repeats}  "
          f"({len(asm['folds'])} outer folds)")
    print(f"selected epochs: E*_marg "
          f"{sorted(asm['folds']['e_marg'].tolist())}")
    print(f"                 E*_inc  "
          f"{sorted(asm['folds']['e_inc'].tolist())}")

    # Repeat averaged predictions, same as Aim 2's headline. Anyone with no
    # prediction in any repeat gives an all NaN column and numpy warns. That
    # only happens when the fold set is incomplete, and the coverage message
    # below reports it, so the warning is silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = {k: np.nanmean(v, axis=0) for k, v in asm["mats"].items()}
        mean_clin_logit = np.nanmean(asm["clinical_logit"], axis=0)

    covered = int(np.isfinite(mean["stacked"]).sum())
    print(f"\ncoverage: {covered} of {len(y)} participants have a prediction")
    if covered < len(y):
        print(f"  {len(y) - covered} participants are unscored because only "
              f"{len(asm['folds'])} of {5 * len(repeats)} outer folds are "
              f"present. Every figure below is computed on the {covered} that "
              f"are scored, so it is a partial result, not the headline.")

    print("\n" + "-" * 76)
    print("HEADLINE PERFORMANCE (repeat averaged out-of-fold)")
    print("-" * 76)
    table = [summarise("deep SpO2 only (calibrated)", y, mean["deep_platt"]),
             summarise("clinical confounders only", y, mean["clinical"]),
             summarise("clinical plus deep", y, mean["stacked"])]
    tdf = pd.DataFrame(table)
    print(f"\n  {'model':<30s} {'AUC':>7s} {'95% CI':>17s} {'AUPRC':>7s} "
          f"{'Brier':>7s} {'slope':>6s}")
    for _, r in tdf.iterrows():
        print(f"  {r['model']:<30s} {r['auc_roc']:7.4f} "
              f"[{r.get('auc_ci_lo', float('nan')):.3f}, "
              f"{r.get('auc_ci_hi', float('nan')):.3f}] "
              f"{r['auprc']:7.4f} {r['brier']:7.4f} "
              f"{r.get('calibration_slope', float('nan')):6.3f}")

    # --- C2: does the deep score add to the clinical model ---
    print("\n" + "-" * 76)
    print("C2, THE MAIN CONTRAST: does the deep score add to the clinical model?")
    print("-" * 76)
    print("\n  Primary inference: likelihood ratio test of the deep coefficient,")
    print("  fitted on held out predictions with the clinical logit as a fixed")
    print("  offset, so nothing is fitted on the rows it is evaluated on.")
    ht = held_out_coefficient_test(y, mean_clin_logit, mean["deep_inc"])
    if "error" in ht:
        print(f"    test failed: {ht['error']}")
    else:
        print(f"    odds ratio per SD of the deep score: "
              f"{ht['or_per_sd']:.3f} "
              f"[{ht['or_ci_lo']:.3f}, {ht['or_ci_hi']:.3f}]")
        print(f"    likelihood ratio chi-square {ht['lr_chi2']:.3f} on 1 df, "
              f"p = {ht['p_value']:.4g}")

    per_repeat = []
    for ri, repeat in enumerate(repeats):
        r = held_out_coefficient_test(
            y, asm["clinical_logit"][ri], asm["mats"]["deep_inc"][ri])
        r["repeat"] = repeat
        per_repeat.append(r)
        if "error" not in r:
            print(f"    repeat {repeat}: OR {r['or_per_sd']:.3f}, "
                  f"p = {r['p_value']:.4g}")

    print("\n  Effect size: change in area under the curve, with a paired")
    print("  bootstrap over participants.")
    ok = np.isfinite(mean["stacked"]) & np.isfinite(mean["clinical"])
    boot = paired_bootstrap_delta_auc(y[ok], mean["clinical"][ok],
                                      mean["stacked"][ok])
    # The bootstrap SE ignores variability between training runs, so the
    # between repeat spread of the difference is added to it. The bootstrap
    # interval alone would overstate the precision.
    deltas_per_repeat = []
    for ri in range(len(repeats)):
        m = np.isfinite(asm["mats"]["stacked"][ri]) & \
            np.isfinite(asm["mats"]["clinical"][ri])
        deltas_per_repeat.append(
            roc_auc_score(y[m], asm["mats"]["stacked"][ri][m])
            - roc_auc_score(y[m], asm["mats"]["clinical"][ri][m]))
    between = float(np.std(deltas_per_repeat, ddof=1)) \
        if len(deltas_per_repeat) > 1 else 0.0
    se_total = float(np.sqrt(boot["se_boot"] ** 2 + between ** 2))
    print(f"    change in AUC {boot['delta']:+.4f}")
    print(f"    bootstrap SE {boot['se_boot']:.4f}, "
          f"between repeat SD {between:.4f}, total SE {se_total:.4f}")
    lo90 = boot["delta"] - 1.645 * se_total
    hi90 = boot["delta"] + 1.645 * se_total
    print(f"    90 percent interval using the total SE: "
          f"[{lo90:+.4f}, {hi90:+.4f}]")
    print(f"    per repeat changes: "
          f"{', '.join(f'{d:+.4f}' for d in deltas_per_repeat)}")
    inside = abs(lo90) < EQUIVALENCE_MARGIN and abs(hi90) < EQUIVALENCE_MARGIN
    print(f"\n    pre-registered equivalence margin "
          f"{EQUIVALENCE_MARGIN:+.3f} AUC")
    print(f"    the interval lies {'INSIDE' if inside else 'OUTSIDE'} the "
          f"margin, so the result is "
          f"{'an informative null' if inside else 'inconclusive on equivalence'}")

    # DeLong as well, to match how Aim 2 reported everything
    dl = MC.delong_test(y[ok], mean["clinical"][ok], mean["stacked"][ok])
    if dl:
        print(f"\n    DeLong (reported for continuity with Aim 2, and "
              f"conservative for nested models):")
        print(f"      z = {dl['z']:+.3f}, p = {dl['p_value']:.4g}")

    # --- C1: deep score vs the Aim 2 biomarker model ---
    print("\n" + "-" * 76)
    print("C1: the deep score against the Aim 2 biomarker model, standing alone")
    print("-" * 76)
    # The comparator has to be matched on repeats as well as participants.
    # Aim 2's headline is an average over three repeats, which is less noisy
    # than any single repeat, so comparing a single repeat deep score with it
    # isn't like for like and favours the biomarker arm. An audit of the write
    # up caught this, so both contrasts are computed and the repeat matched one
    # is primary.
    per_repeat_path = os.path.join(RESULTS, "aim2_oof_per_repeat.npz")
    ref = os.path.join("/Volumes/ThesisHD/Data/processed/aim2/results",
                       "logreg_oof_scores.npz")
    c1 = c1_published = None
    if os.path.exists(per_repeat_path):
        with np.load(per_repeat_path) as z:
            bio_pr = z["obm_only__per_repeat"]
        # average only the repeats this run used, so both sides rest on the
        # same repeats
        bio_matched = np.nanmean(bio_pr[repeats, :], axis=0)
        m = np.isfinite(mean["deep_platt"]) & np.isfinite(bio_matched)
        a_bio = roc_auc_score(y[m], bio_matched[m])
        a_deep = roc_auc_score(y[m], mean["deep_platt"][m])
        c1 = MC.delong_test(y[m], bio_matched[m], mean["deep_platt"][m])
        print(f"  repeat matched (both sides on repeat(s) {repeats}):")
        print(f"    Aim 2, 34 biomarkers      AUC {a_bio:.4f}")
        print(f"    Aim 3, deep SpO2 only     AUC {a_deep:.4f}")
        print(f"    difference {a_deep - a_bio:+.4f}")
        if c1:
            print(f"    paired DeLong z = {c1['z']:+.3f}, "
                  f"p = {c1['p_value']:.4g}")
            lo = c1["difference"] - 1.96 * c1["se"]
            hi = c1["difference"] + 1.96 * c1["se"]
            print(f"    95 percent interval on the difference "
                  f"[{lo:+.4f}, {hi:+.4f}]")
            print(f"    that interval is {abs(hi - lo) / (2 * EQUIVALENCE_MARGIN):.1f} "
                  f"times the width of the equivalence margin, so this is a "
                  f"failure to separate the two models, not a demonstration "
                  f"that they are equivalent")
            c1["ci_lo"], c1["ci_hi"] = float(lo), float(hi)
    if os.path.exists(ref):
        with np.load(ref) as z:
            bio_pub = z["obm_only"][1]
        m2 = np.isfinite(mean["deep_platt"]) & np.isfinite(bio_pub)
        c1_published = MC.delong_test(y[m2], bio_pub[m2],
                                      mean["deep_platt"][m2])
        if c1_published:
            print(f"\n  against Aim 2's published three repeat average "
                  f"(not repeat matched, shown for continuity):")
            print(f"    Aim 2 AUC {c1_published['auc_a']:.4f}, "
                  f"difference {c1_published['difference']:+.4f}, "
                  f"p = {c1_published['p_value']:.4g}")

    # --- per fold detail ---
    print("\n" + "-" * 76)
    print("PER FOLD DETAIL (descriptive only, never tested fold by fold)")
    print("-" * 76)
    f = asm["folds"]
    print(f"  deep SpO2 only AUC  : mean {f['auc_deep_marg'].mean():.4f}, "
          f"SD {f['auc_deep_marg'].std():.4f}, "
          f"range {f['auc_deep_marg'].min():.4f} to "
          f"{f['auc_deep_marg'].max():.4f}")
    print(f"  clinical AUC        : mean {f['auc_clinical'].mean():.4f}, "
          f"SD {f['auc_clinical'].std():.4f}")
    print(f"  stacked AUC        : mean {f['auc_stacked'].mean():.4f}, "
          f"SD {f['auc_stacked'].std():.4f}")
    print(f"  deep coefficient in the stacked model: "
          f"mean {f['deep_coef_in_stack'].mean():+.4f}, "
          f"{int((f['deep_coef_in_stack'] > 0).sum())} of {len(f)} folds "
          f"positive")
    print(f"  spread across the 3 inner models on a test fold: "
          f"mean {f['inner_auc_spread'].mean():.4f}")

    # --- save outputs ---
    out = {
        "arm": args.arm, "repeats": repeats,
        "headline": tdf.to_dict(orient="records"),
        "c2_held_out_test": ht, "c2_per_repeat": per_repeat,
        "c2_delta_auc": boot, "c2_between_repeat_sd": between,
        "c2_se_total": se_total, "c2_ci90": [lo90, hi90],
        "c2_inside_margin": bool(inside),
        "c2_delong": dl, "c1_delong": c1,
        "c1_delong_vs_published_average": c1_published,
        "equivalence_margin": EQUIVALENCE_MARGIN,
    }
    with open(os.path.join(RESULTS,
                           f"stack_eval_{args.arm}{args.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    f.to_csv(os.path.join(RESULTS,
                          f"fold_detail_{args.arm}{args.tag}.csv"), index=False)
    np.savez_compressed(
        os.path.join(RESULTS, f"oof_{args.arm}{args.tag}.npz"),
        y=y, **{k: v for k, v in asm["mats"].items()},
        clinical_logit=asm["clinical_logit"],
        deep_platt_mean=mean["deep_platt"], stacked_mean=mean["stacked"],
        clinical_mean=mean["clinical"])
    print(f"\nwrote {RESULTS}/stack_eval_{args.arm}{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
