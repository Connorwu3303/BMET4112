"""
Checks that each Aim 2 model was fitted correctly and tuned well enough. The
data, cohort, metrics and folds are covered by verify_aim2_results.py instead.

  fast    saved outputs only (seconds): search settings, hyperparameters stuck
          at a search edge, fold stability, selection optimism, calibration
  refit   refits models (minutes): each fold's selected model must reproduce
          its saved test AUC; also train vs test AUC, LR convergence, and each
          model's inner search on one outer fold rerun (no failed candidates,
          same winner), the LR one again with a 30,000 iteration limit
  curves  validation curves past the searched range (tens of minutes) for
          parameters that kept landing on an edge. A range was too narrow only
          if a value outside it beats the selected one by more than
          MATERIAL_AUC_GAIN. Diagnostic only (plain 5-fold CV on the full
          cohort), never reported as model performance.

Output: processed/aim2/results/optimisation_verification_<part>.txt

Usage:
    python3 aim2/verify_optimisation.py --part fast
    python3 aim2/verify_optimisation.py --part refit
    python3 aim2/verify_optimisation.py --part curves
    add --smoke to any part for a quick code-path check (no report written)
"""

import argparse
import inspect
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score

import ml_common as MC
from ml_common import (
    N_INNER_FOLDS, N_OUTER_FOLDS, N_OUTER_REPEATS, RANDOM_STATE, RESULTS_DIR,
    build_pipeline, clean_param_value, clean_params, load_dataset,
)
import train_logistic_regression as LR
import train_random_forest as RF
import train_xgboost as XGB

PRIMARY = "obm_plus_confounders"
MATERIAL_AUC_GAIN = 0.002   # about an eighth of a 95% CI width on this cohort

lines, verdicts = [], []


def emit(text=""):
    print(text, flush=True)
    lines.append(text)


def check(name, ok, detail=""):
    verdicts.append((name, bool(ok)))
    emit(f"  {'PASS' if ok else 'FLAG'}  {name}" + (f"  [{detail}]" if detail else ""))


def section(title):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


def load_results():
    res = {k: json.load(open(os.path.join(RESULTS_DIR, f"{p}_results.json")))
           for k, p in (("lr", "logreg"), ("rf", "rf"), ("xgb", "xgb"))}
    folds = {k: pd.read_csv(os.path.join(RESULTS_DIR, f"{p}_fold_metrics.csv"))
             for k, p in (("lr", "logreg"), ("rf", "rf"), ("xgb", "xgb"))}
    params = {
        "lr": pd.read_csv(os.path.join(RESULTS_DIR, "logreg_selected_params_by_fold.csv")),
        "rf": pd.DataFrame(res["rf"]["stage2"]["selected_params_by_fold"]),
        "xgb": pd.DataFrame(res["xgb"]["stage2"]["selected_params_by_fold"]),
    }
    arms = {
        "lr": next(a for a in LR.IMBALANCE_ARMS if a[0] == res["lr"]["stage1_best_arm"]),
        "rf": next(a for a in RF.IMBALANCE_ARMS if a[0] == res["rf"]["stage1_best_arm"]),
        "xgb": next(a for a in XGB.IMBALANCE_ARMS if a[0] == res["xgb"]["stage1_best_arm"]),
    }
    return res, folds, params, arms


def outer_splits(y):
    """The exact outer folds nested_cv used: seed RANDOM_STATE + repeat."""
    splits = {}
    for rep in range(N_OUTER_REPEATS):
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            splits[(rep, fold)] = (tr, te)
    return splits


def row_params(row):
    """Selected settings for one fold, cleaned, without diagnostic columns."""
    raw = {c: row[c] for c in row.index if str(c).startswith("model__")}
    return clean_params(raw)


# --- part 1: fast checks on the saved outputs ---

def part_fast(smoke=False):
    res, folds, params, arms = load_results()
    X, y, feats, meta = load_dataset(PRIMARY)
    n_features = X.shape[1]

    section("1.1 SEARCH CONFIGURATION (every model must optimise the same thing)")
    lr_search = LR.make_search_factory(LR.penalty_grid(False), arms["lr"][1], arms["lr"][2])()
    rf_broad = RF.make_random_search_factory(arms["rf"][1], arms["rf"][2], RF.broad_search_space(False), 25)()
    rf_grid = RF.make_grid_factory(arms["rf"][1], arms["rf"][2], {"model__n_estimators": [10]})()
    xg_broad = XGB.make_random_search_factory(arms["xgb"][1], arms["xgb"][2], XGB.broad_search_space(False), 30)()
    xg_grid = XGB.make_grid_factory(arms["xgb"][1], arms["xgb"][2], {"model__n_estimators": [10]})()
    for name, s in (("LR grid", lr_search), ("RF broad", rf_broad), ("RF focused", rf_grid),
                    ("XGB broad", xg_broad), ("XGB focused", xg_grid)):
        cv = s.cv
        ok = (s.scoring == "roc_auc" and s.refit is True and isinstance(cv, StratifiedKFold)
              and cv.n_splits == N_INNER_FOLDS and cv.shuffle and cv.random_state == RANDOM_STATE)
        check(f"{name}: selects on AUC, refits the winner, stratified {N_INNER_FOLDS}-fold inner CV", ok)
    src = inspect.getsource(MC.nested_cv)
    check("outer test fold never passed to the search (fit uses train_idx only)",
          "search.fit(X_values[train_idx], y[train_idx])" in src
          and "predict_proba(X_values[test_idx])" in src and "fit(X_values[test_idx]" not in src)
    src2 = inspect.getsource(MC.TwoStageSearch.fit)
    check("two-stage choice uses inner scores only",
          "focused.best_score_ >= broad.best_score_" in src2 and "test" not in src2)

    section("1.2 HYPERPARAMETER BOUNDARY ANALYSIS")
    emit("  A parameter chosen at the edge of what was searched, in many folds, means the")
    emit("  best value may lie outside the range. Flagged here, tested properly in 'curves'.")
    boundary_flags = {}

    # logistic regression
    emit("\n  Logistic regression")
    lp = params["lr"]
    grid = LR.penalty_grid(False)
    by_pen = {g["model__penalty"][0]: g for g in grid}
    rows = []
    for pen, g in by_pen.items():
        sel = lp[lp["model__penalty"] == pen]
        if sel.empty:
            continue
        cvals = np.asarray(g["model__C"], float)
        c = sel["model__C"].astype(float)
        rows.append((pen, "C", f"{cvals.min():.4g} to {cvals.max():.4g} ({len(cvals)} values, x{cvals[1] / cvals[0]:.2f} steps)",
                     len(sel), int(np.isclose(c, cvals.min()).sum()), int(np.isclose(c, cvals.max()).sum())))
        if "model__l1_ratio" in g:
            lv = np.asarray(g["model__l1_ratio"], float); l1 = sel["model__l1_ratio"].astype(float)
            rows.append((pen, "l1_ratio", f"{lv.min()} to {lv.max()} ({len(lv)} values)",
                         len(sel), int(np.isclose(l1, lv.min()).sum()), int(np.isclose(l1, lv.max()).sum())))
    emit(f"    {'penalty':<11s} {'param':<9s} {'searched':<38s} {'folds':>5s} {'at low':>7s} {'at high':>7s}")
    for pen, prm, rng, n, lo, hi in rows:
        emit(f"    {pen:<11s} {prm:<9s} {rng:<38s} {n:>5d} {lo:>7d} {hi:>7d}")
        if max(lo, hi) >= max(3, n // 2):
            boundary_flags.setdefault("lr", []).append(f"{pen} {prm} ({'low' if lo >= hi else 'high'} edge in {max(lo, hi)}/{n})")
    emit("    Note: l1_ratio = 0 (pure L2) is covered separately by the L2 grid.")

    # random forest
    emit("\n  Random forest")
    rp = params["rf"]
    mf_numeric = {"sqrt": int(np.sqrt(n_features)) / n_features, "log2": int(np.log2(n_features)) / n_features}
    def mf_value(v):
        return mf_numeric.get(v, v) if isinstance(v, str) else float(v)
    rf_specs = [
        ("n_estimators", [200, 300, 500, 800], rp["model__n_estimators"].astype(float), "discrete"),
        ("max_depth", [4, 6, 8, 12, 20], rp["model__max_depth"].astype(float), "discrete, plus unlimited"),
        ("min_samples_leaf", [1, 39], rp["model__min_samples_leaf"].astype(float), "1 to 29 broad, +10 in focused grid"),
        ("min_samples_split", [2, 39], rp["model__min_samples_split"].astype(float), "2 to 39"),
        ("max_features (share)", [mf_value("log2"), 0.4], rp["model__max_features"].map(mf_value).astype(float),
         "log2, 0.05, 0.1, sqrt, 0.2, 0.4"),
    ]
    emit(f"    {'param':<22s} {'searched':<36s} {'selected (folds)':<26s} {'at low':>7s} {'at high':>7s}")
    for name, bounds, vals, desc in rf_specs:
        lo_b, hi_b = min(bounds), max(bounds)
        v = vals.dropna()
        lo = int(np.isclose(v, lo_b).sum()); hi = int(np.isclose(v, hi_b).sum())
        shown = ", ".join(f"{k:g} x{c}" for k, c in v.value_counts().head(3).items())
        emit(f"    {name:<22s} {desc:<36s} {shown:<26s} {lo:>7d} {hi:>7d}")
        if max(lo, hi) >= 5:
            boundary_flags.setdefault("rf", []).append(f"{name} ({'low' if lo >= hi else 'high'} edge in {max(lo, hi)}/15)")
    leaf = rp["model__min_samples_leaf"].astype(float)
    emit(f"    Within-fold check for leaf size: the focused grid tops out at the broad winner + 10.")
    emit(f"    The broad candidates are fixed by the seed, so the leaf values the broad stage can")
    emit(f"    return are known; folds whose leaf equals one of those + 10 sat at their grid's top.")
    cands = list(__import__("sklearn.model_selection", fromlist=["ParameterSampler"]).ParameterSampler(
        RF.broad_search_space(False), n_iter=25, random_state=RANDOM_STATE))
    broad_leaves = sorted({int(c["model__min_samples_leaf"]) for c in cands})
    at_top = int(leaf.isin([b + 10 for b in broad_leaves]).sum())
    emit(f"    broad leaf values available: {broad_leaves}; folds at broad value + 10: {at_top}/15")
    if at_top >= 5:
        boundary_flags.setdefault("rf", []).append(f"min_samples_leaf at top of its focused grid in {at_top}/15")

    # xgboost
    emit("\n  XGBoost")
    xp = params["xgb"]
    xgb_specs = [
        ("n_estimators", 50, 1600, False), ("learning_rate", 0.005, 0.5, True), ("max_depth", 1, 7, False),
        ("min_child_weight", 1, 20, False), ("subsample", 0.6, 1.0, False), ("colsample_bytree", 0.4, 1.0, False),
        ("gamma", 0.0, 5.0, False), ("reg_alpha", 1e-3, 10.0, True), ("reg_lambda", 0.1, 20.0, True),
    ]
    emit(f"    {'param':<18s} {'explored':<20s} {'median (min to max)':<30s} {'near low':>8s} {'near high':>9s}")
    for name, lo_b, hi_b, logscale in xgb_specs:
        v = xp[f"model__{name}"].astype(float)
        if logscale:
            t = lambda a: np.log10(a)
            span = t(hi_b) - t(lo_b); lo = int((t(v) - t(lo_b) <= 0.1 * span).sum()); hi = int((t(hi_b) - t(v) <= 0.1 * span).sum())
        else:
            span = hi_b - lo_b; lo = int((v - lo_b <= 0.1 * span + 1e-12).sum()); hi = int((hi_b - v <= 0.1 * span + 1e-12).sum())
        emit(f"    {name:<18s} {f'{lo_b:g} to {hi_b:g}':<20s} {f'{v.median():.4g} ({v.min():.4g} to {v.max():.4g})':<30s} {lo:>8d} {hi:>9d}")
        if max(lo, hi) >= 5:
            boundary_flags.setdefault("xgb", []).append(f"{name} ({'low' if lo >= hi else 'high'} edge in {max(lo, hi)}/15)")

    emit("")
    for model in ("lr", "rf", "xgb"):
        flags = boundary_flags.get(model, [])
        check(f"{model}: no hyperparameter pinned at a search edge in 5+ folds", not flags, "; ".join(flags))

    section("1.3 FOLD STABILITY AND SELECTION OPTIMISM (primary specification)")
    emit(f"    {'model':<6s} {'outer AUC mean':>15s} {'sd':>7s} {'min':>7s} {'max':>7s} {'inner mean':>11s} {'inner-outer':>12s}")
    for model in ("lr", "rf", "xgb"):
        f = folds[model]
        gap = f["inner_best_score"].mean() - f["auc_roc"].mean()
        emit(f"    {model:<6s} {f['auc_roc'].mean():>15.4f} {f['auc_roc'].std():>7.4f} {f['auc_roc'].min():>7.4f} "
             f"{f['auc_roc'].max():>7.4f} {f['inner_best_score'].mean():>11.4f} {gap:>+12.4f}")
        check(f"{model}: fold AUCs stable (sd < 0.03)", f["auc_roc"].std() < 0.03, f"sd {f['auc_roc'].std():.4f}")
        check(f"{model}: no material selection optimism (inner - outer < +{MATERIAL_AUC_GAIN})",
              gap < MATERIAL_AUC_GAIN, f"{gap:+.4f}")

    section("1.4 CALIBRATION OF OUT-OF-FOLD PREDICTIONS (primary specification)")
    import statsmodels.api as sm
    emit("  Calibration slope near 1 = predicted risks are not too extreme or too flat.")
    emit("  Calibration intercept near 0 = average predicted risk matches the event rate.")
    emit(f"    {'model':<6s} {'arm':<32s} {'mean predicted':>15s} {'observed':>9s} {'intercept':>10s} {'slope [95% CI]':>22s}")
    for model, prefix in (("lr", "logreg"), ("rf", "rf"), ("xgb", "xgb")):
        arr = np.load(os.path.join(RESULTS_DIR, f"{prefix}_oof_scores.npz"))[PRIMARY]
        yy, ss = arr[0].astype(int), np.clip(arr[1], 1e-6, 1 - 1e-6)
        lp_ = np.log(ss / (1 - ss))
        slope_fit = sm.Logit(yy, sm.add_constant(lp_)).fit(disp=0)
        slope = slope_fit.params[1]; lo, hi = slope_fit.conf_int()[1]
        icpt = sm.GLM(yy, np.ones_like(lp_), family=sm.families.Binomial(), offset=lp_).fit().params[0]
        emit(f"    {model:<6s} {arms[model][0]:<32s} {ss.mean():>15.4f} {yy.mean():>9.4f} {icpt:>+10.3f} "
             f"{slope:>8.3f} [{lo:.3f}, {hi:.3f}]")
        check(f"{model}: calibrated (|intercept| < 0.2 and slope 0.8 to 1.2)",
              abs(icpt) < 0.2 and 0.8 <= slope <= 1.2, f"intercept {icpt:+.3f}, slope {slope:.3f}")
    emit("  A FLAG here is a finding about probability scale, not a fitting error: class")
    emit("  weighting deliberately shifts predicted risk upwards.")

    section("1.5 IMBALANCE STRATEGY SELECTION")
    for model in ("lr", "rf", "xgb"):
        a = res[model]["stage1_imbalance_arms"]
        aucs = sorted(((m["auc_roc"], k) for k, m in a.items()), reverse=True)
        emit(f"    {model}: chosen {aucs[0][1]} ({aucs[0][0]:.4f}); runner-up {aucs[1][1]} "
             f"({aucs[1][0]:.4f}); margin {aucs[0][0] - aucs[1][0]:.4f}; spread {aucs[0][0] - aucs[-1][0]:.4f}")
    return boundary_flags


# --- part 2: refit fold models ---

def build_fold_model(model, row, arms):
    """Rebuild exactly the pipeline the search refitted for one outer fold."""
    prm = row_params(row)
    if model == "lr":
        est = LogisticRegression(class_weight=arms["lr"][2], max_iter=2000, random_state=RANDOM_STATE)
        est.set_params(**{k: v for k, v in prm.items() if v is not None})
        return build_pipeline(est, imbalance=arms["lr"][1], scale=True, random_state=RANDOM_STATE)
    if model == "rf":
        est = RF.make_estimator(arms["rf"][2]); est.set_params(**prm)
        return build_pipeline(est, imbalance=arms["rf"][1], scale=False, random_state=RANDOM_STATE)
    est = XGB.make_estimator(arms["xgb"][2]); est.set_params(**prm)
    return build_pipeline(est, imbalance=arms["xgb"][1], scale=False, random_state=RANDOM_STATE)


def part_refit(smoke=False):
    res, folds, params, arms = load_results()
    X, y, feats, meta = load_dataset(PRIMARY)
    Xv = X.to_numpy(dtype=float)
    splits = outer_splits(y)

    section("2.1 EVERY OUTER FOLD'S SELECTED MODEL REPRODUCES ITS SAVED TEST AUC")
    emit("  Refit from the saved per-fold settings. Test AUC must match the saved fold")
    emit("  metrics exactly; training AUC is compared with test AUC as an overfitting check.")
    lr_nonconv = []
    for model in ("lr", "xgb", "rf"):
        p = params[model]; fm = folds[model].set_index(["repeat", "fold"])
        todo = p if not smoke else p.head(1)
        if model == "rf" and not smoke:
            todo = p[p["repeat"] == 0]   # 800-tree forests: repeat 0 is enough for this check
        worst, gaps, t0 = 0.0, [], time.time()
        for _, row in todo.iterrows():
            key = (int(row["repeat"]), int(row["fold"]))
            tr, te = splits[key]
            pipe = build_fold_model(model, row, arms)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                pipe.fit(Xv[tr], y[tr])
            # convergence warnings were suppressed in the main run, so check here
            if model == "lr":
                est = pipe.named_steps["model"]
                if int(np.max(est.n_iter_)) >= est.max_iter:
                    lr_nonconv.append((key, row_params(row), int(np.max(est.n_iter_)), est.max_iter))
            test_auc = roc_auc_score(y[te], pipe.predict_proba(Xv[te])[:, 1])
            train_auc = roc_auc_score(y[tr], pipe.predict_proba(Xv[tr])[:, 1])
            worst = max(worst, abs(test_auc - fm.loc[key, "auc_roc"]))
            gaps.append(train_auc - test_auc)
        check(f"{model}: {len(todo)} refitted fold model(s) reproduce saved test AUC",
              worst < 1e-9, f"worst difference {worst:.2e}, {time.time() - t0:.0f}s")
        emit(f"    {model}: training minus test AUC, mean {np.mean(gaps):+.4f} "
             f"(range {np.min(gaps):+.4f} to {np.max(gaps):+.4f})")
    check("logistic regression: selected model converged in every refitted fold",
          not lr_nonconv, "; ".join(f"fold {k}: {n}/{m} iters" for k, _, n, m in lr_nonconv))

    section("2.2 FULL INNER SEARCH RE-RUN ON ONE OUTER FOLD (repeat 0, fold 0)")
    tr, te = splits[(0, 0)]
    Xtr, ytr = Xv[tr], y[tr]

    # searches use error_score = nan, so a failed candidate only shows as a NaN score
    def nan_count(search):
        return int(np.isnan(search.cv_results_["mean_test_score"]).sum()), len(search.cv_results_["mean_test_score"])

    # logistic regression, plus the iteration-limit sensitivity test
    grid = LR.penalty_grid(quick=smoke)
    s_norm = LR.make_search_factory(grid, arms["lr"][1], arms["lr"][2])()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); s_norm.fit(Xtr, ytr)
    n_nan, n_all = nan_count(s_norm)
    check(f"LR: no failed candidates in the inner search", n_nan == 0, f"{n_nan}/{n_all} failed")
    saved = row_params(params["lr"].query("repeat == 0 and fold == 0").iloc[0])
    got = clean_params({k: v for k, v in s_norm.best_params_.items()})
    same = all((str(got.get(k)) == str(v)) or (isinstance(v, float) and np.isclose(float(got.get(k, np.nan)), v))
               for k, v in saved.items())
    check("LR: re-run selects the same settings as saved for this fold", same, f"{got}")

    big = []
    for g in grid:
        g2 = dict(g)
        g2["model__max_iter"] = [30000]
        big.append(g2)
    s_big = LR.make_search_factory(big, arms["lr"][1], arms["lr"][2])()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning); s_big.fit(Xtr, ytr)
    n_warn = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    got_big = clean_params({k: v for k, v in s_big.best_params_.items() if k != "model__max_iter"})
    got_cmp = {k: v for k, v in got.items() if k != "max_iter"}
    same_sel = (str(sorted(got_big.items())) == str(sorted(got_cmp.items())))
    score_diff = float(np.nanmax(np.abs(s_big.cv_results_["mean_test_score"] - s_norm.cv_results_["mean_test_score"])))
    emit(f"    inner scores, study iteration limits vs 30,000 iterations: largest difference {score_diff:.2e}")
    emit(f"    convergence warnings remaining at 30,000 iterations (main process only): {n_warn}")
    check("LR: selection unchanged when every candidate may run 30,000 iterations",
          same_sel and np.isclose(s_big.best_score_, s_norm.best_score_, atol=1e-4),
          f"best inner AUC {s_norm.best_score_:.5f} vs {s_big.best_score_:.5f}")

    for model, mod in (("xgb", XGB), ("rf", RF)):
        n_iter = 6 if smoke else (30 if model == "xgb" else 25)
        broad = mod.make_random_search_factory(arms[model][1], arms[model][2],
                                               mod.broad_search_space(quick=smoke), n_iter)()
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); broad.fit(Xtr, ytr)
        centre = {k: clean_param_value(k.replace("model__", ""), v) for k, v in broad.best_params_.items()}
        focused = mod.make_grid_factory(arms[model][1], arms[model][2], mod.focused_grid(centre))()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); focused.fit(Xtr, ytr)
        b_nan, b_all = nan_count(broad); f_nan, f_all = nan_count(focused)
        check(f"{model}: no failed candidates (broad {b_all}, focused {f_all})",
              b_nan == 0 and f_nan == 0, f"broad {b_nan} failed, focused {f_nan} failed")
        winner = focused if focused.best_score_ >= broad.best_score_ else broad
        got = clean_params(dict(winner.best_params_))
        saved_row = params[model].query("repeat == 0 and fold == 0").iloc[0]
        saved = row_params(saved_row)
        same = (all(str(got.get(k)) == str(v) for k, v in saved.items())
                and np.isclose(saved_row["inner_broad_score"], broad.best_score_)
                and np.isclose(saved_row["inner_focused_score"], focused.best_score_))
        if not smoke:
            check(f"{model}: re-run reproduces the saved selection and both inner scores", same,
                  f"{time.time() - t0:.0f}s")
        # report if the focused grid's winner sat on the edge of its own grid
        fg = mod.focused_grid(centre)
        edges = []
        for k, vals in fg.items():
            if len(vals) < 2:
                continue
            chosen = focused.best_params_[k]
            numeric = [v for v in vals if not isinstance(v, str)]
            if chosen in (min(numeric), max(numeric)) if numeric and not isinstance(chosen, str) else False:
                edges.append(f"{k.replace('model__', '')}={chosen} (grid {min(numeric)} to {max(numeric)})")
        emit(f"    {model}: focused-grid winner at the edge of its own grid for: {', '.join(edges) or 'none'}")


# --- part 3: validation curves past the searched range ---

def cv_auc(pipe, Xv, y, cv):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.mean(cross_val_score(pipe, Xv, y, cv=cv, scoring="roc_auc", n_jobs=5)))


def part_curves(smoke=False):
    res, folds, params, arms = load_results()
    X, y, feats, meta = load_dataset(PRIMARY)
    Xv = X.to_numpy(dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    summary = []

    def curve(model, label, make, values, searched, selected_value):
        t0 = time.time()
        vals = values[:2] if smoke else values
        scores = [(v, cv_auc(make(v), Xv, y, cv)) for v in vals]
        base = next((s for v, s in scores if v == selected_value), None)
        best_v, best_s = max(scores, key=lambda vs: vs[1])
        outside = best_v not in searched
        gain = best_s - base if base is not None else float("nan")
        emit(f"\n  {model} {label}  ({time.time() - t0:.0f}s)")
        for v, s in scores:
            tag = []
            if v == selected_value: tag.append("selected")
            if v not in searched: tag.append("outside searched range")
            if v == best_v: tag.append("BEST")
            emit(f"    {str(v):>10s}  AUC {s:.4f}  {', '.join(tag)}")
        material = outside and gain > MATERIAL_AUC_GAIN
        summary.append((model, label, selected_value, best_v, gain, outside, material))
        return material

    section("3. VALIDATION CURVES (diagnostic only: plain 5-fold CV on the full cohort)")
    emit("  Each curve varies ONE hyperparameter, holding the others at the selected values.")
    emit(f"  A search is judged too narrow only if the best value lies OUTSIDE the searched")
    emit(f"  range AND beats the selected value by more than {MATERIAL_AUC_GAIN} AUC.")

    # logistic regression: elastic net around the selected setting
    sel = LR.most_frequent_joint_config(params["lr"])[0]
    def lr_pipe(C=sel["C"], l1=sel.get("l1_ratio", 0.15)):
        est = LogisticRegression(class_weight=arms["lr"][2], penalty="elasticnet", solver="saga",
                                 C=C, l1_ratio=l1, max_iter=20000, random_state=RANDOM_STATE)
        return build_pipeline(est, imbalance=arms["lr"][1], scale=True, random_state=RANDOM_STATE)
    c_vals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    curve("LR", "C (elastic net, l1_ratio fixed)", lambda v: lr_pipe(C=v), c_vals,
          searched={0.001, 0.01, 0.1, 1.0, 10.0}, selected_value=sel["C"])
    curve("LR", "l1_ratio (C fixed)", lambda v: lr_pipe(l1=v), [0.0, 0.05, 0.1, 0.15, 0.3, 0.5],
          searched={0.15, 0.5, 0.85, 0.0}, selected_value=sel.get("l1_ratio", 0.15))

    # random forest around the most common selected config
    rp = params["rf"]
    modal = clean_params({c: rp[c].mode().iloc[0] for c in rp.columns if c.startswith("model__")})
    trees = 100 if smoke else 500
    def rf_pipe(**over):
        cfg = dict(modal); cfg["n_estimators"] = trees; cfg.update(over)
        est = RF.make_estimator(arms["rf"][2]); est.set_params(**cfg)
        return build_pipeline(est, imbalance=arms["rf"][1], scale=False, random_state=RANDOM_STATE)
    curve("RF", "max_features", lambda v: rf_pipe(max_features=v), [0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
          searched={0.05, 0.1, 0.2, 0.4}, selected_value=modal["max_features"])
    curve("RF", "min_samples_leaf", lambda v: rf_pipe(min_samples_leaf=v), [10, 20, 27, 35, 50, 75, 100],
          searched=set(range(1, 40)), selected_value=modal["min_samples_leaf"])
    curve("RF", "max_depth", lambda v: rf_pipe(max_depth=v), [3, 4, 6, 8, 12],
          searched={4, 6, 8, 12, 20}, selected_value=modal["max_depth"])
    curve("RF", "min_samples_split", lambda v: rf_pipe(min_samples_split=v), [10, 37, 60, 100, 150],
          searched=set(range(2, 40)), selected_value=modal["min_samples_split"])

    # xgboost around the representative config
    rep = clean_params(dict(res["xgb"]["stage2"]["representative_params"]))
    def xg_pipe(**over):
        cfg = dict(rep); cfg.update(over)
        est = XGB.make_estimator(arms["xgb"][2]); est.set_params(**cfg)
        return build_pipeline(est, imbalance=arms["xgb"][1], scale=False, random_state=RANDOM_STATE)
    curve("XGB", "min_child_weight", lambda v: xg_pipe(min_child_weight=v), [5, 10, 20, 40, 80],
          searched={1, 3, 5, 10, 20}, selected_value=rep["min_child_weight"])
    curve("XGB", "gamma", lambda v: xg_pipe(gamma=v), [0.0, 0.5, 1.0, 5.0, 10.0, 20.0],
          searched={0.0, 0.1, 0.5, 1.0, 2.0, 5.0}, selected_value=rep["gamma"])
    curve("XGB", "reg_lambda", lambda v: xg_pipe(reg_lambda=v), [1.0, 8.071418522169695, 20.0, 50.0, 100.0],
          searched={1.0, 8.071418522169695, 20.0}, selected_value=rep["reg_lambda"])
    curve("XGB", "max_depth", lambda v: xg_pipe(max_depth=v), [1, 2, 3, 4, 5],
          searched={1, 2, 3, 4, 5, 6, 7}, selected_value=rep["max_depth"])
    curve("XGB", "n_estimators (learning rate fixed)", lambda v: xg_pipe(n_estimators=v), [200, 400, 800, 1600, 3200],
          searched={50, 100, 200, 300, 400, 500, 800, 1000, 1600}, selected_value=rep["n_estimators"])

    section("3. SUMMARY")
    emit(f"    {'model':<5s} {'hyperparameter':<36s} {'selected':>10s} {'best':>10s} {'gain':>8s} {'best outside?':>14s}")
    for model, label, sv, bv, gain, outside, material in summary:
        emit(f"    {model:<5s} {label:<36s} {str(sv):>10s} {str(bv):>10s} {gain:>+8.4f} {str(outside):>14s}")
    for model in ("LR", "RF", "XGB"):
        mats = [f"{label} (best {bv}, +{gain:.4f})" for m, label, sv, bv, gain, outside, material in summary
                if m == model and material]
        check(f"{model}: searched ranges were wide enough (no material gain outside them)", not mats, "; ".join(mats))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("fast", "refit", "curves"), required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    emit("=" * 78)
    emit(f"MODEL CORRECTNESS AND OPTIMISATION VERIFICATION - part {args.part}"
         + (" (SMOKE)" if args.smoke else ""))
    emit("=" * 78)
    emit(f"run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    {"fast": part_fast, "refit": part_refit, "curves": part_curves}[args.part](smoke=args.smoke)
    section("VERDICT")
    n_flag = sum(1 for _, ok in verdicts if not ok)
    emit(f"  {len(verdicts) - n_flag} passed, {n_flag} flagged, {(time.time() - t0) / 60:.1f} min")
    for name, ok in verdicts:
        if not ok:
            emit(f"  FLAGGED: {name}")
    if not args.smoke:
        path = os.path.join(RESULTS_DIR, f"optimisation_verification_{args.part}.txt")
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"\nReport written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
