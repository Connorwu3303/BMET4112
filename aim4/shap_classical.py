"""
Aim 4: out of fold SHAP for the classical models.

Each of the 15 outer folds is refitted on its training rows and only its test
rows are explained, then the three repeats are averaged. Additivity (base plus
contributions equals the model output) is checked for every participant.

The refits should be the published Aim 2 models. Logistic regression uses the
per fold settings Aim 3 saved and XGBoost's are recovered by re-running the
Aim 2 search, so both match fold for fold. The random forest uses Aim 2's fixed
stage 1 configuration unless --recover-rf has cached its per fold settings, and
its AUC gap is reported (verify_aim4.py tests it). Forest contributions are
probabilities, not log odds, so only rankings compare across all three models.

Outputs (processed/aim4/results/): shap_{model}_{spec}.npz with matching fold,
importance and summary files, combined into shap_importance_all.csv,
shap_folds.csv and shap_summary.json.

Usage:
    python3 shap_classical.py --model logistic_regression
    python3 shap_classical.py --model xgboost        # recovers per fold params
    python3 shap_classical.py --model random_forest
    python3 shap_classical.py --all
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common4 as C                                       # noqa: E402

import ml_common as MC                                    # noqa: E402
from folds import get_verified_folds                      # noqa: E402

from sklearn.metrics import roc_auc_score                 # noqa: E402

FROZEN_LR = os.path.join(C.AIM3_RESULTS, "aim2_frozen_params_by_fold.csv")

# The imbalance arm each model picked in its own Aim 2 stage 1, copied from the
# saved summaries. Re-selecting here would be a second data driven choice and
# wouldn't reproduce the published models.
CHOSEN_ARM = {
    "logistic_regression": {"imbalance": "none", "class_weight": "balanced",
                            "scale": True},
    "random_forest": {"imbalance": "none",
                      "class_weight": "balanced_subsample", "scale": False},
    "xgboost": {"imbalance": "none", "class_weight": None, "scale": False},
}

# Aim 2's fixed forest configuration. Used there for the stage 1 arm comparison,
# and here whenever no recovered per fold settings are passed in.
RF_FIXED = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 5,
            "max_features": "sqrt"}

# published AUCs the refits are checked against
PUBLISHED_AUC = {
    ("logistic_regression", "confounders_only"): 0.7799,
    ("logistic_regression", "obm_only"): 0.6100,
    ("logistic_regression", "obm_plus_confounders"): 0.7792,
    ("random_forest", "confounders_only"): 0.7768,
    ("random_forest", "obm_only"): 0.5953,
    ("random_forest", "obm_plus_confounders"): 0.7680,
    ("xgboost", "confounders_only"): 0.7790,
    ("xgboost", "obm_only"): 0.6016,
    ("xgboost", "obm_plus_confounders"): 0.7745,
}


# --- model construction ---

def frozen_lr_params(spec, repeat, fold):
    """Logistic regression hyperparameters Aim 3 saved for this fold."""
    t = pd.read_csv(FROZEN_LR)
    row = t[(t["specification"] == spec) & (t["repeat"] == repeat)
            & (t["fold"] == fold)]
    if row.empty:
        raise KeyError(f"no frozen parameters for {spec} r{repeat} f{fold}")
    row = row.iloc[0]
    out = {}
    for key in ("C", "penalty", "solver", "l1_ratio", "max_iter"):
        v = row.get(f"model__{key}")
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        out[key] = MC.clean_param_value(key, v)
    return out


def build_model(model, spec, repeat, fold, search_params=None):
    """
    Pipeline to fit on one training fold.

    Returns (pipeline, params_used, units), where units is the scale the SHAP
    values come out in.
    """
    arm = CHOSEN_ARM[model]
    if model == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        params = frozen_lr_params(spec, repeat, fold)
        est = LogisticRegression(class_weight=arm["class_weight"],
                                 random_state=MC.RANDOM_STATE, **params)
        units = "log_odds"
    elif model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        params = dict(search_params) if search_params else dict(RF_FIXED)
        est = RandomForestClassifier(class_weight=arm["class_weight"],
                                     random_state=MC.RANDOM_STATE, n_jobs=-1,
                                     **params)
        # the forest's scikit-learn interface gives probabilities, no margin
        units = "probability"
    elif model == "xgboost":
        from xgboost import XGBClassifier
        from train_xgboost import BASE_XGB_PARAMS
        params = dict(search_params or {})
        est = XGBClassifier(**BASE_XGB_PARAMS)
        if params:
            est.set_params(**params)
        units = "log_odds"
    else:
        raise ValueError(model)
    pipe = MC.build_pipeline(est, imbalance=arm["imbalance"],
                             scale=arm["scale"],
                             random_state=MC.RANDOM_STATE)
    return pipe, params, units


# recovered per fold hyperparameters are cached under these names, in
# processed/aim4/results/
PARAM_CACHE = {"xgboost": "xgb_frozen_params_{spec}.csv",
               "random_forest": "rf_frozen_params_{spec}.csv"}


def from_csv_value(name, value):
    """
    Clean one hyperparameter read back from a CSV.

    The forest's max_features column mixes strings ("sqrt") and fractions, so
    pandas reads it as object dtype and 0.1 comes back as "0.1", which the
    estimator rejects. No hyperparameter here takes a numeric string, so
    anything that parses as a number is converted and the rest is left alone.
    """
    out = MC.clean_param_value(name, value)
    if isinstance(out, str):
        # Booleans too. The forest's bootstrap comes back as a numpy bool when
        # its column is clean, but as the string "True" if a missing value
        # anywhere in the column forces object dtype, and scikit-learn rejects
        # the string.
        if out in ("True", "False"):
            return out == "True"
        try:
            num = float(out)
        except ValueError:
            return out
        return MC.clean_param_value(name, num)
    return out


def cached_params(model, spec):
    """Recovered per fold parameters for this model and spec, or None."""
    name = PARAM_CACHE.get(model)
    if not name:
        return None
    path = os.path.join(C.RESULTS, name.format(spec=spec))
    return pd.read_csv(path) if os.path.exists(path) else None


def recover_params(model, spec, verbose=True):
    """
    Re-run the Aim 2 two stage search to recover a model's per fold choices.

    Calls the Aim 2 code directly, like Aim 3's comparator rebuild did. A
    reimplementation that drifted would give a model that looks like the
    published one but isn't.

    Cached, since for the forest this takes an hour per specification.
    """
    name = PARAM_CACHE[model]
    cache = os.path.join(C.RESULTS, name.format(spec=spec))
    if os.path.exists(cache):
        return pd.read_csv(cache)

    from sklearn.model_selection import StratifiedKFold

    if model == "xgboost":
        import train_xgboost as TM
        arm, weight, n_iter = "none", None, 30
    elif model == "random_forest":
        import train_random_forest as TM
        arm, weight, n_iter = "none", "balanced_subsample", 25
    else:
        raise ValueError(model)

    X, y, _, _ = MC.load_dataset(spec)
    Xv = X.to_numpy(dtype=float)
    space = TM.broad_search_space(quick=False)
    rows = []
    oof = np.full((MC.N_OUTER_REPEATS, len(y)), np.nan)
    for repeat in range(MC.N_OUTER_REPEATS):
        skf = StratifiedKFold(n_splits=MC.N_OUTER_FOLDS, shuffle=True,
                              random_state=MC.RANDOM_STATE + repeat)
        for fold, (tr, te) in enumerate(skf.split(Xv, y)):
            t0 = time.time()
            search = MC.TwoStageSearch(
                TM.make_random_search_factory(arm, weight, space, n_iter),
                TM.focused_grid,
                lambda g: TM.make_grid_factory(arm, weight, g)())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(Xv[tr], y[tr])
            oof[repeat, te] = search.best_estimator_.predict_proba(
                Xv[te])[:, 1]
            keep = {k.replace("model__", ""): MC.clean_param_value(
                        k.replace("model__", ""), v)
                    for k, v in search.best_params_.items()
                    if k.startswith("model__")}
            rows.append({"specification": spec, "repeat": repeat,
                         "fold": fold, "seconds": time.time() - t0,
                         "auc": float(roc_auc_score(y[te], oof[repeat, te])),
                         **keep})
            if verbose:
                print(f"    r{repeat} f{fold}: AUC {rows[-1]['auc']:.4f}  "
                      f"{rows[-1]['seconds']:.0f}s", flush=True)
    t = pd.DataFrame(rows)
    got = float(roc_auc_score(y, np.nanmean(oof, axis=0)))
    os.makedirs(C.RESULTS, exist_ok=True)
    t.to_csv(cache, index=False)
    if verbose:
        print(f"    repeat averaged AUC {got:.4f}, published "
              f"{PUBLISHED_AUC[(model, spec)]:.4f}")
    return t


# --- SHAP ---

def linear_shap(pipe, x_train, x_test):
    """
    Exact SHAP for a linear model on the transformed features, in closed form.

        phi_ij = beta_j * (x_ij - E[x_j]),  base = intercept + beta . E[x]

    The expectation is over the training rows, so one participant's
    explanation doesn't depend on who else landed in their test fold.
    """
    est = pipe.named_steps["model"]
    beta = np.asarray(est.coef_).ravel()
    intercept = float(np.asarray(est.intercept_).ravel()[0])
    mu = x_train.mean(axis=0)
    phi = (x_test - mu) * beta
    base = intercept + float(beta @ mu)
    return phi, base, est.decision_function(x_test)


def tree_shap(pipe, x_test, model):
    """
    Exact TreeSHAP for the two ensembles.

    feature_perturbation stays at the tree path dependent default, the exact
    algorithm using the trees' own cover statistics, which needs no background
    sample. The interventional option needs a background set, and the
    explanation would then depend on which rows were picked for it.
    """
    import shap
    est = pipe.named_steps["model"]
    expl = shap.TreeExplainer(est)
    values = expl.shap_values(x_test, check_additivity=False)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        # scikit-learn classifiers give (rows, features, classes).
        values = values[:, :, 1]
    base = np.asarray(expl.expected_value).ravel()
    base = float(base[1]) if base.size > 1 else float(base[0])
    if model == "xgboost":
        out = est.predict(x_test, output_margin=True)
    else:
        out = est.predict_proba(x_test)[:, 1]
    return values, base, np.asarray(out, dtype=float).ravel()


def run_one(model, spec, data, verbose=True):
    """Fit, explain and check all 15 folds for one model and specification."""
    X, y, features, meta = MC.load_dataset(spec)
    pid = np.asarray(meta["pid"]).astype(int)
    if not np.array_equal(pid, data["pid"].astype(int)):
        raise RuntimeError(
            "the dataset row order does not match the verified fold source; "
            "every fold index would address the wrong participant")
    Xv = X.to_numpy(dtype=float)

    # XGBoost always recovers its per fold choices since its search is cheap.
    # The forest's search costs an hour per specification, so it only uses
    # recovered choices if --recover-rf has already cached them (needed because
    # its fixed configuration was statistically distinguishable from the
    # published tuned model). The route used is saved for each combination.
    table = None
    if model == "xgboost":
        table = recover_params(model, spec)
    elif model == "random_forest":
        table = cached_params(model, spec)

    n_rep = MC.N_OUTER_REPEATS
    shap_sum = np.zeros((len(y), len(features)))
    shap_n = np.zeros(len(y))
    oof = np.full((n_rep, len(y)), np.nan)
    base_values, worst_additivity, rows = [], 0.0, []
    units = None

    for f in data["folds"]:
        repeat, fold = int(f["repeat"]), int(f["fold"])
        tr, te = f["train_idx"], f["test_idx"]
        params = None
        if table is not None:
            r = table[(table["repeat"] == repeat)
                      & (table["fold"] == fold)].iloc[0]
            params = {k: from_csv_value(k, r[k]) for k in r.index
                      if k not in ("specification", "repeat", "fold",
                                   "seconds", "auc", "search_stage",
                                   "inner_broad_score", "inner_focused_score")
                      and pd.notna(r[k])}
        pipe, used, units = build_model(model, spec, repeat, fold, params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xv[tr], y[tr])

        pre = pipe[:-1]
        xt_tr = pre.transform(Xv[tr])
        xt_te = pre.transform(Xv[te])
        oof[repeat, te] = pipe.predict_proba(Xv[te])[:, 1]

        if model == "logistic_regression":
            phi, base, out = linear_shap(pipe, xt_tr, xt_te)
        else:
            phi, base, out = tree_shap(pipe, xt_te, model)

        err = float(np.max(np.abs(base + phi.sum(axis=1) - out)))
        worst_additivity = max(worst_additivity, err)
        base_values.append(base)
        shap_sum[te] += phi
        shap_n[te] += 1
        rows.append({"model": model, "specification": spec, "repeat": repeat,
                     "fold": fold, "base_value": base,
                     "additivity_max_error": err,
                     "auc": float(roc_auc_score(y[te], oof[repeat, te])),
                     "params": json.dumps(used, default=str)})
        if verbose:
            print(f"    r{repeat} f{fold}: AUC {rows[-1]['auc']:.4f}  "
                  f"additivity {err:.2e}", flush=True)

    if not np.all(shap_n == n_rep):
        raise RuntimeError("some participant was not explained once per repeat")
    shap_mean = shap_sum / n_rep
    mean_oof = np.nanmean(oof, axis=0)
    auc = float(roc_auc_score(y, mean_oof))

    raw_values = Xv
    imp = pd.DataFrame({
        "feature": features,
        "mean_abs_shap": np.abs(shap_mean).mean(axis=0),
        "mean_shap": shap_mean.mean(axis=0),
        "sd_shap": shap_mean.std(axis=0),
        # Direction of each feature's effect: rank correlation between a
        # participant's feature value and its contribution. The mean
        # contribution is no use here, since SHAP values are deviations from
        # the background and average to zero. Exactly plus or minus one for a
        # linear model; anything else means the feature pushes different ways
        # for different people.
        "value_shap_spearman": [
            C.spearman_rank_agreement(raw_values[:, j], shap_mean[:, j])["rho"]
            for j in range(len(features))],
    })
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)
    imp = imp.sort_values("mean_abs_shap", ascending=False)

    return {"model": model, "spec": spec, "features": features,
            "recovered": table is not None,
            "shap": shap_mean, "y": y, "pid": pid, "oof": mean_oof,
            "oof_per_repeat": oof, "auc": auc, "units": units,
            "base_values": base_values, "importance": imp,
            "worst_additivity": worst_additivity, "folds": pd.DataFrame(rows)}


HOW_FITTED = {
    "logistic_regression": "per fold, recovered from the Aim 3 comparator "
                           "rebuild",
    "xgboost": "per fold, recovered by re-running the Aim 2 two stage search",
    "random_forest": "per fold, recovered by re-running the Aim 2 two stage "
                     "search",
}


def aggregate(out_dir):
    """
    Rebuild the three combined files from every per combination file on disk.

    Called at the end of every run, so running one model at a time still
    leaves the combined files complete.
    """
    import glob
    imps, folds, rows = [], [], []
    for p in sorted(glob.glob(os.path.join(out_dir,
                                           "shap_summary_*.json"))):
        if os.path.basename(p) == "shap_summary.json":
            continue
        with open(p) as fh:
            rows.append(json.load(fh))
    for model in C.CLASSICAL_MODELS:
        for spec in C.SPECS:
            a = os.path.join(out_dir, f"shap_importance_{model}_{spec}.csv")
            b = os.path.join(out_dir, f"shap_folds_{model}_{spec}.csv")
            if os.path.exists(a):
                imps.append(pd.read_csv(a))
            if os.path.exists(b):
                folds.append(pd.read_csv(b))
    if imps:
        pd.concat(imps, ignore_index=True).to_csv(
            os.path.join(out_dir, "shap_importance_all.csv"), index=False)
    if folds:
        pd.concat(folds, ignore_index=True).to_csv(
            os.path.join(out_dir, "shap_folds.csv"), index=False)
    rows.sort(key=lambda r: (C.CLASSICAL_MODELS.index(r["model"]),
                             C.SPECS.index(r["specification"])))
    with open(os.path.join(out_dir, "shap_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=C.CLASSICAL_MODELS)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="rebuild the aggregate files from what is on disk")
    ap.add_argument("--recover-rf", action="store_true",
                    help="re-run the forest's two stage search to recover its "
                         "per fold hyperparameters, about an hour per "
                         "specification")
    ap.add_argument("--spec", default=None, choices=C.SPECS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    models = C.CLASSICAL_MODELS if (args.all or not args.model) \
        else (args.model,)
    specs = C.SPECS if (args.all or not args.spec) else (args.spec,)

    if args.recover_rf:
        for spec in specs:
            print(f"recovering the forest's per fold hyperparameters for "
                  f"{spec}")
            recover_params("random_forest", spec)

    out_dir = C.ensure_results()
    if args.aggregate_only:
        rows = aggregate(out_dir)
        print(f"rebuilt the aggregates from {len(rows)} saved combinations")
        return 0
    data = get_verified_folds(quiet=True)
    print("=" * 78)
    print("AIM 4 CLASSICAL SHAP")
    print("=" * 78)

    summary, all_folds, all_imp = [], [], []
    for model in models:
        for spec in specs:
            print(f"\n{model} / {spec}")
            t0 = time.time()
            res = run_one(model, spec, data)
            pub = PUBLISHED_AUC[(model, spec)]
            drift = res["auc"] - pub
            print(f"  repeat averaged AUC {res['auc']:.4f}, published "
                  f"{pub:.4f}, difference {drift:+.4f}")
            print(f"  worst additivity violation {res['worst_additivity']:.2e} "
                  f"({res['units']})")
            print(f"  top 8 by mean absolute contribution:")
            for _, r in res["importance"].head(8).iterrows():
                print(f"    {r['feature']:<28s} {r['mean_abs_shap']:.5f}  "
                      f"direction {r['value_shap_spearman']:+.3f}")

            tagname = f"{model}_{spec}"
            res["folds"].to_csv(
                os.path.join(out_dir, f"shap_folds_{tagname}.csv"),
                index=False)
            np.savez_compressed(
                os.path.join(out_dir, f"shap_{tagname}.npz"),
                shap=res["shap"], pid=res["pid"], y=res["y"],
                oof=res["oof"], oof_per_repeat=res["oof_per_repeat"],
                features=np.array(res["features"], dtype=object),
                base_values=np.array(res["base_values"]))
            imp = res["importance"].copy()
            imp.insert(0, "model", model)
            imp.insert(1, "specification", spec)
            imp.insert(2, "units", res["units"])
            imp.to_csv(os.path.join(out_dir, f"shap_importance_{tagname}.csv"),
                       index=False)
            all_imp.append(imp)
            all_folds.append(res["folds"])
            row = {
                "model": model, "specification": spec, "units": res["units"],
                "auc_refit": res["auc"], "auc_published": pub,
                "auc_difference": drift,
                "reproduces_exactly": bool(abs(drift) < 5e-4),
                "worst_additivity_violation": res["worst_additivity"],
                "n_features": len(res["features"]),
                "hyperparameters": HOW_FITTED[model] if res["recovered"]
                    or model == "logistic_regression"
                    else "a declared fixed configuration, Aim 2's own stage 1 "
                         "configuration",
                "seconds": time.time() - t0}
            summary.append(row)
            with open(os.path.join(out_dir, f"shap_summary_{tagname}.json"),
                      "w") as fh:
                json.dump(row, fh, indent=2, default=str)

    # Rebuild the combined files from every per combination file on disk.
    # Running one model at a time is the normal way to use this, and building
    # them from this run alone would quietly drop the other two models.
    summary = aggregate(out_dir)

    print("\n" + "=" * 78)
    print(f"{'model':<22s}{'specification':<24s}{'refit':>8s}{'published':>11s}"
          f"{'diff':>9s}{'additivity':>12s}")
    for s in summary:
        print(f"{s['model']:<22s}{s['specification']:<24s}"
              f"{s['auc_refit']:>8.4f}{s['auc_published']:>11.4f}"
              f"{s['auc_difference']:>+9.4f}"
              f"{s['worst_additivity_violation']:>12.2e}")
    print(f"\nwrote {out_dir}/shap_importance_all.csv")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
