"""
Aim 4: head to head comparison of the two pipelines.

In Aim 3 the CNN alone (0.6146) and the 34 hand engineered biomarkers alone
(0.6040) couldn't be separated. This asks whether they carry the same
information or reach a similar AUC by different routes.

  1. Cross model agreement: do the three classical models rank features alike?
  2. Collinearity: SHAP summed within correlated clusters, and what cancels.
  3. Reducibility: out of fold R squared of the deep score on the biomarkers,
     with the clinical and sleep structure sets for comparison.
  4. Concordance: each biomarker's link to the deep score vs its SHAP rank.
  5. Contrast C3: does the deep score add to clinical plus biomarkers?

Outputs (processed/aim4/results/): headtohead.json plus a CSV per analysis,
and attribution_by_spo2.csv once deep_attrib.py has been run.

Run:
    python3 headtohead.py
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common4 as C                                      # noqa: E402

import ml_common as MC                                   # noqa: E402
import data as D                                         # noqa: E402
from folds import get_verified_folds                     # noqa: E402

from scipy import stats                                  # noqa: E402
from sklearn.linear_model import RidgeCV                 # noqa: E402
from sklearn.metrics import roc_auc_score, r2_score      # noqa: E402

# features whose absolute Spearman correlation is above this go in one cluster
# for the grouped contribution summary. Single linkage, so a chain of pairwise
# correlations above the threshold forms one cluster.
#
# Set to 0.8 after seeing two features in the panel correlate at 0.967 with
# contributions that cancel. The clustering only uses feature values and never
# the outcome, so it could have been fixed before any model was fitted.
CLUSTER_THRESHOLD = 0.80

EQUIVALENCE_MARGIN = 0.005      # same margin as pre-registered in Aim 3
N_BOOT = 2000


# --- 1. cross model agreement ---

def cross_model_agreement(out_dir):
    """Rank agreement between the three classical models, within each spec."""
    rows = []
    for spec in C.SPECS:
        imp = {}
        for model in C.CLASSICAL_MODELS:
            p = os.path.join(out_dir, f"shap_importance_{model}_{spec}.csv")
            if os.path.exists(p):
                t = pd.read_csv(p).set_index("feature")
                imp[model] = t["mean_abs_shap"]
        names = sorted(imp)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = imp[a].index.intersection(imp[b].index)
                r = C.spearman_rank_agreement(imp[a][common].to_numpy(),
                                              imp[b][common].to_numpy())
                # top five overlap as well, since a correlation over fifty
                # features can look healthy while the models disagree completely
                # about the most important ones (the part people quote)
                top_a = set(imp[a].sort_values(ascending=False).head(5).index)
                top_b = set(imp[b].sort_values(ascending=False).head(5).index)
                rows.append({"specification": spec, "model_a": a,
                             "model_b": b, "spearman": r["rho"],
                             "p_value": r["p_value"], "n_features": r["n"],
                             "top5_overlap": len(top_a & top_b),
                             "top5_a": "; ".join(sorted(top_a)),
                             "top5_b": "; ".join(sorted(top_b))})
    return pd.DataFrame(rows)


# --- 2. collinearity and grouped contributions ---

def correlation_clusters(X, threshold=CLUSTER_THRESHOLD):
    """
    Single linkage clusters of features by absolute Spearman correlation.

    Written out by hand since single linkage at a threshold is just connected
    components on the thresholded correlation graph, and this keeps the
    definition clear.
    """
    cols = list(X.columns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = X.corr(method="spearman").to_numpy()
    adj = np.abs(np.nan_to_num(rho)) >= threshold
    np.fill_diagonal(adj, False)
    seen, clusters = set(), []
    for i in range(len(cols)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            comp.append(j)
            stack.extend(np.flatnonzero(adj[j]).tolist())
        clusters.append(sorted(comp))
    return [[cols[i] for i in comp] for comp in clusters], rho


def cluster_threshold_sensitivity(spec, thresholds=(0.70, 0.80, 0.90, 0.95)):
    """
    How the clustering changes with the threshold.

    The threshold is the one parameter in this aim picked after seeing the data,
    so its effect is measured here: number of clusters, size of the largest, and
    how many features sit in clusters of more than one.
    """
    X, _, features, _ = MC.load_dataset(spec)
    rows = []
    for th in thresholds:
        clusters, _ = correlation_clusters(X[features], threshold=th)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        rows.append({"specification": spec, "threshold": th,
                     "n_clusters": len(clusters),
                     "largest_cluster": sizes[0] if sizes else 0,
                     "features_in_multi_clusters":
                         int(sum(s for s in sizes if s > 1)),
                     "n_features": len(features)})
    return pd.DataFrame(rows)


def grouped_contributions(out_dir, model, spec):
    """
    Summed SHAP contributions per correlated cluster, and how much they cancel.

    Cancellation ratio = magnitude of the cluster's summed contribution
    divided by the sum of its members' magnitudes. One means the members all
    push the same way; near zero means they cancel, so individual contributions
    inside that cluster shouldn't be quoted.
    """
    path = os.path.join(out_dir, f"shap_{model}_{spec}.npz")
    if not os.path.exists(path):
        return None, None
    z = np.load(path, allow_pickle=True)
    shap = z["shap"]
    features = [str(f) for f in z["features"]]
    X, _, _, _ = MC.load_dataset(spec)
    X = X[features]
    clusters, _ = correlation_clusters(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_all = np.abs(np.nan_to_num(
            X.corr(method="spearman").to_numpy()))

    rows = []
    for ci, members in enumerate(clusters):
        idx = [features.index(m) for m in members]
        summed = shap[:, idx].sum(axis=1)
        separate = np.abs(shap[:, idx]).mean(axis=0).sum()
        # single linkage joins A to C through B, so members of a big cluster
        # don't all have to correlate above the threshold with each other. The
        # weakest and mean pairwise correlation show how much chaining there is.
        sub = rho_all[np.ix_(idx, idx)]
        off = sub[~np.eye(len(idx), dtype=bool)] if len(idx) > 1 \
            else np.array([1.0])
        rows.append({
            "model": model, "specification": spec, "cluster": ci,
            "size": len(members), "members": "; ".join(members),
            "mean_abs_shap_summed": float(np.abs(summed).mean()),
            "sum_of_member_mean_abs": float(separate),
            "cancellation_ratio": float(np.abs(summed).mean() / separate)
            if separate > 0 else np.nan,
            "min_pairwise_abs_spearman": float(off.min()),
            "mean_pairwise_abs_spearman": float(off.mean()),
        })
    t = pd.DataFrame(rows).sort_values("mean_abs_shap_summed",
                                       ascending=False)
    t["rank"] = range(1, len(t) + 1)
    return t, clusters


# --- 3. reducibility of the deep score ---

def cross_fitted_r2(target, predictors, folds, repeat):
    """
    Out of fold R squared of a ridge regression of target on predictors.

    Ridge, with the penalty picked inside each training fold, because the
    biomarker panel is too collinear for a stable unpenalised fit. The mean in
    sample R squared comes back too, and the gap shows how much is overfitting.
    """
    pred = np.full(len(target), np.nan)
    insample = []
    for f in folds:
        if f["repeat"] != repeat:
            continue
        tr, te = f["train_idx"], f["test_idx"]
        ok_tr = np.isfinite(target[tr])
        tf = D.fit_confounder_transform(predictors[tr][ok_tr])
        xt = D.apply_confounder_transform(predictors[tr][ok_tr], tf)
        xe = D.apply_confounder_transform(predictors[te], tf)
        model = RidgeCV(alphas=np.logspace(-3, 4, 25))
        model.fit(xt, target[tr][ok_tr])
        pred[te] = model.predict(xe)
        insample.append(float(r2_score(target[tr][ok_tr],
                                       model.predict(xt))))
    ok = np.isfinite(pred) & np.isfinite(target)
    return {"r2_out_of_fold": float(r2_score(target[ok], pred[ok])),
            "r2_in_sample_mean": float(np.mean(insample)),
            "pearson": float(np.corrcoef(target[ok], pred[ok])[0, 1]),
            "spearman": float(stats.spearmanr(target[ok], pred[ok])
                              .statistic),
            "n": int(ok.sum()), "prediction": pred}


# --- 5. contrast C3 (step 4 is done inline in main) ---

def contrast_c3(data, deep_z, out_dir):
    """
    Does the deep score add anything to the clinical variables plus biomarkers?

    Set up the same way Aim 3 set up C2 so the two are comparable: the Aim 2
    model for obm_plus_confounders with its own frozen per fold hyperparameters,
    fitted with and without the cross fitted deep score as one extra column. The
    likelihood ratio test on that column uses held out predictions only.
    """
    sys.path.insert(0, C.AIM3_DIR)
    import stack_eval as SE

    y, pid = data["y"], data["pid"]
    pid_to_row = {int(p): i for i, p in enumerate(pid)}
    X, y2, features, meta = MC.load_dataset("obm_plus_confounders")
    if not np.array_equal(np.asarray(meta["pid"]).astype(int),
                          pid.astype(int)):
        raise RuntimeError("row order mismatch between the dataset and folds")
    if not np.array_equal(y2, y):
        raise RuntimeError("outcome mismatch between the dataset and folds")
    Xv = X.to_numpy(dtype=float)

    repeat = C.DEEP_REPEAT
    base = np.full(len(y), np.nan)
    with_deep = np.full(len(y), np.nan)
    base_logit = np.full(len(y), np.nan)
    rows = []

    for fold in range(MC.N_OUTER_FOLDS):
        sel = os.path.join(C.AIM3_RESULTS, "selected",
                           f"{C.DEEP_ARM}_r{repeat}_f{fold}.npz")
        if not os.path.exists(sel):
            return None, None
        z = np.load(sel, allow_pickle=True)
        te = np.array([pid_to_row[int(p)] for p in z["test_pid"].astype(int)])
        tr = np.array([pid_to_row[int(p)]
                       for p in z["outer_train_pid"].astype(int)])
        params = SE.frozen_params(repeat, fold, spec="obm_plus_confounders")

        p0, l0, _, _ = SE.fit_frozen_lr(Xv[tr], y[tr], Xv[te], params)
        base[te] = p0
        base_logit[te] = l0

        train_z = z["train_z_inc"][tr]
        if not np.all(np.isfinite(train_z)):
            raise RuntimeError("cross fitted deep scores missing in training "
                               "rows")
        p1, _, lr1, _ = SE.fit_frozen_lr(
            np.column_stack([Xv[tr], train_z]), y[tr],
            np.column_stack([Xv[te], z["test_z_inc"]]), params)
        with_deep[te] = p1
        rows.append({"fold": fold, "n_test": len(te),
                     "auc_base": float(roc_auc_score(y[te], p0)),
                     "auc_with_deep": float(roc_auc_score(y[te], p1)),
                     "deep_coefficient": float(lr1.coef_[0][-1])})

    ok = np.isfinite(base) & np.isfinite(with_deep) & np.isfinite(deep_z)
    test = SE.held_out_coefficient_test(y[ok], base_logit[ok], deep_z[ok])
    # order matters: this returns AUC(second) minus AUC(first), so the baseline
    # goes first and the delta reads as the improvement
    delta = SE.paired_bootstrap_delta_auc(y[ok], base[ok], with_deep[ok])
    out = {
        "contrast": "C3",
        "description": "clinical plus biomarkers, with and without the deep "
                       "score",
        "n": int(ok.sum()), "n_events": int(y[ok].sum()),
        "auc_base": float(roc_auc_score(y[ok], base[ok])),
        "auc_with_deep": float(roc_auc_score(y[ok], with_deep[ok])),
        "coefficient_test": test, "delta_auc": delta,
        "equivalence_margin": EQUIVALENCE_MARGIN,
    }
    out["delta_auc_point"] = out["auc_with_deep"] - out["auc_base"]
    return out, pd.DataFrame(rows)


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", default=None,
                    help="deep attribution table; defaults to the full run")
    args = ap.parse_args()

    out_dir = C.ensure_results()
    data = get_verified_folds(quiet=True)
    y, pid = data["y"], data["pid"]
    report = {}

    print("=" * 78)
    print("AIM 4 HEAD TO HEAD")
    print("=" * 78)

    # 1. cross model agreement
    print("\n1. DO THE CLASSICAL MODELS AGREE ABOUT WHAT MATTERS?")
    agree = cross_model_agreement(out_dir)
    if len(agree):
        agree.to_csv(os.path.join(out_dir, "cross_model_agreement.csv"),
                     index=False)
        for _, r in agree.iterrows():
            print(f"   {r['specification']:<22s} "
                  f"{r['model_a'][:12]:<13s} vs {r['model_b'][:12]:<13s} "
                  f"Spearman {r['spearman']:+.3f}  "
                  f"top five overlap {int(r['top5_overlap'])}/5")
    report["cross_model_agreement"] = agree.to_dict(orient="records")

    # 2. collinearity
    print("\n2. COLLINEARITY: CONTRIBUTIONS SUMMED WITHIN CORRELATED CLUSTERS")
    grouped_all = []
    for model in C.CLASSICAL_MODELS:
        for spec in ("obm_only", "obm_plus_confounders"):
            t, clusters = grouped_contributions(out_dir, model, spec)
            if t is None:
                continue
            grouped_all.append(t)
            if model == "logistic_regression":
                print(f"   {spec}: {len(clusters)} clusters at "
                      f"|Spearman| >= {CLUSTER_THRESHOLD}")
                for _, r in t.head(6).iterrows():
                    print(f"     {r['mean_abs_shap_summed']:.4f}  "
                          f"cancellation {r['cancellation_ratio']:.2f}  "
                          f"size {int(r['size']):2d}  weakest pair "
                          f"{r['min_pairwise_abs_spearman']:.2f}  "
                          f"{r['members'][:50]}")
    sens = pd.concat([cluster_threshold_sensitivity(sp)
                      for sp in ("obm_only", "obm_plus_confounders")],
                     ignore_index=True)
    sens.to_csv(os.path.join(out_dir, "cluster_threshold_sensitivity.csv"),
                index=False)
    print("\n   how the clustering depends on the threshold:")
    print(f"   {'spec':<22s}{'threshold':>10s}{'clusters':>10s}"
          f"{'largest':>9s}{'in clusters > 1':>17s}")
    for _, r in sens.iterrows():
        print(f"   {r['specification']:<22s}{r['threshold']:>10.2f}"
              f"{int(r['n_clusters']):>10d}{int(r['largest_cluster']):>9d}"
              f"{int(r['features_in_multi_clusters']):>17d}")
    report["cluster_threshold_sensitivity"] = sens.to_dict(orient="records")

    if grouped_all:
        g = pd.concat(grouped_all, ignore_index=True)
        g.to_csv(os.path.join(out_dir, "grouped_contributions.csv"),
                 index=False)
        worst = g[g["size"] > 1].nsmallest(5, "cancellation_ratio")
        print("\n   the clusters whose members cancel most, where individual")
        print("   contributions must NOT be quoted:")
        for _, r in worst.iterrows():
            print(f"     {r['model'][:12]:<13s} {r['specification'][:20]:<21s} "
                  f"ratio {r['cancellation_ratio']:.3f}  {r['members'][:48]}")
        report["worst_cancellation"] = worst.to_dict(orient="records")

    # 3. reducibility
    print("\n3. HOW MUCH OF THE DEEP SCORE DO THE BIOMARKERS REPRODUCE?")
    deep = np.load(os.path.join(C.AIM3_RESULTS, "oof_A.npz"),
                   allow_pickle=True)
    deep_marg = deep["deep_marg"][0]
    deep_inc = deep["deep_inc"][0]

    Xo, _, obm_features, _ = MC.load_dataset("obm_only")
    Xc, _, conf_features, _ = MC.load_dataset("confounders_only")
    structure = pd.read_csv(os.path.join(C.AIM3_RESULTS, "structure_set.csv"))
    structure = structure.set_index("pid").loc[[int(p) for p in pid]] \
        .reset_index()
    sets = {
        "34 biomarkers": Xo.to_numpy(dtype=float),
        "18 clinical variables": Xc.to_numpy(dtype=float),
        "9 structure variables":
            structure[list(D.STRUCTURE_COLUMNS)].to_numpy(dtype=float),
        "biomarkers plus structure": np.column_stack([
            Xo.to_numpy(dtype=float),
            structure[list(D.STRUCTURE_COLUMNS)].to_numpy(dtype=float)]),
    }
    red = []
    for name, mat in sets.items():
        r = cross_fitted_r2(deep_marg, mat, data["folds"], C.DEEP_REPEAT)
        red.append({"predictors": name, "n_predictors": mat.shape[1],
                    **{k: v for k, v in r.items() if k != "prediction"}})
        print(f"   {name:<28s} out of fold R2 {r['r2_out_of_fold']:+.4f}  "
              f"in sample {r['r2_in_sample_mean']:+.4f}  "
              f"Spearman {r['spearman']:+.3f}")
    pd.DataFrame(red).to_csv(
        os.path.join(out_dir, "deep_reducibility.csv"), index=False)
    report["reducibility"] = red

    # 4. SHAP rank vs association with the deep score
    print("\n4. DO THE TWO PIPELINES WEIGHT THE SAME BIOMARKERS?")
    imp_path = os.path.join(out_dir,
                            "shap_importance_logistic_regression_obm_only.csv")
    bridge = None
    if os.path.exists(imp_path):
        imp = pd.read_csv(imp_path).set_index("feature")
        rows = []
        for f in obm_features:
            v = Xo[f].to_numpy(dtype=float)
            m = np.isfinite(v) & np.isfinite(deep_marg)
            rho = float(stats.spearmanr(v[m], deep_marg[m]).statistic)
            rows.append({"feature": f, "abs_spearman_with_deep": abs(rho),
                         "spearman_with_deep": rho,
                         "mean_abs_shap": float(imp.loc[f, "mean_abs_shap"]),
                         "shap_rank": int(imp.loc[f, "rank"])})
        bridge = pd.DataFrame(rows)
        bridge["deep_rank"] = bridge["abs_spearman_with_deep"] \
            .rank(ascending=False).astype(int)
        bridge = bridge.sort_values("abs_spearman_with_deep", ascending=False)
        bridge.to_csv(os.path.join(out_dir, "biomarker_bridge.csv"),
                      index=False)
        r = C.spearman_rank_agreement(bridge["mean_abs_shap"],
                                      bridge["abs_spearman_with_deep"])
        report["bridge_rank_agreement"] = r
        print(f"   rank agreement between the SHAP importance of a biomarker")
        print(f"   and its association with the deep score: "
              f"Spearman {r['rho']:+.3f}, p = {r['p_value']:.3g}, "
              f"n = {r['n']}")
        print(f"   the biomarkers most associated with the deep score:")
        for _, rr in bridge.head(8).iterrows():
            print(f"     {rr['feature']:<28s} rho {rr['spearman_with_deep']:+.3f}"
                  f"   SHAP rank {int(rr['shap_rank'])}")

    # deep attribution by SpO2 band, inside scored events and by sleep stage
    # (only if deep_attrib.py has been run)
    att_path = args.attrib or os.path.join(
        out_dir, f"deep_attrib_{C.DEEP_ARM}.csv")
    if os.path.exists(att_path):
        att = pd.read_csv(att_path)
        print(f"\n   deep attribution profile over {len(att)} participants "
              f"({os.path.basename(att_path)})")
        prof = []
        for i, lab in enumerate(C.bin_labels()):
            s = att[f"exact_bin_{i}_sum"].sum()
            a = att[f"exact_bin_{i}_abs"].sum()
            n = att[f"exact_bin_{i}_n"].sum()
            prof.append({"spo2_bin": lab, "signed_total": float(s),
                         "abs_total": float(a), "samples": int(n),
                         "minutes_per_night": float(n / 60.0 / len(att))})
        pdf = pd.DataFrame(prof)
        tot_abs = pdf["abs_total"].sum()
        pdf["share_of_abs"] = pdf["abs_total"] / tot_abs
        pdf["share_of_time"] = pdf["samples"] / pdf["samples"].sum()
        pdf["enrichment"] = pdf["share_of_abs"] / pdf["share_of_time"]
        pdf.to_csv(os.path.join(out_dir, "attribution_by_spo2.csv"),
                   index=False)
        print(f"   {'SpO2 bin':<12s}{'min/night':>10s}{'signed':>12s}"
              f"{'share abs':>11s}{'enrichment':>12s}")
        for _, r in pdf.iterrows():
            print(f"   {r['spo2_bin']:<12s}{r['minutes_per_night']:>10.1f}"
                  f"{r['signed_total']:>+12.2f}{r['share_of_abs']:>11.3f}"
                  f"{r['enrichment']:>12.2f}")
        report["attribution_by_spo2"] = pdf.to_dict(orient="records")
        for key in ("desat", "resp"):
            e = att[f"exact_{key}_enrichment"].replace(
                [np.inf, -np.inf], np.nan).dropna()
            report[f"{key}_enrichment_median"] = float(e.median())
            print(f"   attribution enrichment inside scored "
                  f"{'desaturations' if key == 'desat' else 'respiratory events'}"
                  f": median {e.median():.3f}")
        nrem = att["exact_nrem_abs"].sum()
        rem = att["exact_rem_abs"].sum()
        report["rem_share_of_attribution"] = float(rem / (nrem + rem))
        report["rem_share_of_time"] = float(
            att["exact_rem_n"].sum()
            / (att["exact_nrem_n"].sum() + att["exact_rem_n"].sum()))
        print(f"   REM carries {rem / (nrem + rem):.1%} of absolute "
              f"attribution on {report['rem_share_of_time']:.1%} of analysable "
              f"sleep time")
        print(f"   note: the pooling allocates half the pooled vector to REM by "
              f"design,\n   so 50 percent is the architectural expectation, not "
              f"the time share")
    else:
        print(f"\n   (no full attribution table at {att_path}; run "
              f"deep_attrib.py)")

    # 5. contrast C3
    print("\n5. CONTRAST C3: DOES THE DEEP SCORE ADD TO CLINICAL PLUS "
          "BIOMARKERS?")
    c3, c3folds = contrast_c3(data, deep_inc, out_dir)
    if c3 is None:
        print("   (the selected outputs for the deep arm are missing)")
    else:
        t = c3["coefficient_test"]
        d = c3["delta_auc"]
        print(f"   clinical plus biomarkers          AUC "
              f"{c3['auc_base']:.4f}")
        print(f"   the same plus the deep score      AUC "
              f"{c3['auc_with_deep']:.4f}")
        print(f"   change {c3['delta_auc_point']:+.4f}, "
              f"90 percent interval "
              f"[{d.get('ci90_lo', float('nan')):+.4f}, "
              f"{d.get('ci90_hi', float('nan')):+.4f}], "
              f"pre-registered margin plus or minus {EQUIVALENCE_MARGIN}")
        inside = (abs(d.get("ci90_lo", 1)) < EQUIVALENCE_MARGIN
                  and abs(d.get("ci90_hi", 1)) < EQUIVALENCE_MARGIN)
        print(f"   the interval lies "
              f"{'INSIDE' if inside else 'OUTSIDE'} the margin, so this is "
              f"{'equivalence' if inside else 'a failure to separate'}")
        c3["equivalence_established"] = bool(inside)
        print(f"   coefficient on the deep score {t.get('beta'):+.4f} per "
              f"standard deviation, odds ratio {t.get('or_per_sd'):.4f} "
              f"[{t.get('or_ci_lo'):.4f}, {t.get('or_ci_hi'):.4f}], "
              f"p = {t.get('p_value'):.4g}")
        c3folds.to_csv(os.path.join(out_dir, "contrast_c3_folds.csv"),
                       index=False)
        report["contrast_c3"] = c3

    with open(os.path.join(out_dir, "headtohead.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nwrote {out_dir}/headtohead.json")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
