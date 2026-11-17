"""Checks how sensitive the logistic regression is to two feature-level choices. Added while writing the report.

S1: no 600 s constant value rule. Only the participants the rule affected are recomputed, from the raw EDF.
S2: desaturation thresholds lowered by 0.1 so a four point fall on the device's integer scale always registers.
Nothing on the drive is written. Results go to thesis_report/derived/ (s1/s2 feature csvs, sensitivity_results.json).

Usage:
    python3 thesis_report/scripts/sensitivity_features.py
"""
import json, os, sys, warnings, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "aim1"), os.path.join(ROOT, "aim2"), os.path.join(ROOT, "aim3")]
OUT = os.path.join(ROOT, "thesis_report", "derived")
P = "/Volumes/ThesisHD/Data/processed/"
WINDOWS = ("sleep", "nrem", "rem")

def _panel():
    return json.load(open(P + "aim2/feature_manifest.json"))["core_obm_panel"]

def one_s1(pid):
    import pyedflib
    import reclean_with_validity_mask as RC
    import obm_library as L
    # switch off the constant value (stuck run) rule, then reclean from the raw EDF
    RC.find_stuck_runs = lambda s, fs, **k: np.zeros(s.size, dtype=bool)
    ev = pd.read_csv(P + f"events/shhs1-{pid}_events.csv")
    f = pyedflib.EdfReader(f"/Volumes/ThesisHD/Data/shhs1-{pid}.edf")
    try:
        i = [k for k, l in enumerate(f.getSignalLabels()) if "sao2" in l.lower() or "spo2" in l.lower()][0]
        fs = f.getSampleFrequency(i); raw = f.readSignal(i)
    finally:
        f.close()
    sig, mask, acc = RC.clean_with_mask(raw, fs, ev)
    st = L.build_stage_array(ev, sig.size, fs)
    feats = L.compute_all_obm(sig, mask, fs, st, windows=WINDOWS)
    row = {k: feats.get(k, np.nan) for k in _panel()}; row["pid"] = int(pid); row["n_valid"] = int(mask.sum())
    return row

def one_s2(pid):
    import obm_library as L
    # patch the detector once per process so every threshold is 0.1 lower
    if not getattr(L, "_patched", False):
        orig = L.detect_desaturations
        L.detect_desaturations = lambda seg, fs, thr: orig(seg, fs, thr - 0.1)
        L._patched = True
    sig = np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy"); mask = np.load(P + f"validity_mask/shhs1-{pid}.npy")
    ev = pd.read_csv(P + f"events/shhs1-{pid}_events.csv")
    st = L.build_stage_array(ev, sig.size, 1.0)
    feats = L.compute_all_obm(sig, mask, 1.0, st, windows=WINDOWS)
    row = {k: feats.get(k, np.nan) for k in _panel()}; row["pid"] = int(pid)
    return row

if __name__ == "__main__":
    from joblib import Parallel, delayed
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from folds import get_verified_folds
    from ml_common import delong_test
    data = get_verified_folds(quiet=True)
    y, pid, folds = np.asarray(data["y"], float), [int(p) for p in data["pid"]], data["folds"]
    man = json.load(open(P + "aim2/feature_manifest.json")); panel = man["core_obm_panel"]; conf = man["confounders"]
    ds = pd.read_csv(P + "aim2/modelling_dataset.csv"); key = "pid" if "pid" in ds.columns else "nsrrid"; ds = ds.set_index(key).loc[pid]
    log = pd.read_csv(P + "reclean_log.csv").set_index("pid")
    affected = [p for p in pid if log.loc[p, "n_stuck_only"] > 0]
    print("cohort", len(pid), "affected by the rule", len(affected), flush=True)
    t0 = time.time()
    s1path, s2path = os.path.join(OUT, "s1_no_constant_rule.csv"), os.path.join(OUT, "s2_thresholds_minus_0p1.csv")
    # features are cached, so a rerun goes straight to the models
    if not os.path.exists(s1path):
        rows = Parallel(n_jobs=5, verbose=1)(delayed(one_s1)(p) for p in affected); pd.DataFrame(rows).to_csv(s1path, index=False)
    print("S1 features done", round(time.time() - t0), "s", flush=True); t0 = time.time()
    if not os.path.exists(s2path):
        rows = Parallel(n_jobs=5, verbose=1)(delayed(one_s2)(p) for p in pid); pd.DataFrame(rows).to_csv(s2path, index=False)
    print("S2 features done", round(time.time() - t0), "s", flush=True)
    s1 = pd.read_csv(s1path).set_index("pid"); s2 = pd.read_csv(s2path).set_index("pid")
    fro = pd.read_csv(P + "aim3/results/aim2_frozen_params_by_fold.csv")
    def fit_spec(X, spec):
        # refit the LR in each fold with the frozen Aim 2 hyperparameters (median impute and scale on the training fold)
        oof = np.zeros((3, len(y)))
        for f in folds:
            r = fro[(fro.specification == spec) & (fro.repeat == f["repeat"]) & (fro.fold == f["fold"])].iloc[0]
            kw = dict(C=float(r.model__C), penalty=r.model__penalty, solver=r.model__solver, class_weight="balanced", random_state=42, max_iter=int(r.model__max_iter) if pd.notna(r.model__max_iter) else 2000)
            if r.model__penalty == "elasticnet": kw["l1_ratio"] = float(r.model__l1_ratio)
            tr, te = f["train_idx"], f["test_idx"]
            med = np.nanmedian(X[tr], axis=0); Xf = np.where(np.isnan(X), med, X); mu, sd = Xf[tr].mean(0), Xf[tr].std(0); sd = np.where(sd < 1e-12, 1, sd); Z = (Xf - mu) / sd
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oof[f["repeat"], te] = LogisticRegression(**kw).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        return oof.mean(0), oof
    base_panel = ds[panel].copy()
    res = {"n_affected_in_cohort": len(affected)}
    variants = {"baseline": base_panel}
    v1 = base_panel.copy(); v1.loc[s1.index, panel] = s1[panel].values; variants["S1_no_constant_rule"] = v1
    v2 = s2.loc[pid, panel].copy(); variants["S2_thresholds_minus_0p1"] = v2
    # S1 only touches the affected rows, the rest keep their baseline values
    Xc = ds[conf].to_numpy(float)
    base_oof = {}
    for name, v in variants.items():
        res[name] = {}
        for spec in ("obm_only", "obm_plus_confounders"):
            cols = man["model_specifications"][spec]
            frame = pd.concat([v[panel], ds[conf]], axis=1)[cols]
            m, per = fit_spec(frame.to_numpy(float), spec)
            res[name][spec] = dict(auc=float(roc_auc_score(y, m)), auc_per_repeat=[float(roc_auc_score(y, per[i])) for i in range(3)])
            if name == "baseline": base_oof[spec] = m
            else:
                d = delong_test(y, base_oof[spec], m); res[name][spec]["delong_vs_baseline"] = {k: float(val) for k, val in d.items()}
        res[name]["panel_shift"] = {c: float(np.nanmedian(np.abs(v[c].to_numpy(float) - base_panel[c].to_numpy(float)))) for c in ["sleep_odi3", "sleep_odi4", "sleep_mean", "sleep_ct90_pct", "sleep_sampen"]}
        res[name]["means"] = {c: float(np.nanmean(v[c].to_numpy(float))) for c in ["sleep_odi3", "sleep_odi4", "sleep_mean", "sleep_ct90_pct"]}
    x = pd.read_csv("/Volumes/ThesisHD/Data/documentation/shhs1-dataset-0.15.0.csv", usecols=["nsrrid", "rdi4p", "rdi3p"]).set_index("nsrrid").loc[pid]
    from scipy import stats
    res["S2_vs_rdi"] = dict(base_odi4_mean=float(np.nanmean(base_panel.sleep_odi4)), s2_odi4_mean=float(np.nanmean(v2.sleep_odi4)), rdi4p_mean=float(np.nanmean(x.rdi4p)),
        base_pearson=float(pd.concat([base_panel.sleep_odi4, x.rdi4p], axis=1).corr().iloc[0, 1]), s2_pearson=float(pd.concat([v2.sleep_odi4, x.rdi4p], axis=1).corr().iloc[0, 1]))
    json.dump(res, open(os.path.join(OUT, "sensitivity_results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
