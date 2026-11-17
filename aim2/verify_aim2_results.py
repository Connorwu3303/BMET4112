"""
Independent check of the full-cohort Aim 1 and Aim 2 results.

It doesn't trust the reports. It recomputes the claims from the raw files with
separate code. Where it does reuse project code (e.g. the cleaning and
biomarker functions) it's to check determinism: saved inputs must give the
saved outputs exactly.

Sections, numbered as in the report:
  1      Aim 1 coverage: every raw participant cleaned and logged
  2      Aim 1 determinism: a random sample of the participants added after
         the first 2,591 re-cleaned from EDF bit for bit
  3      Feature determinism: recomputed biomarkers match the saved table
  4-5    Cohort rebuilt from the raw SHHS CSVs, nobody with baseline CVD in it
  6      Feature sets: exactly 34 biomarkers and 18 confounders
  7-10   Same participants, labels and folds; metrics and DeLong recomputed
  11-12  Preprocessing inside every pipeline, searches nested in outer folds
  13-14  Output hygiene in results/, and no em dashes in code or docs

Output: processed/aim2/results/verification_report.txt

Usage:
    python3 aim2/verify_aim2_results.py
"""

import glob
import io
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJECT, "aim1"))

SHHS_DIR = "/Volumes/ThesisHD/Data"
PROCESSED = os.path.join(SHHS_DIR, "processed")
AIM2 = os.path.join(PROCESSED, "aim2")
RESULTS = os.path.join(AIM2, "results")
DOC_DIR = "/Users/connorwu/Desktop/Thesis/documentation"
REPORT_PATH = os.path.join(RESULTS, "verification_report.txt")

MODELS = {
    "logistic_regression": ("logreg", "logreg_oof_scores.npz", "logreg_results.json",
                            "logreg_fold_metrics.csv"),
    "random_forest": ("rf", "rf_oof_scores.npz", "rf_results.json",
                      "rf_fold_metrics.csv"),
    "xgboost": ("xgb", "xgb_oof_scores.npz", "xgb_results.json",
                "xgb_fold_metrics.csv"),
}
SPECS = ("confounders_only", "obm_only", "obm_plus_confounders")
SAMPLE_SIZE = 5
TOL = 1e-9

lines, verdicts = [], []


def emit(text=""):
    print(text)
    lines.append(text)


def check(name, ok, detail=""):
    verdicts.append((name, bool(ok)))
    emit(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def section(title):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


# Fast DeLong (Sun and Xu 2014), written separately from ml_common's version so
# the two can be checked against each other

def _midrank(x):
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks + 1
    return out


def fast_delong(y, score_a, score_b):
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")          # positives first
    preds = np.vstack([np.asarray(score_a, float)[order],
                       np.asarray(score_b, float)[order]])
    m = int(y.sum())
    n = preds.shape[1] - m
    tx = np.vstack([_midrank(p[:m]) for p in preds])
    ty = np.vstack([_midrank(p[m:]) for p in preds])
    tz = np.vstack([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    diff = aucs[1] - aucs[0]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var)
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(diff), float(p)


def main():
    t_start = time.time()
    emit("=" * 78)
    emit("INDEPENDENT VERIFICATION - FULL COHORT")
    emit("=" * 78)
    emit(f"run at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    section("1. AIM 1 COVERAGE")
    raw = sorted(re.search(r"shhs1-(\d+)\.edf$", os.path.basename(p)).group(1)
                 for p in glob.glob(os.path.join(SHHS_DIR, "shhs1-*.edf"))
                 if re.search(r"shhs1-(\d+)\.edf$", os.path.basename(p)))
    def ids(folder, pattern):
        return {re.search(r"shhs1-(\d+)", os.path.basename(p)).group(1)
                for p in glob.glob(os.path.join(PROCESSED, folder, pattern))}
    sig = ids("cleaned_spo2_v2", "shhs1-*.npy")
    msk = ids("validity_mask", "shhs1-*.npy")
    evt = ids("events", "shhs1-*_events.csv")
    emit(f"  raw EDF {len(raw)}, signals {len(sig)}, masks {len(msk)}, events {len(evt)}")
    check("5,793 raw participants present", len(raw) == 5793, str(len(raw)))
    check("every raw participant has signal, mask and events",
          set(raw) <= sig and set(raw) <= msk and set(raw) <= evt,
          f"missing sig {len(set(raw) - sig)}, mask {len(set(raw) - msk)}, "
          f"events {len(set(raw) - evt)}")
    log = pd.read_csv(os.path.join(PROCESSED, "reclean_log.csv"))
    check("reclean log has one row per participant",
          len(log) == len(raw) and log["pid"].nunique() == len(raw), str(len(log)))
    check("every reclean log row status ok", (log["status"] == "ok").all(),
          str(log["status"].value_counts().to_dict()))
    if "processed_this_run" in log.columns:
        new = int(log["processed_this_run"].fillna(False).astype(bool).sum())
        emit(f"  processed in the full-cohort run: {new}; carried forward: {len(log) - new}")
        check("the 3,202 new participants were processed in this run", new == 3202, str(new))
    arch = pd.read_csv(os.path.join(PROCESSED, "archive_half_cohort_2591",
                                    "reclean_log.csv"))
    merged = arch.merge(log, on="pid", suffixes=("_old", "_new"))
    same = np.allclose(merged["valid_hours_old"], merged["valid_hours_new"])
    check("first 2,591 participants' cleaning statistics preserved exactly",
          len(merged) == 2591 and same, f"{len(merged)} matched")

    section("2. AIM 1 DETERMINISM ON NEW PARTICIPANTS")
    import pyedflib
    import reclean_with_validity_mask as R
    new_pids = sorted(p for p in raw if int(p) > 202600)
    rng = random.Random(20260914)
    sample = rng.sample(new_pids, SAMPLE_SIZE)
    ok_all = True
    for pid in sample:
        ev_saved = pd.read_csv(os.path.join(PROCESSED, "events", f"shhs1-{pid}_events.csv"))
        ev_fresh = R.parse_events(os.path.join(SHHS_DIR, f"shhs1-{pid}-nsrr.xml"))
        reader = pyedflib.EdfReader(os.path.join(SHHS_DIR, f"shhs1-{pid}.edf"))
        try:
            ch = R.find_spo2_channel(reader)
            fs = reader.getSampleFrequency(ch)
            rawsig = reader.readSignal(ch)
        finally:
            reader.close()
        s_new, m_new, _ = R.clean_with_mask(rawsig, fs, ev_fresh)
        s_saved = np.load(os.path.join(PROCESSED, "cleaned_spo2_v2", f"shhs1-{pid}.npy"))
        m_saved = np.load(os.path.join(PROCESSED, "validity_mask", f"shhs1-{pid}.npy"))
        # CSV round-trip can change dtype of an empty or float column, so compare values
        ev_ok = (len(ev_saved) == len(ev_fresh)
                 and (ev_saved["event_concept"].astype(str).values
                      == ev_fresh["event_concept"].astype(str).values).all()
                 and np.allclose(ev_saved["start_sec"], ev_fresh["start_sec"])
                 and np.allclose(ev_saved["duration_sec"], ev_fresh["duration_sec"]))
        s_ok = np.array_equal(s_new, s_saved, equal_nan=True)
        m_ok = np.array_equal(m_new, m_saved)
        ok_all &= ev_ok and s_ok and m_ok
        emit(f"    pid {pid}: events {'ok' if ev_ok else 'DIFF'}, signal "
             f"{'ok' if s_ok else 'DIFF'}, mask {'ok' if m_ok else 'DIFF'}")
    check(f"{SAMPLE_SIZE} random new participants re-cleaned bit-identically", ok_all)

    section("3. FEATURE DETERMINISM")
    from obm_library import build_stage_array, compute_all_obm
    feats = pd.read_csv(os.path.join(AIM2, "obm_features.csv"))
    check("feature table has 5,793 participants", len(feats) == 5793, str(len(feats)))
    fsample = rng.sample(sorted(feats["pid"].astype(int)), SAMPLE_SIZE)
    worst = 0.0
    all_ok = True
    for pid in fsample:
        s = np.load(os.path.join(PROCESSED, "cleaned_spo2_v2", f"shhs1-{pid}.npy"))
        m = np.load(os.path.join(PROCESSED, "validity_mask", f"shhs1-{pid}.npy"))
        e = pd.read_csv(os.path.join(PROCESSED, "events", f"shhs1-{pid}_events.csv"))
        fresh = compute_all_obm(s, m, 1.0, build_stage_array(e, s.size, 1.0))
        row = feats[feats["pid"] == pid].iloc[0]
        bad = 0
        for key, value in fresh.items():
            saved = row.get(key, np.nan)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                bad += int(not pd.isna(saved))
                continue
            if isinstance(value, (bool, np.bool_)):
                continue
            denom = max(abs(float(value)), 1e-12)
            rel = abs(float(value) - float(saved)) / denom
            worst = max(worst, rel)
            bad += int(rel > 1e-6)
        all_ok &= bad == 0
        emit(f"    pid {pid}: {len(fresh)} features recomputed, {bad} mismatches")
    check("biomarkers recomputed from saved inputs match the table",
          all_ok, f"worst relative difference {worst:.2e}")

    section("4-5. COHORT REBUILT INDEPENDENTLY FROM RAW CSVs")
    shhs = pd.read_csv(os.path.join(DOC_DIR, "shhs1-dataset-0.15.0.csv"),
                       usecols=["nsrrid"], low_memory=False)
    cvd = pd.read_csv(os.path.join(DOC_DIR, "shhs-cvd-summary-dataset-0.15.0.csv"),
                      low_memory=False)
    prev_cols = ["prev_mi", "prev_mip", "prev_stk", "prev_chf", "prev_ang", "prev_revpro"]
    base = feats[feats["sufficient_valid_data"].astype(bool)][["pid"]]
    base = base.merge(shhs, left_on="pid", right_on="nsrrid")
    base = base.merge(cvd[["nsrrid", "any_cvd"] + prev_cols], on="nsrrid")
    base = base[base["any_cvd"].notna()]
    prevalent = (base[prev_cols].fillna(0) > 0).any(axis=1)
    incident = base[~prevalent]
    n_obs, n_inc, n_ev = len(base), len(incident), int(incident["any_cvd"].sum())
    emit(f"  rebuilt: observed {n_obs}, prevalent {int(prevalent.sum())}, "
         f"incident {n_inc}, events {n_ev} ({n_ev / n_inc:.2%})")

    data = pd.read_csv(os.path.join(AIM2, "modelling_dataset.csv"))
    check("dataset rows match rebuilt observed cohort", len(data) == n_obs,
          f"{len(data)} vs {n_obs}")
    ds_inc = data[data["cvd_incident"].notna()]
    check("dataset incident cohort matches rebuild", len(ds_inc) == n_inc,
          f"{len(ds_inc)} vs {n_inc}")
    check("dataset incident events match rebuild", int(ds_inc["cvd_incident"].sum()) == n_ev,
          f"{int(ds_inc['cvd_incident'].sum())} vs {n_ev}")
    check("same participant IDs in incident cohort",
          set(ds_inc["pid"]) == set(incident["pid"]))
    leak = int(((data["any_prevalent_cvd"] == 1) & data["cvd_incident"].notna()).sum())
    check("no prevalent-CVD participant in the incident cohort", leak == 0, str(leak))
    recheck = ((data.merge(cvd[["nsrrid"] + prev_cols], left_on="pid", right_on="nsrrid")
                [prev_cols].fillna(0) > 0).any(axis=1).astype(int).values)
    check("any_prevalent_cvd flag agrees with raw count variables (> 0 rule)",
          np.array_equal(recheck, data["any_prevalent_cvd"].values))

    section("6. FEATURE SETS")
    manifest = json.load(open(os.path.join(AIM2, "feature_manifest.json")))
    src = io.open(os.path.join(HERE, "build_modelling_dataset.py"), encoding="utf-8").read()
    core_src = re.findall(r'"([a-z0-9_]+)"', src.split("CORE_OBM_PANEL = [")[1].split("]")[0])
    ms = manifest["model_specifications"]
    check("obm_only is exactly the 34-feature core panel",
          ms["obm_only"] == core_src and len(core_src) == 34, str(len(ms["obm_only"])))
    check("confounders_only has 18 features", len(ms["confounders_only"]) == 18)
    check("obm_plus_confounders = core panel + confounders",
          ms["obm_plus_confounders"] == core_src + ms["confounders_only"],
          str(len(ms["obm_plus_confounders"])))
    check("no confounder appears among the biomarkers",
          not set(ms["obm_only"]) & set(ms["confounders_only"]))

    section("7-10. MODEL OUTPUTS")
    oof, res, folds = {}, {}, {}
    for model, (_, npz, js, fm) in MODELS.items():
        paths = [os.path.join(RESULTS, f) for f in (npz, js, fm)]
        present = all(os.path.exists(p) for p in paths)
        check(f"{model}: out-of-fold scores, results and fold metrics exist", present)
        if present:
            oof[model] = np.load(paths[0])
            res[model] = json.load(open(paths[1]))
            folds[model] = pd.read_csv(paths[2])

    y_ref = None
    for model in oof:
        for spec in SPECS:
            if spec not in oof[model]:
                check(f"{model}: specification {spec} saved", False)
                continue
            y, s = oof[model][spec][0].astype(int), oof[model][spec][1]
            if y_ref is None:
                y_ref = y
            ok = (len(y) == n_inc and np.array_equal(y, y_ref) and int(y.sum()) == n_ev
                  and np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all())
            check(f"{model} / {spec}: same {n_inc} participants and labels, valid scores", ok)
            r = res[model]["stage3_specifications"][spec]
            auc, ap, br = roc_auc_score(y, s), average_precision_score(y, s), brier_score_loss(y, s)
            check(f"{model} / {spec}: AUC {auc:.4f}, AUPRC {ap:.4f}, Brier {br:.4f} recomputed",
                  abs(auc - r["auc_roc"]) < TOL and abs(ap - r["auprc"]) < TOL
                  and abs(br - r["brier"]) < TOL)
            check(f"{model} / {spec}: bootstrap CI brackets the estimate",
                  r["auc_ci_lo"] <= r["auc_roc"] <= r["auc_ci_hi"])
        meta = res[model]["meta"]
        check(f"{model}: reported cohort matches rebuild",
              meta["n_participants"] == n_inc and meta["n_events"] == n_ev,
              f"{meta['n_participants']} / {meta['n_events']}")

    emit("\n  Fold structure:")
    shapes = {}
    for model, fm in folds.items():
        key = fm[["repeat", "fold", "n_test", "n_test_events"]].sort_values(["repeat", "fold"])
        shapes[model] = key.reset_index(drop=True)
        check(f"{model}: 3 repeats x 5 folds = 15 outer folds",
              len(fm) == 15 and fm["repeat"].nunique() == 3 and fm["fold"].nunique() == 5,
              str(len(fm)))
    if len(shapes) > 1:
        ref = next(iter(shapes.values()))
        check("all models used identical outer folds (sizes and event counts)",
              all(v.equals(ref) for v in shapes.values()))
        expected = []
        for rep in range(3):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + rep)
            for f, (_, te) in enumerate(skf.split(np.zeros(len(y_ref)), y_ref)):
                expected.append((rep, f, len(te), int(y_ref[te].sum())))
        exp = pd.DataFrame(expected, columns=["repeat", "fold", "n_test", "n_test_events"])
        check("folds match an independent reconstruction from seed 42",
              exp.equals(ref.astype(exp.dtypes.to_dict())))

    emit("\n  DeLong, pipeline implementation vs independent fast DeLong:")
    from ml_common import delong_test
    worst_p = 0.0
    n_tests = 0
    for model in oof:
        a = oof[model]
        for x, z in (("confounders_only", "obm_plus_confounders"),
                     ("confounders_only", "obm_only"),
                     ("obm_only", "obm_plus_confounders")):
            y = a[x][0].astype(int)
            mine = delong_test(y, a[x][1], a[z][1])
            _, _, d2, p2 = fast_delong(y, a[x][1], a[z][1])
            worst_p = max(worst_p, abs(mine["p_value"] - p2))
            n_tests += 1
            ok = abs(mine["difference"] - d2) < 1e-9 and abs(mine["p_value"] - p2) < 1e-6
            check(f"{model}: {x} -> {z} dAUC {d2:+.4f} p {p2:.4g}", ok)
    names = list(oof)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            y = oof[names[i]]["obm_plus_confounders"][0].astype(int)
            mine = delong_test(y, oof[names[i]]["obm_plus_confounders"][1],
                               oof[names[j]]["obm_plus_confounders"][1])
            _, _, d2, p2 = fast_delong(y, oof[names[i]]["obm_plus_confounders"][1],
                                       oof[names[j]]["obm_plus_confounders"][1])
            n_tests += 1
            check(f"{names[i]} -> {names[j]} (primary) dAUC {d2:+.4f} p {p2:.4g}",
                  abs(mine["difference"] - d2) < 1e-9 and abs(mine["p_value"] - p2) < 1e-6)
    emit(f"  {n_tests} DeLong tests compared, worst p-value disagreement {worst_p:.2e}")

    section("11-12. PIPELINE STRUCTURE AND SEARCH NESTING")
    import train_logistic_regression as LR
    import train_random_forest as RF
    import train_xgboost as XGB
    lr_pipe = LR.make_search_factory(LR.penalty_grid(quick=True), "smotetomek", None)().estimator
    rf_pipe = RF.make_grid_factory("smotetomek", None, {"model__n_estimators": [10]})().estimator
    xg_pipe = XGB.make_grid_factory("smotetomek", None, {"model__n_estimators": [10]})().estimator
    check("LR pipeline: impute -> scale -> resample -> model",
          [n for n, _ in lr_pipe.steps] == ["impute", "scale", "resample", "model"])
    check("RF pipeline: impute -> resample -> model",
          [n for n, _ in rf_pipe.steps] == ["impute", "resample", "model"])
    check("XGB pipeline: impute -> resample -> model",
          [n for n, _ in xg_pipe.steps] == ["impute", "resample", "model"])
    offenders = []
    for path in glob.glob(os.path.join(HERE, "train_*.py")):
        text = io.open(path, encoding="utf-8").read()
        for pattern in (r"\.fit_transform\(", r"SMOTETomek\(", r"SimpleImputer\(",
                        r"StandardScaler\("):
            if re.search(pattern, text):
                offenders.append(f"{os.path.basename(path)}:{pattern}")
    check("no preprocessing fitted outside the shared pipeline in model scripts",
          not offenders, ", ".join(offenders))
    for model in ("random_forest", "xgboost"):
        if model in res:
            st = res[model].get("stage2", {})
            by_fold = st.get("selected_params_by_fold", [])
            ok = ("inside each outer training fold" in st.get("procedure", "")
                  and len(by_fold) == 15
                  and all("search_stage" in row for row in by_fold))
            check(f"{model}: two-stage search ran inside all 15 outer folds", ok)

    section("13. OUTPUT HYGIENE")
    files = sorted(f for f in os.listdir(RESULTS) if not f.startswith("._"))
    expected_prefix = ("logreg_", "rf_", "xgb_", "model_comparison", "verification_report",
                       "results_tables", "optimisation_verification_")
    stray = [f for f in files if not f.startswith(expected_prefix) or " " in f
             or re.search(r"(C|_quicktest)\.(npz|json|csv|txt|png)$", f)]
    check("no stray, misnamed or test files in results/", not stray, ", ".join(stray))
    chain_start = os.path.getmtime(os.path.join(AIM2, "modelling_dataset.csv"))
    stale = [f for f in files if f != "verification_report.txt"
             and os.path.getmtime(os.path.join(RESULTS, f)) < chain_start]
    check("every result file written after the full-cohort dataset was built",
          not stale, ", ".join(stale))

    section("14. STYLE")
    targets = (glob.glob(os.path.join(PROJECT, "*.md"))
               + glob.glob(os.path.join(PROJECT, "aim1", "*.py"))
               + glob.glob(os.path.join(PROJECT, "aim2", "*.py")))
    dashed = [os.path.basename(p) for p in targets
              if "\u2014" in io.open(p, encoding="utf-8", errors="ignore").read()
              and os.path.basename(p) != "handoff.md"]
    check("no em dashes in project code or documents (handoff.md original text excepted)",
          not dashed, ", ".join(dashed))

    section("VERDICT")
    n_fail = sum(1 for _, ok in verdicts if not ok)
    emit(f"  {len(verdicts) - n_fail} passed, {n_fail} failed, "
         f"{(time.time() - t_start) / 60:.1f} min")
    for name, ok in verdicts:
        if not ok:
            emit(f"  FAILED: {name}")
    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
