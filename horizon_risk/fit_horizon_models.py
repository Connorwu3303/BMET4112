"""
Fits the 1, 5 and 10 year risk models on the Aim 2 folds.

Each spec is refitted at each horizon with the label "first event by t". Everything
else is as in Aim 2: same participants, same outer folds, same penalty grid, median
imputation and scaling fitted inside the training data. Three things change because
the output is now read as a probability as well as a ranking:

    no class weighting    class weights push every predicted risk up
    log loss for tuning   inner selection uses weighted log loss (Aim 2 used AUC),
                          which rewards correct risks
    no intercept penalty  liblinear penalises the intercept, which shifts risks, so
                          lbfgs and saga are used instead

Out of fold risks are per participant, so they go on the drive
(processed/horizon_risk/oof_<arm>.npz). Fold level summaries go to
horizon_risk/results/fold_summary_<arm>.csv.

Usage:
    /usr/bin/python3 horizon_risk/fit_horizon_models.py
    /usr/bin/python3 horizon_risk/fit_horizon_models.py --arms corrected s_tiers   # named arms
    /usr/bin/python3 horizon_risk/fit_horizon_models.py --quick   # one repeat, coarse grid
"""

import argparse
import json
import os
import sys
import time as clock
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "aim2")]
import horizon_lib as H
from ml_common import (MANIFEST_PATH, N_INNER_FOLDS, N_OUTER_FOLDS, N_OUTER_REPEATS,
                       RANDOM_STATE, load_dataset)

DOC = "/Volumes/ThesisHD/Data/documentation/"
DRIVE_OUT = "/Volumes/ThesisHD/Data/processed/horizon_risk"
RESULTS = os.path.join(HERE, "results")
OBM_FEATURES = "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv"
EVENT_DATE_COLUMNS = ["mi_date", "mip_date", "chd_dthdt", "cvd_dthdt", "revpro_date",
                      "ptca_date", "cabg_date", "chf_date", "stk_date"]
N_JOBS = int(os.environ.get("HORIZON_JOBS", 4))
PRIMARY_SPECS = ["age_only", "age_sex", "clinical_no_age", "clinical", "oximetry", "combined"]


def inferred_group(cv):
    """Which outcome columns are filled in. Stands in for parent cohort, which the files don't record."""
    return np.select([cv.mi_death.notna() & cv.prev_ang.notna() & cv.prev_revpro.notna(),
                      cv.prev_revpro.notna() & cv.prev_ang.isna() & cv.mi_death.isna(),
                      cv.prev_ang.notna() & cv.prev_revpro.isna() & cv.mi_death.isna()], ["G4", "G2", "G3"], "G1")


def counts_as_event_under_corrected_rule(cv):
    """
    Event test behind the count-consistent label in the thesis (load_cohort only applies it in G2).

    The SHHS documentation logic for ARIC at the April 2006 cut tests whether a component equals 1, but the
    components are counts of records. That test reproduces the released any_cvd in every G2 row, while counts of 2
    or more are always flagged in the other groups. So 50 analysed people in G2, where every recorded component type
    has two or more dated records and none has exactly one, end up as non-events. It looks like a coding anomaly but
    it's inferred: the listing is from an earlier run than the released file and no custodian has confirmed it.
    This is the same rule with >= 1. Only the count test changes, so G2 still doesn't count stroke alone the way the
    other groups do. Heart failure is the only basis for 32 of the 50.
    """
    counted = [cv[c].to_numpy(float) >= 1 for c in ("mi", "ptca", "cabg", "revasc_proc", "chf")]
    counted += [cv[c].to_numpy(float) == 1 for c in ("chd_death", "cvd_death")]
    return np.any(counted, axis=0)


def load_cohort(label="released"):
    """
    Features, event flag, time, cause and audit flags for the 4,352 analysed participants.

    label picks the event flag: "released" is any_cvd as used in Aims 1 to 4, "corrected" applies the
    count-consistent rule in G2, and "all_discordant" also counts everyone with any_cvd = 0 who has a CVD date.
    y stays as the released flag whatever the label, since it defines the Aim 2 folds.
    """
    X, y, clinical, meta = load_dataset("confounders_only")
    obm, _, obm_names, _ = load_dataset("obm_only")
    pid = meta["pid"].astype(int)
    y = np.asarray(y)
    cv = pd.read_csv(DOC + "shhs-cvd-summary-dataset-0.15.0.csv").set_index("nsrrid").loc[pid]
    first = cv[EVENT_DATE_COLUMNS].min(axis=1).to_numpy(float)
    censor = cv["censdate"].to_numpy(float)
    group = inferred_group(cv)
    # any_cvd is 0 for these people even though they have a CVD date
    inconsistent = (y == 0) & np.isfinite(first)
    repaired = (y == 0) & (group == "G2") & counts_as_event_under_corrected_rule(cv)
    event = {"released": y == 1, "corrected": (y == 1) | repaired, "all_discordant": (y == 1) | inconsistent}[label].astype(int)
    time, cause = H.build_time_and_cause(event, first, censor, cv["vital"].to_numpy())
    baseline = pd.read_csv(DOC + "shhs1-dataset-0.15.0.csv", low_memory=False, encoding="latin-1").set_index("nsrrid").loc[pid]
    # self reported disease at the baseline interview, code 1 only (angina isn't part of any_cvd)
    told = ["ANGINA15", "MI15", "STROKE15", "HF15", "CABG15", "CA15"]
    prior_any = (baseline[told] == 1).any(axis=1).to_numpy()
    prior_composite = (baseline[told[1:]] == 1).any(axis=1).to_numpy()
    frame = pd.concat([X, obm], axis=1)
    # whole recording versions of the 26 sleep features (no sleep staging needed); the 8 REM and NREM features have none
    recording = [c.replace("sleep_", "recording_") for c in obm_names if c.startswith("sleep_")]
    whole_night = pd.read_csv(OBM_FEATURES, usecols=["pid"] + recording).set_index("pid").loc[pid, recording]
    frame[recording] = whole_night.to_numpy(float)
    frame["diabetes"] = baseline["ParRptDiab"].fillna(0).to_numpy(float)
    frame["diabetes_missing"] = baseline["ParRptDiab"].isna().to_numpy(float)
    no_bloods = [c for c in clinical if c not in ("Chol", "HDL", "Trig")]
    specs = {
        "age_only": ["age_s1"],
        "age_sex": ["age_s1", "is_male"],
        "clinical_no_age": [c for c in clinical if c != "age_s1"],
        "clinical": list(clinical),
        "oximetry": list(obm_names),
        "combined": list(clinical) + list(obm_names),
        # what's left without a blood sample, then also without a blood pressure cuff
        "no_bloods": no_bloods,
        "no_tests": [c for c in no_bloods if c not in ("SystBP", "DiasBP", "HTNDerv_s1")],
        "clinical_diabetes": list(clinical) + ["diabetes", "diabetes_missing"],
        # no tests plus the 34 features (these still use technician scored sleep staging)
        "no_tests_oximetry": [c for c in no_bloods if c not in ("SystBP", "DiasBP", "HTNDerv_s1")] + list(obm_names),
        # whole recording features need no staging, so this is as close as the data get to what a wearable could
        # compute. Still from a research oximeter though, so an actual wearable wasn't tested.
        "recording_oximetry": recording,
        "no_tests_recording": [c for c in no_bloods if c not in ("SystBP", "DiasBP", "HTNDerv_s1")] + recording,
        "clinical_recording": list(clinical) + recording,
    }
    late_event = (event == 1) & (first > censor)
    return dict(frame=frame, y=y, event=event, time=time, cause=cause, pid=pid, specs=specs, clinical=list(clinical),
                inconsistent=inconsistent, repaired=repaired, late_event=late_event, group=group, label=label,
                prior_any=prior_any, prior_composite=prior_composite, baseline=baseline, cv=cv, first=first)


def penalty_grid(quick=False):
    """Aim 2 grid, but with lbfgs and saga in place of liblinear so the intercept isn't penalised."""
    common = dict(max_iter=5000, tol=1e-5, random_state=RANDOM_STATE)
    if quick:
        return [dict(penalty="l2", solver="lbfgs", C=float(c), **common) for c in np.logspace(-3, 1, 5)]
    grid = [dict(penalty="l2", solver="lbfgs", C=float(c), **common) for c in np.logspace(-4, 1.5, 12)]
    grid += [dict(penalty="l1", solver="saga", C=float(c), **common) for c in np.logspace(-4, 1.5, 12)]
    grid += [dict(penalty="elasticnet", solver="saga", C=float(c), l1_ratio=r, **common)
             for c in np.logspace(-3, 1, 5) for r in (0.15, 0.5, 0.85)]
    return grid


def prepare(fit_rows, *others):
    imputer = SimpleImputer(strategy="median").fit(fit_rows)
    scaler = StandardScaler().fit(imputer.transform(fit_rows))
    return [scaler.transform(imputer.transform(a)) for a in (fit_rows,) + others]


def censoring_model(kind, time, cause, Z, max_time, strata=None):
    if kind == "cox":
        return H.CoxCensoring(time, cause, Z, max_time=max_time), Z
    if kind == "km":
        return H.KMCensoring(time, cause), None
    if kind == "km_group":
        return H.KMCensoring(time, cause, strata), strata
    raise ValueError(kind)


def tune_and_fit(X, label, weight, grid):
    """Inner CV on weighted log loss. Returns the best config and its mean inner loss (fit_fold does the refit on all the training data)."""
    skf = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    losses = np.zeros(len(grid))
    for a, b in skf.split(X, label):
        Xa, Xb = prepare(X[a], X[b])
        for k, config in enumerate(grid):
            model = LogisticRegression(**config).fit(Xa, label[a], sample_weight=weight[a])
            losses[k] += H.weighted_logloss(label[b], model.predict_proba(Xb)[:, 1], weight[b])
    best = int(np.argmin(losses))
    return grid[best], float(losses[best] / N_INNER_FOLDS)


def fit_fold(spec, X, Z_all, time, cause, years, repeat, fold, train, test, grid, censor_kind, use_weights, group):
    horizon = H.horizon_days(years)
    Zt, Zte = prepare(Z_all[train], Z_all[test])
    cmodel, cx = censoring_model(censor_kind, time[train], cause[train], Zt, H.horizon_days(10) + 1, group[train])
    label, weight = H.ipcw_labels(time[train], cause[train], horizon, cmodel, cx)
    keep = weight > 0
    if not use_weights:
        weight = keep.astype(float)
    # rescale to mean weight 1 so the penalty strength stays comparable with an unweighted fit
    weight = weight / weight[keep].mean()
    Xk, yk, wk = X[train][keep], label[keep], weight[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config, inner = tune_and_fit(Xk, yk, wk, grid)
        Xtr, Xte = prepare(Xk, X[test])
        model = LogisticRegression(**config).fit(Xtr, yk, sample_weight=wk)
    return dict(spec=spec, years=years, repeat=repeat, fold=fold, test=test,
                risk=model.predict_proba(Xte)[:, 1], null=H.weighted_incidence(yk, wk),
                penalty=config["penalty"], C=config["C"], l1_ratio=config.get("l1_ratio"),
                inner_logloss=inner, n_fit=int(keep.sum()), events_fit=int(yk.sum()),
                n_nonzero=int((model.coef_ != 0).sum()))


def fit_cox_fold(Z_all, time, cause, repeat, fold, train, test):
    Zt, Zte = prepare(Z_all[train], Z_all[test])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = H.CauseSpecificCox(time[train], cause[train], Zt)
    days = [H.horizon_days(y) for y in H.HORIZONS_YEARS]
    risks = model.risk(Zte, days)
    return dict(repeat=repeat, fold=fold, test=test, risks={y: risks[d] for y, d in zip(H.HORIZONS_YEARS, days)})


def run_arm(tag, cohort, specs, grid, repeats, censor_kind="cox", use_weights=True, cause=None, mask=None, cox=False, cox_spec="clinical"):
    """One full set of out of fold risks. Sensitivity arms change the cause, the weighting or who's included."""
    frame, time, y = cohort["frame"], cohort["time"], cohort["y"]
    cause = cohort["cause"] if cause is None else cause
    Z_all = frame[cohort["clinical"]].to_numpy(float)
    Z_cox = frame[cohort["specs"][cox_spec]].to_numpy(float)
    n = len(y)
    tasks, cox_tasks = [], []
    for repeat in range(repeats):
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)
        # folds come from the released flag on the whole cohort so they match Aim 2 whatever the arm changes
        for fold, (train, test) in enumerate(skf.split(np.zeros((n, 1)), y)):
            if mask is not None:
                train, test = train[mask[train]], test[mask[test]]
            for spec in specs:
                X = frame[cohort["specs"][spec]].to_numpy(float)
                for years in H.HORIZONS_YEARS:
                    tasks.append((spec, X, Z_all, time, cause, years, repeat, fold, train, test, grid, censor_kind, use_weights, cohort["group"]))
            if cox:
                cox_tasks.append((Z_cox, time, cause, repeat, fold, train, test))
    start = clock.time()
    print(f"{tag}: {len(tasks)} outer fits over {N_JOBS} workers", flush=True)
    done = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(fit_fold)(*t) for t in tasks)
    risks = {(s, h): np.full((repeats, n), np.nan) for s in specs for h in H.HORIZONS_YEARS}
    null = {h: np.full((repeats, n), np.nan) for h in H.HORIZONS_YEARS}
    rows = []
    for d in done:
        risks[(d["spec"], d["years"])][d["repeat"], d["test"]] = d["risk"]
        null[d["years"]][d["repeat"], d["test"]] = d["null"]
        rows.append({k: v for k, v in d.items() if k not in ("test", "risk")})
    arrays = {f"{s}__{h}y": a for (s, h), a in risks.items()}
    arrays.update({f"null__{h}y": a for h, a in null.items()})
    if cox:
        done_cox = Parallel(n_jobs=N_JOBS)(delayed(fit_cox_fold)(*t) for t in cox_tasks)
        for h in H.HORIZONS_YEARS:
            a = np.full((repeats, n), np.nan)
            for d in done_cox:
                a[d["repeat"], d["test"]] = d["risks"][h]
            arrays[f"{cox_spec}_cox__{h}y"] = a
    os.makedirs(DRIVE_OUT, exist_ok=True)
    np.savez_compressed(os.path.join(DRIVE_OUT, f"oof_{tag}.npz"), time=time, cause=cause, y=y, **arrays)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"fold_summary_{tag}.csv"), index=False)
    print(f"{tag}: finished in {clock.time() - start:.0f} s", flush=True)


def run_named_arm(name, grid, repeats):
    """
    Runs one arm by name so it can be rerun on its own. The released label arms s_death_as_censoring,
    s_marginal_km and s_no_weights are unchanged from the first run. There used to be an arm that dropped the 84
    discordant participants, but it dropped 50 real events along with correct non-events, so its higher AUC was
    falsely reassuring and it was taken out.
    """
    released = load_cohort()
    if name == "primary":
        run_arm("primary", released, PRIMARY_SPECS, grid, repeats, cox=True)
    elif name == "s_death_as_censoring":
        death_censored = np.where(released["cause"] == H.OTHER_DEATH, H.CENSORED, released["cause"])
        run_arm(name, released, ["clinical"], grid, repeats, cause=death_censored)
    elif name == "s_marginal_km":
        run_arm(name, released, ["clinical"], grid, repeats, censor_kind="km")
    elif name == "s_no_weights":
        run_arm(name, released, ["clinical"], grid, repeats, use_weights=False)
    elif name == "corrected":
        run_arm(name, load_cohort("corrected"), PRIMARY_SPECS, grid, repeats, cox=True)
    elif name == "s_all_discordant":
        run_arm(name, load_cohort("all_discordant"), ["clinical"], grid, repeats, cox=True)
    elif name == "s_no_prior_any":
        run_arm(name, released, ["clinical"], grid, repeats, mask=~released["prior_any"], cox=True)
    elif name == "s_no_prior_composite":
        run_arm(name, released, ["clinical"], grid, repeats, mask=~released["prior_composite"], cox=True)
    elif name == "s_km_by_group":
        run_arm(name, released, ["clinical"], grid, repeats, censor_kind="km_group")
    elif name == "s_diabetes":
        run_arm(name, released, ["clinical_diabetes"], grid, repeats, cox=True, cox_spec="clinical_diabetes")
    elif name == "s_tiers":
        run_arm(name, released, ["no_bloods", "no_tests"], grid, repeats)
    elif name == "s_no_tests_oximetry":
        run_arm(name, released, ["no_tests_oximetry"], grid, repeats)
    elif name == "s_recording":
        run_arm(name, released, ["recording_oximetry", "no_tests_recording", "clinical_recording"], grid, repeats)
    elif name == "corrected_diabetes":
        run_arm(name, load_cohort("corrected"), ["clinical_diabetes"], grid, repeats, cox=True, cox_spec="clinical_diabetes")
    elif name == "corrected_tiers":
        run_arm(name, load_cohort("corrected"), ["no_bloods", "no_tests"], grid, repeats)
    elif name == "corrected_no_tests_oximetry":
        run_arm(name, load_cohort("corrected"), ["no_tests_oximetry"], grid, repeats)
    elif name == "corrected_recording":
        run_arm(name, load_cohort("corrected"), ["recording_oximetry", "no_tests_recording", "clinical_recording"], grid, repeats)
    else:
        raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--arms", nargs="+", default=["primary"])
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.quick:
        run_arm("primary_quick", load_cohort(), PRIMARY_SPECS, penalty_grid(True), 1, cox=True)
        return
    for name in args.arms:
        run_named_arm(name, penalty_grid(False), N_OUTER_REPEATS)


if __name__ == "__main__":
    main()
