"""
Bootstraps the whole pipeline to see what the fixed model intervals leave out.

The CIs in the main tables hold the fitted models fixed. This refits on bootstrap samples so the spread also
includes training variance. To keep the run time manageable it uses a reduced grid (L2 only, 5 values of C,
3 inner folds). That pipeline isn't the same as the saved one: it's L2 only, training rows are duplicated, and
inner folds split rows, so one person can end up on both sides of a tuning split. Its mean is pessimistic, most
of all for the 52 feature model (whose saved fits are mostly L1 or elastic net), so only the SD is borrowed.
A rough interval is the saved estimate plus or minus 1.96 of these SDs. That's a sensitivity width, so don't
read it as a confidence interval. Released label only.

Outputs (horizon_risk/results/): whole_pipeline_bootstrap.csv, whole_pipeline_bootstrap_samples.csv

Usage: /usr/bin/python3 horizon_risk/whole_pipeline_bootstrap.py
"""

import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "aim2")]
import horizon_lib as H
from fit_horizon_models import N_JOBS, RESULTS, load_cohort, prepare

warnings.simplefilter("ignore")
N_BOOT = 400
C_GRID = np.logspace(-3, 1, 5)
SPECS = ("clinical", "oximetry", "combined", "age_sex")
CONTRASTS = [("clinical", "oximetry"), ("combined", "clinical"), ("clinical", "age_sex"), ("clinical_cox", "clinical")]


def pick_and_fit(X_train, label, weight, X_test):
    losses = np.zeros(len(C_GRID))
    for a, b in StratifiedKFold(3, shuffle=True, random_state=0).split(X_train, label):
        A, B = prepare(X_train[a], X_train[b])
        for k, C in enumerate(C_GRID):
            model = LogisticRegression(C=C, max_iter=2000, tol=1e-4).fit(A, label[a], sample_weight=weight[a])
            losses[k] += H.weighted_logloss(label[b], model.predict_proba(B)[:, 1], weight[b])
    A, B = prepare(X_train, X_test)
    return LogisticRegression(C=C_GRID[int(np.argmin(losses))], max_iter=2000, tol=1e-4).fit(A, label, sample_weight=weight).predict_proba(B)[:, 1]


def one_sample(cohort, draw, seed):
    """Folds are made on unique people so nobody ends up in both training and test. Returns within fold AUC per model and horizon."""
    time, cause, y, frame = cohort["time"], cohort["cause"], cohort["y"], cohort["frame"]
    Z = frame[cohort["clinical"]].to_numpy(float)
    matrices = {s: frame[cohort["specs"][s]].to_numpy(float) for s in SPECS}
    unique = np.unique(draw)
    cap = H.horizon_days(10) + 1
    days = [H.horizon_days(h) for h in H.HORIZONS_YEARS]
    risks = {(s, h): np.full(len(draw), np.nan) for s in SPECS + ("clinical_cox",) for h in H.HORIZONS_YEARS}
    fold_of = np.zeros(len(draw), int)
    for fold, (_, test_unique) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, y[unique])):
        in_test = np.isin(draw, unique[test_unique])
        train, test = draw[~in_test], draw[in_test]
        fold_of[in_test] = fold
        Zt, Zte = prepare(Z[train], Z[test])
        censor = H.CoxCensoring(time[train], cause[train], Zt, max_time=cap)
        for years, day in zip(H.HORIZONS_YEARS, days):
            label, weight = H.ipcw_labels(time[train], cause[train], day, censor, Zt)
            keep = weight > 0
            w = weight[keep] / weight[keep].mean()
            for s in SPECS:
                risks[(s, years)][in_test] = pick_and_fit(matrices[s][train][keep], label[keep], w, matrices[s][test])
        cox = H.CauseSpecificCox(time[train], cause[train], Zt).risk(Zte, days)
        for years, day in zip(H.HORIZONS_YEARS, days):
            risks[("clinical_cox", years)][in_test] = cox[day]
    Zd = prepare(Z[draw])[0]
    censor = H.CoxCensoring(time[draw], cause[draw], Zd, max_time=cap)
    out = {}
    for years, day in zip(H.HORIZONS_YEARS, days):
        label, weight = H.ipcw_labels(time[draw], cause[draw], day, censor, Zd)
        for s in SPECS + ("clinical_cox",):
            out[f"{s}__{years}y"] = H.within_fold_auc(label, risks[(s, years)], weight, fold_of)
    return out


def safe_sample(cohort, draw, seed):
    # a resample can leave a model that won't fit (e.g. complete separation in the censoring model), so those draws are skipped and counted
    try:
        return one_sample(cohort, draw, seed)
    except (ValueError, np.linalg.LinAlgError):
        return None


def main():
    cohort = load_cohort()
    n = len(cohort["y"])
    rng = np.random.RandomState(2026)
    draws = [(rng.randint(0, n, n), int(rng.randint(1_000_000))) for _ in range(N_BOOT)]
    done = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(safe_sample)(cohort, d, s) for d, s in draws)
    good = [d for d in done if d is not None]
    table = pd.DataFrame(good)
    table.to_csv(os.path.join(RESULTS, "whole_pipeline_bootstrap_samples.csv"), index=False)
    saved = pd.read_csv(os.path.join(RESULTS, "contrasts.csv"))
    rows = []
    for years in H.HORIZONS_YEARS:
        for a, b in CONTRASTS:
            delta = table[f"{a}__{years}y"] - table[f"{b}__{years}y"]
            fixed = saved[(saved.years == years) & (saved.a == a) & (saved.b == b) & (saved.metric == "auc_wf")].iloc[0]
            sd = float(delta.std(ddof=1))
            rows.append(dict(years=years, a=a, b=b, samples=len(table), failed=N_BOOT - len(table), saved_difference=fixed.difference, saved_lo=fixed.lo, saved_hi=fixed.hi,
                             sd_fixed_models=float((fixed.hi - fixed.lo) / 3.92), sd_whole_pipeline=sd, rough_lo=float(fixed.difference - 1.96 * sd), rough_hi=float(fixed.difference + 1.96 * sd),
                             bootstrap_mean=float(delta.mean()), share_above_zero=float((delta > 0).mean())))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "whole_pipeline_bootstrap.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
