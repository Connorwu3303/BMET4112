"""
Keeps the Aim 2 model as it is and recalibrates it to each horizon:

    logit P(first event by t) = a(t) + b(t) * Aim 2 linear predictor

fit_horizon_models.py refits every coefficient at each horizon, which leaves only
about 67 training events for 18 predictors at 1 year; here only a(t) and b(t) are
learned. They're fitted with censoring weights on training participants only, on
linear predictors from a cross fit inside the training data, so no one's score in
that step comes from a model that saw them. I added this arm after seeing the 1
year result, so it's labelled post hoc wherever it's reported.

On the released label the script stops if the test fold scores don't reproduce the
saved Aim 2 scores. --label corrected retrains the same model on the corrected event
flag with the same folds. --final fits the clinical model once on everyone so it
can be written down, for reporting only.

Outputs: processed/horizon_risk/oof_recalibrated[_corrected].npz (risks plus the
Aim 2 scores) and horizon_risk/results/recalibration_parameters[_corrected].csv,
or with --final horizon_risk/results/recalibration_final[_corrected].json

Usage: /usr/bin/python3 horizon_risk/fit_recalibrated_top_model.py [--label corrected] [--final]
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
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "aim2")]
import horizon_lib as H
from fit_horizon_models import DRIVE_OUT, RESULTS, load_cohort, prepare
from ml_common import N_INNER_FOLDS, N_OUTER_FOLDS, N_OUTER_REPEATS, RANDOM_STATE
from train_logistic_regression import make_search_factory, penalty_grid

R3 = "/Volumes/ThesisHD/Data/processed/aim3/results/"
SPECS = {"clinical": "confounders_only", "oximetry": "obm_only", "combined": "obm_plus_confounders"}


def fit_arm(label):
    cohort = load_cohort(label)
    frame, y, event, time, cause = cohort["frame"], cohort["y"], cohort["event"], cohort["time"], cohort["cause"]
    Z_all = frame[cohort["clinical"]].to_numpy(float)
    n = len(y)
    saved = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)
    # Aim 2 picked class_weight balanced at stage 1, so that's the setting reproduced here
    factory = make_search_factory(penalty_grid(False), "none", "balanced")
    arrays, rows = {}, []
    for spec, aim2_name in SPECS.items():
        start = clock.time()
        X = frame[cohort["specs"][spec]].to_numpy(float)
        risks = {h: np.full((N_OUTER_REPEATS, n), np.nan) for h in H.HORIZONS_YEARS}
        scores = np.full((N_OUTER_REPEATS, n), np.nan)
        for repeat in range(N_OUTER_REPEATS):
            skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)
            # folds always come from the released flag; the model is trained on the chosen event flag
            for fold, (train, test) in enumerate(skf.split(X, y)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search = factory().fit(X[train], event[train])
                    best = search.best_estimator_
                    test_p = best.predict_proba(X[test])[:, 1]
                    scores[repeat, test] = test_p
                    # cross fit inside the training data, at the settings the outer training data picked
                    inner_lp = np.full(len(train), np.nan)
                    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
                    for a, b in inner.split(X[train], event[train]):
                        fit = clone(best).fit(X[train][a], event[train][a])
                        inner_lp[b] = H.logit(fit.predict_proba(X[train][b])[:, 1])
                Zt = prepare(Z_all[train])[0]
                censor = H.CoxCensoring(time[train], cause[train], Zt, max_time=H.horizon_days(10) + 1)
                for years in H.HORIZONS_YEARS:
                    weighted_label, weight = H.ipcw_labels(time[train], cause[train], H.horizon_days(years), censor, Zt)
                    keep = weight > 0
                    a_t, b_t = H.recalibrate(inner_lp[keep], weighted_label[keep], weight[keep])
                    risks[years][repeat, test] = H.expit(a_t + b_t * H.logit(test_p))
                    rows.append(dict(spec=spec, years=years, repeat=repeat, fold=fold, a=a_t, b=b_t,
                                     events_fit=int(weighted_label[keep].sum()), n_fit=int(keep.sum()),
                                     **{k.replace("model__", ""): v for k, v in search.best_params_.items()}))
        auc_here = np.mean([roc_auc_score(event, s) for s in scores])
        if label == "released":
            # reproduction check: these scores have to match what Aim 2 saved
            reference = saved[aim2_name + "__per_repeat"]
            gap = np.abs(scores - reference)
            auc_saved = np.mean([roc_auc_score(y, s) for s in reference])
            print(f"{spec}: largest score gap against saved Aim 2 {gap.max():.2e}, mean gap {gap.mean():.2e}, "
                  f"AUC here {auc_here:.5f}, saved {auc_saved:.5f}, {clock.time() - start:.0f} s", flush=True)
            if abs(auc_here - auc_saved) > 0.002 or gap.max() > 1e-3:
                raise RuntimeError(f"{spec} did not reproduce Aim 2")
        else:
            print(f"{spec}: any event AUC on the {label} label {auc_here:.5f}, {clock.time() - start:.0f} s", flush=True)
        for years, a in risks.items():
            arrays[f"{spec}_recal__{years}y"] = a
        arrays[f"{spec}_aim2score"] = scores
    suffix = "" if label == "released" else f"_{label}"
    np.savez_compressed(os.path.join(DRIVE_OUT, f"oof_recalibrated{suffix}.npz"), **arrays)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"recalibration_parameters{suffix}.csv"), index=False)


def final_recalibrated_equation(label):
    """Aim 2 clinical model fitted on everyone, with a(t) and b(t) from a cross fit, saved as numbers that can be written down."""
    cohort = load_cohort(label)
    frame, event, time, cause = cohort["frame"], cohort["event"], cohort["time"], cohort["cause"]
    names = cohort["clinical"]
    X = frame[names].to_numpy(float)
    Z = prepare(X)[0]
    factory = make_search_factory(penalty_grid(False), "none", "balanced")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = factory().fit(X, event)
        best = search.best_estimator_
        inner_lp = np.full(len(event), np.nan)
        inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for a, b in inner.split(X, event):
            inner_lp[b] = H.logit(clone(best).fit(X[a], event[a]).predict_proba(X[b])[:, 1])
    censor = H.CoxCensoring(time, cause, Z, max_time=H.horizon_days(10) + 1)
    horizons = {}
    for years in H.HORIZONS_YEARS:
        weighted_label, weight = H.ipcw_labels(time, cause, H.horizon_days(years), censor, Z)
        keep = weight > 0
        a_t, b_t = H.recalibrate(inner_lp[keep], weighted_label[keep], weight[keep])
        horizons[years] = dict(a=a_t, b=b_t)
    steps = best.named_steps
    imputer = next(v for k, v in steps.items() if "imput" in k)
    scaler = next(v for k, v in steps.items() if "scal" in k)
    model = steps["model"]
    equation = dict(intercept=float(model.intercept_[0]), coef=dict(zip(names, map(float, model.coef_[0]))),
                    medians=dict(zip(names, map(float, imputer.statistics_))), means=dict(zip(names, map(float, scaler.mean_))),
                    sds=dict(zip(names, map(float, scaler.scale_))), best_params={k: str(v) for k, v in search.best_params_.items()},
                    horizons=horizons)
    suffix = "" if label == "released" else f"_{label}"
    json.dump(equation, open(os.path.join(RESULTS, f"recalibration_final{suffix}.json"), "w"), indent=1)
    print("final equation", {y: (round(v["a"], 3), round(v["b"], 3)) for y, v in horizons.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="released", choices=["released", "corrected"])
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final:
        final_recalibrated_equation(args.label)
    else:
        fit_arm(args.label)


if __name__ == "__main__":
    main()
