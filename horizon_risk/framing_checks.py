"""
A few extra checks the thesis quotes, added after the main evaluation. All write to horizon_risk/results/.

reconciliation.csv (plus reconciliation_by_horizon.csv and reconciliation_groups.csv)
    How the earlier fixed 10 year figure of 0.8031 relates to the 10 year figure here. Same saved
    clinical score, only changing who's left out and how they're treated. The rise comes from removing
    people with high scores who didn't have the event, mostly those who died first.

released_model_on_corrected_label.csv
    Risks from models trained on the released label, scored against the corrected label with no refit.
    Shows how far the absolute risks depend on the label they were trained on.

released_model_within_g2.csv
    Same comparison inside G2, the one group the corrected label changes, with Kaplan Meier weights
    within group like the other rows split by inferred group.

diabetes_model_calibration.csv
    O/E in people with diabetes from the model with diabetes as an input, next to the model without it.

pooled_auc_bias.csv
    Range of fold incidence and the pooled AUC of a model with no predictors, i.e. why the within fold
    AUC is the one quoted.

Usage: /usr/bin/python3 horizon_risk/framing_checks.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import horizon_lib as H
from evaluate_horizon_models import R3, censoring_weights, fold_ids, load_arm
from extended_analyses import subgroups
from fit_horizon_models import load_cohort

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def event_free_auc(label, score, weight, cause, time, years):
    """Cases by t against people still event free and alive at t (the other way to define a control under competing risks)."""
    horizon = H.horizon_days(years)
    keep = ~((cause == H.OTHER_DEATH) & (time <= horizon))
    return H.weighted_auc(label[keep], score[keep], weight[keep])


def reconciliation():
    cohort = load_cohort()
    y, time, cause = cohort["event"], cohort["time"], cohort["cause"]
    saved = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)["confounders_only__per_repeat"]
    score = saved[0]
    horizon = H.horizon_days(10)
    by_10 = (cause == H.CVD) & (time <= horizon)
    death_by_10 = (cause == H.OTHER_DEATH) & (time <= horizon)
    censored_by_10 = (cause == H.CENSORED) & (time <= horizon)
    weights, _ = censoring_weights(cohort, cause)
    label, weight = weights[10]
    rows = [
        ("Any time label, everyone", len(y), roc_auc_score(y, score)),
        ("By 10 years, deaths and censored before 10 years removed", int((~death_by_10 & ~censored_by_10).sum()), roc_auc_score(by_10[~death_by_10 & ~censored_by_10], score[~death_by_10 & ~censored_by_10])),
        ("By 10 years, censored removed, deaths kept as non-events", int((~censored_by_10).sum()), roc_auc_score(by_10[~censored_by_10], score[~censored_by_10])),
        ("By 10 years, deaths removed, censored kept as non-events", int((~death_by_10).sum()), roc_auc_score(by_10[~death_by_10], score[~death_by_10])),
        ("By 10 years, weighted, deaths kept as non-events", int((weight > 0).sum()), H.weighted_auc(label, score, weight)),
        ("By 10 years, weighted, deaths left out of the controls", int(((weight > 0) & ~death_by_10).sum()), event_free_auc(label, score, weight, cause, time, 10)),
        ("Any time label, all 448 other deaths removed", int((cause != H.OTHER_DEATH).sum()), roc_auc_score(y[cause != H.OTHER_DEATH], score[cause != H.OTHER_DEATH])),
    ]
    table = pd.DataFrame(rows, columns=["framing", "n", "auc"])
    table["score"] = "saved Aim 2 clinical logistic regression, repeat 0"
    data = load_arm("primary")
    extra = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        risks = data[f"clinical__{years}y"]
        extra.append(dict(years=years, deaths_as_controls=float(np.mean([H.weighted_auc(label, r, weight) for r in risks])),
                          deaths_left_out=float(np.mean([event_free_auc(label, r, weight, cause, time, years) for r in risks]))))
    extra = pd.DataFrame(extra)
    mean_age = {name: float(cohort["frame"]["age_s1"][mask].mean()) for name, mask in (("event by 10 years", by_10), ("died first by 10 years", death_by_10), ("event free at 10 years", time > horizon))}
    weights_case = (by_10 | death_by_10)
    case_vs_death = roc_auc_score(by_10[weights_case], score[weights_case])
    table.to_csv(os.path.join(RESULTS, "reconciliation.csv"), index=False)
    extra.to_csv(os.path.join(RESULTS, "reconciliation_by_horizon.csv"), index=False)
    pd.DataFrame([dict(mean_age, cases_against_deaths_auc=case_vs_death)]).to_csv(os.path.join(RESULTS, "reconciliation_groups.csv"), index=False)
    print(table.drop(columns="score").round(4).to_string())
    print(extra.round(4).to_string())
    print(mean_age, round(case_vs_death, 4))


def released_model_on_corrected_label():
    released, corrected = load_cohort(), load_cohort("corrected")
    data = load_arm("primary")
    weights, _ = censoring_weights(corrected, corrected["cause"])
    folds = fold_ids(released["y"])
    rows = []
    for spec in ("clinical", "clinical_recal", "clinical_cox", "oximetry", "combined_recal"):
        for years in H.HORIZONS_YEARS:
            label, weight = weights[years]
            risks = data[f"{spec}__{years}y"]
            keep = weight > 0
            per_repeat = []
            for r in range(risks.shape[0]):
                cal = H.calibration(label, risks[r], weight)
                per_repeat.append(dict(auc_wf=H.within_fold_auc(label[keep], risks[r][keep], weight[keep], folds[r][keep]), oe=cal["oe_ratio"], slope=cal["slope"]))
            rows.append(dict(spec=spec, years=years, **pd.DataFrame(per_repeat).mean().to_dict()))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "released_model_on_corrected_label.csv"), index=False)
    print(table.round(4).to_string())


def _oe_with_interval(label, weight, risks, index, rng, resamples=500):
    def oe(idx):
        return np.mean([H.weighted_incidence(label[idx], weight[idx]) / ((weight[idx] * r[idx]).sum() / weight[idx].sum()) for r in risks])
    samples = [oe(rng.choice(index, len(index), replace=True)) for _ in range(resamples)]
    return oe(index), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def released_model_within_g2():
    rng = np.random.RandomState(20260)
    released, corrected = load_cohort(), load_cohort("corrected")
    data = load_arm("primary")
    in_g2 = released["group"] == "G2"
    rows = []
    for name, cohort in (("released", released), ("corrected", corrected)):
        weights, _ = censoring_weights(cohort, cohort["cause"], kind="km_group")
        for spec in ("clinical", "clinical_recal", "clinical_cox"):
            for years in H.HORIZONS_YEARS:
                label, weight = weights[years]
                index = np.where(in_g2 & (weight > 0))[0]
                oe, lo, hi = _oe_with_interval(label, weight, data[f"{spec}__{years}y"], index, rng)
                rows.append(dict(measured_on=name, spec=spec, years=years, n=len(index), events=int(label[index].sum()), oe=oe, oe_lo=lo, oe_hi=hi))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "released_model_within_g2.csv"), index=False)
    print(table.round(3).to_string())


def diabetes_model_calibration():
    rng = np.random.RandomState(20261)
    cohort = load_cohort()
    data = load_arm("primary")
    weights, _ = censoring_weights(cohort, cohort["cause"])
    has_diabetes = subgroups(cohort)["diabetes"]
    rows = []
    for spec in ("clinical", "clinical_recal", "clinical_diabetes"):
        for years in (5, 10):
            label, weight = weights[years]
            index = np.where(has_diabetes & (weight > 0))[0]
            oe, lo, hi = _oe_with_interval(label, weight, data[f"{spec}__{years}y"], index, rng)
            rows.append(dict(spec=spec, years=years, n=len(index), events=int(label[index].sum()), oe=oe, oe_lo=lo, oe_hi=hi))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "diabetes_model_calibration.csv"), index=False)
    print(table.round(3).to_string())


def pooled_auc_bias():
    """Why the within fold AUC is quoted. Folds are stratified on the any time label, so for the 1 year label training and test incidence move in opposite directions and a model with no predictors scores below 0.5 when pooled."""
    cohort = load_cohort()
    weights, _ = censoring_weights(cohort, cohort["cause"])
    data = load_arm("primary")
    folds = fold_ids(cohort["y"])
    rows = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        incidence = [H.weighted_incidence(label[folds[r] == f], weight[folds[r] == f]) for r in range(folds.shape[0]) for f in range(5)]
        nulls = data[f"null__{years}y"]
        keep = weight > 0
        rows.append(dict(years=years, smallest_fold_incidence=min(incidence), largest_fold_incidence=max(incidence),
                         null_model_pooled_auc=float(np.mean([H.weighted_auc(label[keep], n[keep], weight[keep]) for n in nulls]))))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS, "pooled_auc_bias.csv"), index=False)
    print(table.round(4).to_string())


if __name__ == "__main__":
    reconciliation()
    released_model_on_corrected_label()
    released_model_within_g2()
    diabetes_model_calibration()
    pooled_auc_bias()
