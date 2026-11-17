"""
Measures the accuracy of the 1, 5 and 10 year risks (time dependent AUC,
calibration, Brier and IPA, decision metrics such as net benefit) using out of
fold risks only, so each person is scored by a model that never saw them.

Quote the within fold AUC, since the pooled one is biased low at 1 year (see
horizon_lib.within_fold_auc). O/E near 1 is close to automatic because the
intercept is learned from this cohort. Metrics are per repeat then averaged,
except calibration deciles and risk bands, which use the risk averaged over the
three repeats. CIs are percentile bootstraps over participants with the fitted
models and censoring weights held fixed, so they're conditional on those models
(whole_pipeline_bootstrap.py checks what that leaves out).

Runs on any_cvd as released and on the count-consistent label (see
fit_horizon_models.counts_as_event_under_corrected_rule; outputs end _corrected).
The sensitivity arms are released label only.

Reads processed/horizon_risk/oof_*.npz and writes the tables to
horizon_risk/results/ (metrics_primary.csv, contrasts.csv, classification.csv,
risk_equations.json, worked_examples.csv, metrics_sensitivity.csv and others).

Usage: /usr/bin/python3 horizon_risk/evaluate_horizon_models.py
"""

import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "aim2")]
import horizon_lib as H
from fit_horizon_models import (DRIVE_OUT, RESULTS, load_cohort, penalty_grid, prepare, tune_and_fit)
from ml_common import N_OUTER_FOLDS, N_OUTER_REPEATS, RANDOM_STATE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.simplefilter("ignore")
R2 = "/Volumes/ThesisHD/Data/processed/aim2/results/"
R3 = "/Volumes/ThesisHD/Data/processed/aim3/results/"
N_BOOT = 2000
SEED = 42
SPEC_ORDER = ["age_only", "age_sex", "clinical_no_age", "no_tests", "no_tests_oximetry", "no_tests_recording", "no_bloods", "clinical", "clinical_recal", "clinical_cox",
              "clinical_diabetes", "clinical_diabetes_cox", "oximetry", "oximetry_recal", "recording_oximetry", "combined", "combined_recal", "clinical_recording"]
POST_HOC_SPECS = ("clinical_recal", "oximetry_recal", "combined_recal", "no_tests", "no_tests_oximetry", "no_bloods", "clinical_diabetes", "clinical_diabetes_cox",
                  "recording_oximetry", "no_tests_recording", "clinical_recording")
AGE_BANDS = [(39, 50), (50, 60), (60, 70), (70, 80), (80, 120)]
# reference risk bands. 5 year: Australian guideline 2023 (low below 5%, intermediate 5 to below 10%, high 10%
# or more). 10 year: ACC/AHA 2019 (low below 5%, borderline 5 to below 7.5%, intermediate 7.5 to below 20%, high
# 20% or more). Those guidelines use different outcomes and populations, so these bands only describe where risk
# falls and don't classify anyone for treatment. No guideline has a 1 year threshold, so those are just illustrative.
RISK_GROUPS = {5: [0.05, 0.10], 10: [0.05, 0.075, 0.20]}
NET_BENEFIT_THRESHOLDS = {1: [0.01, 0.02, 0.05], 5: [0.05, 0.10], 10: [0.05, 0.075, 0.20]}
# the first three are the contrasts the thesis rests on, the rest are descriptive
PRIMARY_CONTRASTS = [("clinical", "oximetry"), ("combined", "clinical"), ("clinical", "age_sex")]
CONTRASTS = PRIMARY_CONTRASTS + [
    ("clinical", "age_only"), ("clinical_no_age", "oximetry"), ("combined", "oximetry"),
    ("clinical_cox", "clinical"), ("clinical_recal", "clinical"), ("clinical_recal", "clinical_cox"),
    ("clinical_recal", "oximetry_recal"), ("combined_recal", "clinical_recal"), ("clinical_recal", "age_only"),
    ("clinical_recal", "age_sex"), ("clinical_cox", "age_only"), ("clinical_cox", "age_sex"),
    ("clinical", "no_tests"), ("clinical", "no_bloods"), ("no_tests_oximetry", "no_tests"), ("clinical_diabetes", "clinical"), ("clinical_diabetes_cox", "clinical_cox"),
    ("clinical", "recording_oximetry"), ("recording_oximetry", "oximetry"), ("clinical_recording", "clinical"), ("no_tests_recording", "no_tests"), ("no_tests", "age_sex")]
DECISION_SPECS = ("clinical", "clinical_recal", "clinical_cox", "age_only", "age_sex", "no_tests", "oximetry", "oximetry_recal", "recording_oximetry", "combined", "combined_recal")
NET_BENEFIT_PAIRS = [("combined", "clinical"), ("combined_recal", "clinical_recal"), ("clinical", "no_tests"), ("no_tests", "age_sex")]


def fold_ids(y):
    """Which Aim 2 test fold each person is in, one row per repeat."""
    ids = np.zeros((N_OUTER_REPEATS, len(y)), int)
    for repeat in range(N_OUTER_REPEATS):
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (_, test) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
            ids[repeat, test] = fold
    return ids


def not_estimable(tag):
    """
    Horizons where a recalibrated arm can't be estimated: a slope at or below 0 in any fold reverses the
    ranking there. It's a rule on the fitted slopes applied to every arm, so nothing is picked by hand.
    """
    name = "recalibration_parameters_corrected.csv" if tag == "corrected" else "recalibration_parameters.csv"
    path = os.path.join(RESULTS, name)
    if tag not in ("primary", "corrected") or not os.path.exists(path):
        return set()
    table = pd.read_csv(path)
    bad = table[table.b <= 0].groupby(["spec", "years"]).size()
    return {(f"{spec}_recal", years) for spec, years in bad.index}


def load_arm(tag):
    """Out of fold risks for one arm. primary and corrected also pull in the recalibrated Aim 2 model and, if their files exist, the post hoc variants."""
    data = dict(np.load(os.path.join(DRIVE_OUT, f"oof_{tag}.npz")))
    if tag == "primary":
        data.update(np.load(os.path.join(DRIVE_OUT, "oof_recalibrated.npz")))
        extras = ("oof_s_tiers.npz", "oof_s_diabetes.npz", "oof_s_no_tests_oximetry.npz", "oof_s_recording.npz")
    elif tag == "corrected":
        data.update(np.load(os.path.join(DRIVE_OUT, "oof_recalibrated_corrected.npz")))
        extras = ("oof_corrected_tiers.npz", "oof_corrected_diabetes.npz", "oof_corrected_no_tests_oximetry.npz", "oof_corrected_recording.npz")
    else:
        extras = ()
    for extra in extras:
        path = os.path.join(DRIVE_OUT, extra)
        if os.path.exists(path):
            loaded = np.load(path)
            data.update({k: loaded[k] for k in loaded.files if k.split("__")[0] in SPEC_ORDER})
    return data


def censoring_weights(cohort, cause, mask=None, kind="cox"):
    """Label and weight per horizon, with the censoring model fitted on the people being evaluated."""
    keep = np.ones(len(cause), bool) if mask is None else mask
    Z = prepare(cohort["frame"][cohort["clinical"]].to_numpy(float)[keep])[0]
    time, c = cohort["time"][keep], cause[keep]
    if kind == "cox":
        model, x = H.CoxCensoring(time, c, Z, max_time=H.horizon_days(10) + 1), Z
    elif kind == "km_group":
        model, x = H.KMCensoring(time, c, cohort["group"][keep]), cohort["group"][keep]
    else:
        model, x = H.KMCensoring(time, c), None
    out = {}
    for years in H.HORIZONS_YEARS:
        label, weight = H.ipcw_labels(time, c, H.horizon_days(years), model, x)
        l, w = np.zeros(len(cause), int), np.zeros(len(cause))
        l[keep], w[keep] = label, weight
        out[years] = (l, w)
    return out, model


def point_metrics(label, weight, risks, nulls, folds):
    """Metrics for one model at one horizon, per repeat then averaged."""
    rows = []
    for r in range(risks.shape[0]):
        ok = np.isfinite(risks[r])
        y, w, p, n = label[ok], weight[ok], risks[r][ok], nulls[r][ok]
        cal = H.calibration(y, p, w)
        brier, null_brier = H.weighted_brier(y, p, w), H.weighted_brier(y, n, w)
        rows.append(dict(auc=H.weighted_auc(y, p, w), auc_wf=H.within_fold_auc(y, p, w, folds[r][ok]), brier=brier,
                         null_brier=null_brier, ipa=1 - brier / null_brier, oe=cal["oe_ratio"], slope=cal["slope"],
                         in_the_large=cal["in_the_large"], observed=cal["observed"], expected=cal["expected"]))
    frame = pd.DataFrame(rows)
    return frame.mean().to_dict(), frame.std(ddof=1).to_dict() if len(frame) > 1 else {}


def fast_metrics(label, weight, risks, nulls, folds, idx):
    """Bootstrap version of point_metrics (same definitions)."""
    y, w = label[idx], weight[idx]
    keep = w > 0
    auc, wf, brier, ipa, oe, slope = [], [], [], [], [], []
    for r in range(risks.shape[0]):
        p, n, f = risks[r][idx], nulls[r][idx], folds[r][idx]
        k = keep & np.isfinite(p)
        b, nb = H.weighted_brier(y[k], p[k], w[k]), H.weighted_brier(y[k], n[k], w[k])
        cal = H.calibration(y[k], p[k], w[k])
        auc.append(H.weighted_auc(y[k], p[k], w[k])); wf.append(H.within_fold_auc(y[k], p[k], w[k], f[k]))
        brier.append(b); ipa.append(1 - b / nb); oe.append(cal["oe_ratio"]); slope.append(cal["slope"])
    return dict(auc=np.mean(auc), auc_wf=np.mean(wf), brier=np.mean(brier), ipa=np.mean(ipa), oe=np.mean(oe), slope=np.mean(slope))


def percentile(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def evaluate_arm(tag, cohort, cause=None, mask=None, kind="cox", n_boot=0):
    data = load_arm(tag)
    cause = cohort["cause"] if cause is None else cause
    weights, _ = censoring_weights(cohort, cause, mask, kind)
    folds = fold_ids(cohort["y"])
    specs = [s for s in SPEC_ORDER if f"{s}__1y" in data]
    skip = not_estimable(tag)
    rows = []
    rng = np.random.RandomState(SEED)
    n = len(cohort["y"])
    pool = np.where(mask)[0] if mask is not None else np.arange(n)
    draws = [rng.choice(pool, len(pool), replace=True) for _ in range(n_boot)]
    contrasts = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        nulls = data[f"null__{years}y"]
        boot = {}
        for s in specs:
            if (s, years) in skip:
                rows.append(dict(arm=tag, years=years, spec=s, post_hoc=s in POST_HOC_SPECS, not_estimable=True))
                continue
            risks = data[f"{s}__{years}y"]
            mean, sd = point_metrics(label, weight, risks, nulls, folds)
            row = dict(arm=tag, years=years, spec=s, post_hoc=s in POST_HOC_SPECS, **mean)
            row.update({f"{k}_sd_repeats": v for k, v in sd.items() if k in ("auc", "auc_wf", "brier", "ipa", "oe", "slope")})
            if n_boot:
                boot[s] = pd.DataFrame([fast_metrics(label, weight, risks, nulls, folds, d) for d in draws])
                for metric in boot[s].columns:
                    row[f"{metric}_lo"], row[f"{metric}_hi"] = percentile(boot[s][metric])
            rows.append(row)
            print(f"  {tag} {years}y {s:16s} AUC {mean['auc']:.4f} within fold {mean['auc_wf']:.4f} Brier {mean['brier']:.4f} IPA {mean['ipa']:.4f} O/E {mean['oe']:.3f} slope {mean['slope']:.3f}", flush=True)
        if n_boot:
            for a, b in CONTRASTS:
                if a in boot and b in boot:
                    point = {r["spec"]: r for r in rows if r["years"] == years}
                    for metric in ("auc", "auc_wf", "ipa"):
                        delta = boot[a][metric] - boot[b][metric]
                        lo, hi = percentile(delta)
                        contrasts.append(dict(years=years, a=a, b=b, primary=(a, b) in PRIMARY_CONTRASTS, metric=metric,
                                              difference=point[a][metric] - point[b][metric], lo=lo, hi=hi))
    return pd.DataFrame(rows), pd.DataFrame(contrasts), weights


def within_groups(cohort, weights, data, skip, folds):
    """Tests whether the clinical score does more than sort people by age: AUC inside age bands and within sex, pairs compared inside a test fold."""
    age = cohort["frame"]["age_s1"].to_numpy(float)
    male = cohort["frame"]["is_male"].to_numpy(float) == 1
    groups = {f"age {lo}-{hi if hi < 120 else '+'}": (age >= lo) & (age < hi) for lo, hi in AGE_BANDS}
    groups.update({"male": male, "female": ~male})
    rng = np.random.RandomState(SEED)
    rows = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        for name, g in groups.items():
            for spec in ("age_sex", "clinical", "clinical_recal", "clinical_cox", "oximetry", "oximetry_recal", "combined"):
                if (spec, years) in skip:
                    continue
                risks = data[f"{spec}__{years}y"]
                cases = int(((label == 1) & g).sum())
                # under 20 events a within group AUC and its CI are too unstable to report
                if cases < 20:
                    rows.append(dict(years=years, group=name, spec=spec, n=int(g.sum()), events=cases))
                    continue
                def auc(idx):
                    return np.mean([H.stratified_auc(label[idx], risks[r][idx], weight[idx], folds[r][idx]) for r in range(len(risks))])
                index = np.where(g)[0]
                draws = [auc(rng.choice(index, len(index), replace=True)) for _ in range(500)]
                lo, hi = percentile(draws)
                rows.append(dict(years=years, group=name, spec=spec, n=int(g.sum()), events=cases, auc=auc(index), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def stratum_auc(cohort, weights, data, folds, strata, specs, pairs):
    """AUC with pairs compared only within one stratum and one test fold, per repeat then averaged, plus paired differences."""
    rng = np.random.RandomState(SEED)
    n = len(strata)
    draws = [rng.randint(0, n, n) for _ in range(500)]
    rows = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        def value(spec, idx):
            return np.mean([H.stratified_auc(label[idx], r[idx], weight[idx], strata[idx] * 10 + folds[k][idx]) for k, r in enumerate(data[f"{spec}__{years}y"])])
        everyone = np.arange(n)
        boot = {spec: np.array([value(spec, d) for d in draws]) for spec in specs}
        for spec in specs:
            lo, hi = percentile(boot[spec])
            rows.append(dict(years=years, spec=spec, comparison="", auc=value(spec, everyone), lo=lo, hi=hi))
        for a, b in pairs:
            lo, hi = percentile(boot[a] - boot[b])
            rows.append(dict(years=years, spec=a, comparison=b, auc=value(a, everyone) - value(b, everyone), lo=lo, hi=hi))
    return pd.DataFrame(rows)


STRATUM_SPECS = ("age_only", "age_sex", "clinical", "clinical_recal", "clinical_cox", "oximetry", "combined", "no_tests")
STRATUM_PAIRS = [("combined", "clinical"), ("clinical", "age_sex"), ("clinical_recal", "age_sex"), ("clinical", "oximetry")]


def age_band_auc(cohort, weights, data, folds):
    """Pairs compared only within a 10 year age band and one test fold, so neither age nor the fold can do the sorting."""
    age = cohort["frame"]["age_s1"].to_numpy(float)
    bands = np.digitize(age, [lo for lo, _ in AGE_BANDS[1:]])
    return stratum_auc(cohort, weights, data, folds, bands, STRATUM_SPECS, STRATUM_PAIRS)


def inferred_group_auc(cohort, weights, data, folds):
    """Pairs compared only within an inferred parent group and one test fold. G4 is old and holds half the 10 year events, so a lot of the overall AUC is just separating groups."""
    strata = np.searchsorted(["G1", "G2", "G3", "G4"], cohort["group"])
    return stratum_auc(cohort, weights, data, folds, strata, STRATUM_SPECS, STRATUM_PAIRS)


def rank_only_models(cohort, weights):
    """
    Clinical, oximetry and combined inputs under every model in the thesis at each horizon, to show whether
    the oximetry result depends on the model. The saved scores were trained on the any event label, so only
    the AUC (a ranking) means anything. The tree files only hold the score averaged over repeats, so that's the
    basis for logistic regression, random forest and XGBoost; the CNN has one repeat and is compared with
    logistic regression on that same repeat. Differences this small don't rank the models.
    """
    a2 = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)
    a3 = np.load(R3 + "oof_A.npz", allow_pickle=True)
    scores = {}
    for spec in ("confounders_only", "obm_only", "obm_plus_confounders"):
        # row 0 of these files is the label, row 1 the mean out of fold score
        for model, name in (("logistic regression", "logreg"), ("random forest", "rf"), ("xgboost", "xgb")):
            saved = np.load(R2 + f"{name}_oof_scores.npz")[spec]
            if not np.array_equal(saved[0].astype(int), cohort["y"]):
                raise ValueError(f"{name} {spec}: saved scores are not in cohort order")
            scores[("score averaged over 3 repeats", model, spec)] = saved[1:2]
        scores[("repeat 0", "logistic regression", spec)] = a2[spec + "__per_repeat"][:1]
    scores[("repeat 0", "1D CNN", "obm_only")] = a3["deep_marg"][:1]
    scores[("repeat 0", "1D CNN stacked with clinical", "obm_plus_confounders")] = a3["stacked"][:1]
    scores[("repeat 0", "clinical comparator for the CNN", "confounders_only")] = a3["clinical"][:1]
    rows = []
    rng = np.random.RandomState(SEED)
    draws = [rng.randint(0, len(cohort["y"]), len(cohort["y"])) for _ in range(1000)]
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        for (basis, model, spec), s in scores.items():
            value = H.weighted_auc(label, s[0], weight)
            lo, hi = percentile([H.weighted_auc(label[d], s[0][d], weight[d]) for d in draws])
            rows.append(dict(years=years, basis=basis, model=model, input=spec, auc=value, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def saved_score_contrast(cohort, weights):
    """Clinical minus oximetry on the saved Aim 2 logistic regression scores, within folds, per repeat then averaged."""
    a2 = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)
    clinical, oximetry = a2["confounders_only__per_repeat"], a2["obm_only__per_repeat"]
    folds = fold_ids(cohort["y"])
    rng = np.random.RandomState(SEED)
    draws = [rng.randint(0, len(cohort["y"]), len(cohort["y"])) for _ in range(1000)]
    rows = []
    for years in H.HORIZONS_YEARS:
        label, w = weights[years]
        def value(scores, idx):
            return np.mean([H.within_fold_auc(label[idx], scores[r][idx], w[idx], folds[r][idx]) for r in range(len(scores))])
        everyone = np.arange(len(label))
        a, b = np.array([value(clinical, d) for d in draws]), np.array([value(oximetry, d) for d in draws])
        lo, hi = percentile(a - b)
        rows.append(dict(years=years, clinical=value(clinical, everyone), clinical_lo=percentile(a)[0], clinical_hi=percentile(a)[1],
                         oximetry=value(oximetry, everyone), oximetry_lo=percentile(b)[0], oximetry_hi=percentile(b)[1],
                         difference=value(clinical, everyone) - value(oximetry, everyone), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def decision_tables(cohort, weights, data, skip):
    """Accuracy at decision thresholds. The above average threshold only uses the training fold's incidence."""
    rows, risk_groups, curves, deciles, nb_rows, capture = [], [], [], [], [], []
    rng = np.random.RandomState(SEED)
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        nulls = data[f"null__{years}y"]
        for spec in DECISION_SPECS:
            if (spec, years) in skip or f"{spec}__{years}y" not in data:
                continue
            risks = data[f"{spec}__{years}y"]
            mean_risk = risks.mean(axis=0)
            per_repeat = []
            for r in range(len(risks)):
                c = H.classification(label, risks[r] >= nulls[r], weight)
                per_repeat.append(c)
            rows.append(dict(years=years, spec=spec, rule="risk above the training average",
                             **pd.DataFrame(per_repeat).mean().to_dict()))
            for threshold in NET_BENEFIT_THRESHOLDS[years]:
                per = [H.classification(label, risks[r] >= threshold, weight) for r in range(len(risks))]
                rows.append(dict(years=years, spec=spec, rule=f"risk at least {threshold:.1%}",
                                 **pd.DataFrame(per).mean().to_dict()))
                nb = np.mean([H.net_benefit(label, risks[r], weight, threshold) for r in range(len(risks))])
                treat_all = H.net_benefit(label, np.ones(len(label)), weight, threshold)
                nb_rows.append(dict(years=years, spec=spec, threshold=threshold, net_benefit=nb, treat_all=treat_all, treat_none=0.0))
            for top in (0.10, 0.20, 0.30):
                per = []
                for r in range(len(risks)):
                    # exactly this share of everyone, known outcome or not. Age alone gives tied risks within a year of age, so ties break at random
                    order = np.lexsort((rng.random_sample(len(label)), -risks[r]))
                    chosen = np.zeros(len(label), bool)
                    chosen[order[:int(round(top * len(label)))]] = True
                    per.append(H.classification(label, chosen, weight))
                capture.append(dict(years=years, spec=spec, top_share=top, share_of_events_captured=np.mean([c["sensitivity"] for c in per]),
                                    observed_risk_in_group=np.mean([c["ppv"] for c in per])))
            known = weight > 0
            edges = np.quantile(mean_risk[known], np.linspace(0, 1, 11))
            band = np.clip(np.searchsorted(edges, mean_risk, side="right") - 1, 0, 9)
            for row in H.calibration_groups(label, mean_risk, weight):
                members = np.where(known & (band == row["group"] - 1))[0]
                boot = [H.weighted_incidence(label[d], weight[d]) for d in (rng.choice(members, len(members), replace=True) for _ in range(300))]
                lo, hi = percentile(boot)
                deciles.append(dict(years=years, spec=spec, observed_lo=lo, observed_hi=hi, **row))
            if years in RISK_GROUPS:
                edges = [0.0] + RISK_GROUPS[years] + [1.01]
                for lo, hi in zip(edges[:-1], edges[1:]):
                    k = (mean_risk >= lo) & (mean_risk < hi) & (weight > 0)
                    if k.sum() == 0:
                        risk_groups.append(dict(years=years, spec=spec, group=f"{lo:.1%} to {min(hi, 1):.1%}", n=0))
                        continue
                    # n and events are raw counts among people with a known outcome. For the share of the cohort
                    # above a threshold use flagged_share from the classification table, which is weighted
                    risk_groups.append(dict(years=years, spec=spec, group=f"{lo:.1%} to {min(hi, 1):.1%}", n=int(k.sum()),
                                            events=int(label[k].sum()), mean_predicted=float((weight[k] * mean_risk[k]).sum() / weight[k].sum()),
                                            observed=H.weighted_incidence(label[k], weight[k])))
            top = {1: 0.10, 5: 0.40, 10: 0.60}[years]
            for threshold in np.linspace(0.002 if years == 1 else 0.01, top, 60):
                nb = np.mean([H.net_benefit(label, risks[r], weight, threshold) for r in range(len(risks))])
                curves.append(dict(years=years, spec=spec, threshold=threshold, net_benefit=nb,
                                   treat_all=H.net_benefit(label, np.ones(len(label)), weight, threshold)))
    return (pd.DataFrame(rows), pd.DataFrame(risk_groups), pd.DataFrame(curves), pd.DataFrame(deciles),
            pd.DataFrame(nb_rows), pd.DataFrame(capture))


def net_benefit_contrasts(cohort, weights, data, skip):
    """Net benefit with CIs, plus paired differences against age alone and age and sex."""
    rng = np.random.RandomState(SEED)
    draws = [rng.randint(0, len(cohort["y"]), len(cohort["y"])) for _ in range(1000)]
    rows = []
    for years in H.HORIZONS_YEARS:
        label, weight = weights[years]
        for threshold in NET_BENEFIT_THRESHOLDS[years]:
            def nb(spec, idx):
                return np.mean([H.net_benefit(label[idx], r[idx], weight[idx], threshold) for r in data[f"{spec}__{years}y"]])
            specs = [s for s in DECISION_SPECS if (s, years) not in skip and f"{s}__{years}y" in data]
            boot = {s: np.array([nb(s, d) for d in draws]) for s in specs}
            everyone = np.arange(len(label))
            for s in specs:
                lo, hi = percentile(boot[s])
                rows.append(dict(years=years, threshold=threshold, spec=s, comparison="", net_benefit=nb(s, everyone), lo=lo, hi=hi))
                for base in ("age_only", "age_sex"):
                    if s == base:
                        continue
                    lo, hi = percentile(boot[s] - boot[base])
                    rows.append(dict(years=years, threshold=threshold, spec=s, comparison=base, net_benefit=nb(s, everyone) - nb(base, everyone), lo=lo, hi=hi))
            for a, b in NET_BENEFIT_PAIRS:
                if a in boot and b in boot and b not in ("age_only", "age_sex"):
                    lo, hi = percentile(boot[a] - boot[b])
                    rows.append(dict(years=years, threshold=threshold, spec=a, comparison=b, net_benefit=nb(a, everyone) - nb(b, everyone), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def final_equations(cohort):
    """
    Clinical risk equation at each horizon, fitted on everyone so it can be written down. Its accuracy
    is the cross validated accuracy above; these coefficients were never used to measure it.
    """
    frame, time, cause = cohort["frame"], cohort["time"], cohort["cause"]
    names = cohort["clinical"]
    Xraw = frame[names].to_numpy(float)
    Z = prepare(Xraw)[0]
    censor = H.CoxCensoring(time, cause, Z, max_time=H.horizon_days(10) + 1)
    grid = penalty_grid()
    out, rows = {}, []
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    for years in H.HORIZONS_YEARS:
        label, weight = H.ipcw_labels(time, cause, H.horizon_days(years), censor, Z)
        keep = weight > 0
        w = weight[keep] / weight[keep].mean()
        config, _ = tune_and_fit(Xraw[keep], label[keep], w, grid)
        imputer = SimpleImputer(strategy="median").fit(Xraw[keep])
        scaler = StandardScaler().fit(imputer.transform(Xraw[keep]))
        model = LogisticRegression(**config).fit(scaler.transform(imputer.transform(Xraw[keep])), label[keep], sample_weight=w)
        out[years] = dict(intercept=float(model.intercept_[0]), coef=dict(zip(names, map(float, model.coef_[0]))),
                          medians=dict(zip(names, map(float, imputer.statistics_))),
                          means=dict(zip(names, map(float, scaler.mean_))), sds=dict(zip(names, map(float, scaler.scale_))),
                          penalty=config["penalty"], C=config["C"], l1_ratio=config.get("l1_ratio"))
        for n, b in zip(names, model.coef_[0]):
            rows.append(dict(years=years, variable=n, odds_ratio_per_sd=float(np.exp(b)), coefficient_per_sd=float(b)))
    return out, pd.DataFrame(rows)


def final_cox(cohort):
    """Cause specific Cox pair fitted on everyone. Kept as an object since its risk is a sum over event times."""
    Xraw = cohort["frame"][cohort["clinical"]].to_numpy(float)
    return H.CauseSpecificCox(cohort["time"], cohort["cause"], prepare(Xraw)[0])


def worked_examples(equations, cohort, recalibrated):
    """
    Risks for two made-up people from all three estimators, to show what the equations give and how much they
    differ for one person. Neither is a study participant, and the risks describe a 1990s research cohort, so
    they aren't anyone's risk today. Units as in the SHHS file: blood pressure mmHg, lipids mg/dL. Smoking
    status codes are 0 never, 1 current, 2 former; race codes are 1 White, 2 Black, 3 other.
    """
    people = {
        "A: woman aged 52, normal blood pressure, never smoked, no medication": dict(
            age_s1=52, bmi_s1=26, SystBP=118, DiasBP=74, Chol=190, HDL=60, Trig=110, BETA1=0, ACE1=0, CCB1=0, DIURET1=0, LIPID1=0,
            HTNDerv_s1=0, is_male=0, race_2=0, race_3=0, smokstat_s1_1=0, smokstat_s1_2=0),
        "B: man aged 72, treated hypertension, former smoker": dict(
            age_s1=72, bmi_s1=30, SystBP=150, DiasBP=82, Chol=220, HDL=40, Trig=170, BETA1=1, ACE1=1, CCB1=0, DIURET1=0, LIPID1=0,
            HTNDerv_s1=1, is_male=1, race_2=0, race_3=0, smokstat_s1_1=0, smokstat_s1_2=1),
    }
    names = cohort["clinical"]
    Xraw = cohort["frame"][names].to_numpy(float)
    medians = np.nanmedian(Xraw, axis=0)
    filled = np.where(np.isnan(Xraw), medians, Xraw)
    means, sds = filled.mean(axis=0), filled.std(axis=0)
    cox = final_cox(cohort)
    days = [H.horizon_days(y) for y in H.HORIZONS_YEARS]
    rows = []
    for name, person in people.items():
        row = dict(person=name)
        for years, e in equations.items():
            lp = e["intercept"] + sum(e["coef"][k] * (person[k] - e["means"][k]) / e["sds"][k] for k in e["coef"])
            row[f"separate_lr_{years}y"] = float(H.expit(lp))
        lp = recalibrated["intercept"] + sum(recalibrated["coef"][k] * (person[k] - recalibrated["means"][k]) / recalibrated["sds"][k] for k in names)
        for years in H.HORIZONS_YEARS:
            line = recalibrated["horizons"][str(years)]
            row[f"recalibrated_{years}y"] = float(H.expit(line["a"] + line["b"] * lp))
        z = ((np.array([person[k] for k in names], float) - means) / sds)[None, :]
        risks = cox.risk(z, days)
        for years, d in zip(H.HORIZONS_YEARS, days):
            row[f"cox_{years}y"] = float(risks[d][0])
        rows.append(row)
    return pd.DataFrame(rows)


def sample_size_check(cohort, weights):
    """
    Whether there's enough data at each horizon for this many predictors. Riley criteria for small overfitting,
    small optimism and a precise overall risk, using a shrunken R squared. Using them with censoring weights and
    a competing event stretches the published formula, which is for a binary or a survival outcome.
    """
    rows = []
    for spec in ("clinical", "oximetry", "combined"):
        Z = prepare(cohort["frame"][cohort["specs"][spec]].to_numpy(float))[0]
        p_columns = Z.shape[1]
        for years in H.HORIZONS_YEARS:
            y, w = weights[years]
            k = w > 0
            fit = LogisticRegression(C=1e6, max_iter=5000).fit(Z[k], y[k], sample_weight=w[k])
            p = fit.predict_proba(Z[k])[:, 1]
            n_eff = w[k].sum()
            null_rate = H.weighted_incidence(y[k], w[k])
            ll1 = -H.weighted_logloss(y[k], p, w[k]) * n_eff
            ll0 = -H.weighted_logloss(y[k], np.full(k.sum(), null_rate), w[k]) * n_eff
            r2 = 1 - np.exp(-2 * (ll1 - ll0) / n_eff)
            rows.append(dict(spec=spec, years=years, parameters=p_columns, n_with_known_outcome=int(k.sum()), events=int(y[k].sum()),
                             events_per_parameter=float(y[k].sum() / p_columns), r2_apparent=float(r2),
                             n_required_apparent_r2=float(H.required_sample_size(p_columns, r2)),
                             **H.riley_criteria(p_columns, r2, int(k.sum()), null_rate)))
    table = pd.DataFrame(rows)
    table["n_required"] = table[["n_small_overfitting", "n_small_optimism", "n_overall_risk"]].max(axis=1)
    return table


def cohort_audit(cohort, weights, data):
    time, cause, y = cohort["time"], cohort["cause"], cohort["y"]
    info = dict(n=int(len(y)), events=int((cause == H.CVD).sum()), other_deaths=int((cause == H.OTHER_DEATH).sum()), censored_alive=int((cause == H.CENSORED).sum()),
                zero_follow_up=int((time <= 0).sum()), any_cvd_zero_but_cardiovascular_date_present=int(cohort["inconsistent"].sum()),
                event_date_after_last_contact=int(cohort["late_event"].sum()))
    for years in H.HORIZONS_YEARS:
        h = H.horizon_days(years)
        label, w = weights[years]
        info[f"{years}y"] = dict(
            events=int(((cause == H.CVD) & (time <= h)).sum()), other_deaths=int(((cause == H.OTHER_DEATH) & (time <= h)).sum()),
            censored_alive_before=int(((cause == H.CENSORED) & (time <= h)).sum()), still_followed=int((time > h).sum()),
            cumulative_incidence=H.weighted_incidence(label, w), largest_weight=float(w.max()), mean_weight=float(w[w > 0].mean()))
    for spec in ("clinical", "clinical_recal"):
        p1, p5, p10 = (data[f"{spec}__{y}y"] for y in H.HORIZONS_YEARS)
        info[f"share_with_risk_falling_as_horizon_lengthens_{spec}"] = dict(
            one_above_five=float((p1 > p5).mean()), five_above_ten=float((p5 > p10).mean()))
    return info


def evaluate_label(tag, label, suffix):
    """Everything for one event flag. Sensitivity arms are released label only."""
    cohort = load_cohort(label)
    data = load_arm(tag)
    skip = not_estimable(tag)
    def path(name, extension="csv"):
        return os.path.join(RESULTS, f"{name}{suffix}.{extension}")
    print(tag, "with bootstrap. Not estimable:", sorted(skip), flush=True)
    main_table, contrast_table, weights = evaluate_arm(tag, cohort, n_boot=N_BOOT)
    main_table.to_csv(path("metrics_primary"), index=False)
    contrast_table.to_csv(path("contrasts"), index=False)
    json.dump(cohort_audit(cohort, weights, data), open(path("cohort_audit", "json"), "w"), indent=1)
    print("Within groups", flush=True)
    folds = fold_ids(cohort["y"])
    within_groups(cohort, weights, data, skip, folds).to_csv(path("within_groups"), index=False)
    age_band_auc(cohort, weights, data, folds).to_csv(path("age_band_auc"), index=False)
    group_weights = censoring_weights(cohort, cohort["cause"], kind="km_group")[0]
    inferred_group_auc(cohort, group_weights, data, folds).to_csv(path("inferred_group_auc"), index=False)
    print("Other models, ranking only", flush=True)
    # these saved scores were trained on the released label; in the _corrected files they're only scored against the other outcome
    rank_only_models(cohort, weights).assign(training_label="released").to_csv(path("rank_only_all_models"), index=False)
    saved_score_contrast(cohort, weights).assign(training_label="released").to_csv(path("saved_score_contrast"), index=False)
    print("Decision tables", flush=True)
    decisions, groups, curves, deciles, nb, capture = decision_tables(cohort, weights, data, skip)
    decisions.to_csv(path("classification"), index=False)
    groups.to_csv(path("reference_risk_bands"), index=False)
    curves.to_csv(path("net_benefit_curves"), index=False)
    deciles.to_csv(path("calibration_deciles"), index=False)
    nb.to_csv(path("net_benefit"), index=False)
    capture.to_csv(path("risk_capture"), index=False)
    net_benefit_contrasts(cohort, weights, data, skip).to_csv(path("net_benefit_contrasts"), index=False)
    print("Sample size and final equations", flush=True)
    sample_size_check(cohort, weights).to_csv(path("sample_size"), index=False)
    equations, coefficient_table = final_equations(cohort)
    json.dump(equations, open(path("risk_equations", "json"), "w"), indent=1)
    coefficient_table.to_csv(path("risk_equation_coefficients"), index=False)
    recalibrated = json.load(open(path("recalibration_final", "json")))
    worked_examples(equations, cohort, recalibrated).to_csv(path("worked_examples"), index=False)
    return cohort


def main():
    cohort = evaluate_label("primary", "released", "")
    print("Sensitivity arms", flush=True)
    death_censored = np.where(cohort["cause"] == H.OTHER_DEATH, H.CENSORED, cohort["cause"])
    arms = [evaluate_arm("s_death_as_censoring", cohort, cause=death_censored)[0],
            evaluate_arm("s_marginal_km", cohort, kind="km")[0],
            evaluate_arm("s_km_by_group", cohort, kind="km_group")[0],
            evaluate_arm("s_no_weights", cohort)[0],
            evaluate_arm("s_no_prior_any", cohort, mask=~cohort["prior_any"])[0],
            evaluate_arm("s_no_prior_composite", cohort, mask=~cohort["prior_composite"])[0],
            evaluate_arm("s_all_discordant", load_cohort("all_discordant"))[0]]
    pd.concat(arms).to_csv(os.path.join(RESULTS, "metrics_sensitivity.csv"), index=False)
    evaluate_label("corrected", "corrected", "_corrected")
    print("done", flush=True)


if __name__ == "__main__":
    main()
