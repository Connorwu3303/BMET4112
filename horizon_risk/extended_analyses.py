"""
Extra checks on the risk models that evaluate_horizon_models.py doesn't do.

These were added after the first results were reviewed, so they're all post hoc, and none of
them changes a model. Each covers something the headline numbers can't:

    cohort_risk_by_label      how much the cohort risk itself depends on who counts as an event
    endpoint_composition      how much of the outcome a guideline's narrower outcome would leave out
    calibration_by_subgroup   whether the risks hold within sex, race, age, parent cohort group,
                              diabetes and prior disease
    parent_group_validation   what happens to a group when it's held out of training. Closest thing
                              to external validation these files allow (groups are inferred from
                              which columns are filled in)
    flagged_group_outcomes    what actually happens to the people flagged as high risk
    person_level_stability    how far one person's risk moves between repeats and between estimators
    missing_data              how much of each predictor is missing, and where
    severe_subgroups          whether oximetry adds anything where disease is more severe
    learning_curve            whether more participants of the same kind would help
    label_effect              what the corrected label does to the Aims 2 to 4 results
    follow_up                 reverse Kaplan Meier follow up
    recalibration_summary     how stable a(t) and b(t) were

Outputs go to horizon_risk/results/, mostly one csv or json per check named after it.

Usage: /usr/bin/python3 horizon_risk/extended_analyses.py
"""

import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "aim2")]
import horizon_lib as H
from evaluate_horizon_models import R2, R3, SEED, censoring_weights, fold_ids, load_arm, not_estimable, percentile
from fit_horizon_models import DOC, DRIVE_OUT, N_JOBS, RESULTS, load_cohort, penalty_grid, prepare, tune_and_fit
from ml_common import N_OUTER_FOLDS, RANDOM_STATE

warnings.simplefilter("ignore")
DAYS = [H.horizon_days(y) for y in H.HORIZONS_YEARS]
HARD_DATES = ["mi_date", "chd_dthdt", "stk_date"]


def candidate_fatal_strokes(cohort):
    """People in G2 who died on a stroke date but aren't flagged, possibly because of the same count test. Candidates only."""
    cv = cohort["cv"]
    events = pd.read_csv(DOC + "shhs-cvd-events-dataset-0.15.0.csv")
    position = pd.Series(np.arange(len(cohort["pid"])), index=cohort["pid"])
    strokes = events[(events.event == 10) & events.nsrrid.isin(cohort["pid"])]
    index = position.loc[strokes.nsrrid].to_numpy()
    censor = cv["censdate"].to_numpy(float)
    on_death_day = np.zeros(len(censor), bool)
    on_death_day[index[strokes.event_dt.to_numpy(float) == censor[index]]] = True
    return (cohort["group"] == "G2") & on_death_day & (cv["vital"].to_numpy() == 0) & (cv["stroke"].to_numpy(float) >= 2) & (cohort["y"] == 0)


def cohort_risk_by_label():
    """Aalen Johansen cumulative incidence under each event definition. Same estimator throughout so the numbers can be compared directly."""
    released, corrected, discordant = load_cohort(), load_cohort("corrected"), load_cohort("all_discordant")
    rows = []
    def add(name, time, cause, keep=None):
        keep = np.ones(len(time), bool) if keep is None else keep
        aj = H.aalen_johansen(time[keep], cause[keep], DAYS)
        counts = [int(((cause[keep] == H.CVD) & (time[keep] <= d)).sum()) for d in DAYS]
        rows.append(dict(definition=name, n=int(keep.sum()), events_any_time=int((cause[keep] == H.CVD).sum()),
                         **{f"risk_{y}y": float(a) for y, a in zip(H.HORIZONS_YEARS, aj)},
                         **{f"events_{y}y": c for y, c in zip(H.HORIZONS_YEARS, counts)}))
    for name, c in (("released", released), ("corrected", corrected), ("all 84 discordant counted", discordant)):
        add(name, c["time"], c["cause"])
        add(name + ", without 163 reporting prior composite disease", c["time"], c["cause"], ~c["prior_composite"])
        add(name + ", without 263 reporting any prior disease", c["time"], c["cause"], ~c["prior_any"])
    fatal = candidate_fatal_strokes(released)
    time, cause = corrected["time"].copy(), corrected["cause"].copy()
    cause[fatal] = H.CVD
    add("corrected, plus candidate fatal strokes at the death date", time, cause)
    # range of possible readings of the G2 label, no refit. The repair isn't unique: heart failure isn't a clear ARIC endpoint in the documents, and stroke alone counts elsewhere
    cv, group, y, first = released["cv"], released["group"], released["y"], released["first"]
    censor, vital = cv["censdate"].to_numpy(float), cv["vital"].to_numpy()
    g2 = group == "G2"
    coronary = (cv[["mi", "ptca", "cabg", "revasc_proc"]].fillna(0) >= 1).any(axis=1).to_numpy() | (cv["chd_death"] == 1).to_numpy() | (cv["cvd_death"] == 1).to_numpy()
    stroke = (cv["stroke"].fillna(0) >= 1).to_numpy()
    without_chf = ((y == 1) & ~(g2 & ~coronary)) | (corrected["repaired"] & coronary)
    clock = first.copy()
    coronary_dates = cv[["mi_date", "chd_dthdt", "cvd_dthdt", "revpro_date", "ptca_date", "cabg_date"]].min(axis=1).to_numpy(float)
    clock[g2] = np.where(np.isfinite(coronary_dates[g2]), coronary_dates[g2], first[g2])
    t, c = H.build_time_and_cause(without_chf.astype(int), clock, censor, vital)
    add("no heart failure in G2, coronary count test applied", t, c)
    t, c = H.build_time_and_cause(((corrected["event"] == 1) | (g2 & stroke)).astype(int), first, censor, vital)
    add("corrected, plus stroke alone in G2", t, c)
    # in G2, time from the printed components only, leaving out the stroke, MI and procedure dates that don't set the flag there
    counted_dates = cv[["mi_date", "chd_dthdt", "cvd_dthdt", "revpro_date", "ptca_date", "cabg_date", "chf_date"]].min(axis=1).to_numpy(float)
    clock = first.copy()
    clock[g2] = np.where(np.isfinite(counted_dates[g2]), counted_dates[g2], first[g2])
    for name, event in (("released, time from printed components in G2", released["event"]), ("corrected, time from printed components in G2", corrected["event"])):
        t, c = H.build_time_and_cause(event, clock, censor, vital)
        add(name, t, c)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "cohort_risk_by_label.csv"), index=False)
    return int(fatal.sum())


def endpoint_composition(n_fatal):
    """
    A guideline's narrower outcome leaves out heart failure and revascularisation; this gives a rough idea of how
    much that matters. Stroke counts in every group here, even though any_cvd doesn't count it alone in G2.
    """
    c = load_cohort()
    cv = c["cv"]
    def narrow(columns):
        first = cv[columns].min(axis=1).to_numpy(float)
        censor = cv["censdate"].to_numpy(float)
        has = np.isfinite(first)
        time = np.where(has, first, censor)
        cause = np.where(has, H.CVD, np.where(cv["vital"].to_numpy() == 0, H.OTHER_DEATH, H.CENSORED))
        return H.aalen_johansen(time, cause, DAYS)
    hard_first = cv[HARD_DATES].min(axis=1).to_numpy(float)
    by_ten = (c["cause"] == H.CVD) & (c["time"] <= H.horizon_days(10))
    def on_first_day(columns):
        return (cv[columns].to_numpy(float) == c["time"][:, None]).any(axis=1)
    hard_day = by_ten & on_first_day(HARD_DATES)
    hf_revasc_day = by_ten & ~hard_day & on_first_day(["chf_date", "revpro_date", "ptca_date", "cabg_date"])
    mip_day = by_ten & ~hard_day & ~hf_revasc_day & on_first_day(["mip_date"])
    cvd_death_day = by_ten & ~hard_day & ~hf_revasc_day & ~mip_day & on_first_day(["cvd_dthdt"])
    info = dict(first_events_by_10y=int(by_ten.sum()), first_event_mi_chd_death_stroke=int(hard_day.sum()), first_event_heart_failure_or_revascularisation=int(hf_revasc_day.sum()),
                first_event_mi_or_procedure_unspecified=int(mip_day.sum()), first_event_cvd_death_only=int(cvd_death_day.sum()),
                events_with_no_mi_chd_death_stroke_by_10y=int((by_ten & ~(hard_first <= H.horizon_days(10))).sum()),
                narrow_mi_chd_death_stroke=dict(zip(map(str, H.HORIZONS_YEARS), map(float, narrow(HARD_DATES)))),
                narrow_plus_cvd_death=dict(zip(map(str, H.HORIZONS_YEARS), map(float, narrow(HARD_DATES + ["cvd_dthdt"])))),
                released_composite=dict(zip(map(str, H.HORIZONS_YEARS), map(float, H.aalen_johansen(c["time"], c["cause"], DAYS)))),
                candidate_fatal_strokes=n_fatal, discordant_84_split=dict(
                    repaired_by_corrected_rule=int(c["repaired"].sum()), candidate_fatal_stroke=n_fatal,
                    other=int(c["inconsistent"].sum() - c["repaired"].sum() - n_fatal)))
    age = c["frame"]["age_s1"].to_numpy(float)
    info["age_outside_40_79"] = int(((age < 40) | (age > 79)).sum())
    info["age_outside_45_79"] = int(((age < 45) | (age > 79)).sum())
    info["on_lipid_medication_share"] = float(c["frame"]["LIPID1"].mean())
    json.dump(info, open(os.path.join(RESULTS, "endpoint_composition.json"), "w"), indent=1)


def subgroups(cohort):
    frame, base = cohort["frame"], cohort["baseline"]
    age = frame["age_s1"].to_numpy(float)
    diabetes = base["ParRptDiab"].to_numpy(float)
    groups = {"women": frame["is_male"].to_numpy(float) == 0, "men": frame["is_male"].to_numpy(float) == 1,
              "White": (frame["race_2"].to_numpy(float) == 0) & (frame["race_3"].to_numpy(float) == 0),
              "Black": frame["race_2"].to_numpy(float) == 1, "other race": frame["race_3"].to_numpy(float) == 1,
              "diabetes": diabetes == 1, "no diabetes": diabetes == 0, "diabetes not recorded": np.isnan(diabetes),
              "reported prior disease (263)": cohort["prior_any"], "no reported prior disease": ~cohort["prior_any"]}
    for lo, hi in ((39, 50), (50, 60), (60, 70), (70, 80), (80, 120)):
        groups[f"age {lo} to {hi - 1 if hi < 120 else 'over'}"] = (age >= lo) & (age < hi)
    for g in ("G1", "G2", "G3", "G4"):
        groups[f"inferred group {g}"] = cohort["group"] == g
    return groups


def calibration_by_subgroup():
    """
    O/E, slope and AUC within each subgroup, from the out of fold risks. Under 20 events only the counts are given.
    AUC compares pairs inside a test fold. Rows split by inferred group use Kaplan Meier weights within group, since
    most censoring falls in one group and the Cox censoring model has no group term.
    """
    rows = []
    rng = np.random.RandomState(SEED)
    for label, tag in (("released", "primary"), ("corrected", "corrected")):
        cohort = load_cohort(label)
        data = load_arm(tag)
        folds = fold_ids(cohort["y"])
        cox_weights, _ = censoring_weights(cohort, cohort["cause"])
        group_weights, _ = censoring_weights(cohort, cohort["cause"], kind="km_group")
        for years in H.HORIZONS_YEARS:
            for name, g in subgroups(cohort).items():
                y, w = (group_weights if name.startswith("inferred group") else cox_weights)[years]
                for spec in ("clinical", "clinical_recal", "clinical_cox", "oximetry"):
                    known = g & (w > 0)
                    events = int(y[known].sum())
                    row = dict(label=label, years=years, group=name, spec=spec, n=int(known.sum()), events=events)
                    if events >= 20:
                        risks = data[f"{spec}__{years}y"]
                        index = np.where(known)[0]
                        def oe(idx):
                            return np.mean([H.weighted_incidence(y[idx], w[idx]) / ((w[idx] * r[idx]).sum() / w[idx].sum()) for r in risks])
                        lo, hi = percentile([oe(rng.choice(index, len(index), replace=True)) for _ in range(500)])
                        cal = [H.calibration(y[index], r[index], w[index]) for r in risks]
                        row.update(oe=oe(index), oe_lo=lo, oe_hi=hi, slope=float(np.mean([c["slope"] for c in cal])),
                                   observed=float(np.mean([c["observed"] for c in cal])), expected=float(np.mean([c["expected"] for c in cal])),
                                   auc=float(np.mean([H.stratified_auc(y[index], r[index], w[index], folds[k][index]) for k, r in enumerate(risks)])))
                    rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "calibration_by_subgroup.csv"), index=False)


def _hold_out_one(cohort, group_name, evaluation):
    """Train on the other inferred groups, then predict the held out one. Folds aren't used here."""
    frame, time, cause = cohort["frame"], cohort["time"], cohort["cause"]
    names = cohort["clinical"]
    X = frame[names].to_numpy(float)
    held = cohort["group"] == group_name
    train, test = np.where(~held)[0], np.where(held)[0]
    Zt, Zte = prepare(X[train], X[test])
    censor = H.CoxCensoring(time[train], cause[train], Zt, max_time=H.horizon_days(10) + 1)
    cox = H.CauseSpecificCox(time[train], cause[train], Zt).risk(Zte, DAYS)
    rows = []
    for years, day in zip(H.HORIZONS_YEARS, DAYS):
        label, weight = H.ipcw_labels(time[train], cause[train], day, censor, Zt)
        keep = weight > 0
        w = weight[keep] / weight[keep].mean()
        config, _ = tune_and_fit(X[train][keep], label[keep], w, penalty_grid())
        A, B = prepare(X[train][keep], X[test])
        lr = LogisticRegression(**config).fit(A, label[keep], sample_weight=w).predict_proba(B)[:, 1]
        y_all, w_all = evaluation[years]
        y, wt = y_all[test], w_all[test]
        for estimator, risk in (("separate LR", lr), ("cause specific Cox", cox[day])):
            row = dict(label=cohort["label"], held_out=group_name, years=years, estimator=estimator, n=int((wt > 0).sum()), events=int(y[wt > 0].sum()))
            if row["events"] >= 20:
                cal = H.calibration(y, risk, wt)
                index = np.where(wt > 0)[0]
                rng = np.random.RandomState(SEED)
                draws = [rng.choice(index, len(index), replace=True) for _ in range(500)]
                lo, hi = percentile([H.weighted_incidence(y[d], wt[d]) / ((wt[d] * risk[d]).sum() / wt[d].sum()) for d in draws])
                row.update(observed=cal["observed"], expected=cal["expected"], oe=cal["oe_ratio"], oe_lo=lo, oe_hi=hi, slope=cal["slope"], auc=H.weighted_auc(y, risk, wt))
            rows.append(row)
    return rows


def parent_group_validation():
    rows = []
    for label in ("released", "corrected"):
        cohort = load_cohort(label)
        # weights by group, same reason as in calibration_by_subgroup (training weights still come from the Cox model on the other groups)
        evaluation, _ = censoring_weights(cohort, cohort["cause"], kind="km_group")
        done = Parallel(n_jobs=4)(delayed(_hold_out_one)(cohort, g, evaluation) for g in ("G1", "G2", "G3", "G4"))
        rows += [r for part in done for r in part]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "parent_group_validation.csv"), index=False)


def flagged_group_outcomes():
    """A low PPV is partly other deaths, so 'no event' doesn't mean the flagged person stayed well."""
    rows = []
    for name_label, tag in (("released", "primary"), ("corrected", "corrected")):
        cohort = load_cohort(name_label)
        data = load_arm(tag)
        weights, _ = censoring_weights(cohort, cohort["cause"])
        for years, threshold in ((5, 0.10), (10, 0.20)):
            y, w = weights[years]
            day = H.horizon_days(years)
            died = ((cohort["cause"] == H.OTHER_DEATH) & (cohort["time"] <= day)).astype(float)
            for spec in ("clinical", "clinical_recal", "clinical_cox", "age_sex"):
                per = []
                for r in data[f"{spec}__{years}y"]:
                    flagged = r >= threshold
                    for name, k in (("flagged", flagged), ("not flagged", ~flagged)):
                        total = w[k].sum()
                        per.append(dict(group=name, event=(w[k] * y[k]).sum() / total, other_death=(w[k] * died[k]).sum() / total))
                mean = pd.DataFrame(per).groupby("group").mean()
                for name in mean.index:
                    rows.append(dict(label=name_label, years=years, threshold=threshold, spec=spec, group=name, event=mean.loc[name, "event"], other_death=mean.loc[name, "other_death"],
                                     alive_without_event=1 - mean.loc[name, "event"] - mean.loc[name, "other_death"]))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "flagged_group_outcomes.csv"), index=False)


def person_level_stability():
    """Agreement across the cohort doesn't carry over to one person. Spread between repeats reflects models trained on 80% of the data."""
    data = load_arm("primary")
    bands = {1: None, 5: [0.05, 0.10], 10: [0.05, 0.075, 0.20]}
    rows = []
    for years in H.HORIZONS_YEARS:
        direct, recal, cox = (data[f"{s}__{years}y"] for s in ("clinical", "clinical_recal", "clinical_cox"))
        for name, risks in (("separate LR", direct), ("recalibrated", recal), ("cause specific Cox", cox)):
            spread = (risks.max(axis=0) - risks.min(axis=0)) * 100
            cv_ = risks.std(axis=0, ddof=1) / risks.mean(axis=0) * 100
            row = dict(years=years, estimator=name, median_range_points=float(np.median(spread)), p90_range_points=float(np.percentile(spread, 90)), median_cv_percent=float(np.median(cv_)))
            if bands[years]:
                b = np.digitize(risks, bands[years])
                row["share_changing_band_between_repeats"] = float((b.max(axis=0) != b.min(axis=0)).mean())
            rows.append(row)
        ratio = direct.mean(axis=0) / cox.mean(axis=0)
        ratio_recal = recal.mean(axis=0) / cox.mean(axis=0)
        row = dict(years=years, estimator="ratio to Cox", lr_p5=float(np.percentile(ratio, 5)), lr_p95=float(np.percentile(ratio, 95)),
                   recal_p5=float(np.percentile(ratio_recal, 5)), recal_p95=float(np.percentile(ratio_recal, 95)))
        if bands[years]:
            all_bands = np.stack([np.digitize(a.mean(axis=0), bands[years]) for a in (direct, recal, cox)])
            row["share_differing_band_across_estimators"] = float((all_bands.max(axis=0) != all_bands.min(axis=0)).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "person_level_stability.csv"), index=False)


def missing_data():
    cohort = load_cohort()
    frame = cohort["frame"]
    rows = [dict(variable=c, missing=int(frame[c].isna().sum()), percent=float(frame[c].isna().mean() * 100)) for c in cohort["clinical"]]
    any_missing = frame[cohort["clinical"]].isna().any(axis=1).to_numpy()
    for g in ("G1", "G2", "G3", "G4"):
        rows.append(dict(variable=f"any clinical variable missing, {g}", missing=int((any_missing & (cohort["group"] == g)).sum()), percent=float(any_missing[cohort["group"] == g].mean() * 100)))
    rows.append(dict(variable="any clinical variable missing, all", missing=int(any_missing.sum()), percent=float(any_missing.mean() * 100)))
    # the two smoking dummies have no missing values since a missing status falls into the never smoker reference group
    rows.append(dict(variable="smoking status missing in the raw file, coded as never smoker", missing=int(cohort["baseline"]["smokstat_s1"].isna().sum()), percent=float(cohort["baseline"]["smokstat_s1"].isna().mean() * 100)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "missing_data.csv"), index=False)


def severe_subgroups():
    """The narrow overall CIs say little about severe disease, so these are shown separately. Post hoc with few events. Pairs compared inside a test fold."""
    rows = []
    for label, tag in (("released", "primary"), ("corrected", "corrected")):
        cohort = load_cohort(label)
        data = load_arm(tag)
        folds = fold_ids(cohort["y"])
        weights, _ = censoring_weights(cohort, cohort["cause"])
        base = cohort["baseline"]
        groups = {"respiratory disturbance index (4%) 30 or more": base["rdi4p"].to_numpy(float) >= 30,
                  "T90 20% or more": base["pctlt90"].to_numpy(float) >= 20,
                  "men up to 70": (cohort["frame"]["is_male"].to_numpy(float) == 1) & (cohort["frame"]["age_s1"].to_numpy(float) <= 70)}
        rng = np.random.RandomState(SEED)
        for years in (5, 10):
            y, w = weights[years]
            for name, g in groups.items():
                index = np.where(g & (w > 0))[0]
                def auc(spec, idx):
                    return np.mean([H.stratified_auc(y[idx], r[idx], w[idx], folds[k][idx]) for k, r in enumerate(data[f"{spec}__{years}y"])])
                draws = [rng.choice(index, len(index), replace=True) for _ in range(1000)]
                a, b, o = (np.array([auc(s, d) for d in draws]) for s in ("combined", "clinical", "oximetry"))
                lo, hi = percentile(a - b)
                oe = np.mean([H.weighted_incidence(y[index], w[index]) / ((w[index] * r[index]).sum() / w[index].sum()) for r in data[f"clinical__{years}y"]])
                rows.append(dict(label=label, years=years, group=name, n=len(index), events=int(y[index].sum()), combined_minus_clinical=auc("combined", index) - auc("clinical", index), lo=lo, hi=hi,
                                 clinical_auc=auc("clinical", index), oximetry_auc=auc("oximetry", index), oximetry_lo=percentile(o)[0], oximetry_hi=percentile(o)[1], clinical_oe=float(oe)))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "severe_subgroups.csv"), index=False)


def _learning_fold(cohort, spec, repeat, fold, train, test, share, grid):
    frame, time, cause = cohort["frame"], cohort["time"], cohort["cause"]
    rng = np.random.RandomState(RANDOM_STATE + 100 * repeat + fold)
    part = np.sort(rng.choice(train, int(round(share * len(train))), replace=False))
    Z = frame[cohort["clinical"]].to_numpy(float)
    X = frame[cohort["specs"][spec]].to_numpy(float)
    # censoring model uses the whole training fold. A quarter of it only has about 44 censorings for 18 covariates and the Cox fit doesn't converge
    Zt = prepare(Z[train])[0]
    label, weight = H.ipcw_labels(time[train], cause[train], H.horizon_days(10), H.CoxCensoring(time[train], cause[train], Zt, max_time=H.horizon_days(10) + 1), Zt)
    label, weight = label[np.searchsorted(train, part)], weight[np.searchsorted(train, part)]
    keep = weight > 0
    w = weight[keep] / weight[keep].mean()
    config, _ = tune_and_fit(X[part][keep], label[keep], w, grid)
    A, B = prepare(X[part][keep], X[test])
    return dict(spec=spec, repeat=repeat, fold=fold, share=share, n_train=len(part), test=test,
                risk=LogisticRegression(**config).fit(A, label[keep], sample_weight=w).predict_proba(B)[:, 1])


def learning_curve():
    """10 year AUC vs training size for penalised logistic regression on an L2 grid. Says nothing about new kinds of data."""
    cohort = load_cohort()
    y = cohort["y"]
    grid = [dict(penalty="l2", solver="lbfgs", C=float(c), max_iter=5000, tol=1e-5, random_state=RANDOM_STATE) for c in np.logspace(-4, 1.5, 12)]
    tasks = []
    for repeat in range(3):
        skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + repeat)
        for fold, (train, test) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
            for spec in ("clinical", "oximetry", "combined"):
                for share in (0.25, 0.5, 0.75, 1.0):
                    tasks.append((cohort, spec, repeat, fold, train, test, share, grid))
    done = Parallel(n_jobs=N_JOBS)(delayed(_learning_fold)(*t) for t in tasks)
    weights, _ = censoring_weights(cohort, cohort["cause"])
    label, w = weights[10]
    rows = []
    for d in done:
        t = d["test"]
        rows.append(dict(spec=d["spec"], share=d["share"], n_train=d["n_train"], repeat=d["repeat"], fold=d["fold"], auc=H.weighted_auc(label[t], d["risk"], w[t])))
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(RESULTS, "learning_curve_folds.csv"), index=False)
    table = per_fold.groupby(["spec", "share"]).agg(n_train=("n_train", "mean"), auc=("auc", "mean"), sd_between_folds=("auc", "std")).reset_index()
    table.to_csv(os.path.join(RESULTS, "learning_curve.csv"), index=False)


def t90_quintile_hazard(cohort):
    """
    Top fifth of T90 vs T90 of 0 in a Cox model. T90 is exactly 0 in 1,180 people, so an earlier version that split by
    rank depended on row order. Here the other values are split into three groups by value, so row order doesn't matter.
    """
    frame = cohort["frame"]
    t90 = frame["sleep_ct90_pct"].to_numpy(float)
    filled = np.where(np.isfinite(t90), t90, np.nanmedian(t90))
    group = np.full(len(filled), 4)
    group[filled == 0] = 0
    middle = (filled != 0) & (filled < np.quantile(filled, 0.8))
    group[middle] = 1 + pd.qcut(filled[middle], 3, labels=False)
    dummies = np.column_stack([(group == i).astype(float) for i in range(1, 5)])
    Z = prepare(frame[cohort["clinical"]].to_numpy(float))[0]
    names = cohort["clinical"]
    ok = cohort["time"] > 0
    status = (cohort["cause"] == H.CVD).astype(int)
    designs = {"unadjusted": dummies, "age_sex": np.column_stack([dummies, Z[:, names.index("age_s1")], Z[:, names.index("is_male")]]),
               "all_18": np.column_stack([dummies, Z])}
    out = dict(group_sizes=[int((group == i).sum()) for i in range(5)], event_share=[float(status[group == i].mean()) for i in range(5)])
    for name, design in designs.items():
        fit = sm.PHReg(cohort["time"][ok] / 365.25, design[ok], status=status[ok], ties="efron").fit(disp=0)
        b, se = float(fit.params[3]), float(fit.bse[3])
        out[name] = dict(hr=float(np.exp(b)), lo=float(np.exp(b - 1.96 * se)), hi=float(np.exp(b + 1.96 * se)), p=float(2 * stats.norm.sf(abs(b / se))))
    return out


def label_effect():
    """
    Saved Aims 2 to 4 scores against each label (no refit), then the Aim 2 logistic regression refit on the corrected
    label. The CNN wasn't retrained, so its row only shows its saved score against the corrected label.
    """
    released, corrected = load_cohort(), load_cohort("corrected")
    a2 = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)
    a3 = np.load(R3 + "oof_A.npz", allow_pickle=True)
    refit = np.load(os.path.join(DRIVE_OUT, "oof_recalibrated_corrected.npz"))
    rows = []
    g2 = released["group"] == "G2"
    def add(model, spec, scores):
        # overall AUC drops because the new G2 events get compared with people in other groups; inside G2 it can go up
        rows.append(dict(model=model, input=spec, auc_released=float(np.mean([roc_auc_score(released["event"], s) for s in scores])),
                         auc_corrected=float(np.mean([roc_auc_score(corrected["event"], s) for s in scores])),
                         auc_released_within_g2=float(np.mean([roc_auc_score(released["event"][g2], s[g2]) for s in scores])),
                         auc_corrected_within_g2=float(np.mean([roc_auc_score(corrected["event"][g2], s[g2]) for s in scores]))))
    for spec, short in (("confounders_only", "clinical"), ("obm_only", "oximetry"), ("obm_plus_confounders", "combined")):
        add("logistic regression, mean of 3 repeats", spec, a2[spec + "__per_repeat"])
        rows[-1]["auc_refit_on_corrected"] = float(np.mean([roc_auc_score(corrected["event"], s) for s in refit[f"{short}_aim2score"]]))
        for model, name in (("random forest", "rf"), ("xgboost", "xgb")):
            add(model + ", score averaged over repeats", spec, np.load(R2 + f"{name}_oof_scores.npz")[spec][1:2])
    add("1D CNN, repeat 0, uncalibrated", "obm_only", a3["deep_marg"][:1])
    add("1D CNN stacked with clinical, repeat 0", "obm_plus_confounders", a3["stacked"][:1])
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "label_effect.csv"), index=False)
    json.dump(dict(released=t90_quintile_hazard(released), corrected=t90_quintile_hazard(corrected)),
              open(os.path.join(RESULTS, "t90_hazard_by_label.json"), "w"), indent=1)


def follow_up():
    c = load_cohort()
    q = [x / 365.25 for x in H.reverse_km_quantiles(c["time"], c["cause"])]
    info = dict(reverse_km_median_years=q[1], reverse_km_q1_years=q[0], reverse_km_q3_years=q[2], zero_follow_up=int((c["time"] <= 0).sum()),
                no_outcome_recorded=int(pd.read_csv(DOC + "shhs-cvd-summary-dataset-0.15.0.csv")["any_cvd"].isna().sum()))
    json.dump(info, open(os.path.join(RESULTS, "follow_up.json"), "w"), indent=1)


def recalibration_summary():
    rows = []
    for name, file in (("released", "recalibration_parameters.csv"), ("corrected", "recalibration_parameters_corrected.csv")):
        table = pd.read_csv(os.path.join(RESULTS, file))
        for (spec, years), g in table.groupby(["spec", "years"]):
            rows.append(dict(label=name, spec=spec, years=years, mean_a=g.a.mean(), mean_b=g.b.mean(), min_b=g.b.min(), max_b=g.b.max(),
                             folds_b_at_or_below_0=int((g.b <= 0).sum()), training_events_min=int(g.events_fit.min()), training_events_max=int(g.events_fit.max())))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "recalibration_summary.csv"), index=False)


def main():
    steps = [("recalibration", recalibration_summary), ("follow up", follow_up), ("missing data", missing_data),
             ("flagged", flagged_group_outcomes), ("stability", person_level_stability), ("severe", severe_subgroups),
             ("label effect", label_effect), ("subgroups", calibration_by_subgroup), ("learning curve", learning_curve),
             ("parent groups", parent_group_validation)]
    print("cohort risk", flush=True)
    n_fatal = cohort_risk_by_label()
    endpoint_composition(n_fatal)
    for name, step in steps:
        print(name, flush=True)
        step()
    print("done", flush=True)


if __name__ == "__main__":
    main()
