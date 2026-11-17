"""
Baseline characteristics table (Table 1) for the thesis report. Only reads saved
files; no model is refitted and no biomarker is recomputed.

Continuous variables get mean (SD), median [Q1, Q3], Welch t and Mann-Whitney U;
categorical ones get n (% of non-missing) and one Pearson chi-square per variable
on the full r x 2 table, with no continuity correction. The absolute SMD (pooled
SD) is included because p values say very little at this sample size. Categorical
variables come from the raw SHHS-1 file, not the model's dummy columns, so missing
smoking status shows as missing instead of being coded as never smoker.

Inputs: processed/aim2/modelling_dataset.csv and processed/aim2/obm_features.csv,
documentation/shhs1-dataset-0.15.0.csv and
documentation/shhs-cvd-summary-dataset-0.15.0.csv

Outputs (thesis_report/facts/)
    table1_baseline.csv                the 4,352 analysed, overall and by incident event
    table1_id_block_comparison.csv     analysed IDs up to 202600 vs IDs above it
    table1_excluded_comparison.csv     analysed vs each excluded group

Usage:
    /usr/bin/python3 thesis_report/scripts/table1_baseline.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

AIM2 = "/Volumes/ThesisHD/Data/processed/aim2"
DOC = "/Volumes/ThesisHD/Data/documentation"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "facts")

CLINICAL = [
    "nsrrid", "age_s1", "gender", "race", "bmi_s1", "SystBP", "DiasBP", "Chol",
    "HDL", "Trig", "HTNDerv_s1", "BETA1", "ACE1", "CCB1", "DIURET1", "LIPID1",
    "smokstat_s1", "SlpPrdP", "rdi4p", "ahi_a0h4", "ParRptDiab", "HTNMED1",
]
OXIMETRY = [
    "pid", "sufficient_valid_data", "sleep_ct90_pct", "sleep_odi3", "sleep_odi4",
    "sleep_mean", "sleep_min", "rem_odi3", "sleep_valid_hours",
    "recording_valid_hours",
]

CONTINUOUS = [
    # (group, column, label)
    ("Demographics", "age_s1", "Age (years)"),
    ("Clinical", "bmi_s1", "BMI (kg/m2)"),
    ("Clinical", "SystBP", "Systolic BP (mmHg)"),
    ("Clinical", "DiasBP", "Diastolic BP (mmHg)"),
    ("Clinical", "Chol", "Total cholesterol (mg/dL)"),
    ("Clinical", "HDL", "HDL cholesterol (mg/dL)"),
    ("Clinical", "Trig", "Triglycerides (mg/dL)"),
    ("Oximetry", "sleep_ct90_pct", "T90 (% of valid sleep time below 90%)"),
    ("Oximetry", "sleep_odi3", "ODI3 (events/h of valid sleep)"),
    ("Oximetry", "sleep_odi4", "ODI4 (events/h of valid sleep)"),
    ("Oximetry", "rem_odi3", "REM ODI3 (events/h of valid REM)"),
    ("Oximetry", "sleep_mean", "Mean SpO2 in valid sleep (%)"),
    ("Oximetry", "sleep_min", "Nadir SpO2 in valid sleep (%)"),
    ("Sleep", "sleep_valid_hours", "Valid sleep time, pipeline (h)"),
    ("Sleep", "recording_valid_hours", "Valid recording time, pipeline (h)"),
    ("Sleep", "tst_hours", "Total sleep time, published SlpPrdP (h)"),
    ("Sleep", "rdi4p", "RDI at 4% desaturation, published rdi4p (events/h)"),
    ("Sleep", "ahi_a0h4", "AHI at 4% desaturation, published ahi_a0h4 (events/h)"),
    ("Follow-up", "followup_years", "Days to last contact or death, censdate (years)"),
]

CATEGORICAL = [
    # (group, column, label, {value: level label})
    ("Demographics", "age_band", "Age band",
     {"<55": "under 55", "55-64": "55 to 64", "65-74": "65 to 74", ">=75": "75 or over"}),
    ("Demographics", "gender", "Sex", {1: "Male", 2: "Female"}),
    ("Demographics", "race", "Race", {1: "White", 2: "Black", 3: "Other"}),
    ("Clinical", "smokstat_s1", "Smoking status",
     {0: "Never", 1: "Current", 2: "Former"}),
    ("Clinical", "HTNDerv_s1", "Hypertension (derived)", {1: "Yes"}),
    ("Medication", "BETA1", "Beta blocker without diuretic", {1: "Yes"}),
    ("Medication", "ACE1", "ACE inhibitor without diuretic", {1: "Yes"}),
    ("Medication", "CCB1", "Any calcium channel blocker", {1: "Yes"}),
    ("Medication", "DIURET1", "Any diuretic", {1: "Yes"}),
    ("Medication", "LIPID1", "Any lipid lowering medication", {1: "Yes"}),
    ("Not in the model", "HTNMED1", "Any antihypertensive medication", {1: "Yes"}),
    ("Not in the model", "ParRptDiab", "Diabetes, parent cohort report", {1: "Yes"}),
    ("Sleep", "osa_band", "OSA severity by published rdi4p",
     {"<5": "under 5", "5-15": "5 to under 15", "15-30": "15 to under 30", ">=30": "30 or over"}),
    ("Follow-up", "vital", "Vital status at last contact", {1: "Alive", 0: "Dead"}),
]


def fmt_mean_sd(x):
    x = x.dropna()
    return f"{x.mean():.2f} ({x.std(ddof=1):.2f})" if len(x) else ""


def fmt_median_iqr(x):
    x = x.dropna()
    if not len(x):
        return ""
    return f"{x.median():.2f} [{x.quantile(0.25):.2f}, {x.quantile(0.75):.2f}]"


def smd_continuous(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def smd_binary(p1, p2):
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2.0)
    return abs(p1 - p2) / pooled if pooled > 0 else np.nan


def summarise(frame, group_col, name_a, name_b):
    """One row per variable (per level for categorical ones). group_col is 0 for group A, 1 for group B."""
    a = frame[frame[group_col] == 0]
    b = frame[frame[group_col] == 1]
    rows = []

    for group, col, label in CONTINUOUS:
        x_all, x_a, x_b = frame[col], a[col], b[col]
        t_p = stats.ttest_ind(x_a.dropna(), x_b.dropna(), equal_var=False).pvalue
        u_p = stats.mannwhitneyu(x_a.dropna(), x_b.dropna(), alternative="two-sided").pvalue
        rows.append({
            "group": group, "variable": col, "label": label, "level": "",
            "kind": "continuous",
            "overall": fmt_mean_sd(x_all), "overall_median_iqr": fmt_median_iqr(x_all),
            name_a: fmt_mean_sd(x_a), f"{name_a}_median_iqr": fmt_median_iqr(x_a),
            name_b: fmt_mean_sd(x_b), f"{name_b}_median_iqr": fmt_median_iqr(x_b),
            "missing_overall_n": int(x_all.isna().sum()),
            "missing_overall_pct": round(100 * x_all.isna().mean(), 2),
            f"missing_{name_a}_n": int(x_a.isna().sum()),
            f"missing_{name_b}_n": int(x_b.isna().sum()),
            "range_overall": f"{x_all.min():.2f} to {x_all.max():.2f}",
            "p_value": t_p, "p_value_mannwhitney": u_p, "test": "Welch t; Mann-Whitney U",
            "smd": smd_continuous(x_a, x_b),
        })

    for group, col, label, levels in CATEGORICAL:
        x_all, x_a, x_b = frame[col], a[col], b[col]
        table = pd.crosstab(frame[col], frame[group_col])
        chi = stats.chi2_contingency(table.values, correction=False)
        first = True
        for value, level_name in levels.items():
            def share(s):
                nonmissing = s.notna().sum()
                n = int((s == value).sum())
                return n, nonmissing, (100.0 * n / nonmissing if nonmissing else np.nan)
            n_o, d_o, p_o = share(x_all)
            n_a, d_a, p_a = share(x_a)
            n_b, d_b, p_b = share(x_b)
            rows.append({
                "group": group, "variable": col, "label": label, "level": level_name,
                "kind": "categorical",
                "overall": f"{n_o} ({p_o:.1f}%)", "overall_median_iqr": "",
                name_a: f"{n_a} ({p_a:.1f}%)", f"{name_a}_median_iqr": "",
                name_b: f"{n_b} ({p_b:.1f}%)", f"{name_b}_median_iqr": "",
                "missing_overall_n": int(x_all.isna().sum()),
                "missing_overall_pct": round(100 * x_all.isna().mean(), 2),
                f"missing_{name_a}_n": int(x_a.isna().sum()),
                f"missing_{name_b}_n": int(x_b.isna().sum()),
                "range_overall": "",
                "p_value": chi[1] if first else np.nan,
                "p_value_mannwhitney": np.nan,
                "test": f"Pearson chi-square, {chi[2]} df" if first else "",
                "smd": smd_binary(p_a / 100.0, p_b / 100.0),
            })
            first = False
    return pd.DataFrame(rows)


def main():
    clinical = pd.read_csv(os.path.join(DOC, "shhs1-dataset-0.15.0.csv"),
                           low_memory=False, usecols=lambda c: c in CLINICAL)
    cvd = pd.read_csv(os.path.join(DOC, "shhs-cvd-summary-dataset-0.15.0.csv"),
                      low_memory=False,
                      usecols=["nsrrid", "censdate", "vital", "any_cvd"])
    oximetry = pd.read_csv(os.path.join(AIM2, "obm_features.csv"), usecols=OXIMETRY)
    flags = pd.read_csv(os.path.join(AIM2, "modelling_dataset.csv"),
                        usecols=["pid", "any_prevalent_cvd", "cvd_incident"])

    data = oximetry.merge(clinical, left_on="pid", right_on="nsrrid", how="left")
    data = data.drop(columns=["nsrrid"]).merge(
        cvd, left_on="pid", right_on="nsrrid", how="left")
    data = data.merge(flags, on="pid", how="left")

    data["tst_hours"] = data["SlpPrdP"] / 60.0
    data["followup_years"] = data["censdate"] / 365.25
    data["age_band"] = pd.cut(data["age_s1"], [0, 54.5, 64.5, 74.5, 200],
                              labels=["<55", "55-64", "65-74", ">=75"]).astype(object)
    data["osa_band"] = pd.cut(data["rdi4p"], [-1, 4.999999, 14.999999, 29.999999, 1e6],
                              labels=["<5", "5-15", "15-30", ">=30"]).astype(object)

    # each participant with a computed recording gets exactly one cohort state
    def state(row):
        if not bool(row["sufficient_valid_data"]):
            return "under_4h"
        if pd.isna(row["nsrrid"]) or row["pid"] in (204709, 204806):
            return "no_cvd_row"
        if pd.isna(row["any_prevalent_cvd"]):
            return "no_outcome"
        return "prevalent" if row["any_prevalent_cvd"] == 1 else "analysed"

    data["state"] = data.apply(state, axis=1)
    counts = data["state"].value_counts().to_dict()
    print("cohort states:", counts)
    assert counts["analysed"] == 4352 and counts["prevalent"] == 684
    assert counts["no_outcome"] == 753

    cohort = data[data["state"] == "analysed"].copy()
    assert int(cohort["cvd_incident"].sum()) == 829
    assert (cohort["cvd_incident"] == cohort["any_cvd"]).all()

    # Table 1, overall and split by incident event
    cohort["split"] = cohort["cvd_incident"].astype(int)
    table = summarise(cohort, "split", "no_event", "incident_event")
    table.to_csv(os.path.join(OUT, "table1_baseline.csv"), index=False)
    print(f"Table 1 rows: {len(table)}; n = {len(cohort)}, events = {int(cohort['split'].sum())}")

    # analysed participants split at the old half cohort boundary (ID 202600)
    cohort["split_ids"] = (cohort["pid"] > 202600).astype(int)
    blocks = summarise(cohort, "split_ids", "ids_to_202600", "ids_above_202600")
    rate = cohort.groupby("split_ids")["cvd_incident"].agg(["size", "sum", "mean"])
    chi_rate = stats.chi2_contingency(
        pd.crosstab(cohort["split_ids"], cohort["cvd_incident"]).values, correction=False)
    print("ID blocks:", rate.to_dict(), "event rate chi-square p =", chi_rate[1])
    blocks.to_csv(os.path.join(OUT, "table1_id_block_comparison.csv"), index=False)

    # analysed vs each excluded group that's big enough to compare
    frames = []
    for other in ("prevalent", "no_outcome"):
        subset = data[data["state"].isin(["analysed", other])].copy()
        subset["split_state"] = (subset["state"] == other).astype(int)
        result = summarise(subset, "split_state", "analysed", other)
        result.insert(0, "comparison", f"analysed vs {other}")
        frames.append(result)
    excluded = pd.concat(frames, ignore_index=True)
    excluded.to_csv(os.path.join(OUT, "table1_excluded_comparison.csv"), index=False)
    print("Written:", OUT)


if __name__ == "__main__":
    main()
