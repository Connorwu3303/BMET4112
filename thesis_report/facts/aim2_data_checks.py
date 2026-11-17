"""
Read-only checks behind the Aim 2 data fact sheet (aim2_data.md).

Everything is recomputed from saved files: nothing is refitted, no pipeline script is
run and no EDF is opened. One cleaned signal per sampled participant is read to get the
SpO2 quantisation step. The printout is saved as aim2_data_checks_output.txt.

Usage:
    /usr/bin/python3 thesis_report/facts/aim2_data_checks.py > thesis_report/facts/aim2_data_checks_output.txt
"""

import json
import re

import numpy as np
import pandas as pd
from scipy import stats

P = "/Volumes/ThesisHD/Data/processed/"
A = P + "aim2/"
D = "/Volumes/ThesisHD/Data/documentation/"

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 900)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 70)


def section(title):
    print("\n" + "#" * 8 + " " + title)


obm = pd.read_csv(A + "obm_features.csv")
manifest = json.load(open(A + "feature_manifest.json"))
cvd = pd.read_csv(D + "shhs-cvd-summary-dataset-0.15.0.csv", low_memory=False)
core = manifest["core_obm_panel"]

section("1. Column reconciliation of obm_features.csv")
cols = obm.columns.tolist()
print("total columns", len(cols), "rows", len(obm))
non_feature = [c for c in cols if c in ("pid", "sufficient_valid_data")]
cross = ["rem_minus_nrem_mean", "rem_minus_nrem_nadir", "rem_nrem_odi3_ratio"]
for w in ("recording", "sleep", "nrem", "rem"):
    these = [c for c in cols if c.startswith(w + "_") and c not in cross]
    print(w, len(these))
print("cross window", [c for c in cross if c in cols], "identifiers/flags", non_feature)
meta_suffix = ("valid_hours", "n_valid_samples", "n_segments", "longest_segment_min",
               "complexity_segment_min")
meta = [c for c in cols if c.endswith(meta_suffix)]
print("recording-description columns", len(meta))
n_events = [c for c in cols if re.search(r"_odi\d_n_events$", c)]
print("event-count columns", len(n_events))
print("numeric columns that are biomarker values (781-2-meta):", len(cols) - 2 - len(meta))

section("2. Completeness and missingness in the full 5,793")
for w in ("recording", "sleep", "nrem", "rem"):
    c = [x for x in cols if x.startswith(w + "_")]
    print(w, len(c), "mean populated", round(obm[c].notna().mean(axis=1).mean() * 100, 3))
mf = obm.isna().mean().sort_values(ascending=False)
print("columns >50% missing", int((mf > 0.5).sum()), "max missing", mf.index[0], round(mf.iloc[0], 4))
print(mf.head(30).round(4).to_string())
print("no scored REM in 5,793:", int((obm.rem_valid_hours.fillna(0) == 0).sum()))
log = pd.read_csv(A + "obm_extraction_log.csv")
print("extraction log status", log.status.value_counts().to_dict(), "n_features", log.n_features.value_counts().to_dict())
print("seconds per participant: median", round(log.seconds.median(), 2), "sum hours", round(log.seconds.sum() / 3600, 2))

section("3. Reproduce the pruning on the 5,036 rows")
modelling = pd.read_csv(A + "modelling_dataset.csv", usecols=["pid", "any_prevalent_cvd", "cvd_incident"])
frame = obm[obm.pid.isin(modelling.pid)].copy()
print("rows", len(frame))
housekeeping = {"pid", "nsrrid", "sufficient_valid_data", "any_prevalent_cvd", "cvd_incident"}
outcome_names = set(manifest["outcomes"])
candidates = [c for c in frame.columns if c not in housekeeping and c not in outcome_names
              and pd.api.types.is_numeric_dtype(frame[c]) and not c.endswith("_n_valid_samples")]
print("candidates", len(candidates))
nonmissing = frame[candidates].notna().mean()
sparse = nonmissing[nonmissing < 0.80].index.tolist()
print("sparse", len(sparse), sparse)
kept = [c for c in candidates if c not in sparse]
low = [c for c in kept if frame[c].nunique(dropna=True) < 5]
print("low variation", low)
kept = [c for c in kept if c not in low]
corr = frame[kept].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
redundant = [c for c in upper.columns if (upper[c] > 0.95).any()]
kept2 = [c for c in kept if c not in redundant]
print("correlated", len(redundant), "kept", len(kept2))
print("identical to manifest obm_features:", kept2 == manifest["obm_features"])
print("groups", {k: len(v) for k, v in manifest["obm_groups"].items()})
meta_in = [c for c in manifest["obm_features"] if c.endswith(meta_suffix) or c.endswith("_n_events")]
print("recording-description or event-count columns inside the 516:", len(meta_in), meta_in)
print("core panel inside the 516:", len(set(core) & set(manifest["obm_features"])))
print("core panel outside the 516:", sorted(set(core) - set(manifest["obm_features"])))

section("4. Core panel in the 4,352: distribution, missingness, correlation")
coh_ids = modelling.loc[modelling.any_prevalent_cvd == 0, "pid"]
coh = obm[obm.pid.isin(coh_ids)]
rows = []
for c in core:
    x = coh[c]
    y = obm[c]
    rows.append(dict(feature=c, n=int(x.notna().sum()), miss_4352=int(x.isna().sum()),
                     miss_pct=round(x.isna().mean() * 100, 2), mean=round(x.mean(), 4), sd=round(x.std(), 4),
                     median=round(x.median(), 4), q1=round(x.quantile(0.25), 4), q3=round(x.quantile(0.75), 4),
                     min=round(x.min(), 4), max=round(x.max(), 4), miss_5793=int(y.isna().sum())))
print(pd.DataFrame(rows).to_string())
pc = coh[core].corr().abs()
pairs = [(a, b, round(pc.loc[a, b], 4)) for i, a in enumerate(core) for b in core[i + 1:] if pc.loc[a, b] > 0.95]
print("pairs with |r| > 0.95 in the panel:", len(pairs))
for p in sorted(pairs, key=lambda t: -t[2]):
    print("  ", p)
print("rem_nrem_odi3_ratio finite max", coh.rem_nrem_odi3_ratio.max())

# check the feature file and the modelling file agree
mod_core = pd.read_csv(A + "modelling_dataset.csv", usecols=["pid"] + core)
joined = mod_core.merge(obm[["pid"] + core], on="pid", suffixes=("", "_o"))
same = all(np.allclose(joined[c].to_numpy(dtype=float), joined[c + "_o"].to_numpy(dtype=float), equal_nan=True) for c in core)
print("core columns identical between modelling_dataset.csv and obm_features.csv:", same)

section("5. Recompute the validation gates and the checks the report does not contain")
wanted = ["pid", "losao2r", "losao2nr", "sao2rem", "sao2nrem", "pctlt90", "pctlt85", "pctlt80", "pctlt75", "rdi4p",
          "rdi3p", "frac_valid", "rem_min", "nrem_min", "rem_mean", "nrem_mean", "sleep_ct90_pct",
          "sleep_ct85_pct", "sleep_ct80_pct", "sleep_ct75_pct", "sleep_odi4", "sleep_odi3", "sleep_mean",
          "sleep_valid_hours", "timest1", "timest2", "timest34", "timerem", "ahi_a0h4", "ahi_a0h3"]
merged = pd.read_csv(A + "obm_vs_shhs_published.csv", usecols=lambda c: c in wanted)
print("merged", merged.shape)


def agree(pub, our, label):
    b = pd.DataFrame({"p": merged[pub], "o": merged[our]}).dropna()
    d = b.o - b.p
    slope, intercept = np.polyfit(b.p, b.o, 1)
    print(f"{label:<40s} n={len(b):5d} r={np.corrcoef(b.p, b.o)[0, 1]:.4f} rho={stats.spearmanr(b.p, b.o).statistic:.4f} "
          f"slope={slope:.4f} int={intercept:+.3f} bias={d.mean():+.3f} medbias={d.median():+.3f} MAE={d.abs().mean():.3f} "
          f"RMSE={np.sqrt((d ** 2).mean()):.3f} LoA=[{d.mean() - 1.96 * d.std(ddof=0):+.2f},{d.mean() + 1.96 * d.std(ddof=0):+.2f}] "
          f"w1={(d.abs() <= 1).mean() * 100:.1f}% w2={(d.abs() <= 2).mean() * 100:.1f}% w5={(d.abs() <= 5).mean() * 100:.1f}%")


print("--- pairs in the saved report")
agree("losao2r", "rem_min", "losao2r vs rem_min")
agree("losao2nr", "nrem_min", "losao2nr vs nrem_min")
agree("sao2rem", "rem_mean", "sao2rem vs rem_mean")
agree("sao2nrem", "nrem_mean", "sao2nrem vs nrem_mean")
agree("pctlt90", "sleep_ct90_pct", "pctlt90 vs sleep_ct90_pct")
agree("rdi4p", "sleep_odi4", "rdi4p vs sleep_odi4")
print("--- extra pairs, computed here, not part of the pipeline's gates")
agree("pctlt85", "sleep_ct85_pct", "pctlt85 vs sleep_ct85_pct")
agree("pctlt80", "sleep_ct80_pct", "pctlt80 vs sleep_ct80_pct")
agree("pctlt75", "sleep_ct75_pct", "pctlt75 vs sleep_ct75_pct")
agree("rdi3p", "sleep_odi3", "rdi3p vs sleep_odi3")
agree("ahi_a0h4", "sleep_odi4", "ahi_a0h4 vs sleep_odi4")
agree("ahi_a0h3", "sleep_odi3", "ahi_a0h3 vs sleep_odi3")
merged["tst"] = merged[["timest1", "timest2", "timest34", "timerem"]].sum(axis=1, min_count=4)
merged["ours_min"] = merged.sleep_valid_hours * 60
agree("tst", "ours_min", "TST minutes")
print("rdi4p vs sleep_odi4 by severity (rdi4p bands), mean of ODI4 minus rdi4p:")
band = pd.cut(merged.rdi4p, [-1, 5, 15, 30, 1000], labels=["<=5", "5-15", "15-30", ">30"])
print(merged.assign(band=band, d=merged.sleep_odi4 - merged.rdi4p).groupby("band", observed=True).d.agg(["size", "mean", "median"]).round(3).to_string())
print("share where ODI4 above rdi4p", round((merged.sleep_odi4 > merged.rdi4p).mean() * 100, 2))
print("quality strata counts", pd.cut(merged.frac_valid, [-0.01, 0.85, 0.95, 0.99, 1.01]).value_counts().sort_index().to_dict())

section("6. SpO2 quantisation and the effective detection thresholds")
for pid in (200001, 200002, 203500, 205000):
    sig = np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy")
    mask = np.load(P + f"validity_mask/shhs1-{pid}.npy")
    v = sig[mask]
    u = np.unique(np.round(v, 6))
    steps = np.diff(u)
    d = np.abs(np.diff(v))
    print(pid, "samples", sig.size, "valid", int(mask.sum()), "dtype", sig.dtype, "unique", u.size, "min step", round(steps.min(), 6),
          "modal step", round(pd.Series(np.round(steps, 4)).mode().iloc[0], 4), "share of |diff| == 0", round((d == 0).mean(), 4))
print("100/256 =", 100 / 256)
for c in ("sleep_desat2_depth_min", "sleep_desat3_depth_min", "sleep_desat4_depth_min"):
    x = obm[c].dropna()
    print(c, "n", len(x), "mode", round(x.round(4).mode().iloc[0], 4), "median", round(x.median(), 4), "steps of 0.390625:", round(x.round(4).mode().iloc[0] / 0.390625, 3))
for c in ("sleep_desat3_duration_max", "sleep_desat3_duration_mean", "sleep_desat3_depth_max", "sleep_desat3_area_mean"):
    x = coh[c].dropna()
    print(c, "cohort: median", round(x.median(), 2), "q1", round(x.quantile(.25), 2), "q3", round(x.quantile(.75), 2), "max", round(x.max(), 2))
print("participants in 4,352 whose longest ODI3 event exceeds 120 s:", int((coh.sleep_desat3_duration_max > 120).sum()), "of", int(coh.sleep_desat3_duration_max.notna().sum()))
print("no ODI3 events in sleep (5,793):", int((obm.sleep_odi3_n_events == 0).sum()), " (4,352):", int((coh.sleep_odi3_n_events == 0).sum()))
print("REM window: rem_odi3 NaN (5,793):", int(obm.rem_odi3.isna().sum()), "(4,352):", int(coh.rem_odi3.isna().sum()),
      " rem_nrem ratio NaN (4,352):", int(coh.rem_nrem_odi3_ratio.isna().sum()))
print("complexity segment minutes (cohort) median", round(coh.sleep_complexity_segment_min.median(), 2), "min", coh.sleep_complexity_segment_min.min(),
      "max", coh.sleep_complexity_segment_min.max(), "any NaN", int(coh.sleep_sampen.isna().sum()))
print("share of complexity segment over valid sleep minutes, median", round((coh.sleep_complexity_segment_min / (coh.sleep_valid_hours * 60)).median(), 4))
print("sleep_n_segments (cohort) median", coh.sleep_n_segments.median(), "q1", coh.sleep_n_segments.quantile(.25), "q3", coh.sleep_n_segments.quantile(.75))
print("CTM vs fraction: sleep_ctm mean", round(coh.sleep_ctm.mean(), 4))

section("7. Cascade reproduction and the ID block claims")
shhs = pd.read_csv(D + "shhs1-dataset-0.15.0.csv", low_memory=False, usecols=["nsrrid", "age_s1", "gender", "race"])
print("SHHS-1 rows", len(shhs), "CVD rows", len(cvd), "obm rows", len(obm))
print("SHHS-1 IDs with no recording output:", sorted(set(shhs.nsrrid) - set(obm.pid)))
print("Recording IDs with no CVD row:", sorted(set(obm.pid) - set(cvd.nsrrid)))
insufficient = obm.loc[~obm.sufficient_valid_data.astype(bool), ["pid", "recording_valid_hours"]]
print(insufficient.to_string())
s = obm.loc[obm.sufficient_valid_data.astype(bool), ["pid"]].merge(cvd, left_on="pid", right_on="nsrrid")
print("after 4 h and joins", len(s), "any_cvd observed", int(s.any_cvd.notna().sum()), "missing", int(s.any_cvd.isna().sum()))
print("no-outcome IDs range", s.loc[s.any_cvd.isna(), "pid"].min(), s.loc[s.any_cvd.isna(), "pid"].max())
all_ids = obm[["pid"]].merge(modelling, on="pid", how="left").merge(shhs, left_on="pid", right_on="nsrrid", how="left")
all_ids["block"] = np.where(all_ids.pid <= 202600, "to_202600", "above_202600")
print("recordings by block", all_ids.block.value_counts().to_dict())
in_5036 = all_ids[all_ids.any_prevalent_cvd.notna()]
print("outcome observed by block", in_5036.block.value_counts().to_dict())
share = in_5036.groupby("block").any_prevalent_cvd.agg(["size", "sum", "mean"])
print(share.to_string())
print("baseline CVD share chi-square p", stats.chi2_contingency(pd.crosstab(in_5036.block, in_5036.any_prevalent_cvd).values, correction=False)[1])
inc = in_5036[in_5036.any_prevalent_cvd == 0].copy()
inc["age65"] = inc.age_s1 >= 65
g = inc.groupby("block")
print(g.agg(n=("pid", "size"), events=("cvd_incident", "sum"), rate=("cvd_incident", "mean"), age=("age_s1", "mean"), sd=("age_s1", "std"),
            age65=("age65", "mean"), male=("gender", lambda x: (x == 1).mean()), white=("race", lambda x: (x == 1).mean()),
            black=("race", lambda x: (x == 2).mean()), other=("race", lambda x: (x == 3).mean())).T.to_string())
to = inc[inc.block == "to_202600"]
ab = inc[inc.block == "above_202600"]
print("event rate chi-square p (no correction)", stats.chi2_contingency(pd.crosstab(inc.block, inc.cvd_incident).values, correction=False)[1])
print("event rate chi-square p (Yates)", stats.chi2_contingency(pd.crosstab(inc.block, inc.cvd_incident).values, correction=True)[1])
print("age Welch p", stats.ttest_ind(to.age_s1, ab.age_s1, equal_var=False).pvalue, " student p", stats.ttest_ind(to.age_s1, ab.age_s1).pvalue)
print("age>=65 chi-square p", stats.chi2_contingency(pd.crosstab(inc.block, inc.age65).values, correction=False)[1])
print("sex p", stats.chi2_contingency(pd.crosstab(inc.block, inc.gender).values, correction=False)[1])
print("race p", stats.chi2_contingency(pd.crosstab(inc.block, inc.race).values, correction=False)[1])
print("ages in the cohort: min", inc.age_s1.min(), "max", inc.age_s1.max(), "under 40:", int((inc.age_s1 < 40).sum()), "at 90:", int((inc.age_s1 >= 90).sum()),
      "median", inc.age_s1.median(), "q1", inc.age_s1.quantile(.25), "q3", inc.age_s1.quantile(.75))
print("ages in 5,793: min", all_ids.age_s1.min(), "max", all_ids.age_s1.max())
vc = inc.race.value_counts().sort_index()
print("race counts in cohort", vc.to_dict(), (vc / len(inc) * 100).round(2).to_dict())

section("8. Secondary outcome counts in the 4,352")
mod2 = pd.read_csv(A + "modelling_dataset.csv", usecols=lambda c: c.startswith("incident_") or c in ("pid", "any_prevalent_cvd"))
for c in manifest["incident_outcomes"]:
    x = mod2[c].dropna()
    print(c, int(x.sum()), "of", len(x), round(x.mean() * 100, 2))

# sections 9 to 15 cover the outcome, follow-up and how the detector behaves
import os
import shutil
import sys
import tempfile

full = pd.read_csv(A + "modelling_dataset.csv", usecols=["pid", "any_prevalent_cvd", "cvd_incident"])
extra_cols = ["nsrrid", "slptime", "SlpPrdP", "TimeBedP", "slp_time", "timest1", "timest2", "timest34", "timerem",
              "smokstat_s1", "ANGINA15", "MI15", "STROKE15", "HF15", "CABG15", "CA15", "HTNMED1", "HTNDerv_s1",
              "SystBP", "DiasBP", "bmi_s1", "age_s1", "gender", "race"]
sh = pd.read_csv(D + "shhs1-dataset-0.15.0.csv", low_memory=False, usecols=lambda c: c in extra_cols)
a = full.merge(cvd.drop(columns=["age_s1", "gender", "race"]), left_on="pid", right_on="nsrrid").merge(sh, on="nsrrid")
a["group"] = np.select([a.mi_death.notna(), a.prev_ang.notna() & a.mi_death.isna(), a.prev_revpro.notna() & a.mi_death.isna()],
                       ["G4", "G3", "G2"], "G1")
inc = a[a.any_prevalent_cvd == 0].copy()
print("\nchecks: analysed", len(inc), "events", int(inc.any_cvd.sum()))

section("9. What any_cvd contains in the 4,352")
comp = ["mi", "mip", "revasc_proc", "ptca", "cabg", "chf", "stroke", "cvd_death", "chd_death", "angina", "mi_fatal", "stk_fatal", "any_chd"]
for v in comp:
    x = inc[v].fillna(0) > 0
    print(f"{v:12s} n>0 {int(x.sum()):5d}  any_cvd=1 {int(inc.loc[x, 'any_cvd'].sum()):5d}  any_cvd=0 {int((inc.loc[x, 'any_cvd'] == 0).sum()):5d}")
others = (inc[["mi", "mip", "revasc_proc", "ptca", "cabg", "chf", "cvd_death", "chd_death"]].fillna(0) > 0).any(axis=1)
st = inc.stroke.fillna(0) > 0
print("any_cvd=1 with a non-stroke component", int(((inc.any_cvd == 1) & others).sum()))
print("any_cvd=1 with stroke only", int(((inc.any_cvd == 1) & ~others & st).sum()))
print("any_cvd=1 with neither (angina alone or nothing)", int(((inc.any_cvd == 1) & ~others & ~st).sum()))
print("any_cvd=0 with a non-stroke component", int(((inc.any_cvd == 0) & others).sum()))
print("any_cvd=0 with stroke only", int(((inc.any_cvd == 0) & ~others & st).sum()))
print(inc.loc[(inc.any_cvd == 0) & others, ["mi", "mip", "revasc_proc", "ptca", "cabg", "chf", "cvd_death", "chd_death"]].fillna(0).gt(0).sum().to_dict())
print(pd.crosstab([inc.group, st.rename("stroke"), others.rename("other_component")], inc.any_cvd).to_string())
print("date variables <= 0 in cohort:", {d: int((inc[d] <= 0).sum()) for d in ["mi_date", "mip_date", "stk_date", "chd_dthdt", "cvd_dthdt", "revpro_date", "ptca_date", "cabg_date", "chf_date", "ang_date"]})

section("10. Prevalent flags on the 5,036, and residual self-reported disease in the 4,352")
flags = ["prev_mi", "prev_mip", "prev_stk", "prev_chf", "prev_ang", "prev_revpro"]
pos = a[flags].fillna(0) > 0
print("positive", pos.sum().to_dict(), "missing", a[flags].isna().sum().to_dict(), "union", int(pos.any(axis=1).sum()))
print("removed by this flag alone", {f: int((pos[f] & (pos.sum(axis=1) == 1)).sum()) for f in flags})
print("union without prev_ang and prev_revpro", int(pos[["prev_mi", "prev_mip", "prev_stk", "prev_chf"]].any(axis=1).sum()))
print("missing in the 4,352", inc[flags].isna().sum().to_dict())
print("max value of each flag on 5,036", a[flags].max().to_dict())
print(a.groupby("group")[flags].agg(lambda x: x.notna().sum()).to_string())
print(a.groupby("group").any_prevalent_cvd.agg(["size", "sum", "mean"]).round(4).to_string())
print("any_cvd among the 684 prevalent", int(a[a.any_prevalent_cvd == 1].any_cvd.sum()))
sr = ["ANGINA15", "MI15", "STROKE15", "HF15", "CABG15", "CA15"]
for v in sr:
    print(v, inc[v].value_counts(dropna=False).sort_index().to_dict())
srp = inc[sr] == 1
any_sr = srp.any(axis=1)
hard = srp[["MI15", "STROKE15", "HF15", "CABG15", "CA15"]].any(axis=1)
print("any self report yes", int(any_sr.sum()), "events", int(inc.loc[any_sr, "any_cvd"].sum()), "rate", round(inc.loc[any_sr, "any_cvd"].mean(), 4), "rate without", round(inc.loc[~any_sr, "any_cvd"].mean(), 4))
print("excluding angina", int(hard.sum()), "events", int(inc.loc[hard, "any_cvd"].sum()), "rate", round(inc.loc[hard, "any_cvd"].mean(), 4))
print(pd.crosstab(inc.group, any_sr).to_string())

section("11. Follow-up in the 4,352, censdate in days from the baseline PSG")
yr = 365.25
print("all", inc.censdate.describe().round(2).to_dict(), "years median", round(inc.censdate.median() / yr, 3), "Q1", round(inc.censdate.quantile(.25) / yr, 3), "Q3", round(inc.censdate.quantile(.75) / yr, 3), "mean", round(inc.censdate.mean() / yr, 3), "sd", round(inc.censdate.std() / yr, 3), "max", round(inc.censdate.max() / yr, 3))
ev = inc[inc.any_cvd == 1].copy()
ne = inc[inc.any_cvd == 0].copy()
print("events", ev.censdate.describe().round(1).to_dict())
print("non-events", ne.censdate.describe().round(1).to_dict())
print("cohort under 1 y", int((inc.censdate < yr).sum()), "under 5 y", int((inc.censdate < 5 * yr).sum()), "at least 10 y", int((inc.censdate >= 10 * yr).sum()), round((inc.censdate >= 10 * yr).mean() * 100, 2))
print("non-events: censdate 0", int((ne.censdate == 0).sum()), "under 1 y", int((ne.censdate < yr).sum()), "under 5 y", int((ne.censdate < 5 * yr).sum()), "at least 10 y", int((ne.censdate >= 10 * yr).sum()), "died", int((ne.vital == 0).sum()), "alive", int((ne.vital == 1).sum()), "alive under 5 y", int(((ne.censdate < 5 * yr) & (ne.vital == 1)).sum()), "alive under 10 y", int(((ne.censdate < 10 * yr) & (ne.vital == 1)).sum()))
print("events: died", int((ev.vital == 0).sum()), "alive", int((ev.vital == 1).sum()))
dates = ["mi_date", "mip_date", "chd_dthdt", "cvd_dthdt", "revpro_date", "ptca_date", "cabg_date", "chf_date", "stk_date"]
ev["t"] = ev[dates].min(axis=1)
print("time to first component date, events", ev.t.describe().round(1).to_dict(), "years median", round(ev.t.median() / yr, 2), "Q1", round(ev.t.quantile(.25) / yr, 2), "Q3", round(ev.t.quantile(.75) / yr, 2))
print("events with a date", int(ev.t.notna().sum()), "first date after censdate", int((ev.t > ev.censdate).sum()))
print("events by time", {y: int((ev.t <= y * yr).sum()) for y in (1, 3, 5, 8, 10)}, "after 10 y", int((ev.t > 10 * yr).sum()))
print("first component among events", ev[dates].idxmin(axis=1).value_counts().to_dict())
print("non-events holding a component date", int((ne[dates].notna().any(axis=1)).sum()), ne[dates].notna().sum().to_dict())

section("12. Data availability groups in the analysed cohort")
g = inc.groupby("group").agg(n=("pid", "size"), events=("any_cvd", "sum"), rate=("any_cvd", "mean"), age=("age_s1", "mean"), amin=("age_s1", "min"), amax=("age_s1", "max"),
                             male=("gender", lambda x: (x == 1).mean()), white=("race", lambda x: (x == 1).mean()), black=("race", lambda x: (x == 2).mean()), other=("race", lambda x: (x == 3).mean()),
                             fu_med_y=("censdate", lambda x: x.median() / yr), fu_q1_y=("censdate", lambda x: x.quantile(.25) / yr), fu_q3_y=("censdate", lambda x: x.quantile(.75) / yr), fu_max_y=("censdate", lambda x: x.max() / yr),
                             alive=("vital", "mean"), idmin=("pid", "min"), idmax=("pid", "max"))
print(g.round(3).T.to_string())
print(a.groupby("group").agg(n=("pid", "size"), idmin=("pid", "min"), idmax=("pid", "max")).to_string())
print("angina>0 by group", inc.groupby("group").angina.apply(lambda x: round((x > 0).mean(), 3)).to_dict(), "ang_date present", inc.groupby("group").ang_date.apply(lambda x: int(x.notna().sum())).to_dict())
inc["block"] = np.where(inc.pid <= 202600, "to_202600", "above_202600")
print(pd.crosstab(inc.block, inc.group, margins=True).to_string())
no = cvd[cvd.any_cvd.isna()]
print("rows with no outcome in the CVD file", len(no), "censdate median days", no.censdate.median(), "dead", int((no.vital == 0).sum()))

section("13. Sleep time columns in the SHHS-1 file, all 5,804 rows")
sh["tst"] = sh[["timest1", "timest2", "timest34", "timerem"]].sum(axis=1, min_count=4)
print(sh[["slptime", "SlpPrdP", "TimeBedP", "slp_time", "tst"]].describe().round(2).T.to_string())
print(sh[["slptime", "SlpPrdP", "TimeBedP", "slp_time", "tst"]].corr().round(4).to_string())
rl = pd.read_csv(P + "reclean_log.csv")
print("reclean_log: valid_hours mean x60", round(rl.valid_hours.mean() * 60, 1), "total recording minutes mean", round((rl.n_samples / rl.fs / 60).mean(), 1), "frac_valid median", round(rl.frac_valid.median(), 4))

section("14. Hypertension rule and range oddities in the 4,352")
inc["and_rule"] = ((inc.SystBP > 140) & (inc.DiasBP > 90)) | (inc.HTNMED1 == 1)
inc["or_rule"] = (inc.SystBP > 140) | (inc.DiasBP > 90) | (inc.HTNMED1 == 1)
print("agree with SBP>140 AND DBP>90 or medication", round((inc.and_rule == (inc.HTNDerv_s1 == 1)).mean(), 4), " agree with OR rule", round((inc.or_rule == (inc.HTNDerv_s1 == 1)).mean(), 4), " SBP>140 yet not hypertensive", int(((inc.SystBP > 140) & (inc.HTNDerv_s1 == 0)).sum()))
for v in ["bmi_s1", "SystBP", "DiasBP"]:
    x = inc[v].dropna()
    print(v, "min", x.min(), "n", int((x == x.min()).sum()), "max", x.max(), "n", int((x == x.max()).sum()))
print("DBP under 40:", int((inc.DiasBP < 40).sum()), " SBP under 80:", int((inc.SystBP < 80).sum()), " smoking missing:", int(inc.smokstat_s1.isna().sum()))

section("15. SpO2 lattice and detector behaviour")
# import the detector from a temp copy with bytecode off, so nothing gets written into aim2/
tmp = tempfile.mkdtemp()
shutil.copy(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "aim2", "obm_library.py"), tmp)
sys.dont_write_bytecode = True
sys.path.insert(0, tmp)
from obm_library import build_stage_array, central_tendency_measure, contiguous_segments, detect_desaturations, window_mask  # noqa: E402


def level(k):
    # stored value for integer saturation k; samples sit on a 100/256 = 0.390625 point grid
    return np.ceil(k * 2.56) / 2.56


rng = np.random.default_rng(42)
sampled = rng.choice(obm.pid.values, 40, replace=False)
vals = np.concatenate([np.round(np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy")[np.load(P + f"validity_mask/shhs1-{pid}.npy")], 3) for pid in sampled])
levels = np.array([level(k) for k in range(50, 101)])
print("40 random participants, samples", vals.size, "share within 0.01 of ceil(k*2.56)/2.56 for an integer k:", round(float((np.min(np.abs(vals[:, None] - levels[None, :]), axis=1) < 0.01).mean()), 4))
print("level(k) - k for k=88..99:", {k: round(level(k) - k, 4) for k in range(88, 100)}, "mean over k=85..99:", round(np.mean([level(k) - k for k in range(85, 100)]), 4))
for threshold, fall in [(2, 2), (3, 3), (3, 4), (4, 4), (4, 5)]:
    passes = [k for k in range(90, 100) if level(k) - level(k - fall) >= threshold]
    depths = sorted({round(level(k) - level(k - fall), 4) for k in range(90, 100)})
    print(f"fall of {fall} integer points vs threshold {threshold}: depths {depths}, passes from baselines {passes} ({len(passes)} of 10)")

print("\nCTM against the share of triples with no change, longest sleep segment")
for pid in (200002, 203500, 205000):
    sig = np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy")
    val = np.load(P + f"validity_mask/shhs1-{pid}.npy")
    ev_table = pd.read_csv(P + f"events/shhs1-{pid}_events.csv")
    wm = window_mask(val, build_stage_array(ev_table, sig.size, 1.0), "sleep")
    s_all = contiguous_segments(wm, 1.0)
    i0, i1 = max(s_all, key=lambda t: t[1] - t[0])
    x = sig[i0:i1]
    print(pid, "ctm", round(central_tendency_measure(x), 4), "both differences exactly zero", round(float(np.mean((np.diff(x[:-1]) == 0) & (np.diff(x[1:]) == 0))), 4), "segment s", i1 - i0, "sleep segments", len(s_all))

print("\nSynthetic sawtooth, depth uniform 5 to 8 points, 3 h, quantised to 0.390625, threshold 4, 5 seeds")
for period in (30, 40, 60, 90, 120, 150, 200):
    res = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        pieces, total = [], 0
        while total < 3 * 3600:
            d = r.uniform(5, 8)
            fall = int(period * 0.4)
            pieces.append(np.concatenate([np.linspace(96, 96 - d, fall, endpoint=False), np.linspace(96 - d, 96, period - fall, endpoint=False)]))
            total += period
        trace = np.round(np.concatenate(pieces) / 0.390625) * 0.390625
        found = detect_desaturations(trace, 1.0, 4.0)
        res.append((len(pieces), len(found), np.mean([e["duration_sec"] for e in found])))
    res = np.array(res)
    print(f"cycle {period:4d} s: true {3600 / period:6.1f}/h detected {res[:, 1].mean() / 3:6.1f}/h ratio {res[:, 1].mean() / res[:, 0].mean():.3f} mean duration {res[:, 2].mean():.1f} s")

print("\nRerun of the detector on 300 random participants, sleep window only")
pub = pd.read_csv(A + "obm_vs_shhs_published.csv", usecols=["pid", "rdi4p", "rdi3p", "sleep_odi4", "sleep_odi3"])
rows = []
for _, row in pub.sample(300, random_state=7).iterrows():
    pid = int(row.pid)
    sig = np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy")
    val = np.load(P + f"validity_mask/shhs1-{pid}.npy")
    wm = window_mask(val, build_stage_array(pd.read_csv(P + f"events/shhs1-{pid}_events.csv"), sig.size, 1.0), "sleep")
    if wm.sum() == 0:
        continue
    pieces = [sig[i:j] for i, j in contiguous_segments(wm, 1.0)]
    hours = wm.sum() / 3600.0
    def rate(th):
        return sum(len(detect_desaturations(p, 1.0, th)) for p in pieces) / hours
    rows.append(dict(saved4=row.sleep_odi4, saved3=row.sleep_odi3, rdi4p=row.rdi4p, rdi3p=row.rdi3p, o4=rate(4.0), o39=rate(3.9), o3=rate(3.0), o27=rate(2.7)))
t = pd.DataFrame(rows)
print("n", len(t), "max abs difference from the saved values: ODI4", (t.o4 - t.saved4).abs().max(), "ODI3", (t.o3 - t.saved3).abs().max())
for ours, theirs, label in [("o4", "rdi4p", "threshold 4.0, as run, vs rdi4p"), ("o39", "rdi4p", "threshold 3.9, any 4 point fall, vs rdi4p"), ("o3", "rdi3p", "threshold 3.0, as run, vs rdi3p"), ("o27", "rdi3p", "threshold 2.7, any 3 point fall, vs rdi3p")]:
    slope, intercept = np.polyfit(t[theirs], t[ours], 1)
    print(f"{label:45s} r={np.corrcoef(t[ours], t[theirs])[0, 1]:.4f} slope={slope:.3f} intercept={intercept:.3f} bias={(t[ours] - t[theirs]).mean():+.3f} median bias={(t[ours] - t[theirs]).median():+.3f}")
print("means", t[["o4", "o39", "o3", "o27", "rdi4p", "rdi3p"]].mean().round(3).to_dict())
print("Spearman, threshold 4.0 against 3.9:", round(t.o4.corr(t.o39, method="spearman"), 4), " 3.0 against 2.7:", round(t.o3.corr(t.o27, method="spearman"), 4))
shutil.rmtree(tmp, ignore_errors=True)

section("16. Remaining small facts")
c2 = obm[obm.pid.isin(coh_ids)]
print("valid sleep under 4 h in the 4,352:", int((c2.sleep_valid_hours < 4).sum()), " under 2 h:", int((c2.sleep_valid_hours < 2).sum()), " min:", round(c2.sleep_valid_hours.min(), 3))
print("max |sleep_aod100_per_hour - 60*(100 - sleep_mean)|:", float((c2.sleep_aod100_per_hour - 60 * (100 - c2.sleep_mean)).abs().max()))
print("rem_valid_hours == 0 in 4,352:", int((c2.rem_valid_hours.fillna(0) == 0).sum()), " REM minutes 0 to 10:", int(((c2.rem_valid_hours > 0) & (c2.rem_valid_hours * 60 < 10)).sum()), " 0 to 30:", int(((c2.rem_valid_hours > 0) & (c2.rem_valid_hours * 60 < 30)).sum()), " median minutes:", round(c2.rem_valid_hours.median() * 60, 1))
print("nrem_odi3 == 0 in 4,352:", int((c2.nrem_odi3 == 0).sum()), " rem_odi3 at 120:", int((c2.rem_odi3 >= 120).sum()))
head = pd.read_csv(D + "shhs1-dataset-0.15.0.csv", nrows=0).columns.tolist()
print("date or site columns in the SHHS-1 file:", [c for c in head if c.lower() in ("choldt", "clinic", "site", "psgdate", "blpsgdate", "studydt", "parentstudy")])
print("modelling_dataset columns:", pd.read_csv(A + "modelling_dataset.csv", nrows=0).shape[1])
print("recording_n_segments in 4,352: median", c2.recording_n_segments.median(), " sleep_n_segments: median", c2.sleep_n_segments.median(), "Q1", c2.sleep_n_segments.quantile(.25), "Q3", c2.sleep_n_segments.quantile(.75))
target = pd.read_csv(A + "modelling_dataset.csv", usecols=["pid", "any_prevalent_cvd", "cvd_incident", "incident_any_cvd", "any_cvd"])
print("cvd_incident counts", target.cvd_incident.value_counts(dropna=False).to_dict(), " identical to incident_any_cvd:", bool(target.cvd_incident.fillna(-1).eq(target.incident_any_cvd.fillna(-1)).all()), " equals any_cvd where defined:", bool((target.loc[target.any_prevalent_cvd == 0, "cvd_incident"] == target.loc[target.any_prevalent_cvd == 0, "any_cvd"]).all()))
dummies = pd.read_csv(A + "modelling_dataset.csv", usecols=["pid", "any_prevalent_cvd", "smokstat_s1_1", "smokstat_s1_2"]).merge(sh[["nsrrid", "smokstat_s1"]], left_on="pid", right_on="nsrrid")
missing_smoke = dummies[(dummies.any_prevalent_cvd == 0) & dummies.smokstat_s1.isna()]
print("missing smoking in 4,352:", len(missing_smoke), " dummy sum:", int(missing_smoke[["smokstat_s1_1", "smokstat_s1_2"]].sum().sum()), " missing in either dummy column:", int(dummies[["smokstat_s1_1", "smokstat_s1_2"]].isna().sum().sum()))
