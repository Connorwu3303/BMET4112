"""
Builds the Aim 2 modelling dataset.

Joins the oximetry biomarkers (obm_features.csv) to the SHHS clinical
covariates and CVD outcomes, then applies the exclusions.

The outcome is incident CVD. any_cvd in the SHHS summary file also counts
disease that was already there at the sleep study, so anyone with CVD at
baseline is dropped and the target is a first event during follow-up. Leaving
them in would make this a diagnosis problem, and a confounded one, since
existing heart disease changes sleep and medication. The prevalent-inclusive
version is still written out for a sensitivity analysis.

Features are tagged into groups in feature_manifest.json so the model scripts
can fit confounders only, OBM only and OBM + confounders. Confounders only is
the baseline to beat, since age alone predicts CVD well.

Outputs (processed/aim2/): modelling_dataset.csv, feature_manifest.json,
dataset_summary.txt

Usage:
    python3 aim2/build_modelling_dataset.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# --- config ---

OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"
AIM2_DIR = os.path.join(OUTPUT_DIR, "aim2")
FEATURES_PATH = os.path.join(AIM2_DIR, "obm_features.csv")

DOC_DIR = "/Users/connorwu/Desktop/Thesis/documentation"
SHHS1_CSV = os.path.join(DOC_DIR, "shhs1-dataset-0.15.0.csv")
CVD_CSV = os.path.join(DOC_DIR, "shhs-cvd-summary-dataset-0.15.0.csv")

DATASET_PATH = os.path.join(AIM2_DIR, "modelling_dataset.csv")
MANIFEST_PATH = os.path.join(AIM2_DIR, "feature_manifest.json")
SUMMARY_PATH = os.path.join(AIM2_DIR, "dataset_summary.txt")

# hours of usable SpO2 needed before we trust someone's biomarkers
MIN_VALID_HOURS = 4.0

# Continuous confounders, straight from the SHHS-1 file (can't get these from
# the SpO2 signal).
CONTINUOUS_CONFOUNDERS = [
    "age_s1",     # age at baseline, the strongest CVD predictor
    "bmi_s1",     # body mass index, linked to both OSA and CVD
    "SystBP",     # systolic blood pressure
    "DiasBP",     # diastolic blood pressure
    "Chol",       # total cholesterol
    "HDL",        # high density lipoprotein
    "Trig",       # triglycerides
]

# medication and condition flags, already 0/1 in the SHHS file
BINARY_CONFOUNDERS = [
    "BETA1",       # beta blocker
    "ACE1",        # ACE inhibitor
    "CCB1",        # calcium channel blocker
    "DIURET1",     # diuretic
    "LIPID1",      # lipid lowering agent
    "HTNDerv_s1",  # derived hypertension status
]

# nominal variables and how each one gets encoded
CATEGORICAL_CONFOUNDERS = {
    # 1 = male, 2 = female. Two levels only need one indicator column.
    "gender": "binary_male",
    # 1 = white, 2 = black, 3 = other. No ordering, so one-hot.
    "race": "onehot",
    # 0 = never, 1 = current, 2 = former. Not ordered in risk, so one-hot.
    "smokstat_s1": "onehot",
}

# Prevalent disease flags. Anyone positive on any of these had CVD before the
# sleep study and is dropped from the incident analysis.
PREVALENT_FLAGS = [
    "prev_mi",       # prior myocardial infarction
    "prev_mip",      # prior MI including silent / procedural
    "prev_stk",      # prior stroke
    "prev_chf",      # prior congestive heart failure
    "prev_ang",      # prior angina
    "prev_revpro",   # prior revascularisation procedure
]

# outcomes carried through (any_cvd is the main one)
OUTCOMES = [
    "any_cvd",     # composite CVD, primary target
    "any_chd",     # coronary heart disease composite
    "mi",          # myocardial infarction
    "stroke",      # stroke
    "chf",         # congestive heart failure
    "cvd_death",   # cardiovascular death
    "afibincident",
]

PRIMARY_OUTCOME = "any_cvd"

# Feature pruning thresholds. These only look at the feature matrix, never the
# outcome, so it's fine to apply them to the whole dataset before splitting.
MIN_NONMISSING_FRACTION = 0.80   # drop a feature missing in over 20% of rows
MIN_UNIQUE_VALUES = 5            # drop near-constant features
# Was 0.98, which kept 640 features against 423 events (0.64 events per
# variable). The four analysis windows are mostly repeat views of the same
# signal, so 0.95 mainly removes duplicates.
CORRELATION_THRESHOLD = 0.95

# Core biomarker panel. The full biomarker set is several hundred features
# against a few hundred events. That's fine for a data-driven model, but it
# can't give stable coefficients for the main interpretable result, however
# hard it's penalised.
#
# So this is a pre-specified subset: one representative per biomarker family,
# limited to measures that are clinically established or checked against a
# published SHHS variable in validate_obm.py. It's picked from the literature,
# so it doesn't peek at the outcome.
#
# With the 18 confounders this gives roughly 8 to 9 events per variable, close
# to the usual minimum for a stable logistic fit. The full set is kept as a
# separate specification for a sensitivity analysis.
CORE_OBM_PANEL = [
    # Hypoxic burden, graded. ct90_pct is T90, validated at r = 1.000 against
    # the published pctlt90.
    "sleep_ct90_pct", "sleep_ct85_pct", "sleep_ct80_pct",
    "sleep_ca90", "sleep_ihb", "sleep_pod", "sleep_aod100_per_hour",
    # General statistics. sleep_min is the overnight nadir.
    "sleep_mean", "sleep_std", "sleep_min", "sleep_p5",
    # Desaturation family, at both AASM scoring depths.
    "sleep_odi3", "sleep_odi4",
    "sleep_desat3_depth_mean", "sleep_desat3_duration_mean",
    "sleep_desat3_area_mean", "sleep_desat3_area_per_hour",
    "sleep_desat3_slope_desat_mean",
    # Short-term variability.
    "sleep_delta_index", "sleep_zero_crossings_per_hour",
    # Complexity.
    "sleep_sampen", "sleep_lz_complexity", "sleep_ctm", "sleep_dfa_alpha",
    # Periodicity, the spectral analogue of the apnoea-hypopnoea index.
    "sleep_psd_ratio", "sleep_psd_band",
    # Stage stratified. The four means and nadirs are the ones checked
    # directly against losao2r, losao2nr, sao2rem and sao2nrem.
    "rem_min", "nrem_min", "rem_mean", "nrem_mean",
    "rem_odi3", "nrem_odi3",
    "rem_minus_nrem_nadir", "rem_nrem_odi3_ratio",
]


def prune_features(frame, columns, summary):
    """
    Drops degenerate and near-duplicate feature columns.

    Three passes, each logged to the summary so the thesis can say what went:
      1. features missing in too many participants to impute sensibly
      2. near-constant features (no signal, and they upset a regularised fit)
      3. one of each near-perfectly correlated pair, since collinear pairs make
         the logistic regression coefficients hard to read

    Returns the list of columns that survive.
    """
    kept = list(columns)

    # pass 1: missingness
    nonmissing = frame[kept].notna().mean()
    too_sparse = nonmissing[nonmissing < MIN_NONMISSING_FRACTION].index.tolist()
    kept = [c for c in kept if c not in too_sparse]
    summary.append(
        f"  dropped {len(too_sparse)} features missing in more than "
        f"{(1 - MIN_NONMISSING_FRACTION):.0%} of participants"
    )

    # pass 2: near-constant
    low_variation = [
        c for c in kept if frame[c].nunique(dropna=True) < MIN_UNIQUE_VALUES
    ]
    kept = [c for c in kept if c not in low_variation]
    summary.append(
        f"  dropped {len(low_variation)} features with fewer than "
        f"{MIN_UNIQUE_VALUES} distinct values"
    )

    # pass 3: near-duplicates. For any pair above the threshold drop the later
    # column, so the earlier one in column order stays as the representative.
    numeric = frame[kept].select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        corr = numeric.corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape, dtype=bool), k=1)
        )
        redundant = [
            c for c in upper.columns
            if (upper[c] > CORRELATION_THRESHOLD).any()
        ]
        kept = [c for c in kept if c not in redundant]
        summary.append(
            f"  dropped {len(redundant)} features correlated above "
            f"{CORRELATION_THRESHOLD} with an earlier feature"
        )

    summary.append(f"  retained {len(kept)} features")
    return kept


def main():
    summary = []

    def emit(text=""):
        print(text)
        summary.append(text)

    emit("=" * 78)
    emit("BUILDING THE AIM 2 MODELLING DATASET")
    emit("=" * 78)

    if not os.path.exists(FEATURES_PATH):
        emit(f"\nFATAL: {FEATURES_PATH} not found. Run extract_obm_features.py.")
        return 1

    # --- load ---
    obm = pd.read_csv(FEATURES_PATH)
    emit(f"\nOBM features        : {obm.shape[0]} participants, "
         f"{obm.shape[1]} columns")

    shhs1_wanted = set(
        ["nsrrid"]
        + CONTINUOUS_CONFOUNDERS
        + BINARY_CONFOUNDERS
        + list(CATEGORICAL_CONFOUNDERS)
    )
    shhs1 = pd.read_csv(
        SHHS1_CSV, low_memory=False, usecols=lambda c: c in shhs1_wanted
    )
    emit(f"SHHS-1 clinical     : {shhs1.shape[0]} rows, "
         f"{shhs1.shape[1]} columns")

    cvd = pd.read_csv(CVD_CSV, low_memory=False)
    emit(f"CVD summary         : {cvd.shape[0]} rows")

    # --- exclusion cascade (n printed at each step so the final n adds up) ---
    emit("\n" + "=" * 78)
    emit("EXCLUSION CASCADE")
    emit("=" * 78)

    n0 = len(obm)
    emit(f"\n  participants with computed biomarkers        : {n0}")

    # 1. enough usable oximetry
    data = obm[obm["sufficient_valid_data"].astype(bool)].copy()
    emit(f"  after requiring >= {MIN_VALID_HOURS} h analysable SpO2   : "
         f"{len(data)}  (-{n0 - len(data)})")

    # No scored REM means no REM biomarkers. We keep these people anyway
    # (dropping them would bias towards better sleepers) and let the REM
    # columns be imputed later.
    n_no_rem = int((data.get("rem_valid_hours", pd.Series(dtype=float))
                    .fillna(0) == 0).sum())
    emit(f"    (of these, {n_no_rem} have no scored REM, so REM-specific")
    emit(f"     biomarkers are missing and will be imputed, not excluded)")

    # 2. join clinical
    n_before = len(data)
    data = data.merge(shhs1, left_on="pid", right_on="nsrrid", how="inner")
    emit(f"  after joining SHHS-1 clinical variables      : {len(data)}  "
         f"(-{n_before - len(data)})")

    # 3. join outcomes
    n_before = len(data)
    # afibprevalent is pulled in separately because it's not part of the CVD
    # exclusion. It only gates the AF outcome (prevalent AF has to go for an
    # incident AF analysis).
    cvd_cols = ["nsrrid"] + [c for c in PREVALENT_FLAGS + OUTCOMES + ["afibprevalent"]
                             if c in cvd.columns]
    data = data.merge(
        cvd[cvd_cols], on="nsrrid", how="inner", suffixes=("", "_cvd")
    )
    emit(f"  after joining CVD outcomes                   : {len(data)}  "
         f"(-{n_before - len(data)})")

    # 4. primary outcome must be observed
    n_before = len(data)
    data = data[data[PRIMARY_OUTCOME].notna()].copy()
    emit(f"  after requiring {PRIMARY_OUTCOME} to be observed      : "
         f"{len(data)}  (-{n_before - len(data)})")

    # 5. prevalent disease exclusion (the one that matters most)
    available_flags = [f for f in PREVALENT_FLAGS if f in data.columns]
    # A missing prevalent flag counts as absent. prev_ang and prev_revpro are
    # missing for most of the cohort, so treating missing as prevalent would
    # throw out nearly everyone. This is the usual handling for these SHHS
    # variables.
    #
    # Careful: these flags are event counts. prev_mi goes up to 22 (22
    # adjudicated prior MIs). Testing == 1 would only catch people with
    # exactly one prior event and let everyone with two or more into the
    # incident cohort, i.e. the sickest people. So binarise with > 0.
    data["any_prevalent_cvd"] = (
        (data[available_flags].fillna(0) > 0).any(axis=1).astype(int)
    )
    n_prevalent = int(data["any_prevalent_cvd"].sum())

    # Some outcomes are counts too, so binarise them to "at least one event in
    # follow-up". any_cvd and any_chd are already 0/1, but mi, stroke and chf
    # are counts and would otherwise reach the classifier as multi-class
    # labels.
    for outcome in OUTCOMES:
        if outcome not in data.columns:
            continue
        values = data[outcome].dropna().unique()
        if not set(values).issubset({0.0, 1.0}):
            data[outcome] = np.where(
                data[outcome].isna(), np.nan, (data[outcome] > 0).astype(float)
            )
            emit(f"  binarised count-coded outcome '{outcome}' to "
                 f"at-least-one-event")

    data["cvd_incident"] = np.where(
        data["any_prevalent_cvd"] == 1,
        np.nan,                       # undefined for prevalent cases
        data[PRIMARY_OUTCOME],
    )

    # Make an incident version of every outcome, so a secondary analysis can't
    # end up on the prevalent-inclusive column by accident. Otherwise
    # --outcome chf would quietly include the 398 people who already had CVD
    # at baseline.
    #
    # All of them use the same composite exclusion as the primary outcome, so
    # every outcome has the same denominator and results compare directly.
    # That's stricter than only excluding the matching condition (e.g. only
    # prevalent heart failure for incident heart failure), but consistency
    # across outcomes matters more here than the extra sample size.
    incident_outcomes = []
    for outcome in OUTCOMES:
        if outcome not in data.columns:
            continue
        name = f"incident_{outcome}"
        data[name] = np.where(
            data["any_prevalent_cvd"] == 1, np.nan, data[outcome]
        )
        # AF needs an extra exclusion: AF at baseline can't be incident AF,
        # and that's tracked by its own flag (afibprevalent).
        if outcome == "afibincident" and "afibprevalent" in data.columns:
            data[name] = np.where(
                data["afibprevalent"].fillna(0) > 0, np.nan, data[name]
            )
        incident_outcomes.append(name)

    emit("\n  Incident versions derived for every outcome:")
    for name in incident_outcomes:
        observed = data[name].notna()
        n_obs = int(observed.sum())
        n_pos = int(data.loc[observed, name].sum())
        rate = n_pos / n_obs if n_obs else 0.0
        emit(f"    {name:<26s} {n_pos:>5d} events of {n_obs:>5d} observed "
             f"({rate:>6.2%})")

    emit(f"\n  prevalent CVD at baseline (excluded from the")
    emit(f"  incident analysis)                           : {n_prevalent}")
    emit(f"  analysis cohort for incident CVD             : "
         f"{len(data) - n_prevalent}")

    incident = data[data["any_prevalent_cvd"] == 0]
    emit(f"\n  incident {PRIMARY_OUTCOME} events                     : "
         f"{int(incident[PRIMARY_OUTCOME].sum())} "
         f"({incident[PRIMARY_OUTCOME].mean():.1%} of the cohort)")

    emit("\n  Event counts on the incident cohort:")
    for outcome in OUTCOMES:
        if outcome in incident.columns:
            observed = incident[outcome].dropna()
            if len(observed):
                emit(f"    {outcome:<16s} {int(observed.sum()):>5d} positive "
                     f"of {len(observed):>5d} observed "
                     f"({observed.mean():>6.2%})")

    # --- encode the confounders ---
    emit("\n" + "=" * 78)
    emit("CONFOUNDER ENCODING")
    emit("=" * 78)

    confounder_cols = []

    for col in CONTINUOUS_CONFOUNDERS:
        if col in data.columns:
            confounder_cols.append(col)
    emit(f"\n  continuous : {len(confounder_cols)} carried through as-is")

    for col in BINARY_CONFOUNDERS:
        if col in data.columns:
            confounder_cols.append(col)
    emit(f"  binary     : {len(BINARY_CONFOUNDERS)} medication / condition flags")

    for col, scheme in CATEGORICAL_CONFOUNDERS.items():
        if col not in data.columns:
            continue
        if scheme == "binary_male":
            # SHHS codes 1 = male, 2 = female
            data["is_male"] = (data[col] == 1).astype(int)
            confounder_cols.append("is_male")
            emit(f"  {col:<12s} -> is_male indicator")
        elif scheme == "onehot":
            dummies = pd.get_dummies(
                data[col].astype("Int64"), prefix=col, dummy_na=False
            )
            # first level is the reference category (avoids the dummy
            # variable trap in an unregularised linear model)
            if dummies.shape[1] > 1:
                reference = dummies.columns[0]
                dummies = dummies.drop(columns=[reference])
                emit(f"  {col:<12s} -> {list(dummies.columns)} "
                     f"(reference: {reference})")
            for dummy_col in dummies.columns:
                data[dummy_col] = dummies[dummy_col].astype(int)
            confounder_cols.extend(dummies.columns.tolist())

    emit(f"\n  total confounder columns: {len(confounder_cols)}")
    emit("\n  Missingness in the confounders (imputed inside the CV pipeline,")
    emit("  never here, so imputation statistics stay inside the train fold):")
    for col in confounder_cols:
        frac = data[col].isna().mean()
        if frac > 0:
            emit(f"    {col:<20s} {frac:.1%} missing")

    # --- pick out and prune the OBM feature columns ---
    emit("\n" + "=" * 78)
    emit("OBM FEATURE PRUNING")
    emit("=" * 78)

    housekeeping = {
        "pid", "nsrrid", "sufficient_valid_data", "any_prevalent_cvd",
        "cvd_incident",
    }
    obm_candidates = [
        c for c in obm.columns
        if c not in housekeeping
        and c not in OUTCOMES
        and pd.api.types.is_numeric_dtype(data[c])
    ]
    # sample counts only describe how long the recording was, and a model
    # could latch onto that
    obm_candidates = [
        c for c in obm_candidates
        if not c.endswith("_n_valid_samples")
    ]
    emit(f"\n  candidate OBM features: {len(obm_candidates)}")
    emit("  (dropped the raw sample-count columns, which describe recording")
    emit("   length rather than physiology)")

    obm_kept = prune_features(data, obm_candidates, summary)
    print("\n".join(summary[-4:]))

    # group what's left by analysis window
    groups = {
        "obm_recording": [c for c in obm_kept if c.startswith("recording_")],
        "obm_sleep": [c for c in obm_kept if c.startswith("sleep_")],
        "obm_nrem": [c for c in obm_kept if c.startswith("nrem_")],
        "obm_rem": [c for c in obm_kept if c.startswith("rem_")],
    }
    grouped = set().union(*groups.values()) if groups else set()
    groups["obm_cross"] = [c for c in obm_kept if c not in grouped]

    emit("\n  surviving OBM features by analysis window:")
    for name, cols in groups.items():
        emit(f"    {name:<16s} {len(cols):>4d}")

    # The core panel comes from the full candidate list, before pruning. It's
    # pre-specified from the literature, so the correlation rule shouldn't thin
    # it. Only features that are missing entirely or too sparse get dropped.
    core_available = [c for c in CORE_OBM_PANEL if c in data.columns]
    core_missing = [c for c in CORE_OBM_PANEL if c not in data.columns]
    core_nonmissing = data[core_available].notna().mean()
    core_panel = [
        c for c in core_available
        if core_nonmissing[c] >= MIN_NONMISSING_FRACTION
    ]
    core_dropped_sparse = [c for c in core_available if c not in core_panel]

    emit(f"\n  core pre-specified panel:")
    emit(f"    requested            : {len(CORE_OBM_PANEL)}")
    if core_missing:
        emit(f"    absent from features : {len(core_missing)} -> {core_missing}")
    if core_dropped_sparse:
        emit(f"    dropped, too sparse  : {len(core_dropped_sparse)} -> "
             f"{core_dropped_sparse}")
    emit(f"    retained             : {len(core_panel)}")
    emit(f"    NOT correlation-pruned, because this panel is pre-specified")
    emit(f"    from the literature rather than selected from the data.")

    # --- write ---
    keep_cols = (
        ["pid", "sufficient_valid_data", "any_prevalent_cvd", "cvd_incident"]
        + incident_outcomes
        + [c for c in OUTCOMES if c in data.columns]
        + confounder_cols
        + obm_kept
        + core_panel
    )
    keep_cols = list(dict.fromkeys(keep_cols))
    dataset = data[keep_cols].copy()
    dataset.to_csv(DATASET_PATH, index=False)

    manifest = {
        "primary_outcome": PRIMARY_OUTCOME,
        "incident_outcome_column": "cvd_incident",
        "prevalent_exclusion_column": "any_prevalent_cvd",
        "n_participants_total": int(len(dataset)),
        "n_participants_incident_cohort": int(
            (dataset["any_prevalent_cvd"] == 0).sum()
        ),
        "min_valid_hours": MIN_VALID_HOURS,
        "outcomes": [c for c in OUTCOMES if c in dataset.columns],
        "incident_outcomes": [c for c in incident_outcomes
                              if c in dataset.columns],
        "confounders": confounder_cols,
        "obm_features": obm_kept,
        "obm_groups": groups,
        "pruning": {
            "min_nonmissing_fraction": MIN_NONMISSING_FRACTION,
            "min_unique_values": MIN_UNIQUE_VALUES,
            "correlation_threshold": CORRELATION_THRESHOLD,
        },
        "core_obm_panel": core_panel,
        "model_specifications": {
            # reference arm: clinical risk factors on their own, the
            # baseline any oximetry result is compared to
            "confounders_only": confounder_cols,
            # primary arms, core panel (roughly 8 to 9 events per variable)
            "obm_only": core_panel,
            "obm_plus_confounders": core_panel + confounder_cols,
            # sensitivity arms, full biomarker set. Heavily over-parameterised,
            # so the coefficients mean little, but they show whether a
            # data-driven fit finds anything the core panel missed.
            "obm_full": obm_kept,
            "obm_full_plus_confounders": obm_kept + confounder_cols,
        },
    }
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2)

    emit("\n" + "=" * 78)
    emit("OUTPUT")
    emit("=" * 78)
    emit(f"\n  dataset  : {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    emit(f"             {DATASET_PATH}")
    emit(f"  manifest : {MANIFEST_PATH}")
    emit(f"\n  model specifications available:")
    for name, cols in manifest["model_specifications"].items():
        n_events_spec = int(dataset.loc[dataset["any_prevalent_cvd"] == 0,
                                        PRIMARY_OUTCOME].sum())
        epv_spec = n_events_spec / len(cols) if cols else 0
        role = ("reference" if name == "confounders_only"
                else "SENSITIVITY" if "full" in name else "PRIMARY")
        emit(f"    {name:<28s} {len(cols):>4d} features, "
             f"{epv_spec:>5.2f} events/variable  [{role}]")

    # events per variable, a rough guide to how much regularisation the
    # logistic regression will need
    n_events = int(dataset.loc[dataset["any_prevalent_cvd"] == 0,
                               PRIMARY_OUTCOME].sum())
    n_primary = len(core_panel) + len(confounder_cols)
    epv_primary = n_events / n_primary if n_primary else 0
    n_full = len(obm_kept) + len(confounder_cols)
    epv_full = n_events / n_full if n_full else 0
    emit(f"\n  incident events                        : {n_events}")
    emit(f"  events per variable, PRIMARY spec      : {epv_primary:.2f} "
         f"({n_primary} features)")
    emit(f"  events per variable, SENSITIVITY spec  : {epv_full:.2f} "
         f"({n_full} features)")
    if epv_primary >= 8:
        emit("\n  The primary specification is adequately powered by the")
        emit("  conventional standard of roughly 10 events per variable, so its")
        emit("  coefficients can be interpreted as effect estimates.")
    else:
        emit("\n  The primary specification is below the conventional 10 events")
        emit("  per variable, so its coefficients should be read as regularised")
        emit("  and shrunken rather than as unbiased effect estimates.")
    emit("  The sensitivity specification is heavily over-parameterised by")
    emit("  design. It exists to test whether an unconstrained, data-driven")
    emit("  search finds signal the pre-specified panel missed. Its individual")
    emit("  coefficients are not interpretable; only its discrimination is.")

    with open(SUMMARY_PATH, "w") as handle:
        handle.write("\n".join(summary) + "\n")
    print(f"\n  summary  : {SUMMARY_PATH}")
    print("\nNext: python3 aim2/train_logistic_regression.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
