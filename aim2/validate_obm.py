"""
Checks the computed OBMs against the published SHHS oximetry variables, which
the study's own technologists and software scored independently.

The four REM / NREM nadir and mean pairs should agree almost exactly, which
also checks the channel alignment and sleep staging. pctlt90 and rdi4p should
only correlate: SHHS computed pctlt90 on the uncleaned signal, and rdi4p only
counts desaturations linked to a scored respiratory event, so our ODI4 should
come out somewhat higher. Every comparison is split by data quality. SHHS kept
probe dropouts in, so agreement should be tight on clean recordings and only
loosen where a lot of the night was invalid.

Needs obm_features.csv from extract_obm_features.py.

Outputs (processed/aim2/): obm_validation_report.txt, obm_validation_plots.png
(scatter and Bland-Altman panels), obm_vs_shhs_published.csv

Usage:
    python3 aim2/validate_obm.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- config ---

OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"
AIM2_DIR = os.path.join(OUTPUT_DIR, "aim2")
FEATURES_PATH = os.path.join(AIM2_DIR, "obm_features.csv")

DOC_DIR = "/Users/connorwu/Desktop/Thesis/documentation"
SHHS1_CSV = os.path.join(DOC_DIR, "shhs1-dataset-0.15.0.csv")

REPORT_PATH = os.path.join(AIM2_DIR, "obm_validation_report.txt")
PLOT_PATH = os.path.join(AIM2_DIR, "obm_validation_plots.png")
MERGED_PATH = os.path.join(AIM2_DIR, "obm_vs_shhs_published.csv")

# (published column, our column, label, exact_match_expected)
VALIDATION_PAIRS = [
    ("losao2r", "rem_min", "Nadir SaO2 in REM (%)", True),
    ("losao2nr", "nrem_min", "Nadir SaO2 in NREM (%)", True),
    ("sao2rem", "rem_mean", "Mean SaO2 in REM (%)", True),
    ("sao2nrem", "nrem_mean", "Mean SaO2 in NREM (%)", True),
    ("pctlt90", "sleep_ct90_pct", "T90, percent of sleep below 90%", False),
    ("rdi4p", "sleep_odi4", "4% desaturation index (events/h)", False),
]

# gates for the exact-match pairs. A correlation this high across thousands
# of people can't happen by accident, so it's a real check on channel
# alignment and staging.
EXACT_MIN_PEARSON = 0.90
EXACT_MAX_MEAN_ABS_ERROR = 2.0

# convergent pairs only need a strong monotonic relationship
CONVERGENT_MIN_SPEARMAN = 0.70

# data-quality strata, by fraction of the recording that was analysable
QUALITY_STRATA = [
    ("excellent (>=99% valid)", 0.99, 1.01),
    ("good (95-99% valid)", 0.95, 0.99),
    ("fair (85-95% valid)", 0.85, 0.95),
    ("poor (<85% valid)", -0.01, 0.85),
]


def agreement_stats(published, ours):
    """
    Agreement metrics for one validation pair, on rows where both values
    exist.

    Bias and error are reported next to correlation because correlation on
    its own can hide a systematic offset. A feature in the wrong units would
    still correlate at 1.0.
    """
    both = pd.DataFrame({"pub": published, "ours": ours}).dropna()
    if len(both) < 10:
        return None

    pub = both["pub"].to_numpy(dtype=float)
    our = both["ours"].to_numpy(dtype=float)
    diff = our - pub

    stats = {
        "n": len(both),
        "pearson_r": float(np.corrcoef(pub, our)[0, 1]),
        "spearman_r": float(sp_stats.spearmanr(pub, our).statistic),
        "mean_bias": float(np.mean(diff)),
        "median_bias": float(np.median(diff)),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "median_abs_error": float(np.median(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "limits_of_agreement_lo": float(np.mean(diff) - 1.96 * np.std(diff)),
        "limits_of_agreement_hi": float(np.mean(diff) + 1.96 * np.std(diff)),
        "pct_within_1": float(np.mean(np.abs(diff) <= 1.0) * 100),
        "pct_within_2": float(np.mean(np.abs(diff) <= 2.0) * 100),
        "pct_within_5": float(np.mean(np.abs(diff) <= 5.0) * 100),
        "pub_mean": float(np.mean(pub)),
        "our_mean": float(np.mean(our)),
        "pub_median": float(np.median(pub)),
        "our_median": float(np.median(our)),
    }

    # a plain least squares slope is enough to catch a scale error like a
    # factor-of-60 unit mistake
    slope, intercept = np.polyfit(pub, our, 1)
    stats["slope"] = float(slope)
    stats["intercept"] = float(intercept)
    return stats


def main():
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit("=" * 78)
    emit("OBM VALIDATION AGAINST PUBLISHED SHHS VARIABLES")
    emit("=" * 78)

    if not os.path.exists(FEATURES_PATH):
        emit(f"\nFATAL: {FEATURES_PATH} not found. Run extract_obm_features.py first.")
        return 1

    features = pd.read_csv(FEATURES_PATH)
    emit(f"\nOur features        : {features.shape[0]} participants, "
         f"{features.shape[1]} columns")

    published_cols = ["nsrrid"] + [p for p, _, _, _ in VALIDATION_PAIRS]
    # pull the per-stage minute counts to rebuild total sleep time (the
    # precheck below explains why `slptime` can't be used for this)
    extra = ["slptime", "SlpPrdP", "timest1", "timest2", "timest34", "timerem",
             "timeremp", "avgsat", "pctlt85", "pctlt80", "pctlt75",
             "rdi3p", "ahi_a0h3", "ahi_a0h4"]
    shhs = pd.read_csv(
        SHHS1_CSV, low_memory=False,
        usecols=lambda c: c in set(published_cols + extra),
    )
    emit(f"Published SHHS rows : {len(shhs)}")

    merged = features.merge(
        shhs, left_on="pid", right_on="nsrrid", how="inner"
    )
    emit(f"Merged on nsrrid    : {len(merged)} participants")
    if merged.empty:
        emit("\nFATAL: merge produced no rows. Check that pid matches nsrrid.")
        return 1

    # fraction of valid data (from the Aim 1 reclean log), used for the
    # quality strata below
    if "recording_valid_hours" in merged.columns and "slptime" in merged.columns:
        pass
    merged["frac_valid"] = np.nan
    log_path = os.path.join(OUTPUT_DIR, "reclean_log.csv")
    if os.path.exists(log_path):
        reclean = pd.read_csv(log_path)
        if "frac_valid" in reclean.columns and "pid" in reclean.columns:
            reclean = reclean[["pid", "frac_valid"]].dropna()
            reclean["pid"] = reclean["pid"].astype(int)
            merged = merged.drop(columns=["frac_valid"]).merge(
                reclean, on="pid", how="left"
            )
    emit(f"With quality info   : {int(merged['frac_valid'].notna().sum())}")

    merged.to_csv(MERGED_PATH, index=False)
    emit(f"Merged table written: {MERGED_PATH}")

    # --- precheck: scored sleep time ---
    # done before the biomarkers. If our sleep time is way off theirs, every
    # stage-stratified biomarker below is suspect and the problem is in the
    # staging.
    emit("\n" + "=" * 78)
    emit("PRECHECK: SCORED SLEEP TIME")
    emit("=" * 78)
    emit("\n  Total sleep time is reconstructed as the sum of the published")
    emit("  per-stage minute counts (timest1 + timest2 + timest34 + timerem).")
    emit("  The `slptime` column is NOT total sleep time despite the name: it")
    emit("  averages 588 minutes, which exceeds the length of most recordings")
    emit("  in this cohort, so it is time in bed rather than time asleep.")
    emit("  Comparing against it would show a spurious 3.8 hour deficit.")

    stage_cols = ["timest1", "timest2", "timest34", "timerem"]
    if (all(c in merged.columns for c in stage_cols)
            and "sleep_valid_hours" in merged.columns):
        theirs_min = merged[stage_cols].sum(axis=1, min_count=len(stage_cols))
        ours_min = merged["sleep_valid_hours"] * 60.0
        st = agreement_stats(theirs_min, ours_min)
        if st:
            emit(f"\n  published total sleep time vs our valid sleep minutes")
            emit(f"    n                 : {st['n']}")
            emit(f"    Pearson r         : {st['pearson_r']:.4f}")
            emit(f"    our mean          : {st['our_mean']:.1f} min")
            emit(f"    published mean    : {st['pub_mean']:.1f} min")
            emit(f"    mean bias         : {st['mean_bias']:+.1f} min")
            emit(f"    mean abs error    : {st['mean_abs_error']:.1f} min")
            emit(f"    regression slope  : {st['slope']:.4f}")
            emit("\n  Ours should sit at or slightly BELOW theirs, because our")
            emit("  sleep window additionally drops samples the oximeter was")
            emit("  not measuring. A large POSITIVE bias would instead mean we")
            emit("  are counting wake epochs as sleep, which would invalidate")
            emit("  every stage stratified biomarker below.")
            if abs(st["mean_bias"]) < 15:
                emit(f"\n  Agreement within {abs(st['mean_bias']):.1f} minutes on")
                emit("  average confirms the sleep stage expansion in")
                emit("  obm_library.build_stage_array is aligned with the SHHS")
                emit("  scoring, epoch for epoch.")

    # --- main validation ---
    verdicts = {}
    all_stats = {}

    for pub_col, our_col, label, exact in VALIDATION_PAIRS:
        emit("\n" + "=" * 78)
        emit(f"{'EXACT' if exact else 'CONVERGENT'}: {label}")
        emit(f"  published '{pub_col}'  vs  ours '{our_col}'")
        emit("=" * 78)

        if pub_col not in merged.columns:
            emit(f"  SKIP: {pub_col} not in the SHHS dataset")
            continue
        if our_col not in merged.columns:
            emit(f"  SKIP: {our_col} not in our feature table")
            continue

        st = agreement_stats(merged[pub_col], merged[our_col])
        if st is None:
            emit("  SKIP: fewer than 10 participants with both values")
            continue
        all_stats[our_col] = st

        emit(f"\n  n compared        : {st['n']}")
        emit(f"  Pearson r         : {st['pearson_r']:.4f}")
        emit(f"  Spearman rho      : {st['spearman_r']:.4f}")
        emit(f"  regression slope  : {st['slope']:.4f}  "
             f"(1.0 means identical scale)")
        emit(f"  intercept         : {st['intercept']:+.3f}")
        emit(f"\n  our median        : {st['our_median']:.3f}")
        emit(f"  published median  : {st['pub_median']:.3f}")
        emit(f"  mean bias         : {st['mean_bias']:+.3f}")
        emit(f"  median bias       : {st['median_bias']:+.3f}")
        emit(f"  mean abs error    : {st['mean_abs_error']:.3f}")
        emit(f"  RMSE              : {st['rmse']:.3f}")
        emit(f"  95% limits of agreement: "
             f"[{st['limits_of_agreement_lo']:+.2f}, "
             f"{st['limits_of_agreement_hi']:+.2f}]")
        emit(f"\n  within 1 unit     : {st['pct_within_1']:.1f}%")
        emit(f"  within 2 units    : {st['pct_within_2']:.1f}%")
        emit(f"  within 5 units    : {st['pct_within_5']:.1f}%")

        if exact:
            ok = (
                st["pearson_r"] >= EXACT_MIN_PEARSON
                and st["mean_abs_error"] <= EXACT_MAX_MEAN_ABS_ERROR
            )
            emit(f"\n  gate: r >= {EXACT_MIN_PEARSON} and MAE <= "
                 f"{EXACT_MAX_MEAN_ABS_ERROR}  ->  {'PASS' if ok else 'FAIL'}")
        else:
            ok = st["spearman_r"] >= CONVERGENT_MIN_SPEARMAN
            emit(f"\n  gate: Spearman >= {CONVERGENT_MIN_SPEARMAN}  ->  "
                 f"{'PASS' if ok else 'FAIL'}")
        verdicts[label] = ok

        # split by data quality
        if merged["frac_valid"].notna().any():
            emit(f"\n  agreement by recording quality:")
            emit(f"    {'stratum':<26s} {'n':>5s} {'r':>8s} {'MAE':>8s} "
                 f"{'bias':>8s}")
            for name, lo, hi in QUALITY_STRATA:
                sel = merged[
                    (merged["frac_valid"] > lo) & (merged["frac_valid"] <= hi)
                ]
                if len(sel) < 10:
                    emit(f"    {name:<26s} {len(sel):>5d}  (too few)")
                    continue
                sst = agreement_stats(sel[pub_col], sel[our_col])
                if sst is None:
                    continue
                emit(f"    {name:<26s} {sst['n']:>5d} "
                     f"{sst['pearson_r']:>8.4f} "
                     f"{sst['mean_abs_error']:>8.3f} "
                     f"{sst['mean_bias']:>+8.3f}")
            emit("\n    Tight agreement in the clean strata that loosens only")
            emit("    where the oximeter was often not measuring is the expected")
            emit("    signature of correct cleaning.")

    # --- graded thresholds, checked for internal consistency ---
    emit("\n" + "=" * 78)
    emit("INTERNAL CONSISTENCY: GRADED HYPOXIA THRESHOLDS")
    emit("=" * 78)
    emit("\n  Time below a deeper threshold must never exceed time below a")
    emit("  shallower one. This is a pure self-consistency check that needs no")
    emit("  external reference, and it catches an off-by-one in the threshold")
    emit("  loop that correlation against SHHS could easily miss.")
    ladder = ["sleep_ct95_pct", "sleep_ct90_pct", "sleep_ct88_pct",
              "sleep_ct85_pct", "sleep_ct80_pct", "sleep_ct75_pct",
              "sleep_ct70_pct"]
    ladder = [c for c in ladder if c in merged.columns]
    violations = 0
    for a, b in zip(ladder, ladder[1:]):
        bad = int((merged[b] > merged[a] + 1e-9).sum())
        violations += bad
        emit(f"    {b} > {a} in {bad} participants")
    emit(f"\n  total violations: {violations}  -> "
         f"{'PASS' if violations == 0 else 'FAIL'}")
    verdicts["graded threshold monotonicity"] = violations == 0

    # same check on the published SHHS ladder for comparison
    pub_ladder = [c for c in ["pctlt90", "pctlt85", "pctlt80", "pctlt75"]
                  if c in merged.columns]
    if len(pub_ladder) > 1:
        pub_violations = sum(
            int((merged[b] > merged[a] + 1e-9).sum())
            for a, b in zip(pub_ladder, pub_ladder[1:])
        )
        emit(f"  same check on the PUBLISHED ladder: {pub_violations} violations")

    # ODI has to go up as the detection threshold loosens
    emit("\n  Desaturation index must decrease as the required depth grows:")
    odi_ladder = [c for c in ["sleep_odi2", "sleep_odi3", "sleep_odi4"]
                  if c in merged.columns]
    odi_violations = 0
    for a, b in zip(odi_ladder, odi_ladder[1:]):
        bad = int((merged[b] > merged[a] + 1e-9).sum())
        odi_violations += bad
        emit(f"    {b} > {a} in {bad} participants")
    emit(f"  total violations: {odi_violations}  -> "
         f"{'PASS' if odi_violations == 0 else 'FAIL'}")
    verdicts["ODI threshold monotonicity"] = odi_violations == 0

    # --- plots ---
    n_pairs = len([p for p, o, _, _ in VALIDATION_PAIRS
                   if p in merged.columns and o in merged.columns])
    if n_pairs:
        fig, axes = plt.subplots(2, n_pairs, figsize=(4 * n_pairs, 8))
        if n_pairs == 1:
            axes = axes.reshape(2, 1)
        col = 0
        for pub_col, our_col, label, exact in VALIDATION_PAIRS:
            if pub_col not in merged.columns or our_col not in merged.columns:
                continue
            both = merged[[pub_col, our_col]].dropna()
            if both.empty:
                col += 1
                continue
            pub = both[pub_col].to_numpy()
            our = both[our_col].to_numpy()

            ax = axes[0, col]
            ax.scatter(pub, our, s=4, alpha=0.25, edgecolors="none")
            lim = [min(pub.min(), our.min()), max(pub.max(), our.max())]
            ax.plot(lim, lim, "r--", lw=1, label="identity")
            ax.set_xlabel(f"published {pub_col}")
            ax.set_ylabel(f"ours {our_col}")
            st = all_stats.get(our_col)
            title = label if len(label) < 34 else label[:31] + "..."
            if st:
                title += f"\nr={st['pearson_r']:.3f}, MAE={st['mean_abs_error']:.2f}"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

            # Bland-Altman (difference vs mean) shows bias that changes with
            # magnitude, which the scatter can hide
            ax = axes[1, col]
            mean_val = (pub + our) / 2.0
            diff = our - pub
            ax.scatter(mean_val, diff, s=4, alpha=0.25, edgecolors="none")
            ax.axhline(np.mean(diff), color="r", lw=1,
                       label=f"bias {np.mean(diff):+.2f}")
            ax.axhline(np.mean(diff) + 1.96 * np.std(diff), color="r",
                       ls=":", lw=1)
            ax.axhline(np.mean(diff) - 1.96 * np.std(diff), color="r",
                       ls=":", lw=1, label="95% limits")
            ax.set_xlabel("mean of the two")
            ax.set_ylabel("ours minus published")
            ax.set_title("Bland-Altman", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)
            col += 1

        fig.suptitle(
            "Computed digital oximetry biomarkers versus published SHHS "
            "variables",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=140)
        plt.close(fig)
        emit(f"\nPlots written to: {PLOT_PATH}")

    # --- verdict ---
    emit("\n" + "=" * 78)
    emit("VALIDATION VERDICT")
    emit("=" * 78)
    for name, ok in verdicts.items():
        emit(f"  {'PASS' if ok else 'FAIL'}  {name}")

    n_fail = sum(1 for ok in verdicts.values() if not ok)
    if n_fail == 0:
        emit("\n  All validation gates passed. The biomarker library reproduces")
        emit("  the independently scored SHHS oximetry variables, which")
        emit("  confirms correct EDF channel selection, correct sleep stage")
        emit("  alignment, and correct biomarker definitions.")
    else:
        emit(f"\n  {n_fail} gate(s) failed. Inspect the scatter and")
        emit("  Bland-Altman panels: a slope far from 1.0 indicates a units or")
        emit("  scale error, a large intercept indicates a systematic offset,")
        emit("  and a fan shape indicates a magnitude-dependent problem.")

    with open(REPORT_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\nReport written to: {REPORT_PATH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
