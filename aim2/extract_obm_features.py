"""
Runs the OBM extraction over the whole SHHS cohort.

Loads each participant's corrected Aim 1 output (cleaned signal + validity
mask) and their parsed events table, computes the full OBM set with
obm_library and writes one wide feature table.

It has to read the v2 Aim 1 outputs. The original cleaned_spo2 folder
forward-filled long dropouts and shouldn't be used for features (the header
of aim1/reclean_with_validity_mask.py shows how much damage that did).

Participants with less than MIN_VALID_HOURS of analysable time are still
extracted but flagged in `sufficient_valid_data`, so the modelling step can
drop them without recomputing anything.

Output:
    processed/aim2/obm_features.csv       one row per participant
    processed/aim2/obm_extraction_log.csv per-participant status and timing

Usage:
    python3 aim2/extract_obm_features.py
    python3 aim2/extract_obm_features.py --limit 50      # quick smoke test
    python3 aim2/extract_obm_features.py --jobs 4
"""

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from obm_library import build_stage_array, compute_all_obm

# --- config ---

OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"
SIGNAL_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2_v2")
MASK_DIR = os.path.join(OUTPUT_DIR, "validity_mask")
EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")

AIM2_DIR = os.path.join(OUTPUT_DIR, "aim2")
FEATURES_PATH = os.path.join(AIM2_DIR, "obm_features.csv")
LOG_PATH = os.path.join(AIM2_DIR, "obm_extraction_log.csv")

# SHHS records the SaO2 channel at 1 Hz
DEFAULT_FS = 1.0

# minimum analysable hours for someone's overnight biomarkers to count as
# stable. Same as MIN_VALID_HOURS in the Aim 1 recleaning.
MIN_VALID_HOURS = 4.0

ANALYSIS_WINDOWS = ("recording", "sleep", "nrem", "rem")


def extract_one(pid, fs=DEFAULT_FS):
    """
    Compute the full OBM set for one participant.

    Returns (feature_dict, log_dict). Never raises: failures go into the log
    dict so one unreadable participant can't kill a cohort run.
    """
    started = time.time()
    log = {"pid": pid, "status": "ok", "error": ""}

    try:
        signal = np.load(os.path.join(SIGNAL_DIR, f"shhs1-{pid}.npy"))
        mask = np.load(os.path.join(MASK_DIR, f"shhs1-{pid}.npy"))
        events = pd.read_csv(os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv"))
    except Exception as exc:
        log.update({"status": "failed", "error": f"load: {exc}"})
        return None, log

    try:
        stages = build_stage_array(events, signal.size, fs)
        features = compute_all_obm(
            signal, mask, fs, stages, windows=ANALYSIS_WINDOWS
        )
    except Exception as exc:
        log.update({"status": "failed", "error": f"compute: {exc}"})
        return None, log

    features["pid"] = int(pid)

    valid_hours = features.get("recording_valid_hours", np.nan)
    features["sufficient_valid_data"] = bool(
        np.isfinite(valid_hours) and valid_hours >= MIN_VALID_HOURS
    )

    log.update(
        {
            "valid_hours": valid_hours,
            "sleep_hours": features.get("sleep_valid_hours", np.nan),
            "rem_hours": features.get("rem_valid_hours", np.nan),
            "n_features": len(features),
            "seconds": time.time() - started,
        }
    )
    return features, log


def discover_pids():
    """Participant IDs that have a signal, a mask and an events table."""
    def ids(pattern, regex, directory):
        found = set()
        for path in glob.glob(os.path.join(directory, pattern)):
            match = re.search(regex, os.path.basename(path))
            if match:
                found.add(match.group(1))
        return found

    signals = ids("shhs1-*.npy", r"shhs1-(\d+)\.npy$", SIGNAL_DIR)
    masks = ids("shhs1-*.npy", r"shhs1-(\d+)\.npy$", MASK_DIR)
    events = ids("shhs1-*_events.csv", r"shhs1-(\d+)_events\.csv$", EVENTS_DIR)

    complete = signals & masks & events
    return sorted(complete), signals, masks, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N participants")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="parallel workers, -1 for all cores")
    args = parser.parse_args()

    os.makedirs(AIM2_DIR, exist_ok=True)

    pids, signals, masks, events = discover_pids()

    print("=" * 78)
    print("AIM 2 - OBM FEATURE EXTRACTION")
    print("=" * 78)
    print(f"\nsignals available      : {len(signals)}")
    print(f"masks available        : {len(masks)}")
    print(f"events available       : {len(events)}")
    print(f"complete triples       : {len(pids)}")

    missing_mask = sorted(signals - masks)
    if missing_mask:
        print(f"\nWARNING: {len(missing_mask)} signals have no validity mask yet.")
        print("  The Aim 1 correction may still be running. These are skipped;")
        print("  rerun this script once it finishes to pick them up.")

    if not pids:
        print("\nNothing to extract. Has aim1/reclean_with_validity_mask.py run?")
        return 1

    if args.limit:
        pids = pids[: args.limit]
        print(f"\nLIMIT active: processing {len(pids)} participants only.")

    print(f"\nAnalysis windows       : {', '.join(ANALYSIS_WINDOWS)}")
    print(f"Parallel workers       : {args.jobs}")
    print(f"\nExtracting biomarkers for {len(pids)} participants...\n")

    started = time.time()
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(extract_one)(pid) for pid in pids
    )
    elapsed = time.time() - started

    feature_rows = [f for f, _ in results if f is not None]
    log_rows = [lg for _, lg in results]

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False)

    if not feature_rows:
        print("\nAll participants failed. See the log for the reason.")
        print(log_df["error"].value_counts().to_string())
        return 1

    features = pd.DataFrame(feature_rows)

    # pid and the data-quality columns first, then the rest alphabetically
    lead = ["pid", "sufficient_valid_data", "recording_valid_hours",
            "sleep_valid_hours", "nrem_valid_hours", "rem_valid_hours"]
    lead = [c for c in lead if c in features.columns]
    rest = sorted(c for c in features.columns if c not in lead)
    features = features[lead + rest].sort_values("pid").reset_index(drop=True)
    features.to_csv(FEATURES_PATH, index=False)

    # --- summary ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nElapsed                : {elapsed / 60:.1f} min "
          f"({elapsed / max(len(pids), 1):.2f} s per participant)")
    for status, count in log_df["status"].value_counts().items():
        print(f"  status '{status}': {count}")
    if (log_df["status"] == "failed").any():
        print("\n  Failure reasons:")
        for reason, count in (
            log_df.loc[log_df["status"] == "failed", "error"]
            .value_counts()
            .head(10)
            .items()
        ):
            print(f"    {count:>4d}  {reason}")

    print(f"\nFeature table          : {features.shape[0]} rows x "
          f"{features.shape[1]} columns")
    print(f"  written to           : {FEATURES_PATH}")
    print(f"  log written to       : {LOG_PATH}")

    n_sufficient = int(features["sufficient_valid_data"].sum())
    print(f"\nData quality:")
    print(f"  >= {MIN_VALID_HOURS} h analysable    : {n_sufficient}/{len(features)}")
    print(f"  median analysable    : "
          f"{features['recording_valid_hours'].median():.2f} h")
    print(f"  median sleep time    : "
          f"{features['sleep_valid_hours'].median():.2f} h")

    # completeness per window. REM will be patchy since not everyone has
    # scored REM.
    print(f"\nFeature completeness by window:")
    for window in ANALYSIS_WINDOWS:
        cols = [c for c in features.columns if c.startswith(f"{window}_")]
        if not cols:
            continue
        complete = features[cols].notna().mean(axis=1).mean()
        print(f"  {window:<10s} {len(cols):>4d} features, "
              f"{complete:.1%} populated on average")

    # flag mostly-empty columns now so they don't turn up as a surprise
    # during model fitting
    missing_frac = features.isna().mean().sort_values(ascending=False)
    very_sparse = missing_frac[missing_frac > 0.5]
    print(f"\nColumns more than 50% missing: {len(very_sparse)}")
    for col, frac in very_sparse.head(20).items():
        print(f"    {frac:.1%}  {col}")
    if len(very_sparse) > 20:
        print(f"    ... and {len(very_sparse) - 20} more")

    # quick sanity check on the headline biomarkers so a broken run is
    # obvious straight away
    if "sleep_ct90_pct" in features.columns:
        t90 = features["sleep_ct90_pct"].dropna()
        print(f"\nT90 over valid sleep time (sleep_ct90_pct):")
        print(f"  median {t90.median():.2f}%, mean {t90.mean():.2f}%, "
              f"max {t90.max():.2f}%")
        print(f"  participants with T90 > 10%: {(t90 > 10).sum()} "
              f"({(t90 > 10).mean():.1%})")
    if "sleep_odi3" in features.columns:
        odi = features["sleep_odi3"].dropna()
        print(f"\nODI3 over valid sleep time (sleep_odi3):")
        print(f"  median {odi.median():.2f} events/h, "
              f"mean {odi.mean():.2f}, max {odi.max():.2f}")

    print("\nNext: python3 aim2/validate_obm.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
