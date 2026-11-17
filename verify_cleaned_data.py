"""
QC checks on the cleaned SpO2 outputs from preprocess_shhs_batch.py.

Run after preprocess_shhs_batch.py. Existing files can still be wrong, so for
every participant with outputs it checks that:
  1. both files (.npy signal, .csv events) exist and aren't empty
  2. no NaNs are left in the signal (interpolation should have filled them all)
  3. min/max sit inside the plausible range used during cleaning
  4. the recording isn't suspiciously short (truncated/corrupt file) or flat
     (a stuck sensor that got past the artifact and range masks)
  5. the run log agrees with what's actually on disk

Output: processed/qc_report.csv (one row per participant), plus a printed
summary and a list of participants worth a closer look.

Usage:
    python3 verify_cleaned_data.py
"""

import os
import glob

import numpy as np
import pandas as pd

# --- config ---
# OUTPUT_DIR should match preprocess_shhs_batch.py
OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"
CLEAN_SIGNAL_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2")
EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")
LOG_PATH = os.path.join(OUTPUT_DIR, "preprocessing_log.csv")

PLAUSIBLE_SPO2_RANGE = (50.0, 100.0)
MIN_REASONABLE_SAMPLES = 3600  # ~1 hour at the SHHS SaO2 channel's 1Hz rate
FLAT_SIGNAL_STD_THRESHOLD = 0.01  # a real overnight trace should vary at least a little


def check_participant(pid):
    """Run every check for one participant and return a dict of results."""
    npy_path = os.path.join(CLEAN_SIGNAL_DIR, f"shhs1-{pid}.npy")
    csv_path = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")

    if not os.path.exists(npy_path):
        return {"pid": pid, "status": "MISSING_NPY", "issues": "cleaned .npy file missing"}
    if not os.path.exists(csv_path):
        return {"pid": pid, "status": "MISSING_CSV", "issues": "events .csv file missing"}

    signal = np.load(npy_path)
    events_df = pd.read_csv(csv_path)

    issues = []

    if len(signal) == 0:
        issues.append("cleaned signal is empty")
    else:
        if np.isnan(signal).any():
            issues.append(f"{int(np.isnan(signal).sum())} NaN values remain")
        if signal.min() < PLAUSIBLE_SPO2_RANGE[0] - 1e-6 or signal.max() > PLAUSIBLE_SPO2_RANGE[1] + 1e-6:
            issues.append(f"values outside plausible range: min={signal.min():.2f}, max={signal.max():.2f}")
        if len(signal) < MIN_REASONABLE_SAMPLES:
            issues.append(f"suspiciously short recording: {len(signal)} samples")
        if np.nanstd(signal) < FLAT_SIGNAL_STD_THRESHOLD:
            issues.append(f"signal looks flat (std={np.nanstd(signal):.4f}) - possible stuck sensor")

    if len(events_df) == 0:
        issues.append("events table is empty")

    return {
        "pid": pid,
        "status": "OK" if not issues else "FLAGGED",
        "n_samples": len(signal),
        "min_val": float(signal.min()) if len(signal) else None,
        "max_val": float(signal.max()) if len(signal) else None,
        "n_events": len(events_df),
        "issues": "; ".join(issues) if issues else "",
    }


def main():
    npy_files = glob.glob(os.path.join(CLEAN_SIGNAL_DIR, "shhs1-*.npy"))
    pids = sorted(os.path.basename(f).replace("shhs1-", "").replace(".npy", "") for f in npy_files)
    print(f"Found {len(pids)} cleaned participants to check.\n")

    if not pids:
        print("No cleaned files found - has preprocess_shhs_batch.py finished any participants yet?")
        return

    results = [check_participant(pid) for pid in pids]
    results_df = pd.DataFrame(results)

    n_ok = (results_df["status"] == "OK").sum()
    n_flagged = (results_df["status"] == "FLAGGED").sum()
    n_missing = results_df["status"].str.startswith("MISSING").sum()

    print(f"OK:                    {n_ok}")
    print(f"Flagged for review:    {n_flagged}")
    print(f"Missing output files:  {n_missing}")

    if n_flagged or n_missing:
        print("\n--- Participants needing a closer look ---")
        for _, row in results_df[results_df["status"] != "OK"].iterrows():
            print(f"  shhs1-{row['pid']}: {row['status']} - {row['issues']}")

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
        log_ok_pids = set(log_df.loc[log_df["status"] == "ok", "pid"].astype(str))
        disk_pids = set(pids)
        missing_from_disk = log_ok_pids - disk_pids
        if missing_from_disk:
            print(f"\nWARNING: log says these succeeded but files aren't on disk: {sorted(missing_from_disk)}")
    else:
        print("\n(No preprocessing_log.csv found yet - skipping log cross-check.)")

    report_path = os.path.join(OUTPUT_DIR, "qc_report.csv")
    results_df.to_csv(report_path, index=False)
    print(f"\nFull QC report written to:\n  {report_path}")


if __name__ == "__main__":
    main()
