"""
Aim 1 fix: re-clean the SpO2 signals and save a validity mask with each one.

The first pipeline (preprocess_shhs_batch.py) filled every gap, so the hours of
0.0 the SaO2 channel reads after the probe comes off were forward-filled into
made-up constant signal (participant 200020: T90 of 43.9% on the filled signal,
3.3% on valid data only). Here a sample is invalid if it's in an XML artifact
event, outside the plausible range or part of a frozen-sensor run. Following
Levy et al. (2021), short dropouts are recovered and long ones count as time
that can't be analysed: gaps up to MAX_INTERP_GAP_SEC are linearly
interpolated, while longer gaps and any gap at a recording edge stay NaN and
are left out of every Aim 2 denominator.

Outputs (the original outputs are left as they are):
    processed/cleaned_spo2_v2/shhs1-{pid}.npy   float64 signal, NaN = invalid
    processed/validity_mask/shhs1-{pid}.npy     bool, True = valid/analysable
    processed/reclean_log.csv                   per-participant accounting

Usage:
    python3 aim1/reclean_with_validity_mask.py [--force]
"""

import os
import glob
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyedflib

# --- config ---

SHHS_DIR = "/Volumes/ThesisHD/Data"
OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"

EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")
CLEAN_V2_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2_v2")
MASK_DIR = os.path.join(OUTPUT_DIR, "validity_mask")
LOG_PATH = os.path.join(OUTPUT_DIR, "reclean_log.csv")

SPO2_CHANNEL_CANDIDATES = ["sao2", "spo2"]

# Real arterial saturation in a living participant never sits outside this
# band, so anything outside it comes from the instrument. A disconnected probe
# reading exactly 0.0 is the most common case.
PLAUSIBLE_SPO2_RANGE = (50.0, 100.0)

# Stuck-sensor threshold, in seconds. A frozen oximeter can hold one normal
# looking value for tens of minutes (participant 202267 sits at 98.0453% for
# 53 min), which passes the range check and wasn't flagged by the technicians.
# Zero variability pulls every complexity and desaturation biomarker towards a
# "perfectly stable night". The median participant's longest constant run is
# 267 s, so 300 s would start masking ordinary stable sleep; 600 s still removes
# the obvious cases and masks 0.6% of analysable time.
STUCK_SENSOR_MIN_RUN_SEC = 600

# Cut-off between a recoverable dropout and time that can't be analysed. 30 s
# bridges motion artifacts and probe re-seating, and a straight line can't
# create a desaturation (the detector needs a peak then a trough). It could
# hide one shorter than the gap, so the cap limits that to one brief event per
# gap. Mean bridged gap across the cohort is 9.6 s.
MAX_INTERP_GAP_SEC = 30

# Participants with less analysable time than this are still written out
# (nothing is deleted) but flagged so Aim 2 can exclude them. Four hours is
# the usual minimum for stable overnight biomarkers.
MIN_VALID_HOURS = 4.0

FORCE_REPROCESS = False
LOG_CHECKPOINT_EVERY = 100

# The full SHHS-1 cohort; 205804 is the highest ID.
PID_RANGE = (200001, 205804)


def parse_events(xml_path):
    """
    Parse an NSRR annotation XML into a DataFrame of scored events.

    Same behaviour as the original pipeline so the two runs stay comparable.
    NSRR repeats the event name after a pipe ("SpO2 artifact|SpO2 artifact"),
    so only the first field is kept.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for event in root.findall(".//ScoredEvent"):
        concept = event.findtext("EventConcept", default="")
        start = event.findtext("Start")
        duration = event.findtext("Duration")
        if start is None or duration is None:
            continue
        rows.append(
            {
                "event_concept": concept.split("|")[0],
                "start_sec": float(start),
                "duration_sec": float(duration),
            }
        )

    return pd.DataFrame(rows)


def find_spo2_channel(edf_reader):
    """Index of the SpO2 / SaO2 channel in an open EdfReader."""
    labels = edf_reader.getSignalLabels()
    for i, label in enumerate(labels):
        if any(candidate in label.lower() for candidate in SPO2_CHANNEL_CANDIDATES):
            return i
    raise ValueError(f"No SpO2/SaO2 channel found among channels: {labels}")


def find_stuck_runs(signal, fs, min_run_sec=STUCK_SENSOR_MIN_RUN_SEC):
    """
    Boolean mask of samples in a run of identical values longer than
    min_run_sec (a frozen oximeter). Runs on the raw signal, so "stuck" is
    judged on what the device reported; the 0.0 disconnection stretches get
    caught here too, which is harmless.
    """
    n = signal.size
    if n < 2:
        return np.zeros(n, dtype=bool)

    min_run = int(round(min_run_sec * fs))
    stuck = np.zeros(n, dtype=bool)

    change_points = np.flatnonzero(np.diff(signal) != 0)
    boundaries = np.concatenate(([-1], change_points, [n - 1]))
    run_starts = boundaries[:-1] + 1
    run_ends = boundaries[1:]
    run_lengths = run_ends - run_starts + 1

    for start, end, length in zip(run_starts, run_ends, run_lengths):
        if length > min_run:
            stuck[start : end + 1] = True
    return stuck


def find_gaps(invalid_mask):
    """
    Contiguous runs of True in a boolean mask, as a list of (start, stop)
    half-open index pairs.

    Pads with False before differencing so runs at either end of the array
    don't need a special case.
    """
    if invalid_mask.size == 0:
        return []
    padded = np.concatenate(([False], invalid_mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def clean_with_mask(raw_signal, fs, events_df):
    """
    Returns (cleaned_signal, validity_mask, accounting) for one participant.

    cleaned_signal : float64, NaN wherever the sample can't be analysed
    validity_mask  : bool, True wherever the sample can go into a biomarker

    The mask is True exactly where the signal is finite. Returning both is
    redundant, but it lets downstream code use the mask directly instead of
    relying on isnan checks.
    """
    signal = np.asarray(raw_signal, dtype=float).copy()
    n = signal.size

    # --- invalid source 1: technician-scored artifact events ---
    artifact_mask = np.zeros(n, dtype=bool)
    if not events_df.empty and "event_concept" in events_df.columns:
        artifacts = events_df[
            events_df["event_concept"].str.contains("artifact", case=False, na=False)
        ]
        for _, row in artifacts.iterrows():
            start_idx = max(int(row["start_sec"] * fs), 0)
            end_idx = min(int((row["start_sec"] + row["duration_sec"]) * fs), n)
            if end_idx > start_idx:
                artifact_mask[start_idx:end_idx] = True

    # --- invalid source 2: physiologically impossible values ---
    # checked on the raw signal before any masking, so the count describes
    # the input itself
    low, high = PLAUSIBLE_SPO2_RANGE
    implausible_mask = (
        (signal < low) | (signal > high) | ~np.isfinite(signal)
    )

    # --- invalid source 3: a frozen sensor ---
    # Also done on the raw signal (see find_stuck_runs). Catches in-range,
    # unflagged artifact that both checks above miss.
    stuck_mask = find_stuck_runs(np.asarray(raw_signal, dtype=float), fs)

    invalid = artifact_mask | implausible_mask | stuck_mask
    n_artifact_masked = int(artifact_mask.sum())
    n_implausible_masked = int(implausible_mask.sum())
    n_stuck_masked = int(stuck_mask.sum())
    # samples only the stuck detector caught, i.e. what the extra pass adds
    n_stuck_only = int(np.sum(stuck_mask & ~artifact_mask & ~implausible_mask))
    n_invalid_initial = int(invalid.sum())

    signal[invalid] = np.nan

    # --- bridge the short gaps only ---
    max_gap_samples = int(round(MAX_INTERP_GAP_SEC * fs))
    gaps = find_gaps(invalid)

    n_short_gaps = 0
    n_long_gaps = 0
    n_interpolated_samples = 0
    n_edge_unfilled = 0

    valid_idx = np.flatnonzero(~invalid)

    for start, stop in gaps:
        length = stop - start
        if length > max_gap_samples:
            n_long_gaps += 1
            continue

        # A gap at either end of the recording has nothing on one side to
        # interpolate from. Filling it means extrapolating a constant
        # outwards, which is the made-up data this script is meant to
        # remove, so these stay invalid whatever their length.
        if start == 0 or stop == n:
            n_edge_unfilled += length
            continue

        n_short_gaps += 1
        left_val = signal[start - 1]
        right_val = signal[stop]
        if not (np.isfinite(left_val) and np.isfinite(right_val)):
            # Shouldn't happen, since gaps are maximal runs of invalid samples
            # and so both neighbours are valid. Guarded anyway so NaN-derived
            # values can't get written without anyone noticing.
            n_edge_unfilled += length
            continue

        # interpolate strictly between the good samples either side
        signal[start:stop] = np.linspace(
            left_val, right_val, num=length + 2
        )[1:-1]
        invalid[start:stop] = False
        n_interpolated_samples += length

    validity_mask = ~invalid

    # signal is finite exactly where it's valid
    signal[~validity_mask] = np.nan
    assert np.isfinite(signal[validity_mask]).all(), "valid samples must be finite"
    assert np.isnan(signal[~validity_mask]).all(), "invalid samples must be NaN"

    n_valid = int(validity_mask.sum())
    accounting = {
        "n_samples": n,
        "fs": float(fs),
        "n_artifact_masked": n_artifact_masked,
        "n_implausible_masked": n_implausible_masked,
        "n_stuck_masked": n_stuck_masked,
        "n_stuck_only": n_stuck_only,
        "n_invalid_initial": n_invalid_initial,
        "n_short_gaps_interpolated": n_short_gaps,
        "n_interpolated_samples": n_interpolated_samples,
        "n_long_gaps_left_invalid": n_long_gaps,
        "n_edge_samples_left_invalid": n_edge_unfilled,
        "n_valid_samples": n_valid,
        "valid_hours": n_valid / (fs * 3600.0),
        "frac_valid": n_valid / n if n else 0.0,
        # how much signal the original pipeline would have made up for this
        # participant (useful for the thesis write-up)
        "n_samples_original_would_fabricate": n_invalid_initial - n_interpolated_samples,
    }
    return signal, validity_mask, accounting


def process_participant(pid):
    """Re-clean one participant and return its log row."""
    edf_path = os.path.join(SHHS_DIR, f"shhs1-{pid}.edf")
    xml_path = os.path.join(SHHS_DIR, f"shhs1-{pid}-nsrr.xml")
    events_path = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")

    # reuse the parsed events table if there is one, since XML parsing is the
    # slow part and the first Aim 1 pass already wrote these
    if os.path.exists(events_path):
        events_df = pd.read_csv(events_path)
    else:
        # Participants added after the original pipeline ran have no events
        # table yet, so it gets written here as well as parsed. The Aim 1
        # audit needs it, and Aim 2 feature extraction quietly skips anyone
        # without one, which would drop them from the cohort with no error.
        # Same columns and format as the original pipeline.
        events_df = parse_events(xml_path)
        events_df.to_csv(events_path, index=False)

    reader = pyedflib.EdfReader(edf_path)
    try:
        channel_idx = find_spo2_channel(reader)
        fs = reader.getSampleFrequency(channel_idx)
        raw_signal = reader.readSignal(channel_idx)
    finally:
        reader.close()

    signal, mask, accounting = clean_with_mask(raw_signal, fs, events_df)

    np.save(os.path.join(CLEAN_V2_DIR, f"shhs1-{pid}.npy"), signal)
    np.save(os.path.join(MASK_DIR, f"shhs1-{pid}.npy"), mask)

    row = {"pid": pid, "status": "ok"}
    row.update(accounting)
    row["sufficient_valid_data"] = accounting["valid_hours"] >= MIN_VALID_HOURS
    return row


def discover_pids():
    """Participant IDs inside PID_RANGE that have both an EDF and an NSRR XML."""
    pids = []
    for path in glob.glob(os.path.join(SHHS_DIR, "shhs1-*.edf")):
        match = re.search(r"shhs1-(\d+)\.edf$", os.path.basename(path))
        if not match:
            continue
        pid = match.group(1)
        if not (PID_RANGE[0] <= int(pid) <= PID_RANGE[1]):
            continue
        xml_path = os.path.join(SHHS_DIR, f"shhs1-{pid}-nsrr.xml")
        if not os.path.exists(xml_path):
            continue
        # A copy in progress leaves 0-byte placeholders that pass an existence
        # check (Finder creates every destination file before filling any of
        # them). Reading one would log a false failure, so an empty EDF or XML
        # counts as not there yet and gets picked up on a rerun.
        if os.path.getsize(path) == 0 or os.path.getsize(xml_path) == 0:
            continue
        pids.append(pid)
    return sorted(pids)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="reprocess participants whose outputs already exist, needed after "
             "changing the cleaning logic",
    )
    args = parser.parse_args()
    force = FORCE_REPROCESS or args.force

    os.makedirs(CLEAN_V2_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    pids = discover_pids()
    print("=" * 78)
    print("AIM 1 CORRECTION - RE-CLEAN WITH VALIDITY MASK")
    print("=" * 78)
    print(f"\nParticipants found         : {len(pids)}")
    print(f"Max interpolated gap       : {MAX_INTERP_GAP_SEC} s")
    print(f"Stuck-sensor run threshold : {STUCK_SENSOR_MIN_RUN_SEC} s")
    print(f"Force reprocess            : {force}")
    print(f"Plausible range            : {PLAUSIBLE_SPO2_RANGE[0]:.0f}-"
          f"{PLAUSIBLE_SPO2_RANGE[1]:.0f}%")
    print(f"Min analysable time        : {MIN_VALID_HOURS} h")
    print(f"Signal output              : {CLEAN_V2_DIR}")
    print(f"Mask output                : {MASK_DIR}\n")

    log_rows = []

    previous_rows = {}
    if os.path.exists(LOG_PATH) and not force:
        try:
            previous = pd.read_csv(LOG_PATH)
            previous = previous[previous["status"] == "ok"]
            for record in previous.to_dict(orient="records"):
                previous_rows[int(record["pid"])] = record
            print(f"Previous accounting loaded for {len(previous_rows)} "
                  f"participants (carried forward if skipped).\n")
        except Exception as exc:
            print(f"WARNING: could not read {LOG_PATH} ({exc}). Skipped "
                  f"participants will get stub log rows.\n")

    def write_log(rows):
        """
        Write the full log every time, checkpoints included. Previous rows for
        participants not reached yet are appended, so a crash mid-run (the
        external drive has unmounted before) never leaves a truncated log.
        """
        seen = {int(r["pid"]) for r in rows}
        pending = [dict(r, processed_this_run=False)
                   for pid_key, r in previous_rows.items() if pid_key not in seen]
        frame = pd.DataFrame(rows + pending)
        frame["pid"] = frame["pid"].astype(int)
        frame = frame.sort_values("pid").reset_index(drop=True)
        frame.to_csv(LOG_PATH, index=False)
        return frame

    started = time.time()

    for i, pid in enumerate(pids, start=1):
        out_signal = os.path.join(CLEAN_V2_DIR, f"shhs1-{pid}.npy")
        out_mask = os.path.join(MASK_DIR, f"shhs1-{pid}.npy")

        out_events = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")

        if (
            os.path.exists(out_signal)
            and os.path.exists(out_mask)
            and os.path.exists(out_events)
            and not force
        ):
            # carry the previous run's accounting forward instead of writing a
            # stub row, otherwise an incremental run overwrites reclean_log.csv
            # and wipes the cleaning stats for everyone it skipped
            previous = previous_rows.get(int(pid))
            if previous is not None:
                carried = dict(previous)
                carried["processed_this_run"] = False
                log_rows.append(carried)
            else:
                log_rows.append({"pid": int(pid),
                                 "status": "skipped (already done)",
                                 "processed_this_run": False})
            continue

        try:
            row = process_participant(pid)
            if i % 50 == 0 or row["frac_valid"] < 0.75:
                print(
                    f"[{i}/{len(pids)}] shhs1-{pid}: "
                    f"{row['valid_hours']:.1f}h valid "
                    f"({row['frac_valid']:.1%}), "
                    f"{row['n_short_gaps_interpolated']} short gaps bridged, "
                    f"{row['n_long_gaps_left_invalid']} long gaps excluded"
                )
        except Exception as exc:
            row = {"pid": pid, "status": "failed", "error": str(exc)}
            print(f"[{i}/{len(pids)}] shhs1-{pid} FAILED: {exc}")

        row["pid"] = int(row["pid"])
        row["processed_this_run"] = True
        log_rows.append(row)

        if i % LOG_CHECKPOINT_EVERY == 0:
            write_log(log_rows)
            rate = i / (time.time() - started)
            remaining = (len(pids) - i) / rate / 60 if rate else 0
            print(
                f"  --- checkpoint {i}/{len(pids)}, "
                f"{rate:.1f} participants/s, ~{remaining:.1f} min remaining ---"
            )

    log_df = write_log(log_rows)

    # --- summary ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"\nElapsed: {(time.time() - started) / 60:.1f} min")
    for status, count in log_df["status"].value_counts().items():
        print(f"  status '{status}': {count}")
    if "processed_this_run" in log_df.columns:
        n_new = int(log_df["processed_this_run"].fillna(False).astype(bool).sum())
        print(f"  processed in this run     : {n_new}")
        print(f"  carried forward (skipped) : {len(log_df) - n_new}")
        print("  Statistics below cover the WHOLE cohort, not just this run.")

    done = log_df[log_df["status"] == "ok"]
    if done.empty:
        print("\nNothing newly processed. Delete outputs or set FORCE_REPROCESS "
              "to re-run.")
        print(f"Log written to: {LOG_PATH}")
        return 0

    print(f"\nAnalysable time retained across {len(done)} participants:")
    print(f"  median valid              : {done['valid_hours'].median():.2f} h "
          f"({done['frac_valid'].median():.1%} of recording)")
    print(f"  worst case                : {done['valid_hours'].min():.2f} h "
          f"({done['frac_valid'].min():.1%})")
    print(f"  best case                 : {done['valid_hours'].max():.2f} h")

    print(f"\nStuck-sensor detection (frozen device, in range and unflagged):")
    if "n_stuck_masked" in done.columns:
        n_affected = int((done["n_stuck_only"] > 0).sum())
        print(f"  participants affected     : {n_affected}/{len(done)} "
              f"({n_affected / len(done):.1%})")
        print(f"  samples caught ONLY by this pass: "
              f"{int(done['n_stuck_only'].sum()):,} "
              f"({done['n_stuck_only'].sum() / done['n_samples'].sum():.3%} "
              f"of all samples)")
        if n_affected:
            worst = done.nlargest(5, "n_stuck_only")
            print(f"  worst affected participants:")
            for _, r in worst.iterrows():
                print(f"    pid {r['pid']}: {int(r['n_stuck_only']):,} samples "
                      f"({r['n_stuck_only'] / 60:.0f} min frozen)")

    print(f"\nGap handling:")
    print(f"  short gaps bridged  total : "
          f"{int(done['n_short_gaps_interpolated'].sum()):,}")
    print(f"  samples interpolated total: "
          f"{int(done['n_interpolated_samples'].sum()):,} "
          f"({done['n_interpolated_samples'].sum() / done['n_samples'].sum():.2%} "
          f"of all samples)")
    print(f"  long gaps excluded  total : "
          f"{int(done['n_long_gaps_left_invalid'].sum()):,}")

    fabricated = done["n_samples_original_would_fabricate"]
    print(f"\nFabricated data avoided versus the original pipeline:")
    print(f"  total samples             : {int(fabricated.sum()):,} "
          f"({fabricated.sum() / 3600:.1f} participant-hours)")
    print(f"  median per participant    : {fabricated.median():.0f} samples")
    print(f"  worst participant         : {int(fabricated.max()):,} samples "
          f"({fabricated.max() / 3600:.1f} h), pid "
          f"{done.loc[fabricated.idxmax(), 'pid']}")

    insufficient = done[~done["sufficient_valid_data"]]
    print(f"\nParticipants below {MIN_VALID_HOURS} h analysable time: "
          f"{len(insufficient)}/{len(done)}")
    if len(insufficient):
        print("  These are written out but flagged; Aim 2 excludes them so that")
        print("  overnight biomarkers are not computed on a fragment of a night.")
        preview = insufficient.nsmallest(15, "valid_hours")
        for _, r in preview.iterrows():
            print(f"    pid {r['pid']}: {r['valid_hours']:.2f} h valid "
                  f"({r['frac_valid']:.1%})")

    print(f"\nLog written to: {LOG_PATH}")
    print("\nNext: python3 aim1/verify_aim1_v2.py, then build features in aim2/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
