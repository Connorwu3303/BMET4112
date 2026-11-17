"""
Batch SpO2 preprocessing for SHHS (first Aim 1 cleaning pass).

Runs the single-participant prototype (shhs1-200001) over every participant
downloaded locally (EDF + matching NSRR XML). For each one it:
  1. parses the *-nsrr.xml annotations into a table of scored events
     (respiratory events, arousals, desaturations, artifacts, sleep stages)
  2. reads the raw SpO2/SaO2 waveform from the .edf
  3. sets XML-flagged SpO2 artifact segments and out-of-range values to NaN,
     linearly interpolates over them and forward/back-fills at the edges
  4. saves the cleaned waveform (.npy) and events (.csv), and keeps a run log
     so failures can be found without re-running everything

This version fills every gap, however long. The fix for long dropouts is in
aim1/reclean_with_validity_mask.py.

Outputs (processed/): cleaned_spo2/shhs1-{pid}.npy,
events/shhs1-{pid}_events.csv, preprocessing_log.csv

Needs: pip install pyedflib numpy pandas

Usage:
    python3 preprocess_shhs_batch.py
"""

import os
import glob
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyedflib

# --- config ---
# check these two paths match the machine before running

# raw SHHS EDF + XML files (external hard drive)
SHHS_DIR = "/Volumes/ThesisHD/Data"

# outputs go in a "processed" folder on the external drive too, since the
# internal disk is low on space
OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"

CLEAN_SIGNAL_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2")
EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")
LOG_PATH = os.path.join(OUTPUT_DIR, "preprocessing_log.csv")

# channel label fragments to look for (case-insensitive substring match)
SPO2_CHANNEL_CANDIDATES = ["sao2", "spo2"]

# Only process participants that are actually on the external drive. Raise
# the upper bound as more of the cohort is downloaded. The drive has the
# 2,591 participants (200001-202600) confirmed on /Volumes/ThesisHD/Data.
# The start was later narrowed to 200101, so 200001-200100 keep the outputs
# from an earlier run.


#PID_RANGE = (200001, 202600)
PID_RANGE = (200101, 202600)  # inclusive

# SpO2 in a living person doesn't really read below ~50%, so anything lower is
# sensor/motion noise the XML didn't flag as an artifact. We treat it the same
# way: mask and interpolate over it.
PLAUSIBLE_SPO2_RANGE = (50.0, 100.0)

# True reprocesses participants even if their outputs already exist, e.g.
# after changing the cleaning logic (like adding the plausibility clip).
# Leave False for normal incremental runs. Set it True for a full rerun from
# scratch (including the first 116 participants) so the whole cohort goes
# through the same logic and old and new runs aren't mixed.
FORCE_REPROCESS = False

# also write the log every N participants, so progress can be checked
# mid-run and there's still a log if the run gets interrupted
LOG_CHECKPOINT_EVERY = 50


def is_materialized(path, size_check_bytes=1024):
    """
    Rough check that a file is really on disk and not a OneDrive cloud-only
    placeholder. Placeholders report a real size but allocate ~0 blocks, so
    compare allocated blocks with the reported size. Not 100% reliable on
    every filesystem, which is why PID_RANGE above is the real safety net.
    """
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size == 0:
        return True  # empty file, nothing to materialize
    allocated_bytes = st.st_blocks * 512
    return allocated_bytes >= min(size_check_bytes, st.st_size)


def find_participants(shhs_dir, pid_range=PID_RANGE):
    """
    Sorted participant IDs (strings, e.g. "200001") inside pid_range that have
    both an .edf and a matching -nsrr.xml fully downloaded (not cloud-only
    placeholders).
    """
    edf_files = glob.glob(os.path.join(shhs_dir, "shhs1-*.edf"))
    participant_ids = []

    for edf_path in edf_files:
        match = re.search(r"shhs1-(\d+)\.edf$", os.path.basename(edf_path))
        if not match:
            continue
        pid = match.group(1)

        if not (pid_range[0] <= int(pid) <= pid_range[1]):
            continue

        xml_path = os.path.join(shhs_dir, f"shhs1-{pid}-nsrr.xml")
        if not os.path.exists(xml_path):
            continue

        if not (is_materialized(edf_path) and is_materialized(xml_path)):
            print(f"  Skipping shhs1-{pid}: file(s) not fully downloaded yet.")
            continue

        participant_ids.append(pid)

    return sorted(participant_ids)


def parse_events(xml_path):
    """
    Parse an NSRR annotation XML into a DataFrame with columns event_concept,
    start_sec, duration_sec. NSRR repeats the event name after a pipe
    ("SpO2 artifact|SpO2 artifact"), so only the first part is kept.
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
    """Index of the SpO2/SaO2 channel in an open EdfReader."""
    labels = edf_reader.getSignalLabels()
    for i, label in enumerate(labels):
        if any(candidate in label.lower() for candidate in SPO2_CHANNEL_CANDIDATES):
            return i
    raise ValueError(f"No SpO2/SaO2 channel found among channels: {labels}")


def clean_spo2_signal(signal, fs, events_df):
    """
    Set bad SpO2 samples to NaN, linearly interpolate over the gaps, then
    forward/back-fill whatever is left at the start or end of the recording.

    Two kinds of bad sample get masked:
      1. XML-flagged artifact events (scored by the technician, e.g. probe
         disconnection or motion)
      2. readings outside PLAUSIBLE_SPO2_RANGE that the XML didn't flag,
         i.e. noise the original scoring missed

    Returns the cleaned signal and the number of out-of-range raw samples.
    """
    signal = signal.astype(float).copy()
    n_implausible = int(
        np.sum((signal < PLAUSIBLE_SPO2_RANGE[0]) | (signal > PLAUSIBLE_SPO2_RANGE[1]))
    )

    artifact_events = events_df[
        events_df["event_concept"].str.contains("artifact", case=False, na=False)
    ]

    for _, row in artifact_events.iterrows():
        start_idx = max(int(row["start_sec"] * fs), 0)
        end_idx = min(int((row["start_sec"] + row["duration_sec"]) * fs), len(signal))
        signal[start_idx:end_idx] = np.nan

    implausible_mask = (signal < PLAUSIBLE_SPO2_RANGE[0]) | (signal > PLAUSIBLE_SPO2_RANGE[1])
    signal[implausible_mask] = np.nan

    series = pd.Series(signal)
    series = series.interpolate(method="linear", limit_direction="both")
    series = series.ffill().bfill()

    cleaned = series.to_numpy()
    return cleaned, n_implausible


def process_participant(pid, shhs_dir):
    """Run the whole pipeline for one participant and return a summary dict."""
    edf_path = os.path.join(shhs_dir, f"shhs1-{pid}.edf")
    xml_path = os.path.join(shhs_dir, f"shhs1-{pid}-nsrr.xml")

    events_df = parse_events(xml_path)

    reader = pyedflib.EdfReader(edf_path)
    try:
        channel_idx = find_spo2_channel(reader)
        fs = reader.getSampleFrequency(channel_idx)
        raw_signal = reader.readSignal(channel_idx)
    finally:
        reader.close()

    cleaned_signal, n_implausible = clean_spo2_signal(raw_signal, fs, events_df)

    np.save(os.path.join(CLEAN_SIGNAL_DIR, f"shhs1-{pid}.npy"), cleaned_signal)
    events_df.to_csv(os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv"), index=False)

    n_artifacts = int(
        events_df["event_concept"].str.contains("artifact", case=False, na=False).sum()
    )

    return {
        "pid": pid,
        "status": "ok",
        "n_samples": len(cleaned_signal),
        "fs": fs,
        "n_events": len(events_df),
        "n_artifacts": n_artifacts,
        "n_implausible_samples": n_implausible,
    }


def main():
    os.makedirs(CLEAN_SIGNAL_DIR, exist_ok=True)
    os.makedirs(EVENTS_DIR, exist_ok=True)

    participant_ids = find_participants(SHHS_DIR)
    print(f"Found {len(participant_ids)} participants with local EDF + NSRR XML files.\n")

    log_rows = []
    for i, pid in enumerate(participant_ids, start=1):
        out_npy = os.path.join(CLEAN_SIGNAL_DIR, f"shhs1-{pid}.npy")
        out_csv = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")

        already_done = os.path.exists(out_npy) and os.path.exists(out_csv)
        if already_done and not FORCE_REPROCESS:
            print(f"[{i}/{len(participant_ids)}] shhs1-{pid} already processed, skipping.")
            log_rows.append({"pid": pid, "status": "skipped (already done)"})
            continue

        print(f"[{i}/{len(participant_ids)}] Processing shhs1-{pid} ...", end=" ")
        try:
            result = process_participant(pid, SHHS_DIR)
            print(
                f"OK ({result['n_samples']} samples, {result['n_artifacts']} artifact segments, "
                f"{result['n_implausible_samples']} implausible samples caught)"
            )
        except Exception as exc:
            result = {"pid": pid, "status": "failed", "error": str(exc)}
            print(f"FAILED: {exc}")
        log_rows.append(result)

        if i % LOG_CHECKPOINT_EVERY == 0:
            pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
            print(f"  --- checkpoint: progress log written at {i}/{len(participant_ids)} ---")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(LOG_PATH, index=False)

    n_ok = (log_df["status"] == "ok").sum()
    n_failed = (log_df["status"] == "failed").sum()
    n_skipped = (log_df["status"] == "skipped (already done)").sum()
    print(
        f"\nDone. {n_ok} succeeded, {n_failed} failed, {n_skipped} skipped. "
        f"Log written to:\n  {LOG_PATH}"
    )


if __name__ == "__main__":
    main()
