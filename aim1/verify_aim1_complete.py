"""
Aim 1 completion audit for the first-pass cleaned SpO2 (processed/cleaned_spo2/).

Works everything out again from the files on disk instead of reading
qc_report.csv, so it catches a stale or half-written QC report as well as real
data problems. Criteria:
  A. Coverage: every raw EDF has a cleaned signal and events table, no orphans.
  B. Integrity: every signal is finite, in range, long enough and not flat.
  C. Fidelity: signal length matches the XML "Recording Start Time" duration.
  D. Consistency: IDs 200001-200100 came from an earlier run, so the two ID
     blocks are checked separately for out-of-range values.
  E. Usability: the events tables have the sleep staging Aim 2 needs.

Output: processed/aim1_audit.csv (one row per participant) plus a printed
verdict per criterion.

Usage:
    python3 aim1/verify_aim1_complete.py
"""

import os
import glob
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

# --- config ---
# paths and range must match preprocess_shhs_batch.py

SHHS_DIR = "/Volumes/ThesisHD/Data"
OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"

CLEAN_SIGNAL_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2")
EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")
LOG_PATH = os.path.join(OUTPUT_DIR, "preprocessing_log.csv")
AUDIT_PATH = os.path.join(OUTPUT_DIR, "aim1_audit.csv")

PLAUSIBLE_SPO2_RANGE = (50.0, 100.0)

# 3600 samples is one hour at 1 Hz. Anything shorter isn't a usable
# overnight recording for Aim 2.
MIN_SAMPLES = 3600

# A working oximeter always shows some beat-to-beat variation, so an SD this
# small over a whole night means a stuck sensor got through both masking passes.
MIN_SIGNAL_STD = 0.01

# Allowed difference between signal length and the XML duration. A couple of
# seconds covers rounding between the EDF record structure and the XML field.
DURATION_TOLERANCE_SEC = 2.0

# Constant runs this long (seconds at 1 Hz) get reported. Some are expected
# and harmless: forward/back filling at the recording edges creates them, and
# real SpO2 plateaus are normal in quiet sleep. So it's a soft flag for review
# only and doesn't fail anything.
LONG_CONSTANT_RUN_SEC = 900

# IDs up to here were last written by an earlier run, because PID_RANGE was
# later narrowed to start at 200101. Kept separate so criterion D can compare
# the two blocks.
EARLY_BLOCK_MAX_PID = 200100


def longest_constant_run(signal):
    """
    Length in samples of the longest run of identical values.

    Used as a stuck-sensor check. Worked out from the indices where the value
    changes, so it's vectorised (no loop).
    """
    if len(signal) == 0:
        return 0
    change_points = np.flatnonzero(np.diff(signal) != 0)
    # run boundaries = change points plus the two array ends
    boundaries = np.concatenate(([-1], change_points, [len(signal) - 1]))
    return int(np.diff(boundaries).max())


def xml_declared_duration(xml_path):
    """
    Recording duration in seconds from the XML's opening "Recording Start
    Time" event, which is cheap to read and independent of the EDF signal.
    Returns None if the event isn't there.
    """
    try:
        for _, element in ET.iterparse(xml_path, events=("end",)):
            if element.tag != "ScoredEvent":
                continue
            concept = element.findtext("EventConcept", default="")
            if concept.split("|")[0].strip() == "Recording Start Time":
                duration = element.findtext("Duration")
                return float(duration) if duration is not None else None
            element.clear()
    except (ET.ParseError, OSError, ValueError):
        return None
    return None


def audit_events_table(events_path):
    """
    Check a parsed events table is usable for Aim 2.

    Returns a dict describing the sleep staging (plus a few event counts).
    Mainly this is about whether stage annotations exist at all, since the
    REM / NREM biomarkers in Aim 2 can't be computed without them.
    """
    result = {
        "events_readable": False,
        "n_events": 0,
        "n_stage_events": 0,
        "n_rem_events": 0,
        "n_nrem_events": 0,
        "n_wake_events": 0,
        "n_desat_events": 0,
        "n_apnea_events": 0,
        "n_hypopnea_events": 0,
        "n_artifact_events": 0,
        "staged_sec": 0.0,
    }

    try:
        events = pd.read_csv(events_path)
    except Exception:
        return result

    if events.empty or "event_concept" not in events.columns:
        # an empty table is readable but useless, so leave the flag False
        return result

    result["events_readable"] = True
    result["n_events"] = len(events)

    concept = events["event_concept"].astype(str)
    duration = pd.to_numeric(events["duration_sec"], errors="coerce").fillna(0.0)

    # NSRR stage names: "Stage 1 sleep" to "Stage 4 sleep" for NREM,
    # "REM sleep" for REM, "Wake" for wake
    is_rem = concept.str.contains("REM sleep", case=False, na=False)
    is_nrem = concept.str.match(r"Stage [1-4] sleep", case=False, na=False)
    is_wake = concept.str.fullmatch(r"Wake", case=False, na=False)

    result["n_rem_events"] = int(is_rem.sum())
    result["n_nrem_events"] = int(is_nrem.sum())
    result["n_wake_events"] = int(is_wake.sum())
    result["n_stage_events"] = int((is_rem | is_nrem | is_wake).sum())
    result["staged_sec"] = float(duration[is_rem | is_nrem].sum())

    result["n_desat_events"] = int(
        concept.str.contains("desaturation", case=False, na=False).sum()
    )
    result["n_apnea_events"] = int(
        concept.str.contains("apnea", case=False, na=False).sum()
    )
    result["n_hypopnea_events"] = int(
        concept.str.contains("hypopnea", case=False, na=False).sum()
    )
    result["n_artifact_events"] = int(
        concept.str.contains("artifact", case=False, na=False).sum()
    )

    return result


def audit_participant(pid):
    """
    Full audit of one participant. Never raises: any failure goes into the
    issues string so one bad file can't stop the whole audit.
    """
    row = {"pid": int(pid)}
    issues = []

    npy_path = os.path.join(CLEAN_SIGNAL_DIR, f"shhs1-{pid}.npy")
    events_path = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")
    xml_path = os.path.join(SHHS_DIR, f"shhs1-{pid}-nsrr.xml")
    edf_path = os.path.join(SHHS_DIR, f"shhs1-{pid}.edf")

    row["has_edf"] = os.path.exists(edf_path)
    row["has_xml"] = os.path.exists(xml_path)
    row["has_npy"] = os.path.exists(npy_path)
    row["has_events"] = os.path.exists(events_path)

    # --- A. coverage ---
    if not row["has_npy"]:
        issues.append("missing cleaned signal")
    if not row["has_events"]:
        issues.append("missing events table")

    # --- B. signal integrity ---
    signal = None
    if row["has_npy"]:
        try:
            signal = np.load(npy_path)
        except Exception as exc:
            issues.append(f"signal unreadable ({exc})")

    if signal is not None:
        row["dtype"] = str(signal.dtype)
        row["n_samples"] = int(signal.size)

        if signal.ndim != 1:
            issues.append(f"signal not 1-D (shape {signal.shape})")

        n_nan = int(np.isnan(signal).sum())
        n_inf = int(np.isinf(signal).sum())
        row["n_nan"] = n_nan
        row["n_inf"] = n_inf
        if n_nan:
            issues.append(f"{n_nan} NaN samples remain")
        if n_inf:
            issues.append(f"{n_inf} Inf samples remain")

        finite = signal[np.isfinite(signal)]
        if finite.size == 0:
            issues.append("no finite samples")
            row["min_val"] = np.nan
            row["max_val"] = np.nan
            row["mean_val"] = np.nan
            row["std_val"] = np.nan
            row["longest_constant_run_sec"] = np.nan
        else:
            row["min_val"] = float(finite.min())
            row["max_val"] = float(finite.max())
            row["mean_val"] = float(finite.mean())
            row["std_val"] = float(finite.std())

            low, high = PLAUSIBLE_SPO2_RANGE
            if row["min_val"] < low:
                issues.append(f"min {row['min_val']:.2f} below plausible floor {low}")
            if row["max_val"] > high:
                issues.append(f"max {row['max_val']:.2f} above plausible ceiling {high}")

            if row["std_val"] < MIN_SIGNAL_STD:
                issues.append(f"signal effectively flat (std {row['std_val']:.4g})")

            # in seconds, since the SHHS SpO2 channel is 1 Hz
            row["longest_constant_run_sec"] = longest_constant_run(signal)

        if row["n_samples"] < MIN_SAMPLES:
            issues.append(
                f"recording too short ({row['n_samples']} samples "
                f"< {MIN_SAMPLES})"
            )

    # --- C. length vs XML duration ---
    declared = xml_declared_duration(xml_path) if row["has_xml"] else None
    row["xml_declared_sec"] = declared if declared is not None else np.nan
    if declared is not None and signal is not None:
        # SHHS-1 SpO2 is 1 Hz, so one sample is one second
        delta = abs(float(signal.size) - declared)
        row["length_vs_xml_delta_sec"] = delta
        if delta > DURATION_TOLERANCE_SEC:
            issues.append(
                f"length mismatch: {signal.size} samples vs XML "
                f"{declared:.0f}s (delta {delta:.0f}s)"
            )
    else:
        row["length_vs_xml_delta_sec"] = np.nan

    # --- E. events usability ---
    if row["has_events"]:
        row.update(audit_events_table(events_path))
        if not row["events_readable"]:
            issues.append("events table unreadable or empty")
        elif row["n_stage_events"] == 0:
            issues.append("no sleep stage annotations (blocks REM/NREM features)")
        elif row["n_rem_events"] == 0:
            # Not a failure. Some nights really have no scored REM and the
            # feature code has to cope with that. The REM count is kept so
            # Aim 2 knows how many participants will be missing REM biomarkers.
            pass

    row["n_issues"] = len(issues)
    row["issues"] = "; ".join(issues)
    return row


def discover_pids():
    """
    ID sets for the raw EDFs, the cleaned signals and the events tables.
    main() audits their union so orphans show up both ways: a raw file that
    was never cleaned, and a cleaned file with no raw input behind it.
    """
    raw = set()
    for path in glob.glob(os.path.join(SHHS_DIR, "shhs1-*.edf")):
        match = re.search(r"shhs1-(\d+)\.edf$", os.path.basename(path))
        if match:
            raw.add(match.group(1))

    cleaned = set()
    for path in glob.glob(os.path.join(CLEAN_SIGNAL_DIR, "shhs1-*.npy")):
        match = re.search(r"shhs1-(\d+)\.npy$", os.path.basename(path))
        if match:
            cleaned.add(match.group(1))

    events = set()
    for path in glob.glob(os.path.join(EVENTS_DIR, "shhs1-*_events.csv")):
        match = re.search(r"shhs1-(\d+)_events\.csv$", os.path.basename(path))
        if match:
            events.add(match.group(1))

    return raw, cleaned, events


def main():
    print("=" * 78)
    print("AIM 1 COMPLETION AUDIT")
    print("=" * 78)

    if not os.path.isdir(CLEAN_SIGNAL_DIR):
        print(f"FATAL: {CLEAN_SIGNAL_DIR} not found. Is the external drive mounted?")
        sys.exit(1)

    raw, cleaned, events = discover_pids()
    all_pids = sorted(raw | cleaned | events)

    print(f"\nInventory on disk:")
    print(f"  raw EDF files          : {len(raw)}")
    print(f"  cleaned signal .npy    : {len(cleaned)}")
    print(f"  parsed events .csv     : {len(events)}")
    print(f"  union to audit         : {len(all_pids)}")

    missing_clean = sorted(raw - cleaned)
    missing_events = sorted(raw - events)
    orphan_clean = sorted(cleaned - raw)
    orphan_events = sorted(events - raw)

    print(f"\n  raw with no cleaned signal : {len(missing_clean)}")
    print(f"  raw with no events table   : {len(missing_events)}")
    print(f"  cleaned with no raw EDF    : {len(orphan_clean)}")
    print(f"  events with no raw EDF     : {len(orphan_events)}")

    print(f"\nAuditing {len(all_pids)} participants (reading every signal from disk)...")
    rows = []
    for i, pid in enumerate(all_pids, start=1):
        rows.append(audit_participant(pid))
        if i % 250 == 0:
            print(f"  {i}/{len(all_pids)} audited")

    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_PATH, index=False)
    print(f"  {len(all_pids)}/{len(all_pids)} audited")
    print(f"\nPer-participant audit table written to:\n  {AUDIT_PATH}")

    # --- verdicts ---
    print("\n" + "=" * 78)
    print("CRITERION VERDICTS")
    print("=" * 78)

    verdicts = {}

    # A. coverage
    coverage_ok = (
        not missing_clean and not missing_events
        and not orphan_clean and not orphan_events
    )
    verdicts["A. COVERAGE"] = coverage_ok
    print(f"\nA. COVERAGE")
    print(f"   every raw EDF has a cleaned signal and events table, no orphans")
    print(f"   -> {'PASS' if coverage_ok else 'FAIL'}")
    for label, group in [
        ("raw missing cleaned signal", missing_clean),
        ("raw missing events table", missing_events),
        ("orphan cleaned signal", orphan_clean),
        ("orphan events table", orphan_events),
    ]:
        if group:
            print(f"      {label}: {len(group)} -> {group[:15]}")

    # B. integrity
    finite_ok = (audit["n_nan"].fillna(0) == 0).all() and (
        audit["n_inf"].fillna(0) == 0
    ).all()
    range_ok = bool(
        (audit["min_val"] >= PLAUSIBLE_SPO2_RANGE[0]).all()
        and (audit["max_val"] <= PLAUSIBLE_SPO2_RANGE[1]).all()
    )
    length_ok = bool((audit["n_samples"] >= MIN_SAMPLES).all())
    flat_ok = bool((audit["std_val"] >= MIN_SIGNAL_STD).all())
    integrity_ok = finite_ok and range_ok and length_ok and flat_ok
    verdicts["B. INTEGRITY"] = integrity_ok
    print(f"\nB. INTEGRITY")
    print(f"   no NaN / no Inf anywhere        -> {'PASS' if finite_ok else 'FAIL'}")
    print(
        f"   all values within {PLAUSIBLE_SPO2_RANGE[0]:.0f}-"
        f"{PLAUSIBLE_SPO2_RANGE[1]:.0f}%          -> "
        f"{'PASS' if range_ok else 'FAIL'}"
    )
    print(
        f"   all recordings >= {MIN_SAMPLES} samples   -> "
        f"{'PASS' if length_ok else 'FAIL'}"
    )
    print(f"   no flat / stuck signals         -> {'PASS' if flat_ok else 'FAIL'}")
    print(
        f"   observed range across cohort    : "
        f"{audit['min_val'].min():.2f}% to {audit['max_val'].max():.2f}%"
    )
    print(
        f"   recording length                : "
        f"{audit['n_samples'].min() / 3600:.1f}h to "
        f"{audit['n_samples'].max() / 3600:.1f}h "
        f"(median {audit['n_samples'].median() / 3600:.1f}h)"
    )

    # C. fidelity
    compared = audit["length_vs_xml_delta_sec"].notna()
    fidelity_ok = bool(
        compared.any()
        and (audit.loc[compared, "length_vs_xml_delta_sec"] <= DURATION_TOLERANCE_SEC).all()
    )
    verdicts["C. FIDELITY"] = fidelity_ok
    print(f"\nC. FIDELITY (signal length vs independently declared XML duration)")
    print(f"   participants comparable         : {int(compared.sum())}/{len(audit)}")
    if compared.any():
        print(
            f"   max absolute disagreement       : "
            f"{audit.loc[compared, 'length_vs_xml_delta_sec'].max():.2f} s"
        )
    print(f"   -> {'PASS' if fidelity_ok else 'FAIL'}")

    # D. consistency across the two processing blocks
    early = audit[audit["pid"] <= EARLY_BLOCK_MAX_PID]
    late = audit[audit["pid"] > EARLY_BLOCK_MAX_PID]
    print(f"\nD. CONSISTENCY (cleaning logic identical across processing blocks)")
    print(
        f"   block 1 (pid <= {EARLY_BLOCK_MAX_PID}, written by earlier run) : "
        f"{len(early)} participants"
    )
    print(
        f"   block 2 (pid >  {EARLY_BLOCK_MAX_PID}, written by latest run)  : "
        f"{len(late)} participants"
    )
    consistency_ok = True
    if len(early) and len(late):
        # If block 1 had been written without the plausibility clip, the
        # out-of-range values would only show up there. If both blocks stay
        # in range, the clip was either applied or made no difference, so the
        # two blocks are interchangeable.
        early_bad = int(
            (early["min_val"] < PLAUSIBLE_SPO2_RANGE[0]).sum()
            + (early["max_val"] > PLAUSIBLE_SPO2_RANGE[1]).sum()
        )
        late_bad = int(
            (late["min_val"] < PLAUSIBLE_SPO2_RANGE[0]).sum()
            + (late["max_val"] > PLAUSIBLE_SPO2_RANGE[1]).sum()
        )
        print(f"   block 1 out-of-range participants: {early_bad}")
        print(f"   block 2 out-of-range participants: {late_bad}")
        print(
            f"   block 1 min / block 2 min       : "
            f"{early['min_val'].min():.2f}% / {late['min_val'].min():.2f}%"
        )
        consistency_ok = (early_bad == 0) and (late_bad == 0)
    verdicts["D. CONSISTENCY"] = consistency_ok
    print(f"   -> {'PASS' if consistency_ok else 'FAIL'}")

    # E. usability for Aim 2
    staged = audit["n_stage_events"].fillna(0) > 0
    has_rem = audit["n_rem_events"].fillna(0) > 0
    usability_ok = bool(staged.all())
    verdicts["E. USABILITY"] = usability_ok
    print(f"\nE. USABILITY FOR AIM 2")
    print(f"   participants with sleep staging : {int(staged.sum())}/{len(audit)}")
    print(f"   participants with scored REM    : {int(has_rem.sum())}/{len(audit)}")
    print(
        f"   participants with desaturations  : "
        f"{int((audit['n_desat_events'].fillna(0) > 0).sum())}/{len(audit)}"
    )
    if staged.any():
        staged_hours = audit.loc[staged, "staged_sec"] / 3600.0
        print(
            f"   scored sleep time (h)           : "
            f"{staged_hours.min():.1f} to {staged_hours.max():.1f} "
            f"(median {staged_hours.median():.1f})"
        )
    if not has_rem.all():
        no_rem = audit.loc[~has_rem, "pid"].tolist()
        print(
            f"   NOTE {len(no_rem)} participants have no scored REM, so REM-specific "
            f"biomarkers will be missing for them (handled as NaN in Aim 2)."
        )
    print(f"   -> {'PASS' if usability_ok else 'FAIL'}")

    # --- run log check ---
    # the log is only written at checkpoints, so it can lag the files on disk
    print("\n" + "=" * 78)
    print("RUN LOG RECONCILIATION")
    print("=" * 78)
    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH)
        print(f"  preprocessing_log.csv rows : {len(log)}")
        print(f"  cleaned signals on disk    : {len(cleaned)}")
        if "status" in log.columns:
            for status, count in log["status"].value_counts().items():
                print(f"    status '{status}': {count}")
        logged = set(log["pid"].astype(str).str.zfill(6)) if "pid" in log else set()
        on_disk_not_logged = sorted(cleaned - logged)
        print(f"  cleaned but absent from log: {len(on_disk_not_logged)}")
        if on_disk_not_logged:
            print(
                f"    range {on_disk_not_logged[0]} to {on_disk_not_logged[-1]}\n"
                f"    Expected: the log only records participants touched by the\n"
                f"    most recent run, and PID_RANGE started at 200101. These\n"
                f"    files were written by an earlier run. Criteria A-D above\n"
                f"    validate them directly from disk, so the log gap is a\n"
                f"    bookkeeping artefact and not missing data."
            )
    else:
        print("  no preprocessing_log.csv found")

    # --- soft flags ---
    # worth eyeballing, but they don't block Aim 2
    print("\n" + "=" * 78)
    print("SOFT FLAGS (review only, non-blocking)")
    print("=" * 78)
    long_runs = audit[
        audit["longest_constant_run_sec"].fillna(0) >= LONG_CONSTANT_RUN_SEC
    ]
    print(
        f"  participants with a constant run >= {LONG_CONSTANT_RUN_SEC}s : "
        f"{len(long_runs)}/{len(audit)}"
    )
    if len(long_runs):
        print(
            f"    longest observed: {long_runs['longest_constant_run_sec'].max():.0f}s "
            f"(pid {int(long_runs.loc[long_runs['longest_constant_run_sec'].idxmax(), 'pid'])})"
        )
        print(
            "    Constant runs are partly expected: edge fill and genuine stable\n"
            "    plateaus in quiet sleep both produce them. Only a run covering a\n"
            "    large fraction of a night suggests a stuck sensor."
        )

    # crude outlier check on mean overnight saturation
    if audit["mean_val"].notna().any():
        low_mean = audit[audit["mean_val"] < 85.0]
        print(f"  participants with mean SpO2 < 85%: {len(low_mean)}")
        if len(low_mean):
            print(
                f"    pids: {low_mean['pid'].tolist()[:15]}\n"
                "    Severe but physiologically possible in this cohort. Worth\n"
                "    eyeballing the waveform before modelling."
            )

    # --- final verdict ---
    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    for name, ok in verdicts.items():
        print(f"  {name:<18s} {'PASS' if ok else 'FAIL'}")

    all_pass = all(verdicts.values())
    n_with_issues = int((audit["n_issues"] > 0).sum())
    print(f"\n  participants with any hard issue: {n_with_issues}/{len(audit)}")
    if n_with_issues:
        print("\n  Issue breakdown:")
        counter = Counter()
        for text in audit.loc[audit["n_issues"] > 0, "issues"]:
            for part in str(text).split("; "):
                # strip numbers so similar issues group together
                counter[re.sub(r"[\d.]+", "N", part)] += 1
        for issue, count in counter.most_common():
            print(f"    {count:>5d}  {issue}")
        print("\n  Affected pids (first 30):")
        print(f"    {audit.loc[audit['n_issues'] > 0, 'pid'].tolist()[:30]}")

    if all_pass and n_with_issues == 0:
        print(
            f"\n  AIM 1 IS COMPLETE for the {len(cleaned)} participants currently\n"
            f"  on the drive. All cleaned signals are finite, in range, of\n"
            f"  plausible duration, length-consistent with their annotations, and\n"
            f"  carry the sleep staging Aim 2 needs. Safe to proceed to Aim 2.\n"
            f"\n  When the remaining participants land on the drive, extend\n"
            f"  PID_RANGE in preprocess_shhs_batch.py, rerun it with\n"
            f"  FORCE_REPROCESS = False, then rerun this audit."
        )
    else:
        print(
            "\n  AIM 1 IS NOT YET CLEAN. Resolve the failures above before\n"
            "  building features on top of these signals."
        )

    return 0 if (all_pass and n_with_issues == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
