"""
Aim 1 completion audit for the corrected (v2) outputs.

The first audit (verify_aim1_complete.py) passed the original outputs, whose
real defect (hours of forward-filled constant signal) only showed up as a soft
flag. Here invalid samples are NaN with a validity mask, and F is a regression
test for that defect. Criteria:
  A. Coverage: every raw EDF has a signal, a mask and an events table.
  B. Invariants: bool mask of the same shape, True exactly where the signal is
     finite.
  C. Range: every valid sample is inside the plausible range.
  D. Fidelity: signal length matches the recording duration in the XML.
  E. Analysable time: enough valid time for overnight biomarkers (reported only).
  F. No fabrication: no implausibly long constant run among valid samples.
  G. Usability: sleep staging exists and overlaps the valid samples.

Output: processed/aim1_v2_audit.csv plus a printed verdict per criterion.

Usage:
    python3 aim1/verify_aim1_v2.py
"""

import os
import glob
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pandas as pd

SHHS_DIR = "/Volumes/ThesisHD/Data"
OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"

SIGNAL_DIR = os.path.join(OUTPUT_DIR, "cleaned_spo2_v2")
MASK_DIR = os.path.join(OUTPUT_DIR, "validity_mask")
EVENTS_DIR = os.path.join(OUTPUT_DIR, "events")
AUDIT_PATH = os.path.join(OUTPUT_DIR, "aim1_v2_audit.csv")

PLAUSIBLE_SPO2_RANGE = (50.0, 100.0)
MIN_VALID_HOURS = 4.0
DURATION_TOLERANCE_SEC = 2.0

# Threshold for criterion F. A real SpO2 plateau at 1 Hz can easily last
# several minutes in stable sleep, but identical values for most of an hour
# isn't physiology. The original outputs had runs up to 3.8 h, every one of
# them ending exactly at the end of the recording.
MAX_PLAUSIBLE_CONSTANT_RUN_SEC = 1800

# Apart from being long, a fill artefact runs right to the end (or start) of
# the record. Tracked separately since that's the exact failure being tested.
EDGE_PROXIMITY_SAMPLES = 2


def longest_constant_run_in_valid(signal, mask):
    """
    Longest run of identical values among the valid samples, and whether that
    run touches a recording edge.

    Runs are found inside each contiguous valid segment, so a value repeated
    either side of an unmeasured gap doesn't count as one run.
    """
    best_len = 0
    best_touches_edge = False
    n = signal.size

    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)

    for seg_start, seg_stop in zip(starts, stops):
        seg = signal[seg_start:seg_stop]
        if seg.size == 0:
            continue
        change_points = np.flatnonzero(np.diff(seg) != 0)
        boundaries = np.concatenate(([-1], change_points, [seg.size - 1]))
        run_lengths = np.diff(boundaries)
        if run_lengths.size == 0:
            continue
        idx = int(np.argmax(run_lengths))
        run_len = int(run_lengths[idx])
        if run_len > best_len:
            best_len = run_len
            run_start = seg_start + int(boundaries[idx]) + 1
            run_end = seg_start + int(boundaries[idx + 1])
            best_touches_edge = (
                run_start <= EDGE_PROXIMITY_SAMPLES
                or run_end >= n - 1 - EDGE_PROXIMITY_SAMPLES
            )
    return best_len, best_touches_edge


def xml_declared_duration(xml_path):
    """Recording duration in seconds, from the XML's own opening event."""
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


def audit_participant(pid, fs=1.0):
    row = {"pid": int(pid)}
    issues = []

    signal_path = os.path.join(SIGNAL_DIR, f"shhs1-{pid}.npy")
    mask_path = os.path.join(MASK_DIR, f"shhs1-{pid}.npy")
    events_path = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")
    xml_path = os.path.join(SHHS_DIR, f"shhs1-{pid}-nsrr.xml")
    edf_path = os.path.join(SHHS_DIR, f"shhs1-{pid}.edf")

    row["has_edf"] = os.path.exists(edf_path)
    row["has_signal"] = os.path.exists(signal_path)
    row["has_mask"] = os.path.exists(mask_path)
    row["has_events"] = os.path.exists(events_path)

    if not row["has_signal"]:
        issues.append("missing v2 signal")
    if not row["has_mask"]:
        issues.append("missing validity mask")
    if not row["has_events"]:
        issues.append("missing events table")

    if not (row["has_signal"] and row["has_mask"]):
        row["n_issues"] = len(issues)
        row["issues"] = "; ".join(issues)
        return row

    try:
        signal = np.load(signal_path)
        mask = np.load(mask_path)
    except Exception as exc:
        issues.append(f"unreadable ({exc})")
        row["n_issues"] = len(issues)
        row["issues"] = "; ".join(issues)
        return row

    row["n_samples"] = int(signal.size)
    row["mask_dtype"] = str(mask.dtype)

    # --- B. invariants ---
    if signal.shape != mask.shape:
        issues.append(f"shape mismatch {signal.shape} vs {mask.shape}")
    else:
        if mask.dtype != np.bool_:
            issues.append(f"mask is {mask.dtype}, expected bool")
        finite = np.isfinite(signal)
        # the signal/mask pairing, checked both ways
        n_valid_not_finite = int(np.sum(mask & ~finite))
        n_invalid_but_finite = int(np.sum(~mask & finite))
        row["n_valid_not_finite"] = n_valid_not_finite
        row["n_invalid_but_finite"] = n_invalid_but_finite
        if n_valid_not_finite:
            issues.append(f"{n_valid_not_finite} valid samples are not finite")
        if n_invalid_but_finite:
            issues.append(
                f"{n_invalid_but_finite} invalid samples carry a value"
            )

    n_valid = int(mask.sum())
    row["n_valid"] = n_valid
    row["valid_hours"] = n_valid / (fs * 3600.0)
    row["frac_valid"] = n_valid / signal.size if signal.size else 0.0

    valid_values = signal[mask]

    # --- C. range ---
    if valid_values.size:
        row["min_val"] = float(valid_values.min())
        row["max_val"] = float(valid_values.max())
        row["mean_val"] = float(valid_values.mean())
        row["std_val"] = float(valid_values.std())
        low, high = PLAUSIBLE_SPO2_RANGE
        if row["min_val"] < low:
            issues.append(f"min {row['min_val']:.2f} below {low}")
        if row["max_val"] > high:
            issues.append(f"max {row['max_val']:.2f} above {high}")
    else:
        issues.append("no valid samples at all")
        row["min_val"] = row["max_val"] = np.nan
        row["mean_val"] = row["std_val"] = np.nan

    # --- D. fidelity ---
    declared = xml_declared_duration(xml_path)
    row["xml_declared_sec"] = declared if declared is not None else np.nan
    if declared is not None:
        delta = abs(float(signal.size) - declared)
        row["length_vs_xml_delta_sec"] = delta
        if delta > DURATION_TOLERANCE_SEC:
            issues.append(f"length vs XML delta {delta:.0f}s")
    else:
        row["length_vs_xml_delta_sec"] = np.nan

    # --- E. analysable time ---
    row["sufficient"] = row["valid_hours"] >= MIN_VALID_HOURS
    # recorded but not counted as an issue: a short night is a limit of the
    # data, and Aim 2 deals with it by excluding the participant

    # --- F. no fabrication ---
    run_len, touches_edge = longest_constant_run_in_valid(signal, mask)
    row["longest_constant_run_sec"] = run_len
    row["constant_run_touches_edge"] = touches_edge
    row["constant_run_frac"] = run_len / signal.size if signal.size else 0.0
    if run_len > MAX_PLAUSIBLE_CONSTANT_RUN_SEC:
        issues.append(
            f"implausible constant run of {run_len}s among valid samples"
            + (" reaching a recording edge" if touches_edge else "")
        )

    # --- G. usability ---
    if row["has_events"]:
        try:
            events = pd.read_csv(events_path)
            concept = events["event_concept"].astype(str)
            is_rem = concept.str.contains("REM sleep", case=False, na=False)
            is_nrem = concept.str.match(r"Stage [1-4] sleep", case=False,
                                        na=False)
            row["n_rem_events"] = int(is_rem.sum())
            row["n_nrem_events"] = int(is_nrem.sum())
            row["n_stage_events"] = int((is_rem | is_nrem).sum())
            if row["n_stage_events"] == 0:
                issues.append("no sleep stage annotations")

            # Staging has to overlap the valid samples, or the stage-specific
            # biomarkers come out empty even though the staging and the mask
            # each look fine on their own.
            stage_mask = np.zeros(signal.size, dtype=bool)
            for _, r in events[is_rem | is_nrem].iterrows():
                a = max(int(float(r["start_sec"]) * fs), 0)
                b = min(
                    int((float(r["start_sec"]) + float(r["duration_sec"])) * fs),
                    signal.size,
                )
                if b > a:
                    stage_mask[a:b] = True
            overlap = int(np.sum(stage_mask & mask))
            row["valid_sleep_hours"] = overlap / (fs * 3600.0)
            if row["n_stage_events"] > 0 and overlap == 0:
                issues.append("sleep staging does not overlap any valid sample")
        except Exception as exc:
            issues.append(f"events unreadable ({exc})")

    row["n_issues"] = len(issues)
    row["issues"] = "; ".join(issues)
    return row


def main():
    print("=" * 78)
    print("AIM 1 COMPLETION AUDIT - CORRECTED (v2) OUTPUTS")
    print("=" * 78)

    if not os.path.isdir(SIGNAL_DIR):
        print(f"FATAL: {SIGNAL_DIR} not found. Run "
              f"aim1/reclean_with_validity_mask.py first.")
        return 1

    def ids(directory, pattern, regex):
        found = set()
        for path in glob.glob(os.path.join(directory, pattern)):
            m = re.search(regex, os.path.basename(path))
            if m:
                found.add(m.group(1))
        return found

    raw = ids(SHHS_DIR, "shhs1-*.edf", r"shhs1-(\d+)\.edf$")
    signals = ids(SIGNAL_DIR, "shhs1-*.npy", r"shhs1-(\d+)\.npy$")
    masks = ids(MASK_DIR, "shhs1-*.npy", r"shhs1-(\d+)\.npy$")
    events = ids(EVENTS_DIR, "shhs1-*_events.csv", r"shhs1-(\d+)_events\.csv$")

    all_pids = sorted(raw | signals | masks)
    print(f"\nInventory:")
    print(f"  raw EDF          : {len(raw)}")
    print(f"  v2 signals       : {len(signals)}")
    print(f"  validity masks   : {len(masks)}")
    print(f"  events tables    : {len(events)}")

    missing_signal = sorted(raw - signals)
    missing_mask = sorted(raw - masks)
    missing_events = sorted(raw - events)
    orphans = sorted((signals | masks) - raw)

    print(f"\n  raw without v2 signal : {len(missing_signal)}")
    print(f"  raw without mask      : {len(missing_mask)}")
    print(f"  raw without events    : {len(missing_events)}")
    print(f"  orphan outputs        : {len(orphans)}")

    print(f"\nAuditing {len(all_pids)} participants...")
    rows = []
    for i, pid in enumerate(all_pids, start=1):
        rows.append(audit_participant(pid))
        if i % 500 == 0:
            print(f"  {i}/{len(all_pids)}")
    audit = pd.DataFrame(rows)
    audit.to_csv(AUDIT_PATH, index=False)
    print(f"  {len(all_pids)}/{len(all_pids)} audited")
    print(f"\nAudit table: {AUDIT_PATH}")

    verdicts = {}

    print("\n" + "=" * 78)
    print("CRITERION VERDICTS")
    print("=" * 78)

    # A
    coverage_ok = not (missing_signal or missing_mask or missing_events
                       or orphans)
    verdicts["A. COVERAGE"] = coverage_ok
    print(f"\nA. COVERAGE                      -> "
          f"{'PASS' if coverage_ok else 'FAIL'}")

    # B
    inv_ok = bool(
        (audit["n_valid_not_finite"].fillna(0) == 0).all()
        and (audit["n_invalid_but_finite"].fillna(0) == 0).all()
        and (audit["mask_dtype"].dropna() == "bool").all()
    )
    verdicts["B. INVARIANTS"] = inv_ok
    print(f"\nB. INVARIANTS (signal finite exactly where mask is True)")
    print(f"   valid-but-not-finite violations  : "
          f"{int(audit['n_valid_not_finite'].fillna(0).sum())}")
    print(f"   invalid-but-valued violations    : "
          f"{int(audit['n_invalid_but_finite'].fillna(0).sum())}")
    print(f"   -> {'PASS' if inv_ok else 'FAIL'}")

    # C
    range_ok = bool(
        (audit["min_val"].dropna() >= PLAUSIBLE_SPO2_RANGE[0]).all()
        and (audit["max_val"].dropna() <= PLAUSIBLE_SPO2_RANGE[1]).all()
    )
    verdicts["C. RANGE"] = range_ok
    print(f"\nC. RANGE")
    print(f"   observed across cohort           : "
          f"{audit['min_val'].min():.2f}% to {audit['max_val'].max():.2f}%")
    print(f"   -> {'PASS' if range_ok else 'FAIL'}")

    # D
    compared = audit["length_vs_xml_delta_sec"].notna()
    fidelity_ok = bool(
        compared.any()
        and (audit.loc[compared, "length_vs_xml_delta_sec"]
             <= DURATION_TOLERANCE_SEC).all()
    )
    verdicts["D. FIDELITY"] = fidelity_ok
    print(f"\nD. FIDELITY")
    print(f"   max length disagreement vs XML   : "
          f"{audit.loc[compared, 'length_vs_xml_delta_sec'].max():.2f} s")
    print(f"   -> {'PASS' if fidelity_ok else 'FAIL'}")

    # E
    n_sufficient = int(audit["sufficient"].fillna(False).sum())
    print(f"\nE. ANALYSABLE TIME")
    print(f"   median retained                  : "
          f"{audit['valid_hours'].median():.2f} h "
          f"({audit['frac_valid'].median():.1%} of recording)")
    print(f"   worst case                       : "
          f"{audit['valid_hours'].min():.2f} h "
          f"({audit['frac_valid'].min():.1%})")
    print(f"   >= {MIN_VALID_HOURS} h                        : "
          f"{n_sufficient}/{len(audit)}")
    below = audit[~audit["sufficient"].fillna(False)]
    if len(below):
        print(f"   below threshold (excluded in Aim 2):")
        for _, r in below.nsmallest(10, "valid_hours").iterrows():
            print(f"     pid {int(r['pid'])}: {r['valid_hours']:.2f} h "
                  f"({r['frac_valid']:.1%})")
    # not a pipeline failure, so this one only reports and never fails
    verdicts["E. ANALYSABLE TIME"] = True

    # F: the regression test
    bad_runs = audit[
        audit["longest_constant_run_sec"].fillna(0)
        > MAX_PLAUSIBLE_CONSTANT_RUN_SEC
    ]
    fabrication_ok = len(bad_runs) == 0
    verdicts["F. NO FABRICATION"] = fabrication_ok
    print(f"\nF. NO FABRICATION (regression test on the forward-fill defect)")
    print(f"   longest constant run among valid samples:")
    print(f"     median  : {audit['longest_constant_run_sec'].median():.0f} s")
    print(f"     p99     : "
          f"{audit['longest_constant_run_sec'].quantile(0.99):.0f} s")
    print(f"     max     : {audit['longest_constant_run_sec'].max():.0f} s "
          f"({audit['longest_constant_run_sec'].max() / 3600:.2f} h)")
    print(f"   runs exceeding {MAX_PLAUSIBLE_CONSTANT_RUN_SEC} s        : "
          f"{len(bad_runs)}")
    n_edge = int(audit["constant_run_touches_edge"].fillna(False).sum())
    print(f"   longest run reaching a record edge: {n_edge}")
    print(f"   (the original outputs had runs to 13,562 s, every one of them")
    print(f"    terminating exactly at the recording end)")
    print(f"   -> {'PASS' if fabrication_ok else 'FAIL'}")
    if len(bad_runs):
        for _, r in bad_runs.nlargest(10, "longest_constant_run_sec").iterrows():
            print(f"     pid {int(r['pid'])}: "
                  f"{r['longest_constant_run_sec']:.0f} s "
                  f"({r['constant_run_frac']:.1%} of record)")

    # G
    staged = audit["n_stage_events"].fillna(0) > 0
    overlap_ok = (audit["valid_sleep_hours"].fillna(0) > 0)
    usability_ok = bool(staged.all() and overlap_ok.all())
    verdicts["G. USABILITY"] = usability_ok
    print(f"\nG. USABILITY FOR AIM 2")
    print(f"   with sleep staging               : "
          f"{int(staged.sum())}/{len(audit)}")
    print(f"   staging overlaps valid samples   : "
          f"{int(overlap_ok.sum())}/{len(audit)}")
    print(f"   with scored REM                  : "
          f"{int((audit['n_rem_events'].fillna(0) > 0).sum())}/{len(audit)}")
    print(f"   median valid sleep time          : "
          f"{audit['valid_sleep_hours'].median():.2f} h")
    print(f"   -> {'PASS' if usability_ok else 'FAIL'}")

    # --- final verdict ---
    print("\n" + "=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    for name, ok in verdicts.items():
        print(f"  {name:<22s} {'PASS' if ok else 'FAIL'}")

    n_with_issues = int((audit["n_issues"] > 0).sum())
    print(f"\n  participants with any issue: {n_with_issues}/{len(audit)}")
    if n_with_issues:
        counter = Counter()
        for text in audit.loc[audit["n_issues"] > 0, "issues"]:
            for part in str(text).split("; "):
                counter[re.sub(r"[\d.]+", "N", part)] += 1
        print("\n  Issue breakdown:")
        for issue, count in counter.most_common():
            print(f"    {count:>5d}  {issue}")

    all_pass = all(verdicts.values())
    if all_pass and n_with_issues == 0:
        print(
            f"\n  AIM 1 IS COMPLETE AND CORRECT for the {len(signals)}\n"
            f"  participants currently on the drive.\n"
            f"\n  Signals and masks satisfy the pairing invariant, every valid\n"
            f"  sample is physiologically plausible, lengths agree with the\n"
            f"  annotations, the forward-fill fabrication is gone, and sleep\n"
            f"  staging overlaps the measured data everywhere.\n"
            f"\n  {n_sufficient} participants carry enough analysable time for\n"
            f"  overnight biomarkers and proceed to Aim 2.\n"
            f"\n  If more participants are added later: widen PID_RANGE in\n"
            f"  aim1/reclean_with_validity_mask.py, rerun it (it skips work\n"
            f"  already done), then rerun this audit."
        )
    else:
        print("\n  Resolve the failures above before building features.")

    return 0 if (all_pass and n_with_issues == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
