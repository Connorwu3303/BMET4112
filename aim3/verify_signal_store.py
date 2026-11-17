"""
Aim 3 step 3: checks the packed signal store independently of how it was built.

A packing bug wouldn't crash anything. Training would just model the wrong data.
So a few biomarkers are recomputed from the store with plain numpy and compared,
participant by participant, with the Aim 2 feature table, which came from
separate code on the original files.

Checks:
  A  structural: lengths match the index, offsets tile, values are plausible
  B  quantisation: on a random sample, round trip error against the original
     files is at most half an int16 step
  C  fidelity: recording mean, minimum, T90 and sleep mean match Aim 2
  D  stages: sleep and REM sample counts match the Aim 2 window sizes
  E  alignment: shifting participants by one must break the C agreement
  F  cohort: every participant in the incident CVD cohort is present, with at
     least 4 h of analysable signal

Output: processed/aim3/signal_store/store_verification.csv (exit code 1 if any
check fails)

Usage (from aim3/):
    python3 verify_signal_store.py
"""

import os
import sys

import numpy as np
import pandas as pd

DATA_ROOT = "/Volumes/ThesisHD/Data/processed"
STORE_DIR = os.path.join(DATA_ROOT, "aim3", "signal_store")
SIGNAL_DIR = os.path.join(DATA_ROOT, "cleaned_spo2_v2")
MASK_DIR = os.path.join(DATA_ROOT, "validity_mask")
FEATURES_PATH = os.path.join(DATA_ROOT, "aim2", "obm_features.csv")
SCALE = 100
RNG = np.random.default_rng(42)

results = []


def check(name, passed, detail):
    results.append({"check": name, "status": "PASS" if passed else "FAIL",
                    "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return passed


def load_store():
    index = pd.read_csv(os.path.join(STORE_DIR, "index.csv"))
    signal = np.load(os.path.join(STORE_DIR, "signal_q.npy"))
    valid = np.load(os.path.join(STORE_DIR, "valid.npy"))
    stage = np.load(os.path.join(STORE_DIR, "stage.npy"))
    return index, signal, valid, stage


def get(index_row, signal, valid, stage):
    """Return (spo2_percent, valid_bool, stage_codes) for one index row."""
    a, b = int(index_row["offset"]), int(index_row["offset"] + index_row["length"])
    v = valid[a:b].astype(bool)
    return signal[a:b].astype(np.float64) / SCALE, v, stage[a:b]


def recompute(index, signal, valid, stage):
    """
    Recompute four simple biomarkers per participant from scratch.

    Plain numpy on purpose. Reusing obm_library would only show that the same
    code gives the same answer twice, which says nothing about whether the store
    holds the right data.
    """
    rows = []
    for _, r in index.iterrows():
        x, v, st = get(r, signal, valid, stage)
        xv = x[v]
        sleep = v & np.isin(st, (2, 3, 4, 5))
        rem = v & (st == 5)
        rows.append({
            "pid": int(r["pid"]),
            "rec_mean": float(xv.mean()) if xv.size else np.nan,
            "rec_min": float(xv.min()) if xv.size else np.nan,
            # T90 as a percentage of analysable time, same as Aim 2 (valid time
            # is the denominator)
            "rec_ct90": float(100.0 * np.mean(xv < 90.0)) if xv.size else np.nan,
            "sleep_mean": float(x[sleep].mean()) if sleep.sum() else np.nan,
            "n_sleep": int(sleep.sum()),
            "n_rem": int(rem.sum()),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("AIM 3 SIGNAL STORE VERIFICATION")
    print("=" * 72)

    index, signal, valid, stage = load_store()
    print(f"\nstore: {len(index)} participants, {len(signal):,} samples, "
          f"{len(signal) / 3600.0:,.0f} hours")

    print("\nA. structural integrity")
    total = int(index["length"].sum())
    check("A1 array lengths agree with index",
          len(signal) == total and len(valid) == total and len(stage) == total,
          f"signal {len(signal):,}, valid {len(valid):,}, stage {len(stage):,}, "
          f"index sum {total:,}")

    # offsets have to tile the store exactly: sorted by offset, each participant
    # starts where the previous one ended
    srt = index.sort_values("offset")
    expected = np.concatenate([[0], np.cumsum(srt["length"].to_numpy())[:-1]])
    check("A2 offsets tile the store with no gaps or overlaps",
          np.array_equal(srt["offset"].to_numpy(), expected),
          f"{len(srt)} contiguous blocks" if
          np.array_equal(srt["offset"].to_numpy(), expected) else
          f"first mismatch at block "
          f"{int(np.argmax(srt['offset'].to_numpy() != expected))}")

    nz = signal[signal != 0]
    check("A3 stored values are plausible saturations",
          nz.size > 0 and nz.min() >= 50 * SCALE and nz.max() <= 100 * SCALE,
          f"non-zero range {nz.min() / SCALE:.2f} to {nz.max() / SCALE:.2f} "
          f"percent")

    check("A4 signal is zero exactly where the mask is false",
          bool(np.all(signal[valid == 0] == 0)) and
          bool(np.all(signal[valid == 1] != 0)),
          f"{int(np.sum(signal[valid == 0] != 0))} invalid samples non-zero, "
          f"{int(np.sum(signal[valid == 1] == 0))} valid samples zero")

    check("A5 stage codes are in range",
          bool(np.all(np.isin(np.unique(stage), (0, 1, 2, 3, 4, 5)))),
          f"codes present {sorted(np.unique(stage).tolist())}")

    print("\nB. quantisation round trip against the original files")
    sample = index.sample(n=min(60, len(index)), random_state=42)
    worst, worst_pid, n_len_bad, n_mask_bad = 0.0, None, 0, 0
    for _, r in sample.iterrows():
        pid = int(r["pid"])
        orig = np.load(os.path.join(SIGNAL_DIR, f"shhs1-{pid}.npy"))
        omask = np.load(os.path.join(MASK_DIR, f"shhs1-{pid}.npy")).astype(bool)
        x, v, _ = get(r, signal, valid, stage)
        if len(orig) != len(x):
            n_len_bad += 1
            continue
        if not np.array_equal(omask, v):
            n_mask_bad += 1
            continue
        if omask.any():
            err = float(np.max(np.abs(orig[omask] - x[v])))
            if err > worst:
                worst, worst_pid = err, pid
    # Half an int16 step at scale 100 is 0.005 percent, 1/156 of the 0.78
    # percent hardware quantum, so this tolerance is much tighter than the
    # oximeter's own precision.
    check("B1 lengths match the original files", n_len_bad == 0,
          f"{n_len_bad} of {len(sample)} sampled participants differ in length")
    check("B2 masks match the original files", n_mask_bad == 0,
          f"{n_mask_bad} of {len(sample)} sampled participants differ in mask")
    check("B3 round trip error within half a quantisation step",
          worst <= 0.005 + 1e-12,
          f"worst absolute error {worst:.6f} percent "
          f"(pid {worst_pid}) across {len(sample)} participants")

    print("\nC. fidelity against the Aim 2 biomarker table")
    print("   (recomputing from the store, this takes a minute)")
    mine = recompute(index, signal, valid, stage)
    feats = pd.read_csv(FEATURES_PATH, low_memory=False)
    merged = mine.merge(feats, on="pid", how="inner", suffixes=("", "_aim2"))
    check("C0 participants merged with Aim 2 features", len(merged) > 5000,
          f"{len(merged)} of {len(index)} store participants matched")

    pairs = [
        ("rec_mean", "recording_mean", "recording mean saturation"),
        ("rec_min", "recording_min", "recording minimum saturation"),
        ("rec_ct90", "recording_ct90_pct", "recording T90 percent"),
        ("sleep_mean", "sleep_mean", "sleep mean saturation"),
    ]
    for ours, theirs, label in pairs:
        if theirs not in merged.columns:
            check(f"C {label}", False, f"Aim 2 column {theirs} not found")
            continue
        ok = merged[ours].notna() & merged[theirs].notna()
        a, b = merged.loc[ok, ours].to_numpy(), merged.loc[ok, theirs].to_numpy()
        r = float(np.corrcoef(a, b)[0, 1])
        mad = float(np.max(np.abs(a - b)))
        # Same arithmetic on the same samples, so anything worse than
        # r = 0.9999 means the store and the features disagree about which
        # samples belong to which participant.
        check(f"C {label}", r > 0.9999 and mad < 0.05,
              f"r = {r:.6f}, worst absolute difference {mad:.4f}, n = {ok.sum()}")

    print("\nD. sleep staging")
    # Aim 2 reports window duration in analysable hours, so the store's sample
    # counts are converted at 1 Hz. Agreement means the store's stage painting
    # puts the same seconds in the same window as Aim 2 did.
    for ours, theirs, label in [("n_sleep", "sleep_valid_hours", "sleep hours"),
                                ("n_rem", "rem_valid_hours", "REM hours")]:
        if theirs not in merged.columns:
            check(f"D {label}", False, f"Aim 2 column {theirs} not found")
            continue
        ok = merged[theirs].notna()
        a = merged.loc[ok, ours].to_numpy(dtype=float) / 3600.0
        b = merged.loc[ok, theirs].to_numpy(dtype=float)
        r = float(np.corrcoef(a, b)[0, 1])
        mad = float(np.max(np.abs(a - b)))
        check(f"D {label}", r > 0.9999 and mad < 1e-6,
              f"r = {r:.6f}, worst absolute difference {mad:.2e} hours, "
              f"n = {int(ok.sum())}")

    print("\nE. positive control: misalignment must be detectable")
    # Shift the store rows by one participant and check the correlation
    # collapses. Without this, a pass in section C could just mean the
    # comparison has no power.
    shifted = mine.copy()
    shifted["pid"] = np.roll(shifted["pid"].to_numpy(), 1)
    sm = shifted.merge(feats[["pid", "recording_mean"]], on="pid", how="inner")
    ok = sm["rec_mean"].notna() & sm["recording_mean"].notna()
    r_shift = float(np.corrcoef(sm.loc[ok, "rec_mean"],
                                sm.loc[ok, "recording_mean"])[0, 1])
    check("E1 shifting participants by one destroys agreement",
          abs(r_shift) < 0.5,
          f"r falls to {r_shift:.4f} under a one participant shift, "
          f"so section C has power to detect misalignment")

    print("\nF. cohort coverage")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from folds import load_cohort  # noqa: E402
    pid_cohort, y, _, _, _ = load_cohort()
    missing = set(pid_cohort.tolist()) - set(index["pid"].tolist())
    check("F1 every modelled participant is in the store", not missing,
          f"cohort {len(pid_cohort)} participants, {len(missing)} missing")

    have = index[index["pid"].isin(pid_cohort)]
    check("F2 modelled participants all have analysable signal",
          bool((have["n_valid"] > 0).all()) and
          bool((have["valid_hours"] >= 4.0).all()),
          f"minimum analysable hours {have['valid_hours'].min():.2f}, "
          f"median {have['valid_hours'].median():.2f}")

    print("\n" + "=" * 72)
    table = pd.DataFrame(results)
    n_pass = int((table["status"] == "PASS").sum())
    print(f"RESULT: {n_pass} of {len(table)} checks passed")
    if n_pass < len(table):
        print("\nFAILED CHECKS:")
        for _, r in table[table["status"] == "FAIL"].iterrows():
            print(f"  {r['check']}: {r['detail']}")
    print("=" * 72)

    out = os.path.join(STORE_DIR, "store_verification.csv")
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0 if n_pass == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
