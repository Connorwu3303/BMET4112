"""
Aim 3 step 1: packs the cleaned SpO2 recordings into one compact store.

Reading Aim 1's three files per participant off the external drive every epoch
is too slow, so this packs them once into arrays concatenated across
participants, small enough to hold in memory (about 0.7 GB):

  signal_q.npy  int16   SpO2 as percent times 100, 0 where not analysable
  valid.npy     uint8   1 where the sample is analysable, 0 otherwise
  stage.npy     uint8   0 unknown, 1 wake, 2 N1, 3 N2, 4 N3, 5 REM
  index.csv             pid, offset, length and summary counts per participant

Percent x 100 in int16 is exact to 0.01 percent, far finer than the oximeter's
0.78 percent step. All 5,793 participants are packed, not just the 4,352 in the
incident CVD cohort, so pretraining or a different outcome won't need a rebuild.

Outputs (processed/aim3/signal_store/): the four files above, README.txt, and
failures.csv if any participant failed

Run:
    python3 build_signal_store.py            # skips work if outputs exist
    python3 build_signal_store.py --force    # rebuild from scratch
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# stage painting and stage codes come from the Aim 2 library, so there's one
# definition of a stage label across Aim 2 and Aim 3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
from obm_library import build_stage_array  # noqa: E402

DATA_ROOT = "/Volumes/ThesisHD/Data/processed"
SIGNAL_DIR = os.path.join(DATA_ROOT, "cleaned_spo2_v2")
MASK_DIR = os.path.join(DATA_ROOT, "validity_mask")
EVENT_DIR = os.path.join(DATA_ROOT, "events")
OUT_DIR = os.path.join(DATA_ROOT, "aim3", "signal_store")

FS = 1.0            # SaO2 channel is 1 Hz throughout SHHS-1
SCALE = 100         # int16 store holds percent x 100
PID_RANGE = (200001, 205804)
CHUNK = 200         # participants per parallel batch, bounds peak memory
N_JOBS = 6          # 8 cores, 2 left for the OS and the drive


def participant_paths(pid):
    return (
        os.path.join(SIGNAL_DIR, f"shhs1-{pid}.npy"),
        os.path.join(MASK_DIR, f"shhs1-{pid}.npy"),
        os.path.join(EVENT_DIR, f"shhs1-{pid}_events.csv"),
    )


def discover_participants():
    """
    List participants that have all three Aim 1 outputs.

    Anyone with only some of the three gets reported, so the cohort can't
    quietly shrink between pipeline stages.
    """
    found, incomplete = [], []
    for pid in range(PID_RANGE[0], PID_RANGE[1] + 1):
        sig, mask, ev = participant_paths(pid)
        have = [os.path.exists(sig), os.path.exists(mask), os.path.exists(ev)]
        if all(have):
            found.append(pid)
        elif any(have):
            incomplete.append((pid, have))
    return found, incomplete


def load_one(pid):
    """
    Load one participant and return the packed arrays plus a summary row.

    On failure it returns the pid and an error message instead of raising, so
    one unreadable file can't abort a run over thousands of recordings.
    """
    sig_path, mask_path, ev_path = participant_paths(pid)
    try:
        signal = np.load(sig_path)
        mask = np.load(mask_path).astype(bool)
    except Exception as exc:
        return {"pid": pid, "error": f"load failed: {exc}"}

    if signal.shape != mask.shape:
        return {"pid": pid,
                "error": f"shape mismatch {signal.shape} vs {mask.shape}"}

    n = int(signal.shape[0])

    # Re-check the Aim 1 invariant here. Aim 1 already audited it across the
    # cohort, but a mask out of step with its signal would quietly corrupt every
    # weighted average downstream, and it's cheap to check while the arrays are
    # loaded.
    finite = np.isfinite(signal)
    if not np.array_equal(finite, mask):
        n_bad = int(np.sum(finite != mask))
        return {"pid": pid,
                "error": f"validity invariant broken at {n_bad} samples"}

    try:
        events = pd.read_csv(ev_path)
    except Exception as exc:
        return {"pid": pid, "error": f"events unreadable: {exc}"}

    stages = build_stage_array(events, n, FS).astype(np.uint8)

    # Quantise to int16. Invalid samples are written as 0, outside the plausible
    # 50 to 100 percent range, so a bug that ignored the mask would give an
    # obviously absurd value.
    q = np.zeros(n, dtype=np.int16)
    vals = signal[mask]
    if vals.size:
        if vals.min() < 0 or vals.max() > 327:
            return {"pid": pid,
                    "error": f"values out of int16 range: "
                             f"{vals.min()} to {vals.max()}"}
        q[mask] = np.rint(vals * SCALE).astype(np.int16)

    row = {
        "pid": pid,
        "length": n,
        "n_valid": int(mask.sum()),
        "valid_hours": float(mask.sum() / FS / 3600.0),
        "hours_total": float(n / FS / 3600.0),
        # Sleep is defined as in Aim 2: any scored stage other than wake or
        # unknown. Kept per participant so the store can be sanity checked
        # against the Aim 2 features.
        "n_sleep": int(np.isin(stages, (2, 3, 4, 5)).sum()),
        "n_rem": int((stages == 5).sum()),
        "n_stage_unknown": int((stages == 0).sum()),
        "error": "",
    }
    return {"row": row, "q": q, "valid": mask.view(np.uint8), "stage": stages,
            "pid": pid, "error": ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if the store already exists")
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    index_path = os.path.join(OUT_DIR, "index.csv")
    if os.path.exists(index_path) and not args.force:
        existing = pd.read_csv(index_path)
        print(f"store already present with {len(existing)} participants; "
              f"pass --force to rebuild")
        return 0

    t0 = time.time()
    pids, incomplete = discover_participants()
    print(f"participants with all three Aim 1 outputs: {len(pids)}")
    if incomplete:
        print(f"WARNING: {len(incomplete)} participants have partial outputs "
              f"and were skipped, first few: {incomplete[:5]}")

    # Two passes. The first reads each recording's length so the offsets into
    # the concatenated store are known exactly, the second fills the arrays.
    # Reading a .npy header is much cheaper than reading the data, so pass 1 is
    # quick and nothing has to grow as we go.
    print("pass 1: reading lengths from .npy headers")
    lengths = {}
    for pid in pids:
        sig_path, _, _ = participant_paths(pid)
        try:
            lengths[pid] = int(np.load(sig_path, mmap_mode="r").shape[0])
        except Exception as exc:
            print(f"  pid {pid}: header unreadable ({exc}), will be skipped")
    pids = [p for p in pids if p in lengths]
    total = int(sum(lengths.values()))
    print(f"  {len(pids)} participants, {total:,} samples, "
          f"{total / 3600.0:,.0f} recording hours")
    print(f"  store size: signal {total * 2 / 1e9:.2f} GB, "
          f"mask {total / 1e9:.2f} GB, stage {total / 1e9:.2f} GB")

    signal_store = np.zeros(total, dtype=np.int16)
    valid_store = np.zeros(total, dtype=np.uint8)
    stage_store = np.zeros(total, dtype=np.uint8)

    print(f"pass 2: packing with {args.jobs} workers")
    rows, failures = [], []
    cursor = 0
    for start in range(0, len(pids), CHUNK):
        batch = pids[start:start + CHUNK]
        results = Parallel(n_jobs=args.jobs, backend="loky")(
            delayed(load_one)(pid) for pid in batch
        )
        for res in results:
            if res is None or res.get("error"):
                failures.append({"pid": res.get("pid") if res else None,
                                 "error": res.get("error") if res else "none"})
                continue
            n = res["row"]["length"]
            stop = cursor + n
            signal_store[cursor:stop] = res["q"]
            valid_store[cursor:stop] = res["valid"]
            stage_store[cursor:stop] = res["stage"]
            res["row"]["offset"] = cursor
            rows.append(res["row"])
            cursor = stop
        done = start + len(batch)
        print(f"  {done}/{len(pids)} participants, "
              f"{cursor:,}/{total:,} samples, {time.time() - t0:.0f}s",
              flush=True)

    if failures:
        print(f"WARNING: {len(failures)} participants failed:")
        for f in failures[:10]:
            print(f"    pid {f['pid']}: {f['error']}")

    # Trim to what was actually written. A failure leaves the tail unused, and
    # keeping it would make the arrays disagree with the index.
    if cursor != total:
        print(f"trimming store from {total:,} to {cursor:,} samples "
              f"after {len(failures)} failures")
        signal_store = signal_store[:cursor]
        valid_store = valid_store[:cursor]
        stage_store = stage_store[:cursor]

    index = pd.DataFrame(rows)[[
        "pid", "offset", "length", "n_valid", "valid_hours", "hours_total",
        "n_sleep", "n_rem", "n_stage_unknown",
    ]].sort_values("pid").reset_index(drop=True)

    np.save(os.path.join(OUT_DIR, "signal_q.npy"), signal_store)
    np.save(os.path.join(OUT_DIR, "valid.npy"), valid_store)
    np.save(os.path.join(OUT_DIR, "stage.npy"), stage_store)
    index.to_csv(index_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(
            os.path.join(OUT_DIR, "failures.csv"), index=False)

    with open(os.path.join(OUT_DIR, "README.txt"), "w") as fh:
        fh.write(
            "Aim 3 packed SpO2 signal store\n"
            "==============================\n\n"
            f"participants       {len(index)}\n"
            f"samples            {cursor}\n"
            f"sampling rate      {FS} Hz\n"
            f"signal_q.npy       int16, percent times {SCALE}, "
            "0 where not analysable\n"
            "valid.npy          uint8, 1 where analysable\n"
            "stage.npy          uint8, 0 unknown 1 wake 2 N1 3 N2 4 N3 5 REM\n"
            "index.csv          pid, offset, length and per-night summaries\n\n"
            "Read participant i as signal_q[offset : offset + length] / "
            f"{SCALE}, valid only where valid[...] == 1.\n"
        )

    print(f"\nwrote store to {OUT_DIR}")
    print(f"  {len(index)} participants, {cursor:,} samples, "
          f"{cursor / 3600.0:,.0f} hours, {time.time() - t0:.0f}s total")
    print(f"  median analysable hours {index['valid_hours'].median():.2f}, "
          f"range {index['valid_hours'].min():.2f} to "
          f"{index['valid_hours'].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
