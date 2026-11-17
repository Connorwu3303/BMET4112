"""
Aim 3 stage 0: label free gates that have to pass before the outcome is used.

Every fix made because the CVD numbers looked wrong leaks some of the outcome
into the design, so the pipeline is debugged here on targets built from the
signal itself, or on permuted labels:

  1  ODI3 regression: arm A recovers ODI3 from the waveform (bar r >= 0.9)
  2  learning rate sensitivity: checks the pre-declared 1e-3 is in a stable region
  3  positive control: full classification path on moderate or severe ODI3
  4  permuted control: same path on permuted CVD labels, expecting AUC near 0.5

Gates 1 and 2 use only the 1,441 participants with a recording but no usable
outcome, so they never appear in any CVD fit. Gates 3 and 4 are actually run by
run_gates_classification.sh.

Outputs (processed/aim3/results/): gate_odi3.json, gate_lr.json

Usage:
    python3 gates.py --gate {odi3,lr,positive,permuted}
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D            # noqa: E402
import models as M          # noqa: E402
import trainer as T         # noqa: E402
from folds import get_verified_folds  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
FEATURES = "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv"
SEED = 42


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def no_outcome_pids(store, data):
    """
    The 1,441 participants with a recording but no usable incident outcome.

    They had CVD at baseline or a missing outcome, so Aim 2 gave them NaN
    instead of 0 and they aren't in any fold. That's what makes them safe to
    debug on.
    """
    cohort = set(int(p) for p in data["pid"])
    return np.array(sorted(int(p) for p in store.index["pid"]
                           if int(p) not in cohort))


def odi3_targets(pids):
    """log1p of sleep ODI3 for each pid (callers standardise it on train rows)."""
    f = pd.read_csv(FEATURES, low_memory=False,
                    usecols=["pid", "sleep_odi3"]).set_index("pid")
    vals = f.loc[[int(p) for p in pids], "sleep_odi3"].to_numpy(dtype=float)
    return np.log1p(np.clip(vals, 0, None))


def run_odi3(store, data, epochs=12, lr=1e-3, batch=32, arm="A", quiet=False):
    """
    Gate 1: checks the network can read ODI3 off the waveform.

    Returns the validation Pearson r for each epoch and the best one.
    """
    pids = no_outcome_pids(store, data)
    raw = odi3_targets(pids)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pids))
    n_val = int(0.24 * len(pids))
    va, tr = order[:n_val], order[n_val:]

    # Standardise on training rows only. Even in a label free gate, using
    # validation stats is a habit that could get copy-pasted into the real
    # pipeline.
    mu, sd = raw[tr].mean(), raw[tr].std()
    target = (raw - mu) / max(sd, 1e-9)

    print(f"  ODI3 gate: {len(tr)} train, {len(va)} validation "
          f"(of {len(pids)} participants with no usable outcome)")
    print(f"  raw sleep ODI3: median {np.expm1(np.median(raw)):.1f} per hour, "
          f"range {np.expm1(raw.min()):.1f} to {np.expm1(raw.max()):.1f}")

    tl, tds = D.make_loader(store, pids[tr], target[tr],
                            arm="primary",
                            train=True, batch_size=batch, seed=SEED)
    vl, _ = D.make_loader(store, pids[va], target[va], arm="primary",
                          train=False, batch_size=16, seed=SEED)

    net = M.build(arm)
    res = T.train_unit(net, tl, tds, vl, device(), epochs=epochs, lr=lr,
                       loss_kind="huber", verbose=not quiet)
    rs = [h["val_metric"] for h in res["history"]]
    return {"lr": lr, "epochs": epochs, "r_per_epoch": rs,
            "best_r": float(np.nanmax(rs)),
            "best_epoch": int(np.nanargmax(rs)),
            "seconds": res["seconds"],
            "seconds_per_epoch": res["seconds"] / max(len(rs), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True,
                    choices=["odi3", "lr", "positive", "permuted"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print(f"AIM 3 LABEL FREE GATE: {args.gate}")
    print("=" * 72)
    dev = device()
    print(f"device {dev}, torch {torch.__version__}")

    t0 = time.time()
    data = get_verified_folds(quiet=True)
    store = D.SignalStore()
    print(f"store and folds loaded in {time.time() - t0:.1f}s\n")

    if args.gate == "odi3":
        res = run_odi3(store, data, epochs=args.epochs, batch=args.batch)
        print(f"\n  best validation Pearson r = {res['best_r']:.4f} "
              f"at epoch {res['best_epoch']}")
        print(f"  {res['seconds_per_epoch']:.1f}s per epoch, "
              f"{res['seconds'] / 60:.1f} min total")
        # bar set from OxiNet, which reached an intraclass correlation of 0.96
        # for AHI from the same channel
        verdict = "PASS" if res["best_r"] >= 0.9 else "FAIL"
        print(f"\n  GATE {verdict}: the bar is r of at least 0.90")
        if verdict == "FAIL":
            print("  Do not start any cardiovascular run. A network that cannot "
                  "recover a biomarker computed from its own input has a "
                  "pipeline fault, not a weak label.")
        with open(os.path.join(RESULTS, "gate_odi3.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        return 0 if verdict == "PASS" else 1

    if args.gate == "lr":
        out = []
        for lr in (3e-4, 1e-3, 3e-3):
            print(f"\n  learning rate {lr:g}")
            out.append(run_odi3(store, data, epochs=max(6, args.epochs // 2),
                                lr=lr, batch=args.batch, quiet=True))
            print(f"    best r {out[-1]['best_r']:.4f} at epoch "
                  f"{out[-1]['best_epoch']}, "
                  f"{out[-1]['seconds_per_epoch']:.1f}s per epoch")
        best = max(out, key=lambda r: r["best_r"])
        print(f"\n  best learning rate on the label free task: {best['lr']:g} "
              f"(r {best['best_r']:.4f})")
        chosen = [r for r in out if abs(r["lr"] - 1e-3) < 1e-12][0]
        margin = best["best_r"] - chosen["best_r"]
        print(f"  the pre-declared 1e-3 reaches r {chosen['best_r']:.4f}, "
              f"{margin:.4f} below the best of the three")
        print(f"  VERDICT: {'1e-3 is in a stable region, keeping it' if margin < 0.05 else 'consider the better rate, and record the change'}")
        with open(os.path.join(RESULTS, "gate_lr.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        return 0

    print(f"gate {args.gate} is run by run_gates_classification.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
