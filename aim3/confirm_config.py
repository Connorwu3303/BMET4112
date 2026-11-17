"""
Aim 3: rechecks the config adopted by the search at the real epoch budget.

The search (optimise.py) uses 10 epoch trials, because 18 trials at the full
budget would cost too much. Every trial gets the same budget, so the ranking is
fair for a 10 epoch schedule. The real runs use a 16 epoch cap with a cosine
schedule over the cap, so a 10 epoch trial decays faster than the first 10
epochs of a real run. Learning rate especially depends on schedule length, since
a shorter schedule decays faster and tolerates a higher peak.

So the winner and the pre-registered config are each trained once at 16 epochs
on the same label free task (ODI3 on the 1,441 participants who aren't in any
CVD fold). If the winner doesn't come out ahead, its advantage came from the
shorter schedule and optimisation_result.json is reverted to the pre-registered
config. It's only two runs, done before the long training starts.

Outputs (processed/aim3/results/): config_confirmation.json, and
optimisation_result.json is rewritten if the choice is reverted

Usage:
    python3 confirm_config.py --epochs 16
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D            # noqa: E402
import models as M          # noqa: E402
import trainer as T         # noqa: E402
import run_unit as RU       # noqa: E402
from folds import get_verified_folds  # noqa: E402
from gates import no_outcome_pids, odi3_targets  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
SEED = 42


def run_one(store, pids, target, tr, va, cfg, device, epochs, label):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tl, tds = D.make_loader(store, pids[tr], target[tr], arm="primary",
                            train=True, batch_size=32, seed=SEED)
    vl, _ = D.make_loader(store, pids[va], target[va], arm="primary",
                          train=False, batch_size=16, seed=SEED)
    net = M.build("A", overrides=RU.model_overrides(cfg))
    print(f"\n  {label}: {M.count_parameters(net):,} parameters, "
          f"lr {cfg['lr']:g}, weight decay {cfg['weight_decay']:g}, "
          f"dropout {cfg['dropout']:g}/{cfg['head_dropout']:g}, "
          f"width x{cfg['width_mult']:g}, k {cfg['k_fraction']:g}", flush=True)
    res = T.train_unit(net, tl, tds, vl, device, epochs=epochs,
                       lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                       loss_kind="huber", verbose=False)
    rs = [h["val_metric"] for h in res["history"]]
    print(f"    best r {np.nanmax(rs):.4f} at epoch {int(np.nanargmax(rs))}, "
          f"final r {rs[-1]:.4f}, {res['seconds'] / 60:.1f} min")
    return {"label": label, "config": cfg, "best_r": float(np.nanmax(rs)),
            "final_r": float(rs[-1]), "best_epoch": int(np.nanargmax(rs)),
            "r_per_epoch": rs, "minutes": res["seconds"] / 60.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 76)
    print("CONFIRMING THE ADOPTED CONFIGURATION AT THE PRODUCTION BUDGET")
    print("=" * 76)
    print(f"device {device}, {args.epochs} epochs, the production cap")

    data = get_verified_folds(quiet=True)
    store = D.SignalStore()
    pids = no_outcome_pids(store, data)
    raw = odi3_targets(pids)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pids))
    n_val = int(0.24 * len(pids))
    va, tr = order[:n_val], order[n_val:]
    mu, sd = raw[tr].mean(), raw[tr].std()
    target = (raw - mu) / max(sd, 1e-9)

    adopted, source = RU.resolve_config()
    prereg = dict(RU.PREREGISTERED)

    if adopted == prereg:
        print("\nThe optimisation confirmed the pre-registered configuration, so "
              "there is nothing to compare. Nothing to do.")
        return 0

    out = [run_one(store, pids, target, tr, va, prereg, device, args.epochs,
                   "pre-registered"),
           run_one(store, pids, target, tr, va, adopted, device, args.epochs,
                   "adopted by the search")]

    gain = out[1]["best_r"] - out[0]["best_r"]
    print("\n" + "-" * 76)
    print(f"  pre-registered      r = {out[0]['best_r']:.4f}")
    print(f"  adopted by search   r = {out[1]['best_r']:.4f}")
    print(f"  difference at the production budget: {gain:+.4f}")

    # Decision rule, set before seeing the numbers: keep the search's choice only
    # if it also wins at the full budget. Otherwise its edge came from the
    # shorter schedule, so go back to the pre-registered config (the
    # conservative default).
    keep = gain > 0
    print(f"\n  DECISION: {'keep the adopted configuration' if keep else 'REVERT to the pre-registered configuration'}")
    if not keep:
        print("  The search's advantage did not survive the production epoch "
              "budget, so it was an artefact of the shorter schedule.")

    result = {"epochs": args.epochs, "runs": out, "gain": gain,
              "keep_adopted": bool(keep), "source": source}
    with open(os.path.join(RESULTS, "config_confirmation.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    if not keep:
        # write the pre-registered config back as "adopted" so run_unit picks it
        # up and every unit records the revert
        path = RU.OPTIMISED_PATH
        with open(path) as fh:
            res = json.load(fh)
        res["adopted_before_confirmation"] = res.get("adopted")
        res["adopted"] = prereg
        res.setdefault("notes", []).append(
            f"REVERTED to the pre-registered configuration: the search's choice "
            f"scored {gain:+.4f} against it at the production budget of "
            f"{args.epochs} epochs, so its advantage at 10 epochs was an "
            f"artefact of the shorter schedule.")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"  updated {path}")

    print(f"\nwrote {RESULTS}/config_confirmation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
