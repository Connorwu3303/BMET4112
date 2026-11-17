"""
Aim 3: trains one unit in its own process and saves everything.

A unit is one (arm, repeat, outer fold, inner model). Each writes its own
folder and skips itself if that folder is already complete, so the full run is
a shell loop that can be stopped and restarted (the external drive has
unmounted mid run before). A fresh process per unit also resets the GPU
allocator, which slowed down badly when several models shared one process.

Nothing is selected here. The unit trains to the fixed epoch cap and saves the
validation logits and EMA weights for every epoch; select_epoch.py then picks
the epoch and scores the outer test fold once.

Outputs (processed/aim3/results/units/<arm>_r<repeat>_f<fold>_i<inner>/):
    val_logits.npz, ema_epochNN.pt for every epoch, meta.json

Usage (one arm and repeat, as a resumable shell loop):
    for f in 0 1 2 3 4; do for i in 0 1 2; do
        python3 run_unit.py --arm A --repeat 0 --fold $f --inner $i
    done; done
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D            # noqa: E402
import models as M          # noqa: E402
import trainer as T         # noqa: E402
from folds import get_verified_folds  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
UNITS = os.path.join(RESULTS, "units")

# Fixed before the first run that used the CVD labels. Changing them after that
# would break the pre-registration, so they're kept in one place and saved into
# every unit's meta.json.
EPOCH_CAP = 16
LR = 1e-3
WEIGHT_DECAY = 1e-2
BATCH = 32
EMA_DECAY = 0.99
GRAD_CLIP = 1.0

# The config actually used. It starts as the pre-registered one and only gets
# replaced by the config adopted from the label free optimisation (optimise.py),
# if that has been run. It's read from a file so the pre-registered values above
# stay visible in the source, and each unit records which of the two it used.
OPTIMISED_PATH = os.path.join(RESULTS, "optimisation_result.json")

PREREGISTERED = {"lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": 0.1,
                 "head_dropout": 0.5, "width_mult": 1.0, "k_fraction": 0.10}


def widths_for(mult):
    """Scales the network widths by mult, keeping the same shape."""
    base = (16, 32, 48, 64, 64)
    return (tuple(max(4, int(round(c * mult))) for c in base),
            max(4, int(round(64 * mult))), max(4, int(round(16 * mult))))


def resolve_config(use_optimised=True):
    """
    Works out which hyperparameters to use and where they came from.

    Returns (config, source). source goes into every unit's meta.json so it's
    always clear which config produced a result.
    """
    if use_optimised and os.path.exists(OPTIMISED_PATH):
        try:
            with open(OPTIMISED_PATH) as fh:
                res = json.load(fh)
            cfg = dict(PREREGISTERED)
            cfg.update({k: v for k, v in res.get("adopted", {}).items()
                        if k in PREREGISTERED})
            if cfg != PREREGISTERED:
                return cfg, "label-free optimisation (optimisation_result.json)"
            return cfg, ("label-free optimisation confirmed the pre-registered "
                         "configuration")
        except Exception as exc:
            print(f"  WARNING: could not read {OPTIMISED_PATH} ({exc}); "
                  f"falling back to the pre-registered configuration")
    return dict(PREREGISTERED), "pre-registered"


def model_overrides(cfg, arm="A"):
    """
    Maps a config onto the arguments a given arm actually takes.

    This has to be arm aware. The primary network and the window model both
    take "widths" but they mean different things (five strided stages against
    four window encoder stages), and passing the primary's widths to the window
    model once quietly took it from 36,099 to 65,187 parameters. So each arm
    scales its own base widths and only gets the parameters it has.
    """
    mult = cfg["width_mult"]
    if arm in ("A", "D", "NCA"):
        w, dil, stem = widths_for(mult)
        return {"widths": w, "dilated_width": dil, "stem_width": stem,
                "dropout": cfg["dropout"], "head_dropout": cfg["head_dropout"],
                "k_fraction": cfg["k_fraction"]}
    if arm in ("B", "NC"):
        base = (16, 32, 48, 48)
        w = tuple(max(4, int(round(c * mult))) for c in base)
        return {"widths": w,
                "context_width": max(4, int(round(48 * mult))),
                "dropout": cfg["head_dropout"]}
    raise ValueError(f"no override mapping for arm {arm!r}")


# seed comes from the unit ids, so each unit is reproducible on its own and no
# two units share an initialisation
def unit_seed(arm, repeat, fold, inner):
    base = {"A": 0, "B": 1, "NC": 2, "D": 3, "NCA": 4}.get(arm, 9)
    return 10000 + 1000 * repeat + 100 * fold + 10 * inner + base


def unit_dir(arm, repeat, fold, inner):
    return os.path.join(UNITS, f"{arm}_r{repeat}_f{fold}_i{inner}")


def is_complete(path, epochs):
    """A unit is complete when it has logits for every epoch and all checkpoints."""
    logits = os.path.join(path, "val_logits.npz")
    if not os.path.exists(logits):
        return False
    try:
        with np.load(logits) as z:
            if z["logits"].shape[0] != epochs:
                return False
    except Exception:
        return False
    return all(os.path.exists(os.path.join(path, f"ema_epoch{e:02d}.pt"))
               for e in range(epochs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(M.ARMS))
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--inner", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCH_CAP)
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate; normally left unset so "
                         "the resolved configuration is used")
    ap.add_argument("--preregistered", action="store_true",
                    help="ignore the label free optimisation and use the "
                         "pre-registered configuration, for a sensitivity arm")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--tag", default="",
                    help="suffix for a sensitivity run, keeping it separate")
    ap.add_argument("--permute-labels", action="store_true",
                    help="label free control: permute the outcome with a fixed "
                         "seed, so nothing real is observed")
    ap.add_argument("--odi3-label", action="store_true",
                    help="label free positive control: predict a moderate or "
                         "severe desaturation index instead of the outcome")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = unit_dir(args.arm, args.repeat, args.fold, args.inner) + args.tag
    if is_complete(out, args.epochs) and not args.force:
        print(f"unit {os.path.basename(out)} already complete, skipping")
        return 0
    os.makedirs(out, exist_ok=True)

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    data = get_verified_folds(quiet=True)
    store = D.SignalStore()
    y = data["y"].copy()
    pid = data["pid"]

    label_note = "cvd_incident"
    if args.permute_labels:
        # fixed seed. Every later stage is scored against these permuted labels,
        # so no real association is ever seen
        y = np.random.default_rng(20260928).permutation(y)
        label_note = "PERMUTED cvd_incident (label free control)"
    elif args.odi3_label:
        import pandas as pd
        f = pd.read_csv(
            "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv",
            low_memory=False, usecols=["pid", "sleep_odi3"]).set_index("pid")
        odi = f.loc[[int(p) for p in pid], "sleep_odi3"].to_numpy(dtype=float)
        y = (odi >= 15.0).astype(int)
        label_note = "1[sleep_odi3 >= 15] (label free positive control)"

    outer = [f for f in data["folds"]
             if f["repeat"] == args.repeat and f["fold"] == args.fold]
    if not outer:
        raise SystemExit(f"no fold for repeat {args.repeat} fold {args.fold}")
    outer = outer[0]
    splits = T.inner_splits(outer["train_idx"], y)
    if not 0 <= args.inner < len(splits):
        raise SystemExit(f"inner must be 0 to {len(splits) - 1}")
    itr, iva = splits[args.inner]

    seed = unit_seed(args.arm, args.repeat, args.fold, args.inner)
    torch.manual_seed(seed)
    np.random.seed(seed)

    arm_input = M.arm_input(args.arm)
    print(f"unit {os.path.basename(out)}  arm {args.arm} "
          f"(input {arm_input}, {M.ARMS[args.arm]['channels']} channel(s))")
    print(f"  label   : {label_note}")
    print(f"  device  : {dev}, seed {seed}")
    print(f"  inner   : {len(itr)} train ({int(y[itr].sum())} events), "
          f"{len(iva)} validation ({int(y[iva].sum())} events)")
    print(f"  outer   : {len(outer['train_idx'])} training, "
          f"{len(outer['test_idx'])} test (not scored here)")
    print(f"  setup   : {time.time() - t0:.1f}s", flush=True)

    train_loader, train_ds = D.make_loader(
        store, pid[itr], y[itr], arm=arm_input, train=True,
        batch_size=args.batch, seed=seed)
    val_loader, _ = D.make_loader(
        store, pid[iva], y[iva], arm=arm_input, train=False,
        batch_size=16, seed=seed)

    cfg, cfg_source = resolve_config(use_optimised=not args.preregistered)
    if args.lr is not None:
        cfg["lr"] = args.lr
    net = M.build(args.arm, overrides=model_overrides(cfg, args.arm),
                  strict=True)
    print(f"  config  : {cfg_source}")
    print(f"            lr {cfg['lr']:g}, weight decay {cfg['weight_decay']:g}, "
          f"dropout {cfg['dropout']:g}/{cfg['head_dropout']:g}, "
          f"width x{cfg['width_mult']:g}, pooling fraction "
          f"{cfg['k_fraction']:g}")
    print(f"  model   : {M.count_parameters(net):,} parameters", flush=True)

    res = T.train_unit(net, train_loader, train_ds, val_loader, dev,
                       epochs=args.epochs, lr=cfg["lr"],
                       weight_decay=cfg["weight_decay"], grad_clip=GRAD_CLIP,
                       ema_decay=EMA_DECAY, out_dir=out, verbose=True)

    # Validation logits, one row per epoch. pids are saved with them because the
    # eval loader sorts by length, so row order isn't dataset order and later
    # stages have to realign on pid.
    recs = res["val_records"]
    np.savez_compressed(
        os.path.join(out, "val_logits.npz"),
        logits=np.stack([r["logit"] for r in recs]),
        y=recs[0]["y"], pid=recs[0]["pid"],
        epochs=np.arange(len(recs)))

    meta = {
        "arm": args.arm, "repeat": args.repeat, "fold": args.fold,
        "inner": args.inner, "seed": seed, "label": label_note,
        "epochs": args.epochs, "lr": cfg["lr"], "batch": args.batch,
        "weight_decay": cfg["weight_decay"], "ema_decay": EMA_DECAY,
        "config": cfg, "config_source": cfg_source,
        "grad_clip": GRAD_CLIP, "arm_input": arm_input,
        "n_parameters": M.count_parameters(net),
        "n_inner_train": int(len(itr)), "n_inner_val": int(len(iva)),
        "n_inner_train_events": int(y[itr].sum()),
        "n_inner_val_events": int(y[iva].sum()),
        "inner_train_pid": [int(p) for p in pid[itr]],
        "inner_val_pid": [int(p) for p in pid[iva]],
        "outer_test_pid": [int(p) for p in pid[outer["test_idx"]]],
        "history": res["history"],
        "seconds": res["seconds"],
        "seconds_per_epoch": res["seconds"] / max(args.epochs, 1),
        # backend fault counts (trainer.predict explains why these exist)
        "batches_skipped_in_training": res.get("batches_skipped_in_training", 0),
        "validation_batches": res.get("validation_batches", 0),
        "validation_batches_repaired": res.get(
            "validation_batches_repaired", 0),
        "torch": torch.__version__, "device": str(dev),
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\n  done in {res['seconds'] / 60:.1f} min "
          f"({meta['seconds_per_epoch']:.1f}s per epoch)")
    print(f"  backend faults: {meta['batches_skipped_in_training']} training "
          f"batches skipped, {meta['validation_batches_repaired']} of "
          f"{meta['validation_batches']} validation batches repaired")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
