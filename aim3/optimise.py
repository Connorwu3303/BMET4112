"""
Aim 3: label free hyperparameter search for the CNN.

With 829 events, the true differences between configurations are much smaller
than the noise in a CVD based criterion (SE roughly 0.012 AUC), and any choice
made on the outcome would count against the final evaluation. So the search
regresses sleep ODI3 from the waveform on the 1,441 participants with no usable
incident outcome, who aren't in any CVD fold.

The proxy target is much easier than the real one, so the adoption rule was set
before the search ran. Learning rate, weight decay, width and pooling fraction
are adopted, since they govern optimisation and capacity. Either dropout is
only adopted if the search wants more than the pre-registered value, because an
easy target leans towards less regularisation. That rule is a judgement call.

Outputs (processed/aim3/results/): optimisation_trials.csv,
optimisation_result.json

Usage:
    python3 optimise.py --trials 20
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
from gates import no_outcome_pids, odi3_targets  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
SEED = 42

# pre-registered config, also run as the first (baseline) trial
BASELINE = {"lr": 1e-3, "weight_decay": 1e-2, "dropout": 0.1,
            "head_dropout": 0.5, "width_mult": 1.0, "k_fraction": 0.10}

# Trials are shorter than a real fit so the search is affordable. Only the
# ranking of configs matters here, and the gate run showed the ordering is
# already clear well before convergence.
TRIAL_EPOCHS = 10


def widths_for(mult):
    """Scales the network widths by mult, keeping the same shape."""
    base = (16, 32, 48, 64, 64)
    w = tuple(max(4, int(round(c * mult))) for c in base)
    return w, max(4, int(round(64 * mult))), max(4, int(round(16 * mult)))


def build_overrides(params):
    w, dil, stem = widths_for(params["width_mult"])
    return {"widths": w, "dilated_width": dil, "stem_width": stem,
            "dropout": params["dropout"],
            "head_dropout": params["head_dropout"],
            "k_fraction": params["k_fraction"]}


def run_trial(store, pids, target, tr, va, params, device, epochs=TRIAL_EPOCHS,
              verbose=False):
    """Trains one config on the label free task and returns its scores."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tl, tds = D.make_loader(store, pids[tr], target[tr], arm="primary",
                            train=True, batch_size=32, seed=SEED)
    vl, _ = D.make_loader(store, pids[va], target[va], arm="primary",
                          train=False, batch_size=16, seed=SEED)
    net = M.build("A", overrides=build_overrides(params))
    n_par = M.count_parameters(net)
    res = T.train_unit(net, tl, tds, vl, device, epochs=epochs,
                       lr=params["lr"], weight_decay=params["weight_decay"],
                       loss_kind="huber", verbose=verbose)
    rs = [h["val_metric"] for h in res["history"]]
    return {"best_r": float(np.nanmax(rs)),
            "final_r": float(rs[-1]),
            "best_epoch": int(np.nanargmax(rs)),
            "r_per_epoch": rs,
            "n_parameters": int(n_par),
            "seconds": res["seconds"],
            "skipped_batches": res.get("batches_skipped_in_training", 0),
            "repaired_batches": res.get("validation_batches_repaired", 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=TRIAL_EPOCHS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 76)
    print("AIM 3 HYPERPARAMETER OPTIMISATION, ON A LABEL FREE PROXY TASK")
    print("=" * 76)
    print(f"device {device}, {args.trials} trials of {args.epochs} epochs each")

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

    print(f"\ntask: regress log1p(sleep ODI3) from the waveform")
    print(f"  {len(tr)} train, {len(va)} validation, from the {len(pids)} "
          f"participants with no usable incident outcome")
    print(f"  these participants appear in NO cardiovascular fold, so nothing "
          f"here can contaminate the evaluation")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    records = []

    def objective(trial):
        params = {
            "lr": trial.suggest_float("lr", 2e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 5e-2,
                                                log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
            "head_dropout": trial.suggest_float("head_dropout", 0.2, 0.6,
                                                step=0.1),
            "width_mult": trial.suggest_categorical("width_mult",
                                                    [0.75, 1.0, 1.5]),
            "k_fraction": trial.suggest_categorical("k_fraction",
                                                    [0.05, 0.10, 0.20]),
        }
        t0 = time.time()
        out = run_trial(store, pids, target, tr, va, params, device,
                        epochs=args.epochs)
        rec = {"trial": trial.number, **params, **out}
        records.append(rec)
        print(f"  trial {trial.number:2d}: r {out['best_r']:.4f}  "
              f"lr {params['lr']:.2e}  wd {params['weight_decay']:.2e}  "
              f"drop {params['dropout']:.2f}/{params['head_dropout']:.1f}  "
              f"width {params['width_mult']}  k {params['k_fraction']}  "
              f"{out['n_parameters'] / 1000:.0f}k par  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        return out["best_r"]

    # queue the pre-registered config as trial 0, so the search is always
    # compared against the baseline it might replace
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True))
    study.enqueue_trial({k: v for k, v in BASELINE.items()})
    t0 = time.time()
    study.optimize(objective, n_trials=args.trials)

    table = pd.DataFrame(records).sort_values("best_r", ascending=False)
    table.to_csv(os.path.join(RESULTS, "optimisation_trials.csv"), index=False)

    best = table.iloc[0]
    base_row = table[
        (np.isclose(table["lr"], BASELINE["lr"])) &
        (np.isclose(table["weight_decay"], BASELINE["weight_decay"])) &
        (np.isclose(table["dropout"], BASELINE["dropout"])) &
        (np.isclose(table["head_dropout"], BASELINE["head_dropout"]))]
    base_r = float(base_row["best_r"].iloc[0]) if len(base_row) else None

    print("\n" + "-" * 76)
    print("RESULTS")
    print("-" * 76)
    print(f"\n  {'rank':>4s} {'r':>7s} {'lr':>9s} {'wd':>9s} {'drop':>5s} "
          f"{'head':>5s} {'width':>6s} {'k':>5s} {'params':>8s}")
    for i, (_, r) in enumerate(table.head(8).iterrows()):
        print(f"  {i + 1:>4d} {r['best_r']:>7.4f} {r['lr']:>9.2e} "
              f"{r['weight_decay']:>9.2e} {r['dropout']:>5.2f} "
              f"{r['head_dropout']:>5.1f} {r['width_mult']:>6.2f} "
              f"{r['k_fraction']:>5.2f} {r['n_parameters'] / 1000:>7.0f}k")

    if base_r is not None:
        print(f"\n  pre-registered configuration: r = {base_r:.4f}")
        print(f"  best configuration found    : r = {best['best_r']:.4f}")
        print(f"  improvement on the proxy    : {best['best_r'] - base_r:+.4f}")

    # Adoption rule from the module docstring. Optimisation and capacity
    # settings are adopted. A lower dropout isn't, because a search on an easy
    # target leans that way and the real target is weak.
    adopted = dict(BASELINE)
    notes = []
    for key in ("lr", "weight_decay", "width_mult", "k_fraction"):
        if not np.isclose(best[key], BASELINE[key]):
            adopted[key] = float(best[key])
            notes.append(f"{key}: {BASELINE[key]} -> {adopted[key]:g} "
                         f"(adopted, governs optimisation or capacity)")
    for key in ("dropout", "head_dropout"):
        if best[key] > BASELINE[key]:
            adopted[key] = float(best[key])
            notes.append(f"{key}: {BASELINE[key]} -> {adopted[key]:g} "
                         f"(adopted, more regularisation than pre-registered)")
        elif best[key] < BASELINE[key]:
            notes.append(f"{key}: search preferred {best[key]:g} but "
                         f"{BASELINE[key]} is KEPT, because a search on a "
                         f"strongly learnable target is biased toward less "
                         f"regularisation and the real target is weak")

    print("\n" + "-" * 76)
    print("WHAT IS ADOPTED")
    print("-" * 76)
    if notes:
        for n in notes:
            print(f"  {n}")
    else:
        print("  nothing changes: the pre-registered configuration was best, or "
              "no difference was material")
    print(f"\n  final configuration: {json.dumps(adopted)}")

    # Spread of the score across the top few configs. If they're all within
    # noise, the choice barely matters and the printout should say so.
    top = table.head(min(5, len(table)))["best_r"].to_numpy()
    print(f"\n  spread among the top {len(top)} configurations: "
          f"{top.max() - top.min():.4f} in r")
    if base_r is not None and best["best_r"] - base_r < 0.01:
        print("  the gain over the pre-registered configuration is under 0.01, "
              "so this is a confirmation of the original choice rather than a "
              "meaningful improvement")

    with open(os.path.join(RESULTS, "optimisation_result.json"), "w") as fh:
        json.dump({"baseline": BASELINE, "baseline_r": base_r,
                   "best_trial": {k: (float(v) if isinstance(v, (int, float,
                                                                 np.floating))
                                      else v)
                                  for k, v in best.to_dict().items()
                                  if k != "r_per_epoch"},
                   "adopted": adopted, "notes": notes,
                   "n_trials": len(table), "trial_epochs": args.epochs,
                   "top_spread": float(top.max() - top.min()),
                   "minutes": (time.time() - t0) / 60.0,
                   "task": "log1p(sleep_odi3) regression on the 1,441 "
                           "participants with no usable incident outcome",
                   "contaminates_evaluation": False}, fh, indent=2)
    print(f"\nwrote {RESULTS}/optimisation_result.json")
    print(f"wrote {RESULTS}/optimisation_trials.csv")
    print(f"total {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
