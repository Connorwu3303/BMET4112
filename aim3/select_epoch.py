"""
Aim 3: picks the epoch for one outer fold, then scores its outer test fold once.

run_unit.py trains to a fixed cap and saves every epoch. The epoch is the only
data driven choice in the deep pipeline, and it's picked here once, on the
three inner models' validation logits pooled together (3,481 rows), not per
model on 1,161 rows each. That carries much less selection optimism into the
stacker, which is fitted on those same rows.

Two epochs come from the same runs. E*_marg (best pooled AUC) is for the head
to head with the biomarker model. E*_inc (best deviance reduction over a cross
fitted clinical predictor) is for every incremental comparison, since the best
standalone checkpoint might just be the best age proxy. The outer test fold is
only read after both are fixed.

Outputs (processed/aim3/results/selected/): <arm>_r<repeat>_f<fold>.npz plus a
matching .json with the selection table

Usage:
    python3 select_epoch.py --arm A --repeat 0 --fold 0
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
from run_unit import unit_dir, EPOCH_CAP  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
SELECT_DIR = os.path.join(RESULTS, "selected")

# epochs before this can't be picked. The first two include the warmup and are
# barely trained, so the criterion would just be picking noise
MIN_EPOCH = 3


def confounder_predictor(conf, y, fit_idx, apply_idx):
    """
    Unweighted logistic regression on the 18 standardised confounders.

    Fitted on fit_idx and applied to apply_idx, returns the linear predictor.
    Kept plain on purpose (C = 1, no class weighting, no grid). It stands in for
    what the clinical variables already explain, so the epoch criterion can ask
    what the waveform adds on top. A tuned model would make that reference move.
    """
    from sklearn.linear_model import LogisticRegression
    tf = D.fit_confounder_transform(conf[fit_idx])
    x_fit = D.apply_confounder_transform(conf[fit_idx], tf)
    x_app = D.apply_confounder_transform(conf[apply_idx], tf)
    lr = LogisticRegression(C=1.0, max_iter=2000, penalty="l2")
    lr.fit(x_fit, y[fit_idx])
    return lr.decision_function(x_app)


def deviance(y, *columns):
    """
    Residual deviance of a logistic regression of y on an intercept plus columns.

    Uses statsmodels because the deviance and the likelihood ratio test need an
    unpenalised fit. scikit-learn penalises by default, and how you switch that
    off depends on the version.
    """
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack(columns)) if columns else \
        np.ones((len(y), 1))
    model = sm.GLM(y, X, family=sm.families.Binomial())
    try:
        fit = model.fit()
        return float(fit.deviance), fit
    except Exception:
        return float("nan"), None


def load_units(arm, repeat, fold, tag="", epochs=EPOCH_CAP, n_inner=3):
    """Loads the per epoch validation logits for the three inner models."""
    units = []
    for inner in range(n_inner):
        path = unit_dir(arm, repeat, fold, inner) + tag
        f = os.path.join(path, "val_logits.npz")
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing {f}; run run_unit.py first")
        with np.load(f) as z:
            units.append({"path": path, "logits": z["logits"],
                          "y": z["y"], "pid": z["pid"].astype(int),
                          "inner": inner})
        if units[-1]["logits"].shape[0] < epochs:
            raise ValueError(
                f"{f} has {units[-1]['logits'].shape[0]} epochs, expected "
                f"{epochs}")
    return units


def select_epochs(units, conf, y_all, pid_to_row, epochs=EPOCH_CAP,
                  min_epoch=MIN_EPOCH, verbose=True):
    """
    Picks E*_marg and E*_inc from the pooled, standardised validation logits.

    Returns the per epoch table (with an eligible flag), the two chosen epochs
    and the pooled rows, labels and clinical predictor.
    """
    from sklearn.metrics import roc_auc_score

    # Cross fitted clinical predictor: rows in inner fold k get a model fitted on
    # the other two inner folds, so no row's own outcome feeds into its own
    # clinical prediction.
    rows_per_unit = [np.array([pid_to_row[p] for p in u["pid"]])
                     for u in units]
    o_per_unit = []
    for k, rows in enumerate(rows_per_unit):
        others = np.concatenate([r for j, r in enumerate(rows_per_unit)
                                 if j != k])
        o_per_unit.append(confounder_predictor(conf, y_all, others, rows))

    pooled_rows = np.concatenate(rows_per_unit)
    pooled_y = y_all[pooled_rows]
    pooled_o = np.concatenate(o_per_unit)

    # baseline deviance (clinical predictor only), same for every epoch
    dev_base, _ = deviance(pooled_y, pooled_o)

    table = []
    for e in range(epochs):
        # Standardise each model by its own validation mean and SD (no labels
        # used). Separately trained models sit at different logit scales, so
        # pooled raw the AUC would partly measure how well the three line up.
        z_parts = []
        for u in units:
            v = u["logits"][e]
            mu, sd = float(v.mean()), float(v.std())
            z_parts.append((v - mu) / (sd if sd > 1e-12 else 1.0))
        z = np.concatenate(z_parts)

        auc = float(roc_auc_score(pooled_y, z)) \
            if len(np.unique(pooled_y)) > 1 else float("nan")
        dev_with, _ = deviance(pooled_y, pooled_o, z)
        per_fold = [float(roc_auc_score(y_all[rows], zp))
                    if len(np.unique(y_all[rows])) > 1 else float("nan")
                    for rows, zp in zip(rows_per_unit, z_parts)]
        table.append({
            "epoch": e, "pooled_auc": auc,
            "mean_inner_auc": float(np.nanmean(per_fold)),
            "inner_auc_spread": float(np.nanmax(per_fold)
                                      - np.nanmin(per_fold)),
            "deviance_base": dev_base, "deviance_with": dev_with,
            "deviance_reduction": dev_base - dev_with,
            "eligible": e >= min_epoch,
        })

    df = pd.DataFrame(table)
    elig = df[df["eligible"]]
    if elig.empty:
        # Only happens when the epoch cap is at or below min_epoch, i.e. in the
        # short chain validation runs. The real runs use a cap of 16, so epochs 3
        # to 15 are always eligible. Falling back to all epochs with a printed
        # warning beats crashing, and the warning means it can't slip through
        # unnoticed in a real run.
        print(f"    WARNING: no epoch is eligible with min_epoch={min_epoch} "
              f"and a cap of {epochs}; falling back to all epochs. This is "
              f"expected only in a short validation run.")
        df = df.assign(eligible=True)
        elig = df
    # ties go to the earlier epoch, the more conservative pick, since fewer
    # epochs means less fitting to this particular training fold
    e_marg = int(elig.loc[elig["pooled_auc"].idxmax(), "epoch"])
    e_inc = int(elig.loc[elig["deviance_reduction"].idxmax(), "epoch"])

    if verbose:
        print(f"    {'epoch':>5s} {'pooled AUC':>11s} {'mean inner':>11s} "
              f"{'spread':>7s} {'dev reduction':>14s}")
        for _, r in df.iterrows():
            mark = ""
            if int(r["epoch"]) == e_marg:
                mark += " <- E*_marg"
            if int(r["epoch"]) == e_inc:
                mark += " <- E*_inc"
            flag = "" if r["eligible"] else " (not eligible)"
            print(f"    {int(r['epoch']):5d} {r['pooled_auc']:11.4f} "
                  f"{r['mean_inner_auc']:11.4f} {r['inner_auc_spread']:7.4f} "
                  f"{r['deviance_reduction']:14.3f}{mark}{flag}")

    return df, e_marg, e_inc, {
        "pooled_rows": pooled_rows, "pooled_y": pooled_y, "pooled_o": pooled_o,
        "rows_per_unit": rows_per_unit}


@torch.no_grad()
def score_test(arm, units, epoch, store, pid_test, y_test, dev, batch=16):
    """
    Scores the outer test fold with the three models' EMA weights at this epoch.

    Each model's test logits are standardised with its own validation mean and
    SD at this epoch, then the three are averaged. Averaging narrows the spread
    a little compared with the single model logits the stacker was trained on,
    so the stacked test prediction is if anything conservative. It's left
    uncorrected, since any scale factor would have to come from held out data.
    """
    arm_input = M.arm_input(arm)
    loader, _ = D.make_loader(store, pid_test, y_test, arm=arm_input,
                              train=False, batch_size=batch, seed=0)
    per_model = []
    for u in units:
        net = M.build(arm)
        ck = os.path.join(u["path"], f"ema_epoch{epoch:02d}.pt")
        net.load_state_dict(torch.load(ck, map_location="cpu"))
        net = net.to(dev).eval()
        logit, y_out, pid_out = T.predict(net, loader, dev)
        v = u["logits"][epoch]
        mu, sd = float(v.mean()), float(v.std())
        per_model.append({"inner": u["inner"], "logit": logit,
                          "z": (logit - mu) / (sd if sd > 1e-12 else 1.0),
                          "mu": mu, "sd": sd, "pid": pid_out, "y": y_out})
        del net
        if dev.type == "mps":
            torch.mps.empty_cache()

    # all three scored the same loader so the pid order should match, but check
    for p in per_model[1:]:
        if not np.array_equal(p["pid"], per_model[0]["pid"]):
            raise RuntimeError("test identifier order differs between models")

    return {"pid": per_model[0]["pid"], "y": per_model[0]["y"],
            "z_mean": np.mean([p["z"] for p in per_model], axis=0),
            "z_per_model": np.stack([p["z"] for p in per_model]),
            "standardisation": [{"inner": p["inner"], "mu": p["mu"],
                                 "sd": p["sd"]} for p in per_model]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(M.ARMS))
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=EPOCH_CAP)
    ap.add_argument("--tag", default="")
    ap.add_argument("--permute-labels", action="store_true")
    ap.add_argument("--odi3-label", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(SELECT_DIR, exist_ok=True)
    out = os.path.join(
        SELECT_DIR, f"{args.arm}_r{args.repeat}_f{args.fold}{args.tag}.npz")
    if os.path.exists(out) and not args.force:
        print(f"{os.path.basename(out)} exists, skipping")
        return 0

    t0 = time.time()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    data = get_verified_folds(quiet=True)
    store = D.SignalStore()
    y = data["y"].copy()
    pid = data["pid"]
    conf = data["confounders"]

    if args.permute_labels:
        y = np.random.default_rng(20260928).permutation(y)
    elif args.odi3_label:
        f = pd.read_csv(
            "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv",
            low_memory=False, usecols=["pid", "sleep_odi3"]).set_index("pid")
        y = (f.loc[[int(p) for p in pid], "sleep_odi3"].to_numpy(dtype=float)
             >= 15.0).astype(int)

    pid_to_row = {int(p): i for i, p in enumerate(pid)}
    outer = [f for f in data["folds"] if f["repeat"] == args.repeat
             and f["fold"] == args.fold][0]

    print(f"selecting for arm {args.arm}, repeat {args.repeat}, "
          f"fold {args.fold}")
    units = load_units(args.arm, args.repeat, args.fold, args.tag, args.epochs)

    # The three inner validation folds have to partition this outer fold's
    # training set exactly, otherwise the cross fitted predictions aren't what
    # the stacker assumes.
    train_set = set(int(p) for p in pid[outer["train_idx"]])
    all_val = np.concatenate([u["pid"] for u in units])
    if set(all_val.tolist()) != train_set:
        raise RuntimeError(
            f"the three validation folds cover {len(set(all_val.tolist()))} "
            f"participants but the outer training fold has {len(train_set)}")
    if len(all_val) != len(set(all_val.tolist())):
        raise RuntimeError("a participant appears in more than one inner "
                           "validation fold")
    test_set = set(int(p) for p in pid[outer["test_idx"]])
    if train_set & test_set:
        raise RuntimeError("outer training and test folds overlap")
    print(f"  cross fitted validation rows: {len(all_val)}, "
          f"partitioning the outer training fold exactly")

    df, e_marg, e_inc, pooled = select_epochs(
        units, conf, y, pid_to_row, args.epochs)
    print(f"\n  E*_marg = {e_marg} (pooled AUC "
          f"{df.loc[e_marg, 'pooled_auc']:.4f})")
    print(f"  E*_inc  = {e_inc} (deviance reduction "
          f"{df.loc[e_inc, 'deviance_reduction']:.3f})")

    scored = {}
    for name, epoch in [("marg", e_marg), ("inc", e_inc)]:
        print(f"  scoring the outer test fold at E*_{name} = {epoch}",
              flush=True)
        scored[name] = score_test(args.arm, units, epoch, store,
                                  pid[outer["test_idx"]],
                                  y[outer["test_idx"]], dev)

    # cross fitted training logits at a chosen epoch, in cohort row order (this
    # is what the stacker reads)
    def pooled_z(epoch):
        z = np.full(len(y), np.nan)
        for u in units:
            v = u["logits"][epoch]
            mu, sd = float(v.mean()), float(v.std())
            for p, val in zip(u["pid"], (v - mu) / (sd if sd > 1e-12 else 1.0)):
                z[pid_to_row[int(p)]] = val
        return z

    np.savez_compressed(
        out,
        e_marg=e_marg, e_inc=e_inc,
        selection_table=df.to_records(index=False),
        train_z_marg=pooled_z(e_marg), train_z_inc=pooled_z(e_inc),
        train_o=_scatter(pooled["pooled_rows"], pooled["pooled_o"], len(y)),
        test_pid=scored["marg"]["pid"], test_y=scored["marg"]["y"],
        test_z_marg=scored["marg"]["z_mean"],
        test_z_inc=scored["inc"]["z_mean"],
        test_z_marg_per_model=scored["marg"]["z_per_model"],
        test_z_inc_per_model=scored["inc"]["z_per_model"],
        outer_train_pid=pid[outer["train_idx"]],
        outer_test_pid=pid[outer["test_idx"]])

    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump({"arm": args.arm, "repeat": args.repeat, "fold": args.fold,
                   "e_marg": e_marg, "e_inc": e_inc,
                   "selection": df.to_dict(orient="records"),
                   "standardisation_marg": scored["marg"]["standardisation"],
                   "standardisation_inc": scored["inc"]["standardisation"],
                   "seconds": time.time() - t0}, fh, indent=2)

    print(f"\n  wrote {out}  ({time.time() - t0:.0f}s)")
    return 0


def _scatter(rows, values, n):
    """Puts values at their cohort row positions, NaN everywhere else."""
    out = np.full(n, np.nan)
    out[rows] = values
    return out


if __name__ == "__main__":
    sys.exit(main())
