"""
Aim 4: checks the deep attribution by intervention instead of by gradients.

The methods in deep_attrib.py all read something out of the model at a single
point in input space. Occlusion blanks part of the night and measures the
change in logit, so it doesn't rely on any gradient being faithful.

Two ways of removing a stretch, both reported:
  mask_cleared  set to the fill value and dropped from the pooling masks, the
                way wake and artefact look to the network. This also changes
                the masked mean denominators, so the remaining steps re-weight.
  mask_kept     set to the fill value but kept in the pooling, so only the
                content of those samples changes.

The most positively attributed steps are compared with the same number of
random steps (three draws) and the most negatively attributed ones. A faithful
attribution predicts the change in logit runs top < random < bottom, which is
checked per participant.

Outputs (processed/aim4/results/): occlusion_A.csv, occlusion_A.json

Usage:
    python3 occlusion.py --n 300
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

import common4 as C                                      # noqa: E402
import deep_attrib as A                                  # noqa: E402

import data as D                                         # noqa: E402
import models as M                                       # noqa: E402
from folds import get_verified_folds                     # noqa: E402

# Fraction of valid steps to occlude. A tenth matches the top decile the pooling
# selects, so the test removes about as much of the night as the top decile
# pool looks at.
OCCLUDE_FRACTION = 0.10
N_RANDOM_DRAWS = 3
SEED = 20261011


@torch.no_grad()
def logit_of(net, x, mn, mr):
    v = float(net(x, mn, mr).squeeze().item())
    if not np.isfinite(v) or abs(v) > C.SANE_LOGIT:
        # The Apple backend fault seen in Aim 3. Retry once, then fall back to
        # the deterministic CPU path so an impossible logit never gets saved.
        v = float(net(x, mn, mr).squeeze().item())
        if not np.isfinite(v) or abs(v) > C.SANE_LOGIT:
            cpu = net.to("cpu")
            v = float(cpu(x.cpu(), mn.cpu(), mr.cpu()).squeeze().item())
            net.to(x.device)
    return v


def occlude(x, mn, mr, steps, stride, fill_scaled, clear_mask):
    """
    Copies of the inputs with the given steps blanked.

    Steps are indices on the network's own step grid, so the blanked region
    lines up with the grid the attribution was computed on. Blanking part of a
    step would credit some of that step's contribution to samples outside it.
    """
    x2, mn2, mr2 = x.clone(), mn.clone(), mr.clone()
    for s in steps:
        a, b = int(s) * stride, (int(s) + 1) * stride
        x2[..., a:b] = fill_scaled
        if clear_mask:
            mn2[..., a:b] = 0.0
            mr2[..., a:b] = 0.0
    return x2, mn2, mr2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=C.DEEP_ARM)
    ap.add_argument("--repeat", type=int, default=C.DEEP_REPEAT)
    ap.add_argument("--epoch-kind", default=C.DEEP_EPOCH_KIND)
    ap.add_argument("--n", type=int, default=300,
                    help="participants to test, stratified over folds and "
                         "outcome")
    ap.add_argument("--fraction", type=float, default=OCCLUDE_FRACTION)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    out_dir = C.ensure_results()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    stride = 2 ** 5

    print("=" * 78)
    print(f"AIM 4 OCCLUSION TEST: arm {args.arm}, "
          f"{args.fraction:.0%} of valid steps")
    print("=" * 78)

    data = get_verified_folds(quiet=True)
    pid_all, y_all = data["pid"], data["y"]
    assign = A.fold_of_each_participant(data, args.repeat)
    store = D.SignalStore()

    take = []
    per = max(2, args.n // 5)
    for f in range(5):
        rows = np.flatnonzero(assign == f)
        ev = rows[y_all[rows] == 1]
        ne = rows[y_all[rows] == 0]
        ev = rng.permutation(ev)[:per // 2]
        ne = rng.permutation(ne)[:per - len(ev)]
        take.append(np.concatenate([ev, ne]))
    order = np.concatenate(take)
    order = order[np.argsort(assign[order], kind="stable")]

    rows, current_fold, nets = [], -1, None
    t0 = time.time()
    for count, row in enumerate(order):
        pid = int(pid_all[row])
        fold = int(assign[row])
        if fold != current_fold:
            if nets is not None:
                del nets
                if device.type == "mps":
                    torch.mps.empty_cache()
            nets = A.load_fold_models(args.arm, args.repeat, fold,
                                      args.epoch_kind, device)
            current_fold = fold
            print(f"\n  fold {fold}", flush=True)

        item = A.single_item(store, pid, M.arm_input(args.arm))
        x = item["x"].to(device)
        mn = item["m_nrem"].to(device)
        mr = item["m_rem"].to(device)
        fill_scaled = float(D.scale_signal(np.float32(item["fill"])))

        rec = {"pid": pid, "y": int(y_all[row]), "fold": fold,
               "n_samples": item["n"]}

        for mdl in nets:
            net = mdl["net"]
            ex = A.exact_step_contributions(net, x, mn, mr)
            contrib = ex["contrib"]
            base = ex["logit_actual"]

            # Only occlude steps the pooling uses. Blanking a step that's
            # already outside every mask does nothing and would pull the
            # result towards zero.
            m_any = ((mn + mr) > 0).to(x.dtype)
            valid = M._match_length(
                M.pool_mask(m_any, net.total_stride),
                contrib.shape[0])[0].cpu().numpy() > 0
            vidx = np.flatnonzero(valid)
            k = max(1, int(round(args.fraction * vidx.size)))
            ordered = vidx[np.argsort(-contrib[vidx])]
            top, bottom = ordered[:k], ordered[-k:]

            for clear in (True, False):
                suffix = "cleared" if clear else "kept"
                for name, sel in (("top", top), ("bottom", bottom)):
                    xo, mno, mro = occlude(x, mn, mr, sel, stride,
                                           fill_scaled, clear)
                    rec.setdefault(f"d_{name}_{suffix}", []).append(
                        logit_of(net, xo, mno, mro) - base)
                drops = []
                for d in range(N_RANDOM_DRAWS):
                    sel = rng.choice(vidx, size=k, replace=False)
                    xo, mno, mro = occlude(x, mn, mr, sel, stride,
                                           fill_scaled, clear)
                    drops.append(logit_of(net, xo, mno, mro) - base)
                rec.setdefault(f"d_random_{suffix}", []).append(
                    float(np.mean(drops)))
                rec.setdefault(f"d_random_sd_{suffix}", []).append(
                    float(np.std(drops)))

            rec.setdefault("base_logit", []).append(base)
            # Change in logit the decomposition predicts for the top and bottom
            # occlusions, so each measurement can be compared with a predicted
            # value as well as ranked against the other conditions.
            rec.setdefault("predicted_top", []).append(
                float(-contrib[top].sum()))
            rec.setdefault("predicted_bottom", []).append(
                float(-contrib[bottom].sum()))
            rec.setdefault("k_steps", []).append(int(k))
            rec.setdefault("n_valid_steps", []).append(int(vidx.size))

        flat = {}
        for key, val in rec.items():
            flat[key] = float(np.mean(val)) if isinstance(val, list) else val
        rows.append(flat)
        if (count + 1) % 25 == 0:
            el = time.time() - t0
            print(f"    {count + 1}/{len(order)}  {el:.0f}s  "
                  f"eta {(len(order) - count - 1) * el / (count + 1) / 60:.0f} min",
                  flush=True)

    t = pd.DataFrame(rows)
    path = os.path.join(out_dir, f"occlusion_{args.arm}{args.tag}.csv")
    t.to_csv(path, index=False)

    summary = {"arm": args.arm, "n": len(t), "fraction": args.fraction,
               "n_random_draws": N_RANDOM_DRAWS,
               "median_k_steps": float(t["k_steps"].median()),
               "minutes_occluded":
                   float(t["k_steps"].median() * C.STEP_SECONDS / 60.0),
               "seconds": time.time() - t0}
    print("\n" + "=" * 78)
    print(f"{'condition':<24s}{'median change in logit':>24s}"
          f"{'mean':>10s}{'predicted':>12s}")
    for suffix in ("cleared", "kept"):
        for name in ("top", "random", "bottom"):
            col = f"d_{name}_{suffix}"
            pred = {"top": "predicted_top",
                    "bottom": "predicted_bottom"}.get(name)
            pv = f"{t[pred].median():+12.4f}" if pred else " " * 12
            summary[f"{col}_median"] = float(t[col].median())
            summary[f"{col}_mean"] = float(t[col].mean())
            print(f"{name + ', ' + suffix:<24s}{t[col].median():>+24.4f}"
                  f"{t[col].mean():>+10.4f}{pv}")

    # Check the predicted ordering per participant. The medians can satisfy an
    # ordering that most individuals break.
    for suffix in ("cleared", "kept"):
        ok = (t[f"d_top_{suffix}"] < t[f"d_random_{suffix}"]) & \
             (t[f"d_random_{suffix}"] < t[f"d_bottom_{suffix}"])
        summary[f"ordering_holds_{suffix}"] = float(ok.mean())
        print(f"\nthe predicted ordering (top < random < bottom) holds for "
              f"{ok.mean():.1%} of participants, {suffix}")
        r = C.spearman_rank_agreement(t["predicted_top"],
                                      t[f"d_top_{suffix}"])
        summary[f"predicted_vs_measured_spearman_{suffix}"] = r["rho"]
        print(f"predicted against measured drop, across participants: "
              f"Spearman {r['rho']:+.3f} (n {r['n']})")

    with open(os.path.join(out_dir,
                           f"occlusion_{args.arm}{args.tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nwrote {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
