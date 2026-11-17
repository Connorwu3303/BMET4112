"""
Aim 4: attribution for the deep model at three levels of exactness.

  1. Exact step decomposition (32 s grid), checked against the forward pass.
  2. Integrated gradients (1 s grid), completeness checked per participant.
  3. Grad-CAM (32 s grid), the standard tool, scored against method 1.

Method 1 works because architecture A's head is one linear layer on a pooled
vector, and both pools (masked mean and masked top decile mean) are fixed
weighted sums over steps, so the logit is exactly a bias plus a sum of per
step terms. Integrated gradients uses two baselines: flat_own, the night's own
median sleep SpO2 held flat (only the shape of the night is attributed), and
flat_97, a fixed 97 percent (the level counts too).

Everything is out of fold: each participant is explained by the three inner
models of their outer test fold, at the epoch Aim 3 picked for standalone
discrimination (E*_marg), and the attributions are averaged.

Outputs (processed/aim4/results/): deep_attrib_A.csv, deep_attrib_A.json and
deep_traces_A.npz, with a _random suffix for the random weights control.

Usage:
    python3 deep_attrib.py --limit 40            # pilot
    python3 deep_attrib.py                       # whole cohort
    python3 deep_attrib.py --random-weights      # sanity control
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

import data as D                                         # noqa: E402
import models as M                                       # noqa: E402
from folds import get_verified_folds                     # noqa: E402
from run_unit import unit_dir                            # noqa: E402

EVENTS_DIR = "/Volumes/ThesisHD/Data/processed/events"

# Integrated gradients settings. The step count trades cost against accuracy
# and the completeness error shows where it lands, so that error is reported.
# Midpoint rule instead of the usual left endpoint sum, since it's second order
# accurate for the same number of forward passes.
IG_STEPS = 16
IG_CHUNK = 8

# The flat_97 baseline is further from most nights than the night's own median,
# so the straight line path is longer and the integrand varies more along it.
# On four participants, 16 steps left errors up to 0.45 logits from flat_97 and
# under 0.01 from flat_own. So the step count doubles per participant until the
# completeness identity holds (capped at IG_STEPS_CAP), and the count used is
# saved.
IG_STEPS_FAR = 64
IG_STEPS_CAP = 256

# A completeness error (in logits) above this counts as a real failure of the
# path integral. The cohort logit spread is about one, so this is a fortieth of
# that.
IG_ABS_TOLERANCE = 0.025

# a priori baseline level, snapped to the oximeter's ladder so it's a value the
# device could have reported
BASELINE_97 = float(np.round(97.0 / D.QUANTUM) * D.QUANTUM)

DESAT_CONCEPTS = ("SpO2 desaturation",)
RESP_CONCEPTS = ("Hypopnea", "Obstructive apnea", "Central apnea",
                 "Mixed apnea", "Unsure")


# --- fold and model loading ---

def fold_of_each_participant(data, repeat):
    """
    Map each participant to the outer fold that had them as a test case.

    The outer folds of a repeat should partition the cohort, but it's checked
    anyway: a participant in two test folds would get an attribution averaged
    over models that had seen their outcome.
    """
    pid = data["pid"]
    assign = np.full(len(pid), -1, dtype=int)
    for f in data["folds"]:
        if f["repeat"] != repeat:
            continue
        if np.any(assign[f["test_idx"]] >= 0):
            raise RuntimeError("a participant is a test case in two folds of "
                               "the same repeat")
        assign[f["test_idx"]] = f["fold"]
    if np.any(assign < 0):
        raise RuntimeError(f"{int((assign < 0).sum())} participants belong to "
                           f"no test fold of repeat {repeat}")
    return assign


def load_fold_models(arm, repeat, fold, epoch_kind, device,
                     random_weights=False):
    """
    The three inner models of one outer fold, at the selected epoch.

    Returns a list of dicts with the network plus that model's validation mean
    and SD. Those are needed to reproduce the standardised score Aim 3
    published, which confirms the network being explained is the one that
    produced the result.
    """
    sel_path = os.path.join(C.AIM3_RESULTS, "selected",
                            f"{arm}_r{repeat}_f{fold}.json")
    with open(sel_path) as fh:
        sel = json.load(fh)
    epoch = int(sel[f"e_{epoch_kind}"])
    std = sel[f"standardisation_{epoch_kind}"]

    out = []
    for inner in range(3):
        net = M.build(arm)
        if not random_weights:
            ck = os.path.join(unit_dir(arm, repeat, fold, inner),
                              f"ema_epoch{epoch:02d}.pt")
            net.load_state_dict(torch.load(ck, map_location="cpu"))
        else:
            # Adebayo sanity control: same architecture, untrained weights. The
            # head starts at zero, which would make every attribution exactly
            # zero and the control pointless, so the head gets random weights
            # too.
            g = torch.Generator().manual_seed(20261011 + inner)
            with torch.no_grad():
                net.head.weight.normal_(0.0, 0.1, generator=g)
        rec = [s for s in std if int(s["inner"]) == inner][0]
        net = net.to(device).eval()
        out.append({"net": net, "inner": inner, "mu": float(rec["mu"]),
                    "sd": float(rec["sd"]), "epoch": epoch})
    return out


# --- one participant's input, built the same way as at evaluation ---

def single_item(store, pid, arm_input="primary"):
    """
    One participant's padded input, exactly as the evaluation loader builds it.

    Goes through the same dataset class, so the attribution can't end up running
    on a differently prepared input from the one behind the published score.
    Augmentation is off, same as at evaluation.
    """
    ds = D.SpO2Dataset(store, [pid], [0.0], arm=arm_input, train=False)
    chan, m_nrem, m_rem, _, _ = ds[0]
    night = store.sleep_period(pid) if arm_input != "d" \
        else store.recording(pid)
    return {"x": chan.unsqueeze(0), "m_nrem": m_nrem.unsqueeze(0),
            "m_rem": m_rem.unsqueeze(0), "fill": night["fill"],
            "n": int(night["x"].shape[0]), "pad": D.PAD_EACH_SIDE,
            "percent": night["x"].astype(np.float32),
            "m_sleep": night["m_sleep"], "night_nrem": night["m_nrem"],
            "night_rem": night["m_rem"]}


def sleep_period_offset(store, pid):
    """
    Index in the full recording where the sleep period window starts.

    Needed to place scored events (timed from the start of the recording) onto
    the cropped window the network sees. Recomputed with the same rule the
    store uses so the two can't drift apart.
    """
    _, _, st = store.raw(pid)
    idx = np.flatnonzero(np.isin(st, D.SLEEP_STAGES))
    return int(idx[0]) if idx.size else 0


def event_intervals(pid, concepts, offset, length):
    """
    Scored events for one participant as a boolean mask over the sleep period
    window, plus the number of events.

    These are the technician's scoring from the NSRR annotation files (parsed
    in Aim 1), so they don't depend on the project's own event detector.
    """
    path = os.path.join(EVENTS_DIR, f"shhs1-{pid}_events.csv")
    mask = np.zeros(length, dtype=bool)
    if not os.path.exists(path):
        return mask, 0
    ev = pd.read_csv(path)
    sel = ev[ev["event_concept"].isin(concepts)]
    n = 0
    for s, d in zip(sel["start_sec"].to_numpy(dtype=float),
                    sel["duration_sec"].to_numpy(dtype=float)):
        a = int(round(s)) - offset
        b = a + max(1, int(round(d)))
        a, b = max(0, a), min(length, b)
        if b > a:
            mask[a:b] = True
            n += 1
    return mask, n


# --- method 1: exact step decomposition ---

def _pooled_parts(net, z, sn, sr):
    """
    The four pooled blocks and the per step top decile selection, rebuilt so
    the logit can be split over steps.

    Mirrors masked_mean_top and StageStratifiedPool, including the rule that a
    night with fewer than MIN_REM_STEPS valid REM steps copies its REM block
    from its NREM block. A unit test checks that the rebuilt logit matches the
    network's own forward pass.
    """
    k_frac = net.pool.k_fraction

    def blocks(mask):
        n_valid = mask.sum()
        count = torch.clamp(n_valid, min=1.0)
        k = torch.clamp(torch.ceil(n_valid * k_frac), min=1.0)
        kk = max(1, int(k.item()))
        mean_w = (mask / count).squeeze(0)                     # (L,)
        filled = z[0].masked_fill(mask[0].unsqueeze(0) == 0, -1e4)
        order = torch.argsort(filled, dim=1, descending=True)[:, :kk]
        sel = torch.zeros_like(filled)
        sel.scatter_(1, order, 1.0)
        return mean_w, sel / k
    return blocks(sn), blocks(sr)


@torch.no_grad()
def exact_step_contributions(net, x, mn, mr):
    """
    Per step contributions to the logit on the 32 second grid, plus the
    reconstructed logit and the network's own logit so the caller can check
    they agree.

    Exact in feature space only: each step's feature depends on about 70
    minutes of surrounding signal, not just the samples inside that step.
    """
    m_any = ((mn + mr) > 0).to(x.dtype)
    z = net.features(x, m_any)
    length = z.shape[-1]
    sn = M._match_length(M.pool_mask(mn, net.total_stride), length)
    sr = M._match_length(M.pool_mask(mr, net.total_stride), length)

    w = net.head.weight.detach()[0]
    bias = float(net.head.bias.detach()[0])
    width = z.shape[1]
    w_nm, w_nt = w[:width], w[width:2 * width]
    w_rm, w_rt = w[2 * width:3 * width], w[3 * width:]

    (nm_w, nt_w), (rm_w, rt_w) = _pooled_parts(net, z, sn, sr)
    enough = bool((sr.sum() >= M.MIN_REM_STEPS).item())
    if not enough:
        rm_w, rt_w = nm_w, nt_w

    zc = z[0]                                               # (C, L)
    contrib = (zc * w_nm.view(-1, 1)).sum(0) * nm_w
    contrib = contrib + ((zc * nt_w) * w_nt.view(-1, 1)).sum(0)
    contrib = contrib + (zc * w_rm.view(-1, 1)).sum(0) * rm_w
    contrib = contrib + ((zc * rt_w) * w_rt.view(-1, 1)).sum(0)

    recon = bias + float(contrib.sum().item())
    actual = float(net(x, mn, mr).squeeze().item())
    return {"contrib": contrib.float().cpu().numpy(), "bias": bias,
            "logit_reconstructed": recon, "logit_actual": actual,
            "rem_fallback": not enough,
            "n_steps_nrem": int(sn.sum().item()),
            "n_steps_rem": int(sr.sum().item())}


# --- method 2: integrated gradients ---

def integrated_gradients(net, x, mn, mr, x0, steps=IG_STEPS, chunk=IG_CHUNK):
    """
    Integrated gradients from x0 to x along a straight line (midpoint rule).

    Only the SpO2 channel is varied and the pooling masks stay fixed. Fading
    the masks as well would pass through nights with impossible sleep
    structure, and the attribution wouldn't mean much.
    """
    diff = x - x0
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    total = torch.zeros_like(x)
    for start in range(0, steps, chunk):
        a = alphas[start:start + chunk].to(x.device).view(-1, 1, 1)
        xi = (x0 + a * diff).detach().requires_grad_(True)
        n = xi.shape[0]
        out = net(xi, mn.expand(n, -1), mr.expand(n, -1))
        grad, = torch.autograd.grad(out.sum(), xi)
        total = total + grad.detach().sum(0, keepdim=True)
    attrib = diff * (total / steps)
    with torch.no_grad():
        f_x = float(net(x, mn, mr).squeeze().item())
        f_0 = float(net(x0, mn, mr).squeeze().item())
    got = float(attrib.sum().item())
    return {"attrib": attrib[0, 0].detach().float().cpu().numpy(),
            "f_x": f_x, "f_baseline": f_0, "sum_attrib": got,
            "completeness_error": got - (f_x - f_0)}


# --- method 3: Grad-CAM ---

def _logit_from_features(net, z, mn, mr):
    """Pooling and head applied to a feature map, for Grad-CAM."""
    length = z.shape[-1]
    sn = M._match_length(M.pool_mask(mn, net.total_stride), length)
    sr = M._match_length(M.pool_mask(mr, net.total_stride), length)
    return net.head(net.pool(z, sn, sr))


def grad_cam(net, x, mn, mr):
    """
    Grad-CAM on the last dilated block, plus gradient times activation.

    Both come from one backward pass. Grad-CAM averages each channel's gradient
    over steps, which throws away the top decile selection in the pooling.
    Gradient times activation keeps the per step gradient, so it recovers the
    exact decomposition apart from the top k selection not being
    differentiable. Scoring both against the exact answer isolates the cost of
    the averaging. Channel weights use valid steps only, so padding can't
    dilute them.
    """
    m_any = ((mn + mr) > 0).to(x.dtype)
    z = net.features(x, m_any)
    z.retain_grad()
    out = _logit_from_features(net, z, mn, mr)
    net.zero_grad(set_to_none=True)
    out.sum().backward()
    g = z.grad.detach()[0]                                   # (C, L)
    a = z.detach()[0]
    length = z.shape[-1]
    valid = M._match_length(
        M.pool_mask(m_any, net.total_stride), length)[0]
    denom = torch.clamp(valid.sum(), min=1.0)
    weights = (g * valid.unsqueeze(0)).sum(1) / denom        # (C,)
    cam = (a * weights.view(-1, 1)).sum(0) * valid
    gxa = (g * a).sum(0) * valid
    net.zero_grad(set_to_none=True)
    return {"cam_signed": cam.float().cpu().numpy(),
            "cam_relu": torch.clamp(cam, min=0.0).float().cpu().numpy(),
            "grad_x_activation": gxa.float().cpu().numpy()}


# --- aggregation ---

def to_sample_grid(step_values, stride, length):
    """
    Spread a step level quantity over its samples, dividing by the stride.

    Dividing keeps the total: a step contributing 0.4 to the logit becomes 32
    samples of 0.0125 each, so sums over any set of samples stay comparable
    with sums over steps.
    """
    rep = np.repeat(np.asarray(step_values, dtype=np.float64), stride) / stride
    if rep.shape[0] >= length:
        return rep[:length]
    return np.concatenate([rep, np.zeros(length - rep.shape[0])])


def step_sums(sample_values, stride):
    """
    Sum a per sample quantity within each step of the network's grid.

    Inverse of to_sample_grid for additive quantities. Used to compare a one
    second attribution with a step level one on the same footing.
    """
    v = np.asarray(sample_values, dtype=np.float64)
    n = (v.shape[0] // stride) * stride
    if n == 0:
        return np.zeros(1)
    return v[:n].reshape(-1, stride).sum(axis=1)


def profile(values, percent, m_nrem, m_rem, desat, resp, m_sleep):
    """
    Reduce a per sample attribution to the pre declared summaries.

    Every summary is a sum over a set of samples, so the parts add up to the
    whole and the arithmetic can be checked. A per sample mean would hide that
    a bin holding 0.1 percent of the night can't matter much, however intense
    it looks.
    """
    out = {}
    total = float(np.sum(values))
    out["total"] = total
    out["total_abs"] = float(np.sum(np.abs(values)))

    bins = C.bin_index(percent)
    for i, lab in enumerate(C.bin_labels()):
        sel = (bins == i) & m_sleep
        out[f"bin_{i}_sum"] = float(np.sum(values[sel]))
        out[f"bin_{i}_abs"] = float(np.sum(np.abs(values[sel])))
        out[f"bin_{i}_n"] = int(sel.sum())

    for name, sel in (("nrem", m_nrem), ("rem", m_rem),
                      ("desat", desat & m_sleep), ("resp", resp & m_sleep),
                      ("sleep", m_sleep)):
        out[f"{name}_sum"] = float(np.sum(values[sel]))
        out[f"{name}_abs"] = float(np.sum(np.abs(values[sel])))
        out[f"{name}_n"] = int(sel.sum())

    # Enrichment: share of absolute attribution inside scored events divided by
    # the share of sleep time they cover. One means event time is treated like
    # any other time, above one means attribution piles up there. Kept as a
    # ratio since both shares are small.
    denom_abs = float(np.sum(np.abs(values[m_sleep])))
    n_sleep = max(1, int(m_sleep.sum()))
    for name, sel in (("desat", desat & m_sleep), ("resp", resp & m_sleep)):
        share_a = (float(np.sum(np.abs(values[sel]))) / denom_abs) \
            if denom_abs > 0 else np.nan
        share_t = float(sel.sum()) / n_sleep
        out[f"{name}_share_abs"] = share_a
        out[f"{name}_share_time"] = share_t
        out[f"{name}_enrichment"] = (share_a / share_t) if share_t > 0 \
            else np.nan

    idx = np.flatnonzero(m_sleep)
    if idx.size:
        cuts = np.array_split(idx, 3)
        for t, part in enumerate(cuts):
            out[f"third_{t}_sum"] = float(np.sum(values[part]))
            out[f"third_{t}_abs"] = float(np.sum(np.abs(values[part])))
    return out


# --- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=C.DEEP_ARM)
    ap.add_argument("--repeat", type=int, default=C.DEEP_REPEAT)
    ap.add_argument("--epoch-kind", default=C.DEEP_EPOCH_KIND,
                    choices=("marg", "inc"))
    ap.add_argument("--limit", type=int, default=0,
                    help="explain only the first N participants, for a pilot")
    ap.add_argument("--ig-steps", type=int, default=IG_STEPS)
    ap.add_argument("--no-ig", action="store_true")
    ap.add_argument("--second-baseline-every", type=int, default=4,
                    help="run the flat_97 baseline on every Nth participant")
    ap.add_argument("--keep-traces", type=int, default=40,
                    help="save full resolution traces for this many")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or ("_random" if args.random_weights else "")
    out_dir = C.ensure_results()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 78)
    print(f"AIM 4 DEEP ATTRIBUTION: arm {args.arm}, repeat {args.repeat}, "
          f"E*_{args.epoch_kind}")
    if args.random_weights:
        print("RANDOM WEIGHTS CONTROL: untrained network, same architecture")
    print("=" * 78)
    print(f"device {device}, integrated gradients {'off' if args.no_ig else args.ig_steps} steps")

    data = get_verified_folds(quiet=True)
    pid_all, y_all = data["pid"], data["y"]
    assign = fold_of_each_participant(data, args.repeat)
    store = D.SignalStore()

    order = np.argsort(assign, kind="stable")
    if args.limit:
        # take a few from each fold and outcome class so a pilot isn't all
        # from the first fold
        take = []
        per = max(1, args.limit // 5)
        for f in range(5):
            rows = np.flatnonzero(assign == f)
            ev = rows[y_all[rows] == 1][:max(1, per // 2)]
            ne = rows[y_all[rows] == 0][:per - len(ev)]
            take.append(np.concatenate([ev, ne]))
        order = np.concatenate(take)

    stride = 2 ** 5
    rows, traces = [], {}
    fault = {"logit_mismatch": 0, "ig_completeness": 0, "insane": 0}
    t0 = time.time()
    current_fold, nets = -1, None

    for count, row in enumerate(order):
        pid = int(pid_all[row])
        fold = int(assign[row])
        if fold != current_fold:
            if nets is not None:
                del nets
                if device.type == "mps":
                    torch.mps.empty_cache()
            nets = load_fold_models(args.arm, args.repeat, fold,
                                    args.epoch_kind, device,
                                    args.random_weights)
            current_fold = fold
            print(f"\n  fold {fold}: loaded 3 models at epoch "
                  f"{nets[0]['epoch']}", flush=True)

        item = single_item(store, pid, M.arm_input(args.arm))
        x = item["x"].to(device)
        mn = item["m_nrem"].to(device)
        mr = item["m_rem"].to(device)
        n = item["n"]
        pad = item["pad"]
        percent = item["percent"]
        m_sleep = item["m_sleep"]
        off = sleep_period_offset(store, pid)
        desat, n_desat = event_intervals(pid, DESAT_CONCEPTS, off, n)
        resp, n_resp = event_intervals(pid, RESP_CONCEPTS, off, n)

        rec = {"pid": pid, "y": int(y_all[row]), "fold": fold,
               "n_samples": n, "n_desat_events": n_desat,
               "n_resp_events": n_resp,
               "fill_percent": float(item["fill"])}

        # method 1: exact decomposition, averaged over the three models
        step_sum = None
        logits, recons = [], []
        for mdl in nets:
            ex = exact_step_contributions(mdl["net"], x, mn, mr)
            logits.append(ex["logit_actual"])
            recons.append(ex["logit_reconstructed"])
            step_sum = ex["contrib"] if step_sum is None \
                else step_sum + ex["contrib"]
            rec["rem_fallback"] = bool(ex["rem_fallback"])
        step_mean = step_sum / len(nets)
        rec["logit_mean"] = float(np.mean(logits))
        rec["z_mean"] = float(np.mean(
            [(l - m["mu"]) / (m["sd"] if m["sd"] > 1e-12 else 1.0)
             for l, m in zip(logits, nets)]))
        worst = float(np.max(np.abs(np.array(logits) - np.array(recons))))
        rec["exact_reconstruction_error"] = worst
        if worst > 1e-3:
            fault["logit_mismatch"] += 1
        if np.any(np.abs(logits) > C.SANE_LOGIT):
            fault["insane"] += 1

        # the step grid covers the padded night, so slice out the real window
        step_lo, step_hi = pad // stride, (pad + n) // stride + 1
        exact_samples = to_sample_grid(step_mean, stride,
                                       pad + n + stride)[pad:pad + n]
        p = profile(exact_samples, percent, item["night_nrem"],
                    item["night_rem"], desat, resp, m_sleep)
        for k, v in p.items():
            rec[f"exact_{k}"] = v

        # method 3: Grad-CAM and gradient times activation, from one backward
        # pass. Gradient times activation is the same thing without the channel
        # averaging, so any gap in how well each matches the exact
        # decomposition comes down to the averaging.
        cam_sum, gxa_sum = None, None
        for mdl in nets:
            cm = grad_cam(mdl["net"], x, mn, mr)
            cam_sum = cm["cam_signed"] if cam_sum is None \
                else cam_sum + cm["cam_signed"]
            gxa_sum = cm["grad_x_activation"] if gxa_sum is None \
                else gxa_sum + cm["grad_x_activation"]
        cam_mean = cam_sum / len(nets)
        gxa_mean = gxa_sum / len(nets)
        ref = step_mean[step_lo:step_hi]
        ref_window = step_mean[pad // stride:(pad + n) // stride]
        if step_hi > step_lo + 2:
            rec["cam_vs_exact_pearson"] = float(np.corrcoef(
                cam_mean[step_lo:step_hi], ref)[0, 1])
            rec["gxa_vs_exact_pearson"] = float(np.corrcoef(
                gxa_mean[step_lo:step_hi], ref)[0, 1])
        else:
            rec["cam_vs_exact_pearson"] = np.nan
            rec["gxa_vs_exact_pearson"] = np.nan
        cam_samples = to_sample_grid(cam_mean, stride,
                                     pad + n + stride)[pad:pad + n]
        pc = profile(cam_samples, percent, item["night_nrem"],
                     item["night_rem"], desat, resp, m_sleep)
        for k, v in pc.items():
            rec[f"cam_{k}"] = v

        # method 2: integrated gradients
        ig_store = {}
        if not args.no_ig:
            want = [("flat_own", float(item["fill"]), args.ig_steps)]
            if args.second_baseline_every and \
                    count % args.second_baseline_every == 0:
                want.append(("flat_97", BASELINE_97, IG_STEPS_FAR))
            for name, level, start_steps in want:
                x0 = torch.full_like(x, float(D.scale_signal(
                    np.float32(level))))
                acc, comp, fx, f0, used = None, [], [], [], []
                for mdl in nets:
                    # double the steps until completeness holds for this model
                    # and participant (up to IG_STEPS_CAP), so we don't save an
                    # attribution that's known not to add up
                    steps = start_steps
                    while True:
                        ig = integrated_gradients(mdl["net"], x, mn, mr, x0,
                                                  steps=steps)
                        if abs(ig["completeness_error"]) <= IG_ABS_TOLERANCE \
                                or steps >= IG_STEPS_CAP:
                            break
                        steps *= 2
                    used.append(steps)
                    acc = ig["attrib"] if acc is None else acc + ig["attrib"]
                    comp.append(ig["completeness_error"])
                    fx.append(ig["f_x"])
                    f0.append(ig["f_baseline"])
                acc = acc / len(nets)
                rec[f"ig_{name}_steps_used"] = int(max(used))
                span = float(np.mean(fx) - np.mean(f0))
                err = float(np.mean(comp))
                rel = abs(err) / max(1e-9, abs(span))
                rec[f"ig_{name}_f_x"] = float(np.mean(fx))
                rec[f"ig_{name}_f_baseline"] = float(np.mean(f0))
                rec[f"ig_{name}_span"] = span
                rec[f"ig_{name}_completeness_error"] = err
                rec[f"ig_{name}_completeness_relative"] = rel
                # Flag on the absolute error, since logits are order one across
                # this cohort. The relative error blows up when a night scores
                # almost the same as its own flat counterfactual (denominator
                # near zero), so it would flag the wrong people.
                if abs(err) > IG_ABS_TOLERANCE:
                    fault["ig_completeness"] += 1
                win = acc[pad:pad + n]
                ig_store[name] = win
                pi = profile(win, percent, item["night_nrem"],
                             item["night_rem"], desat, resp, m_sleep)
                for k, v in pi.items():
                    rec[f"ig_{name}_{k}"] = v
                if name == "flat_own":
                    by_step = step_sums(acc[pad:pad + n], stride)
                    k = min(len(by_step), len(ref_window))
                    rec["ig_vs_exact_pearson"] = float(
                        np.corrcoef(by_step[:k], ref_window[:k])[0, 1]) \
                        if k > 2 else np.nan
                    rec["ig_sample_vs_exact_pearson"] = float(np.corrcoef(
                        win, to_sample_grid(step_mean, stride,
                                            pad + n + stride)[pad:pad + n]
                    )[0, 1])

        rows.append(rec)
        if len(traces) < args.keep_traces:
            traces[f"{pid}_percent"] = percent.astype(np.float32)
            traces[f"{pid}_sleep"] = m_sleep
            traces[f"{pid}_rem"] = item["night_rem"]
            traces[f"{pid}_desat"] = desat
            traces[f"{pid}_exact_step"] = step_mean[step_lo:step_hi] \
                .astype(np.float32)
            traces[f"{pid}_cam_step"] = cam_mean[step_lo:step_hi] \
                .astype(np.float32)
            for name, v in ig_store.items():
                traces[f"{pid}_ig_{name}"] = v.astype(np.float32)

        if (count + 1) % 25 == 0:
            el = time.time() - t0
            print(f"    {count + 1}/{len(order)}  {el:.0f}s  "
                  f"{el / (count + 1):.2f}s each  "
                  f"eta {(len(order) - count - 1) * el / (count + 1) / 60:.0f} min",
                  flush=True)

    table = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"deep_attrib_{args.arm}{tag}.csv")
    table.to_csv(csv_path, index=False)
    np.savez_compressed(
        os.path.join(out_dir, f"deep_traces_{args.arm}{tag}.npz"), **traces)

    meta = {"arm": args.arm, "repeat": args.repeat,
            "epoch_kind": args.epoch_kind, "n_explained": len(table),
            "ig_steps": 0 if args.no_ig else args.ig_steps,
            "random_weights": bool(args.random_weights),
            "baseline_97_percent": BASELINE_97,
            "faults": fault,
            "worst_exact_reconstruction_error":
                float(table["exact_reconstruction_error"].max()),
            "seconds": time.time() - t0}
    if "ig_flat_own_completeness_relative" in table:
        meta["ig_completeness_relative_median"] = float(
            table["ig_flat_own_completeness_relative"].median())
        meta["ig_completeness_relative_worst"] = float(
            table["ig_flat_own_completeness_relative"].max())
        meta["ig_vs_exact_pearson_median"] = float(
            table["ig_vs_exact_pearson"].median())
    meta["cam_vs_exact_pearson_median"] = float(
        table["cam_vs_exact_pearson"].median())
    meta["gxa_vs_exact_pearson_median"] = float(
        table["gxa_vs_exact_pearson"].median())
    if "ig_flat_own_completeness_error" in table:
        meta["ig_completeness_abs_median"] = float(
            table["ig_flat_own_completeness_error"].abs().median())
        meta["ig_completeness_abs_worst"] = float(
            table["ig_flat_own_completeness_error"].abs().max())
        meta["ig_abs_tolerance"] = IG_ABS_TOLERANCE
        for name in ("flat_own", "flat_97"):
            col = f"ig_{name}_steps_used"
            if col in table:
                v = table[col].dropna()
                if len(v):
                    meta[f"{name}_steps_median"] = float(v.median())
                    meta[f"{name}_steps_max"] = float(v.max())
                    meta[f"{name}_refined_fraction"] = float(
                        (v > v.min()).mean())
    with open(os.path.join(out_dir,
                           f"deep_attrib_{args.arm}{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print(f"explained {len(table)} participants in "
          f"{(time.time() - t0) / 60:.1f} min")
    print(f"worst exact reconstruction error "
          f"{meta['worst_exact_reconstruction_error']:.2e}  "
          f"(this is the check that the decomposition is the network)")
    if "ig_completeness_relative_worst" in meta:
        print(f"integrated gradients completeness, relative: median "
              f"{meta['ig_completeness_relative_median']:.4f}, worst "
              f"{meta['ig_completeness_relative_worst']:.4f}")
        print(f"integrated gradients against the exact decomposition: "
              f"median r {meta['ig_vs_exact_pearson_median']:.3f}")
    if "ig_completeness_abs_worst" in meta:
        print(f"integrated gradients completeness, absolute logits: median "
              f"{meta['ig_completeness_abs_median']:.2e}, worst "
              f"{meta['ig_completeness_abs_worst']:.2e} "
              f"(tolerance {IG_ABS_TOLERANCE})")
    print(f"Grad-CAM against the exact decomposition: median r "
          f"{meta['cam_vs_exact_pearson_median']:.3f}")
    print(f"gradient times activation against it: median r "
          f"{meta['gxa_vs_exact_pearson_median']:.3f}  "
          f"(the gap isolates the cost of averaging gradients over steps)")
    print(f"faults: {fault}")
    print(f"wrote {csv_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
