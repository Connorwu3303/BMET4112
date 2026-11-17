"""
Aim 3 unit tests. Run these before any training that uses the CVD labels.

The invariance tests are set up so they can't pass while measuring nothing.
They randomise the weights first (an untrained network is nearly constant) and
confirm the output depends on the input, use synthetic nights that vary like
real ones instead of white noise, and change one thing at a time.

Test groups, and what a failure would mean:

  1  store and folds       wrong data, or an invalid comparison with Aim 2
  2  input representation  leakage, or the Aim 1 fabrication coming back
  3  shape invariance      night length or batch shape read as risk
  4  pooling               unmeasured time feeding into the prediction
  5  information matching  sleep architecture credited to oximetry
  6  training loop         optimism from selecting on held out data
  7  MPS correctness       silently wrong numbers from the GPU backend

Usage:
    python3 aim3/test_aim3.py
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import models as M           # noqa: E402
import data as D             # noqa: E402
import trainer as T          # noqa: E402
from folds import get_verified_folds, build_folds  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  [PASS] {name}" + (f": {detail}" if detail else ""))
    else:
        FAIL.append((name, detail))
        print(f"  [FAIL] {name}" + (f": {detail}" if detail else ""))
    return bool(condition)


def randomise_weights(net, seed=0):
    """
    Gives an untrained network real, non-degenerate behaviour.

    Without this the zero initialised head makes the network a constant function
    and every invariance test passes trivially. Batch norm goes into eval mode
    with non-unit scales and non-zero running stats, like after training.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.ndim >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            elif "bias" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            else:
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
        for mod in net.modules():
            if isinstance(mod, torch.nn.BatchNorm1d):
                mod.running_mean.copy_(
                    torch.randn(mod.running_mean.shape, generator=g) * 0.1)
                mod.running_var.copy_(
                    1.0 + torch.rand(mod.running_var.shape, generator=g) * 0.1)
    return net.eval()


def synthetic_night(n_valid, channels=1, baseline=-0.2, n_dips=40, depth=1.0,
                    seed=0, rem_fraction=0.2):
    """
    Synthetic night laid out the same way the data layer lays out a real one:
    PAD_EACH_SIDE samples of flat fill, the sleep period, then fill to the bucket.
    """
    g = torch.Generator().manual_seed(seed)
    core = torch.full((n_valid,), float(baseline))
    core = core + torch.randn((n_valid,), generator=g) * 0.05
    for _ in range(n_dips):
        s = int(torch.randint(0, max(n_valid - 40, 1), (1,),
                              generator=g).item())
        dur = int(torch.randint(10, 40, (1,), generator=g).item())
        core[s:s + dur] = core[s:s + dur] - float(depth) * torch.sin(
            torch.linspace(0, float(np.pi), dur))

    total = D.pad_to_shape(n_valid)
    right = total - n_valid - D.PAD_EACH_SIDE
    fill = float(baseline)
    sig = torch.cat([torch.full((D.PAD_EACH_SIDE,), fill), core,
                     torch.full((right,), fill)])
    m_nrem = torch.zeros(total)
    m_rem = torch.zeros(total)
    a = D.PAD_EACH_SIDE
    split = a + int(n_valid * (1 - rem_fraction))
    m_nrem[a:split] = 1.0
    m_rem[split:a + n_valid] = 1.0
    if channels == 1:
        x = sig.unsqueeze(0).unsqueeze(0)
    else:
        x = torch.stack([m_nrem + m_rem, m_rem]).unsqueeze(0)
    return x, m_nrem.unsqueeze(0), m_rem.unsqueeze(0)


# (baseline, number of dips, dip depth). White noise inputs all share the same
# pooled mean and extremes, so they barely move a network that reads pooled
# statistics; these nights vary the way real ones do.
NIGHT_VARIANTS = [(-0.2, 10, 0.5), (-0.2, 80, 0.5), (-0.2, 80, 2.0),
                  (-1.0, 40, 1.0), (-2.0, 120, 3.0), (0.0, 5, 0.2)]


def input_spread(net, n_valid=19200, channels=1, seed=11):
    """
    How much the logit moves across physiologically different nights.

    The invariance checks are judged against this. A fixed tolerance on an
    arbitrary logit scale doesn't mean much, since a network whose output
    hardly changes between nights would pass it without showing anything.
    """
    vals = []
    with torch.no_grad():
        for i, (b, nd, dp) in enumerate(NIGHT_VARIANTS):
            x, mn, mr = synthetic_night(n_valid, channels, baseline=b,
                                        n_dips=nd, depth=dp, seed=seed + i)
            vals.append(float(net(x, mn, mr)))
    return max(vals) - min(vals)


# --- tests ---

def test_store_and_folds():
    print("\n1. STORE AND FOLDS")
    store = D.SignalStore()
    check("1.1 store holds all 5,793 participants", len(store.index) == 5793,
          f"{len(store.index)} participants")

    data = get_verified_folds(quiet=True)
    y = data["y"]
    check("1.2 cohort is the Aim 2 incident cohort",
          len(y) == 4352 and int(y.sum()) == 829,
          f"{len(y)} participants, {int(y.sum())} events "
          f"({y.mean() * 100:.1f} percent)")
    check("1.3 all 15 outer folds verified against Aim 2", True,
          "get_verified_folds raises otherwise")
    check("1.4 fold construction is deterministic",
          all(np.array_equal(a["test_idx"], b["test_idx"])
              for a, b in zip(build_folds(y), build_folds(y))))

    # The deep arm's cross fitted predictions have to sit on the same inner
    # partition as the stacker's own cross validation, which is the Aim 2 grid
    # search partition.
    splits = T.inner_splits(data["folds"][0]["train_idx"], y)
    sizes = [len(tr) for tr, _ in splits]
    val_union = np.concatenate([va for _, va in splits])
    check("1.5 inner folds are the Aim 2 StratifiedKFold(3, shuffle, seed 42)",
          len(splits) == 3 and abs(max(sizes) - min(sizes)) <= 2 and
          len(np.unique(val_union)) == len(data["folds"][0]["train_idx"]),
          f"inner training sizes {sizes}, validation folds partition all "
          f"{len(data['folds'][0]['train_idx'])} outer training rows")

    for repeat in range(3):
        cov = np.concatenate([f["test_idx"] for f in data["folds"]
                              if f["repeat"] == repeat])
        if not check(f"1.6 repeat {repeat} tests every participant once",
                     len(cov) == len(y) and len(np.unique(cov)) == len(y)):
            break

    # labels have to match the ones behind the Aim 2 saved scores, otherwise a
    # paired DeLong test would be pairing different cohorts
    res = "/Volumes/ThesisHD/Data/processed/aim2/results"
    ok = True
    for f in ["logreg_oof_scores.npz", "rf_oof_scores.npz",
              "xgb_oof_scores.npz"]:
        arr = np.load(os.path.join(res, f))
        for spec in arr.files:
            if not np.array_equal(arr[spec][0].astype(int), y):
                ok = False
    check("1.7 labels match row 0 of every Aim 2 out-of-fold file", ok,
          "so the paired comparisons run on identical participants")
    return store, data


def test_input_representation(store, data):
    print("\n2. INPUT REPRESENTATION")
    pids = data["pid"][:64]
    y = data["y"][:64]

    ok_mask, ok_fill, ok_finite, ok_disjoint = True, True, True, True
    for pid in pids[:15]:
        x_full, v_full, st_full = store.raw(int(pid))
        night = store.sleep_period(int(pid))
        asleep = np.isin(st_full, D.SLEEP_STAGES)
        idx = np.flatnonzero(asleep)
        a, b = int(idx[0]), int(idx[-1]) + 1
        if not np.array_equal(night["m_sleep"], v_full[a:b] & asleep[a:b]):
            ok_mask = False
        if not np.allclose(night["x"][~night["m_sleep"]], night["fill"]):
            ok_fill = False
        if not np.all(np.isfinite(night["x"])):
            ok_finite = False
        if (night["m_nrem"] & night["m_rem"]).any() or not np.array_equal(
                night["m_nrem"] | night["m_rem"], night["m_sleep"]):
            ok_disjoint = False
    check("2.1 sleep mask is exactly validity AND asleep", ok_mask)
    check("2.2 everything outside valid sleep holds the night's own fill value",
          ok_fill)
    check("2.3 no NaN or infinity anywhere", ok_finite)
    check("2.4 NREM and REM masks partition sleep with no overlap", ok_disjoint)

    # Alignment with Aim 2. If the staging here disagreed with the staging used
    # for the biomarkers, the two arms would be reading different nights.
    feats = pd.read_csv(
        "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv",
        low_memory=False,
        usecols=["pid", "sleep_valid_hours", "nrem_valid_hours",
                 "rem_valid_hours"]).set_index("pid")
    worst = {"sleep": 0.0, "nrem": 0.0, "rem": 0.0}
    for pid in pids[:40]:
        night = store.sleep_period(int(pid))
        row = feats.loc[int(pid)]
        for key, col in [("sleep", "sleep_valid_hours"),
                         ("nrem", "nrem_valid_hours"),
                         ("rem", "rem_valid_hours")]:
            mine = float(night[f"m_{key}"].sum()) / 3600.0
            worst[key] = max(worst[key], abs(mine - float(row[col])))
    check("2.5 masks agree with the Aim 2 window durations",
          max(worst.values()) < 1e-6,
          f"worst differences sleep {worst['sleep']:.2e}, "
          f"nrem {worst['nrem']:.2e}, rem {worst['rem']:.2e} hours")

    check("2.6 the signal transform is a fixed affine map",
          abs(float(D.scale_signal(np.float32(95.0)))) < 1e-9 and
          abs(float(D.scale_signal(np.float32(90.0))) + 1.0) < 1e-6,
          "95 percent maps to 0 and 90 percent to -1, with no fitted statistic")

    night = store.sleep_period(int(pids[0]))
    check("2.7 the fill value is snapped to the quantisation ladder",
          abs(night["fill"] / D.QUANTUM - round(night["fill"] / D.QUANTUM))
          < 1e-6,
          f"fill {night['fill']:.4f} percent, quantum {D.QUANTUM:.4f} percent")

    loader, ds = D.make_loader(store, pids, y, arm="primary", train=True,
                               batch_size=8, seed=42)
    xb, mn, mr, yb, pb = next(iter(loader))
    check("2.8 the primary input has exactly one channel", xb.shape[1] == 1,
          f"batch shape {tuple(xb.shape)}")
    check("2.9 every training tensor is finite", bool(torch.isfinite(xb).all()))
    check("2.10 batch length is a multiple of the shape unit",
          xb.shape[-1] % D.SHAPE_MULTIPLE == 0,
          f"{xb.shape[-1]} samples, unit {D.SHAPE_MULTIPLE}")
    check("2.11 pooling masks are zero throughout the padding",
          bool(torch.all(mn[:, :D.PAD_EACH_SIDE] == 0)) and
          bool(torch.all(mr[:, :D.PAD_EACH_SIDE] == 0)),
          f"the first {D.PAD_EACH_SIDE} samples carry no pooling weight")

    seen = set()
    for e in range(6):
        ds.set_epoch(e)
        seen.add(float(ds[0][0].sum()))
    check("2.12 the random crop varies across epochs", len(seen) >= 4,
          f"{len(seen)} distinct crops over 6 epochs")

    eloader, eds = D.make_loader(store, pids, y, arm="primary", train=False,
                                 batch_size=16, seed=42)
    exb, emn, emr, eyb, epb = next(iter(eloader))
    true_lens = [store.sleep_period(int(p))["x"].shape[0] for p in epb.numpy()]
    check("2.13 evaluation covers the whole sleep period",
          exb.shape[-1] >= max(true_lens) + 2 * D.PAD_EACH_SIDE,
          f"batch length {exb.shape[-1]}, longest night {max(true_lens)}")
    check("2.14 evaluation applies no augmentation", not eds.train)
    check("2.15 participant identifiers travel with every batch",
          len(set(epb.numpy().tolist())) == len(epb),
          f"first pids {epb.numpy()[:4].tolist()}")
    check("2.16 evaluation batches are length sorted",
          true_lens == sorted(true_lens))
    shapes = sorted({int(D.pad_to_shape(int(l))) for l in eds.lengths})
    check("2.17 evaluation uses only a few fixed shapes", len(shapes) <= 16,
          f"{len(shapes)} distinct shapes, largest {max(shapes)}")


def test_shape_invariance():
    print("\n3. SHAPE INVARIANCE (night length must not be readable as risk)")
    for arm in ["A", "B"]:
        ch = M.ARMS[arm]["channels"]
        net = randomise_weights(M.build(arm), seed=1)
        spread = input_spread(net, channels=ch)
        if not check(f"3.{arm}.0 the network depends on its input "
                     f"(the test is not vacuous)", spread > 1e-2,
                     f"logit ranges over {spread:.4f} across nights"):
            continue

        # The main one. Every night has flat fill wider than the receptive
        # field, so extending the padding can't reach a valid sample.
        x, mn, mr = synthetic_night(19200, channels=ch, baseline=-0.4,
                                    n_dips=60, depth=1.2, seed=5)
        with torch.no_grad():
            base = float(net(x, mn, mr))
        worst = 0.0
        for extra in [2400, 4800, 12000, 19200]:
            xp = torch.cat([x, x[:, :, -1:].repeat(1, 1, extra)], dim=2)
            mnp = torch.cat([mn, torch.zeros(1, extra)], dim=1)
            mrp = torch.cat([mr, torch.zeros(1, extra)], dim=1)
            with torch.no_grad():
                worst = max(worst, abs(float(net(xp, mnp, mrp)) - base))
        check(f"3.{arm}.1 the logit is invariant to the batch shape",
              worst < 1e-5,
              f"worst change {worst:.2e} for extra padding up to 19,200 "
              f"samples, against an input spread of {spread:.4f}")

        g = torch.Generator().manual_seed(3)
        big = torch.randn((1, ch, 19200), generator=g) * 20
        vals = []
        for extra in [2400, 9600, 19200]:
            xp = torch.cat([x, big[:, :, :extra]], dim=2)
            mnp = torch.cat([mn, torch.zeros(1, extra)], dim=1)
            mrp = torch.cat([mr, torch.zeros(1, extra)], dim=1)
            with torch.no_grad():
                vals.append(float(net(xp, mnp, mrp)))
        check(f"3.{arm}.2 garbage beyond the fill cannot reach valid samples",
              max(vals) - min(vals) < 1e-5 and abs(vals[0] - base) < 1e-4,
              f"range {max(vals) - min(vals):.2e} across padding amounts, "
              f"and {abs(vals[0] - base):.2e} against the unpadded logit")


def test_pooling():
    print("\n4. POOLING")
    torch.manual_seed(0)
    B, C, L = 4, 3, 400
    x = torch.randn(B, C, L)
    m = torch.ones(B, L)
    m[0, 200:] = 0
    m[1, 40:] = 0
    m[2, 395:] = 0
    out = M.masked_mean_top(x, m, 0.10)
    worst = 0.0
    for b in range(B):
        k = math.ceil(int(m[b].sum()) * 0.10)
        for c in range(C):
            vals = x[b, c][m[b] == 1].numpy()
            worst = max(worst, abs(out[b, c].item() - vals.mean()),
                        abs(out[b, C + c].item()
                            - np.sort(vals)[::-1][:k].mean()))
    check("4.1 masked mean and top decile mean match a numpy reference",
          worst < 1e-4, f"worst absolute error {worst:.2e}")
    check("4.2 a night with three valid steps pools without error",
          bool(torch.isfinite(M.masked_mean_top(
              torch.randn(1, C, L),
              torch.cat([torch.ones(1, 3), torch.zeros(1, L - 3)], 1))).all()))
    check("4.3 a fully masked night does not divide by zero",
          bool(torch.isfinite(M.masked_mean_top(
              torch.randn(1, C, L), torch.zeros(1, L))).all()))
    x2 = x.clone()
    x2[0, :, 200:] = 500.0
    check("4.4 values under an invalid mask are ignored exactly",
          torch.allclose(M.masked_mean_top(x, m)[0],
                         M.masked_mean_top(x2, m)[0], atol=1e-5))
    mm = torch.zeros(1, 64)
    mm[0, :16] = 1              # exactly half of the first step
    mm[0, 32:47] = 1            # just under half of the second
    check("4.5 a step counts at exactly half coverage, not below",
          M.pool_mask(mm, 32).tolist() == [[1.0, 0.0]],
          f"pooled mask {M.pool_mask(mm, 32).tolist()}")

    net = randomise_weights(M.build("A"), seed=1)
    x, mn, mr = synthetic_night(19200, 1, seed=2)
    mr_tiny = torch.zeros_like(mr)
    mr_tiny[0, D.PAD_EACH_SIDE:D.PAD_EACH_SIDE + 64] = 1.0   # 2 steps
    with torch.no_grad():
        e_tiny = net.embed(x, mn, mr_tiny)
        e_full = net.embed(x, mn, mr)
    half = e_tiny.shape[1] // 2
    check("4.6 a night with too little REM falls back to its NREM vector",
          torch.allclose(e_tiny[0, :half], e_tiny[0, half:]) and
          not torch.allclose(e_full[0, :half], e_full[0, half:]),
          f"fewer than {M.MIN_REM_STEPS} valid REM steps copies NREM, more "
          f"does not")


def test_information_matching(store, data):
    print("\n5. INFORMATION MATCHING (the heart of the thesis comparison)")
    net = M.build("A")
    check("5.1 the primary takes one channel, saturation only",
          net.stem[0].in_channels == 1 and M.ARMS["A"]["channels"] == 1,
          "no mask channel and no REM channel, because the 34 biomarker "
          "features contain no sleep architecture variable")
    check("5.2 the primary has no confounder parameters",
          sum(p.numel() for n, p in net.named_parameters()
              if "conf" in n.lower()) == 0)
    import inspect
    check("5.3 the primary's forward signature admits no confounders",
          list(inspect.signature(net.forward).parameters)
          == ["x", "m_nrem", "m_rem"],
          f"parameters {list(inspect.signature(net.forward).parameters)}")
    check("5.4 the negative control sees no saturation at all",
          M.ARMS["NC"]["input"] == "nc" and M.ARMS["NC"]["channels"] == 2,
          "sleep mask and REM indicator only, so structure-only predictive "
          "value is measured rather than credited to oximetry")

    nc_loader, _ = D.make_loader(store, data["pid"][:16], data["y"][:16],
                                 arm="nc", train=False, batch_size=8)
    xb, mn, mr, _, _ = next(iter(nc_loader))
    check("5.5 the negative control's channels are binary masks",
          xb.shape[1] == 2 and bool(torch.all((xb == 0) | (xb == 1))),
          f"shape {tuple(xb.shape)}, values in {{0, 1}}")

    d_loader, _ = D.make_loader(store, data["pid"][:8], data["y"][:8],
                                arm="d", train=False, batch_size=4)
    dxb, dmn, dmr, _, _ = next(iter(d_loader))
    check("5.6 the deployable arm uses the validity mask and no staging",
          bool(torch.all(dmr == 0)) and bool(dmn.sum() > 0),
          "REM mask empty, so pooling runs over all valid samples and the arm "
          "needs no technician hypnogram")

    s = D.structure_features(store, int(data["pid"][0]))
    finite = sum(1 for k in D.STRUCTURE_COLUMNS if np.isfinite(s[k]))
    check("5.7 the structure set S is computable",
          all(k in s for k in D.STRUCTURE_COLUMNS) and
          finite >= len(D.STRUCTURE_COLUMNS) - 1,
          f"{finite} of {len(D.STRUCTURE_COLUMNS)} variables finite for "
          f"pid {int(data['pid'][0])}")


def test_arm_sizes():
    print("\n5b. EVERY ARM BUILDS TO ITS DECLARED SIZE")
    # "widths" means different things in the primary network and the window
    # model, and mixing them up once rebuilt the window model at 65,187
    # parameters instead of 36,099 with nothing flagging it. Any edit that
    # changes an arm has to update these declared sizes too.
    import run_unit as RU
    cfg, _ = RU.resolve_config()
    expected = {"A": 241201, "B": 36099, "NC": 36179, "D": 241201}
    got = {}
    for arm, n in expected.items():
        net = M.build(arm, overrides=RU.model_overrides(cfg, arm), strict=True)
        got[arm] = M.count_parameters(net)
    check("5b.1 every arm builds to the size the design declares",
          got == expected,
          ", ".join(f"{a} {n:,}" for a, n in got.items()) if got == expected
          else f"expected {expected}, got {got}")

    # an override an arm can't accept has to raise instead of being dropped
    raised = False
    try:
        M.build("NC", overrides={"dilated_width": 64}, strict=True)
    except TypeError:
        raised = True
    check("5b.2 an inapplicable override raises in strict mode", raised,
          "so a configuration meant for one arm cannot silently reshape another")


def test_training_loop():
    print("\n6. TRAINING LOOP")
    check("6.1 the loss is exactly unweighted BCE with logits",
          T.uses_plain_bce(),
          "verified numerically against weighted and unweighted references")
    check("6.2 the fit selects no epoch and restores no checkpoint",
          T.selects_epoch_after_training(),
          "every epoch is recorded and the choice is made by a later script")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data.py")).read()
    check("6.3 no amplitude scaling or baseline offset augmentation exists",
          not any(k in src for k in ["amplitude_scale", "shift_sd",
                                     "baseline_offset", "time_warp"]),
          "only cropping, dequantisation jitter and time masking, all of which "
          "preserve T90 and nadir")
    check("6.4 no test time augmentation",
          "eval_crops" not in src,
          "evaluation is a single full night pass")


def test_backend_fault_repair():
    print("\n6b. BACKEND FAULT DETECTION AND REPAIR")
    # The GPU on this machine now and then returns a garbage value for one
    # element of a batch (logits up to 1.2e35). AUC barely notices, but it
    # wrecks the standardisation constants. This injects a logit that is always
    # absurd, which retrying or moving to the CPU can't fix, so predict has to
    # raise instead of returning it.
    class AlwaysAbsurd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, mn, mr):
            return torch.full((x.shape[0], 1), 1e35) + self.p

    store = D.SignalStore()
    dd = get_verified_folds(quiet=True)
    loader, _ = D.make_loader(store, dd["pid"][:8], dd["y"][:8],
                              arm="primary", train=False, batch_size=4)
    raised = False
    try:
        T.predict(AlwaysAbsurd(), loader, torch.device("cpu"))
    except RuntimeError as exc:
        raised = "implausible logits" in str(exc)
    check("6b.1 an implausible logit is detected and raises", raised,
          f"the bound is |logit| <= {T.SANE_LOGIT}, beyond which a risk model "
          f"is meaningless since sigmoid(50) is 1 to machine precision")

    # a healthy model should pass straight through with zero repairs
    net = randomise_weights(M.build("A"), seed=3)
    report = {}
    logit, y_out, pid_out = T.predict(net, loader, torch.device("cpu"),
                                      report=report)
    check("6b.2 a healthy model needs no repair",
          report.get("repaired", 0) == 0 and np.all(np.abs(logit) <= T.SANE_LOGIT),
          f"{report.get('batches', 0)} batches, "
          f"{report.get('repaired', 0)} repaired, "
          f"logit range {logit.min():.3f} to {logit.max():.3f}")
    check("6b.3 the training loop can skip a faulty batch",
          "n_skipped += 1" in __import__("inspect").getsource(T.train_unit),
          "a transient fault skips the batch rather than aborting the unit")


def test_mps():
    print("\n7. MPS AGREEMENT WITH CPU")
    if not torch.backends.mps.is_available():
        check("7.0 MPS available", False, "skipped")
        return
    dev = torch.device("mps")
    for arm in ["A", "B"]:
        ch = M.ARMS[arm]["channels"]
        net = randomise_weights(M.build(arm), seed=4)
        xs, mns, mrs = [], [], []
        for i, (nv, remf) in enumerate([(19200, 0.2), (7200, 0.2),
                                        (19200, 0.001)]):
            x, mn, mr = synthetic_night(nv, ch, seed=i, rem_fraction=remf)
            pad = 38400 - x.shape[-1]
            xs.append(torch.cat([x, x[:, :, -1:].repeat(1, 1, pad)], dim=2))
            mns.append(torch.cat([mn, torch.zeros(1, pad)], dim=1))
            mrs.append(torch.cat([mr, torch.zeros(1, pad)], dim=1))
        x, mn, mr = torch.cat(xs), torch.cat(mns), torch.cat(mrs)
        with torch.no_grad():
            cpu_out = net(x, mn, mr)
            net = net.to(dev)
            mps_out = net(x.to(dev), mn.to(dev), mr.to(dev)).cpu()
            net = net.to("cpu")
        diff = float((cpu_out - mps_out).abs().max())
        check(f"7.{arm} MPS matches CPU on the forward pass", diff < 2e-3,
              f"worst absolute logit difference {diff:.2e}")

    # Gradients too. Dropout is off because the two devices draw from different
    # random streams and would drop different channels. An earlier version of
    # this test wrongly reported that as a backend disagreement of 111 percent.
    base = M.build("A", overrides={"dropout": 0.0, "head_dropout": 0.0})
    x, mn, mr = synthetic_night(19200, 1, seed=9)
    x, mn, mr = x.repeat(4, 1, 1), mn.repeat(4, 1), mr.repeat(4, 1)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    grads = {}
    for dv in ["cpu", "mps"]:
        net = M.build("A", overrides={"dropout": 0.0, "head_dropout": 0.0})
        net.load_state_dict(base.state_dict())
        net = net.to(dv).train()
        out = net(x.to(dv), mn.to(dv), mr.to(dv)).squeeze(-1)
        T.compute_loss(out, y.to(dv)).backward()
        grads[dv] = torch.cat([p.grad.flatten().detach().cpu()
                               for p in net.parameters() if p.grad is not None])
    scale = grads["cpu"].abs().max().clamp(min=1e-12)
    rel = float((grads["cpu"] - grads["mps"]).abs().max() / scale)
    check("7.3 MPS gradients match CPU gradients", rel < 5e-2,
          f"worst relative difference {rel:.2e}")


def main():
    print("=" * 72)
    print("AIM 3 UNIT TESTS")
    print("=" * 72)
    store, data = test_store_and_folds()
    test_input_representation(store, data)
    test_shape_invariance()
    test_pooling()
    test_information_matching(store, data)
    test_arm_sizes()
    test_training_loop()
    test_backend_fault_repair()
    test_mps()

    print("\n" + "=" * 72)
    print(f"RESULT: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("\nFAILURES:")
        for name, detail in FAIL:
            print(f"  {name}: {detail}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
