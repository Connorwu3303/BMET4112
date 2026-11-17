"""
Aim 4: unit tests.

Every invariance or identity test first checks that the quantity being tested
moves when the input moves, since a constant function passes them all. Where a
trained model isn't needed the weights are randomised, because architecture
A's head is zero initialised and would otherwise make every attribution zero.

Groups 0 to 10, in order: the test network responds (without it every other
group is vacuous), cohort and folds, the exact decomposition, integrated
gradients, the gradient identity, Grad-CAM, classical SHAP, occlusion,
aggregation arithmetic, event placement and hygiene.

Run:
    python3 test_aim4.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common4 as C                                      # noqa: E402

import data as D                                         # noqa: E402
import models as M                                       # noqa: E402

sys.path.insert(0, C.AIM3_DIR)
from test_aim3 import randomise_weights, synthetic_night  # noqa: E402

import deep_attrib as A                                  # noqa: E402
import occlusion as OC                                   # noqa: E402
import shap_classical as S                               # noqa: E402
import headtohead as H                                   # noqa: E402

PASSED, FAILED = [], []


def check(group, name, condition, detail=""):
    (PASSED if condition else FAILED).append((group, name, detail))
    mark = "pass" if condition else "FAIL"
    print(f"  [{mark}] {group}.{name}" + (f"  {detail}" if detail else ""))
    return bool(condition)


def small_net(seed=0, channels=1):
    """
    A cut down architecture A so the tests run in seconds.

    Only the dilated stack is reduced. Widths, kernels, pooling and head are
    the production ones. With the widths cut to 8 the network was almost
    constant (six distinct synthetic nights spanned 0.04 logits), so every
    identity test would pass trivially. At the production widths they span 2.2.
    """
    net = M.SpO2Net(n_channels=channels, widths=(16, 32, 48, 64, 64),
                    kernels=(7, 7, 5, 5, 5), dilated_width=64,
                    dilations=(1, 2, 4), dropout=0.0, head_dropout=0.0)
    return randomise_weights(net, seed=seed)


def night_pair(n_valid=9600, seed=3):
    """
    A severe night and a mild one, differing in both level and desaturations.

    A real severe vs mild contrast involves both. Varying only the dips also
    gives two nights this architecture scores almost the same, because the
    pooled mean over several hundred steps absorbs a change in the number of
    dips far more easily than a shift in where the whole night sits.
    """
    severe = synthetic_night(n_valid, baseline=-1.2, n_dips=90, depth=2.0,
                             seed=seed)
    mild = synthetic_night(n_valid, baseline=-0.1, n_dips=5, depth=0.2,
                           seed=seed)
    return severe, mild


# --- 1. cohort and folds ---

def group1_cohort_and_folds():
    print("\n1. COHORT AND FOLDS")
    from folds import get_verified_folds
    data = get_verified_folds(quiet=True)
    y, pid = data["y"], data["pid"]
    check(1, "cohort_size", len(y) == 4352, f"{len(y)} participants")
    check(1, "event_count", int(y.sum()) == 829, f"{int(y.sum())} events")

    assign = A.fold_of_each_participant(data, 0)
    counts = np.bincount(assign, minlength=5)
    check(1, "test_folds_partition_cohort", counts.sum() == len(y),
          f"fold sizes {counts.tolist()}")
    check(1, "every_participant_has_a_fold", bool(np.all(assign >= 0)))

    # row order of the modelling dataset has to match the fold source, or every
    # fold index would point at a different participant in the two files
    import ml_common as MC
    _, y2, _, meta = MC.load_dataset("obm_plus_confounders")
    check(1, "dataset_row_order_matches_folds",
          bool(np.array_equal(np.asarray(meta["pid"]).astype(int),
                              pid.astype(int))))
    check(1, "outcome_matches_folds", bool(np.array_equal(y2, y)))

    # nobody can be a test case twice in one repeat. True by construction above,
    # but checked here in case the fold source ever changes
    for repeat in (0, 1, 2):
        a = A.fold_of_each_participant(data, repeat)
        check(1, f"repeat{repeat}_folds_disjoint",
              bool(np.all(np.bincount(a, minlength=5) > 800)),
              f"sizes {np.bincount(a, minlength=5).tolist()}")


# --- 0. the test network responds ---

def group0_the_test_network_responds():
    print("\n0. THE TEST NETWORK IS NOT A CONSTANT FUNCTION")
    # The rest of the suite means nothing without this group. A constant
    # function trivially passes every identity and invariance test below, so the
    # test network has to be shown to respond to its input first.
    net = small_net(seed=1)
    logits = []
    for base, dips, depth in ((-0.2, 10, 0.5), (-0.2, 80, 2.0), (-1.0, 40, 1.0),
                              (-1.2, 90, 2.0), (-0.1, 5, 0.2)):
        x, mn, mr = synthetic_night(9600, baseline=base, n_dips=dips,
                                    depth=depth, seed=3)
        logits.append(A.exact_step_contributions(net, x, mn, mr)
                      ["logit_actual"])
    spread = max(logits) - min(logits)
    check(0, "logit_spans_physiological_nights", spread > 0.5,
          f"logit spans {spread:.3f} over five night types: "
          + ", ".join(f"{v:+.3f}" for v in logits))
    distinct = len({round(v, 4) for v in logits})
    check(0, "every_night_type_scores_differently", distinct == len(logits),
          f"{distinct} distinct logits from {len(logits)} nights")


# --- 2. exact step decomposition ---

def group2_exact_decomposition():
    print("\n2. THE EXACT STEP DECOMPOSITION")
    net = small_net(seed=1)
    (x, mn, mr), (x2, mn2, mr2) = night_pair()

    ex = A.exact_step_contributions(net, x, mn, mr)
    err = abs(ex["logit_actual"] - ex["logit_reconstructed"])
    check(2, "reconstructs_the_logit", err < 1e-4,
          f"error {err:.2e}, logit {ex['logit_actual']:+.4f}")

    # input dependence first, otherwise the identity above could hold for a
    # constant network and prove nothing
    ex2 = A.exact_step_contributions(net, x2, mn2, mr2)
    spread = abs(ex["logit_actual"] - ex2["logit_actual"])
    check(2, "responds_to_different_nights", spread > 1e-3,
          f"logit moved {spread:.4f} between a severe and a mild night")
    check(2, "contributions_differ_between_nights",
          float(np.max(np.abs(ex["contrib"] - ex2["contrib"]))) > 1e-6)

    # only steps inside a pooling mask may contribute. A non-zero contribution
    # from a padding step would mean the per step attribution includes time that
    # was never measured.
    m_any = ((mn + mr) > 0).float()
    length = ex["contrib"].shape[0]
    valid = M._match_length(M.pool_mask(m_any, net.total_stride),
                            length)[0].numpy() > 0
    outside = float(np.max(np.abs(ex["contrib"][~valid]))) \
        if (~valid).any() else 0.0
    check(2, "padding_contributes_nothing", outside < 1e-6,
          f"worst contribution outside the mask {outside:.2e}")

    # the REM fallback has to reconstruct too. A night with almost no REM takes
    # a different branch through the pooling, and that's the branch a
    # reimplementation is most likely to get wrong.
    xs, mns, mrs = synthetic_night(9600, n_dips=40, seed=5, rem_fraction=0.0005)
    exs = A.exact_step_contributions(net, xs, mns, mrs)
    check(2, "rem_fallback_branch_taken", exs["rem_fallback"],
          f"{exs['n_steps_rem']} valid REM steps")
    check(2, "rem_fallback_reconstructs",
          abs(exs["logit_actual"] - exs["logit_reconstructed"]) < 1e-4,
          f"error {abs(exs['logit_actual'] - exs['logit_reconstructed']):.2e}")

    # and the two channel arm: same head shape, different input shape
    netc = small_net(seed=7, channels=2)
    xc, mnc, mrc = synthetic_night(9600, channels=2, seed=9)
    exc = A.exact_step_contributions(netc, xc, mnc, mrc)
    check(2, "two_channel_arm_reconstructs",
          abs(exc["logit_actual"] - exc["logit_reconstructed"]) < 1e-4)


# --- 3. integrated gradients ---

def group3_integrated_gradients():
    print("\n3. INTEGRATED GRADIENTS")
    net = small_net(seed=2)
    (x, mn, mr), _ = night_pair(seed=11)
    # baseline is a flat healthy level so the path to a severe night covers a
    # real change in logit. Using the night's own level would make the span near
    # zero, and then any implementation returning something small would pass
    # completeness.
    x0 = torch.full_like(x, -0.1)

    ig = A.integrated_gradients(net, x, mn, mr, x0, steps=64)
    span = ig["f_x"] - ig["f_baseline"]
    check(3, "the_span_is_not_degenerate", abs(span) > 1e-3,
          f"f(x) - f(baseline) = {span:+.4f}")
    check(3, "completeness", abs(ig["completeness_error"]) < 0.02,
          f"error {ig['completeness_error']:+.2e} on a span of {span:+.4f}")

    # more integration steps should shrink the error; that's what separates a
    # discretisation error from a bug in the path
    coarse = A.integrated_gradients(net, x, mn, mr, x0, steps=4)
    check(3, "refinement_reduces_the_error",
          abs(ig["completeness_error"]) <= abs(coarse["completeness_error"])
          + 1e-9,
          f"4 steps {abs(coarse['completeness_error']):.2e} to 64 steps "
          f"{abs(ig['completeness_error']):.2e}")

    # a null path has to give a null attribution. An implementation returning
    # the gradient instead of gradient times difference would pass completeness
    # on most inputs and only fail here.
    ig0 = A.integrated_gradients(net, x, mn, mr, x.clone(), steps=8)
    check(3, "identical_baseline_gives_zero",
          float(np.max(np.abs(ig0["attrib"]))) < 1e-6,
          f"worst {float(np.max(np.abs(ig0['attrib']))):.2e}")

    # attribution should only sit where the input differs from the baseline,
    # since the (x - x0) factor is exactly zero everywhere else
    diff = (x - x0)[0, 0].numpy()
    same = np.abs(diff) < 1e-12
    if same.any():
        check(3, "zero_where_input_equals_baseline",
              float(np.max(np.abs(ig["attrib"][same]))) < 1e-9)

    # averaging over models is exact for IG since it's linear in the model
    # output. The full run relies on this, so check it.
    net_b = small_net(seed=12)
    a1 = A.integrated_gradients(net, x, mn, mr, x0, steps=16)["attrib"]
    a2 = A.integrated_gradients(net_b, x, mn, mr, x0, steps=16)["attrib"]
    mean_of_attrib = (a1 + a2) / 2.0

    class Mean(torch.nn.Module):
        def __init__(self, nets):
            super().__init__()
            self.nets = torch.nn.ModuleList(nets)

        def forward(self, x, mn, mr):
            return sum(n(x, mn, mr) for n in self.nets) / len(self.nets)

    attrib_of_mean = A.integrated_gradients(
        Mean([net, net_b]).eval(), x, mn, mr, x0, steps=16)["attrib"]
    worst = float(np.max(np.abs(mean_of_attrib - attrib_of_mean)))
    check(3, "averaging_models_equals_averaging_attributions", worst < 1e-6,
          f"worst difference {worst:.2e}")


# --- 4. gradient identity ---

def group4_gradient_identity():
    print("\n4. GRADIENT TIMES ACTIVATION EQUALS THE EXACT DECOMPOSITION")
    # The head is linear in the pooled vector and the pooling is a fixed
    # weighted sum over steps, so d(logit)/d(feature) times the feature is
    # exactly that feature's contribution. The decomposition and the backward
    # pass are separate code paths, so if this fails one of them is wrong.
    net = small_net(seed=4)
    (x, mn, mr), _ = night_pair(seed=13)
    ex = A.exact_step_contributions(net, x, mn, mr)
    cm = A.grad_cam(net, x, mn, mr)
    gxa = cm["grad_x_activation"]
    ref = ex["contrib"]
    check(4, "the_reference_is_not_constant", float(np.std(ref)) > 1e-8,
          f"standard deviation {float(np.std(ref)):.2e}")
    r = float(np.corrcoef(gxa, ref)[0, 1])
    check(4, "correlates_at_one", r > 0.999, f"Pearson r = {r:.6f}")
    scale = max(1e-12, float(np.max(np.abs(ref))))
    worst = float(np.max(np.abs(gxa - ref))) / scale
    check(4, "agrees_elementwise", worst < 0.02,
          f"worst relative difference {worst:.4f}")


# --- 5. Grad-CAM ---

def group5_grad_cam():
    print("\n5. GRAD-CAM AND WHY IT DISAGREES")
    net = small_net(seed=6)
    (x, mn, mr), _ = night_pair(seed=17)
    ex = A.exact_step_contributions(net, x, mn, mr)
    cam = A.grad_cam(net, x, mn, mr)["cam_signed"]
    r_full = float(np.corrcoef(cam, ex["contrib"])[0, 1])
    check(5, "grad_cam_responds_to_input", float(np.std(cam)) > 1e-10)

    # Now zero the top decile half of the head, leaving mean pooling only. With
    # mean pooling the per step gradient is constant across steps, so Grad-CAM's
    # channel averaging loses nothing and it should match the exact
    # decomposition. If agreement jumps, the disagreement above comes from the
    # averaging interacting with top decile selection, which is down to the
    # architecture and isn't a coding error in either method.
    width = net.head.weight.shape[1] // 4
    with torch.no_grad():
        w = net.head.weight.clone()
        w[0, width:2 * width] = 0.0
        w[0, 3 * width:] = 0.0
        net.head.weight.copy_(w)
    ex_mean = A.exact_step_contributions(net, x, mn, mr)
    cam_mean = A.grad_cam(net, x, mn, mr)["cam_signed"]
    r_meanonly = float(np.corrcoef(cam_mean, ex_mean["contrib"])[0, 1])
    check(5, "mean_pooling_only_restores_agreement", r_meanonly > r_full,
          f"r rises from {r_full:.3f} with top decile pooling to "
          f"{r_meanonly:.3f} without it")


# --- 6. classical SHAP ---

def group6_classical_shap():
    print("\n6. CLASSICAL SHAP")
    rng = np.random.default_rng(0)
    n, p = 400, 6
    X = rng.normal(size=(n, p))
    X[:, 3] = X[:, 0] * 0.97 + rng.normal(scale=0.2, size=n)   # collinear pair
    y = (X @ np.array([1.5, -1.0, 0.0, 0.5, 0.2, 0.0])
         + rng.normal(scale=0.5, size=n) > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    import ml_common as MC
    pipe = MC.build_pipeline(LogisticRegression(C=1.0, max_iter=2000),
                             imbalance="none", scale=True)
    pipe.fit(X, y)
    xt = pipe[:-1].transform(X)
    phi, base, out = S.linear_shap(pipe, xt, xt)
    err = float(np.max(np.abs(base + phi.sum(axis=1) - out)))
    check(6, "linear_additivity", err < 1e-10, f"worst error {err:.2e}")
    check(6, "contributions_are_not_all_zero",
          float(np.max(np.abs(phi))) > 1e-6)

    # exactness against the closed form, worked out a second way here
    beta = pipe.named_steps["model"].coef_.ravel()
    manual = (xt - xt.mean(axis=0)) * beta
    check(6, "linear_matches_closed_form",
          float(np.max(np.abs(phi - manual))) < 1e-12)

    # locality: changing one feature's values should only change that feature's
    # contribution, with the fitted model held fixed
    xt2 = xt.copy()
    xt2[:, 2] = xt2[:, 2] + 1.0
    phi2, _, _ = S.linear_shap(pipe, xt, xt2)
    moved = np.abs(phi2 - phi).max(axis=0)
    check(6, "locality", moved[2] > 1e-6
          and float(np.max(np.delete(moved, 2))) < 1e-12,
          f"feature 2 moved {moved[2]:.3f}, others at most "
          f"{float(np.max(np.delete(moved, 2))):.2e}")

    # TreeSHAP additivity for both ensembles, each in its own output units
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    for name, est in (("random_forest",
                       RandomForestClassifier(n_estimators=40, random_state=0)),
                      ("xgboost",
                       XGBClassifier(n_estimators=40, max_depth=3,
                                     verbosity=0, random_state=0))):
        pipe = MC.build_pipeline(est, imbalance="none", scale=False)
        pipe.fit(X, y)
        xt = pipe[:-1].transform(X)
        phi, base, out = S.tree_shap(pipe, xt, name)
        err = float(np.max(np.abs(base + phi.sum(axis=1) - out)))
        check(6, f"{name}_additivity", err < 1e-4,
              f"worst error {err:.2e}")
        check(6, f"{name}_not_degenerate",
              float(np.max(np.abs(phi))) > 1e-6)

    # the collinearity code should find the planted pair and report that their
    # contributions cancel
    import pandas as pd
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    clusters, _ = H.correlation_clusters(Xdf, threshold=0.8)
    together = any({"f0", "f3"} <= set(c) for c in clusters)
    check(6, "collinear_pair_clustered", together,
          f"{len(clusters)} clusters from {p} features")


# --- 7. occlusion ---

def group7_occlusion():
    print("\n7. OCCLUSION MECHANICS")
    net = small_net(seed=8)
    (x, mn, mr), _ = night_pair(seed=19)
    stride = net.total_stride
    base = OC.logit_of(net, x, mn, mr)

    # input dependence first: occluding inside the sleep period has to move the
    # logit, or every invariance check below is vacuous
    inside = list(range(D.PAD_EACH_SIDE // stride + 10,
                        D.PAD_EACH_SIDE // stride + 40))
    xo, mno, mro = OC.occlude(x, mn, mr, inside, stride, -1.2, True)
    moved = abs(OC.logit_of(net, xo, mno, mro) - base)
    check(7, "occluding_sleep_moves_the_logit", moved > 1e-4,
          f"logit moved {moved:.4f}")

    # occluding padding (already outside every mask and already at the fill
    # value) should change nothing at all
    pad_steps = list(range(2, 20))
    xp, mnp, mrp = OC.occlude(x, mn, mr, pad_steps, stride, -1.2, True)
    unmoved = abs(OC.logit_of(net, xp, mnp, mrp) - base)
    check(7, "occluding_padding_changes_nothing", unmoved < 1e-4,
          f"logit moved {unmoved:.2e}")

    # clearing the mask and keeping it are different interventions, so one
    # number for both would mix up removal with re-weighting
    xk, mnk, mrk = OC.occlude(x, mn, mr, inside, stride, -1.2, False)
    d_cleared = OC.logit_of(net, xo, mno, mro) - base
    d_kept = OC.logit_of(net, xk, mnk, mrk) - base
    check(7, "cleared_and_kept_differ", abs(d_cleared - d_kept) > 1e-6,
          f"cleared {d_cleared:+.4f}, kept {d_kept:+.4f}")
    check(7, "keeping_the_mask_preserves_the_denominators",
          int(mnk.sum().item() + mrk.sum().item())
          == int(mn.sum().item() + mr.sum().item()))
    check(7, "clearing_the_mask_shrinks_the_denominators",
          int(mno.sum().item() + mro.sum().item())
          < int(mn.sum().item() + mr.sum().item()))

    # the decomposition should predict the direction of a large occlusion
    ex = A.exact_step_contributions(net, x, mn, mr)
    contrib = ex["contrib"]
    m_any = ((mn + mr) > 0).float()
    valid = M._match_length(M.pool_mask(m_any, stride),
                            contrib.shape[0])[0].numpy() > 0
    vidx = np.flatnonzero(valid)
    k = max(1, int(0.1 * vidx.size))
    top = vidx[np.argsort(-contrib[vidx])][:k]
    bottom = vidx[np.argsort(-contrib[vidx])][-k:]
    xt_, mnt, mrt = OC.occlude(x, mn, mr, top, stride, -1.2, False)
    xb_, mnb, mrb = OC.occlude(x, mn, mr, bottom, stride, -1.2, False)
    d_top = OC.logit_of(net, xt_, mnt, mrt) - base
    d_bot = OC.logit_of(net, xb_, mnb, mrb) - base
    check(7, "top_occlusion_lowers_more_than_bottom", d_top < d_bot,
          f"top {d_top:+.4f} against bottom {d_bot:+.4f}")


# --- 8. aggregation ---

def group8_aggregation():
    print("\n8. AGGREGATION ARITHMETIC")
    rng = np.random.default_rng(1)
    n = 3200
    vals = rng.normal(size=n)
    stride = 32
    steps = A.step_sums(vals, stride)
    check(8, "step_sums_preserve_the_total",
          abs(steps.sum() - vals.sum()) < 1e-9,
          f"difference {abs(steps.sum() - vals.sum()):.2e}")
    back = A.to_sample_grid(steps, stride, n)
    check(8, "round_trip_preserves_the_total",
          abs(back.sum() - vals.sum()) < 1e-9)
    check(8, "round_trip_is_not_the_identity",
          float(np.max(np.abs(back - vals))) > 1e-6,
          "spreading a step over its samples loses within step detail, as it "
          "must")

    # the profile's parts have to add up to the whole (the first thing anyone
    # reading the tables will check)
    percent = rng.uniform(85, 100, size=n)
    m_sleep = np.ones(n, bool)
    m_nrem = np.zeros(n, bool)
    m_nrem[: n // 2] = True
    m_rem = ~m_nrem
    desat = np.zeros(n, bool)
    desat[100:200] = True
    resp = np.zeros(n, bool)
    resp[150:260] = True
    p = A.profile(vals, percent, m_nrem, m_rem, desat, resp, m_sleep)
    bin_total = sum(p[f"bin_{i}_sum"] for i in range(len(C.bin_labels())))
    check(8, "bins_partition_the_total", abs(bin_total - p["total"]) < 1e-9,
          f"difference {abs(bin_total - p['total']):.2e}")
    check(8, "stages_partition_the_total",
          abs(p["nrem_sum"] + p["rem_sum"] - p["total"]) < 1e-9)
    check(8, "thirds_partition_the_total",
          abs(sum(p[f"third_{t}_sum"] for t in range(3)) - p["total"]) < 1e-9)
    check(8, "enrichment_is_a_ratio_of_shares",
          abs(p["desat_enrichment"]
              - p["desat_share_abs"] / p["desat_share_time"]) < 1e-9)

    # saturation bins follow the biomarker definitions: T90 counts strictly
    # below 90, so exactly 90 goes above the boundary
    idx = C.bin_index(np.array([89.99, 90.0, 84.99, 85.0, 100.0]))
    check(8, "bin_edges_match_the_T90_convention",
          idx[0] == 3 and idx[1] == 4 and idx[2] == 1 and idx[3] == 2
          and idx[4] == len(C.bin_labels()) - 1,
          f"indices {idx.tolist()}")


# --- 9. event placement ---

def group9_event_placement():
    print("\n9. EVENT PLACEMENT")
    import pandas as pd
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        pid = 999999
        pd.DataFrame({
            "event_concept": ["SpO2 desaturation", "SpO2 desaturation",
                              "Hypopnea"],
            "start_sec": [1000.0, 2000.0, 1500.0],
            "duration_sec": [20.0, 10.0, 15.0],
        }).to_csv(os.path.join(tmp, f"shhs1-{pid}_events.csv"), index=False)
        saved = A.EVENTS_DIR
        A.EVENTS_DIR = tmp
        try:
            mask, n = A.event_intervals(pid, A.DESAT_CONCEPTS, 0, 3000)
            check(9, "counts_the_right_events", n == 2, f"found {n}")
            check(9, "places_the_first_event",
                  bool(mask[1000] and mask[1019] and not mask[1020]),
                  "samples 1000 to 1019 inclusive")
            check(9, "total_event_samples", int(mask.sum()) == 30,
                  f"{int(mask.sum())} samples for 20 s plus 10 s")

            # the offset matters because the network reads a window starting at
            # the first scored sleep sample, so an event timed from the start of
            # the recording has to be shifted by it
            shifted, _ = A.event_intervals(pid, A.DESAT_CONCEPTS, 900, 3000)
            check(9, "offset_shifts_events",
                  bool(shifted[100] and not shifted[1000]),
                  "an offset of 900 moves the event from 1000 to 100")

            # events before the window have to be dropped; a negative index
            # would silently wrap them round to the end
            dropped, _ = A.event_intervals(pid, A.DESAT_CONCEPTS, 2500, 3000)
            check(9, "events_before_the_window_are_dropped",
                  int(dropped.sum()) == 0, f"{int(dropped.sum())} samples kept")

            resp_mask, rn = A.event_intervals(pid, A.RESP_CONCEPTS, 0, 3000)
            check(9, "respiratory_events_are_separate",
                  rn == 1 and int(resp_mask.sum()) == 15)
            check(9, "a_missing_file_is_not_an_error",
                  int(A.event_intervals(123, A.DESAT_CONCEPTS, 0, 10)[1]) == 0)
        finally:
            A.EVENTS_DIR = saved


# --- 10. hygiene ---

def group10_hygiene():
    print("\n10. HYGIENE")
    here = os.path.dirname(os.path.abspath(__file__))
    files = [os.path.join(here, f) for f in sorted(os.listdir(here))
             if f.endswith((".py", ".sh"))]
    bad = C.no_em_dash(files)
    check(10, "no_em_dashes_in_aim4_source", not bad,
          f"{len(files)} files checked"
          + (f", offending: {[os.path.basename(b) for b in bad]}" if bad
             else ""))
    check(10, "results_directory_is_on_the_drive",
          C.RESULTS.startswith("/Volumes/ThesisHD"), C.RESULTS)
    check(10, "deep_arm_and_epoch_are_declared",
          C.DEEP_ARM == "A" and C.DEEP_EPOCH_KIND == "marg",
          f"arm {C.DEEP_ARM}, epoch E*_{C.DEEP_EPOCH_KIND}")
    check(10, "baseline_is_on_the_quantisation_ladder",
          abs(A.BASELINE_97 / D.QUANTUM
              - round(A.BASELINE_97 / D.QUANTUM)) < 1e-9,
          f"{A.BASELINE_97:.4f} percent")
    check(10, "published_aucs_cover_every_model_and_spec",
          len(S.PUBLISHED_AUC) == len(C.CLASSICAL_MODELS) * len(C.SPECS),
          f"{len(S.PUBLISHED_AUC)} entries")


def main():
    print("=" * 78)
    print("AIM 4 UNIT TESTS")
    print("=" * 78)
    torch.manual_seed(0)
    for fn in (group0_the_test_network_responds,
               group1_cohort_and_folds, group2_exact_decomposition,
               group3_integrated_gradients, group4_gradient_identity,
               group5_grad_cam, group6_classical_shap, group7_occlusion,
               group8_aggregation, group9_event_placement, group10_hygiene):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            FAILED.append((fn.__name__, "raised", str(exc)))
    total = len(PASSED) + len(FAILED)
    print("\n" + "=" * 78)
    print(f"RESULT: {len(PASSED)} of {total} checks pass")
    if FAILED:
        print("\nfailures:")
        for g, n, d in FAILED:
            print(f"  {g}.{n}  {d}")
    print("=" * 78)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
