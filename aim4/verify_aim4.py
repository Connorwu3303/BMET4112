"""
Aim 4: verification against the saved outputs.

The unit tests check the code on inputs they build themselves. This script
reads the output files that were actually written and recomputes whatever can
be recomputed. They catch different things: a unit test can't catch a run that
used the wrong arm, and this can't catch an identity that was never true.

Checks are grouped by what a failure would mean, most serious first. Each one
prints what it compared as well as pass/fail, so the output doubles as a record.

Run:
    python3 verify_aim4.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common4 as C                                      # noqa: E402

import ml_common as MC                                   # noqa: E402
from folds import get_verified_folds                     # noqa: E402

ROWS = []


def record(group, name, ok, detail, matters=""):
    ROWS.append({"group": group, "check": name, "pass": bool(ok),
                 "detail": detail, "a_failure_would_mean": matters})
    print(f"  [{'pass' if ok else 'FAIL'}] {group}{name}  {detail}")
    return bool(ok)


def skip(group, name, why):
    ROWS.append({"group": group, "check": name, "pass": None,
                 "detail": f"not run: {why}", "a_failure_would_mean": ""})
    print(f"  [skip] {group}{name}  {why}")


# --- A: provenance ---

def group_a_provenance(data):
    print("\nA. PROVENANCE: IS THE MODEL EXPLAINED THE MODEL PUBLISHED?")
    att_path = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.csv")
    if not os.path.exists(att_path):
        skip("A", "1", "the full deep attribution run has not completed")
        skip("A", "2", "the full deep attribution run has not completed")
        skip("A", "3", "the full deep attribution run has not completed")
        return

    att = pd.read_csv(att_path)
    pid = data["pid"].astype(int)
    record("A", "1", set(att["pid"]) == set(pid.tolist())
           and len(att) == len(pid),
           f"{len(att)} participants explained, cohort is {len(pid)}",
           "some participants were explained by a model that saw their outcome,"
           " or some were not explained at all")

    # the standardised score recomputed in the attribution run has to equal the
    # score Aim 3 published for the same participant. This is what ties the
    # networks here to Aim 3's results; it fails if the wrong arm, epoch, fold or
    # checkpoint was loaded.
    sel_dir = os.path.join(C.AIM3_RESULTS, "selected")
    published = {}
    for fold in range(5):
        p = os.path.join(sel_dir, f"{C.DEEP_ARM}_r{C.DEEP_REPEAT}_f{fold}.npz")
        z = np.load(p, allow_pickle=True)
        key = "test_z_marg" if C.DEEP_EPOCH_KIND == "marg" else "test_z_inc"
        for q, v in zip(z["test_pid"].astype(int), z[key]):
            published[int(q)] = float(v)
    got = att.set_index("pid")["z_mean"]
    common = [q for q in got.index if q in published]
    diff = np.array([abs(got[q] - published[q]) for q in common])
    record("A", "2", float(diff.max()) < 1e-4,
           f"worst difference from the Aim 3 published score "
           f"{float(diff.max()):.2e} over {len(common)} participants",
           "the networks explained here are not the networks Aim 3 evaluated")

    meta = json.load(open(os.path.join(
        C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.json")))
    record("A", "3", meta["arm"] == C.DEEP_ARM
           and meta["epoch_kind"] == C.DEEP_EPOCH_KIND
           and int(meta["repeat"]) == C.DEEP_REPEAT,
           f"arm {meta['arm']}, repeat {meta['repeat']}, "
           f"E*_{meta['epoch_kind']}",
           "the run used a different arm or epoch from the one declared")


# --- B: classical SHAP ---

def group_b_shap(data):
    print("\nB. CLASSICAL SHAP")
    import shap_classical as S
    path = os.path.join(C.RESULTS, "shap_summary.json")
    if not os.path.exists(path):
        skip("B", "1", "shap_classical.py has not run")
        return
    summary = json.load(open(path))
    y = data["y"]

    exact_models = ("logistic_regression", "xgboost")
    bad_exact = [s for s in summary
                 if s["model"] in exact_models
                 and abs(s["auc_difference"]) >= 5e-4]
    n_exact = len([s for s in summary if s["model"] in exact_models])
    record("B", "1", not bad_exact,
           f"{n_exact - len(bad_exact)} of {n_exact} refits with recovered "
           f"hyperparameters reproduce the published AUC to 4 decimals",
           "the model being explained is not the model published")

    worst = max(s["worst_additivity_violation"] for s in summary)
    record("B", "2", worst < 1e-4,
           f"worst additivity violation {worst:.2e} across "
           f"{len(summary)} model and specification pairs",
           "the contributions do not sum to the prediction they claim to "
           "decompose")

    # the forest uses a fixed configuration in place of the per fold tuned one,
    # which is only ok if the two are statistically indistinguishable, so test it
    rf = [s for s in summary if s["model"] == "random_forest"]
    if rf:
        from numpy import load
        details, all_ok = [], True
        for s in rf:
            spec = s["specification"]
            z = load(os.path.join(
                C.RESULTS, f"shap_random_forest_{spec}.npz"),
                allow_pickle=True)
            ref = load(os.path.join(C.AIM2_RESULTS, "rf_oof_scores.npz"))
            if spec not in ref.files:
                continue
            published = ref[spec][1]
            t = MC.delong_test(y, z["oof"], published)
            if t:
                sig = t["p_value"] < 0.05
                all_ok = all_ok and not sig
                details.append(f"{spec} {t['difference']:+.4f} "
                               f"p={t['p_value']:.3g}")
        record("B", "3", all_ok,
               "the fixed configuration forest against the published tuned "
               "forest, paired DeLong: " + "; ".join(details),
               "the forest explained here discriminates differently from the "
               "one Aim 2 published, so its contributions describe a different "
               "model")

    folds = pd.read_csv(os.path.join(C.RESULTS, "shap_folds.csv"))
    per = folds.groupby(["model", "specification"]).size()
    record("B", "4", bool((per == 15).all()),
           f"every model and specification covers 15 outer folds "
           f"(min {int(per.min())}, max {int(per.max())})",
           "a fold was skipped, so some participants were explained fewer "
           "times than others")

    # for the linear model the contributions are deviations from the training
    # mean, so they should average to about zero over the cohort. A large mean
    # means the background expectation came from the wrong rows.
    z = np.load(os.path.join(
        C.RESULTS, "shap_logistic_regression_obm_plus_confounders.npz"),
        allow_pickle=True)
    col_means = np.abs(z["shap"].mean(axis=0))
    scale = np.abs(z["shap"]).mean(axis=0).max()
    record("B", "5", float(col_means.max()) < 0.02 * scale,
           f"largest mean contribution {float(col_means.max()):.2e} against a "
           f"typical magnitude of {scale:.3f}",
           "the background expectation came from the explained rows rather "
           "than the training rows")


# --- C: deep attribution ---

def group_c_deep(data):
    print("\nC. DEEP ATTRIBUTION")
    att_path = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.csv")
    if not os.path.exists(att_path):
        skip("C", "1", "the full deep attribution run has not completed")
        return
    att = pd.read_csv(att_path)
    meta = json.load(open(os.path.join(
        C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.json")))

    worst = float(att["exact_reconstruction_error"].max())
    record("C", "1", worst < 1e-3,
           f"worst reconstruction error {worst:.2e} over {len(att)} "
           f"participants",
           "the per step decomposition is not the network's own arithmetic, so "
           "the primary attribution is not exact after all")

    col = "ig_flat_own_completeness_error"
    if col in att:
        w = float(att[col].abs().max())
        record("C", "2", w <= 0.025 + 1e-9,
               f"worst integrated gradients completeness error {w:.2e} logits, "
               f"tolerance 0.025; integration refined on "
               f"{meta.get('flat_own_refined_fraction', 0):.1%} of "
               f"participants",
               "the sample level attribution does not sum to the change in "
               "logit it claims to explain")

    record("C", "3", float(att["gxa_vs_exact_pearson"].median()) > 0.99,
           f"gradient times activation against the exact decomposition, "
           f"median r {float(att['gxa_vs_exact_pearson'].median()):.4f}",
           "two independently coded attributions of the same quantity "
           "disagree, so at least one is wrong")

    # SpO2 bins and sleep stages should each add up to the sleep period total,
    # recomputed from the saved table
    nb = len(C.bin_labels())
    bin_total = att[[f"exact_bin_{i}_sum" for i in range(nb)]].sum(axis=1)
    stage_total = att["exact_nrem_sum"] + att["exact_rem_sum"]
    record("C", "4",
           float((bin_total - att["exact_sleep_sum"]).abs().max()) < 1e-6
           and float((stage_total - att["exact_sleep_sum"]).abs().max()) < 1e-6,
           f"saturation bins and sleep stages each partition the sleep period "
           f"total to {float((bin_total - att['exact_sleep_sum']).abs().max()):.2e}",
           "the tables in the write up do not add up")

    # The aggregation also has to hold all the way to the published logit. Each
    # step's contribution is spread over its 32 samples and summed back over the
    # whole sleep period window, so the window total should equal the logit minus
    # the head bias, which is one constant per fold. The check is the spread of
    # that difference within each fold.
    #
    # It has to be the window total here. Using the valid sleep total was a
    # mistake made once: a step goes into the pooling when at least half its
    # samples are valid sleep of a stage, so a pooled step can still hold samples
    # that aren't valid sleep. Its contribution counts in full towards the logit,
    # but summing over valid sleep samples only picks up part of it. So the sleep
    # only total misses the logit by a real, systematic amount even when the
    # aggregation is fine. C8 measures that shortfall directly.
    diff = att["logit_mean"] - att["exact_total"]
    spread = float(diff.groupby(att["fold"]).std().max())
    scale = float(att["logit_mean"].std())
    bias = float(diff.mean())
    record("C", "7", spread < 0.01 * scale,
           f"within each fold the logit minus the aggregated window total "
           f"varies by at most {spread:.2e}, which is "
           f"{100 * spread / scale:.2f} percent of the logit spread of "
           f"{scale:.3f}; its mean is {bias:+.4f}, recovering the head bias "
           f"(initialised at {np.log(0.19 / 0.81):+.4f})",
           "spreading step contributions over their samples and summing them "
           "back does not recover the logit, so every aggregate table in the "
           "write up is detached from the model's actual output")

    # the shortfall of the valid sleep total against the window total has a
    # specific cause, so measure it. If it came from anything other than non
    # sleep samples inside pooled steps, the correlation below wouldn't hold and
    # the aggregation would need another look.
    gap = att["exact_total"] - att["exact_sleep_sum"]
    nonsleep = 1.0 - att["exact_sleep_n"] / att["n_samples"]
    rho = C.spearman_rank_agreement(gap, nonsleep)["rho"]
    record("C", "8", rho > 0.25 and float(gap.abs().median()) < 0.02 * scale,
           f"the valid sleep total falls short of the window total by a median "
           f"{float(gap.median()):+.5f}, and that shortfall correlates with the "
           f"share of the window that is not valid sleep at Spearman "
           f"{rho:+.3f}, which is the expected cause: a pooled step contributes "
           f"in full but only its valid sleep samples are summed",
           "the difference between the two totals is not explained by non "
           "sleep samples inside pooled steps, so some other and unidentified "
           "effect is moving the aggregates")

    faults = meta.get("faults", {})
    record("C", "5", int(faults.get("insane", 0)) == 0,
           f"backend faults: {faults}",
           "a logit beyond the sane range entered the published attribution")

    # Adebayo sanity check: if an untrained network gives the same attribution
    # profile, the profile is just the architecture's prior and says nothing
    # about what the network learned
    rpath = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}_random.csv")
    if not os.path.exists(rpath):
        skip("C", "6", "the random weights control has not run")
    else:
        rnd = pd.read_csv(rpath)
        shared = set(att["pid"]) & set(rnd["pid"])
        a = att[att["pid"].isin(shared)].sort_values("pid")
        b = rnd[rnd["pid"].isin(shared)].sort_values("pid")
        cols = [f"exact_bin_{i}_abs" for i in range(nb)]
        pa = a[cols].sum().to_numpy()
        pb = b[cols].sum().to_numpy()
        pa = pa / max(1e-12, pa.sum())
        pb = pb / max(1e-12, pb.sum())
        l1 = float(np.abs(pa - pb).sum())
        record("C", "6", l1 > 0.10,
               f"the trained and untrained saturation profiles differ by "
               f"{l1:.3f} in total variation over {len(shared)} shared "
               f"participants",
               "the attribution reflects the architecture rather than "
               "anything the network learned, which would invalidate every "
               "interpretation drawn from it")


# --- D: occlusion ---

def group_d_occlusion():
    print("\nD. OCCLUSION, THE INTERVENTION TEST")
    path = os.path.join(C.RESULTS, f"occlusion_{C.DEEP_ARM}.json")
    if not os.path.exists(path):
        skip("D", "1", "occlusion.py has not run")
        return
    s = json.load(open(path))
    for suffix in ("cleared", "kept"):
        top = s[f"d_top_{suffix}_median"]
        rnd = s[f"d_random_{suffix}_median"]
        bot = s[f"d_bottom_{suffix}_median"]
        record("D", f"1_{suffix}", top < rnd < bot,
               f"median change in logit: top {top:+.4f}, random {rnd:+.4f}, "
               f"bottom {bot:+.4f}",
               "removing the stretches the attribution calls important does "
               "not lower the score more than removing random stretches, so "
               "the attribution does not describe the model's behaviour")
        r = s.get(f"predicted_vs_measured_spearman_{suffix}")
        record("D", f"2_{suffix}", r is not None and r > 0.2,
               f"predicted against measured drop across participants, "
               f"Spearman {r:+.3f}",
               "the size of the predicted effect carries no information about "
               "the size of the measured one")


# --- E: head to head arithmetic ---

def group_e_arithmetic():
    print("\nE. ARITHMETIC IN THE HEAD TO HEAD")
    path = os.path.join(C.RESULTS, "headtohead.json")
    if not os.path.exists(path):
        skip("E", "1", "headtohead.py has not run")
        return
    r = json.load(open(path))

    c3 = r.get("contrast_c3")
    if c3:
        stated = c3["delta_auc_point"]
        recomputed = c3["auc_with_deep"] - c3["auc_base"]
        record("E", "1", abs(stated - recomputed) < 1e-9,
               f"C3 change in AUC {stated:+.6f}, recomputed "
               f"{recomputed:+.6f}",
               "a reported difference does not equal the difference of the "
               "numbers it was computed from")
        boot = c3["delta_auc"]["delta"]
        record("E", "2", abs(boot - recomputed) < 1e-6,
               f"the bootstrap point estimate {boot:+.6f} equals the direct "
               f"difference",
               "the bootstrap was given its two score vectors the wrong way "
               "round, which would reverse the sign of the reported effect")

    red = r.get("reducibility", [])
    if red:
        bad = [x for x in red
               if x["r2_out_of_fold"] > x["r2_in_sample_mean"] + 1e-6]
        record("E", "3", not bad,
               f"out of fold R squared is at or below the in sample figure for "
               f"all {len(red)} predictor sets",
               "an out of fold estimate exceeds its in sample counterpart, "
               "which means the folds are not what they claim to be")

    gpath = os.path.join(C.RESULTS, "grouped_contributions.csv")
    if os.path.exists(gpath):
        g = pd.read_csv(gpath)
        ok = g["cancellation_ratio"].dropna()
        record("E", "4", bool(((ok >= -1e-9) & (ok <= 1 + 1e-9)).all()),
               f"every cancellation ratio lies in [0, 1] "
               f"(min {ok.min():.3f}, max {ok.max():.3f})",
               "a cancellation ratio outside its definable range means the "
               "grouped and individual magnitudes were computed "
               "inconsistently")
        singles = g[g["size"] == 1]["cancellation_ratio"].dropna()
        record("E", "5", bool((np.abs(singles - 1.0) < 1e-9).all()),
               f"all {len(singles)} single member clusters have a ratio of "
               f"exactly 1, as they must",
               "the grouped sum of a one feature cluster does not equal that "
               "feature's own contribution")


# --- F: pre-registration and hygiene ---

def group_f_hygiene():
    print("\nF. PRE-REGISTRATION AND HYGIENE")
    here = os.path.dirname(os.path.abspath(__file__))
    srcs = [os.path.join(here, f) for f in sorted(os.listdir(here))
            if f.endswith((".py", ".sh"))]
    docs = [os.path.join(here, "..", d) for d in
            ("AIM4_METHODS.md", "AIM4_RESULTS.md", "AIM4_README.md",
             "AIM4_TABLES.md")]
    bad = C.no_em_dash(srcs + docs)
    record("F", "1", not bad,
           f"{len([p for p in srcs + docs if os.path.exists(p)])} source and "
           f"document files contain no em dash"
           + (f"; offending: {[os.path.basename(b) for b in bad]}"
              if bad else ""),
           "a project convention was broken in published text")

    import headtohead as H
    sys.path.insert(0, C.AIM3_DIR)
    import stack_eval as SE
    record("F", "2", H.EQUIVALENCE_MARGIN == SE.EQUIVALENCE_MARGIN,
           f"the equivalence margin is {H.EQUIVALENCE_MARGIN}, matching Aim 3",
           "C3 was judged against a different margin from C2, so the two "
           "contrasts are not comparable")

    # every declared constant that shapes a published number has to be in a
    # saved file, so anyone reading later can tell what the run used
    need = {"deep_attrib": ("ig_steps", "baseline_97_percent"),
            "occlusion": ("fraction", "n_random_draws")}
    missing = []
    for stem, keys in need.items():
        p = os.path.join(C.RESULTS, f"{stem}_{C.DEEP_ARM}.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        missing += [f"{stem}.{k}" for k in keys if k not in j]
    record("F", "3", not missing,
           "the run settings that shape published numbers are saved alongside "
           "them" + (f"; missing {missing}" if missing else ""),
           "a published number cannot be reproduced because the settings that "
           "produced it were not recorded")


def main():
    print("=" * 78)
    print("AIM 4 VERIFICATION")
    print("=" * 78)
    data = get_verified_folds(quiet=True)
    for fn in (group_a_provenance, group_b_shap, group_c_deep):
        try:
            fn(data)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            ROWS.append({"group": fn.__name__, "check": "raised",
                         "pass": False, "detail": str(exc),
                         "a_failure_would_mean": ""})
    for fn in (group_d_occlusion, group_e_arithmetic, group_f_hygiene):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            ROWS.append({"group": fn.__name__, "check": "raised",
                         "pass": False, "detail": str(exc),
                         "a_failure_would_mean": ""})

    t = pd.DataFrame(ROWS)
    os.makedirs(C.RESULTS, exist_ok=True)
    t.to_csv(os.path.join(C.RESULTS, "verification_aim4.csv"), index=False)
    ran = t[t["pass"].notna()]
    n_ok = int(ran["pass"].sum())
    print("\n" + "=" * 78)
    print(f"RESULT: {n_ok} of {len(ran)} checks pass, "
          f"{int(t['pass'].isna().sum())} not applicable yet")
    fails = ran[~ran["pass"].astype(bool)]
    if len(fails):
        print("\nFAILURES, and what each would mean:")
        for _, r in fails.iterrows():
            print(f"  {r['group']}{r['check']}: {r['detail']}")
            if r["a_failure_would_mean"]:
                print(f"      {r['a_failure_would_mean']}")
    print(f"wrote {C.RESULTS}/verification_aim4.csv")
    print("=" * 78)
    return 0 if not len(fails) else 1


if __name__ == "__main__":
    sys.exit(main())
