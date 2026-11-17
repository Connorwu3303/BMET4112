"""
Aim 3: writes the results tables straight from the saved outputs.

Cross checking the Aim 2 write ups turned up four errors in 116 hand typed
figures, e.g. "all p greater than 0.11" when the smallest was 0.107, and a p
value written as "less than 0.001" twice when it was actually larger. None of
them changed a conclusion, but they were all the same mistake: a number read
from one file and typed into another. So every number in the Aim 3 results is
generated here from the saved JSON and npz outputs. Only the prose around the
tables is written by hand.

Inputs (processed/aim3/results/): stack_eval_<arm>.json, fold_detail_<arm>.csv,
diagnostics_<arm>.json, gate_odi3.json, aim2_rebuild_summary.json,
verification_aim3.csv
Output: AIM3_TABLES.md in the repo root (or the path given by --out)

Usage:
    python3 aim3/report.py --arm A
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
AIM2 = "/Volumes/ThesisHD/Data/processed/aim2/results"


def load(path, default=None):
    if not os.path.exists(path):
        return default
    if path.endswith(".json"):
        with open(path) as fh:
            return json.load(fh)
    if path.endswith(".npz"):
        return np.load(path, allow_pickle=True)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return default


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{nd}f}"


def p_fmt(p):
    """Formats a p value without writing a 'less than' it can't back up."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-4:
        return "< 0.0001"
    return f"{p:.4g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ev = load(os.path.join(RESULTS, f"stack_eval_{args.arm}.json"))
    if ev is None:
        raise SystemExit(f"stack_eval_{args.arm}.json not found; "
                         f"run stack_eval.py first")
    folds = load(os.path.join(RESULTS, f"fold_detail_{args.arm}.csv"))
    diag = load(os.path.join(RESULTS, f"diagnostics_{args.arm}.json"))
    gate = load(os.path.join(RESULTS, "gate_odi3.json"))
    rebuild = load(os.path.join(RESULTS, "aim2_rebuild_summary.json"))
    verif = load(os.path.join(RESULTS, "verification_aim3.csv"))

    L = []
    w = L.append

    w("## Performance")
    w("")
    reps = ev.get("repeats", [])
    if len(reps) == 1:
        w(f"Out of fold predictions from repeat {reps[0]} of the outer cross")
        w("validation. Aim 2 reports its headline figures averaged over three")
        w("repeats, so every comparator below is taken from the matching")
        w(f"repeat {reps[0]} fit rather than from Aim 2's published average, and")
        w("the two are therefore compared on the same repeat.")
    else:
        w(f"Out of fold predictions averaged over repeats {reps}, matching how")
        w("Aim 2 reports its headline figures.")
    w("")
    w("| Model | AUC [95% CI] | AUPRC | Brier | Calibration slope |")
    w("|---|---|---|---|---|")
    for r in ev["headline"]:
        ci = f"[{fmt(r.get('auc_ci_lo'), 3)}, {fmt(r.get('auc_ci_hi'), 3)}]"
        w(f"| {r['model']} | {fmt(r['auc_roc'])} {ci} | {fmt(r['auprc'])} | "
          f"{fmt(r['brier'])} | {fmt(r.get('calibration_slope'), 3)} |")
    w("")
    if ev["headline"]:
        w(f"Event rate {fmt(ev['headline'][0]['n_events'] / ev['headline'][0]['n'] * 100, 1)} "
          f"percent, so the no skill AUPRC is "
          f"{fmt(ev['headline'][0].get('auprc_baseline'))}.")
        w("")

    w("## The main contrast: does the waveform add to the clinical variables?")
    w("")
    ht = ev.get("c2_held_out_test", {})
    if "or_per_sd" in ht:
        w("Primary inference, a likelihood ratio test of the deep score's")
        w("coefficient fitted on held out predictions with the clinical logit as")
        w("a fixed offset:")
        w("")
        w(f"- odds ratio per standard deviation: **{fmt(ht['or_per_sd'], 3)}** "
          f"[{fmt(ht['or_ci_lo'], 3)}, {fmt(ht['or_ci_hi'], 3)}]")
        w(f"- likelihood ratio chi square {fmt(ht['lr_chi2'], 3)} on 1 degree "
          f"of freedom, p = {p_fmt(ht['p_value'])}")
        w(f"- fitted on {ht['n']} held out predictions")
        w("")
    pr = ev.get("c2_per_repeat", [])
    if pr and all("or_per_sd" in r for r in pr):
        w("Per repeat, to show whether the direction is stable:")
        w("")
        w("| Repeat | Odds ratio per SD | p |")
        w("|---|---|---|")
        for r in pr:
            w(f"| {r['repeat']} | {fmt(r['or_per_sd'], 3)} | "
              f"{p_fmt(r['p_value'])} |")
        w("")
    b = ev.get("c2_delta_auc", {})
    if b:
        w("Effect size, as the change in AUC from adding the deep score to the")
        w("clinical model:")
        w("")
        w(f"- change in AUC **{b['delta']:+.4f}**")
        w(f"- paired bootstrap standard error {fmt(b['se_boot'])} "
          f"({b['n_boot_used']} resamples), between repeat "
          f"standard deviation {fmt(ev.get('c2_between_repeat_sd'))}, "
          f"total standard error {fmt(ev.get('c2_se_total'))}")
        ci90 = ev.get("c2_ci90", [None, None])
        w(f"- 90 percent interval on the total standard error "
          f"[{ci90[0]:+.4f}, {ci90[1]:+.4f}]")
        w(f"- pre-registered equivalence margin plus or minus "
          f"{ev['equivalence_margin']:.3f}: the interval lies "
          f"**{'inside' if ev.get('c2_inside_margin') else 'outside'}** it")
        w("")
    dl = ev.get("c2_delong")
    if dl:
        w(f"DeLong test, reported for continuity with Aim 2 and conservative for")
        w(f"nested models: z = {dl['z']:+.3f}, p = {p_fmt(dl['p_value'])}.")
        w("")

    w("## Head to head against the hand engineered biomarkers")
    w("")
    c1 = ev.get("c1_delong")
    if c1:
        reps = ev.get("repeats", [])
        w(f"Both sides on repeat(s) {reps}, so the comparison is repeat matched.")
        w("")
        w(f"| Model | AUC |")
        w(f"|---|---|")
        w(f"| Aim 2 logistic regression, 34 biomarkers | {fmt(c1['auc_a'])} |")
        w(f"| Aim 3 deep network, raw waveform | {fmt(c1['auc_b'])} |")
        w("")
        w(f"Difference {c1['difference']:+.4f}, paired DeLong z = "
          f"{c1['z']:+.3f}, p = {p_fmt(c1['p_value'])}.")
        if c1.get("ci_lo") is not None:
            width = (c1["ci_hi"] - c1["ci_lo"])
            w(f"95 percent interval on the difference "
              f"[{c1['ci_lo']:+.4f}, {c1['ci_hi']:+.4f}], which is "
              f"{width / (2 * ev['equivalence_margin']):.1f} times the width of "
              f"the equivalence margin. This is a failure to separate the two "
              f"models, not a demonstration that they are equivalent.")
        w("")
        pub = ev.get("c1_delong_vs_published_average")
        if pub:
            w(f"Against Aim 2's published three repeat average (AUC "
              f"{fmt(pub['auc_a'])}), which is a less noisy vector and so not "
              f"repeat matched, the difference is {pub['difference']:+.4f}, "
              f"p = {p_fmt(pub['p_value'])}. Reported for continuity only.")
            w("")

    if folds is not None and len(folds):
        w("## Per fold detail")
        w("")
        w("Reported descriptively. Fold level differences are never tested")
        w("individually, because the folds share most of their training data and")
        w("such tests are not valid.")
        w("")
        w("| Quantity | Mean | SD | Min | Max |")
        w("|---|---|---|---|---|")
        for col, lab in [("auc_deep_marg", "deep waveform only AUC"),
                         ("auc_clinical", "clinical only AUC"),
                         ("auc_stacked", "clinical plus deep AUC"),
                         ("deep_coef_in_stack", "deep coefficient in the stack"),
                         ("inner_auc_spread",
                          "spread across the 3 inner models")]:
            if col in folds.columns:
                s = folds[col]
                w(f"| {lab} | {fmt(s.mean())} | {fmt(s.std())} | "
                  f"{fmt(s.min())} | {fmt(s.max())} |")
        w("")
        # kept in fold order. Sorting hid which fold was the exception, and the
        # interesting bit is that the two criteria have their exceptions in
        # different folds
        ordered = folds.sort_values("fold")
        w(f"Selected epochs by fold 0 to 4: E*_marg "
          f"{ordered['e_marg'].tolist()}, E*_inc "
          f"{ordered['e_inc'].tolist()}.")
        w("")

    if diag:
        w("## Is it oximetry, or is it age and sleep structure?")
        w("")
        w("| Comparison | AUC |")
        w("|---|---|")
        w(f"| deep waveform only | {fmt(diag['auc_deep_spo2_only'])} |")
        w(f"| nuisance model: sleep structure and data quality only | "
          f"{fmt(diag['auc_nuisance_structure_only'])} |")
        if diag.get("auc_negative_control_matched") is not None:
            w(f"| architecture matched control: same network, sleep mask and "
              f"REM only, no saturation | "
              f"{fmt(diag['auc_negative_control_matched'])} |")
        if diag.get("auc_negative_control") is not None:
            w(f"| negative control network: sleep mask and REM, no saturation | "
              f"{fmt(diag['auc_negative_control'])} |")
        w("")
        if diag.get("auc_negative_control_matched") is not None:
            gap = diag["auc_deep_spo2_only"] - \
                diag["auc_negative_control_matched"]
            w("")
            w(f"The architecture matched control is the attribution test: it is "
              f"the same network given the same two mask channels and no "
              f"saturation, so the only difference from the primary arm is what "
              f"it can see. Removing saturation costs {gap:+.4f} AUC.")
            w("")
        w(f"The pre-registered rule flags the deep score as a sleep structure")
        w(f"score rather than an oximetry score if it comes within "
          f"{diag['nuisance_flag_margin']:.2f} AUC of the nuisance model. "
          f"Gap {diag['nuisance_gap']:+.4f}: "
          f"**{'FLAGGED' if diag['nuisance_flagged'] else 'not flagged'}**.")
        w("")
        if diag.get("correlations"):
            w("What the deep score correlates with (Spearman):")
            w("")
            w("| Variable | rho | p |")
            w("|---|---|---|")
            for r in sorted(diag["correlations"],
                            key=lambda x: -abs(x["spearman"])):
                w(f"| {r['variable']} | {r['spearman']:+.3f} | "
                  f"{p_fmt(r['p_value'])} |")
            w("")
        if diag.get("strata"):
            w("Discrimination within strata. If the score were mostly an age")
            w("proxy, it would weaken sharply inside narrow age bands.")
            w("")
            w("| Stratum | n | Events | AUC |")
            w("|---|---|---|---|")
            for s in diag["strata"]:
                w(f"| {s['stratum']} | {s['n']} | {s['n_events']} | "
                  f"{fmt(s['auc'])} |")
            w("")

    w("## Validity checks")
    w("")
    if gate:
        w(f"- Label free learnability gate: architecture A recovers the sleep")
        w(f"  oxygen desaturation index from the waveform at Pearson r = "
          f"**{fmt(gate['best_r'], 4)}** on 345 held out nights from the 1,441")
        w(f"  participants who appear in no cardiovascular fold "
          f"(best of {gate['epochs']} epochs, at epoch {gate['best_epoch']}).")
    if rebuild:
        ok = all(r["reproduces"] for r in rebuild)
        worst = max(abs(r["difference"]) for r in rebuild)
        w(f"- The Aim 2 comparators were rebuilt with the Aim 2 code on the same")
        w(f"  folds: all {len(rebuild)} specifications reproduce "
          f"({'yes' if ok else 'NO'}), worst AUC difference {worst:.5f}.")
        for r in rebuild:
            if r.get("max_abs_diff_with_saved") is not None:
                w(f"  - {r['specification']}: rebuilt "
                  f"{fmt(r['auc_rebuilt'])} against reported "
                  f"{fmt(r['auc_aim2_reported'])}, worst per participant "
                  f"difference {r['max_abs_diff_with_saved']:.1e}")
    if verif is not None and len(verif):
        n_pass = int((verif["status"] == "PASS").sum())
        w(f"- Independent verification: {n_pass} of {len(verif)} checks passed.")
        for _, r in verif[verif["status"] == "FAIL"].iterrows():
            w(f"  - FAILED: {r['check']}: {r['detail']}")
    w("")

    text = "\n".join(L) + "\n"
    # Built from its code point so this file has no literal em dash, and so a
    # blanket search and replace over the file can't break the guard. An earlier
    # fix did exactly that: it turned the condition into ", ", which fired on
    # every comma and quietly stopped the report being written for a day.
    if chr(0x2014) in text:
        raise SystemExit("an em dash reached the report; fix the generator")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "AIM3_TABLES.md")
    with open(out, "w") as fh:
        fh.write(text)
    print(text)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
