"""
Aim 3: checks the results from the saved artefacts, independently of how they
were produced.

Re-running the pipeline only shows the code is deterministic, so this
recomputes each reported number from the saved intermediate files, by a
different route where there is one, and checks leakage that a single run can't
show. Check groups:

  A  identity          same people in the same partitions as Aim 2
  B  leakage           no participant's outcome fed into their own prediction
  C  selection         chosen epochs follow the stated rules
  D  reproduction      saved weights reproduce the test predictions
  E  arithmetic        reported metrics reproduce from the saved predictions
  F  pre-registration  constants match the ones fixed before the labelled run
  G  hygiene           no em dashes anywhere (project convention)

Output: processed/aim3/results/verification_aim3.csv

Usage:
    python3 aim3/verify_aim3.py --arm A
"""

import argparse
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import roc_auc_score, brier_score_loss  # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
SELECT_DIR = os.path.join(RESULTS, "selected")
AIM2 = "/Volumes/ThesisHD/Data/processed/aim2/results"
AIM3_DIR = os.path.dirname(os.path.abspath(__file__))

# built from its code point so this file has no literal em dash and doesn't
# fail its own hygiene check
EM_DASH = chr(0x2014)

checks = []


def check(group, name, ok, detail=""):
    checks.append({"group": group, "check": name,
                   "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" +
          (f": {detail}" if detail else ""))
    return bool(ok)


def selections(arm, tag=""):
    out = []
    if not os.path.isdir(SELECT_DIR):
        return out
    for f in sorted(os.listdir(SELECT_DIR)):
        if f.startswith(f"{arm}_r") and f.endswith(f"{tag}.npz") \
                and (tag or not any(f.endswith(f"{t}.npz")
                                    for t in ("_perm", "_pos", "_smoke"))):
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reproduce-folds", type=int, default=2,
                    help="how many folds to re-score from saved weights")
    args = ap.parse_args()

    print("=" * 76)
    print(f"AIM 3 INDEPENDENT VERIFICATION: arm {args.arm}")
    print("=" * 76)

    from folds import get_verified_folds
    data = get_verified_folds(quiet=True)
    y, pid = data["y"], data["pid"]
    pid_to_row = {int(p): i for i, p in enumerate(pid)}
    files = selections(args.arm, args.tag)

    # --- A. identity with Aim 2 ---
    print("\nA. IDENTITY WITH AIM 2")
    check("A", "A1 cohort is 4,352 participants with 829 events",
          len(y) == 4352 and int(y.sum()) == 829,
          f"{len(y)} participants, {int(y.sum())} events")

    ok = True
    for f in ["logreg_oof_scores.npz", "rf_oof_scores.npz",
              "xgb_oof_scores.npz"]:
        p = os.path.join(AIM2, f)
        if not os.path.exists(p):
            ok = False
            continue
        with np.load(p) as z:
            for spec in z.files:
                if not np.array_equal(z[spec][0].astype(int), y):
                    ok = False
    check("A", "A2 labels identical to every Aim 2 out-of-fold file", ok,
          "paired comparisons are on the same participants in the same order")

    reb = os.path.join(RESULTS, "aim2_rebuild_summary.json")
    if os.path.exists(reb):
        with open(reb) as fh:
            summary = json.load(fh)
        worst = max(abs(s["difference"]) for s in summary)
        allrep = all(s["reproduces"] for s in summary)
        diffs = [s.get("max_abs_diff_with_saved") for s in summary
                 if s.get("max_abs_diff_with_saved") is not None]
        check("A", "A3 the Aim 2 comparators rebuild exactly", allrep,
              f"worst AUC difference {worst:.5f}" +
              (f", worst prediction difference {max(diffs):.2e}"
               if diffs else ""))
    else:
        check("A", "A3 the Aim 2 comparators rebuild exactly", False,
              "aim2_rebuild_summary.json not found; run aim2_comparators.py")

    check("A", "A4 selections exist for this arm", len(files) > 0,
          f"{len(files)} outer folds selected")
    if not files:
        print("\nNothing further can be verified until the run has produced "
              "selections.")
        return finish()

    # --- B. leakage ---
    print("\nB. LEAKAGE")
    bad_overlap, bad_partition, bad_cover = [], [], []
    for f in files:
        z = np.load(os.path.join(SELECT_DIR, f), allow_pickle=True)
        tr = set(int(p) for p in z["outer_train_pid"])
        te = set(int(p) for p in z["test_pid"])
        if tr & te:
            bad_overlap.append(f)
        # Cross fitted training scores should exist for every outer training row
        # and for no test row. A score on a test row would mean the network had
        # scored someone it was training on.
        zi = z["train_z_inc"]
        tr_rows = np.array([pid_to_row[p] for p in tr])
        te_rows = np.array([pid_to_row[p] for p in te])
        if not np.all(np.isfinite(zi[tr_rows])):
            bad_partition.append(f)
        if np.any(np.isfinite(zi[te_rows])):
            bad_cover.append(f)
    check("B", "B1 outer training and test folds never overlap",
          not bad_overlap, f"{len(files)} folds checked")
    check("B", "B2 cross fitted scores exist for every outer training row",
          not bad_partition)
    check("B", "B3 cross fitted scores are absent for every test row",
          not bad_cover,
          "a test row carrying a cross fitted score would mean a network "
          "scored a participant it trained on")

    # The inner validation folds should partition the outer training fold. This
    # reads the unit metadata instead of the selection file, so it also checks
    # the two agree.
    mism = []
    for f in files:
        stem = f[:-4]
        parts = stem.split("_")
        repeat, fold = int(parts[1][1:]), int(parts[2][1:])
        val_union, tr_sets = [], []
        for inner in range(3):
            mp = os.path.join(RESULTS, "units",
                              f"{args.arm}_r{repeat}_f{fold}_i{inner}"
                              f"{args.tag}", "meta.json")
            if not os.path.exists(mp):
                mism.append((f, inner, "missing meta"))
                continue
            with open(mp) as fh:
                m = json.load(fh)
            val_union += m["inner_val_pid"]
            tr_sets.append(set(m["inner_train_pid"]))
            # a unit's own training rows must never include the outer test fold
            if set(m["inner_train_pid"]) & set(m["outer_test_pid"]):
                mism.append((f, inner, "trained on outer test rows"))
            if set(m["inner_val_pid"]) & set(m["outer_test_pid"]):
                mism.append((f, inner, "validated on outer test rows"))
        if len(val_union) != len(set(val_union)):
            mism.append((f, None, "a participant is in two validation folds"))
    check("B", "B4 no unit trains or validates on its outer test fold",
          not mism, f"{len(files) * 3} units checked" if not mism
          else f"{len(mism)} problems, first {mism[:3]}")

    # --- B2. saved logits are plausible ---
    print("\nB2. SAVED LOGITS ARE PHYSICALLY PLAUSIBLE")
    # The first full run produced isolated logits as large as 1.2e35 (one
    # participant in one epoch) from a transient GPU backend fault, while
    # training was otherwise fine. A rank based AUC barely moves, so it slipped
    # past every obvious check, but it would have wrecked the standardisation
    # constants and any regression using the score as a column. This catches it
    # if it happens again.
    import glob
    import trainer as TT
    worst_mag, offenders, n_units = 0.0, [], 0
    for d in sorted(glob.glob(os.path.join(RESULTS, "units", "*"))):
        f = os.path.join(d, "val_logits.npz")
        if not os.path.exists(f) or "smoke" in d:
            continue
        n_units += 1
        with np.load(f) as z:
            L = z["logits"].astype(np.float64)
        mag = float(np.nanmax(np.abs(L)))
        worst_mag = max(worst_mag, mag)
        if not np.all(np.isfinite(L)) or mag > TT.SANE_LOGIT:
            offenders.append((os.path.basename(d), mag))
    check("B", "B5 every saved validation logit is physically plausible",
          not offenders,
          f"{n_units} units, worst magnitude {worst_mag:.4g}, bound "
          f"{TT.SANE_LOGIT}" if not offenders
          else f"{len(offenders)} units corrupted, worst "
               f"{max(o[1] for o in offenders):.4g}: "
               f"{[o[0] for o in offenders][:4]}")

    # same check for the test predictions that feed the contrasts
    bad_sel = []
    for f in files:
        z = np.load(os.path.join(SELECT_DIR, f), allow_pickle=True)
        for key in ("test_z_marg", "test_z_inc", "train_z_marg", "train_z_inc"):
            v = z[key]
            v = v[np.isfinite(v)]
            if v.size and np.max(np.abs(v)) > 100.0:
                bad_sel.append((f, key, float(np.max(np.abs(v)))))
    check("B", "B6 standardised scores are on a sane scale", not bad_sel,
          "standardised scores should sit within a few units of zero"
          if not bad_sel else f"{bad_sel[:3]}")

    # record how often the backend misbehaved
    skipped = repaired = total_val = 0
    for d in sorted(glob.glob(os.path.join(RESULTS, "units", "*"))):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp) or "smoke" in d:
            continue
        with open(mp) as fh:
            m = json.load(fh)
        skipped += m.get("batches_skipped_in_training", 0)
        repaired += m.get("validation_batches_repaired", 0)
        total_val += m.get("validation_batches", 0)
    check("B", "B7 backend fault rate is recorded", True,
          f"{skipped} training batches skipped, {repaired} of {total_val} "
          f"validation batches repaired across {n_units} units")

    # --- C. epoch selection ---
    print("\nC. EPOCH SELECTION REPRODUCES FROM THE SAVED LOGITS")
    import select_epoch as SE
    conf = data["confounders"]
    mism_marg, mism_inc, detail = [], [], []
    for f in files[:max(1, len(files))]:
        stem = f[:-4]
        parts = stem.split("_")
        repeat, fold = int(parts[1][1:]), int(parts[2][1:])
        z = np.load(os.path.join(SELECT_DIR, f), allow_pickle=True)
        try:
            units = SE.load_units(args.arm, repeat, fold, args.tag,
                                  z["selection_table"].shape[0])
        except Exception as exc:
            detail.append(f"{f}: {exc}")
            continue
        df, e_marg, e_inc, _ = SE.select_epochs(
            units, conf, y, pid_to_row, z["selection_table"].shape[0],
            verbose=False)
        if e_marg != int(z["e_marg"]):
            mism_marg.append((f, e_marg, int(z["e_marg"])))
        if e_inc != int(z["e_inc"]):
            mism_inc.append((f, e_inc, int(z["e_inc"])))
    check("C", "C1 E*_marg reproduces from the saved validation logits",
          not mism_marg and not detail,
          f"{len(files)} folds" if not mism_marg
          else f"mismatches {mism_marg[:3]}")
    check("C", "C2 E*_inc reproduces from the saved validation logits",
          not mism_inc and not detail,
          f"{len(files)} folds" if not mism_inc
          else f"mismatches {mism_inc[:3]}")

    # The chosen epoch shouldn't be the last one in every fold. If it always is,
    # the cap is binding and the model is undertrained. That's a finding about
    # the model more than a bug, but it has to be reported.
    last = []
    for f in files:
        z = np.load(os.path.join(SELECT_DIR, f), allow_pickle=True)
        n_ep = z["selection_table"].shape[0]
        last.append(int(z["e_inc"]) == n_ep - 1)
    frac = float(np.mean(last)) if last else 0.0
    check("C", "C3 the epoch cap is not binding in every fold", frac < 1.0,
          f"E*_inc is the final epoch in {int(sum(last))} of {len(last)} folds"
          + ("; the cap is binding, so the model is undertrained and this "
             "must be reported" if frac == 1.0 else ""))

    # --- D. re-score from the saved weights ---
    print("\nD. TEST PREDICTIONS REPRODUCE FROM THE SAVED WEIGHTS")
    if args.reproduce_folds > 0:
        import torch
        import data as D
        store = D.SignalStore()
        dev = torch.device("mps" if torch.backends.mps.is_available()
                           else "cpu")
        worst = 0.0
        done = 0
        for f in files[:args.reproduce_folds]:
            stem = f[:-4]
            parts = stem.split("_")
            repeat, fold = int(parts[1][1:]), int(parts[2][1:])
            z = np.load(os.path.join(SELECT_DIR, f), allow_pickle=True)
            units = SE.load_units(args.arm, repeat, fold, args.tag,
                                  z["selection_table"].shape[0])
            got = SE.score_test(args.arm, units, int(z["e_marg"]), store,
                                z["test_pid"].astype(int),
                                y[[pid_to_row[int(p)]
                                   for p in z["test_pid"]]], dev)
            if not np.array_equal(got["pid"], z["test_pid"].astype(int)):
                check("D", f"D identifier order for {f}", False,
                      "identifier order differs on re-scoring")
                continue
            worst = max(worst, float(np.max(np.abs(got["z_mean"]
                                                  - z["test_z_marg"]))))
            done += 1
        check("D", "D1 re-scoring the saved weights reproduces the test logits",
              worst < 1e-3,
              f"worst absolute difference {worst:.2e} over {done} folds "
              f"(the backend is not bitwise deterministic, so a tolerance of "
              f"1e-3 is the honest bar)")
    else:
        print("  (skipped by request)")

    # --- E. reported metrics ---
    print("\nE. REPORTED METRICS REPRODUCE FROM THE SAVED VECTORS")
    ev = os.path.join(RESULTS, f"stack_eval_{args.arm}{args.tag}.json")
    oofp = os.path.join(RESULTS, f"oof_{args.arm}{args.tag}.npz")
    if os.path.exists(ev) and os.path.exists(oofp):
        with open(ev) as fh:
            rep = json.load(fh)
        z = np.load(oofp)
        name_to_key = {"deep SpO2 only (calibrated)": "deep_platt_mean",
                       "clinical confounders only": "clinical_mean",
                       "clinical plus deep": "stacked_mean"}
        worst_auc, worst_brier = 0.0, 0.0
        for row in rep["headline"]:
            k = name_to_key.get(row["model"])
            if k is None or k not in z.files:
                continue
            v = z[k]
            m = np.isfinite(v)
            worst_auc = max(worst_auc,
                            abs(roc_auc_score(y[m], v[m]) - row["auc_roc"]))
            worst_brier = max(worst_brier,
                              abs(brier_score_loss(y[m], v[m]) - row["brier"]))
        check("E", "E1 reported AUC reproduces from the saved predictions",
              worst_auc < 1e-6, f"worst difference {worst_auc:.2e}")
        check("E", "E2 reported Brier score reproduces", worst_brier < 1e-6,
              f"worst difference {worst_brier:.2e}")

        # The stacked and clinical vectors have to differ. If they don't, the
        # deep column is doing nothing and any reported difference is just
        # rounding.
        d = float(np.nanmax(np.abs(z["stacked_mean"] - z["clinical_mean"])))
        check("E", "E3 the stacked and clinical predictions actually differ",
              d > 1e-6,
              f"largest per participant difference in predicted risk {d:.4f}")

        margin = rep.get("equivalence_margin")
        check("E", "E4 the equivalence margin is the pre-registered 0.005",
              margin == 0.005, f"margin {margin}")
    else:
        check("E", "E1 reported metrics reproduce", False,
              "stack_eval output not found; run stack_eval.py")

    # --- F. pre-registered constants ---
    print("\nF. PRE-REGISTERED CONSTANTS MATCH WHAT WAS USED")
    src = open(os.path.join(AIM3_DIR, "run_unit.py")).read()
    tree = ast.parse(src)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    expected = {"EPOCH_CAP": 16, "LR": 1e-3, "WEIGHT_DECAY": 1e-2,
                "BATCH": 32, "EMA_DECAY": 0.99, "GRAD_CLIP": 1.0}
    wrong = {k: (consts.get(k), v) for k, v in expected.items()
             if consts.get(k) != v}
    check("F", "F1 the frozen training constants are unchanged", not wrong,
          "epoch cap 16, lr 1e-3, weight decay 1e-2, batch 32, EMA 0.99, "
          "clip 1.0" if not wrong else f"changed: {wrong}")

    # which configuration was used, and whether the optimisation changed it
    opt = os.path.join(RESULTS, "optimisation_result.json")
    if os.path.exists(opt):
        with open(opt) as fh:
            o = json.load(fh)
        check("F", "F1b the optimisation used no cardiovascular label",
              o.get("contaminates_evaluation") is False,
              f"search ran on: {o.get('task', 'unknown')}")
        import run_unit as RU
        resolved, source = RU.resolve_config()
        expect = dict(RU.PREREGISTERED)
        expect.update({k: v for k, v in o.get("adopted", {}).items()
                       if k in expect})
        check("F", "F1c the resolved configuration matches what was adopted",
              resolved == expect,
              f"source: {source}" if resolved == expect
              else f"resolved {resolved} against adopted {expect}")

    # check the units on disk actually used them
    used = []
    for f in files:
        stem = f[:-4]
        parts = stem.split("_")
        repeat, fold = int(parts[1][1:]), int(parts[2][1:])
        for inner in range(3):
            mp = os.path.join(RESULTS, "units",
                              f"{args.arm}_r{repeat}_f{fold}_i{inner}"
                              f"{args.tag}", "meta.json")
            if os.path.exists(mp):
                with open(mp) as fh:
                    m = json.load(fh)
                used.append((m["lr"], m["batch"], m["weight_decay"],
                             m["ema_decay"], m["epochs"],
                             json.dumps(m.get("config", {}), sort_keys=True),
                             m.get("config_source", "")))
    uniq = sorted(set(used))
    check("F", "F2 every unit on disk used one identical configuration",
          len(uniq) <= 1, f"{len(uniq)} distinct configurations: {uniq[:3]}")

    import models as M
    check("F", "F3 the primary arm reads one channel only",
          M.ARMS["A"]["channels"] == 1 and M.ARMS["A"]["input"] == "primary",
          "so it cannot see the sleep mask or the REM indicator as inputs")
    check("F", "F4 the negative control arm carries no saturation",
          M.ARMS["NC"]["channels"] == 2 and M.ARMS["NC"]["input"] == "nc")

    import trainer as T
    check("F", "F5 the loss is unweighted cross entropy", T.uses_plain_bce())
    check("F", "F6 no epoch is chosen inside a fit",
          T.selects_epoch_after_training())

    # --- G. output hygiene ---
    print("\nG. OUTPUT HYGIENE")
    offenders = []
    for fn in sorted(os.listdir(AIM3_DIR)):
        if fn.endswith((".py", ".sh", ".md")):
            txt = open(os.path.join(AIM3_DIR, fn), encoding="utf-8").read()
            if EM_DASH in txt:
                offenders.append(fn)
    for extra in ["AIM3_RESULTS.md"]:
        p = os.path.join(AIM3_DIR, "..", extra)
        if os.path.exists(p):
            if EM_DASH in open(p, encoding="utf-8").read():
                offenders.append(extra)
    check("G", "G1 no em dashes in Aim 3 code or reports", not offenders,
          f"{len(os.listdir(AIM3_DIR))} files scanned" if not offenders
          else f"found in {offenders}")

    return finish()


def finish():
    t = pd.DataFrame(checks)
    n_pass = int((t["status"] == "PASS").sum())
    print("\n" + "=" * 76)
    print(f"RESULT: {n_pass} of {len(t)} checks passed")
    if n_pass < len(t):
        print("\nFAILURES:")
        for _, r in t[t["status"] == "FAIL"].iterrows():
            print(f"  {r['check']}: {r['detail']}")
    print("=" * 76)
    os.makedirs(RESULTS, exist_ok=True)
    t.to_csv(os.path.join(RESULTS, "verification_aim3.csv"), index=False)
    print(f"wrote {RESULTS}/verification_aim3.csv")
    return 0 if n_pass == len(t) else 1


if __name__ == "__main__":
    sys.exit(main())
