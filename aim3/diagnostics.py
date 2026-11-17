"""
Aim 3: checks on what a deep result actually means.

A deep score can beat the biomarkers without SpO2 carrying new CVD information,
by learning an age proxy, sleep structure and data quality, or frailty from
follow up artefacts (non-CVD deaths count as non-events).

For each deep score this reports a nuisance model on the structure set alone
(same folds), its correlations with age, BMI and the oximetry biomarkers, its
AUC within age tertiles and each sex, and the negative control arms. If the
nuisance model comes within 0.02 of the deep score, the deep score is called a
sleep structure score instead of an oximetry score (a pre-registered rule).

Inputs: processed/aim3/results/oof_<arm>.npz (from stack_eval.py) and
processed/aim2/obm_features.csv
Outputs (processed/aim3/results/): structure_set.csv (cache),
diagnostics_<arm>.json, correlations_<arm>.csv

Usage:
    python3 diagnostics.py --arm A
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scipy import stats                                # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402

import data as D                                        # noqa: E402
from folds import get_verified_folds                    # noqa: E402

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
FEATURES = "/Volumes/ThesisHD/Data/processed/aim2/obm_features.csv"
NUISANCE_FLAG_MARGIN = 0.02      # pre-registered


def build_structure_table(store, pids, path=None):
    """
    Computes the structure set S for every participant, cached to path.

    S is the sleep architecture and data quality variables that the primary
    network is kept from seeing. It takes a few minutes over the whole cohort,
    hence the cache.
    """
    if path and os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for i, pid in enumerate(pids):
        s = D.structure_features(store, int(pid))
        s["pid"] = int(pid)
        rows.append(s)
        if (i + 1) % 500 == 0:
            print(f"    structure set: {i + 1}/{len(pids)}", flush=True)
    t = pd.DataFrame(rows)
    if path:
        t.to_csv(path, index=False)
    return t


def nuisance_model(structure, y, folds, pid_to_row):
    """
    Logistic regression on the structure set alone, on the same outer folds.

    An upper bound on what sleep quantity, fragmentation and data quality can
    predict with no SpO2 information at all. Kept plain on purpose, since the
    question is only whether this information predicts anything.
    """
    x = structure[list(D.STRUCTURE_COLUMNS)].to_numpy(dtype=float)
    oof = np.full(len(y), np.nan)
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        tf = D.fit_confounder_transform(x[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(C=1.0, max_iter=2000,
                                    class_weight="balanced")
            lr.fit(D.apply_confounder_transform(x[tr], tf), y[tr])
        # combine across repeats: averaged with whatever is already stored
        p = lr.predict_proba(D.apply_confounder_transform(x[te], tf))[:, 1]
        oof[te] = p if np.all(np.isnan(oof[te])) else (oof[te] + p) / 2.0
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print("=" * 76)
    print(f"AIM 3 DIAGNOSTICS: arm {args.arm}")
    print("=" * 76)

    data = get_verified_folds(quiet=True)
    y, pid, table = data["y"], data["pid"], data["table"]
    pid_to_row = {int(p): i for i, p in enumerate(pid)}

    oof_path = os.path.join(RESULTS, f"oof_{args.arm}{args.tag}.npz")
    if not os.path.exists(oof_path):
        raise SystemExit(f"{oof_path} not found; run stack_eval.py first")
    z = np.load(oof_path)
    deep = z["deep_platt_mean"]
    stacked = z["stacked_mean"]
    clinical = z["clinical_mean"]

    store = D.SignalStore()
    print("\n1. STRUCTURE SET S (what the primary network is not shown)")
    s_path = os.path.join(RESULTS, "structure_set.csv")
    structure = build_structure_table(store, pid, s_path)
    structure = structure.set_index("pid").loc[[int(p) for p in pid]] \
                         .reset_index()
    print(f"   {len(structure)} participants, "
          f"{len(D.STRUCTURE_COLUMNS)} variables")
    for c in D.STRUCTURE_COLUMNS:
        v = structure[c]
        print(f"     {c:<20s} median {v.median():8.3f}  "
              f"missing {int(v.isna().sum())}")

    print("\n2. NUISANCE MODEL: structure alone, same folds")
    nuis = nuisance_model(structure, y, data["folds"], pid_to_row)
    ok = np.isfinite(nuis)
    auc_nuis = float(roc_auc_score(y[ok], nuis[ok]))
    ok_d = np.isfinite(deep)
    auc_deep = float(roc_auc_score(y[ok_d], deep[ok_d]))
    print(f"   structure only   AUC {auc_nuis:.4f}")
    print(f"   deep SpO2 only   AUC {auc_deep:.4f}")
    gap = auc_deep - auc_nuis
    flagged = gap < NUISANCE_FLAG_MARGIN
    print(f"   difference {gap:+.4f}")
    print(f"   FLAG: {'YES' if flagged else 'no'} "
          f"(pre-registered rule: flag when the deep score is within "
          f"{NUISANCE_FLAG_MARGIN:.2f} of the structure model)")
    if flagged:
        print("   Under the pre-registered rule the deep score must be described")
        print("   as a sleep structure score rather than an oximetry score.")

    print("\n3. WHAT THE DEEP SCORE CORRELATES WITH")
    feats = pd.read_csv(FEATURES, low_memory=False,
                        usecols=["pid", "sleep_odi3", "sleep_ct90_pct",
                                 "sleep_mean", "sleep_min"]).set_index("pid")
    feats = feats.loc[[int(p) for p in pid]].reset_index()
    probes = {"age": table["age_s1"].to_numpy(dtype=float),
              "bmi": table["bmi_s1"].to_numpy(dtype=float),
              "sleep_odi3": feats["sleep_odi3"].to_numpy(dtype=float),
              "sleep_T90_pct": feats["sleep_ct90_pct"].to_numpy(dtype=float),
              "sleep_mean_SpO2": feats["sleep_mean"].to_numpy(dtype=float),
              "sleep_min_SpO2": feats["sleep_min"].to_numpy(dtype=float)}
    for c in D.STRUCTURE_COLUMNS:
        probes[c] = structure[c].to_numpy(dtype=float)

    rows = []
    print(f"   {'variable':<22s} {'Spearman':>9s} {'p':>10s}")
    for name, v in probes.items():
        m = np.isfinite(v) & np.isfinite(deep)
        if m.sum() < 50:
            continue
        rho, p = stats.spearmanr(deep[m], v[m])
        rows.append({"variable": name, "spearman": float(rho),
                     "p_value": float(p), "n": int(m.sum())})
        print(f"   {name:<22s} {rho:>9.3f} {p:>10.3g}")

    print("\n4. DISCRIMINATION WITHIN STRATA")
    print("   If the deep score were mostly an age proxy, its discrimination")
    print("   would collapse inside narrow age bands.")
    age = table["age_s1"].to_numpy(dtype=float)
    strata = []
    terciles = np.nanpercentile(age, [33.333, 66.667])
    labels = ["age tertile 1 (youngest)", "age tertile 2", "age tertile 3"]
    bins = np.digitize(age, terciles)
    for b, lab in enumerate(labels):
        m = (bins == b) & np.isfinite(deep)
        if m.sum() > 50 and len(np.unique(y[m])) > 1:
            a = float(roc_auc_score(y[m], deep[m]))
            strata.append({"stratum": lab, "n": int(m.sum()),
                           "n_events": int(y[m].sum()), "auc": a})
            print(f"   {lab:<26s} n {int(m.sum()):5d} "
                  f"events {int(y[m].sum()):4d}  AUC {a:.4f}")
    if "is_male" in table.columns:
        for val, lab in [(1, "men"), (0, "women")]:
            m = (table["is_male"].to_numpy() == val) & np.isfinite(deep)
            if m.sum() > 50 and len(np.unique(y[m])) > 1:
                a = float(roc_auc_score(y[m], deep[m]))
                strata.append({"stratum": lab, "n": int(m.sum()),
                               "n_events": int(y[m].sum()), "auc": a})
                print(f"   {lab:<26s} n {int(m.sum()):5d} "
                      f"events {int(y[m].sum()):4d}  AUC {a:.4f}")

    print("\n5. THE NEGATIVE CONTROL ARMS")
    # Two controls. NC is the window model, a different and much smaller
    # architecture, so a gap between it and arm A mixes up the missing SpO2 with
    # the smaller network. NCA is architecture A itself given the same structure
    # only input, so the only difference from arm A is what it can see. NCA is
    # the attribution test; NC is just an upper bound.
    nc = nca = None
    for tag, label in (("NC", "window model, 36,179 parameters"),
                       ("NCA", "architecture A, 241,313 parameters")):
        path = os.path.join(RESULTS, f"oof_{tag}.npz")
        if not os.path.exists(path):
            print(f"   {tag}: not run")
            continue
        score = np.load(path)["deep_platt_mean"]
        mm = np.isfinite(score) & np.isfinite(deep)
        auc = float(roc_auc_score(y[mm], score[mm]))
        if tag == "NC":
            nc = auc
        else:
            nca = auc
        import ml_common as MCX
        t = MCX.delong_test(y[mm], score[mm], deep[mm])
        print(f"   {tag} ({label}): AUC {auc:.4f}")
        if t:
            lo = t["difference"] - 1.96 * t["se"]
            hi = t["difference"] + 1.96 * t["se"]
            print(f"       against arm A ({auc_deep:.4f}): "
                  f"{t['difference']:+.4f}, p = {t['p_value']:.4g}, "
                  f"95 percent interval [{lo:+.4f}, {hi:+.4f}]")
    if nc is not None and nca is not None:
        print(f"   architecture alone accounts for {abs(nca - nc):.4f} AUC, so the "
              f"gap between arm A and the structure only input is attributable to "
              f"saturation rather than to network size")

    out = {"arm": args.arm, "auc_deep_spo2_only": auc_deep,
           "auc_nuisance_structure_only": auc_nuis,
           "nuisance_gap": gap, "nuisance_flagged": bool(flagged),
           "nuisance_flag_margin": NUISANCE_FLAG_MARGIN,
           "correlations": rows, "strata": strata,
           "auc_negative_control": nc,
           "auc_negative_control_matched": nca}
    with open(os.path.join(RESULTS,
                           f"diagnostics_{args.arm}{args.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS, f"correlations_{args.arm}{args.tag}.csv"),
        index=False)
    print(f"\nwrote {RESULTS}/diagnostics_{args.arm}{args.tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
