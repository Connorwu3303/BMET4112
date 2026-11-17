"""
Aim 3: rebuilds the Aim 2 logistic regression comparators per fold and per repeat.

The main contrast needs the clinical model with and without the deep score to
differ by exactly one column. Aim 2 never saved per fold hyperparameters for
its confounders only model, or per repeat out of fold scores, so this re-runs
its nested cross validation for all three specifications using the Aim 2 code
itself and saves both.

The rebuilt repeat averaged AUC has to match the Aim 2 figure to within 5e-4.
Logistic regression is deterministic given its folds and grid, so a mismatch
means the two aims aren't running the same model.

Inputs: processed/aim2/modelling_dataset.csv (via ml_common),
processed/aim2/results/logreg_oof_scores.npz for the cross check
Outputs (processed/aim3/results/): aim2_frozen_params_by_fold.csv,
aim2_oof_per_repeat.npz, aim2_rebuild_summary.json

Usage:
    python3 aim3/aim2_comparators.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402

import ml_common as MC                               # noqa: E402
from train_logistic_regression import (              # noqa: E402
    penalty_grid, make_search_factory, IMBALANCE_ARMS,
)

RESULTS = "/Volumes/ThesisHD/Data/processed/aim3/results"
AIM2_RESULTS = "/Volumes/ThesisHD/Data/processed/aim2/results"

# Imbalance strategy Aim 2 picked in stage 1 and used throughout. Hard coded
# from the saved record: picking it again would be another data driven choice,
# and the saved value is what the published comparator numbers came from.
CHOSEN_ARM = "class_weight_balanced"

# Aim 2 figures the rebuild has to reproduce
EXPECTED_AUC = {"confounders_only": 0.7799, "obm_only": 0.6100,
                "obm_plus_confounders": 0.7792}


def rebuild(spec, verbose=True):
    """
    Re-runs the Aim 2 nested cross validation for one specification.

    Returns per fold parameters, per repeat out of fold predictions and the
    labels. No new modelling here, just the Aim 2 pipeline re-run so its
    intermediate values can be saved.
    """
    X, y, features, meta = MC.load_dataset(spec)
    label, imbalance, class_weight = [a for a in IMBALANCE_ARMS
                                      if a[0] == CHOSEN_ARM][0]
    grid = penalty_grid(quick=False)
    X_values = X.to_numpy(dtype=float)

    n_repeats = MC.N_OUTER_REPEATS
    oof = np.full((n_repeats, len(y)), np.nan)
    rows = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=MC.N_OUTER_FOLDS, shuffle=True,
                              random_state=MC.RANDOM_STATE + repeat)
        for fold, (tr, te) in enumerate(skf.split(X_values, y)):
            t0 = time.time()
            search = make_search_factory(grid, imbalance, class_weight)()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_values[tr], y[tr])
            scores = search.best_estimator_.predict_proba(X_values[te])[:, 1]
            oof[repeat, te] = scores
            rows.append({
                "specification": spec, "repeat": repeat, "fold": fold,
                "n_train": len(tr), "n_test": len(te),
                "n_test_events": int(y[te].sum()),
                "auc": float(roc_auc_score(y[te], scores)),
                "inner_best_score": float(search.best_score_),
                "seconds": time.time() - t0,
                **{k: v for k, v in search.best_params_.items()},
            })
            if verbose:
                print(f"    repeat {repeat} fold {fold}: "
                      f"AUC {rows[-1]['auc']:.4f}  "
                      f"{rows[-1]['seconds']:.0f}s  "
                      f"{search.best_params_.get('model__penalty')} "
                      f"C={search.best_params_.get('model__C'):.4g}",
                      flush=True)

    mean_oof = np.nanmean(oof, axis=0)
    return {"spec": spec, "y": y, "oof_per_repeat": oof,
            "mean_oof": mean_oof, "folds": pd.DataFrame(rows),
            "features": features, "meta": meta,
            "auc_mean_oof": float(roc_auc_score(y, mean_oof)),
            "auc_per_repeat": [float(roc_auc_score(y, oof[r]))
                               for r in range(n_repeats)]}


def main():
    os.makedirs(RESULTS, exist_ok=True)
    print("=" * 72)
    print("REBUILDING THE AIM 2 LOGISTIC REGRESSION COMPARATORS")
    print("=" * 72)
    print(f"imbalance arm: {CHOSEN_ARM} (as selected by Aim 2 stage 1)")

    all_folds, summary = [], []
    saved = {}
    for spec in ("confounders_only", "obm_only", "obm_plus_confounders"):
        print(f"\n{spec}")
        res = rebuild(spec)
        all_folds.append(res["folds"])
        saved[spec] = np.vstack([res["y"], res["mean_oof"]])
        saved[f"{spec}__per_repeat"] = res["oof_per_repeat"]

        expected = EXPECTED_AUC[spec]
        got = res["auc_mean_oof"]
        ok = abs(got - expected) < 5e-4
        summary.append({"specification": spec, "auc_rebuilt": got,
                        "auc_aim2_reported": expected,
                        "difference": got - expected,
                        "reproduces": bool(ok),
                        "auc_per_repeat": res["auc_per_repeat"]})
        print(f"  repeat averaged AUC {got:.4f}, Aim 2 reported {expected:.4f}, "
              f"difference {got - expected:+.5f}  "
              f"[{'reproduces' if ok else 'DOES NOT REPRODUCE'}]")
        print(f"  per repeat AUC: "
              f"{', '.join(f'{v:.4f}' for v in res['auc_per_repeat'])}")

        # cross check against the scores Aim 2 actually saved. Stronger than
        # comparing one summary number since it compares the 4,352 predictions
        # one by one
        f = os.path.join(AIM2_RESULTS, "logreg_oof_scores.npz")
        if os.path.exists(f):
            with np.load(f) as z:
                if spec in z.files:
                    ref = z[spec]
                    r = float(np.corrcoef(ref[1], res["mean_oof"])[0, 1])
                    worst = float(np.max(np.abs(ref[1] - res["mean_oof"])))
                    print(f"  against the saved Aim 2 predictions: "
                          f"r = {r:.6f}, worst absolute difference {worst:.2e}")
                    summary[-1]["r_with_saved"] = r
                    summary[-1]["max_abs_diff_with_saved"] = worst

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(os.path.join(RESULTS, "aim2_frozen_params_by_fold.csv"),
                 index=False)
    np.savez_compressed(os.path.join(RESULTS, "aim2_oof_per_repeat.npz"),
                        **saved)
    with open(os.path.join(RESULTS, "aim2_rebuild_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n" + "=" * 72)
    n_ok = sum(1 for s in summary if s["reproduces"])
    print(f"RESULT: {n_ok} of {len(summary)} specifications reproduce Aim 2")
    if n_ok < len(summary):
        print("A specification that does not reproduce means Aim 2 and Aim 3 are "
              "not running the same comparator. Stop and resolve it before any "
              "contrast is computed.")
    print(f"wrote {RESULTS}/aim2_frozen_params_by_fold.csv")
    print(f"wrote {RESULTS}/aim2_oof_per_repeat.npz")
    print("=" * 72)
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
