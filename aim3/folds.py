"""
Aim 3 step 2: rebuilds the Aim 2 cross validation folds exactly.

The deep and classical models have to be scored on the same participants in the
same held out folds so they can be compared with a paired test.
ml_common.nested_cv makes its outer folds with

    StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + repeat)

which only depends on the labels, their row order and the seed, so the same call
on the same label vector gives identical folds. load_cohort keeps the same rows
in the same file order as Aim 2's load_dataset. verify_against_aim2 checks
training size, test size and test events in all 15 folds against
logreg_fold_metrics.csv and raises on any mismatch.

Usage (from aim3/):
    python3 folds.py    # rebuilds, verifies and prints a summary
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_ROOT = "/Volumes/ThesisHD/Data/processed"
DATASET_PATH = os.path.join(DATA_ROOT, "aim2", "modelling_dataset.csv")
MANIFEST_PATH = os.path.join(DATA_ROOT, "aim2", "feature_manifest.json")
RESULTS_DIR = os.path.join(DATA_ROOT, "aim2", "results")

# These four have to match ml_common. They're copied here so Aim 3 doesn't need
# the Aim 2 modelling stack at import time, and assert_constants_match_aim2
# checks they still agree.
RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_OUTER_REPEATS = 3
PRIMARY_OUTCOME = "cvd_incident"


def assert_constants_match_aim2():
    """
    Check the constants above still match the Aim 2 source.

    If they drifted apart (say Aim 2 was rerun with four repeats), the folds
    would stop lining up while every other check in this file still passed.
    Reading the values back from the Aim 2 module makes that fail loudly.
    """
    import sys
    aim2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "aim2")
    sys.path.insert(0, aim2)
    import ml_common  # noqa: E402

    pairs = [
        ("RANDOM_STATE", RANDOM_STATE, ml_common.RANDOM_STATE),
        ("N_OUTER_FOLDS", N_OUTER_FOLDS, ml_common.N_OUTER_FOLDS),
        ("N_OUTER_REPEATS", N_OUTER_REPEATS, ml_common.N_OUTER_REPEATS),
        ("PRIMARY_OUTCOME", PRIMARY_OUTCOME, ml_common.PRIMARY_OUTCOME),
    ]
    bad = [f"{name}: aim3 has {mine!r}, aim2 has {theirs!r}"
           for name, mine, theirs in pairs if mine != theirs]
    if bad:
        raise AssertionError("Aim 3 constants no longer match Aim 2:\n  "
                             + "\n  ".join(bad))
    return True


def load_cohort(outcome=PRIMARY_OUTCOME):
    """
    Load the analysis cohort in the same row order Aim 2 used.

    Returns (pid, y, confounders, confounder_names, table). confounders is the
    18 column clinical block, unscaled and still with NaN, since imputation and
    scaling have to happen inside each training fold.
    """
    table = pd.read_csv(DATASET_PATH, low_memory=False)
    with open(MANIFEST_PATH) as fh:
        manifest = json.load(fh)

    # same filter as load_dataset: rows with no outcome are dropped (never
    # treated as event free), file order kept
    table = table[table[outcome].notna()].copy().reset_index(drop=True)

    y = table[outcome].astype(int).to_numpy()
    pid = table["pid"].to_numpy()

    spec = manifest["model_specifications"]["confounders_only"]
    names = [c for c in spec if c in table.columns]
    confounders = table[names].astype(float).to_numpy()

    return pid, y, confounders, names, table


def build_folds(y, n_outer=N_OUTER_FOLDS, n_repeats=N_OUTER_REPEATS,
                random_state=RANDOM_STATE):
    """
    Rebuild the Aim 2 outer folds.

    The feature matrix passed to split is just a column of zeros, since
    StratifiedKFold only uses the row count and labels. Using zeros makes it
    obvious no feature values can affect the partition.
    """
    dummy = np.zeros((len(y), 1))
    folds = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True,
                              random_state=random_state + repeat)
        for fold, (train_idx, test_idx) in enumerate(skf.split(dummy, y)):
            folds.append({"repeat": repeat, "fold": fold,
                          "train_idx": train_idx, "test_idx": test_idx})
    return folds


def verify_against_aim2(y, folds, quiet=False):
    """
    Check the reconstructed folds against the folds Aim 2 actually ran.

    Raises AssertionError on any mismatch. Returns the comparison table.
    """
    path = os.path.join(RESULTS_DIR, "logreg_fold_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; cannot verify folds against Aim 2")
    aim2 = pd.read_csv(path)

    rows = []
    for f in folds:
        ref = aim2[(aim2["repeat"] == f["repeat"]) &
                   (aim2["fold"] == f["fold"])]
        if len(ref) != 1:
            raise AssertionError(
                f"expected exactly one Aim 2 row for repeat {f['repeat']} "
                f"fold {f['fold']}, found {len(ref)}")
        ref = ref.iloc[0]
        got = {"n_train": len(f["train_idx"]), "n_test": len(f["test_idx"]),
               "n_test_events": int(y[f["test_idx"]].sum())}
        want = {k: int(ref[k]) for k in got}
        rows.append({"repeat": f["repeat"], "fold": f["fold"],
                     **{f"{k}_aim3": v for k, v in got.items()},
                     **{f"{k}_aim2": v for k, v in want.items()},
                     "match": got == want})

    table = pd.DataFrame(rows)
    if len(table) != len(aim2):
        raise AssertionError(
            f"reconstructed {len(table)} folds but Aim 2 recorded {len(aim2)}")
    if not table["match"].all():
        bad = table[~table["match"]]
        raise AssertionError(
            "reconstructed folds do not match Aim 2:\n" + bad.to_string())

    # each participant tested exactly once per repeat, so the pooled out of
    # fold score never mixes seen and unseen predictions
    for repeat in sorted({f["repeat"] for f in folds}):
        covered = np.concatenate([f["test_idx"] for f in folds
                                 if f["repeat"] == repeat])
        if len(covered) != len(y) or len(np.unique(covered)) != len(y):
            raise AssertionError(
                f"repeat {repeat} does not test each participant exactly once")

    # Train and test have to be disjoint in every fold. StratifiedKFold already
    # guarantees this, so it's really here to catch a later edit to this file.
    for f in folds:
        if np.intersect1d(f["train_idx"], f["test_idx"]).size:
            raise AssertionError(
                f"repeat {f['repeat']} fold {f['fold']} has overlapping "
                f"train and test indices")

    if not quiet:
        print(f"folds verified against Aim 2: {len(table)} of {len(table)} "
              f"folds match on train size, test size and test event count")
    return table


def get_verified_folds(outcome=PRIMARY_OUTCOME, quiet=False):
    """
    Load the cohort, build the folds and verify them.

    Every Aim 3 training script gets its folds through here, so no script can
    quietly make its own split.
    """
    assert_constants_match_aim2()
    pid, y, conf, conf_names, table = load_cohort(outcome)
    folds = build_folds(y)
    verify_against_aim2(y, folds, quiet=quiet)
    return {"pid": pid, "y": y, "confounders": conf,
            "confounder_names": conf_names, "folds": folds, "table": table}


if __name__ == "__main__":
    data = get_verified_folds()
    y = data["y"]
    print(f"\ncohort: {len(y)} participants, {int(y.sum())} events "
          f"({y.mean() * 100:.1f} percent)")
    print(f"confounders: {len(data['confounder_names'])} columns")
    print(f"  {data['confounder_names']}")
    print(f"folds: {len(data['folds'])} "
          f"({N_OUTER_REPEATS} repeats x {N_OUTER_FOLDS} outer folds)")
    f0 = data["folds"][0]
    print(f"\nfirst fold: {len(f0['train_idx'])} train, "
          f"{len(f0['test_idx'])} test, "
          f"{int(y[f0['test_idx']].sum())} test events")
    print(f"first 10 test pids: {data['pid'][f0['test_idx']][:10]}")
