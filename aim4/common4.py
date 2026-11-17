"""
Aim 4: shared paths, settings and small helpers.

Aim 4 looks at what the models use rather than how well they do. It has three
parts: out of fold SHAP for the classical models (shap_classical.py),
attribution for the deep model checked by occlusion (deep_attrib.py,
occlusion.py), and a head to head comparison of the two pipelines
(headtohead.py).

Everything is out of fold: each of the 15 outer folds explains only its own
test rows, with the model fitted on its training rows. The folds are the Aim 2
folds, loaded through the same verified source Aim 3 uses.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AIM2_DIR = os.path.join(HERE, "..", "aim2")
AIM3_DIR = os.path.join(HERE, "..", "aim3")
for p in (AIM2_DIR, AIM3_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

DRIVE = "/Volumes/ThesisHD/Data/processed"
AIM2_RESULTS = os.path.join(DRIVE, "aim2", "results")
AIM3_RESULTS = os.path.join(DRIVE, "aim3", "results")
RESULTS = os.path.join(DRIVE, "aim4", "results")
FEATURES = os.path.join(DRIVE, "aim2", "obm_features.csv")

# The three nested feature sets from Aim 2. The names are the manifest keys, so
# a typo raises an error instead of selecting some other column set.
SPECS = ("confounders_only", "obm_only", "obm_plus_confounders")

# Classical models to explain. Logistic regression is the main one: it's the
# best of the three on both specs that matter here, and it's the only one whose
# per fold hyperparameters Aim 2 and Aim 3 already saved, so the model explained
# is exactly the published model.
CLASSICAL_MODELS = ("logistic_regression", "xgboost", "random_forest")

# Deep arm and checkpoint to explain. E*_marg is the epoch picked for standalone
# discrimination, which is what the head to head comparison is about. E*_inc was
# picked for what the waveform adds on top of the clinical variables, and that
# came out null, so there's nothing to attribute.
DEEP_ARM = "A"
DEEP_REPEAT = 0
DEEP_EPOCH_KIND = "marg"

# SpO2 bins for summarising sample level attribution. Fixed before any
# attribution was run, using the clinical cut points the biomarker library
# already uses (90, 85 and 80 percent for T90, T85 and T80) plus a finer split
# of the normal range where most of the night sits.
SPO2_BIN_EDGES = (50.0, 80.0, 85.0, 88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.01)

# architecture A pools by two, five times, so one step is 32 one second samples
STEP_SECONDS = 32

# logits beyond this come from a backend fault (same cutoff as Aim 3)
SANE_LOGIT = 50.0

EM_DASH = chr(0x2014)


def ensure_results():
    os.makedirs(RESULTS, exist_ok=True)
    return RESULTS


def bin_index(percent, edges=SPO2_BIN_EDGES):
    """
    SpO2 bin index for each sample, or -1 for missing (non-finite) values.

    np.digitize with right=False puts a value sitting on an edge in the upper
    bin, which matches the biomarker definitions (T90 counts strictly below 90).
    Values outside the outer edges go into the first or last bin.
    """
    idx = np.digitize(percent, np.asarray(edges[1:-1], dtype=float),
                      right=False)
    out = np.where(np.isfinite(percent), idx, -1)
    return out.astype(np.int16)


def bin_labels(edges=SPO2_BIN_EDGES):
    lab = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        hi = 100.0 if hi > 100.0 else hi
        lab.append(f"{lo:.0f} to {hi:.0f}")
    return lab


def no_em_dash(paths):
    """
    Return the files in paths that contain an em dash.

    The character is built from its code point so this function can't trip its
    own check.
    """
    bad = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            if EM_DASH in fh.read():
                bad.append(p)
    return bad


def spearman_rank_agreement(a, b):
    """
    Spearman correlation between two importance rankings, plus the n used.

    Shared because three analyses compare two orderings of the same features
    and they should all do it the same way.
    """
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return {"rho": float("nan"), "p_value": float("nan"),
                "n": int(m.sum())}
    rho, p = stats.spearmanr(a[m], b[m])
    return {"rho": float(rho), "p_value": float(p), "n": int(m.sum())}
