"""
Shared code for the Aim 2 classical models: data loading, the preprocessing
pipeline, metrics and DeLong, nested CV, the two-stage search and plots.

Logistic regression, random forest and XGBoost all go through this module, so
they get the same folds, preprocessing, metrics and seeds and the Aim 4
comparison is fair.

Imputation, scaling and SMOTETomek are fitted from data, so all three sit in
an imblearn Pipeline that's refit on each training fold, and the test fold is
never resampled. Hyperparameters are picked in the inner CV loop and every
reported number comes from the outer folds. Accuracy isn't a headline metric
for an imbalanced outcome, so the main ones are AUC-ROC and AUPRC (whose
no-skill baseline is the event rate, not 0.5), plus Brier score for
calibration, which resampling is known to hurt.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- warning filters ---
# Set at module level because the searches run under joblib, and the workers
# re-import this module but don't inherit a catch_warnings block. Only two
# noisy types are filtered: a FutureWarning from imbalanced-learn 0.12 calling
# a scikit-learn internal that 1.6 deprecated, and ConvergenceWarning from saga
# at the extremes of the penalty grid (those candidates just lose).
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"sklearn\..*")
try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# --- config ---

OUTPUT_DIR = "/Volumes/ThesisHD/Data/processed"
AIM2_DIR = os.path.join(OUTPUT_DIR, "aim2")
DATASET_PATH = os.path.join(AIM2_DIR, "modelling_dataset.csv")
MANIFEST_PATH = os.path.join(AIM2_DIR, "feature_manifest.json")
RESULTS_DIR = os.path.join(AIM2_DIR, "results")

# one seed everywhere so every model gets the same folds, and any difference
# between models can't be down to the split
RANDOM_STATE = 42

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

# Outer loop is repeated with different fold assignments. A single 5-fold
# estimate on a few hundred events has a standard error of several AUC points,
# so repeats show how much the result moves with the split.
N_OUTER_REPEATS = 3

PRIMARY_OUTCOME = "cvd_incident"

# the nested model specifications (build_modelling_dataset.py explains why
# confounders only is the reference)
SPECIFICATIONS = ("confounders_only", "obm_only", "obm_plus_confounders")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


# --- hyperparameter types ---

# Parameters scikit-learn checks as strict ints. A pandas column that mixes
# ints with None (max_depth does) hands them back as floats like 6.0, which
# RandomForestClassifier rejects. max_features, min_samples_leaf and
# min_samples_split are left out because they can also take a fraction.
# clean_param_value casts the two min_samples ones only when they're whole
# numbers, and leaves max_features alone.
INTEGER_PARAMS = frozenset({
    "max_depth", "max_iter", "n_estimators", "random_state", "verbose",
    "n_jobs", "degree", "n_neighbors",
})


def clean_param_value(name, value):
    """
    Turns a hyperparameter read back from a pandas column into the plain
    Python type scikit-learn expects. Missing becomes None (max_depth=None is
    valid), numpy scalars become Python ones, and a whole-number float becomes
    an int where it can't be read as a fraction. The name is needed because
    5.0 means 5 for min_samples_leaf but 0.05 means 5% of features for
    max_features.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and value.is_integer():
        # whole-number floats -> int for the strict-int params, and for the
        # fraction-capable ones too, where a whole number means a count
        if name in INTEGER_PARAMS or name in (
            "min_samples_leaf", "min_samples_split"
        ):
            return int(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def clean_params(params, strip_prefix="model__"):
    """
    Cleans a dict of chosen hyperparameters and strips the pipeline prefix,
    so it can go straight into an estimator with **params.
    """
    cleaned = {}
    for key, value in params.items():
        name = key.replace(strip_prefix, "") if strip_prefix else key
        cleaned[name] = clean_param_value(name, value)
    return cleaned


# --- data loading ---

def load_dataset(specification="obm_plus_confounders",
                 outcome=PRIMARY_OUTCOME):
    """
    Loads the modelling dataset for one specification.

    Returns (X, y, feature_names, meta).

    Rows with an undefined outcome are dropped. For the incident outcome that's
    the people with CVD at baseline, who are set to NaN in
    build_modelling_dataset so they can't be counted as event-free by mistake.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run build_modelling_dataset.py first."
        )
    data = pd.read_csv(DATASET_PATH)
    with open(MANIFEST_PATH) as handle:
        manifest = json.load(handle)

    if specification not in manifest["model_specifications"]:
        raise ValueError(
            f"unknown specification {specification}; available: "
            f"{list(manifest['model_specifications'])}"
        )
    features = [
        c for c in manifest["model_specifications"][specification]
        if c in data.columns
    ]

    n_before = len(data)
    data = data[data[outcome].notna()].copy()
    n_dropped = n_before - len(data)

    X = data[features].astype(float)
    y = data[outcome].astype(int).to_numpy()

    meta = {
        "specification": specification,
        "outcome": outcome,
        "n_participants": int(len(data)),
        "n_dropped_outcome_undefined": int(n_dropped),
        "n_features": len(features),
        "n_events": int(y.sum()),
        "event_rate": float(y.mean()),
        "pid": data["pid"].to_numpy() if "pid" in data.columns else None,
    }
    return X, y, features, meta


# --- pipeline ---

def build_pipeline(estimator, imbalance="smotetomek", scale=True,
                   random_state=RANDOM_STATE):
    """
    Wraps an estimator so all fitted preprocessing happens inside the pipeline.

    imbalance is "smotetomek" (SMOTE oversampling, then Tomek-link cleaning of
    the class boundary) or "none" for no resampling (class weights, if used,
    are set on the estimator). scale matters for logistic regression, where
    the penalty depends on feature scale; tree models don't need it.
    """
    steps = [
        # median fill because several biomarkers are very right skewed, and a
        # mean fill would give imputed people implausible values
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    if imbalance == "smotetomek":
        steps.append(
            ("resample", SMOTETomek(random_state=random_state))
        )
    elif imbalance != "none":
        raise ValueError(f"unknown imbalance strategy: {imbalance}")
    steps.append(("model", estimator))
    return ImbPipeline(steps)


# --- metrics ---

def evaluate_predictions(y_true, y_score, threshold=None):
    """
    All the metrics from true labels and predicted probabilities.

    With no threshold given it picks the one that maximises Youden's J
    (sensitivity + specificity - 1), a reasonable default for screening, and
    returns it in the output. The usual 0.5 means very little for an
    imbalanced outcome.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    out = {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        # no-skill AUPRC is the event rate, so read AUPRC against this
        "auprc_baseline": float(np.mean(y_true)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    out["auprc_lift"] = out["auprc"] / out["auprc_baseline"] \
        if out["auprc_baseline"] > 0 else np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    if threshold is None:
        youden = tpr - fpr
        best = int(np.argmax(youden))
        threshold = float(thresholds[best])
    out["threshold"] = float(threshold)

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else np.nan,
        "npv": float(tn / (tn + fn)) if (tn + fn) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y_true)),
        # balanced accuracy averages sensitivity and specificity, so the
        # majority class can't carry the score on its own
        "balanced_accuracy": float(
            0.5 * (tp / (tp + fn) + tn / (tn + fp))
        ) if (tp + fn) and (tn + fp) else np.nan,
    })
    return out


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, alpha=0.05,
                     random_state=RANDOM_STATE):
    """
    Percentile bootstrap CIs for AUC-ROC and AUPRC.

    Resamples participants with replacement (label and score stay paired).
    Resamples with only one class are skipped since AUC is undefined there.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)

    aucs, auprcs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        auprcs.append(average_precision_score(y_true[idx], y_score[idx]))

    def ci(values):
        if not values:
            return (np.nan, np.nan)
        return (
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))),
        )

    lo_auc, hi_auc = ci(aucs)
    lo_ap, hi_ap = ci(auprcs)
    return {
        "auc_ci_lo": lo_auc, "auc_ci_hi": hi_auc,
        "auprc_ci_lo": lo_ap, "auprc_ci_hi": hi_ap,
        "n_boot_used": len(aucs),
    }


def delong_test(y_true, score_a, score_b):
    """
    DeLong test for two correlated ROC curves on the same participants.

    Two models scored on the same people have correlated AUCs, so an unpaired
    comparison overstates the standard error of the difference. DeLong
    estimates that covariance from the placement values, which makes it the
    right test for whether model B beats model A on this cohort.

    Returns the AUC difference, its SE, a z statistic and a two-sided p value.
    """
    y_true = np.asarray(y_true)
    pos = y_true == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return None

    def structural_components(score):
        """
        Placement values per observation (their variances and covariance give
        the DeLong estimate). v10: for each positive, the fraction of
        negatives it outranks, ties counting as half. v01: the same for each
        negative.
        """
        x = np.asarray(score, dtype=float)[pos]
        y = np.asarray(score, dtype=float)[neg]
        # positives vs negatives, every pair
        greater = (x[:, None] > y[None, :]).astype(float)
        equal = (x[:, None] == y[None, :]).astype(float)
        kernel = greater + 0.5 * equal
        v10 = kernel.mean(axis=1)   # one value per positive
        v01 = kernel.mean(axis=0)   # one value per negative
        auc = kernel.mean()
        return auc, v10, v01

    auc_a, v10_a, v01_a = structural_components(score_a)
    auc_b, v10_b, v01_b = structural_components(score_b)

    # covariance matrix of the two AUCs
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))
    cov = s10 / m + s01 / n

    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = auc_b - auc_a
    if var_diff <= 0:
        return {
            "auc_a": float(auc_a), "auc_b": float(auc_b),
            "difference": float(diff), "se": 0.0,
            "z": np.nan, "p_value": np.nan,
        }
    se = float(np.sqrt(var_diff))
    z = float(diff / se)
    p = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(auc_a), "auc_b": float(auc_b),
        "difference": float(diff), "se": se, "z": z, "p_value": p,
    }


# --- nested cross validation ---

def nested_cv(X, y, search_factory, n_outer=N_OUTER_FOLDS,
              n_repeats=N_OUTER_REPEATS, random_state=RANDOM_STATE,
              verbose=True):
    """
    Repeated stratified nested cross validation.

    search_factory takes no arguments and returns a new, unfitted search
    around the full pipeline, so each outer fold starts from scratch. Returns
    per-fold metrics, out-of-fold predictions (per repeat and averaged) and the
    hyperparameters picked in each fold. Headline numbers come from the
    out-of-fold predictions.
    """
    X_values = X.to_numpy(dtype=float) if hasattr(X, "to_numpy") else np.asarray(X)
    y = np.asarray(y)

    fold_rows = []
    chosen_params = []
    # one row of out-of-fold scores per repeat, so repeats stay separate
    oof = np.full((n_repeats, len(y)), np.nan)

    for repeat in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_outer, shuffle=True,
            random_state=random_state + repeat,
        )
        for fold, (train_idx, test_idx) in enumerate(
            skf.split(X_values, y)
        ):
            search = search_factory()
            with warnings.catch_warnings():
                # convergence warnings at the ends of the penalty grid are
                # expected and nothing to act on
                warnings.simplefilter("ignore")
                search.fit(X_values[train_idx], y[train_idx])

            best = search.best_estimator_
            scores = best.predict_proba(X_values[test_idx])[:, 1]
            oof[repeat, test_idx] = scores

            metrics = evaluate_predictions(y[test_idx], scores)
            metrics.update({
                "repeat": repeat, "fold": fold,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "n_test_events": int(y[test_idx].sum()),
                "inner_best_score": float(search.best_score_),
            })
            fold_rows.append(metrics)
            chosen_params.append(
                {"repeat": repeat, "fold": fold, **search.best_params_}
            )

            if verbose:
                print(
                    f"    repeat {repeat} fold {fold}: "
                    f"AUC {metrics['auc_roc']:.4f}  "
                    f"AUPRC {metrics['auprc']:.4f}  "
                    f"(inner {search.best_score_:.4f})"
                )

    folds = pd.DataFrame(fold_rows)
    params = pd.DataFrame(chosen_params)

    # pooled out-of-fold metrics per repeat (fine to pool within a repeat,
    # each participant is in exactly one test fold), averaged later
    pooled = []
    for repeat in range(n_repeats):
        scores = oof[repeat]
        ok = ~np.isnan(scores)
        if ok.sum() and len(np.unique(y[ok])) > 1:
            pooled.append(evaluate_predictions(y[ok], scores[ok]))
    pooled_df = pd.DataFrame(pooled)

    # mean out-of-fold score across repeats, the most stable per-person risk
    # estimate we have (used for the plots)
    mean_oof = np.nanmean(oof, axis=0)

    return {
        "folds": folds,
        "params": params,
        "oof": oof,
        "mean_oof": mean_oof,
        "pooled_per_repeat": pooled_df,
        "y": y,
    }


def summarise_nested_cv(result, label=""):
    """Prints and returns a readable summary of a nested_cv result."""
    folds = result["folds"]
    pooled = result["pooled_per_repeat"]
    lines = []

    def emit(text=""):
        print(text)
        lines.append(text)

    emit(f"\n  {label} nested CV over {len(folds)} outer folds "
         f"({folds['repeat'].nunique()} repeats x "
         f"{folds['fold'].nunique()} folds)")
    emit(f"    {'metric':<22s} {'mean':>8s} {'sd':>8s} {'min':>8s} {'max':>8s}")
    for metric in ("auc_roc", "auprc", "sensitivity", "specificity",
                   "balanced_accuracy", "brier"):
        if metric not in folds.columns:
            continue
        col = folds[metric].dropna()
        emit(f"    {metric:<22s} {col.mean():>8.4f} {col.std():>8.4f} "
             f"{col.min():>8.4f} {col.max():>8.4f}")

    if not pooled.empty:
        emit(f"\n    pooled out-of-fold AUC-ROC : "
             f"{pooled['auc_roc'].mean():.4f} "
             f"(sd {pooled['auc_roc'].std():.4f} across repeats)")
        emit(f"    pooled out-of-fold AUPRC   : "
             f"{pooled['auprc'].mean():.4f} "
             f"(no-skill {pooled['auprc_baseline'].mean():.4f}, "
             f"lift {pooled['auprc_lift'].mean():.2f}x)")

    # inner vs outer gap from hyperparameter selection (the printout below
    # explains why it isn't a clean measure of optimism)
    if "inner_best_score" in folds.columns:
        gap = folds["inner_best_score"].mean() - folds["auc_roc"].mean()
        emit(f"\n    inner selection score      : "
             f"{folds['inner_best_score'].mean():.4f}")
        emit(f"    outer (honest) AUC         : {folds['auc_roc'].mean():.4f}")
        emit(f"    inner minus outer          : {gap:+.4f}")
        emit("    Two effects compete in this number. Selection optimism pushes")
        emit("    the inner score UP, because it is the best of many candidates")
        emit("    scored on the same data. Training set size pushes it DOWN,")
        emit("    because inner models see only a fraction of each training")
        emit("    fold while the outer model sees all of it.")
        if gap > 0:
            emit("    Positive here, so selection optimism dominates. Quoting the")
            emit("    inner score as performance would overstate the model by")
            emit("    roughly this much, which is exactly why the outer loop")
            emit("    exists.")
        else:
            emit("    Negative here, so the training-size effect dominates and")
            emit("    selection optimism is small. That is reassuring: the")
            emit("    hyperparameter search is not overfitting the inner folds.")
            emit("    The outer score remains the one to report either way.")
    return lines


# --- two-stage hyperparameter search ---

class TwoStageSearch:
    """
    Broad random search, then a focused grid around its winner, keeping
    whichever scores better on inner CV. All of it runs inside one outer
    training fold, so the outer test fold never affects the choice.

    Exposes best_estimator_, best_score_ and best_params_ like a scikit-learn
    search, and best_params_ also records search_stage and both inner scores.
    broad_factory() and grid_factory(grid) return unfitted searches, and
    grid_builder turns the broad winner's params into that grid.
    """

    def __init__(self, broad_factory, grid_builder, grid_factory):
        self.broad_factory = broad_factory
        self.grid_builder = grid_builder
        self.grid_factory = grid_factory

    def fit(self, X, y):
        broad = self.broad_factory()
        broad.fit(X, y)

        # clean the winner's values first: random search hands back numpy
        # scalars and some estimators are strict about int params
        centre = {
            key: clean_param_value(key.replace("model__", ""), value)
            for key, value in broad.best_params_.items()
        }
        focused = self.grid_factory(self.grid_builder(centre))
        focused.fit(X, y)

        # same inner splitter on the same training fold, so the two scores
        # compare directly. Ties go to the focused grid.
        use_focused = focused.best_score_ >= broad.best_score_
        winner = focused if use_focused else broad

        self.best_estimator_ = winner.best_estimator_
        self.best_score_ = float(winner.best_score_)
        self.best_params_ = dict(winner.best_params_)
        self.best_params_["search_stage"] = "focused" if use_focused else "broad"
        self.best_params_["inner_broad_score"] = float(broad.best_score_)
        self.best_params_["inner_focused_score"] = float(focused.best_score_)
        return self


# --- plotting ---

def plot_model_performance(y_true, y_score, title, path,
                           extra_curves=None):
    """
    Four-panel figure: ROC, precision-recall, calibration and the predicted
    probability distribution split by outcome.

    extra_curves: optional list of (label, y_score) pairs added to the ROC and
    PR panels for comparison.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    # ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr, lw=2,
            label=f"model (AUC {roc_auc_score(y_true, y_score):.3f})")
    for name, other in (extra_curves or []):
        f2, t2, _ = roc_curve(y_true, other)
        ax.plot(f2, t2, lw=1.2, ls="--",
                label=f"{name} (AUC {roc_auc_score(y_true, other):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)

    # precision-recall
    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ax.plot(recall, precision, lw=2,
            label=f"model (AP {average_precision_score(y_true, y_score):.3f})")
    for name, other in (extra_curves or []):
        p2, r2, _ = precision_recall_curve(y_true, other)
        ax.plot(r2, p2, lw=1.2, ls="--",
                label=f"{name} (AP {average_precision_score(y_true, other):.3f})")
    baseline = float(np.mean(y_true))
    ax.axhline(baseline, color="k", ls=":", lw=1,
               label=f"no skill ({baseline:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # calibration, binned by predicted probability decile. On the diagonal
    # the scores can be read as risks; far off it they're only good for
    # ranking.
    ax = axes[2]
    n_bins = 10
    edges = np.quantile(y_score, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) > 2:
        idx = np.clip(np.digitize(y_score, edges[1:-1]), 0, len(edges) - 2)
        pred_mean, obs_mean, counts = [], [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() >= 5:
                pred_mean.append(y_score[sel].mean())
                obs_mean.append(y_true[sel].mean())
                counts.append(int(sel.sum()))
        ax.plot(pred_mean, obs_mean, "o-", lw=1.5, ms=5, label="observed")
        for px, py, c in zip(pred_mean, obs_mean, counts):
            ax.annotate(str(c), (px, py), fontsize=6,
                        textcoords="offset points", xytext=(3, 4))
    ax.plot([0, max(y_score.max(), 0.01)], [0, max(y_score.max(), 0.01)],
            "k:", lw=1, label="perfect")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed event rate")
    ax.set_title(f"Calibration (Brier {brier_score_loss(y_true, y_score):.4f})")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # score distribution
    ax = axes[3]
    bins = np.linspace(y_score.min(), y_score.max(), 30)
    ax.hist(y_score[y_true == 0], bins=bins, alpha=0.6, density=True,
            label=f"no event (n={int((y_true == 0).sum())})")
    ax.hist(y_score[y_true == 1], bins=bins, alpha=0.6, density=True,
            label=f"event (n={int((y_true == 1).sum())})")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("density")
    ax.set_title("Score separation")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_specification_comparison(results_by_spec, y_true, title, path):
    """
    ROC and PR curves for each nested specification on one pair of axes.

    Shows whether adding the oximetry biomarkers moves the curves beyond the
    clinical confounders alone, which is the main thesis question.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for spec, scores in results_by_spec.items():
        y_spec, s_spec = scores
        fpr, tpr, _ = roc_curve(y_spec, s_spec)
        axes[0].plot(fpr, tpr, lw=1.8,
                     label=f"{spec} ({roc_auc_score(y_spec, s_spec):.3f})")
        precision, recall, _ = precision_recall_curve(y_spec, s_spec)
        axes[1].plot(recall, precision, lw=1.8,
                     label=f"{spec} "
                           f"({average_precision_score(y_spec, s_spec):.3f})")
    axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC by specification")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.2)

    axes[1].axhline(float(np.mean(y_true)), color="k", ls=":", lw=1,
                    label="no skill")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall by specification")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
