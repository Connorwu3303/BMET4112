"""
XGBoost helper class for Aim 2.

Kept out of train_xgboost.py because the searches run under joblib, and the
worker processes have to import a custom estimator class to unpickle it. A
class defined in a script's __main__ only gets there through cloudpickle's
by-value fallback, which is slower and breaks more easily.
"""

import numpy as np
from xgboost import XGBClassifier


class AutoWeightXGBClassifier(XGBClassifier):
    """
    XGBClassifier that sets scale_pos_weight from the labels it's fitted on.

    scale_pos_weight is XGBoost's class weighting, usually negatives divided
    by positives, and plays the same role as class_weight="balanced" in the
    logistic regression and random forest.

    Working out that ratio once on the whole dataset would use the labels of
    people who later end up in test folds. It's only the overall event rate,
    which stratified folds keep almost constant, so the leak would be tiny,
    but computing it from the training fold at fit time avoids it and follows
    the same leakage rules as ml_common.py.

    Cloning still works: scikit-learn clones before fitting, and the value is
    reset on every fit.
    """

    def fit(self, X, y, **kwargs):
        y = np.asarray(y)
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))
        self.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
        return super().fit(X, y, **kwargs)
