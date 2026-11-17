"""
Fixed horizon risk tools for the clinical logistic regression.

The Aim 2 models predict whether someone had an event at any point in follow up.
Follow up length differs between people, so that isn't a risk over a set period.
This module has the pieces for the proper question: the probability of a first
CVD event within t years.

Three things can happen before a horizon t:

    event of interest   label 1
    death from another cause, before t
                        label 0. They can't have the event any more, so they
                        stay in as a non-event. This gives the cumulative
                        incidence, i.e. the real world risk with other deaths
                        allowed for.
    censored alive, before t
                        outcome at t unknown, so weight 0. Their weight passes to
                        similar people through inverse probability of censoring
                        weighting (IPCW).

Weights are 1 / G(T- | x) for anyone with an event of either kind by t, and
1 / G(t | x) for anyone still under observation at t, where G is the probability
of not yet being censored. Nothing here touches files, so it can all be unit
tested on simulated data with a known answer.
"""

import numpy as np
import statsmodels.api as sm

CENSORED, CVD, OTHER_DEATH = 0, 1, 2
DAYS_PER_YEAR = 365.25
HORIZONS_YEARS = (1, 5, 10)


def horizon_days(years):
    return years * DAYS_PER_YEAR


def build_time_and_cause(y, first_event_day, censor_day, vital):
    """
    Time in days and cause of whichever comes first: event, other death or end of follow up.

    y is the event flag, normally the Aim 2 any_cvd label so this uses the same outcome.
    vital is 0 for dead and 1 for alive. A death with y = 0 is a death without a recorded
    event, which is close to a non-CVD death but not exactly the same thing.
    """
    y = np.asarray(y).astype(bool)
    time = np.where(y, first_event_day, censor_day).astype(float)
    cause = np.where(y, CVD, np.where(np.asarray(vital) == 0, OTHER_DEATH, CENSORED))
    if not np.isfinite(time).all():
        raise ValueError("every participant needs a finite time")
    return time, cause.astype(int)


# --- censoring models ---

def _step(times, values, query, left):
    """Right continuous step function (or its left limit), equal to 1 before the first step."""
    index = np.searchsorted(times, query, side="left" if left else "right") - 1
    return np.where(index < 0, 1.0, values[np.maximum(index, 0)])


class KMCensoring:
    """Kaplan Meier of the censoring time, optionally within strata."""

    def __init__(self, time, cause, strata=None):
        time = np.asarray(time, float)
        is_censored = np.asarray(cause) == CENSORED
        strata = np.zeros(len(time), int) if strata is None else np.asarray(strata)
        self.curves = {}
        for level in np.unique(strata):
            keep = strata == level
            self.curves[level] = self._curve(time[keep], is_censored[keep])

    @staticmethod
    def _curve(time, is_censored):
        unique = np.unique(time)
        at_risk = len(time) - np.searchsorted(np.sort(time), unique, side="left")
        index = np.searchsorted(unique, time)
        censored = np.bincount(index[is_censored], minlength=len(unique))
        events = np.bincount(index[~is_censored], minlength=len(unique))
        # an event on the same day as a censoring counts as first, so that person
        # is no longer at risk of being censored that day
        risk = at_risk - events
        factor = np.where(risk > 0, 1.0 - censored / np.maximum(risk, 1), 1.0)
        return unique, np.cumprod(factor)

    def G(self, time, strata=None, left=False):
        time = np.asarray(time, float)
        strata = np.zeros(len(time), int) if strata is None else np.asarray(strata)
        out = np.ones(len(time))
        for level, (unique, survival) in self.curves.items():
            keep = strata == level
            out[keep] = _step(unique, survival, time[keep], left)
        return out


class CoxCensoring:
    """
    Cox model for censoring time on the baseline covariates, capped at max_time.

    The cap keeps the model to censoring that can change a weight at the horizons.
    Most censoring after 10 years is just the end of follow up, which would otherwise
    dominate the fit and flip the age coefficient. Censoring before 10 years isn't
    all early loss either: most of it falls at 8 to 10 years and in one group of
    parent cohorts whose follow up stops sooner, so age here partly stands in for
    cohort. People with no follow up at all tell us nothing about the rate of later
    censoring, so they're left out of the fit. They still get weight 0.
    """

    def __init__(self, time, cause, covariates, max_time):
        time = np.asarray(time, float)
        covariates = np.asarray(covariates, float)
        ok = time > 0
        capped = np.minimum(time[ok], max_time)
        status = ((np.asarray(cause)[ok] == CENSORED) & (time[ok] <= max_time)).astype(int)
        model = sm.PHReg(capped, covariates[ok], status=status, ties="breslow")
        # undamped Newton can run away when censorings are few (e.g. the group holding most of them is left out)
        # and it returns without raising, so check the score at the solution
        def converged(params):
            return np.all(np.isfinite(params)) and np.abs(model.score(params)).max() / len(capped) < 1e-3
        fit = model.fit(disp=0)
        if not converged(fit.params):
            fit = model.fit(disp=0, method="bfgs", maxiter=2000)
        if not converged(fit.params):
            raise ValueError("censoring model did not converge")
        self.beta = np.asarray(fit.params)
        self.times, self.hazard = breslow_baseline(capped, status, covariates[ok], self.beta)

    def G(self, time, covariates, left=False):
        time = np.asarray(time, float)
        relative = np.exp(np.asarray(covariates, float) @ self.beta)
        index = np.searchsorted(self.times, time, side="left" if left else "right") - 1
        cumulative = np.where(index < 0, 0.0, self.hazard[np.maximum(index, 0)])
        return np.exp(-cumulative * relative)


def breslow_baseline(time, status, covariates, beta):
    """Breslow cumulative baseline hazard at the unique event times."""
    time = np.asarray(time, float)
    status = np.asarray(status).astype(bool)
    relative = np.exp(np.asarray(covariates, float) @ beta)
    order = np.argsort(time)
    sorted_time, sorted_relative = time[order], relative[order]
    # sum of relative hazards over everyone still at risk, at each sorted position
    at_risk = np.cumsum(sorted_relative[::-1])[::-1]
    unique = np.unique(time[status])
    positions = np.searchsorted(sorted_time, unique, side="left")
    counts = np.bincount(np.searchsorted(unique, time[status]), minlength=len(unique))
    return unique, np.cumsum(counts / at_risk[positions])


def ipcw_labels(time, cause, horizon, censoring, x, event_of_interest=CVD):
    """
    Label and IPCW weight at a horizon. x is whatever the censoring model conditions
    on (strata for KMCensoring, covariates for CoxCensoring).
    """
    time = np.asarray(time, float)
    cause = np.asarray(cause)
    label = ((cause == event_of_interest) & (time <= horizon)).astype(int)
    known_before = (cause != CENSORED) & (time <= horizon)
    still_observed = time > horizon
    weight = np.zeros(len(time))
    g_before = censoring.G(time, x, left=True)
    g_at = censoring.G(np.full(len(time), float(horizon)), x, left=False)
    weight[known_before] = 1.0 / g_before[known_before]
    weight[still_observed] = 1.0 / g_at[still_observed]
    # callers keep rows with weight > 0 and NaN > 0 is False, so a bad weight would quietly drop someone
    if not np.all(np.isfinite(weight)):
        raise ValueError("non-finite censoring weight")
    return label, weight


# --- cause specific Cox (comparator that gives risks at every horizon from one model) ---

class CauseSpecificCox:
    """
    Two Cox models, one per cause, combined into a cumulative incidence.

    Risk at t is the sum over event times u <= t of S(u-) dH1(u), where S is the
    probability of having had neither event and H1 is the hazard for the event of
    interest. So risks from this model can't drop as the horizon gets longer.
    """

    def __init__(self, time, cause, covariates):
        time = np.asarray(time, float)
        covariates = np.asarray(covariates, float)
        ok = time > 0
        self.fits = {}
        for code in (CVD, OTHER_DEATH):
            status = (np.asarray(cause)[ok] == code).astype(int)
            beta = np.asarray(sm.PHReg(time[ok], covariates[ok], status=status, ties="breslow").fit(disp=0).params)
            self.fits[code] = (beta,) + breslow_baseline(time[ok], status, covariates[ok], beta)

    def risk(self, covariates, horizons):
        covariates = np.asarray(covariates, float)
        grid = np.unique(np.concatenate([self.fits[CVD][1], self.fits[OTHER_DEATH][1]]))
        grid = grid[grid <= max(horizons)]
        increments, cumulative = {}, {}
        for code, (beta, times, hazard) in self.fits.items():
            index = np.searchsorted(times, grid, side="right") - 1
            baseline = np.where(index < 0, 0.0, hazard[np.maximum(index, 0)])
            relative = np.exp(covariates @ beta)[:, None]
            cumulative[code] = baseline[None, :] * relative
            increments[code] = np.diff(np.concatenate([[0.0], baseline]))[None, :] * relative
        total = cumulative[CVD] + cumulative[OTHER_DEATH]
        survival_before = np.exp(-np.hstack([np.zeros((len(covariates), 1)), total[:, :-1]]))
        incidence = np.cumsum(survival_before * increments[CVD], axis=1)
        positions = np.searchsorted(grid, horizons, side="right") - 1
        return {h: incidence[:, p] for h, p in zip(horizons, positions)}


# --- metrics (all IPCW weighted, so weight 0 means not counted) ---

def weighted_auc(label, score, weight):
    """Time dependent AUC: cases by t against everyone known to be event free at t."""
    _, inverse = np.unique(score, return_inverse=True)
    cases = np.bincount(inverse, weight * (label == 1))
    controls = np.bincount(inverse, weight * (label == 0))
    below = np.cumsum(controls) - controls
    return float((cases * (below + 0.5 * controls)).sum() / (cases.sum() * controls.sum()))


def weighted_incidence(label, weight):
    return float((weight * label).sum() / weight.sum())


def weighted_brier(label, risk, weight):
    return float((weight * (label - risk) ** 2).sum() / weight.sum())


def weighted_logloss(label, risk, weight):
    risk = np.clip(risk, 1e-12, 1 - 1e-12)
    return float(-(weight * (label * np.log(risk) + (1 - label) * np.log(1 - risk))).sum() / weight.sum())


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _irls(design, label, weight, offset=None, iterations=30):
    offset = np.zeros(len(label)) if offset is None else offset
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        mu = 1.0 / (1.0 + np.exp(-(design @ beta + offset)))
        v = np.maximum(mu * (1 - mu), 1e-9) * weight
        step = np.linalg.solve((design * v[:, None]).T @ design + 1e-10 * np.eye(design.shape[1]),
                               design.T @ (weight * (label - mu)))
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def recalibrate(linear_predictor, label, weight):
    """Intercept and slope of a weighted logistic regression on an existing score."""
    design = np.column_stack([np.ones(len(linear_predictor)), linear_predictor])
    a, b = _irls(design, label, weight)
    return float(a), float(b)


def calibration(label, risk, weight):
    """Observed to expected ratio, calibration slope, and calibration in the large."""
    keep = weight > 0
    y, r, w = label[keep], risk[keep], weight[keep]
    lp = logit(r)
    slope_fit = _irls(np.column_stack([np.ones(len(lp)), lp]), y, w)
    in_large = _irls(np.ones((len(lp), 1)), y, w, offset=lp)
    return dict(
        observed=weighted_incidence(y, w), expected=float((w * r).sum() / w.sum()),
        oe_ratio=weighted_incidence(y, w) / float((w * r).sum() / w.sum()),
        slope=float(slope_fit[1]), in_the_large=float(in_large[0]),
    )


def classification(label, flagged, weight):
    """Weighted confusion matrix measures. flagged is a boolean decision per person."""
    flagged = np.asarray(flagged).astype(bool)
    tp = (weight * ((label == 1) & flagged)).sum()
    fp = (weight * ((label == 0) & flagged)).sum()
    fn = (weight * ((label == 1) & ~flagged)).sum()
    tn = (weight * ((label == 0) & ~flagged)).sum()
    total = tp + fp + fn + tn
    return dict(
        flagged_share=float((tp + fp) / total),
        sensitivity=float(tp / max(tp + fn, 1e-12)), specificity=float(tn / max(tn + fp, 1e-12)),
        ppv=float(tp / max(tp + fp, 1e-12)), npv=float(tn / max(tn + fn, 1e-12)),
        accuracy=float((tp + tn) / total),
    )


def net_benefit(label, risk, weight, threshold):
    """Net benefit of treating everyone at or above a risk threshold (treat all and treat none are the references)."""
    total = weight.sum()
    flagged = risk >= threshold
    tp = (weight * ((label == 1) & flagged)).sum() / total
    fp = (weight * ((label == 0) & flagged)).sum() / total
    return float(tp - fp * threshold / (1 - threshold))


def calibration_groups(label, risk, weight, n_groups=10):
    """Mean predicted and observed risk in equal sized groups of predicted risk."""
    keep = weight > 0
    y, r, w = label[keep], risk[keep], weight[keep]
    edges = np.quantile(r, np.linspace(0, 1, n_groups + 1))
    group = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_groups - 1)
    rows = []
    for g in range(n_groups):
        k = group == g
        if not k.any():
            continue
        rows.append(dict(group=g + 1, n=int(k.sum()), predicted=float((w[k] * r[k]).sum() / w[k].sum()),
                         observed=weighted_incidence(y[k], w[k]), events=int(y[k].sum())))
    return rows


def required_sample_size(n_parameters, r2_cs, target_shrinkage=0.9):
    """Riley et al. 2020, criterion for small overfitting: n = p / ((S - 1) ln(1 - R2 / S))."""
    return n_parameters / ((target_shrinkage - 1) * np.log(1 - r2_cs / target_shrinkage))


def riley_criteria(n_parameters, r2_apparent, n, incidence):
    """
    The three Riley et al. 2019 and 2020 sample size criteria for a binary outcome. The apparent R squared
    is optimistic, so it's shrunk first by the expected van Houwelingen and Le Cessie factor.
    """
    shrinkage = 1 + n_parameters / (n * np.log(1 - r2_apparent))
    r2_adjusted = shrinkage * r2_apparent
    null_loglik = n * (incidence * np.log(incidence) + (1 - incidence) * np.log(1 - incidence))
    r2_max = 1 - np.exp(2 * null_loglik / n)
    optimism_shrinkage = r2_adjusted / (r2_adjusted + 0.05 * r2_max)
    return dict(r2_adjusted=float(r2_adjusted), expected_shrinkage=float(shrinkage),
                n_small_overfitting=float(required_sample_size(n_parameters, r2_adjusted)),
                n_small_optimism=float(required_sample_size(n_parameters, r2_adjusted, optimism_shrinkage)),
                n_overall_risk=float((1.96 / 0.05) ** 2 * incidence * (1 - incidence)))


# --- descriptive estimates (no model needed) ---

def aalen_johansen(time, cause, horizons, event_of_interest=CVD):
    """Cumulative incidence of one cause, other causes as competing events, censoring handled through the risk sets."""
    time = np.asarray(time, float)
    cause = np.asarray(cause)
    unique = np.unique(time)
    position = np.searchsorted(unique, time)
    at_risk = len(time) - np.searchsorted(np.sort(time), unique, side="left")
    of_interest = np.bincount(position[cause == event_of_interest], minlength=len(unique))
    any_event = np.bincount(position[cause != CENSORED], minlength=len(unique))
    survival_before = np.concatenate([[1.0], np.cumprod(1 - any_event / at_risk)[:-1]])
    incidence = np.cumsum(survival_before * of_interest / at_risk)
    index = np.searchsorted(unique, np.asarray(horizons, float), side="right") - 1
    return np.where(index < 0, 0.0, incidence[np.maximum(index, 0)])


def reverse_km_quantiles(time, cause, quantiles=(0.25, 0.5, 0.75)):
    """Potential follow up: Kaplan Meier of time to censoring, with any event counted as censoring it."""
    unique, survival = KMCensoring._curve(np.asarray(time, float), np.asarray(cause) == CENSORED)
    return [float(unique[np.argmax(survival <= 1 - q)]) for q in quantiles]


# --- AUC without comparing people across folds or strata ---

def within_fold_auc(label, risk, weight, fold):
    """
    Mean AUC across test folds. Pooling risks from different folds compares people across models whose
    average risk moves opposite to the test fold's incidence, which drags a pooled AUC down when events
    are few. A fold with no cases or no controls (can happen in a bootstrap) is skipped.
    """
    values = []
    for f in np.unique(fold):
        k = (fold == f) & (weight > 0)
        if (weight[k] * (label[k] == 1)).sum() > 0 and (weight[k] * (label[k] == 0)).sum() > 0:
            values.append(weighted_auc(label[k], risk[k], weight[k]))
    return float(np.mean(values))


def stratified_auc(label, score, weight, strata):
    """AUC where case-control pairs are only compared within a stratum, e.g. an age band."""
    concordant, pairs = 0.0, 0.0
    for s in np.unique(strata):
        k = (strata == s) & (weight > 0)
        cases, controls = (weight[k] * (label[k] == 1)).sum(), (weight[k] * (label[k] == 0)).sum()
        if cases > 0 and controls > 0:
            concordant += weighted_auc(label[k], score[k], weight[k]) * cases * controls
            pairs += cases * controls
    return float(concordant / pairs)
