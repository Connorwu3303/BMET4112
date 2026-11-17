"""
Tests for horizon_lib (plus a couple of fit_horizon_models functions). Run before reading any results.

Most use simulated cohorts where the true risk is known, including censoring that
depends on age, which is what happens in the SHHS data.

Usage: /usr/bin/python3 horizon_risk/test_horizon_lib.py
"""

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.duration.survfunc import CumIncidenceRight, SurvfuncRight

import horizon_lib as H

warnings.simplefilter("ignore")
PASSED = []


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED.append(name)
    print(f"  ok  {name}  {detail}")


def simulate(n, seed, censor_depends_on_x=True, censoring=True):
    """Exponential event and other death times with hazards that depend on x, plus optional censoring."""
    rng = np.random.RandomState(seed)
    x = rng.normal(size=n)
    event = rng.exponential(1.0 / (0.02 * np.exp(0.8 * x)))
    death = rng.exponential(1.0 / (0.01 * np.exp(0.5 * x)))
    if censoring:
        rate = 0.04 * np.exp(-1.0 * x) if censor_depends_on_x else np.full(n, 0.04)
        censor = rng.exponential(1.0 / rate)
    else:
        censor = np.full(n, 1e9)
    time = np.minimum.reduce([event, death, censor])
    cause = np.where(time == event, H.CVD, np.where(time == death, H.OTHER_DEATH, H.CENSORED))
    return x, time, cause, event, death


def true_risk(x, t):
    """Closed form cumulative incidence for constant hazards."""
    h1, h2 = 0.02 * np.exp(0.8 * x), 0.01 * np.exp(0.5 * x)
    return h1 / (h1 + h2) * (1 - np.exp(-(h1 + h2) * t))


def test_km_matches_statsmodels():
    _, time, cause, _, _ = simulate(3000, 1, censor_depends_on_x=False)
    time = np.round(time, 1) + 0.1
    km = H.KMCensoring(time, cause)
    ref = SurvfuncRight(time, (cause == H.CENSORED).astype(int))
    # statsmodels orders ties the same way (the status 1 "event" comes first), so the censoring KM only
    # differs from it in tie handling, which should be tiny with this few ties
    query = np.array([2.0, 5.0, 10.0, 20.0])
    ours = km.G(query)
    theirs = np.array([ref.surv_prob[np.searchsorted(ref.surv_times, q, side="right") - 1] for q in query])
    check("KM censoring matches statsmodels", np.abs(ours - theirs).max() < 0.01, f"max diff {np.abs(ours - theirs).max():.5f}")


def test_step_left_limit():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    cause = np.array([H.CENSORED, H.CVD, H.CENSORED, H.CVD])
    km = H.KMCensoring(time, cause)
    check("G before first censoring is 1", km.G(np.array([0.5]))[0] == 1.0)
    check("right limit includes the step", abs(km.G(np.array([1.0]))[0] - 0.75) < 1e-12)
    check("left limit excludes the step", km.G(np.array([1.0]), left=True)[0] == 1.0)
    # after t=1: 3 at risk; t=2 is an event; t=3 has 2 at risk, one censored
    check("product limit value", abs(km.G(np.array([3.0]))[0] - 0.75 * 0.5) < 1e-12)


def test_weighted_auc_equals_sklearn_when_unweighted():
    rng = np.random.RandomState(2)
    label = rng.binomial(1, 0.2, 2000)
    score = np.round(rng.normal(size=2000) + label, 1)
    a, b = H.weighted_auc(label, score, np.ones(2000)), roc_auc_score(label, score)
    check("weighted AUC with unit weights equals sklearn, with ties", abs(a - b) < 1e-12, f"{a:.6f} vs {b:.6f}")
    w = rng.randint(1, 4, 2000)
    rep = np.repeat(np.arange(2000), w)
    a, b = H.weighted_auc(label, score, w.astype(float)), roc_auc_score(label[rep], score[rep])
    check("integer weights equal repeating the rows", abs(a - b) < 1e-12)


def test_incidence_matches_aalen_johansen():
    _, time, cause, _, _ = simulate(6000, 3, censor_depends_on_x=False)
    t = 10.0
    km = H.KMCensoring(time, cause)
    label, w = H.ipcw_labels(time, cause, t, km, None)
    aj = CumIncidenceRight(time, cause)
    index = np.searchsorted(aj.times, t, side="right") - 1
    theirs = aj.cinc[0][index]
    ours = (w * label).sum() / len(time)
    check("IPCW incidence with KM weights equals Aalen Johansen", abs(ours - theirs) < 1e-3, f"{ours:.5f} vs {theirs:.5f}")


def test_weights_exclude_censored_before_horizon():
    time = np.array([1.0, 2.0, 3.0, 8.0, 12.0, 0.0])
    cause = np.array([H.CVD, H.CENSORED, H.OTHER_DEATH, H.CVD, H.CENSORED, H.CENSORED])
    label, w = H.ipcw_labels(time, cause, 5.0, H.KMCensoring(time, cause), None)
    check("censored before t has weight 0", w[1] == 0)
    check("zero follow up has weight 0", w[5] == 0)
    check("other death before t is kept as a non event", w[2] > 0 and label[2] == 0)
    check("event before t is a case", w[0] > 0 and label[0] == 1)
    check("event after t is a non event", w[3] > 0 and label[3] == 0)


def test_ipcw_lr_recovers_true_risk_under_dependent_censoring():
    n, t = 40000, 10.0
    x, time, cause, _, _ = simulate(n, 4)
    truth = true_risk(x, t)
    z = x[:, None]
    full_incidence = truth.mean()
    # a logistic model only approximates this truth, so the checks are on averages and rank
    cox = H.CoxCensoring(time, cause, z, max_time=t + 1)
    label, w = H.ipcw_labels(time, cause, t, cox, z)
    keep = w > 0
    fit = LogisticRegression(C=1e6, max_iter=2000).fit(z[keep], label[keep], sample_weight=w[keep])
    risk = fit.predict_proba(z)[:, 1]
    # the unweighted complete case fit is what IPCW replaces, so it should be visibly worse
    naive = LogisticRegression(C=1e6, max_iter=2000).fit(z[keep], label[keep]).predict_proba(z)[:, 1]
    weighted_error, naive_error = abs(risk.mean() - full_incidence), abs(naive.mean() - full_incidence)
    check("IPCW recovers mean risk under dependent censoring", weighted_error < 0.005, f"true {full_incidence:.4f}, ipcw {risk.mean():.4f}, naive {naive.mean():.4f}")
    check("naive complete case is worse than IPCW", naive_error > 2 * weighted_error, f"{naive_error:.4f} vs {weighted_error:.4f}")
    check("marginal KM is worse than the Cox censoring model", True if abs(
        (lambda lw: (lw[0] * lw[1]).sum() / lw[1].sum())(H.ipcw_labels(time, cause, t, H.KMCensoring(time, cause), None)) - full_incidence
    ) > abs((w * label).sum() / w.sum() - full_incidence) else False)
    check("individual risk tracks truth", np.mean(np.abs(risk - truth)) < 0.02, f"mean absolute error {np.mean(np.abs(risk - truth)):.4f}")


def test_auc_under_censoring_matches_uncensored_truth():
    n, t = 40000, 10.0
    x, time, cause, event, death = simulate(n, 5)
    # uncensored truth from the same people
    first = np.minimum(event, death)
    true_label = ((event <= death) & (event <= t)).astype(int)
    true_auc = roc_auc_score(true_label, x)
    z = x[:, None]
    label, w = H.ipcw_labels(time, cause, t, H.CoxCensoring(time, cause, z, max_time=t + 1), z)
    ours = H.weighted_auc(label, x, w)
    plain = roc_auc_score(label[w > 0], x[w > 0])
    check("IPCW AUC matches the uncensored AUC", abs(ours - true_auc) < 0.01, f"truth {true_auc:.4f}, ipcw {ours:.4f}, complete case {plain:.4f}")
    brier_truth = np.mean((true_label - true_risk(x, t)) ** 2)
    brier = H.weighted_brier(label, true_risk(x, t), w)
    check("IPCW Brier matches the uncensored Brier", abs(brier - brier_truth) < 0.005, f"{brier_truth:.4f} vs {brier:.4f}")


def test_cause_specific_cox_recovers_true_risk():
    x, time, cause, _, _ = simulate(30000, 6)
    # censoring depends on x here, which Cox handles since x is in the model
    model = H.CauseSpecificCox(time, cause, x[:, None])
    probe = np.array([-1.0, 0.0, 1.0, 2.0])
    risks = model.risk(probe[:, None], [1.0, 5.0, 10.0])
    for t in (1.0, 5.0, 10.0):
        diff = np.abs(risks[t] - true_risk(probe, t)).max()
        check(f"cause specific Cox risk matches closed form at t={t:g}", diff < 0.015, f"max diff {diff:.4f}")
    check("Cox risks never fall with horizon", np.all(risks[1.0] <= risks[5.0]) and np.all(risks[5.0] <= risks[10.0]))


def test_calibration_of_a_true_model():
    rng = np.random.RandomState(7)
    risk = rng.beta(2, 10, 50000)
    label = rng.binomial(1, risk)
    c = H.calibration(label, risk, np.ones(50000))
    check("perfect model has O/E near 1", abs(c["oe_ratio"] - 1) < 0.02, f"{c['oe_ratio']:.3f}")
    check("perfect model has slope near 1", abs(c["slope"] - 1) < 0.05, f"{c['slope']:.3f}")
    check("perfect model has in the large near 0", abs(c["in_the_large"]) < 0.03, f"{c['in_the_large']:.3f}")
    over = np.clip(risk * 2, 0, 0.99)
    check("doubled risk gives O/E near 0.5", abs(H.calibration(label, over, np.ones(50000))["oe_ratio"] - 0.5) < 0.03)
    pushed = 1 / (1 + np.exp(-2 * H.logit(risk)))
    check("overconfident risk has slope near 0.5", abs(H.calibration(label, pushed, np.ones(50000))["slope"] - 0.5) < 0.05)


def test_recalibration_recovers_known_values():
    rng = np.random.RandomState(8)
    lp = rng.normal(size=200000)
    label = rng.binomial(1, H.expit(-2.0 + 0.6 * lp))
    a, b = H.recalibrate(lp, label, np.ones(len(lp)))
    check("recalibration recovers intercept and slope", abs(a + 2.0) < 0.02 and abs(b - 0.6) < 0.02, f"{a:.3f}, {b:.3f}")
    w = rng.randint(1, 4, len(lp))
    rep = np.repeat(np.arange(len(lp)), w)
    a1, b1 = H.recalibrate(lp, label, w.astype(float))
    a2, b2 = H.recalibrate(lp[rep], label[rep], np.ones(len(rep)))
    check("recalibration weights equal repeating the rows", abs(a1 - a2) < 1e-7 and abs(b1 - b2) < 1e-7)


def test_classification_and_net_benefit():
    label = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    risk = np.array([.9, .1, .8, .1, .1, .1, .1, .1, .1, .1])
    c = H.classification(label, risk >= 0.5, np.ones(10))
    check("sensitivity", c["sensitivity"] == 0.5)
    check("specificity", c["specificity"] == 7 / 8)
    check("ppv", c["ppv"] == 0.5)
    check("npv", c["npv"] == 7 / 8)
    check("accuracy", c["accuracy"] == 0.8)
    # one TP and one FP out of ten at threshold 0.5: 0.1 - 0.1 * 1 = 0
    check("net benefit", abs(H.net_benefit(label, risk, np.ones(10), 0.5)) < 1e-12)


def test_sample_size_formula():
    # Riley et al. 2020 worked example: 24 parameters and R2 of 0.288 gives about 623 participants at S = 0.9
    n = H.required_sample_size(24, 0.288)
    check("Riley sample size formula", abs(n - 623) < 3, f"{n:.1f}")


def test_time_and_cause_are_built_from_the_right_columns():
    # added because bugs planted in these paths weren't caught by any other test
    y = np.array([1, 0, 0])
    time, cause = H.build_time_and_cause(y, np.array([100.0, np.nan, np.nan]), np.array([900.0, 800.0, 700.0]), np.array([1, 0, 1]))
    check("an event takes the event date and not last contact", time[0] == 100)
    check("a non event takes last contact", time[1] == 800 and time[2] == 700)
    check("vital 0 is dead, vital 1 is alive", cause.tolist() == [H.CVD, H.OTHER_DEATH, H.CENSORED])


def test_weights_change_the_answer():
    # each weighted measure should equal the same measure on rows repeated by their integer weight
    rng = np.random.RandomState(7)
    n = 400
    label = rng.binomial(1, 0.3, n)
    risk = np.clip(0.3 + 0.4 * (label - 0.3) + rng.normal(0, 0.2, n), 0.01, 0.99)
    w = rng.randint(1, 5, n).astype(float)
    rep = np.repeat(np.arange(n), w.astype(int))
    one = np.ones(len(rep))
    flagged = risk >= 0.35
    a, b = H.classification(label, flagged, w), H.classification(label[rep], flagged[rep], one)
    check("classification uses the weights", all(abs(a[k] - b[k]) < 1e-12 for k in a) and a["ppv"] != H.classification(label, flagged, np.ones(n))["ppv"])
    a, b = H.calibration(label, risk, w), H.calibration(label[rep], risk[rep], one)
    check("calibration slope uses the weights", abs(a["slope"] - b["slope"]) < 1e-6 and abs(a["slope"] - H.calibration(label, risk, np.ones(n))["slope"]) > 1e-4)
    check("net benefit uses the weights", abs(H.net_benefit(label, risk, w, 0.2) - H.net_benefit(label[rep], risk[rep], one, 0.2)) < 1e-12)
    check("Brier uses the weights", abs(H.weighted_brier(label, risk, w) - H.weighted_brier(label[rep], risk[rep], one)) < 1e-12)
    check("log loss uses the weights", abs(H.weighted_logloss(label, risk, w) - H.weighted_logloss(label[rep], risk[rep], one)) < 1e-12)
    check("AUC uses the weights", abs(H.weighted_auc(label, risk, w) - roc_auc_score(label[rep], risk[rep])) < 1e-12)
    weighted_groups = [g["observed"] for g in H.calibration_groups(label, risk, w, 5)]
    plain_groups = [g["observed"] for g in H.calibration_groups(label, risk, np.ones(n), 5)]
    check("group observed risk uses the weights", weighted_groups != plain_groups)


def test_net_benefit_at_a_low_threshold():
    # two TP and one FP out of ten at threshold 0.2: 2/10 - (1/10) * (0.2 / 0.8) = 0.175. A threshold away from 0.5 catches inverted odds
    label = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    risk = np.array([.9, .3, .4, .1, .1, .1, .1, .1, .1, .1])
    check("net benefit odds at 20%", abs(H.net_benefit(label, risk, np.ones(10), 0.2) - 0.175) < 1e-12)


def test_log_loss_has_both_terms():
    check("log loss for one event and one non event", abs(H.weighted_logloss(np.array([1, 0]), np.array([0.8, 0.8]), np.ones(2)) - (-(np.log(0.8) + np.log(0.2)) / 2)) < 1e-12)


def test_aalen_johansen():
    _, time, cause, _, _ = simulate(4000, 11)
    days = np.array([5.0, 20.0])
    ours = H.aalen_johansen(time, cause, days)
    reference = CumIncidenceRight(time, cause)
    index = np.searchsorted(reference.times, days, side="right") - 1
    check("Aalen Johansen equals statsmodels", np.abs(ours - reference.cinc[H.CVD - 1][index]).max() < 1e-10, str(ours.round(4)))
    no_death = np.where(cause == H.OTHER_DEATH, H.CENSORED, cause)
    km = SurvfuncRight(time, (no_death == H.CVD).astype(int))
    check("with no competing event it is the complement of Kaplan Meier",
          np.abs(H.aalen_johansen(time, no_death, days) - (1 - km.surv_prob[np.searchsorted(km.surv_times, days, side="right") - 1])).max() < 1e-10)


def test_within_fold_auc():
    # each fold ranks perfectly but the folds sit on different scales, so pooled AUC is below 1 and within fold AUC is 1
    label = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    risk = np.array([.1, .2, .1, .2, .5, .6, .5, .6])
    fold = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    w = np.ones(8)
    check("within fold AUC ignores differences between folds", H.within_fold_auc(label, risk, w, fold) == 1.0 and H.weighted_auc(label, risk, w) < 1.0)
    check("a fold without cases is skipped", H.within_fold_auc(np.array([0, 0, 0, 1]), np.array([.1, .2, .1, .2]), np.ones(4), np.array([0, 0, 1, 1])) == 1.0)


def test_stratified_auc():
    label = np.array([0, 1, 0, 1, 0, 1])
    # inside each stratum the case outranks the control, across strata it doesn't
    score = np.array([.1, .2, .5, .6, .9, 1.0])
    strata = np.array([0, 0, 1, 1, 2, 2])
    check("stratified AUC only compares inside a stratum", H.stratified_auc(label, score, np.ones(6), strata) == 1.0 and H.weighted_auc(label, score, np.ones(6)) < 1.0)


def test_riley_criteria():
    n = H.required_sample_size(20, 0.1)
    check("Riley sample size at R2 0.1 and 20 parameters", abs(n - 1698.04) < 0.1, f"{n:.2f}")
    r = H.riley_criteria(18, 0.1434, 4116, 0.1682)
    check("the shrunken R squared is lower than the apparent one", r["r2_adjusted"] < 0.1434 and r["n_small_overfitting"] > H.required_sample_size(18, 0.1434))
    check("overall risk criterion", abs(r["n_overall_risk"] - (1.96 / 0.05) ** 2 * 0.1682 * (1 - 0.1682)) < 1e-9)


def test_censoring_model_survives_a_failed_newton_fit_and_bad_weights_raise():
    # a rare covariate with few censorings can send undamped Newton off to infinity. Weights have to stay finite or the fit has to refuse
    rng = np.random.RandomState(11)
    n = 3000
    x = np.column_stack([rng.normal(size=n), (rng.uniform(size=n) < 0.06).astype(float), rng.normal(size=n)])
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    time = rng.uniform(3000, 6000, n)
    censored = rng.uniform(size=n) < 0.02
    time[censored] = rng.uniform(100, 3000, censored.sum())
    cause = np.where(censored, H.CENSORED, H.CVD)
    model = H.CoxCensoring(time, cause, x, max_time=4000)
    _, weight = H.ipcw_labels(time, cause, 3000, model, x)
    check("censoring weights are finite", np.all(np.isfinite(weight)))
    check("censoring coefficients stay small", np.abs(model.beta).max() < 5, f"{np.abs(model.beta).max():.2f}")

    class Broken:
        def G(self, t, x, left=False):
            return np.full(len(t), np.nan)
    try:
        H.ipcw_labels(time, cause, 3000, Broken(), x)
        raised = False
    except ValueError:
        raised = True
    check("a non-finite weight raises instead of dropping a person", raised)


def test_count_consistent_rule_on_small_examples():
    import pandas as pd
    import fit_horizon_models as F
    nan = float("nan")
    cv = pd.DataFrame(dict(mi=[2, 0, 0, nan, 1], ptca=[0, 0, 3, nan, 0], cabg=[0, 0, 0, nan, 0], revasc_proc=[0, 0, 0, nan, 0], chf=[0, 0, 0, nan, 0],
                           chd_death=[0, 0, 0, nan, 0], cvd_death=[0, 0, 0, nan, 0]))
    check("two records count, none do not, missing does not", F.counts_as_event_under_corrected_rule(cv).tolist() == [True, False, True, False, True])


def test_test_rows_cannot_change_test_risks():
    # goes through the real fitting path on a simulated frame. Scrambling the test fold outcomes should change nothing
    import fit_horizon_models as F
    rng = np.random.RandomState(3)
    n = 500
    x = rng.normal(size=(n, 4))
    time = rng.exponential(2000 * np.exp(-0.6 * x[:, 0])) + 1
    cause = rng.choice([H.CVD, H.OTHER_DEATH, H.CENSORED], n, p=[0.3, 0.1, 0.6])
    train, test = np.arange(400), np.arange(400, 500)
    grid = F.penalty_grid(True)
    group = np.full(n, "G1")
    a = F.fit_fold("s", x, x, time, cause, 5, 0, 0, train, test, grid, "cox", True, group)
    scrambled_time, scrambled_cause = time.copy(), cause.copy()
    scrambled_time[test] = rng.permutation(time[test])
    scrambled_cause[test] = rng.permutation(cause[test])
    noisy = x.copy()
    noisy[test] = rng.normal(size=(100, 4))
    b = F.fit_fold("s", x, noisy, scrambled_time, scrambled_cause, 5, 0, 0, train, test, grid, "cox", True, group)
    check("test fold outcomes and censoring covariates do not touch test risks", np.abs(a["risk"] - b["risk"]).max() == 0.0)


if __name__ == "__main__":
    for name, function in sorted(globals().copy().items()):
        if name.startswith("test_"):
            print(name)
            function()
    print(f"\n{len(PASSED)} of {len(PASSED)} checks passed")
