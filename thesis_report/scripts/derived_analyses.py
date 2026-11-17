"""Extra analyses added while writing the final report. Only reads from the drive and writes to thesis_report/derived/.

A. Sleep structure logistic regression refit on the repeat 0 folds, so it's repeat matched with the deep score.
B. How sensitive the equivalence verdict is to the missing between-repeat term.
C. Time to event and fixed horizon sensitivity, using the saved out-of-fold scores.
E. Which part of the deep score carries the discrimination: the part the biomarkers reproduce, or the rest.
No network is trained.

Outputs: derived_results.json, structure_oof_repeat0.npy

Usage:
    python3 thesis_report/scripts/derived_analyses.py
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "aim3"), os.path.join(ROOT, "aim2")]
OUT = os.path.join(ROOT, "thesis_report", "derived"); os.makedirs(OUT, exist_ok=True)
R3 = "/Volumes/ThesisHD/Data/processed/aim3/results/"
DOC = "/Volumes/ThesisHD/Data/documentation/"
from folds import get_verified_folds
import data as D
from ml_common import delong_test
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from stack_eval import held_out_coefficient_test, paired_bootstrap_delta_auc
import statsmodels.api as sm

res = {}
data = get_verified_folds(quiet=True)
y, pid, folds = data["y"], data["pid"], data["folds"]
y = np.asarray(y, float)
print("folds", len(folds), list(folds[0].keys()))
oof = np.load(R3 + "oof_A.npz", allow_pickle=True)
deep = oof["deep_marg"][0]; deep_inc = oof["deep_inc"][0]; clin_logit = oof["clinical_logit"][0]; clin = oof["clinical"][0]
a2 = np.load(R3 + "aim2_oof_per_repeat.npz", allow_pickle=True)

# A. sleep structure LR, all repeats and repeat 0 only
st = pd.read_csv(R3 + "structure_set.csv").set_index("pid").loc[[int(p) for p in pid]]
x = st[list(D.STRUCTURE_COLUMNS)].to_numpy(float)
def nuis(rep_filter):
    # out-of-fold structure LR scores (rep_filter picks one repeat, None runs all of them and averages as it goes)
    o = np.full(len(y), np.nan)
    for f in folds:
        if rep_filter is not None and f["repeat"] != rep_filter: continue
        tr, te = f["train_idx"], f["test_idx"]
        tf = D.fit_confounder_transform(x[tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(D.apply_confounder_transform(x[tr], tf), y[tr])
        p = lr.predict_proba(D.apply_confounder_transform(x[te], tf))[:, 1]
        o[te] = p if np.all(np.isnan(o[te])) else (o[te] + p) / 2.0
    return o
all_rep = nuis(None); rep0 = nuis(0)
logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
dl = delong_test(y, deep, rep0)
print("delong keys", dl if not isinstance(dl, dict) else list(dl.keys()))
res["A_structure"] = dict(
    auc_saved_weighting_reproduced=float(roc_auc_score(y, all_rep)),
    auc_repeat0=float(roc_auc_score(y, rep0)),
    auc_deep=float(roc_auc_score(y, deep)),
    delong_deep_vs_structure_repeat0={k: (float(v) if np.isscalar(v) else v) for k, v in dl.items()} if isinstance(dl, dict) else list(map(float, dl)),
    spearman_deep_structure=float(stats.spearmanr(deep, rep0)[0]),
    increment_over_clinical=held_out_coefficient_test(y, clin_logit, logit(rep0)))
np.save(os.path.join(OUT, "structure_oof_repeat0.npy"), rep0)

# B. how big the between-repeat SD would have to be before the bound on delta (1.645 x SE) reaches the 0.005 margin,
# plus the Aim 2 LR delta in each repeat to compare against
se0 = None
ev = json.load(open(R3 + "stack_eval_A.json"))
delta = ev["c2_delta_auc"]["delta"]; seb = ev["c2_delta_auc"]["se_boot"]
se_needed = (0.005 - abs(delta)) / 1.645
per = [roc_auc_score(y, a2["obm_plus_confounders__per_repeat"][r]) - roc_auc_score(y, a2["confounders_only__per_repeat"][r]) for r in range(3)]
per_auc = [roc_auc_score(y, a2["confounders_only__per_repeat"][r]) for r in range(3)]
res["B_equivalence"] = dict(delta=delta, se_boot=seb, se_total_needed_to_reach_margin=se_needed,
    between_repeat_sd_needed=float(np.sqrt(max(se_needed**2 - seb**2, 0))),
    aim2_lr_delta_per_repeat=[float(v) for v in per], aim2_lr_delta_between_repeat_sd=float(np.std(per, ddof=1)),
    aim2_lr_clinical_auc_per_repeat=[float(v) for v in per_auc], aim2_lr_clinical_auc_between_repeat_sd=float(np.std(per_auc, ddof=1)))

# C. survival framing. Time in years: first component date for events (censdate if there's no date), censdate for everyone else
cv = pd.read_csv(DOC + "shhs-cvd-summary-dataset-0.15.0.csv").set_index("nsrrid").loc[[int(p) for p in pid]]
dates = ["mi_date", "mip_date", "chd_dthdt", "cvd_dthdt", "revpro_date", "ptca_date", "cabg_date", "chf_date", "stk_date"]
first = cv[dates].min(axis=1).to_numpy(float); cens = cv["censdate"].to_numpy(float)
T = np.where(y == 1, np.where(np.isfinite(first), first, cens), cens) / 365.25
ok = T > 0
print("n with T>0", ok.sum(), "events", int(y[ok].sum()), "events without date", int(((y==1)&~np.isfinite(first)).sum()))
ds = pd.read_csv("/Volumes/ThesisHD/Data/processed/aim2/modelling_dataset.csv")
ds = ds.set_index("pid" if "pid" in ds.columns else "nsrrid").loc[[int(p) for p in pid]]
man = json.load(open("/Volumes/ThesisHD/Data/processed/aim2/feature_manifest.json"))
print("manifest keys", list(man.keys()))
conf_cols = man.get("confounders") or man.get("confounder_features") or man.get("confounder_columns")
panel = man.get("core_obm_panel")
Xc = SimpleImputer(strategy="median").fit_transform(ds[conf_cols].to_numpy(float))
Xc = StandardScaler().fit_transform(Xc)
def z(v):
    v = np.asarray(v, float); v = np.where(np.isfinite(v), v, np.nanmedian(v)); return (v - v.mean()) / v.std()
def cox(exog, names):
    m = sm.PHReg(T[ok], exog[ok], status=y[ok].astype(int), ties="efron").fit(disp=0)
    return m
def cox_row(name, pred):
    # HR per SD: unadjusted, age and sex, all 18 clinical, plus a likelihood ratio test for adding it to the 18
    zp = z(pred)
    out = {}
    un = cox(zp[:, None], [name])
    agesex = np.column_stack([zp, Xc[:, conf_cols.index("age_s1")], Xc[:, conf_cols.index("is_male")]])
    as_ = cox(agesex, [name])
    full = cox(np.column_stack([zp, Xc]), [name] + conf_cols)
    base = cox(Xc, conf_cols)
    lr = 2 * (full.llf - base.llf)
    for lab, m in (("unadjusted", un), ("age_sex", as_), ("all_18", full)):
        b, s = float(m.params[0]), float(m.bse[0])
        out[lab] = dict(hr_per_sd=float(np.exp(b)), lo=float(np.exp(b - 1.96 * s)), hi=float(np.exp(b + 1.96 * s)), p=float(2 * stats.norm.sf(abs(b / s))))
    out["lr_test_add_to_18"] = dict(chi2=float(lr), p=float(stats.chi2.sf(lr, 1)))
    return out
preds = {"sleep_ct90_pct": ds["sleep_ct90_pct"], "sleep_mean": ds["sleep_mean"], "sleep_odi3": ds["sleep_odi3"], "sleep_odi4": ds["sleep_odi4"], "sleep_min": ds["sleep_min"], "obm_oof_score": a2["obm_only__per_repeat"][0], "deep_score": deep}
res["C_cox_per_sd"] = {k: cox_row(k, np.asarray(v, float)) for k, v in preds.items()}
# T90 groups. T90 is exactly 0 in 1,180 participants, so splitting by rank would depend on row order (an earlier version of this code did that).
# Reference group is T90 of 0, top group is the top fifth, and everything in between is cut into three groups by value.
t90 = ds["sleep_ct90_pct"].to_numpy(float); t90f = np.where(np.isfinite(t90), t90, np.nanmedian(t90))
q = np.full(len(t90f), 4); q[t90f == 0] = 0
_mid = (t90f != 0) & (t90f < np.quantile(t90f, 0.8))
q[_mid] = 1 + pd.qcut(t90f[_mid], 3, labels=False)
dm = np.column_stack([(q == i).astype(float) for i in range(1, 5)])
mq_un = cox(dm, []); mq_as = cox(np.column_stack([dm, Xc[:, conf_cols.index("age_s1")], Xc[:, conf_cols.index("is_male")]]), []); mq_full = cox(np.column_stack([dm, Xc]), [])
def qr(m): 
    b, s = float(m.params[3]), float(m.bse[3]); return dict(hr_top_vs_zero=float(np.exp(b)), lo=float(np.exp(b - 1.96 * s)), hi=float(np.exp(b + 1.96 * s)), p=float(2 * stats.norm.sf(abs(b / s))))
res["C_t90_groups"] = dict(top_cut=float(np.quantile(t90f, 0.8)), group_sizes=[int((q == i).sum()) for i in range(5)], unadjusted=qr(mq_un), age_sex=qr(mq_as), all_18=qr(mq_full),
    event_rate_by_group=[float(y[q == i].mean()) for i in range(5)])
# Harrell's C using the event times
def harrell(score, t=T, e=y, mask=ok):
    s, tt, ee = score[mask], t[mask], e[mask]
    idx = np.argsort(tt); s, tt, ee = s[idx], tt[idx], ee[idx]
    conc = disc = tied = 0.0
    for i in np.where(ee == 1)[0]:
        later = tt > tt[i]
        d = s[i] - s[later]
        conc += (d > 0).sum(); disc += (d < 0).sum(); tied += (d == 0).sum()
    return float((conc + 0.5 * tied) / (conc + disc + tied))
r0 = lambda k: a2[k + "__per_repeat"][0]
res["C_harrell"] = {"clinical_repeat0": harrell(r0("confounders_only")), "obm_only_repeat0": harrell(r0("obm_only")), "combined_repeat0": harrell(r0("obm_plus_confounders")), "deep": harrell(deep), "clinical_plus_deep_stacked": harrell(oof["stacked"][0]), "clinical_stack_comparator": harrell(clin)}
# fixed 10 year horizon, dropping anyone censored or dead before 10 years without an event
H = 10.0
known = ((y == 1) & (T <= H)) | (T >= H)
yh = ((y == 1) & (T <= H)).astype(float)
res["C_10y"] = dict(n=int(known.sum()), events=int(yh[known].sum()), excluded_censored_or_died_before_10y=int((~known).sum()),
    auc={k: float(roc_auc_score(yh[known], s[known])) for k, s in dict(clinical=r0("confounders_only"), obm=r0("obm_only"), combined=r0("obm_plus_confounders"), deep=deep, stacked=oof["stacked"][0], clinical_comparator=clin).items()})
dl10 = delong_test(yh[known], oof["stacked"][0][known], clin[known]); res["C_10y"]["delong_stacked_vs_clinical"] = {k: (float(v) if np.isscalar(v) else v) for k, v in dl10.items()} if isinstance(dl10, dict) else list(map(float, dl10))
# non-CVD death as a competing event: AUC with those deaths left out
died_other = (y == 0) & (cv["vital"].to_numpy() == 0)
res["C_exclude_noncv_deaths"] = dict(n_excluded=int(died_other.sum()), auc={k: float(roc_auc_score(y[~died_other], s[~died_other])) for k, s in dict(clinical=clin, obm=r0("obm_only"), deep=deep, stacked=oof["stacked"][0]).items()})

# E. ridge regression of the deep score on the OBM panel (repeat 0 folds), then compare
# the part the panel reproduces with the residual
Xo = ds[panel].to_numpy(float); zdeep = z(deep)
pred = np.full(len(y), np.nan)
for f in folds:
    if f["repeat"] != 0: continue
    tr, te = f["train_idx"], f["test_idx"]
    imp = SimpleImputer(strategy="median").fit(Xo[tr]); sc = StandardScaler().fit(imp.transform(Xo[tr]))
    m = RidgeCV(alphas=np.logspace(-3, 4, 25)).fit(sc.transform(imp.transform(Xo[tr])), zdeep[tr])
    pred[te] = m.predict(sc.transform(imp.transform(Xo[te])))
resid = zdeep - pred
res["E_reproduced_part"] = dict(r2=float(1 - ((zdeep - pred) ** 2).sum() / ((zdeep - zdeep.mean()) ** 2).sum()),
    auc_deep=float(roc_auc_score(y, deep)), auc_ridge_reproduced=float(roc_auc_score(y, pred)), auc_residual=float(roc_auc_score(y, resid)),
    delong_deep_vs_reproduced={k: (float(v) if np.isscalar(v) else v) for k, v in delong_test(y, deep, pred).items()} if isinstance(delong_test(y, deep, pred), dict) else None,
    spearman_residual_with_age=float(stats.spearmanr(resid, ds["age_s1"], nan_policy="omit")[0]),
    spearman_residual_with_structure={c: float(stats.spearmanr(resid, st[c])[0]) for c in D.STRUCTURE_COLUMNS},
    spearman_residual_with_clinical_logit=float(stats.spearmanr(resid, clin_logit)[0]))
json.dump(res, open(os.path.join(OUT, "derived_results.json"), "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
