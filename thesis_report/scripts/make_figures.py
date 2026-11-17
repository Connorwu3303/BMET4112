"""Draws the Chapter 4 figures from saved outputs on the drive.

Nothing is retrained and no participant id gets printed. PDFs go to thesis_report/latex/figures/.

Usage:
    python3 thesis_report/scripts/make_figures.py
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
import pyedflib

T = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
OUT = T + "latex/figures/"
P = "/Volumes/ThesisHD/Data/processed/"
R2, R3, R4 = P + "aim2/results/", P + "aim3/results/", P + "aim4/results/"
RAW = "/Volumes/ThesisHD/Data/"

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "axes.axisbelow": True,
    "lines.linewidth": 1.3, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 6.69  # text width in inches, 17 cm


def save(fig, name):
    fig.savefig(OUT + name + ".pdf", dpi=300)
    plt.close(fig)
    print("wrote", name)


def fig_cleaning():
    pid, pid2 = "200020", "202267"
    with pyedflib.EdfReader(RAW + f"shhs1-{pid}.edf") as f:
        i = [k for k, l in enumerate(f.getSignalLabels()) if l == "SaO2"][0]
        raw = f.readSignal(i)
    filled = np.load(P + f"cleaned_spo2/shhs1-{pid}.npy")
    clean = np.load(P + f"cleaned_spo2_v2/shhs1-{pid}.npy")
    mask = np.load(P + f"validity_mask/shhs1-{pid}.npy")
    ev = pd.read_csv(P + f"events/shhs1-{pid}_events.csv")
    art = ev[ev.event_concept.str.contains("artifact", case=False)]
    t = np.arange(len(raw)) / 3600
    fig = plt.figure(figsize=(W, 3.5))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.45, 1], hspace=0.28, wspace=0.2, height_ratios=[1, 1, 1])
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[1, 0], sharex=a); c = fig.add_subplot(gs[2, 0], sharex=a)
    a.plot(t, raw, color=MUTED, lw=0.5)
    for s, d in zip(art.start_sec, art.duration_sec):
        a.axvspan(s / 3600, (s + d) / 3600, color=ORANGE, alpha=0.25, lw=0)
    a.set_ylim(-3, 104); a.set_ylabel("SaO$_2$ (%)"); a.set_title("(a) Raw channel, scored artefact shaded", fontsize=8.5)
    b.plot(t, filled, color=MUTED, lw=0.5)
    tail = filled.copy(); tail[:18748] = np.nan
    b.plot(t, tail, color=ORANGE, lw=1.6, label="constant fill")
    b.axhline(90, color=INK, lw=0.5, ls=":")
    b.set_ylim(50, 102); b.set_ylabel("SaO$_2$ (%)"); b.set_title("(b) Earlier output, every gap filled", fontsize=8.5)
    b.text(6.2, 82, "constant fill below 90%", color=INK, fontsize=7.5)
    c.plot(t, clean, color=BLUE, lw=0.5)
    c.axhline(90, color=INK, lw=0.5, ls=":")
    c.fill_between(t, 50, 52.5, where=mask, color=AQUA, lw=0, step="mid")
    c.set_ylim(50, 102); c.set_ylabel("SaO$_2$ (%)"); c.set_xlabel("Time from start of recording (h)")
    c.set_title("(c) Corrected signal, validity mask as a green bar", fontsize=8.5)
    c.text(6.2, 74, "gap left invalid", color=INK, fontsize=7.5)
    plt.setp(a.get_xticklabels(), visible=False); plt.setp(b.get_xticklabels(), visible=False)
    with pyedflib.EdfReader(RAW + f"shhs1-{pid2}.edf") as f:
        lab = f.getSignalLabels()
        s2 = f.readSignal(lab.index("SaO2")); hr = f.readSignal(lab.index("H.R."))
    lo, hi, r0, r1 = 2000, 6000, 2457, 5635
    tt = np.arange(lo, hi) / 60
    gs2 = gs[:, 1].subgridspec(2, 1, hspace=0.28)
    d = fig.add_subplot(gs2[0]); e = fig.add_subplot(gs2[1], sharex=d)
    d.plot(tt, s2[lo:hi], color=BLUE, lw=0.7)
    d.axvspan(r0 / 60, r1 / 60, color=YELLOW, alpha=0.22, lw=0)
    d.set_ylim(90, 100.5); d.set_ylabel("SaO$_2$ (%)"); d.set_title("(d) A long constant-value run", fontsize=8.5)
    e.plot(tt, hr[lo:hi], color=MUTED, lw=0.5)
    e.axvspan(r0 / 60, r1 / 60, color=YELLOW, alpha=0.22, lw=0)
    e.set_ylabel("H.R. channel (bpm)"); e.set_xlabel("Time from start of recording (min)")
    e.set_title("(e) H.R. channel during the same run", fontsize=8.5)
    plt.setp(d.get_xticklabels(), visible=False)
    save(fig, "fig_cleaning")


def fig_roc_pr():
    lr = np.load(R2 + "logreg_oof_scores.npz")
    y = lr["confounders_only"][0].astype(int)
    z = np.load(R3 + "oof_A.npz")
    assert (z["y"].astype(int) == y).all()
    rows = [("Clinical LR (18)", lr["confounders_only"][1], BLUE, "-"), ("Combined LR (52)", lr["obm_plus_confounders"][1], ORANGE, "--"),
            ("Deep score, arm A", z["deep_platt_mean"], AQUA, "-"), ("Biomarker LR (34)", lr["obm_only"][1], YELLOW, "-.")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.65), gridspec_kw={"wspace": 0.24})
    for n, s, c, ls in rows:
        f, t, _ = roc_curve(y, s); a.plot(f, t, color=c, ls=ls, label=f"{n}, AUC {roc_auc_score(y, s):.4f}")
        p, r, _ = precision_recall_curve(y, s); b.plot(r, p, color=c, ls=ls, label=f"{n}, AUPRC {average_precision_score(y, s):.4f}")
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":"); a.set_xlabel("1 minus specificity"); a.set_ylabel("Sensitivity"); a.set_title("(a) ROC curve")
    b.axhline(y.mean(), color=MUTED, lw=0.6, ls=":"); b.text(0.98, y.mean() - 0.045, "event rate 0.1905", color=MUTED, ha="right", fontsize=7.5)
    b.set_xlabel("Recall"); b.set_ylabel("Precision"); b.set_ylim(0, 1); b.set_title("(b) Precision-recall curve")
    a.legend(loc="lower right"); b.legend(loc="upper right")
    a.set_xlim(0, 1); a.set_ylim(0, 1); b.set_xlim(0, 1)
    save(fig, "fig_roc_pr")


def net_benefit(y, s, pts):
    # net benefit at threshold pt: TP/n - FP/n * pt / (1 - pt)
    n = len(y); out = []
    for pt in pts:
        pos = s >= pt
        tp = (pos & (y == 1)).sum(); fp = (pos & (y == 0)).sum()
        out.append(tp / n - fp / n * pt / (1 - pt))
    return np.array(out)


def fig_calib_dca():
    fs = {"LR": np.load(R2 + "logreg_oof_scores.npz"), "RF": np.load(R2 + "rf_oof_scores.npz"), "XGB": np.load(R2 + "xgb_oof_scores.npz")}
    y = fs["LR"]["confounders_only"][0].astype(int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.24})
    for (m, c, ls, mk) in [("LR", BLUE, "-", "o"), ("RF", ORANGE, "--", "s"), ("XGB", AQUA, "-.", "^")]:
        s = fs[m]["obm_plus_confounders"][1]
        q = pd.qcut(s, 10, labels=False, duplicates="drop")
        d = pd.DataFrame({"s": s, "y": y, "q": q}).groupby("q").mean()
        a.plot(d.s, d.y, color=c, ls=ls, marker=mk, ms=3.2, label=m)
    a.plot([0, 1], [0, 1], color=MUTED, lw=0.6, ls=":")
    a.set_xlim(0, 0.9); a.set_ylim(0, 0.72); a.set_xlabel("Mean predicted risk, by decile"); a.set_ylabel("Observed event rate")
    a.set_title("(a) Calibration on 52 features, raw scores"); a.legend(loc="upper left")
    pts = np.linspace(0.01, 0.40, 60)
    for sp, lab, c, ls in [("confounders_only", "XGB, 18 clinical", BLUE, "-"), ("obm_plus_confounders", "XGB, 52 combined", ORANGE, "--"), ("obm_only", "XGB, 34 biomarkers", YELLOW, "-.")]:
        b.plot(pts, net_benefit(y, fs["XGB"][sp][1], pts), color=c, ls=ls, label=lab)
    pv = y.mean()
    b.plot(pts, pv - (1 - pv) * pts / (1 - pts), color=MUTED, lw=0.8, ls=":", label="Treat all")
    b.axhline(0, color=INK, lw=0.5)
    b.set_ylim(-0.03, 0.2); b.set_xlim(0, 0.4); b.set_xlabel("Threshold risk"); b.set_ylabel("Net benefit")
    b.set_title("(b) Decision curve, XGB, descriptive"); b.legend(loc="upper right")
    save(fig, "fig_calib_dca")


def boot_ci(y, s, n=2000, seed=42):
    # percentile bootstrap CI for the AUC, skipping resamples that only have one class
    rng = np.random.default_rng(seed); v = []
    idx = np.arange(len(y))
    for _ in range(n):
        i = rng.choice(idx, len(idx))
        if y[i].min() == y[i].max():
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def fig_forest():
    mc = pd.read_csv(R2 + "model_comparison.csv")
    print(mc.columns.tolist()[:8])
    def row(model, spec):
        r = mc[(mc.model == model) & (mc.specification == spec)].iloc[0]
        return r.auc_roc, r.auc_ci_lo, r.auc_ci_hi
    names = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB"}
    cols = {"LR": BLUE, "XGB": AQUA, "RF": ORANGE}
    rows = []
    for sp, lab in [("confounders_only", "18 clinical"), ("obm_plus_confounders", "52 combined"), ("obm_only", "34 biomarkers")]:
        for m in ["logistic_regression", "xgboost", "random_forest"]:
            au, lo, hi = row(m, sp)
            rows.append((f"{names[m]}, {lab}", au, lo, hi, cols[names[m]], "o"))
    y = np.load(R3 + "oof_A.npz")["y"].astype(int)
    st = np.load(T + "derived/structure_oof_repeat0.npy")
    lo, hi = boot_ci(y, st); print("structure CI", roc_auc_score(y, st), lo, hi)
    # deep model rows here and the increments in panel (b) are fixed values typed in
    rows += [("Deep, arm A (saturation)", 0.6146, 0.5939, 0.6353, VIOLET, "s"), ("Sleep structure LR (9)", roc_auc_score(y, st), lo, hi, MUTED, "D"),
             ("Deep control, arm NCA", 0.5796, 0.5579, 0.6018, VIOLET, "s"), ("Deep control, arm NC", 0.5712, 0.5504, 0.5938, VIOLET, "s")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.95), gridspec_kw={"wspace": 0.62, "width_ratios": [1.1, 1]})
    for k, (n, au, l, h, c, mk) in enumerate(rows):
        yy = len(rows) - k
        a.plot([l, h], [yy, yy], color=c, lw=1.1); a.plot(au, yy, marker=mk, color=c, ms=4)
    a.set_yticks([len(rows) - k for k in range(len(rows))]); a.set_yticklabels([r[0] for r in rows], fontsize=7.8)
    a.axvline(0.5, color=MUTED, lw=0.6, ls=":"); a.set_xlim(0.49, 0.82); a.set_xlabel("AUC with 95% CI"); a.set_title("(a) Discrimination"); a.grid(axis="y", visible=False)
    inc = [("LR, biomarkers added", -0.0007, -0.0023, 0.0009, BLUE, "o"), ("XGB, biomarkers added", -0.0045, -0.0081, -0.0009, AQUA, "o"), ("RF, biomarkers added", -0.0088, -0.0140, -0.0037, ORANGE, "o"),
           ("C2, deep on clinical LR", -0.00024, -0.00063, 0.00016, VIOLET, "s"), ("C3, deep on combined LR", -0.00039, -0.00085, 0.00006, VIOLET, "s")]
    b.axvspan(-0.005, 0.005, color=VIOLET, alpha=0.08, lw=0)
    for k, (n, d, l, h, c, mk) in enumerate(inc):
        yy = len(inc) - k
        b.plot([l, h], [yy, yy], color=c, lw=1.1); b.plot(d, yy, marker=mk, color=c, ms=4)
    b.set_yticks([len(inc) - k for k in range(len(inc))]); b.set_yticklabels([r[0] for r in inc], fontsize=7.8)
    b.axvline(0, color=INK, lw=0.5); b.set_xlim(-0.016, 0.008); b.set_ylim(0.3, 5.7); b.grid(axis="y", visible=False)
    b.set_xlabel("Change in AUC"); b.set_title("(b) Increment over the base model")
    b.text(0.0048, 0.42, r"margin $\pm$0.005", ha="right", fontsize=7.2, color=MUTED)
    save(fig, "fig_forest")


CL = {"age_s1": "Age", "is_male": "Male sex", "SystBP": "Systolic BP", "DiasBP": "Diastolic BP", "HTNDerv_s1": "Hypertension", "HDL": "HDL cholesterol", "smokstat_s1_1": "Current smoker", "smokstat_s1_2": "Former smoker",
      "CCB1": "Calcium channel blocker", "Chol": "Total cholesterol", "DIURET1": "Diuretic", "BETA1": "Beta blocker", "ACE1": "ACE inhibitor", "LIPID1": "Lipid-lowering drug", "Trig": "Triglycerides", "bmi_s1": "BMI", "race_2": "Black", "race_3": "Other race"}


def fig_shap():
    fig = plt.figure(figsize=(W, 2.75))
    gs = GridSpec(1, 4, figure=fig, wspace=1.35, width_ratios=[1, 1, 1, 1.05])
    for k, (m, t, unit) in enumerate([("logistic_regression", "(a) LR", "log odds"), ("xgboost", "(b) XGB", "log odds"), ("random_forest", "(c) RF", "probability")]):
        d = pd.read_csv(R4 + f"shap_importance_{m}_obm_plus_confounders.csv").sort_values("mean_abs_shap", ascending=False).head(10)
        ax = fig.add_subplot(gs[0, k])
        cl = [f in CL for f in d.feature]
        ax.barh(range(len(d))[::-1], d.mean_abs_shap, color=[BLUE if c else ORANGE for c in cl], height=0.72, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(d))[::-1])
        ax.set_yticklabels([{"Calcium channel blocker": "Calcium channel\nblocker", "sleep_zero_crossings_per_hour": "sleep_zero_crossings\n_per_hour"}.get(CL.get(f, f), CL.get(f, f)) for f in d.feature], fontsize=6.8, linespacing=0.9)
        ax.set_xlabel(f"Mean |SHAP|, {unit}", fontsize=7.5); ax.set_title(t); ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=7)
    h = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    fig.legend(h, ["Clinical", "Oximetry"], loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.36, 1.08))
    z = np.load(R4 + "shap_logistic_regression_obm_only.npz", allow_pickle=True)
    print(z.files)
    imp = pd.read_csv(R4 + "shap_importance_logistic_regression_obm_only.csv")
    names = list(z["features"]) if "features" in z.files else list(imp.feature)
    sv = z["shap"] if "shap" in z.files else z[z.files[0]]
    i, j = names.index("sleep_odi3"), names.index("nrem_odi3")
    ax = fig.add_subplot(gs[0, 3])
    ax.scatter(sv[:, i], sv[:, j], s=1.4, color=VIOLET, alpha=0.25, lw=0, rasterized=True)
    ax.axhline(0, color=MUTED, lw=0.4); ax.axvline(0, color=MUTED, lw=0.4)
    ax.set_xlabel(r"SHAP, sleep_odi3", fontsize=7.5); ax.set_ylabel(r"SHAP, nrem_odi3", fontsize=7.5); ax.set_title("(d) Cancelling pair"); ax.tick_params(labelsize=7)
    print("pair r", np.corrcoef(sv[:, i], sv[:, j])[0, 1])
    save(fig, "fig_shap")


def fig_attr():
    a4 = pd.read_csv(R4 + "deep_attrib_A.csv")
    ctl = pd.read_csv(R4 + "deep_attrib_A_random.csv")
    bands = ["50-80", "80-85", "85-88", "88-90", "90-92", "92-94", "94-96", "96-98", "98-100"]
    abs_ = np.array([a4[f"exact_bin_{i}_abs"].sum() for i in range(9)]); nn = np.array([a4[f"exact_bin_{i}_n"].sum() for i in range(9)])
    sh, tm = abs_ / abs_.sum(), nn / nn.sum()
    print("shares", np.round(sh, 4), np.round(tm, 4))
    tr = a4[a4.pid.isin(ctl.pid)]
    s_t = np.array([tr[f"exact_bin_{i}_abs"].sum() for i in range(9)]); s_u = np.array([ctl[f"exact_bin_{i}_abs"].sum() for i in range(9)])
    s_t, s_u = s_t / s_t.sum(), s_u / s_u.sum()
    print("L1", np.abs(s_t - s_u).sum())
    fig = plt.figure(figsize=(W, 3.95))
    gs = GridSpec(3, 2, figure=fig, hspace=0.85, wspace=0.2, height_ratios=[1.1, 0.8, 0.5])
    a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
    x = np.arange(9); w = 0.38
    a.bar(x - w / 2, sh, w, color=BLUE, label="Share of |attribution|"); a.bar(x + w / 2, tm, w, color=MUTED, alpha=0.55, label="Share of sleep time")
    a.set_xticks(x); a.set_xticklabels(bands, fontsize=6.8, rotation=35, ha="right"); a.set_title("(a) By saturation band (%), all 4,352", fontsize=8.5); a.legend(loc="upper left"); a.grid(axis="x", visible=False)
    b.bar(x - w / 2, s_t, w, color=BLUE, label="Trained"); b.bar(x + w / 2, s_u, w, color=ORANGE, label="Untrained")
    b.set_xticks(x); b.set_xticklabels(bands, fontsize=6.8, rotation=35, ha="right"); b.set_title("(b) Trained and untrained, same 450", fontsize=8.5); b.legend(loc="upper left"); b.grid(axis="x", visible=False)
    d = np.load(R4 + "deep_traces_A.npz")
    pid = "200096"
    pct, sl, rem, ex = d[f"{pid}_percent"], d[f"{pid}_sleep"], d[f"{pid}_rem"], d[f"{pid}_exact_step"]
    t = np.arange(len(pct)) / 3600
    c = fig.add_subplot(gs[1, :])
    pp = pct.astype(float).copy(); pp[~sl] = np.nan
    c.plot(t, pp, color=BLUE, lw=0.55)
    c.fill_between(t, 80, 100.5, where=rem, color=YELLOW, alpha=0.18, lw=0)
    c.set_ylabel("SpO$_2$ (%)"); c.set_ylim(80, 100.5); c.set_title("(c) One night: saturation in valid sleep, REM shaded")
    plt.setp(c.get_xticklabels(), visible=False)
    c2 = fig.add_subplot(gs[2, :], sharex=c)
    ts = (np.arange(len(ex)) * 32 + 16) / 3600
    c2.bar(ts, ex, width=32 / 3600, color=[ORANGE if v > 0 else AQUA for v in ex], lw=0)
    c2.axhline(0, color=INK, lw=0.4); c2.set_ylabel("$C_t$ (logit)"); c2.set_xlabel("Time within the padded night (h)"); c2.grid(axis="x", visible=False)
    c2.set_title("(d) Exact contribution of each 32 s step", fontsize=8.5)
    save(fig, "fig_attr")


if __name__ == "__main__":
    for f in [fig_cleaning, fig_roc_pr, fig_calib_dca, fig_forest, fig_shap, fig_attr]:
        f()
