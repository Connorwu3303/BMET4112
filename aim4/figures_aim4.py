"""
Aim 4 figures, drawn from the saved outputs.

Every panel reads a file written by one of the analysis scripts and nothing is
recomputed here, so a figure can't disagree with the table next to it. Each
figure is saved as PNG and PDF in processed/aim4/results/figures/.

  fig1_shap_importance     classical SHAP importance, oximetry vs clinical
  fig2_shap_beeswarm       per participant spread behind figure 1
  fig3_collinearity        individual vs grouped contributions
  fig4_attribution_spo2    what the network reads, by saturation band
  fig5_example_nights      two nights with attribution and scored desaturations
  fig6_method_agreement    deep attribution methods vs the exact decomposition
  fig7_occlusion           the occlusion test
  fig8_sanity_control      trained vs untrained attribution profile

Run:
    python3 figures_aim4.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common4 as C                                      # noqa: E402

FIG_DIR = os.path.join(C.RESULTS, "figures")
CLINICAL_COLOUR = "#4C6EF5"
OXIMETRY_COLOUR = "#E8590C"
NEUTRAL = "#495057"

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.png and .pdf")


def clinical_features():
    import ml_common as MC
    _, _, feats, _ = MC.load_dataset("confounders_only")
    return set(feats)


def have(name):
    return os.path.exists(os.path.join(C.RESULTS, name))


def csv(name):
    return pd.read_csv(os.path.join(C.RESULTS, name))


# --- figures ---

def fig1_shap_importance():
    if not have("shap_importance_all.csv"):
        return
    imp = csv("shap_importance_all.csv")
    clin = clinical_features()
    fig, axes = plt.subplots(1, 3, figsize=(13, 6))
    for ax, spec in zip(axes, C.SPECS):
        t = imp[(imp["model"] == "logistic_regression")
                & (imp["specification"] == spec)]
        t = t.sort_values("mean_abs_shap", ascending=True).tail(16)
        colours = [CLINICAL_COLOUR if f in clin else OXIMETRY_COLOUR
                   for f in t["feature"]]
        ax.barh(range(len(t)), t["mean_abs_shap"], color=colours)
        ax.set_yticks(range(len(t)))
        ax.set_yticklabels(t["feature"], fontsize=7)
        ax.set_xlabel("mean absolute contribution, log odds")
        ax.set_title(spec.replace("_", " "), fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLINICAL_COLOUR),
               plt.Rectangle((0, 0), 1, 1, color=OXIMETRY_COLOUR)]
    axes[0].legend(handles, ["clinical variable", "oximetry biomarker"],
                   loc="lower right", fontsize=8)
    fig.suptitle("Out of fold SHAP importance, logistic regression. "
                 "Top 16 features per specification", fontsize=11)
    save(fig, "fig1_shap_importance")


def fig2_shap_beeswarm():
    path = os.path.join(
        C.RESULTS, "shap_logistic_regression_obm_plus_confounders.npz")
    if not os.path.exists(path):
        return
    z = np.load(path, allow_pickle=True)
    shap = z["shap"]
    feats = [str(f) for f in z["features"]]
    import ml_common as MC
    X, _, _, _ = MC.load_dataset("obm_plus_confounders")
    order = np.argsort(-np.abs(shap).mean(axis=0))[:14]
    clin = clinical_features()

    fig, ax = plt.subplots(figsize=(8, 6))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order[::-1]):
        v = shap[:, j]
        raw = X[feats[j]].to_numpy(dtype=float)
        ok = np.isfinite(raw)
        # colour by the participant's percentile on that feature, so you can
        # see whether high values push risk up or down. Percentiles because the
        # features have wildly different scales.
        pct = np.full(len(raw), np.nan)
        pct[ok] = pd.Series(raw[ok]).rank(pct=True).to_numpy()
        jitter = rng.uniform(-0.18, 0.18, size=len(v))
        ax.scatter(v, row + jitter, c=pct, cmap="coolwarm", s=2, alpha=0.45,
                   vmin=0, vmax=1, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [f"{feats[j]}{' *' if feats[j] not in clin else ''}"
         for j in order[::-1]], fontsize=8)
    ax.axvline(0.0, color=NEUTRAL, lw=0.8)
    ax.set_xlabel("contribution to the predicted log odds")
    ax.set_title("Per participant contributions, clinical plus oximetry model.\n"
                 "Colour is the participant's percentile on that feature; "
                 "an asterisk marks an oximetry biomarker", fontsize=10)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax,
                      fraction=0.03)
    cb.set_label("feature percentile", fontsize=8)
    save(fig, "fig2_shap_beeswarm")


def fig3_collinearity():
    if not have("grouped_contributions.csv"):
        return
    g = csv("grouped_contributions.csv")
    g = g[(g["model"] == "logistic_regression")
          & (g["specification"] == "obm_only")]
    if g.empty:
        return
    g = g.sort_values("sum_of_member_mean_abs", ascending=True).tail(12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    y = np.arange(len(g))
    ax1.barh(y + 0.2, g["sum_of_member_mean_abs"], height=0.38,
             color=NEUTRAL, label="sum of the members' individual magnitudes")
    ax1.barh(y - 0.2, g["mean_abs_shap_summed"], height=0.38,
             color=OXIMETRY_COLOUR, label="magnitude of the summed contribution")
    ax1.set_yticks(y)
    ax1.set_yticklabels([m[:44] for m in g["members"]], fontsize=6.5)
    ax1.set_xlabel("mean absolute contribution, log odds")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.set_title("Correlated clusters, oximetry only model", fontsize=10)

    ax2.barh(y, g["cancellation_ratio"], color=[
        OXIMETRY_COLOUR if r < 0.5 else NEUTRAL
        for r in g["cancellation_ratio"]])
    ax2.axvline(1.0, color="black", lw=0.8, ls="--")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"size {int(s)}" for s in g["size"]], fontsize=7)
    ax2.set_xlabel("cancellation ratio")
    ax2.set_title("A ratio near zero means the members cancel,\n"
                  "so their individual contributions must not be quoted",
                  fontsize=10)
    save(fig, "fig3_collinearity")


def fig4_attribution_spo2():
    if not have("attribution_by_spo2.csv"):
        return
    t = csv("attribution_by_spo2.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.5))
    x = np.arange(len(t))
    cols = [OXIMETRY_COLOUR if v > 0 else CLINICAL_COLOUR
            for v in t["signed_total"]]
    ax1.bar(x, t["signed_total"], color=cols)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(t["spo2_bin"], rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("total contribution to the logit, cohort sum")
    ax1.set_title("Where time spent raises or lowers the score", fontsize=10)

    ax2.bar(x, t["minutes_per_night"], color=NEUTRAL)
    ax2.set_xticks(x)
    ax2.set_xticklabels(t["spo2_bin"], rotation=55, ha="right", fontsize=7)
    ax2.set_ylabel("minutes per night, cohort mean")
    ax2.set_title("How much time is actually spent there", fontsize=10)

    ax3.bar(x, t["enrichment"], color=NEUTRAL)
    ax3.axhline(1.0, color=OXIMETRY_COLOUR, lw=1.2, ls="--")
    ax3.set_xticks(x)
    ax3.set_xticklabels(t["spo2_bin"], rotation=55, ha="right", fontsize=7)
    ax3.set_ylabel("share of attribution over share of time")
    ax3.set_title("Enrichment. Above the dashed line means\n"
                  "the network concentrates there", fontsize=10)
    fig.suptitle("What the deep network reads from the saturation waveform, "
                 "by saturation band", fontsize=11)
    save(fig, "fig4_attribution_spo2")


def fig5_example_nights():
    tp = os.path.join(C.RESULTS, f"deep_traces_{C.DEEP_ARM}.npz")
    ap = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.csv")
    if not (os.path.exists(tp) and os.path.exists(ap)):
        return
    z = np.load(tp, allow_pickle=True)
    att = pd.read_csv(ap).set_index("pid")
    pids = sorted({int(k.split("_")[0]) for k in z.files})
    pids = [p for p in pids if p in att.index and f"{p}_exact_step" in z.files]
    if not pids:
        return
    # highest and lowest scoring nights among the saved traces, so the figure
    # shows both ends of what the model does
    scored = sorted(pids, key=lambda p: float(att.loc[p, "z_mean"]))
    pick = [scored[-1], scored[0]]
    labels = ["highest scoring night of those saved",
              "lowest scoring night of those saved"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5), sharex="col")
    for col, (pid, lab) in enumerate(zip(pick, labels)):
        pct = z[f"{pid}_percent"]
        sleep = z[f"{pid}_sleep"]
        desat = z[f"{pid}_desat"]
        step = z[f"{pid}_exact_step"]
        hours = np.arange(len(pct)) / 3600.0
        a_top, a_bot = axes[0, col], axes[1, col]

        shown = np.where(sleep, pct, np.nan)
        a_top.plot(hours, shown, lw=0.45, color=NEUTRAL)
        ymin = max(60.0, np.nanmin(shown) - 2) if np.isfinite(shown).any() \
            else 80.0
        a_top.fill_between(hours, ymin, 100.5, where=desat,
                           color=OXIMETRY_COLOUR, alpha=0.22, lw=0,
                           label="technician scored desaturation")
        a_top.axhline(90.0, color=CLINICAL_COLOUR, lw=0.7, ls="--",
                      label="90 percent")
        a_top.set_ylim(ymin, 100.5)
        a_top.set_ylabel("SpO2, percent")
        a_top.set_title(f"{lab}: participant {pid}, "
                        f"score {float(att.loc[pid, 'z_mean']):+.2f}, "
                        f"outcome {int(att.loc[pid, 'y'])}", fontsize=9)
        a_top.legend(fontsize=6.5, loc="lower left")

        sh = np.arange(len(step)) * C.STEP_SECONDS / 3600.0
        a_bot.bar(sh, step, width=C.STEP_SECONDS / 3600.0,
                  color=[OXIMETRY_COLOUR if v > 0 else CLINICAL_COLOUR
                         for v in step], linewidth=0)
        a_bot.axhline(0, color="black", lw=0.7)
        a_bot.set_xlabel("hours from the start of the sleep period")
        a_bot.set_ylabel("exact contribution\nto the logit, per 32 s step")
    fig.suptitle("The exact step decomposition over two nights. Orange raises "
                 "the predicted risk, blue lowers it", fontsize=11)
    save(fig, "fig5_example_nights")


def fig6_method_agreement():
    ap = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.csv")
    if not os.path.exists(ap):
        return
    att = pd.read_csv(ap)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    series = [("gxa_vs_exact_pearson", "gradient times activation", "#2B8A3E"),
              ("cam_vs_exact_pearson", "Grad-CAM", OXIMETRY_COLOUR),
              ("ig_vs_exact_pearson", "integrated gradients", CLINICAL_COLOUR)]
    bins = np.linspace(-1, 1, 81)
    for col, lab, colour in series:
        if col not in att:
            continue
        v = att[col].dropna()
        ax1.hist(v, bins=bins, histtype="step", lw=1.6, color=colour,
                 label=f"{lab}, median {v.median():+.3f}")
    ax1.set_xlabel("Pearson correlation with the exact step decomposition")
    ax1.set_ylabel("participants")
    ax1.legend(fontsize=7.5)
    ax1.set_title("How faithful each approximate method is", fontsize=10)

    if "ig_flat_own_completeness_error" in att:
        e = att["ig_flat_own_completeness_error"].abs()
        ax2.hist(e, bins=np.logspace(-7, 0, 60), color=NEUTRAL)
        ax2.set_xscale("log")
        ax2.axvline(0.025, color=OXIMETRY_COLOUR, lw=1.3, ls="--",
                    label="tolerance, 0.025 logits")
        ax2.set_xlabel("absolute completeness error, logits")
        ax2.set_ylabel("participants")
        ax2.legend(fontsize=7.5)
        ax2.set_title("Integrated gradients completeness after refinement",
                      fontsize=10)
    save(fig, "fig6_method_agreement")


def fig7_occlusion():
    p = os.path.join(C.RESULTS, f"occlusion_{C.DEEP_ARM}.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, suffix in zip(axes, ("cleared", "kept")):
        cols = [f"d_top_{suffix}", f"d_random_{suffix}", f"d_bottom_{suffix}"]
        if not all(c in t for c in cols):
            continue
        ax.boxplot([t[c].dropna() for c in cols], showfliers=False,
                   widths=0.55,
                   tick_labels=["most positively\nattributed",
                                "chosen at\nrandom",
                                "most negatively\nattributed"])
        ax.axhline(0, color=NEUTRAL, lw=0.9, ls="--")
        ax.set_title(f"mask {suffix}", fontsize=10)
        ax.set_ylabel("change in logit when occluded")
    fig.suptitle("The occlusion test. A faithful attribution predicts the "
                 "ordering left to right", fontsize=11)
    save(fig, "fig7_occlusion")


def fig8_sanity_control():
    a = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}.csv")
    b = os.path.join(C.RESULTS, f"deep_attrib_{C.DEEP_ARM}_random.csv")
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    ta, tb = pd.read_csv(a), pd.read_csv(b)
    shared = sorted(set(ta["pid"]) & set(tb["pid"]))
    ta = ta[ta["pid"].isin(shared)]
    tb = tb[tb["pid"].isin(shared)]
    nb = len(C.bin_labels())
    cols = [f"exact_bin_{i}_abs" for i in range(nb)]
    pa = ta[cols].sum().to_numpy()
    pb = tb[cols].sum().to_numpy()
    pa = pa / max(1e-12, pa.sum())
    pb = pb / max(1e-12, pb.sum())
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(nb)
    ax.bar(x - 0.2, pa, width=0.38, color=OXIMETRY_COLOUR,
           label="trained network")
    ax.bar(x + 0.2, pb, width=0.38, color=NEUTRAL,
           label="untrained, randomised weights")
    ax.set_xticks(x)
    ax.set_xticklabels(C.bin_labels(), rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("share of absolute attribution")
    ax.legend(fontsize=8)
    ax.set_title(f"The sanity control. Total variation distance "
                 f"{float(np.abs(pa - pb).sum()):.3f}, {len(shared)} "
                 f"participants.\nMatching profiles would mean the attribution "
                 f"shows the architecture,\nnot what the network learned",
                 fontsize=9.5)
    save(fig, "fig8_sanity_control")


def main():
    print("=" * 78)
    print("AIM 4 FIGURES")
    print("=" * 78)
    for fn in (fig1_shap_importance, fig2_shap_beeswarm, fig3_collinearity,
               fig4_attribution_spo2, fig5_example_nights,
               fig6_method_agreement, fig7_occlusion, fig8_sanity_control):
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  {fn.__name__} failed: {exc}")
    print(f"\nfigures in {FIG_DIR}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
