"""Figures for the 1, 5 and 10 year risk models, drawn from the tables in horizon_risk/results (nothing is refitted).

Saves PDF and PNG copies to horizon_risk/results/figures/.

Usage: /usr/bin/python3 horizon_risk/make_horizon_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, MUTED_INK, VIOLET, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#4a3aa7", "#c0398c"
INK, GRID = "#0b0b0b", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED_INK, "xtick.color": MUTED_INK, "ytick.color": MUTED_INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})
YEARS = (1, 5, 10)
LABEL = {"clinical": "Clinical LR, per horizon", "clinical_recal": "Clinical LR, recalibrated (post hoc)", "combined": "Clinical and oximetry LR, per horizon", "age_sex": "Age and sex",
         "oximetry": "Oximetry LR, per horizon", "recording_oximetry": "Whole-night oximetry LR (post hoc)"}
COLOUR = {"clinical": BLUE, "clinical_recal": VIOLET, "combined": AQUA, "age_sex": MUTED_INK, "oximetry": ORANGE, "recording_oximetry": PINK}
MARKER = {"clinical": "o", "clinical_recal": "P", "combined": "s", "age_sex": "D", "oximetry": "^", "recording_oximetry": "v"}


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{extension}"), dpi=220)
    plt.close(fig)


def percent_axis(axis, years):
    """Whole percent labels are wrong on the 1 year panels (ticks land on 2.5% and 7.5%), so those get their own spacing."""
    axis.set_major_formatter(ticker.PercentFormatter(1.0, decimals=0))
    if years == 1:
        axis.set_major_locator(ticker.MultipleLocator(0.02))


def fig_auc(specs=("clinical", "clinical_recal", "combined", "age_sex", "oximetry", "recording_oximetry"), name="fig_horizon_auc"):
    table = pd.read_csv(os.path.join(RESULTS, "metrics_primary.csv"))
    # like with like: every model fitted per horizon, with the recalibrated model and whole night features labelled post hoc
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    centres = np.arange(3.0)
    width = 0.13
    for i, spec in enumerate(specs):
        rows = table[table.spec == spec].set_index("years").loc[list(YEARS)]
        x = centres + (i - (len(specs) - 1) / 2) * width
        ax.errorbar(x, rows.auc_wf, yerr=[rows.auc_wf - rows.auc_wf_lo, rows.auc_wf_hi - rows.auc_wf], fmt=MARKER[spec], color=COLOUR[spec],
                    ms=5, capsize=2, lw=1.1, label=LABEL[spec])
    ax.axhline(0.5, color=MUTED_INK, lw=0.8, ls="--")
    ax.text(2.42, 0.505, "chance", color=MUTED_INK, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(centres)
    ax.set_xticklabels([f"By {y} year{'s' if y > 1 else ''}" for y in YEARS])
    ax.set_ylabel("Time-dependent AUC (95% CI)")
    ax.set_ylim(0.45, 0.92)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3 if len(specs) > 4 else 2, loc="upper center", bbox_to_anchor=(0.5, 1.23), columnspacing=1.0, handletextpad=0.3)
    save(fig, name)


def fig_calibration(specs=("clinical", "clinical_recal", "oximetry"), name="fig_horizon_calibration"):
    table = pd.read_csv(os.path.join(RESULTS, "calibration_deciles.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
    for ax, years in zip(axes, YEARS):
        rows = table[table.years == years]
        shown = rows[rows.spec.isin(list(specs))]
        top = max(shown.predicted.max(), shown.observed_hi.max()) * 1.08
        ax.plot([0, top], [0, top], color=MUTED_INK, lw=0.8, ls="--", label="Perfect agreement")
        for spec in specs:
            r = rows[rows.spec == spec]
            ax.errorbar(r.predicted, r.observed, yerr=[(r.observed - r.observed_lo).clip(lower=0), (r.observed_hi - r.observed).clip(lower=0)], marker=MARKER[spec],
                        color=COLOUR[spec], ms=3.5, lw=1.1, elinewidth=0.7, capsize=1.5, label=LABEL[spec])
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_title(f"By {y_label(years)}")
        ax.set_xlabel("Predicted risk")
        percent_axis(ax.xaxis, years)
        percent_axis(ax.yaxis, years)
    axes[0].set_ylabel("Observed risk")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, name)


def y_label(years):
    return f"{years} year" if years == 1 else f"{years} years"


def fig_net_benefit():
    table = pd.read_csv(os.path.join(RESULTS, "net_benefit_curves.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4))
    for ax, years in zip(axes, YEARS):
        rows = table[table.years == years]
        clinical = rows[rows.spec == "clinical"]
        ax.plot(clinical.threshold, clinical.treat_all, color=MUTED_INK, lw=0.9, ls=":", label="Treat everyone")
        ax.axhline(0, color=MUTED_INK, lw=0.9, ls="--", label="Treat no one")
        for spec in ("clinical", "age_sex", "oximetry"):
            r = rows[rows.spec == spec]
            ax.plot(r.threshold, r.net_benefit, color=COLOUR[spec], label=LABEL[spec])
        top = clinical.net_benefit.max()
        ax.set_ylim(-0.1 * top, top * 1.15)
        ax.set_xlim(clinical.threshold.min(), clinical.threshold.max())
        ax.set_title(f"By {y_label(years)}")
        ax.set_xlabel("Risk threshold")
        percent_axis(ax.xaxis, years)
    axes[0].set_ylabel("Net benefit")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07), columnspacing=1.0)
    fig.tight_layout()
    save(fig, "fig_horizon_net_benefit")


if __name__ == "__main__":
    fig_auc()
    # simpler figure for the main text: the four models fitted the same way at each horizon
    fig_auc(specs=("clinical", "combined", "age_sex", "oximetry"), name="fig_horizon_auc_simple")
    fig_calibration()
    # simpler figure for the shortened report: just the two per horizon models the chapter discusses
    fig_calibration(specs=("clinical", "oximetry"), name="fig_horizon_calibration_simple")
    fig_net_benefit()
    print("figures written to", OUT)
