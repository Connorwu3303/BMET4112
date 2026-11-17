"""Builds the Aim 5 LaTeX tables from the saved outputs in horizon_risk/results, so no value is retyped.

Writes tab_aim5_*.tex to thesis_report/latex/tables/. Run after evaluate_horizon_models.py.

Usage:
    python3 thesis_report/scripts/make_aim5_tables.py
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = os.path.join(ROOT, "thesis_report") + "/"
R = os.path.join(ROOT, "horizon_risk", "results") + "/"
OUT = T + "latex/tables/"
HORIZONS = (1, 5, 10)


def read(name):
    return pd.read_csv(os.path.join(R, name))


def num(x, d=4, plus=False):
    # d decimals with a proper LaTeX minus; plus=True also puts a + on positive values
    s = f"{abs(x):.{d}f}"
    if round(x, d) < 0:
        return "$-$" + s
    return ("+" + s) if plus and round(x, d) > 0 else s


def ci(x, lo, hi, d=4, plus=False):
    return f"{num(x, d, plus)} [{num(lo, d, plus)}, {num(hi, d, plus)}]"


def write(name, lines):
    open(OUT + name, "w").write("\n".join(lines) + "\n")


metrics = read("metrics_primary.csv")
metrics_c = read("metrics_primary_corrected.csv")
contrasts = read("contrasts.csv")
contrasts_c = read("contrasts_corrected.csv")
audit = json.load(open(R + "cohort_audit.json"))
audit_c = json.load(open(R + "cohort_audit_corrected.json"))
risk = read("cohort_risk_by_label.csv").set_index("definition")


def m(table, spec, years):
    # metrics row for one model at one horizon
    return table[(table.spec == spec) & (table.years == years)].iloc[0]


def c(table, a, b, years, metric="auc_wf"):
    # contrast row, model a minus model b
    return table[(table.a == a) & (table.b == b) & (table.years == years) & (table.metric == metric)].iloc[0]


# --- main performance table ---
ROWS = [("clinical", "Clinical LR, per horizon"), ("clinical_recal", "Clinical LR, Aim 2 model recalibrated (post hoc)"), ("clinical_cox", "Clinical cause-specific Cox"),
        ("age_only", "Age alone"), ("age_sex", "Age and sex"), ("oximetry", "Oximetry LR, per horizon"), ("oximetry_recal", "Oximetry LR, Aim 2 model recalibrated (post hoc)"), ("recording_oximetry", "Whole-night oximetry LR, per horizon (post hoc)"),
        ("combined", "Clinical and oximetry LR, per horizon")]
FULL_ROWS = ROWS[:5] + [("clinical_no_age", "Clinical LR without age, per horizon")] + ROWS[5:]
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xllll@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{AUC [95\% CI]} & \textbf{IPA [95\% CI]} & \textbf{O/E [95\% CI]} & \textbf{Slope [95\% CI]} \\", r"\midrule"]
for years in HORIZONS:
    a = audit[f"{years}y"]
    head = f"By {years} year{'s' if years > 1 else ''}: {a['events']} events, cumulative incidence {100 * risk.loc['released', f'risk_{years}y']:.2f}\\%"
    lines.append(r"\multicolumn{5}{@{}l}{\textit{" + head + r"}} \\")
    for spec, label in FULL_ROWS:
        r = m(metrics, spec, years)
        if pd.isna(r.auc_wf):
            lines.append(f"{label} & not estimable & & & \\\\")
            continue
        slope = ci(r.slope, r.slope_lo, r.slope_hi, 2)
        lines.append(f"{label} & {ci(r.auc_wf, r.auc_wf_lo, r.auc_wf_hi, 4)} & {ci(r.ipa, r.ipa_lo, r.ipa_hi, 3)} & {ci(r.oe, r.oe_lo, r.oe_hi, 2)} & {slope} \\\\")
    if years != HORIZONS[-1]:
        lines.append(r"\addlinespace[2pt]")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_main.tex", lines)

# short version, AUC only
head = [f"\\shortstack[l]{{By {y} year{'s' if y > 1 else ''}\\\\ {audit[f'{y}y']['events']} events, {100 * risk.loc['released', f'risk_{y}y']:.2f}\\%}}" for y in HORIZONS]
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xlll@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{" + head[0] + r"} & \textbf{" + head[1] + r"} & \textbf{" + head[2] + r"} \\", r"\midrule"]
for spec, label in [x for x in ROWS if x[0] != "age_only"]:
    cells = ["not estimable" if pd.isna(m(metrics, spec, y).auc_wf) else ci(m(metrics, spec, y).auc_wf, m(metrics, spec, y).auc_wf_lo, m(metrics, spec, y).auc_wf_hi, 4) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_main_short.tex", lines)

# --- contrasts ---
band = read("age_band_auc.csv")


def band_row(spec, comparison, years):
    return band[(band.spec == spec) & (band.comparison == comparison) & (band.years == years)].iloc[0]


lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xlll@{}}", r"\toprule",
         r"\textbf{Difference in AUC [95\% CI]} & \textbf{By 1 year} & \textbf{By 5 years} & \textbf{By 10 years} \\", r"\midrule"]
for a, b, label in (("clinical", "oximetry", "Clinical minus oximetry, per-horizon LR"), ("combined", "clinical", "Clinical and oximetry minus clinical, per-horizon LR"), ("clinical", "age_sex", "Clinical minus age and sex, per-horizon LR"),
                    ("clinical_recal", "clinical", "Recalibrated minus per-horizon clinical LR (post hoc)"), ("clinical_recal", "clinical_cox", "Recalibrated clinical LR minus clinical Cox (post hoc)"), ("recording_oximetry", "oximetry", "Whole-night oximetry minus oximetry, per-horizon LR (post hoc)"),
                    ("combined_recal", "clinical_recal", "Clinical and oximetry minus clinical, recalibrated (post hoc)"), ("clinical_recal", "age_sex", "Clinical minus age and sex, recalibrated (post hoc)"),
                    ("clinical", "no_tests", "Clinical minus no-test model, per-horizon LR (post hoc)"), ("no_tests_oximetry", "no_tests", "No-test model and oximetry minus no-test model (post hoc)"),
                    ("clinical_recording", "clinical", "Clinical and whole-night oximetry minus clinical (post hoc)"), ("no_tests_recording", "no_tests", "No-test model and whole-night oximetry minus no-test model (post hoc)"),
                    ("clinical_diabetes", "clinical", "Clinical and diabetes minus clinical, per-horizon LR (post hoc)")):
    cells = [ci(c(contrasts, a, b, y).difference, c(contrasts, a, b, y).lo, c(contrasts, a, b, y).hi, 4, True) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines.append(r"\addlinespace[2pt]")
lines.append(r"\multicolumn{4}{@{}l}{\textit{Pairs compared only inside 10 year age bands and folds (post hoc)}} \\")
for spec, comp, label in (("combined", "clinical", "Clinical and oximetry minus clinical"), ("clinical", "age_sex", "Clinical minus age and sex")):
    cells = [ci(band_row(spec, comp, y).auc, band_row(spec, comp, y).lo, band_row(spec, comp, y).hi, 4, True) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
group_auc = read("inferred_group_auc.csv")
lines += [r"\addlinespace[2pt]", r"\multicolumn{4}{@{}l}{\textit{Pairs compared only inside inferred parent groups and folds (post hoc)}} \\"]
for spec, comp, label in (("combined", "clinical", "Clinical and oximetry minus clinical"), ("clinical", "age_sex", "Clinical minus age and sex")):
    rows_ = [group_auc[(group_auc.spec == spec) & (group_auc.comparison == comp) & (group_auc.years == y)].iloc[0] for y in HORIZONS]
    cells = [ci(r.auc, r.lo, r.hi, 4, True) for r in rows_]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines += [r"\addlinespace[2pt]", r"\multicolumn{4}{@{}l}{\textit{Difference in IPA, per-horizon LR}} \\"]
for a, b, label in (("combined", "clinical", "Clinical and oximetry minus clinical"), ("clinical", "age_sex", "Clinical minus age and sex")):
    cells = [ci(c(contrasts, a, b, y, "ipa").difference, c(contrasts, a, b, y, "ipa").lo, c(contrasts, a, b, y, "ipa").hi, 4, True) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_contrasts.tex", lines)

# short version with the main contrasts only
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xlll@{}}", r"\toprule",
         r"\textbf{Difference in AUC [95\% CI]} & \textbf{By 1 year} & \textbf{By 5 years} & \textbf{By 10 years} \\", r"\midrule"]
for a, b, label in (("clinical", "oximetry", "Clinical minus oximetry"), ("combined", "clinical", "Clinical and oximetry minus clinical"), ("clinical", "age_sex", "Clinical minus age and sex"),
                    ("no_tests_oximetry", "no_tests", "Model without tests, with oximetry, minus model without tests")):
    cells = [ci(c(contrasts, a, b, y).difference, c(contrasts, a, b, y).lo, c(contrasts, a, b, y).hi, 4, True) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
for spec, comp, label in (("combined", "clinical", "Clinical and oximetry minus clinical, inside age bands"), ("clinical", "age_sex", "Clinical minus age and sex, inside age bands")):
    cells = [ci(band_row(spec, comp, y).auc, band_row(spec, comp, y).lo, band_row(spec, comp, y).hi, 4, True) for y in HORIZONS]
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_contrasts_short.tex", lines)

# AUC inside 10 year age bands and inside inferred parent groups
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}", r"\toprule", r" & \multicolumn{3}{c}{\textbf{Inside age bands}} & \multicolumn{3}{c}{\textbf{Inside inferred groups}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", r"\textbf{Model} & 1 y & 5 y & 10 y & 1 y & 5 y & 10 y \\", r"\midrule"]
for spec, label in (("clinical", "Clinical LR, per horizon"), ("clinical_recal", "Clinical LR, recalibrated (post hoc)"), ("clinical_cox", "Clinical Cox"), ("no_tests", "No-test LR (post hoc)"), ("age_sex", "Age and sex"), ("age_only", "Age alone"), ("oximetry", "Oximetry LR, per horizon"), ("combined", "Clinical and oximetry LR, per horizon")):
    a = [num(band[(band.spec == spec) & band.comparison.isna() & (band.years == y)].iloc[0].auc, 4) for y in HORIZONS]
    b = [num(group_auc[(group_auc.spec == spec) & group_auc.comparison.isna() & (group_auc.years == y)].iloc[0].auc, 4) for y in HORIZONS]
    lines.append(f"{label} & " + " & ".join(a + b) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_strata.tex", lines)

# --- decisions at the two reference thresholds (10% by 5 years, 20% by 10 years) ---
classification = read("classification.csv")
nb = read("net_benefit_contrasts.csv")
capture = read("risk_capture.csv")
treat = read("net_benefit.csv")
SPECS = [("clinical", "Clinical LR, per horizon"), ("clinical_recal", "Clinical LR, recalibrated (post hoc)"), ("clinical_cox", "Clinical Cox"), ("no_tests", "No-test LR (post hoc)"), ("age_sex", "Age and sex"), ("age_only", "Age alone"),
         ("oximetry", "Oximetry LR, per horizon"), ("recording_oximetry", "Whole-night oximetry LR (post hoc)"), ("combined", "Clinical and oximetry LR, per horizon")]
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrrrlr@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Flagged} & \textbf{Sens.} & \textbf{Spec.} & \textbf{PPV} & \textbf{NPV} & \textbf{Net benefit [95\% CI]} & \textbf{Top 20\%} \\", r"\midrule"]
for years, threshold, name in ((5, 0.10, "By 5 years, risk of 10\\% or more"), (10, 0.20, "By 10 years, risk of 20\\% or more")):
    t = treat[(treat.years == years) & (treat.threshold.round(4) == threshold)].iloc[0].treat_all
    lines.append(r"\multicolumn{8}{@{}l}{\textit{" + name + f" (treating everyone: net benefit {num(t, 4)})" + r"}} \\")
    for spec, label in SPECS:
        k = classification[(classification.years == years) & (classification.spec == spec) & (classification.rule == f"risk at least {100 * threshold:.1f}%")].iloc[0]
        n = nb[(nb.years == years) & (nb.spec == spec) & (nb.threshold.round(4) == threshold) & nb.comparison.isna()].iloc[0]
        top = capture[(capture.years == years) & (capture.spec == spec) & (capture.top_share == 0.2)].iloc[0]
        lines.append(f"{label} & {100 * k.flagged_share:.1f} & {100 * k.sensitivity:.1f} & {100 * k.specificity:.1f} & {100 * k.ppv:.1f} & {100 * k.npv:.1f} & {ci(n.net_benefit, n.lo, n.hi, 4)} & {100 * top.share_of_events_captured:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_decisions.tex", lines)

# short version: five models and no treat-all line
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrrrlr@{}}", r"\toprule",
         r"\textbf{Model} & \textbf{Flagged} & \textbf{Sens.} & \textbf{Spec.} & \textbf{PPV} & \textbf{NPV} & \textbf{Net benefit [95\% CI]} & \textbf{Top 20\%} \\", r"\midrule"]
for years, threshold, name in ((5, 0.10, "By 5 years, risk of 10\\% or more"), (10, 0.20, "By 10 years, risk of 20\\% or more")):
    lines.append(r"\multicolumn{8}{@{}l}{\textit{" + name + r"}} \\")
    for spec, label in [x for x in SPECS if x[0] in ("clinical", "clinical_recal", "age_sex", "oximetry", "combined")]:
        k = classification[(classification.years == years) & (classification.spec == spec) & (classification.rule == f"risk at least {100 * threshold:.1f}%")].iloc[0]
        n = nb[(nb.years == years) & (nb.spec == spec) & (nb.threshold.round(4) == threshold) & nb.comparison.isna()].iloc[0]
        top = capture[(capture.years == years) & (capture.spec == spec) & (capture.top_share == 0.2)].iloc[0]
        lines.append(f"{label} & {100 * k.flagged_share:.1f} & {100 * k.sensitivity:.1f} & {100 * k.specificity:.1f} & {100 * k.ppv:.1f} & {100 * k.npv:.1f} & {ci(n.net_benefit, n.lo, n.hi, 4)} & {100 * top.share_of_events_captured:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_decisions_short.tex", lines)

# --- released and corrected (count-consistent) label side by side ---
frozen = read("released_model_on_corrected_label.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{\textbf{By 1 year}} & \multicolumn{2}{c}{\textbf{By 5 years}} & \multicolumn{2}{c}{\textbf{By 10 years}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"\textbf{Quantity} & Rel. & C-c. & Rel. & C-c. & Rel. & C-c. \\", r"\midrule"]


def pair(fn, fmt=lambda v: num(v, 4)):
    # released then corrected value for each horizon, as table cells
    cells = []
    for y in HORIZONS:
        cells += [fmt(fn(metrics, contrasts, audit, "released", y)), fmt(fn(metrics_c, contrasts_c, audit_c, "corrected", y))]
    return " & ".join(cells)


lines.append("Events & " + pair(lambda _m, _c, a, _l, y: a[f"{y}y"]["events"], lambda v: f"{v:,}") + r" \\")
lines.append(r"Cumulative incidence, Aalen-Johansen (\%) & " + pair(lambda _m, _c, _a, l, y: 100 * risk.loc[l, f"risk_{y}y"], lambda v: f"{v:.2f}") + r" \\")
for spec, label in (("clinical_recal", "Clinical LR, recalibrated (post hoc), AUC"), ("clinical_cox", "Clinical Cox, AUC"), ("age_sex", "Age and sex, AUC"), ("oximetry", "Oximetry LR, AUC")):
    lines.append(f"{label} & " + pair(lambda t, _c, _a, _l, y: m(t, spec, y).auc_wf, lambda v: num(v, 4)) + r" \\")
lines.append(r"Clinical minus oximetry, per-horizon LR, AUC & " + pair(lambda _m, t, _a, _l, y: c(t, "clinical", "oximetry", y).difference, lambda v: num(v, 4, True)) + r" \\")
lines.append(r"Clinical and oximetry minus clinical, per-horizon LR, AUC & " + pair(lambda _m, t, _a, _l, y: c(t, "combined", "clinical", y).difference, lambda v: num(v, 4, True)) + r" \\")
lines.append(r"Clinical LR, recalibrated (post hoc), O/E on own label & " + pair(lambda t, _c, _a, _l, y: m(t, "clinical_recal", y).oe, lambda v: num(v, 2)) + r" \\")
own = [num(m(metrics, "clinical_recal", y).oe, 2) for y in HORIZONS]
cross = [num(frozen[(frozen.spec == "clinical_recal") & (frozen.years == y)].iloc[0].oe, 2) for y in HORIZONS]
lines.append(r"Same model trained on released, O/E on count-consistent & -- & " + cross[0] + " & -- & " + cross[1] + " & -- & " + cross[2] + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_labels.tex", lines)

# events and cumulative risk under each reading of the label
ladder = [("no heart failure in G2, coronary count test applied", "No heart failure in G2, coronary count test applied"), ("released", "Released"), ("corrected", "Count-consistent"),
          ("corrected, plus stroke alone in G2", "Count-consistent, plus stroke alone in G2"), ("all 84 discordant counted", "All 84 discordant counted")]
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrrr@{}}", r"\toprule", r" & & \multicolumn{3}{c}{\textbf{Events by}} & \multicolumn{3}{c}{\textbf{Risk by (\%)}} \\", r"\cmidrule(lr){3-5}\cmidrule(l){6-8}",
         r"\textbf{Reading of the label} & \textbf{Events} & 1 y & 5 y & 10 y & 1 y & 5 y & 10 y \\", r"\midrule"]
for key, name in ladder:
    r = risk.loc[key]
    lines.append(f"{name} & {int(r.events_any_time)} & {int(r.events_1y)} & {int(r.events_5y)} & {int(r.events_10y)} & {100 * r.risk_1y:.2f} & {100 * r.risk_5y:.2f} & {100 * r.risk_10y:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_ladder.tex", lines)

# --- reconciliation with the earlier 10 year figure ---
rec = read("reconciliation.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrr@{}}", r"\toprule", r"\textbf{Framing of the same saved clinical score} & \textbf{$n$} & \textbf{AUC} \\", r"\midrule"]
for _, r in rec.iterrows():
    lines.append(f"{r.framing.replace('all 448 other deaths removed', 'all 448 deaths without a recorded event removed')} & {r.n:,} & {r.auc:.4f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_reconcile.tex", lines)

# --- appendix tables ---
# predicted against observed risk in each reference band
bands = read("reference_risk_bands.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xlrrrr@{}}", r"\toprule", r"\textbf{Model} & \textbf{Band of predicted risk} & \textbf{$n$} & \textbf{Events} & \textbf{Predicted (\%)} & \textbf{Observed (\%)} \\", r"\midrule"]
for years in (5, 10):
    lines.append(r"\multicolumn{6}{@{}l}{\textit{By " + str(years) + r" years}} \\")
    for spec, label in (("clinical", "Clinical LR, per horizon"), ("clinical_recal", "Clinical LR, recalibrated (post hoc)"), ("clinical_cox", "Clinical Cox")):
        for _, b in bands[(bands.years == years) & (bands.spec == spec)].iterrows():
            lo, hi = b.group.replace("%", "").split(" to ")
            group = f"{float(lo):g}\\% or more" if float(hi) >= 100 else (f"Below {float(hi):g}\\%" if float(lo) == 0 else f"{float(lo):g}\\% to below {float(hi):g}\\%")
            lines.append(f"{label} & {group} & {b.n:,} & {b.events} & {100 * b.mean_predicted:.2f} & {100 * b.observed:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_bands.tex", lines)

# sensitivity analyses for the per-horizon clinical LR
sens = read("metrics_sensitivity.csv")
NAMES = {"s_death_as_censoring": "Deaths censored, not competing", "s_marginal_km": "Censoring by Kaplan-Meier", "s_km_by_group": "Censoring by Kaplan-Meier within inferred group",
         "s_no_weights": "No weights in training, evaluation still weighted", "s_no_prior_any": "Without 263 reporting any prior disease", "s_no_prior_composite": "Without 163 reporting prior composite disease", "s_all_discordant": "All 84 discordant counted as events"}
DIAB = "clinical_diabetes"
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{By 1 year}} & \multicolumn{2}{c}{\textbf{By 5 years}} & \multicolumn{2}{c}{\textbf{By 10 years}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"\textbf{Analysis, clinical LR fitted per horizon} & AUC & O/E & AUC & O/E & AUC & O/E \\", r"\midrule"]
base = [m(metrics, "clinical", y) for y in HORIZONS]
lines.append("Main analysis & " + " & ".join(f"{b.auc_wf:.4f} & {b.oe:.2f}" for b in base) + r" \\")
lines.append("Diabetes added to the 18 variables & " + " & ".join(f"{m(metrics, DIAB, y).auc_wf:.4f} & {m(metrics, DIAB, y).oe:.2f}" for y in HORIZONS) + r" \\")
for arm, name in NAMES.items():
    rows = [sens[(sens.arm == arm) & (sens.spec == "clinical") & (sens.years == y)].iloc[0] for y in HORIZONS]
    lines.append(f"{name} & " + " & ".join(f"{r.auc_wf:.4f} & {r.oe:.2f}" for r in rows) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_sensitivity.tex", lines)

# O/E by subgroup for the recalibrated clinical LR
sub = read("calibration_by_subgroup.csv")
GROUPS = ["women", "men", "White", "Black", "diabetes", "no diabetes", "reported prior disease (263)", "no reported prior disease", "age 39 to 49", "age 50 to 59", "age 60 to 69", "age 70 to 79", "age 80 to over", "inferred group G1", "inferred group G2", "inferred group G3", "inferred group G4"]
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrlrlrl@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{By 1 year}} & \multicolumn{2}{c}{\textbf{By 5 years}} & \multicolumn{2}{c}{\textbf{By 10 years}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"\textbf{Subgroup} & Events & O/E [95\% CI] & Events & O/E [95\% CI] & Events & O/E [95\% CI] \\", r"\midrule"]
for g in GROUPS:
    cells = []
    for y in HORIZONS:
        r = sub[(sub.label == "released") & (sub.spec == "clinical_recal") & (sub.group == g) & (sub.years == y)].iloc[0]
        cells += [str(r.events), "--" if pd.isna(r.oe) else ci(r.oe, r.oe_lo, r.oe_hi, 2)]
    lines.append((g[0].upper() + g[1:]).replace("to over", "and over") + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_subgroups.tex", lines)

# Cox model with each inferred parent group held out in turn
hold = read("parent_group_validation.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrlrrlr@{}}", r"\toprule", r" & \multicolumn{3}{c}{\textbf{Released label}} & \multicolumn{3}{c}{\textbf{Count-consistent label}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(l){5-7}", r"\textbf{Held-out inferred group, Cox model} & AUC & O/E [95\% CI] & Slope & AUC & O/E [95\% CI] & Slope \\", r"\midrule"]
for years in (5, 10):
    lines.append(r"\multicolumn{7}{@{}l}{\textit{By " + str(years) + r" years}} \\")
    for g in ("G1", "G2", "G3", "G4"):
        a = hold[(hold.label == "released") & (hold.held_out == g) & (hold.years == years) & (hold.estimator == "cause specific Cox")].iloc[0]
        b = hold[(hold.label == "corrected") & (hold.held_out == g) & (hold.years == years) & (hold.estimator == "cause specific Cox")].iloc[0]
        lines.append(f"{g} ({a.events} events on the released label) & {a.auc:.4f} & {ci(a.oe, a.oe_lo, a.oe_hi, 2)} & {a.slope:.2f} & {b.auc:.4f} & {ci(b.oe, b.oe_lo, b.oe_hi, 2)} & {b.slope:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_holdout.tex", lines)

# coefficients of the Aim 2 LR and the per-horizon LRs, plus the a(t), b(t) recalibration terms
final = json.load(open(R + "recalibration_final.json"))
equations = json.load(open(R + "risk_equations.json"))
VARIABLES = {"age_s1": "Age (years)", "bmi_s1": r"BMI (kg/m\textsuperscript{2})", "SystBP": "Systolic BP (mmHg)", "DiasBP": "Diastolic BP (mmHg)", "Chol": "Total cholesterol (mg/dL)", "HDL": "HDL cholesterol (mg/dL)", "Trig": "Triglycerides (mg/dL)", "BETA1": "Beta blocker, no diuretic", "ACE1": "ACE inhibitor, no diuretic",
             "CCB1": "Calcium channel blocker", "DIURET1": "Any diuretic", "LIPID1": "Lipid-lowering drug", "HTNDerv_s1": "Derived hypertension", "is_male": "Male", "race_2": "Black race", "race_3": "Other race", "smokstat_s1_1": "Current smoker", "smokstat_s1_2": "Former smoker"}
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrrr@{}}", r"\toprule", r"\textbf{Variable} & \textbf{Median} & \textbf{Mean} & \textbf{SD} & \textbf{Aim 2 LR} & \textbf{LR 1 y} & \textbf{LR 5 y} & \textbf{LR 10 y} \\", r"\midrule"]
lines.append("Intercept & & & & " + num(final["intercept"], 3) + " & " + " & ".join(num(equations[str(y)]["intercept"], 3) for y in HORIZONS) + r" \\")
for key, name in VARIABLES.items():
    cells = [num(equations[str(y)]["coef"][key], 3) for y in HORIZONS]
    lines.append(f"{name} & {final['medians'][key]:.1f} & {final['means'][key]:.2f} & {final['sds'][key]:.2f} & {num(final['coef'][key], 3)} & {cells[0]} & {cells[1]} & {cells[2]} " + r"\\")
lines += [r"\addlinespace[2pt]", r"Recalibration of the Aim 2 LR, $a(t)$ & & & & & " + " & ".join(num(final["horizons"][str(y)]["a"], 3) for y in HORIZONS) + r" \\",
          r"Recalibration of the Aim 2 LR, $b(t)$ & & & & & " + " & ".join(num(final["horizons"][str(y)]["b"], 3) for y in HORIZONS) + r" \\", r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_equation.tex", lines)

# worked examples: predicted risk for a few example people under each approach
examples = read("worked_examples.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrrrrr@{}}", r"\toprule", r" & \multicolumn{3}{c}{\textbf{LR per horizon}} & \multicolumn{3}{c}{\textbf{LR recalibrated}} & \multicolumn{3}{c}{\textbf{Cox}} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}", r"\textbf{Risk (\%) by} & 1 y & 5 y & 10 y & 1 y & 5 y & 10 y & 1 y & 5 y & 10 y \\", r"\midrule"]
for _, e in examples.iterrows():
    v = [e.separate_lr_1y, e.separate_lr_5y, e.separate_lr_10y, e.recalibrated_1y, e.recalibrated_5y, e.recalibrated_10y, e.cox_1y, e.cox_5y, e.cox_10y]
    lines.append(e.person + " & " + " & ".join(f"{100 * x:.1f}" for x in v) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_examples.tex", lines)

# Riley sample size check
riley = read("sample_size.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}", r"\toprule", r"\textbf{Input} & \textbf{Horizon} & \textbf{Parameters} & \textbf{Events} & \textbf{Events needed} & \textbf{$n$ available} & \textbf{$n$ required} \\", r"\midrule"]
for _, r in riley.iterrows():
    needed = r.n_required * r.events / r.n_with_known_outcome
    lines.append(f"{r.spec.capitalize()} & {r.years} y & {r.parameters} & {r.events} & {needed:,.0f} & {r.n_with_known_outcome:,} & {r.n_required:,.0f} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_riley.tex", lines)

# mean a(t), b(t) over the 15 folds; shown as not estimable if any fold had b at or below 0
recal = read("recalibration_summary.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrrrr@{}}", r"\toprule", r" & \multicolumn{2}{c}{\textbf{By 1 year}} & \multicolumn{2}{c}{\textbf{By 5 years}} & \multicolumn{2}{c}{\textbf{By 10 years}} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}", r"\textbf{Mean across 15 folds, for stability only, not for use} & $a(t)$ & $b(t)$ & $a(t)$ & $b(t)$ & $a(t)$ & $b(t)$ \\", r"\midrule"]
for spec, label in (("clinical", "Clinical"), ("combined", "Clinical and oximetry"), ("oximetry", "Oximetry")):
    cells = []
    for y in HORIZONS:
        r = recal[(recal.label == "released") & (recal.spec == spec) & (recal.years == y)].iloc[0]
        cells += (["not estimable", ""] if r.folds_b_at_or_below_0 > 0 else [num(r.mean_a, 3), num(r.mean_b, 3)])
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_recal.tex", lines)

# AUC at each horizon of the saved oximetry scores from every model (ranking only)
rank = read("rank_only_all_models.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xlll@{}}", r"\toprule", r"\textbf{Saved oximetry score, AUC [95\% CI]} & \textbf{By 1 year} & \textbf{By 5 years} & \textbf{By 10 years} \\", r"\midrule"]
for basis, model, label in (("score averaged over 3 repeats", "logistic regression", "Logistic regression, averaged over repeats"), ("score averaged over 3 repeats", "random forest", "Random forest, averaged over repeats"), ("score averaged over 3 repeats", "xgboost", "XGBoost, averaged over repeats"), ("repeat 0", "1D CNN", "Deep score, repeat 0")):
    cells = []
    for y in HORIZONS:
        r = rank[(rank.years == y) & (rank.basis == basis) & (rank.model == model) & (rank.input == "obm_only")].iloc[0]
        cells.append(ci(r.auc, r.lo, r.hi, 4))
    lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_rank.tex", lines)

# learning curve, then the severe disease subgroups
curve = read("learning_curve.csv")
sizes = " & ".join(f"{v:,.0f}" for v in curve[curve.spec == "clinical"].n_train)
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrrr@{}}", r"\toprule", r"\textbf{Mean AUC by 10 years at a training size of} & " + sizes + r" \\", r"\midrule"]
for spec, name in (("clinical", "Clinical LR"), ("combined", "Clinical and oximetry LR"), ("oximetry", "Oximetry LR")):
    lines.append(name + " & " + " & ".join(f"{v:.4f}" for v in curve[curve.spec == spec].auc) + r" \\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_learning.tex", lines)

severe = read("severe_subgroups.csv")
lines = [r"\begin{tabularx}{\textwidth}{@{}Xrrll@{}}", r"\toprule", r"\textbf{Subgroup} & \textbf{$n$} & \textbf{Events} & \textbf{Oximetry AUC [95\% CI]} & \textbf{Added to clinical [95\% CI]} \\", r"\midrule"]
for name, label in (("respiratory disturbance index (4%) 30 or more", r"Respiratory disturbance index at 4\% of 30 or more"), ("T90 20% or more", r"T90 of 20\% or more"), ("men up to 70", "Men aged up to 70")):
    r = severe[(severe.label == "released") & (severe.group == name) & (severe.years == 10)].iloc[0]
    lines.append(f"{label} & {r.n:,} & {r.events} & {ci(r.oximetry_auc, r.oximetry_lo, r.oximetry_hi, 4)} & {ci(r.combined_minus_clinical, r.lo, r.hi, 4, True)} " + r"\\")
lines += [r"\bottomrule", r"\end{tabularx}"]
write("tab_aim5_severe.tex", lines)
print("tables written")
