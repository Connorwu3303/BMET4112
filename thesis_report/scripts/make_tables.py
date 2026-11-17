"""Builds the Chapter 4 LaTeX tables from saved outputs so the numbers aren't typed by hand.
The exceptions are the binary AUCs in bin0 and the group sizes (4,352, 3,523 and 829).

Reads facts/table1_baseline.csv and derived/*.json, writes latex/tables/tab_baseline.tex and tab_sensitivity.tex.
Run after table1_baseline.py, derived_analyses.py and sensitivity_features.py.

Usage:
    python3 thesis_report/scripts/make_tables.py
"""
import json
import os
import pandas as pd
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
d = pd.read_csv(T + "facts/table1_baseline.csv")

def row(label, lv=None):
    r = d[(d.label == label)] if lv is None else d[(d.label == label) & (d.level == lv)]
    return r.iloc[0]

def esc(s):
    # add the thousands comma back to 4 digit counts and escape % and [ ] for LaTeX
    import re
    s = re.sub(r"^(\d)(\d{3}) \(", r"\1,\2 (", s)
    return s.replace("%", r"\%").replace("[", "{[}").replace("]", "{]}")

# baseline table: (row label, label in table1_baseline.csv, level, kind)
# kind: ms = mean (SD), mi = median [IQR], n = count (%), sev = rdi4p 15 or more, summed below
ROWS = [("Age (years)", "Age (years)", None, "ms"), ("Male sex", "Sex", "Male", "n"), ("Race, Black", "Race", "Black", "n"), ("Race, Other", "Race", "Other", "n"),
        ("BMI (kg/m\\textsuperscript{2})", "BMI (kg/m2)", None, "ms"), ("Systolic BP (mmHg)", "Systolic BP (mmHg)", None, "ms"), ("Diastolic BP (mmHg)", "Diastolic BP (mmHg)", None, "ms"),
        ("Total cholesterol (mg/dL)", "Total cholesterol (mg/dL)", None, "ms"), ("HDL cholesterol (mg/dL)", "HDL cholesterol (mg/dL)", None, "ms"), ("Triglycerides (mg/dL)", "Triglycerides (mg/dL)", None, "ms"),
        ("Current smoker", "Smoking status", "Current", "n"), ("Former smoker", "Smoking status", "Former", "n"), ("Derived hypertension", "Hypertension (derived)", "Yes", "n"),
        ("Beta blocker, no diuretic", "Beta blocker without diuretic", "Yes", "n"), ("ACE inhibitor, no diuretic", "ACE inhibitor without diuretic", "Yes", "n"), ("Calcium channel blocker", "Any calcium channel blocker", "Yes", "n"),
        ("Diuretic", "Any diuretic", "Yes", "n"), ("Lipid-lowering drug", "Any lipid lowering medication", "Yes", "n"),
        ("Diabetes (not in the model)", "Diabetes, parent cohort report", "Yes", "n"),
        ("T90 (\\% of valid sleep)", "T90 (% of valid sleep time below 90%)", None, "mi"), ("ODI at 3 points (events/h)", "ODI3 (events/h of valid sleep)", None, "mi"), ("ODI at 4 points (events/h)", "ODI4 (events/h of valid sleep)", None, "mi"),
        ("Mean \\spo{} in valid sleep (\\%)", "Mean SpO2 in valid sleep (%)", None, "ms"), ("Nadir \\spo{} in valid sleep (\\%)", "Nadir SpO2 in valid sleep (%)", None, "ms"), ("Valid sleep time (h)", "Valid sleep time, pipeline (h)", None, "ms"),
        ("Published \\var{rdi4p} below 5", "OSA severity by published rdi4p", "under 5", "n"), ("Published \\var{rdi4p} of 15 or more", None, None, "sev"), ("Follow-up (years)", "Days to last contact or death, censdate (years)", None, "mi"), ("Dead at last contact", "Vital status at last contact", "Dead", "n")]
out = [r"\begin{tabular}{@{}lllll@{}}", r"\toprule", r"\textbf{Characteristic} & \textbf{All ($n=4{,}352$)} & \textbf{No event ($n=3{,}523$)} & \textbf{Incident event ($n=829$)} & \textbf{SMD} \\", r"\midrule"]
for lab, key, lv, kind in ROWS:
    if kind == "sev":
        # no single row for this in the csv, so add the two top bands and redo the SMD here
        a = d[(d.label == "OSA severity by published rdi4p") & (d.level.isin(["15 to under 30", "30 or over"]))]
        def tot(col):
            n = sum(int(x.split(" ")[0]) for x in a[col])
            return n
        an, bn, cn = tot("overall"), tot("no_event"), tot("incident_event")
        p1, p2 = bn / 3523, cn / 829
        smd = abs(p2 - p1) / ((p1 * (1 - p1) + p2 * (1 - p2)) / 2) ** 0.5
        out.append(f"{lab} & {an} ({100*an/4352:.1f}\\%) & {bn} ({100*bn/3523:.1f}\\%) & {cn} ({100*cn/829:.1f}\\%) & {smd:.2f} \\\\")
        continue
    r = row(key, lv)
    if kind == "mi":
        v = [r.overall_median_iqr, r.no_event_median_iqr, r.incident_event_median_iqr]
    else:
        v = [r.overall, r.no_event, r.incident_event]
    v = [esc(x) for x in v]
    out.append(f"{lab} & {v[0]} & {v[1]} & {v[2]} & {r.smd:.2f} \\\\")
out += [r"\bottomrule", r"\end{tabular}"]
open(T + "latex/tables/tab_baseline.tex", "w").write("\n".join(out) + "\n")

# sensitivity table: A is the Cox hazard ratios, B is AUC and Harrell's C under the other framings
j = json.load(open(T + "derived/derived_results.json"))
s = json.load(open(T + "derived/sensitivity_results.json"))
cx = j["C_cox_per_sd"]
print(list(cx.keys()))
NAMES = {"sleep_ct90_pct": "T90", "sleep_mean": "Mean \\spo{}", "sleep_odi3": "ODI at 3 points", "sleep_odi4": "ODI at 4 points", "sleep_min": "Nadir \\spo{}"}
def hr(e):
    return f"{e['hr_per_sd']:.3f} ({e['lo']:.3f}, {e['hi']:.3f})"
L = [r"\begin{tabularx}{\textwidth}{@{}Xlll@{}}", r"\toprule", r"\multicolumn{4}{@{}l}{\textbf{A. Cox hazard ratio with 95\% CI, per standard deviation unless stated ($n=4{,}352$)}} \\", r"\textbf{Variable} & \textbf{Unadjusted} & \textbf{Age and sex} & \textbf{All 18 clinical} \\", r"\midrule"]
for k in cx:
    if not isinstance(cx[k], dict) or "unadjusted" not in cx[k]:
        continue
    nm = NAMES.get(k, {"obm_oof_score": "OBM LR score", "deep_score": "Deep score, arm A"}.get(k, k))
    L.append(f"{nm} & {hr(cx[k]['unadjusted'])} & {hr(cx[k]['age_sex'])} & {hr(cx[k]['all_18'])} \\\\")
q = j["C_t90_groups"]
def hq(e):
    return f"{e['hr_top_vs_zero']:.2f} ({e['lo']:.2f}, {e['hi']:.2f})"
L.append(r"\midrule")
L.append(f"T90, top fifth against T90 of zero & {hq(q['unadjusted'])} & {hq(q['age_sex'])} & {hq(q['all_18'])} \\\\")
L += [r"\bottomrule", r"\end{tabularx}", "", r"\vspace{6pt}", ""]
L += [r"\begin{tabularx}{\textwidth}{@{}Xllll@{}}", r"\toprule", r"\multicolumn{5}{@{}l}{\textbf{B. Discrimination under alternative framings}} \\",
      r"\textbf{Model} & \textbf{\shortstack[l]{AUC,\\ binary}} & \textbf{\shortstack[l]{Harrell's\\ C}} & \textbf{\shortstack[l]{AUC, 10 years, no\\ deaths or censored}} & \textbf{\shortstack[l]{AUC, no deaths\\ without an event}} \\", r"\midrule"]
h, y10, x = j["C_harrell"], j["C_10y"]["auc"], j["C_exclude_noncv_deaths"]["auc"]
bin0 = {"clinical": 0.7805, "stacked": 0.7802, "combined": 0.7783, "deep": 0.6145, "obm": 0.6040}
for nm, key, hk in [("Clinical LR", "clinical", "clinical_repeat0"), ("Clinical LR plus deep score", "stacked", "clinical_plus_deep_stacked"), ("Clinical and OBM LR", "combined", "combined_repeat0"), ("Deep score, arm A, uncalibrated", "deep", "deep"), ("OBM LR", "obm", "obm_only_repeat0")]:
    ex = f"{x[key]:.4f}" if key in x else "not computed"
    L.append(f"{nm} & {bin0[key]:.4f} & {h[hk]:.4f} & {y10[key]:.4f} & {ex} \\\\")
L += [r"\bottomrule", r"\end{tabularx}"]
open(T + "latex/tables/tab_sensitivity.tex", "w").write("\n".join(L) + "\n")
print("done")
