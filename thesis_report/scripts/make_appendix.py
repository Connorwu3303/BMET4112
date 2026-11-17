"""Builds Appendix A (thesis_report/latex/chapters/appendix.tex).

The 34 feature panel table is parsed from the checked fact sheet facts/aim2_data.md so no value is retyped.
The rest of the appendix is fixed LaTeX text inside this script.

Usage:
    python3 thesis_report/scripts/make_appendix.py
"""
import os
import re
T = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
# rows 1 to 34 of the panel table in aim2_data.md (10 columns each)
rows = []
for line in open(T + "facts/aim2_data.md", encoding="utf-8"):
    m = re.match(r"^\| (\d+) \| `([^`]+)` \| ([^|]+) \| (.+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line.rstrip())
    if m and 1 <= int(m.group(1)) <= 34:
        rows.append(m.groups())
assert len(rows) == 34, len(rows)

def tex(s):
    # markdown cell to LaTeX: `code` becomes \var{}, everything else gets escaped
    parts = re.split(r"(`[^`]+`)", s)
    out = []
    for p in parts:
        if p.startswith("`"):
            out.append(r"\var{" + p[1:-1].replace("_", r"\_") + "}")
        else:
            p = p.replace("%", r"\%").replace(r"\%^2", r"\%\textsuperscript{2}").replace("_", r"\_").replace("<", r"$<$").replace(">", r"$>$").replace("^", r"\^{}").replace("[", "{[}").replace("]", "{]}").replace("-ln", "$-$ln")
            out.append(p)
    return "".join(out).strip()

lines = []
for g in rows:
    num, feat, grp, defn, unit, code, miss, ms, med, rng = g
    # drop the "Validated against ..." note and spell out where a definition is T90
    defn = re.sub(r"\s*Validated against `[^`]+`", "", defn).replace(" T90.", " This is T90.")
    lines.append(f"{num} & \\var{{{feat.replace('_', chr(92) + '_' + chr(92) + 'allowbreak{}')}}} & {tex(defn)} & {tex(unit)} & {tex(med)} \\\\")
table = "\n".join(lines)

BODY = r"""\chapter{Parameter Tables}
\label{app:parameters}

This appendix lists the parameter values used in each aim, which are summarised in Chapter~\ref{ch:methods}. Values are as recorded in the project code and saved outputs.

\section{Signal Cleaning}

\begin{table}[H]
\centering
\caption[Signal cleaning parameters]{Signal cleaning parameters and the status of each value.}
\label{tab:app-cleaning}
\small
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{3.8cm}>{\raggedright\arraybackslash}p{2.4cm}X@{}}
\toprule
\textbf{Parameter} & \textbf{Value} & \textbf{Basis} \\
\midrule
Plausible saturation range & 50\% to 100\% & Project judgement; the probe-off value of 0.0015 is the main case \\
Longest bridged gap & 30 s & Project choice; linear bridge between two real samples, interior gaps only \\
Constant-value run limit & Over 600 s, strict & Project choice from the trade-off in Table~\ref{tab:threshold} \\
Minimum analysable time & 4.0 h & Project choice to keep overnight biomarkers stable \\
Saturation channel & \var{SaO2} & Present once in all 5,793 files, read at 1 Hz in physical units \\
Artefact events & Concept contains ``artifact'' & Case-insensitive match in the NSRR annotation file \\
Audit duration tolerance & 2.0 s & Recording length against the annotation file \\
Audit constant-run limit & 1,800 s & Set above the cleaning limit because bridges can join runs \\
\bottomrule
\end{tabularx}
\end{table}

\section{Biomarker Library and Panel}

Each family is computed per window, where $H$ is the valid hours of the window and $x$ is the set of valid samples in it. General statistics give 19 values, hypoxic burden gives 25 values at thresholds of 95, 90, 88, 85, 80, 75 and 70\%, desaturations give 46 values at each of three thresholds, and the variability family gives two. Periodicity (six values) and complexity (six values) are computed only for the recording and sleep windows, on segments of at least 300 s. Four descriptors of the recording are also stored per window (valid hours, valid sample count, segment count and longest segment), and three cross-window values are added, which gives 779 numeric values per participant. Table~\ref{tab:app-panel} defines the 34 features in the panel, with the median and quartiles in the analysed cohort of 4,352.

\footnotesize
\begin{longtable}{@{}r>{\raggedright\arraybackslash}p{4.15cm}>{\raggedright\arraybackslash}p{6.15cm}>{\raggedright\arraybackslash}p{1.7cm}>{\raggedright\arraybackslash}p{2.75cm}@{}}
\caption[Definitions of the 34 pre-specified features]{Definitions of the 34 pre-specified features. Unless stated, $x$ is the set of valid samples in sleep, $H$ is valid sleep hours, $b$ is the median of $x$ and $f_s$ is the sampling rate.}
\label{tab:app-panel} \\
\toprule
\textbf{\#} & \textbf{Feature} & \textbf{Definition} & \textbf{Unit} & \textbf{Median {[}Q1, Q3{]}} \\
\midrule
\endfirsthead
\toprule
\textbf{\#} & \textbf{Feature} & \textbf{Definition} & \textbf{Unit} & \textbf{Median {[}Q1, Q3{]}} \\
\midrule
\endhead
PANEL_ROWS
\bottomrule
\end{longtable}
\normalsize

\section{Classical Model Search Spaces}

\begin{table}[H]
\centering
\caption[Imbalance arms and search spaces for the classical models]{Imbalance arms and hyperparameter search spaces for the classical models. All selection used AUC on three inner folds. LR, logistic regression; RF, random forest; XGB, XGBoost.}
\label{tab:app-search}
\footnotesize
\begin{tabularx}{\textwidth}{@{}lX@{}}
\toprule
\textbf{Item} & \textbf{Values} \\
\midrule
LR, stage 1 arms & SMOTETomek, class weight balanced (chosen), none; full grid inside each arm \\
LR, L2 and L1 & \var{liblinear}; $C$ in 12 log-spaced values from $10^{-4}$ to $10^{1.5}$, 12 candidates each \\
LR, elastic net & \var{saga}; $C$ in $\{0.001, 0.01, 0.1, 1, 10\}$, \var{l1\_ratio} in $\{0.15, 0.5, 0.85\}$, 15 candidates \\
RF, stage 1 & Arms balanced subsample (chosen), balanced, SMOTETomek, none, at 300 trees, unlimited depth, minimum leaf 5, square root features \\
RF, broad search & 25 random draws: \var{n\_estimators} in $\{200, 300, 500, 800\}$; \var{max\_depth} in \{none, 4, 6, 8, 12, 20\}; \var{min\_samples\_split} 2 to 39; \var{min\_samples\_leaf} 1 to 29; \var{max\_features} in \{sqrt, log2, 0.05, 0.1, 0.2, 0.4\}; \var{bootstrap} true or false \\
RF, focused grid & 800 trees; depth, split and bootstrap held at the broad winner; leaf at the winner $-$5, $-$2, 0, +2, +5 and +10; \var{max\_features} in \{sqrt, 0.05, 0.1, 0.2\} plus the winner \\
XGB, stage 1 & Arms positive class weight, SMOTETomek, none (chosen), at 300 trees, learning rate 0.05, depth 4, minimum child weight 5, subsample and column sample 0.8 \\
XGB, broad search & 30 random draws: \var{n\_estimators} in $\{100, 200, 300, 500, 800\}$; learning rate log-uniform 0.01 to 0.3; depth 2 to 6; \var{min\_child\_weight} in $\{1, 3, 5, 10, 20\}$; subsample 0.6 to 1.0; column sample 0.4 to 1.0; $\gamma$ in $\{0, 0.1, 0.5, 1, 2, 5\}$; $\alpha$ log-uniform $10^{-3}$ to 10; $\lambda$ log-uniform 0.1 to 20 \\
XGB, focused grid & Learning rate at half, equal and double the winner; trees at half, equal and double the winner; depth at $-$1, 0 and +1; 27 candidates \\
\bottomrule
\end{tabularx}
\end{table}

The optimisation check passed 28 of 33 items and flagged five. Three were selections at a search boundary, namely the elastic net mixing ratio at its low edge in LR, the number of trees and share of features at the top of the range in RF, and the regulariser $\gamma$ at mixed edges in XGB. Two were the calibration of LR and RF, which follows from class weighting. One-at-a-time curves outside the searched ranges gained at most 0.0003 in inner AUC, which is about one thirty-seventh of the deficit of RF against LR, so the boundaries do not appear to explain the differences between models.

\section{Deep Learning Configuration}

\begin{table}[H]
\centering
\caption[Training configuration of the deep learning arm]{Training configuration of the deep learning arm, the same for all 45 networks.}
\label{tab:app-deep}
\small
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{4.3cm}X@{}}
\toprule
\textbf{Setting} & \textbf{Value} \\
\midrule
Loss & Binary cross-entropy with no class weight \\
Optimiser & AdamW, learning rate $10^{-3}$, weight decay $10^{-2}$ except normalisation scales and bias terms \\
Schedule & One warm-up epoch, then cosine decay to 1\% of the learning rate, per step \\
Epoch cap and batch size & 16 epochs with no early stopping; 32 in training, 16 in evaluation \\
Gradient clipping & Norm 1.0 \\
Dropout & Channel dropout 0.1 in every residual block; dropout 0.5 before the head \\
Moving average & Decay 0.99 on all state, the averaged weights being evaluated and saved \\
Random crop & Length drawn from 16,800, 19,200, 21,600, 24,000 or 26,400 samples \\
Time masking & Probability 0.5; one to three blocks of 60 to 600 samples set to the fill value with masks cleared \\
Dequantisation jitter & Uniform within half of the 0.78 point ladder, $\pm$0.078 in scaled units \\
Selection & Epoch index 3 or later; ties to the earlier epoch \\
Seeds & $10000 + 1000 r + 100 f + 10 i + a$ for repeat $r$, fold $f$, inner model $i$ and arm offset $a$ \\
Software & PyTorch 2.8 with the MPS backend; Apple M2 with 8 GB of memory \\
\bottomrule
\end{tabularx}
\end{table}

\section{Attribution Settings}

Integrated gradients used the midpoint rule, starting from 16 steps for \var{flat\_own} and 64 for \var{flat\_97}, and doubled steps for each inner model until the completeness error was at most 0.025 logits or 256 steps were reached. No participant exceeded the tolerance, and the worst errors were 0.018 and 0.015 logits. Saturation bands had edges at 50, 80, 85, 88, 90, 92, 94, 96, 98 and 100.01\%, with a value on an edge going to the upper band, so that the T90 convention holds. In the randomised weights control the head weights were drawn from a normal distribution with a standard deviation of 0.1, with seeds 20261011 plus the inner model index. Occlusion used seed 20261011, 30 participants with events and 30 without in each fold, and three random draws per inner model per condition.
"""
open(T + "latex/chapters/appendix.tex", "w", encoding="utf-8").write(BODY.replace("PANEL_ROWS", table))
print("ok")
