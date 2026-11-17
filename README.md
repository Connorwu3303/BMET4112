# Overnight oximetry and cardiovascular disease risk (SHHS)

Code for my final-year biomedical engineering thesis (BMET4112) at the University of Sydney, supervised by Prof. Philip de Chazal. The question is whether the overnight blood oxygen signal (SpO2) from a sleep study helps predict future cardiovascular disease (CVD) on top of the standard clinical risk factors. Everything here uses the visit 1 recordings from the Sleep Heart Health Study (SHHS).

The work is split into five aims, and the folders follow the same split:

| Aim | Folder | What it covers |
|---|---|---|
| 1 | repo root, `aim1/` | Cleaning the SpO2 signal and building a per-sample validity mask |
| 2 | `aim2/` | Oximetry biomarkers (OBMs) and classical models: logistic regression, random forest, XGBoost |
| 3 | `aim3/` | A CNN trained on the raw overnight SpO2, tested against the clinical model |
| 4 | `aim4/` | Interpretability (SHAP, exact decomposition, integrated gradients, occlusion) and a head to head of the two pipelines |
| 5 | `horizon_risk/` | 1, 5 and 10 year absolute CVD risk from the clinical logistic regression |
| | `thesis_report/` | Scripts that build the thesis tables and figures from saved outputs |

## Data

The SHHS data come from the National Sleep Research Resource ([sleepdata.org](https://sleepdata.org)) and need an approved data access request, so none of it is in this repo. Raw recordings, processed signals and results are all gitignored.

The scripts expect the data on an external drive laid out like this:

```
/Volumes/ThesisHD/Data/
    shhs1-<id>.edf, shhs1-<id>-nsrr.xml    raw recordings and NSRR annotations
    documentation/                         SHHS dataset CSVs (shhs1-dataset-0.15.0.csv, shhs-cvd-summary-dataset-0.15.0.csv, ...)
    processed/                             everything the pipeline writes
```

Paths are set as constants near the top of each script (`SHHS_DIR`, `OUTPUT_DIR`, `RESULTS`, `DOC`, `DOC_DIR` and so on). To run it on another machine, change those. Three of the Aim 2 scripts read the SHHS CSVs from a separate `DOC_DIR`, and the `thesis_report/` scripts also expect the LaTeX project folder next to them (not included here).

## Setup

Python 3.9. The package versions the results were produced with are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The deep learning code (Aims 3 and 4) uses PyTorch's MPS backend on Apple silicon and falls back to CPU. The shell scripts use macOS `caffeinate` so the machine doesn't sleep during long runs.

## Running order

Commands are run from the repo root unless noted.

**Aim 1, signal cleaning**

```bash
python3 preprocess_shhs_batch.py
python3 verify_cleaned_data.py
python3 aim1/reclean_with_validity_mask.py
python3 aim1/verify_aim1_v2.py
```

`preprocess_shhs_batch.py` was the first cleaning pass and filled every gap however long. `reclean_with_validity_mask.py` replaced it: only gaps of 30 s or less are interpolated and everything else is marked invalid in the mask. `aim1/verify_aim1_complete.py` is the audit of the first-pass outputs.

**Aim 2, biomarkers and classical models**

```bash
python3 aim2/test_obm_library.py
python3 aim2/extract_obm_features.py --jobs 4
python3 aim2/validate_obm.py
python3 aim2/build_modelling_dataset.py
python3 aim2/train_logistic_regression.py
python3 aim2/train_random_forest.py
python3 aim2/train_xgboost.py
python3 aim2/compare_models.py
python3 aim2/summarise_results.py
python3 aim2/verify_aim2_results.py
python3 aim2/verify_optimisation.py --part fast
```

**Aim 3, deep learning**

```bash
cd aim3
python3 build_signal_store.py
python3 verify_signal_store.py
python3 gates.py --gate odi3
python3 optimise.py --trials 20
bash run_after_optimisation.sh
```

`run_after_optimisation.sh` confirms the chosen config, runs the classification gates and then `run_all.sh`, which does the unit tests, the Aim 2 comparators, a smoke test, training for arms A, NC and B, epoch selection, evaluation, diagnostics, verification and the tables, one job at a time. `run_nc.sh` and `run_nca.sh` run the two negative control arms on their own.

**Aim 4, interpretability**

```bash
bash aim4/run_aim4.sh
```

**Aim 5, fixed horizon risk**

```bash
python3 horizon_risk/test_horizon_lib.py
python3 horizon_risk/fit_horizon_models.py
python3 horizon_risk/fit_recalibrated_top_model.py
python3 horizon_risk/evaluate_horizon_models.py
python3 horizon_risk/extended_analyses.py
python3 horizon_risk/framing_checks.py
python3 horizon_risk/whole_pipeline_bootstrap.py
python3 horizon_risk/make_horizon_figures.py
```

**Thesis tables and figures**

```bash
python3 thesis_report/facts/aim2_data_checks.py
python3 thesis_report/scripts/table1_baseline.py
python3 thesis_report/scripts/derived_analyses.py
python3 thesis_report/scripts/sensitivity_features.py
python3 thesis_report/scripts/make_tables.py
python3 thesis_report/scripts/make_figures.py
python3 thesis_report/scripts/make_aim5_tables.py
python3 thesis_report/scripts/make_appendix.py
```

## Directory

```
preprocess_shhs_batch.py   first cleaning pass
verify_cleaned_data.py     QC of the first pass
aim1/                      signal cleaning and validity mask
aim2/                      oximetry biomarkers and classical models
aim3/                      CNN on the raw SpO2 signal
aim4/                      interpretability and head to head
horizon_risk/              Aim 5, 1, 5 and 10 year risk
thesis_report/
    facts/                 data checks behind the numbers quoted in the thesis
    scripts/               thesis tables, figures and extra analyses
requirements.txt
README.md
```

Each script has a short docstring at the top with its inputs, outputs and usage. Output paths below are relative to `/Volumes/ThesisHD/Data/` unless they start with `horizon_risk/` or `thesis_report/`.

### Aim 1: preprocessing and cleaning

| File | What it does | Main output |
|---|---|---|
| [`preprocess_shhs_batch.py`](preprocess_shhs_batch.py) | First cleaning pass. Reads the SpO2 channel and NSRR events for each participant, masks artifact and out of range samples, fills the gaps and saves the signal and events table. | `processed/cleaned_spo2/`, `processed/events/` |
| [`verify_cleaned_data.py`](verify_cleaned_data.py) | Quick QC of the first pass: files exist, no NaNs left, values in range, recordings not too short or flat, log matches what's on disk. | `processed/qc_report.csv` |
| [`aim1/verify_aim1_complete.py`](aim1/verify_aim1_complete.py) | Full audit of the first-pass outputs: coverage, integrity, length against the XML duration, consistency across processing blocks and whether staging is usable. | `processed/aim1_audit.csv` |
| [`aim1/reclean_with_validity_mask.py`](aim1/reclean_with_validity_mask.py) | The Aim 1 fix. Re-cleans everyone so only gaps of 30 s or less are interpolated, long dropouts and frozen sensor stretches stay NaN, and a validity mask is saved with each signal. | `processed/cleaned_spo2_v2/`, `processed/validity_mask/` |
| [`aim1/verify_aim1_v2.py`](aim1/verify_aim1_v2.py) | Audit of the corrected outputs: signal finite exactly where the mask is True, range, length, analysable time, old forward-fill runs gone, staging overlaps the valid data. | `processed/aim1_v2_audit.csv` |

### Aim 2: oximetry biomarkers and classical models

| File | What it does | Main output |
|---|---|---|
| [`aim2/obm_library.py`](aim2/obm_library.py) | The oximetry biomarker library (general statistics, complexity, periodicity, desaturations, hypoxic burden). Respects the validity mask and works on the whole night, sleep, NREM or REM. | imported module |
| [`aim2/test_obm_library.py`](aim2/test_obm_library.py) | Unit tests for the library on synthetic signals with known answers. | printed pass/fail |
| [`aim2/extract_obm_features.py`](aim2/extract_obm_features.py) | Runs the library over every participant in parallel and writes one feature table, flagging anyone with under 4 h of analysable signal. | `processed/aim2/obm_features.csv` |
| [`aim2/validate_obm.py`](aim2/validate_obm.py) | Checks my biomarkers against the published SHHS oximetry variables (nadir and mean SaO2, T90, rdi4p), split by recording quality. | `processed/aim2/obm_validation_report.txt` |
| [`aim2/build_modelling_dataset.py`](aim2/build_modelling_dataset.py) | Joins the biomarkers to the clinical covariates and outcomes, drops anyone with CVD at baseline so the target is incident CVD, and writes the feature group manifest. | `processed/aim2/modelling_dataset.csv`, `feature_manifest.json` |
| [`aim2/ml_common.py`](aim2/ml_common.py) | Shared Aim 2 code: data loading per feature set, the leakage-safe impute/scale/SMOTETomek pipeline, repeated nested CV, metrics, bootstrap CIs and the DeLong test. | imported module |
| [`aim2/xgb_support.py`](aim2/xgb_support.py) | XGBClassifier subclass that sets `scale_pos_weight` from each training fold, in its own module so joblib workers can import it. | imported module |
| [`aim2/train_logistic_regression.py`](aim2/train_logistic_regression.py) | Logistic regression with nested CV on confounders only, OBM only and OBM plus confounders, plus the final model's odds ratios. | `processed/aim2/results/logreg_*` |
| [`aim2/train_random_forest.py`](aim2/train_random_forest.py) | Random forest with the same nested CV and feature sets, ranked by held-out permutation importance. | `processed/aim2/results/rf_*` |
| [`aim2/train_xgboost.py`](aim2/train_xgboost.py) | XGBoost with the same setup, with permutation and gain importance. | `processed/aim2/results/xgb_*` |
| [`aim2/compare_models.py`](aim2/compare_models.py) | Compares the three models on the same out of fold predictions: paired DeLong tests, bootstrap AUC CIs and decision curves. | `processed/aim2/results/model_comparison.*` |
| [`aim2/summarise_results.py`](aim2/summarise_results.py) | Builds the Aim 2 results tables straight from the saved outputs. | `processed/aim2/results/results_tables.md` |
| [`aim2/verify_aim2_results.py`](aim2/verify_aim2_results.py) | Independent recheck of the Aim 1 and Aim 2 results with separate code: coverage, determinism, cohort, folds, metrics, DeLong and leakage. | `processed/aim2/results/verification_report.txt` |
| [`aim2/verify_optimisation.py`](aim2/verify_optimisation.py) | Checks each model was tuned properly: fast checks on saved outputs, fold refits and validation curves past the searched ranges. | `processed/aim2/results/optimisation_verification_*.txt` |

### Aim 3: deep learning on the raw signal

Data and training code:

| File | What it does | Main output |
|---|---|---|
| [`aim3/common.py`](aim3/common.py) | Small shared helpers: device choice (MPS or CPU), seeding, masked pooling, parameter counts. | imported module |
| [`aim3/build_signal_store.py`](aim3/build_signal_store.py) | Packs every cleaned recording, mask and stage array into one store that fits in RAM, so training doesn't read thousands of small files off the drive. | `processed/aim3/signal_store/` |
| [`aim3/folds.py`](aim3/folds.py) | Rebuilds the exact Aim 2 outer CV folds and checks them, so both pipelines are compared on the same people. | imported module |
| [`aim3/verify_signal_store.py`](aim3/verify_signal_store.py) | Checks the store against the original files and the Aim 2 biomarker table. | `signal_store/store_verification.csv` |
| [`aim3/data.py`](aim3/data.py) | Turns the store into padded batches for each arm, and holds the sleep structure features and confounder scaling. | imported module |
| [`aim3/models.py`](aim3/models.py) | The CNNs: architecture A, the five minute window model B, the clinical offset wrapper and the NC, NCA and D control arms. | imported module |
| [`aim3/trainer.py`](aim3/trainer.py) | Trains one inner model (BCE, AdamW, warmup plus cosine decay, EMA weights) and saves a checkpoint and validation predictions every epoch. | imported module |
| [`aim3/test_aim3.py`](aim3/test_aim3.py) | Unit tests for the data layer, folds, networks, pooling, training loop and MPS backend. | printed report |

Gates, tuning and training runs:

| File | What it does | Main output |
|---|---|---|
| [`aim3/gates.py`](aim3/gates.py) | Label free gates that have to pass before any CVD label is used (ODI3 regression with r >= 0.9, learning rate check). | `aim3/results/gate_*.json` |
| [`aim3/optimise.py`](aim3/optimise.py) | Label free Optuna search on ODI3 regression in the participants outside every CVD fold. | `aim3/results/optimisation_result.json` |
| [`aim3/confirm_config.py`](aim3/confirm_config.py) | Retrains the search winner and the pre-set config at the full epoch budget and keeps the winner only if it comes out ahead. | `aim3/results/config_confirmation.json` |
| [`aim3/profile_run.py`](aim3/profile_run.py) | Times the loader, GPU and epochs to see where training time goes. One-off check. | printed report |
| [`aim3/run_unit.py`](aim3/run_unit.py) | Trains one unit (arm, repeat, outer fold, inner model) in its own process. | `aim3/results/units/` |
| [`aim3/select_epoch.py`](aim3/select_epoch.py) | Picks the epoch from the three inner models of an outer fold and scores that fold's test set once. | `aim3/results/selected/` |
| [`aim3/run_after_optimisation.sh`](aim3/run_after_optimisation.sh) | Everything after the search: confirm config, classification gates, then `run_all.sh`. | logs |
| [`aim3/run_all.sh`](aim3/run_all.sh) | The whole Aim 3 run, one job at a time, skipping stages already done. | `aim3/results/` |
| [`aim3/run_fast.sh`](aim3/run_fast.sh) | Quicker path that trains and evaluates arm A only, for one repeat. | `aim3/results/` |
| [`aim3/run_stage1.sh`](aim3/run_stage1.sh) | Older standalone version of the training part of `run_all.sh`. | `aim3/results/units/`, `selected/` |
| [`aim3/run_gates_classification.sh`](aim3/run_gates_classification.sh) | Positive control (label = ODI3 >= 15) and permuted label control on one fold, with a pass/fail for each. | logs |
| [`aim3/run_nc.sh`](aim3/run_nc.sh) | Runs the NC negative control arm (sleep mask and REM indicator only, no SpO2). | `aim3/results/` |
| [`aim3/run_nca.sh`](aim3/run_nca.sh) | Runs the NCA arm (architecture A given only the sleep mask and REM indicator). | `aim3/results/` |
| [`aim3/smoke_test.sh`](aim3/smoke_test.sh) | Runs the whole chain at 3 epochs on one fold to check every stage works. | `stack_eval_A_smoke.json` |

Evaluation:

| File | What it does | Main output |
|---|---|---|
| [`aim3/aim2_comparators.py`](aim3/aim2_comparators.py) | Re-runs the Aim 2 logistic regression CV so its per fold settings and predictions are saved, and checks the AUCs match Aim 2. | `aim3/results/aim2_frozen_params_by_fold.csv` |
| [`aim3/stack_eval.py`](aim3/stack_eval.py) | Adds the deep score to the clinical model fold by fold and tests whether it helps (likelihood ratio test, change in AUC against a 0.005 equivalence margin), and compares it with the Aim 2 biomarker model. | `aim3/results/stack_eval_<arm>.json` |
| [`aim3/diagnostics.py`](aim3/diagnostics.py) | What the deep score is picking up: a structure-only nuisance model, correlations with age, BMI and the biomarkers, AUC within subgroups, and the control arms. | `aim3/results/diagnostics_<arm>.json` |
| [`aim3/verify_aim3.py`](aim3/verify_aim3.py) | Re-checks the Aim 3 results from the saved artefacts: leakage, epoch selection, rescoring from saved weights and the metric arithmetic. | `aim3/results/verification_aim3.csv` |
| [`aim3/report.py`](aim3/report.py) | Writes the Aim 3 results tables from the saved outputs. | `AIM3_TABLES.md` |

### Aim 4: interpretability and head to head

| File | What it does | Main output |
|---|---|---|
| [`aim4/common4.py`](aim4/common4.py) | Shared paths, settings and helpers for Aim 4. | imported module |
| [`aim4/run_aim4.sh`](aim4/run_aim4.sh) | Runs all of Aim 4 in order, one heavy job at a time. | `processed/aim4/results/` |
| [`aim4/test_aim4.py`](aim4/test_aim4.py) | Unit tests on small random networks and synthetic nights. Each identity test first checks the network actually responds to its input. | printed report |
| [`aim4/shap_classical.py`](aim4/shap_classical.py) | Out of fold SHAP values for the three classical models and all three feature sets, checking the refits match the Aim 2 AUCs. | `aim4/results/shap_*` |
| [`aim4/deep_attrib.py`](aim4/deep_attrib.py) | Explains the deep model for every participant: exact step decomposition, integrated gradients with two baselines and Grad-CAM, summed by SpO2 band, sleep stage and scored events. | `aim4/results/deep_attrib_A.*` |
| [`aim4/occlusion.py`](aim4/occlusion.py) | Tests the attribution by blanking the top, bottom and random tenth of each night and checking the logit moves as predicted. | `aim4/results/occlusion_A.*` |
| [`aim4/headtohead.py`](aim4/headtohead.py) | Compares the two pipelines: agreement between classical models, collinear biomarkers, how much of the CNN score the biomarkers reproduce, and whether the deep score adds to clinical plus biomarkers. | `aim4/results/headtohead.json` |
| [`aim4/figures_aim4.py`](aim4/figures_aim4.py) | Draws the eight Aim 4 figures from the saved outputs. | `aim4/results/figures/` |
| [`aim4/verify_aim4.py`](aim4/verify_aim4.py) | Checks the Aim 4 outputs against each other and against Aim 3. | `aim4/results/verification_aim4.csv` |
| [`aim4/report_aim4.py`](aim4/report_aim4.py) | Writes the Aim 4 tables from the saved outputs. | `AIM4_TABLES.md` |

### Aim 5: 1, 5 and 10 year risk (`horizon_risk/`)

| File | What it does | Main output |
|---|---|---|
| [`horizon_risk/horizon_lib.py`](horizon_risk/horizon_lib.py) | The maths: time and cause of first event, censoring models, IPCW labels and weights, cause specific Cox, weighted metrics, Aalen Johansen and the Riley sample size criteria. No file access. | imported module |
| [`horizon_risk/test_horizon_lib.py`](horizon_risk/test_horizon_lib.py) | Tests the library on simulated cohorts with a known true risk. | printed pass list |
| [`horizon_risk/fit_horizon_models.py`](horizon_risk/fit_horizon_models.py) | Refits the Aim 2 logistic regression at 1, 5 and 10 years on the same folds with IPCW, plus a cause specific Cox comparator and the sensitivity arms. | `processed/horizon_risk/oof_<arm>.npz` |
| [`horizon_risk/fit_recalibrated_top_model.py`](horizon_risk/fit_recalibrated_top_model.py) | Keeps the Aim 2 model as trained and only learns an intercept and slope per horizon. | `horizon_risk/results/recalibration_*` |
| [`horizon_risk/evaluate_horizon_models.py`](horizon_risk/evaluate_horizon_models.py) | Scores every model at each horizon (within fold AUC, calibration, Brier and IPA, net benefit, risk bands) with bootstrap CIs, and fits the final risk equations. | `horizon_risk/results/metrics_primary.csv`, `risk_equations.json` |
| [`horizon_risk/extended_analyses.py`](horizon_risk/extended_analyses.py) | Follow-up checks: subgroup calibration, holding out each parent group, person level stability, learning curve and the effect of the corrected label on Aims 2 to 4. | `horizon_risk/results/` |
| [`horizon_risk/framing_checks.py`](horizon_risk/framing_checks.py) | A few extra numbers the thesis quotes, like reconciling the earlier 10 year AUC with this one and why the pooled AUC is biased. | `horizon_risk/results/reconciliation*.csv` |
| [`horizon_risk/whole_pipeline_bootstrap.py`](horizon_risk/whole_pipeline_bootstrap.py) | Refits a cut-down pipeline on 400 bootstrap samples to see how much training variance the fixed model CIs leave out. | `horizon_risk/results/whole_pipeline_bootstrap.csv` |
| [`horizon_risk/make_horizon_figures.py`](horizon_risk/make_horizon_figures.py) | Draws the Aim 5 figures (AUC by horizon, calibration, net benefit). | `horizon_risk/results/figures/` |

### Thesis tables and figures (`thesis_report/`)

| File | What it does | Main output |
|---|---|---|
| [`thesis_report/facts/aim2_data_checks.py`](thesis_report/facts/aim2_data_checks.py) | Read-only checks behind the Aim 2 data facts quoted in the thesis (missingness, pruning, agreement with SHHS values, SpO2 quantisation, follow-up). | printed report |
| [`thesis_report/scripts/table1_baseline.py`](thesis_report/scripts/table1_baseline.py) | Baseline characteristics table (Table 1) for the 4,352 analysed participants, split by incident event. | `thesis_report/facts/table1_baseline.csv` |
| [`thesis_report/scripts/derived_analyses.py`](thesis_report/scripts/derived_analyses.py) | Extra analyses added while writing up: repeat-matched sleep structure model, equivalence margin check, Cox and fixed 10 year versions, and how much of the deep score the biomarkers reproduce. | `thesis_report/derived/derived_results.json` |
| [`thesis_report/scripts/sensitivity_features.py`](thesis_report/scripts/sensitivity_features.py) | How much the logistic regression changes without the 600 s constant value rule (S1) or with desaturation thresholds lowered by 0.1 (S2). | `thesis_report/derived/sensitivity_results.json` |
| [`thesis_report/scripts/make_tables.py`](thesis_report/scripts/make_tables.py) | Chapter 4 LaTeX tables (baseline and sensitivity). | `latex/tables/tab_baseline.tex`, `tab_sensitivity.tex` |
| [`thesis_report/scripts/make_figures.py`](thesis_report/scripts/make_figures.py) | Chapter 4 figures: cleaning example, ROC and PR, calibration and decision curves, forest plot, SHAP and deep attribution. | `latex/figures/*.pdf` |
| [`thesis_report/scripts/make_aim5_tables.py`](thesis_report/scripts/make_aim5_tables.py) | All the Aim 5 LaTeX tables. | `latex/tables/tab_aim5_*.tex` |
| [`thesis_report/scripts/make_appendix.py`](thesis_report/scripts/make_appendix.py) | Appendix A, including the 34 feature table. | `latex/chapters/appendix.tex` |

## Tests

```bash
python3 aim2/test_obm_library.py          # synthetic signals, no data needed
python3 horizon_risk/test_horizon_lib.py  # simulated cohorts, no data needed
python3 aim3/test_aim3.py                 # needs the signal store and Aim 2 outputs
python3 aim4/test_aim4.py                 # needs the Aim 2 and Aim 3 outputs
```
