#!/bin/bash
# Runs all of Aim 4 in dependency order.
#
# Only one heavy job runs at a time. On this 8 GB machine, two heavy jobs at once
# got a training process killed by the OS with no error and a zero byte log,
# which cost a day of work in Aim 3. So each stage finishes before the next one
# starts, and set -e stops the script at the first failure so a broken output
# never gets passed along.
#
# From scratch it takes about five and a half hours: two hours recovering the
# forest's hyperparameters, two hours of deep attribution, and the rest spread
# over the other stages. With the caches in place a re-run is about three hours.
set -e
cd "$(dirname "$0")"

echo "=== 1. unit tests (about 1 min) ==="
python3 test_aim4.py

echo "=== 2a. recover the forest's per fold hyperparameters (about 2 h, cached) ==="
# Skips itself once the tables exist. It's a separate stage because it's the
# most expensive classical step. Check B3 in verify_aim4.py showed it's needed:
# the fixed forest configuration was statistically distinguishable from the
# published tuned forest.
python3 shap_classical.py --model random_forest --recover-rf

echo "=== 2b. classical SHAP, three models and three specifications (about 25 min) ==="
python3 shap_classical.py --all

echo "=== 3. deep attribution over the whole cohort (about 2 h) ==="
python3 deep_attrib.py

echo "=== 4. the random weights control (about 25 min) ==="
# a tenth of the cohort is enough to compare two profiles, and this control is
# only used for that comparison
python3 deep_attrib.py --random-weights --limit 450

echo "=== 5. the occlusion intervention test (about 20 min) ==="
python3 occlusion.py --n 300

echo "=== 6. head to head, including contrast C3 (about 2 min) ==="
python3 headtohead.py

echo "=== 7. verification ==="
python3 verify_aim4.py

echo "=== 8. regenerate the tables ==="
python3 report_aim4.py

echo "=== 9. figures ==="
python3 figures_aim4.py

echo "=== Aim 4 complete ==="
