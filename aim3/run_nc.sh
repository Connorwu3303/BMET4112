#!/bin/bash
# Aim 3: the negative control arm (NC).
#
# This arm decides how the headline result gets described. The primary network
# reads one channel (SpO2) and gets AUC 0.6146 on its own. A plain logistic
# regression on nine sleep structure and data quality variables gets 0.6138 on
# the same folds. Those are indistinguishable, so there's no way to tell whether
# the primary arm's standalone discrimination comes from oximetry or from how
# fragmented the night was and how much of it was measurable.
#
# The nuisance model only gets at that from the tabular side. This arm tests it
# directly: the same class of network, same folds, same training, given the sleep
# mask and REM indicator and no SpO2 at all. Whatever it gets is an upper bound on
# what sleep structure and missingness alone can predict, this time learned by a
# network.
#
# Usage: ./run_nc.sh
set -u
cd "$(dirname "$0")"
R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"; mkdir -p "$LOG"

echo "======================================================================"
echo "NEGATIVE CONTROL ARM: sleep mask and REM only, no saturation"
echo "started $(date '+%H:%M:%S')"
echo "======================================================================"
for fold in 0 1 2 3 4; do
  for inner in 0 1 2; do
    tag="NC_r0_f${fold}_i${inner}"
    [ -e "$R/units/$tag/meta.json" ] && { echo "[skip] $tag"; continue; }
    echo ""; echo "---- $tag  $(date '+%H:%M:%S') ----"
    caffeinate -dimsu python3 -u run_unit.py --arm NC --repeat 0 \
      --fold "$fold" --inner "$inner" 2>&1 | tee "$LOG/${tag}.log" \
      | grep -E "config|inner|model|done|backend"
  done
done
for fold in 0 1 2 3 4; do
  [ -e "$R/selected/NC_r0_f${fold}.npz" ] && { echo "[skip] select f$fold"; continue; }
  caffeinate -dimsu python3 -u select_epoch.py --arm NC --repeat 0 --fold "$fold" \
    2>&1 | tee "$LOG/select_NC_r0_f${fold}.log" | grep -E "E\*|wrote|scoring"
done
echo ""; echo "--- evaluating the negative control ---"
caffeinate -dimsu python3 -u stack_eval.py --arm NC 2>&1 | tee "$LOG/stack_eval_NC.log" | tail -30
echo ""; echo "--- rerunning diagnostics, which now include the control ---"
caffeinate -dimsu python3 -u diagnostics.py --arm A 2>&1 | tee "$LOG/diagnostics_A.log" | tail -25
echo ""; echo "--- regenerating tables ---"
python3 -u report.py --arm A > /dev/null 2>&1
python3 -u verify_aim3.py --arm A 2>&1 | tail -5
echo ""
echo "NEGATIVE CONTROL COMPLETE $(date '+%H:%M:%S')"
