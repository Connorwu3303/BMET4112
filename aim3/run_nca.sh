#!/bin/bash
# Aim 3: the architecture matched negative control arm (NCA).
#
# Same question as run_nc.sh. The primary network reads one channel (SpO2) and
# gets AUC 0.6146 on its own, and a plain logistic regression on nine sleep
# structure and data quality variables gets 0.6138 on the same folds. Those are
# indistinguishable, so there's no way to tell whether the primary arm's
# standalone discrimination comes from oximetry or from how fragmented the night
# was and how much of it was measurable.
#
# NC uses the smaller window model, so a gap between NC and arm A mixes up the
# missing SpO2 with the smaller network. NCA is architecture A itself, same folds
# and training, given only the sleep mask and REM indicator and no SpO2, so the
# only difference from arm A is what the network can see.
#
# Usage: ./run_nca.sh
set -u
cd "$(dirname "$0")"
R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"; mkdir -p "$LOG"

echo "======================================================================"
echo "ARCHITECTURE MATCHED NEGATIVE CONTROL: same network as arm A, mask and REM only"
echo "started $(date '+%H:%M:%S')"
echo "======================================================================"
for fold in 0 1 2 3 4; do
  for inner in 0 1 2; do
    tag="NCA_r0_f${fold}_i${inner}"
    [ -e "$R/units/$tag/meta.json" ] && { echo "[skip] $tag"; continue; }
    echo ""; echo "---- $tag  $(date '+%H:%M:%S') ----"
    caffeinate -dimsu python3 -u run_unit.py --arm NCA --repeat 0 \
      --fold "$fold" --inner "$inner" 2>&1 | tee "$LOG/${tag}.log" \
      | grep -E "config|inner|model|done|backend"
  done
done
for fold in 0 1 2 3 4; do
  [ -e "$R/selected/NCA_r0_f${fold}.npz" ] && { echo "[skip] select f$fold"; continue; }
  caffeinate -dimsu python3 -u select_epoch.py --arm NCA --repeat 0 --fold "$fold" \
    2>&1 | tee "$LOG/select_NCA_r0_f${fold}.log" | grep -E "E\*|wrote|scoring"
done
echo ""; echo "--- evaluating the negative control ---"
caffeinate -dimsu python3 -u stack_eval.py --arm NCA 2>&1 | tee "$LOG/stack_eval_NCA.log" | tail -30
echo ""; echo "--- rerunning diagnostics, which now include the control ---"
caffeinate -dimsu python3 -u diagnostics.py --arm A 2>&1 | tee "$LOG/diagnostics_A.log" | tail -25
echo ""; echo "--- regenerating tables ---"
python3 -u report.py --arm A > /dev/null 2>&1
python3 -u verify_aim3.py --arm A 2>&1 | tail -5
echo ""
echo "ARCHITECTURE MATCHED CONTROL COMPLETE $(date '+%H:%M:%S')"
