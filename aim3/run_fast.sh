#!/bin/bash
# Aim 3 fast path: arm A, one repeat, fully evaluated. The quickest way to a
# working model with real performance numbers.
#
# Left out: the negative control arm, the window model arm, repeats 1 and 2, and
# the hyperparameter search. Those add interpretation and precision but aren't
# needed for the numbers to be correct, and any of them can be added later
# without redoing anything here.
#
# Kept, because the numbers wouldn't mean what they say without them:
#   - the exact Aim 2 outer folds, so the comparison with the classical models is
#     paired on the same participants
#   - three inner models per fold, since the stacker is fitted on their cross
#     fitted scores; using one would break the fusion design
#   - the 16 epoch cap and the pooled two criterion epoch selection
#   - the frozen Aim 2 hyperparameters in the stacker, so the main contrast is
#     the Aim 2 model plus exactly one column
#   - scoring the outer test fold once, after the epoch is fixed
#
# One repeat still scores all 4,352 participants, so every primary contrast is
# complete. Repeats 2 and 3 would narrow the intervals without changing the answer.
#
# A short chain validation runs first (three units of 5 epochs through selection
# and evaluation, about 10 minutes) so a three hour run doesn't fall over at the
# last step.
#
# Usage: ./run_fast.sh

set -u
cd "$(dirname "$0")"
R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"
mkdir -p "$LOG" "$R/units" "$R/selected"

banner() {
  echo ""
  echo "======================================================================"
  echo "$1   $(date '+%H:%M:%S')"
  echo "======================================================================"
}

if [ -e "$R/stack_eval_A_smoke.json" ]; then
  banner "CHAIN VALIDATION already passed, skipping"
else
banner "CHAIN VALIDATION (5 epochs, meaningless numbers, proves the plumbing)"
for inner in 0 1 2; do
  caffeinate -dimsu python3 -u run_unit.py --arm A --repeat 0 --fold 0 \
    --inner "$inner" --epochs 5 --tag _smoke --force \
    2>&1 | tee "$LOG/smoke_i${inner}.log" | grep -E "config|epoch|done|backend"
done
caffeinate -dimsu python3 -u select_epoch.py --arm A --repeat 0 --fold 0 \
  --epochs 5 --tag _smoke --force 2>&1 | tee "$LOG/smoke_select.log" \
  | grep -E "E\*|pooled|cross fitted|wrote|scoring"
python3 -u stack_eval.py --arm A --tag _smoke 2>&1 \
  | tee "$LOG/smoke_stack.log" | tail -25

if [ ! -e "$R/stack_eval_A_smoke.json" ]; then
  echo ""
  echo "CHAIN VALIDATION FAILED. Not starting the three hour run."
  echo "See $LOG/smoke_stack.log"
  exit 1
fi
echo ""
echo "chain validation passed: every stage produced output"
fi

banner "TRAINING ARM A, REPEAT 0 (15 units, about 3.4 hours)"
for fold in 0 1 2 3 4; do
  for inner in 0 1 2; do
    tag="A_r0_f${fold}_i${inner}"
    if [ -e "$R/units/$tag/meta.json" ]; then
      echo "[skip] $tag"
      continue
    fi
    echo ""
    echo "---- $tag  $(date '+%H:%M:%S') ----"
    caffeinate -dimsu python3 -u run_unit.py --arm A --repeat 0 \
      --fold "$fold" --inner "$inner" \
      2>&1 | tee "$LOG/${tag}.log" \
      | grep -E "config|inner|model|epoch 1[0-9]|done|backend"
  done
done

banner "EPOCH SELECTION AND TEST SCORING"
for fold in 0 1 2 3 4; do
  if [ -e "$R/selected/A_r0_f${fold}.npz" ]; then
    echo "[skip] selection fold $fold"
    continue
  fi
  caffeinate -dimsu python3 -u select_epoch.py --arm A --repeat 0 --fold "$fold" \
    2>&1 | tee "$LOG/select_A_r0_f${fold}.log" \
    | grep -E "E\*|wrote|cross fitted|scoring"
done

banner "PERFORMANCE"
caffeinate -dimsu python3 -u stack_eval.py --arm A 2>&1 \
  | tee "$LOG/stack_eval_A.log" | tail -60

banner "DIAGNOSTICS: is it oximetry, or age and sleep structure?"
caffeinate -dimsu python3 -u diagnostics.py --arm A 2>&1 \
  | tee "$LOG/diagnostics_A.log" | tail -40

banner "INDEPENDENT VERIFICATION"
caffeinate -dimsu python3 -u verify_aim3.py --arm A 2>&1 \
  | tee "$LOG/verify_aim3.log" | tail -50

banner "RESULTS TABLES"
python3 -u report.py --arm A 2>&1 | tee "$LOG/report.log" | tail -15

banner "FAST PATH COMPLETE"
echo "tables: $(cd .. && pwd)/AIM3_TABLES.md"
