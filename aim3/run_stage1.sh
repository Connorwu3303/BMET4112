#!/bin/bash
# Aim 3 stage 1: trains every unit, one at a time, and can be resumed.
#
# It's a shell loop with one process per unit, for three reasons that all came up
# on this machine:
#
#   Memory. This is an 8 GB machine. Running two heavy jobs at once got a
#   training process killed by the OS with no error message and a zero byte log,
#   so it's one job at a time.
#
#   Resuming. The external drive has unmounted mid run before. Each unit checks
#   whether its own output is already complete and skips itself if so, so the
#   script can be stopped and restarted at any point and carries on from there.
#
#   Fresh allocator. Building several models in one process slows down badly on
#   this backend.
#
# Repeat 0 runs first for every arm. One repeat already scores all 4,352
# participants, so after repeat 0 there's a complete result for every primary
# contrast and repeats 1 and 2 only add precision. If the machine has to stop
# early, what's already done is usable.
#
# The first argument is the repeat list (quote it for more than one), the rest
# are the arms.
#
# Usage:
#   ./run_stage1.sh 0            # repeat 0, all arms
#   ./run_stage1.sh 0 A          # repeat 0, primary arm only
#   ./run_stage1.sh "1 2"        # repeats 1 and 2, all arms

set -u
cd "$(dirname "$0")"

REPEATS=${1:-0}
shift || true
ARMS=${*:-"A B NC"}

LOG_DIR="/Volumes/ThesisHD/Data/processed/aim3/results/logs"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "AIM 3 STAGE 1"
echo "repeats: $REPEATS"
echo "arms   : $ARMS"
echo "started: $(date)"
echo "======================================================================"

for repeat in $REPEATS; do
  for arm in $ARMS; do
    for fold in 0 1 2 3 4; do
      for inner in 0 1 2; do
        tag="${arm}_r${repeat}_f${fold}_i${inner}"
        echo ""
        echo "---- $tag  $(date '+%H:%M:%S') ----"
        # caffeinate only for the length of this unit, so an interrupted run
        # doesn't leave a sleep assertion held
        caffeinate -dimsu python3 -u run_unit.py \
          --arm "$arm" --repeat "$repeat" --fold "$fold" --inner "$inner" \
          2>&1 | tee "$LOG_DIR/${tag}.log" | grep -E "epoch|done|already|param|inner|label"
        status=${PIPESTATUS[0]}
        if [ "$status" -ne 0 ]; then
          echo "UNIT $tag FAILED with status $status"
          echo "see $LOG_DIR/${tag}.log"
          # keep going. One failed unit shouldn't stop the run; select_epoch.py
          # will refuse that fold and say which unit is missing
        fi
      done
    done

    # epoch selection and test scoring for each fold of this arm
    for fold in 0 1 2 3 4; do
      echo ""
      echo "---- selecting $arm r$repeat f$fold  $(date '+%H:%M:%S') ----"
      caffeinate -dimsu python3 -u select_epoch.py \
        --arm "$arm" --repeat "$repeat" --fold "$fold" \
        2>&1 | tee "$LOG_DIR/select_${arm}_r${repeat}_f${fold}.log" \
        | grep -E "E\*|epoch|wrote|cross fitted|skipping"
    done
  done
done

echo ""
echo "======================================================================"
echo "STAGE 1 COMPLETE: $(date)"
echo "next: python3 stack_eval.py --arm A"
echo "======================================================================"
