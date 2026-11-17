#!/bin/bash
# Runs all of Aim 3, one job at a time. Each stage skips itself if its output
# already exists, so it's safe to kill and restart. Two heavy jobs running at once
# ran out of memory before, which is why nothing runs in parallel.
#
# Arm order is A, NC, then B. A is the main model and NC is the negative control
# (sleep structure only, no oximetry), so if the run has to stop early the two
# arms the main result needs are already done. B is a secondary architecture
# comparison.
#
# Usage:
#   ./run_all.sh              # everything, repeat 0
#   ./run_all.sh 1 2          # add repeats 1 and 2 later

set -u
cd "$(dirname "$0")"

R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"
mkdir -p "$LOG"
REPEATS=${*:-0}

banner() {
  echo ""
  echo "======================================================================"
  echo "$1   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "======================================================================"
}

# run a stage unless its sentinel output file already exists
stage() {
  local name="$1"; local sentinel="$2"; shift 2
  if [ -e "$sentinel" ]; then
    echo "[skip] $name (found $(basename "$sentinel"))"
    return 0
  fi
  banner "$name"
  caffeinate -dimsu "$@" 2>&1 | tee "$LOG/${name}.log" | tail -40
  local st=${PIPESTATUS[0]}
  if [ "$st" -ne 0 ]; then
    echo "[WARN] $name exited $st; see $LOG/${name}.log"
  fi
  return 0
}

banner "AIM 3 FULL RUN. repeats: $REPEATS"
echo "machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"
echo "one heavy job at a time, by design"

# --- stage 0: unit tests and the matched Aim 2 comparators ---
banner "unit tests"
caffeinate -dimsu python3 -u test_aim3.py 2>&1 | tee "$LOG/unit_tests.log" | tail -12
if ! grep -q "0 failed" "$LOG/unit_tests.log"; then
  echo ""
  echo "UNIT TESTS FAILED. Stopping: nothing downstream is trustworthy."
  exit 1
fi

stage "aim2_comparators" "$R/aim2_frozen_params_by_fold.csv" \
  python3 -u aim2_comparators.py

if [ ! -e "$R/aim2_frozen_params_by_fold.csv" ]; then
  echo "The Aim 2 comparators did not rebuild. The stacker cannot freeze its"
  echo "hyperparameters, so C2 would not be 'one added column'. Stopping."
  exit 1
fi

# --- stage 0b: tiny end to end smoke test, then the long label free gate ---
stage "smoke_test" "$R/stack_eval_A_smoke.json" ./smoke_test.sh

stage "gate_odi3_long" "$R/gate_odi3_long.json" \
  python3 -u gates.py --gate odi3 --epochs 26
# gates.py writes gate_odi3.json, so copy it to gate_odi3_long.json. That marks
# the stage as done and keeps both results.
if [ -e "$R/gate_odi3.json" ] && [ ! -e "$R/gate_odi3_long.json" ]; then
  cp "$R/gate_odi3.json" "$R/gate_odi3_long.json"
fi

# --- stage 1: train every unit, then select and evaluate each arm ---
for repeat in $REPEATS; do
  for arm in A NC B; do
    for fold in 0 1 2 3 4; do
      for inner in 0 1 2; do
        tag="${arm}_r${repeat}_f${fold}_i${inner}"
        if [ -e "$R/units/$tag/meta.json" ]; then
          echo "[skip] unit $tag"
          continue
        fi
        banner "unit $tag"
        caffeinate -dimsu python3 -u run_unit.py \
          --arm "$arm" --repeat "$repeat" --fold "$fold" --inner "$inner" \
          2>&1 | tee "$LOG/${tag}.log" \
          | grep -E "epoch|done|wrote|label|inner|model" | tail -20
      done
    done

    for fold in 0 1 2 3 4; do
      sel="$R/selected/${arm}_r${repeat}_f${fold}.npz"
      if [ -e "$sel" ]; then
        echo "[skip] selection ${arm} r${repeat} f${fold}"
        continue
      fi
      banner "select ${arm} r${repeat} f${fold}"
      caffeinate -dimsu python3 -u select_epoch.py \
        --arm "$arm" --repeat "$repeat" --fold "$fold" \
        2>&1 | tee "$LOG/select_${arm}_r${repeat}_f${fold}.log" \
        | grep -E "E\*|wrote|cross fitted|scoring" | tail -12
    done

    # evaluate after each arm, so there's a usable result early even if the run
    # gets interrupted later
    banner "evaluate arm $arm"
    caffeinate -dimsu python3 -u stack_eval.py --arm "$arm" \
      2>&1 | tee "$LOG/stack_eval_${arm}.log" | tail -45
  done
done

# --- stage 2: diagnostics, verification, report ---
banner "diagnostics"
caffeinate -dimsu python3 -u diagnostics.py --arm A \
  2>&1 | tee "$LOG/diagnostics_A.log" | tail -40

banner "independent verification"
caffeinate -dimsu python3 -u verify_aim3.py --arm A \
  2>&1 | tee "$LOG/verify_aim3.log" | tail -45

banner "results tables"
caffeinate -dimsu python3 -u report.py --arm A \
  2>&1 | tee "$LOG/report.log" | tail -20

banner "AIM 3 RUN COMPLETE"
echo "results  : $R"
echo "tables   : $(cd .. && pwd)/AIM3_TABLES.md"
