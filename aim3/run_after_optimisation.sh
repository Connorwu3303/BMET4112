#!/bin/bash
# Aim 3: everything after the label free optimisation, in order.
#
# 1. If the adopted config differs from the pre-registered one, check it at 16
#    epochs (confirm_config.py) and, if it still differs, rerun the learnability
#    gate with it so the gate figure in the thesis describes the model that
#    produced the results. Skipped if the optimisation kept the pre-registered
#    config, since the existing gate already covers it.
# 2. The classification gates (positive control and permuted labels). These test
#    the selection, stacking and held out testing code, which the regression gate
#    doesn't touch. They run before the real outcome is modelled, because a leak
#    found afterwards would already have been seen.
# 3. The full run (run_all.sh): smoke test, every unit, evaluation, diagnostics,
#    independent verification and the tables.
#
# One heavy job at a time throughout, and every stage can be resumed.
#
# Usage: ./run_after_optimisation.sh

set -u
cd "$(dirname "$0")"
R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"
mkdir -p "$LOG"

echo "======================================================================"
echo "AIM 3 POST-OPTIMISATION SEQUENCE   $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# --- 1. check whether the adopted config differs from the pre-registered one ---
CHANGED=$(python3 - <<'PY'
import json, os, sys
sys.path.insert(0, ".")
import run_unit as RU
cfg, src = RU.resolve_config()
print("yes" if cfg != RU.PREREGISTERED else "no")
PY
)
echo ""
echo "configuration adopted differs from pre-registered: $CHANGED"
python3 - <<'PY'
import sys; sys.path.insert(0, ".")
import run_unit as RU, json
cfg, src = RU.resolve_config()
print(f"  source: {src}")
print(f"  config: {json.dumps(cfg)}")
PY

if [ "$CHANGED" = "yes" ]; then
  echo ""
  echo "--- confirming the choice at the production epoch budget ---"
  # The search used 10 epoch trials and the real runs use 16, and a cosine
  # schedule over 10 epochs decays differently from the first 10 epochs of a 16
  # epoch one. confirm_config.py reverts the choice if its edge doesn't hold up.
  caffeinate -dimsu python3 -u confirm_config.py --epochs 16     2>&1 | tee "$LOG/confirm_config.log" | tail -14
  CHANGED=$(python3 - <<'PY2'
import sys; sys.path.insert(0, ".")
import run_unit as RU
cfg, src = RU.resolve_config()
print("yes" if cfg != RU.PREREGISTERED else "no")
PY2
)
  echo "  after confirmation, configuration still differs: $CHANGED"
fi

if [ "$CHANGED" = "yes" ]; then
  echo ""
  echo "--- re-establishing learnability with the adopted configuration ---"
  rm -f "$R/gate_odi3_long.json"
  caffeinate -dimsu python3 -u gates.py --gate odi3 --epochs 26 \
    2>&1 | tee "$LOG/gate_odi3_adopted.log" | tail -8
  cp "$R/gate_odi3.json" "$R/gate_odi3_long.json" 2>/dev/null || true
else
  echo "  the existing learnability gate already describes this configuration"
fi

# --- 2. classification gates ---
echo ""
echo "--- classification gates (no real outcome is modelled) ---"
./run_gates_classification.sh 2>&1 | tee "$LOG/gates_classification.log" \
  | grep -E "CONTROL|VERDICT|expectation|held out|test events|pooled|====" | tail -30

# --- 3. the full run ---
echo ""
echo "--- definitive run ---"
./run_all.sh 0

echo ""
echo "======================================================================"
echo "AIM 3 COMPLETE   $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"
