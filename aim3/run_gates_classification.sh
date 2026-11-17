#!/bin/bash
# Aim 3: the two gates that test the classification path, without the real outcome.
#
# The learnability gate (gates.py --gate odi3) shows the network can read
# oximetry from the waveform, but it's a regression and doesn't run any of the
# classification code: the epoch selection rules, the cross fitted stacking or
# the held out coefficient test. A subtle bug there would do the most damage and
# be the hardest to spot, because a leak just makes a result look better.
#
#   Positive control. The whole path, end to end, against a label derived from
#   the signal itself: whether sleep ODI3 is at least 15 (moderate to severe).
#   The network should recover this strongly. If it doesn't, the classification,
#   selection or scoring code is broken, and we need to know that before
#   trusting any real result.
#
#   Permuted label control. The same path against CVD labels permuted with a
#   fixed seed. Every stage, including the stacking and the held out test, is
#   scored against the permuted labels, so no real association is ever seen.
#   Expected: AUC near 0.5 and an increment inside the noise band. This checks
#   the pipeline doesn't make up signal from nothing, which cross validation on
#   its own can't show.
#
# Both use one outer fold (enough to run every code path) and arm A, so the model
# is the real one.
#
# Usage: ./run_gates_classification.sh

set -u
cd "$(dirname "$0")"
R="/Volumes/ThesisHD/Data/processed/aim3/results"
LOG="$R/logs"
mkdir -p "$LOG"

run_gate() {
  local name="$1"; local tag="$2"; local flag="$3"
  echo ""
  echo "======================================================================"
  echo "$name   $(date '+%H:%M:%S')"
  echo "======================================================================"
  for inner in 0 1 2; do
    if [ -e "$R/units/A_r0_f0_i${inner}${tag}/meta.json" ]; then
      echo "[skip] unit A_r0_f0_i${inner}${tag}"
      continue
    fi
    caffeinate -dimsu python3 -u run_unit.py --arm A --repeat 0 --fold 0 \
      --inner "$inner" --tag "$tag" $flag \
      2>&1 | tee "$LOG/gate_${tag}_i${inner}.log" \
      | grep -E "label|epoch|done|backend" | tail -6
  done
  if [ ! -e "$R/selected/A_r0_f0${tag}.npz" ]; then
    caffeinate -dimsu python3 -u select_epoch.py --arm A --repeat 0 --fold 0 \
      --tag "$tag" $flag \
      2>&1 | tee "$LOG/gate_${tag}_select.log" \
      | grep -E "E\*|pooled|wrote|scoring" | tail -8
  fi
}

run_gate "POSITIVE CONTROL: label is 1[sleep ODI3 >= 15]" "_pos" "--odi3-label"
run_gate "PERMUTED LABEL CONTROL: cardiovascular labels permuted" "_perm" "--permute-labels"

echo ""
echo "======================================================================"
echo "GATE VERDICTS   $(date '+%H:%M:%S')"
echo "======================================================================"
python3 -u - <<'PY'
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
R = "/Volumes/ThesisHD/Data/processed/aim3/results"

for tag, name, expect in (
        ("_pos", "positive control (ODI3 >= 15)",
         "the deep score should recover this strongly, AUC well above 0.8"),
        ("_perm", "permuted cardiovascular labels",
         "the deep score should be near chance, AUC about 0.5")):
    f = os.path.join(R, "selected", f"A_r0_f0{tag}.npz")
    if not os.path.exists(f):
        print(f"\n{name}: NOT RUN")
        continue
    z = np.load(f, allow_pickle=True)
    y = z["test_y"].astype(int)
    auc_m = roc_auc_score(y, z["test_z_marg"]) if len(set(y.tolist())) > 1 else float("nan")
    auc_i = roc_auc_score(y, z["test_z_inc"]) if len(set(y.tolist())) > 1 else float("nan")
    tab = z["selection_table"]
    print(f"\n{name}")
    print(f"  expectation      : {expect}")
    print(f"  held out test AUC: {auc_m:.4f} at E*_marg (epoch {int(z['e_marg'])})")
    print(f"                     {auc_i:.4f} at E*_inc  (epoch {int(z['e_inc'])})")
    print(f"  test events      : {int(y.sum())} of {len(y)} "
          f"({100 * y.mean():.1f} percent)")
    best_pooled = float(np.nanmax(tab["pooled_auc"]))
    print(f"  best pooled inner validation AUC: {best_pooled:.4f}")
    if tag == "_pos":
        verdict = "PASS" if auc_m > 0.80 else "FAIL"
        note = ("the classification, selection and scoring path works"
                if verdict == "PASS" else
                "the path cannot recover a label that is a function of its own "
                "input, so it is broken")
    else:
        verdict = "PASS" if abs(auc_m - 0.5) < 0.05 else "FAIL"
        note = ("the pipeline does not manufacture signal from nothing"
                if verdict == "PASS" else
                "the pipeline finds signal in permuted labels, which means a leak")
    print(f"  VERDICT: {verdict}. {note}")
PY
