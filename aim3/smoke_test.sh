#!/bin/bash
# Aim 3: runs the whole chain end to end at a tiny scale before the long runs.
#
# The chain is run_unit -> select_epoch -> stack_eval -> diagnostics, and a bug in
# the last step would only show up after every unit had been trained. A 3 epoch
# run of one outer fold goes through every interface: the training loop, the per
# epoch checkpoints, the pooled epoch selection, the test scoring, the frozen
# hyperparameter stacking and the held out coefficient test.
#
# The numbers it prints mean nothing. A 3 epoch network has learned almost
# nothing; this only checks that values get carried through every stage. Outputs
# are tagged "_smoke" so they can't be mixed up with the real run.
#
# Usage: ./smoke_test.sh

set -eu
cd "$(dirname "$0")"
EPOCHS=3
TAG="_smoke"

echo "======================================================================"
echo "AIM 3 CHAIN SMOKE TEST: 3 epochs, one outer fold, arm A"
echo "======================================================================"

for inner in 0 1 2; do
  echo ""
  echo "---- training inner model $inner ----"
  caffeinate -dimsu python3 -u run_unit.py --arm A --repeat 0 --fold 0 \
    --inner "$inner" --epochs "$EPOCHS" --tag "$TAG" --force
done

echo ""
echo "---- selecting the epoch and scoring the test fold ----"
caffeinate -dimsu python3 -u select_epoch.py --arm A --repeat 0 --fold 0 \
  --epochs "$EPOCHS" --tag "$TAG" --force

echo ""
echo "---- fusion and evaluation ----"
python3 -u stack_eval.py --arm A --tag "$TAG"

echo ""
echo "======================================================================"
echo "SMOKE TEST COMPLETE"
echo "The numbers above are meaningless: 3 epochs of training on one fold."
echo "What matters is that every stage ran and produced finite values."
echo "======================================================================"
