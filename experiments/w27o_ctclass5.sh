#!/usr/bin/env bash
# w27o -- the cross-class CT scorecard on ALL FIVE folds, not just fold 0.
# Pre-registered at experiments/w27_prereg_slot4.txt SS M4.
#
# ORDER MATTERS. The LightGBM cell costs ~600s of fit against ~57s (xgb) / ~37s (cat),
# and lgb is only the CONTROL -- w27g already holds its fold-0 number. So the cheap,
# genuinely new xgb/cat cells for folds 1-4 run FIRST and the lgb sweep runs last:
# a kill at any point then costs the control, not the result.
#
# Per-(class,fold) checkpointed in cache/ctclassckpt/: an identical re-run resumes and
# refits nothing already on disk (fold 0 xgb/cat land there from w27j).
# niced hard because five other jobs are on the critical path.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/w27o_ctclass5.log
: > "$LOG"
run () {   # run <classes> <fold>
  echo "=== classes=$1 fold=$2 ===" >> "$LOG"
  nice -n 15 .venv/bin/python experiments/w27j_ctclass.py \
      --name ctclass --classes "$1" --fold "$2" \
      --rounds 400 --seed 42 --threads 3 >> "$LOG" 2>&1
}
for f in 1 2 3 4; do run xgb,cat "$f"; done      # the result
for f in 0 1 2 3 4; do run lgb "$f"; done        # the control / harness check
echo "=== all cells done ===" >> "$LOG"
