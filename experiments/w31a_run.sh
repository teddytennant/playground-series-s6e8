#!/usr/bin/env bash
# w31a -- the §K follow-on (R-K3): 5 folds x 3 configs x 4 stages x 2 CT arms, 2000 rounds.
# Pre-registered at experiments/w31_prereg_slot2.txt §N before this was started.
#
# ONE PROCESS PER FOLD, --jobs 3 each = 15 of 16 threads. Per-(config,fold) checkpointed in
# cache/lgb5fckpt/ (booster .txt + prediction cube .npz), so an identical re-run resumes and
# refits nothing already on disk. GRID order is control -> leaves255_dinf -> lr0.05, i.e. the
# two harness gates (N2a/N2b) land first and the expensive 255-leaf config second: a kill at
# any point costs the least informative cell.
set -u
cd "$(dirname "$0")/.."
for f in 0 1 2 3 4; do
  nohup nice -n 15 .venv/bin/python experiments/w31a_lgb5f.py --fold "$f" \
      --rounds 2000 --seed 42 --jobs 3 > "experiments/w31a_f${f}.log" 2>&1 &
done
wait
echo "=== all folds done; now: .venv/bin/python experiments/w31a_lgb5f.py --report ==="
