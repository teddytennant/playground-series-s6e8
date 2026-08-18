#!/usr/bin/env bash
# One job at a time -- w25 §8 triple-booked 16 cores and turned 123s fits into 270s ones.
# So this WAITS for w26a to exit before it starts. Every cell is checkpointed, so re-running
# this after a kill resumes rather than restarts.
#
# No `set -e`: a failure in one transform must not cost the others, and the report at the end
# prints whatever is on disk.
cd "$(dirname "$0")/.."

echo "waiting for w26a to exit..."
while :; do
  alive=0
  for p in /proc/[0-9]*/cmdline; do
    case "$( { tr "\0" " " < "$p"; } 2>/dev/null )" in *w26a_sensitivity*) alive=1;; esac
  done
  [ "$alive" = 0 ] && break
  sleep 20
done
echo "w26a clear at $(date -u), starting w26f"

# ORDER MATTERS. The --kind arm is the SELECTOR (prereg §D3) and all three transforms of it
# come before any of the --cv arm, because the cross-fitted arm is a companion reading that
# never picks C. If this run is cut short, the arm that decides the question is the one
# already on disk.
echo "===== SELECTOR ARM (held-out rows, prereg §D3) ====="
for k in hybrid rankraw rescale; do
  echo "########## $k ##########"
  .venv/bin/python experiments/w26f_csweep.py --kind "$k"
done

echo "===== COMPANION ARM (cross-fitted CV, NOT a selector) ====="
for k in hybrid rankraw rescale; do
  echo "########## cv $k ##########"
  .venv/bin/python experiments/w26f_csweep.py --cv "$k"
done

echo "########## report ##########"
.venv/bin/python experiments/w26f_csweep.py --report
