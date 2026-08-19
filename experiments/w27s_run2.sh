#!/usr/bin/env bash
# w27s arm 3 -- the M9 grid extension. Pre-registered at
# experiments/w27_prereg_slot6.txt SS M9, appended BEFORE this file was written.
#
# M8's registered grid ran 1.0 down to 1e-6 and came back strictly monotone-decreasing:
# there is no interior optimum ANYWHERE below C=1.0, and the mechanism I registered had
# the sign backwards -- standardising raises ||w||^2 by s^2 = 22.7, so the same C buys
# MORE shrinkage, and the optimal C moves UP rather than down. This arm searches the side
# the registered grid never covered.
#
# ⚠ NEW FILE, not an edit of w27s_run.sh, which is currently executing: bash re-reads a
# running script from its current byte offset, so editing one silently switches the live
# job to the new text mid-flight (w27 slot 4 SS12).
#
# Waits on the arm-1/2 launcher rather than running alongside it, so the two never
# contend for the same cached matrix load or the same cores.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w27s_lamstd2.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6
# C=1.0 FIRST so it is Cs[0] and every delta is quoted against the shipped cell, matching
# arm 1 exactly. The grid then runs both ways from it.
G=1.0,30,10,3,0.3,0.1,0.03
: > "$LOG"

while kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "=== arms 1/2 finished, starting M9 extension ===" >> "$LOG"

nice -n 15 "$P" experiments/ridge_sweep.py --transform hybrid --key hybrid188 \
    --drop "$D" --extra-dirs "$X" --expect 188 --standardize \
    --cs "$G" --reps 3 >> "$LOG" 2>&1

echo "=== w27s arm 3 done ===" >> "$LOG"
