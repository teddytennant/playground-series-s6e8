#!/usr/bin/env bash
# w27s -- the L2 penalty on the STANDARDISED 188-member pack. Pre-registered at
# experiments/w27_prereg_slot6.txt SS M8.
#
# The one cell RESEARCH labels "built, not yet measured": --standardize makes the L2
# prior isotropic, and every shipped build switches it on and then leaves C=1.0, i.e.
# a penalty per row of 1.8e-6 against a log-loss of order 0.5. Completely unpenalised.
#
# ORDER MATTERS. The standardised arm is the OPEN cell and runs FIRST; the
# unstandardised arm is only M8(b)'s transfer control and runs second, reusing the
# cached meta_hybrid188.npz so the 188-member load is paid exactly once. A kill at
# any point therefore costs the control, not the result.
#
# Grid runs two decades LOWER than the closed 156-member sweep because standardising
# divides the columns by their sd, which moves the optimal C down by roughly the
# square of that scale (ridge_sweep.c_for / standardize docstrings). C=1.0 is Cs[0]
# and is the reference cell every delta is quoted against.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w27s_lamstd.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6
G=1.0,1e-2,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6
: > "$LOG"

echo "=== ARM standardised (M8a, the open cell) ===" >> "$LOG"
nice -n 15 "$P" experiments/ridge_sweep.py --transform hybrid --key hybrid188 \
    --drop "$D" --extra-dirs "$X" --expect 188 --standardize \
    --cs "$G" --reps 3 >> "$LOG" 2>&1

echo "" >> "$LOG"
echo "=== ARM unstandardised (M8b, transfer control) ===" >> "$LOG"
nice -n 15 "$P" experiments/ridge_sweep.py --transform hybrid --key hybrid188 \
    --drop "$D" --extra-dirs "$X" --expect 188 \
    --cs "$G" --reps 3 >> "$LOG" 2>&1

echo "=== w27s both arms done ===" >> "$LOG"
