#!/usr/bin/env bash
# w97b -- wait for w96d to finish, then price the windowed-TE member INTO THE PACK and
# apply the w97 prereg's rule. No human read in the middle, same as w96d.
#
#   1. wait for the w96d chain (pid in $1) to exit
#   2. if it built both arms, run w26i's paired 50/50 harness on them
#   3. apply experiments/w97a_gate.py -- report only, it enrols NOTHING
#
# Step 3 deliberately does NOT copy anything into oof/. Enrolment is a one-line `cp` that a
# human or a later run does on purpose after reading the verdict; automating it would make
# oof/ change without anyone deciding to change it, which is w96 §10's whole lesson.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
WAIT_PID="${1:-32247}"

echo "== 0. waiting for w96d (pid $WAIT_PID) =="
# /proc poll: ps and pgrep do not exist on this box. Re-read cmdline every time so a
# recycled pid cannot be mistaken for the chain still running.
while [ -r "/proc/$WAIT_PID/cmdline" ] && \
      tr '\0' ' ' < "/proc/$WAIT_PID/cmdline" 2>/dev/null | grep -q w96d; do
  sleep 60
done
echo "w96d gone at $(date -u)"

W=oof_w96/oof_lgbm_teprior_windowed.npy
G=oof_w96/oof_lgbm_teprior_global.npy
if [ ! -f "$W" ] || [ ! -f "$G" ]; then
  echo "NO ARMS ON DISK -- w96d withdrew, or crashed before its build step."
  echo "Per the w96 addendum a WITHDRAW is the end of the line: do NOT go looking for a"
  echo "third operating point. Nothing to price, nothing to enrol."
  exit 0
fi

echo
echo "== 1. what is it worth INSIDE the pack (w26i, unchanged instrument) =="
"$P" -u experiments/w26i_value.py \
    --new-dir "$PWD/oof_w96" \
    --new-names lgbm_teprior_windowed,lgbm_teprior_global \
    --reps 5 --out w97a_teprior_value \
    --prereg-note "PREREG w97 §2 said: Delta = d(windowed) - d(global) in [0, +3e-6], modal +1e-6.
Priced from w24b: a redundant family is worth +1.07e-06 per member, an orthogonal one +1.21e-05.
This member is the same model, folds, features and seeds as the pack; only the encoding differs." \
  || { echo "VALUE HARNESS CRASHED -- nothing is decided, re-run this script."; exit 1; }

echo
echo "== 2. the w97 prereg's rule, in code =="
"$P" experiments/w97a_gate.py || true

echo
echo "REPORT ONLY. Nothing was copied into oof/. See experiments/w97_prereg.txt §4."
echo "w97b done $(date -u)"
