#!/usr/bin/env bash
# w60a -- the 5-arm scheme-average c_avg correction on the **ens4** base of ARM 199 (the PICK's
# own pack) and, as a matched control, of ARM 202. w30c_corr_*.log did exactly this for ARM 194;
# this is that script pointed at the two es-CLEARED packs that carry the strongest ens4 bases.
#
# WHY: w59 §7's lever needs an ELIGIBLE file with CV >= 0.9701294160 AND pred_lb ~ 0.97118 at
# the same time. `corr` (+13.72e-6 of pred_lb) and `ens4` (+13.41e-6) together clear the tier on
# a CV that is already over the bar. Predictions and the ⛔ not-built clause: w60_prereg.txt,
# committed at 45c2498 BEFORE this ran.
#
# No refit, no new members, no search -- w21a is a post-hoc correction over an existing OOF
# vector, and both bases are on disk. Nothing is chosen from a public score.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w60a_corr.log
: > "$LOG"

echo "=== w60a start $(date -u) ===" | tee -a "$LOG"

# Run BOTH concurrently, as w30c did with three. w21a is rank/AUC arithmetic over numpy, not a
# thread-hungry GBDT -- RESEARCH's "never two n_jobs=-1 LightGBM jobs at once" does not apply.
# `nice` so an interactive shell stays responsive. Each writes its own log and its own
# checkpoint (w21a_ckpt_<TAG>.json), so a kill costs at most the fold in flight.
W21A_BASE=w36_ad199std W21A_TAG=w36_ad199stdcorr_ens4 \
    nice -n 15 "$P" experiments/w21a_ad187corr.py > experiments/w60a_ad199_ens4.log 2>&1 &
PID1=$!
sleep 5
W21A_BASE=w38_ad202std W21A_TAG=w38_ad202stdcorr_ens4 \
    nice -n 15 "$P" experiments/w21a_ad187corr.py > experiments/w60a_ad202_ens4.log 2>&1 &
PID2=$!

wait $PID1; R1=$?
wait $PID2; R2=$?
echo "ad199_ens4 exit $R1   ad202_ens4 exit $R2" | tee -a "$LOG"
[ $R1 -eq 0 ] && [ $R2 -eq 0 ] || { echo "=== w60a FAILED $(date -u) ===" | tee -a "$LOG"; exit 1; }
echo "=== w60a done $(date -u) ===" | tee -a "$LOG"
