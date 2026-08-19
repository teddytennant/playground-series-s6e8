#!/usr/bin/env bash
# w29e -- the generative second batch, into data/ext_members9.
# Opened by qda_raw landing at maxcorr 0.88773, decorrelation rank 1 of 191, below the
# 190-pack's previous minimum of 0.91797. These vary the FRAME (qda_lat) and the DENSITY
# (gnb_raw = diagonal covariance, gmm_raw = K-component mixture per class), not the
# hyperparameters -- w27q's finding is that hyperparameters do not move a member.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w29e_build.log
O=data/ext_members9
: > "$LOG"
nice -n 18 $P experiments/w29a_newclass.py --arm gnb_raw --outdir $O               >> "$LOG" 2>&1
nice -n 18 $P experiments/w29a_newclass.py --arm qda_lat --outdir $O --reg 0.05    >> "$LOG" 2>&1
nice -n 18 $P experiments/w29a_newclass.py --arm gmm_raw --outdir $O --k 6 --gmm-fit 120000 >> "$LOG" 2>&1
echo "=== w29e done ===" >> "$LOG"
