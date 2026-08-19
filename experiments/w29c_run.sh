#!/usr/bin/env bash
# w29c -- the 194-member standardised build: w27t's 190-member pack plus the four
# function-class members from data/ext_members8 (qda_raw, rff_raw, rff_lat, poly2_raw).
# Registered at experiments/w29_prereg_slot10.txt SS M14; all four cleared the M14(b)3
# acceptance gate (maxcorr < 0.985) on the 190-pack, and qda_raw came in at maxcorr
# 0.88773 -- decorrelation rank 1 of 191, below the pack's previous minimum of 0.91797.
#
# w27t_run.sh verbatim with ext_members8 added to --extra-dirs and NOTHING else changed,
# so submissions/w27_ad190std*.csv on disk are a matched control.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w29c_ad194.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8
: > "$LOG"

nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name w29_ad194std \
    --standardize --extra-dirs "$X" --drop "$D" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py w29_ad194std >> "$LOG" 2>&1 || exit 1
W21A_BASE=w29_ad194std_h3 W21A_TAG=w29_ad194stdcorr \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w29c done: compare w29_ad194std_h3 against w27_ad190std_h3 0.9701133391 ===" >> "$LOG"
