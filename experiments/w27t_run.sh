#!/usr/bin/env bash
# w27t -- the 190-member standardised build: w27h's 188-member pack plus the two
# feature-block-ablation members from ext_members7. Pre-registered at
# experiments/w27_prereg_slot6.txt SS M10.
#
# w27h_run.sh verbatim, with ext_members7 added to --extra-dirs and NOTHING else changed,
# so submissions/w27_ad188std*.csv on disk are a byte-identical matched control.
#
# ⚠ --extra-dirs points at ext_members7PIN, a frozen hardlink dir holding exactly the two
# registered members. ext_members7 itself is still being written by w27r_blockdrop (tedrop
# exports on completion), so pointing at it would silently load 191 members. See the M10
# amendment in the prereg.
#
# ⚠ Waits on the M9 launcher rather than running alongside it (R-M10e): the box already
# carries w26k_ctscale, w27g_tunect, w27o_ctclass5, w27p_serveclass and w27r_blockdrop.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w27t_ad190.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin
: > "$LOG"

# R-M10e deviation, recorded in the prereg: started immediately rather than chained behind
# the M9 arm, which is running ~6x slower per fit than registered. No wait.
echo "=== M9 arm finished, starting the 190-member build ===" >> "$LOG"

nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name w27_ad190std \
    --standardize --extra-dirs "$X" --drop "$D" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py w27_ad190std >> "$LOG" 2>&1 || exit 1
W21A_BASE=w27_ad190std_h3 W21A_TAG=w27_ad190stdcorr \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w27t done: compare w27_ad190std_h3 against w27_ad188std_h3 0.9701114720 ===" >> "$LOG"
