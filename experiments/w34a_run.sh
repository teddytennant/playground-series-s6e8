#!/usr/bin/env bash
# w34a -- the om_ftt build. `w29c_run.sh` VERBATIM with `ext_members10` added to
# --extra-dirs and NOTHING else changed, so submissions/w29_ad194std*.csv on disk are a
# matched control and the delta is attributable to the new member alone.
#
# WHY. w33b_value priced the two clean omidbaghchehsaraei members into the 187-member
# paired combiner at, over two reps with a stable sign in both:
#     om_ftt  +7.5e-6      om_cat  -2.5e-6      both  +7.0e-6
# and the `cat4` reproduction gate returned +4.35e-5 against w20d's required +4.1e-5, so
# those rows are comparable to the rest of RESEARCH. +7.5e-6 is the largest per-member
# value measured here since the adarsh import (+2.1e-6/member) and w32a prices the gold
# line at +10.4e-6.
#
# ⚠ THE PAIRED NUMBER IS NOT THE SHIPPING NUMBER. w26i's 50/50 paired instrument measures
# value into a 187-member hybrid combiner; this builds a 194-member cross-fitted h3/ens4
# object on the frozen folds. RESEARCH's own record is that paired-delta magnitudes do not
# transfer one-for-one. The bar this has to clear is the CV leader 0.9701182875.
#
# TWO ARMS, both built so om_cat's sign is confirmed on the shipping instrument rather than
# assumed from the paired one:
#   ARM 1  w34_ad195std   = 194 + om_ftt              (om_cat dropped)   <- the lead candidate
#   ARM 2  w34_ad196std   = 194 + om_ftt + om_cat
# The five es-on-val members in data/ext_members10es/ are NOT on either --extra-dirs list:
# their OOF is selected on the scored fold (w33 §3) and would buy CV the LB will not pay.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w34a_ad195.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10
: > "$LOG"

echo "=== ARM 1: w34_ad195std = 194 + om_ftt (om_cat dropped) ===" >> "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name w34_ad195std \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py w34_ad195std >> "$LOG" 2>&1 || exit 1
W21A_BASE=w34_ad195std_h3 W21A_TAG=w34_ad195stdcorr \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== ARM 2: w34_ad196std = 194 + om_ftt + om_cat ===" >> "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name w34_ad196std \
    --standardize --extra-dirs "$X" --drop "$D" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py w34_ad196std >> "$LOG" 2>&1 || exit 1
W21A_BASE=w34_ad196std_h3 W21A_TAG=w34_ad196stdcorr \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w34a done. Bars: w29_ad194std_h3 0.9701140064 / w29_ad194stdcorr 0.9701182875 ===" >> "$LOG"
