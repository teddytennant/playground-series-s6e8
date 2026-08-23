#!/usr/bin/env bash
# w69b -- ARM 208: ARM 211's pack MINUS ext_members14 (the three `zhukovoleksiy` lexb
# members). `w40f_run.sh` with ext_members14 removed from --extra-dirs and NOTHING else
# changed, so ARM 211 on disk is a matched control for the lexb trio and ARM 199 on disk is
# the matched control for the nine `yadoy666` streams. Together with ARM 202 that closes the
# 2x2: 199 = -lexb -yadoy, 202 = +lexb -yadoy, 208 = -lexb +yadoy, 211 = +lexb +yadoy.
#
# WHY THIS PACK. w65a measured the lexb trio at -1.058e-6 (deleting them GAINS) and 3.80e-6
# worse than a placebo trio, 4/4 partitions. ext_members14 is in every ad202/ad211/ad216
# build and nothing has priced its removal from the packs ABOVE 202. w68 section 7 item 4.
#
# ⛔ ARM 208 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV. Registered in
# experiments/w69_prereg.txt section 2.1, committed 39843c6 BEFORE this file existed and
# before the CV existed. ARM 208 is a DIFFERENT PACK from ARM 211 and deleting three columns
# re-weights the nine `yadoy666` streams, which is exactly the case the w40_ad211 retirement
# clause in check_selection.WANTED_RETIRED names as not carrying over. It may be built,
# queued, priced and sent; it may not become a deadline pick. Only its OWN matched control
# (208 - 199) within +-4e-6 on all four criterion bases lifts it, and not by the run that
# built it.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w69b_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12,ext_members15
NAME=w69_ad208std
: > "$LOG"

echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w69_ad208stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w69b done $(date -u). Controls on disk: ARM 211 (w40f_build.log) for the lexb" >> "$LOG"
echo "    trio, ARM 199 (w36b) for the nine yadoy streams. ⛔ NOT WANTED-eligible on CV," >> "$LOG"
echo "    whatever this reads -- w69_prereg section 2.1. ===" >> "$LOG"
