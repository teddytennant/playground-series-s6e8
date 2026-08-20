#!/usr/bin/env bash
# w38d -- ARM 202: the 199-member pack plus the three `zhukovoleksiy` members vetted in
# experiments/w38c_lexvet.py. `w36b_run.sh` with ext_members14 appended to --extra-dirs and
# NOTHING else changed, so submissions/w36_ad199std*.csv on disk is a matched control and the
# 202-199 delta is attributable to the three added members as a GROUP.
#
# The pack, the bar (0.9701440) and the expectation are fixed in experiments/w38d_prereg.txt,
# committed BEFORE this ran (git d063408). Nothing here is chosen from a number.
#
# Chained behind w36g (ARM 197), which is a registered control queued before these members
# were found and is not displaced by them. See the prereg's ordering note.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w38d_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12,ext_members14
NAME=w38_ad202std
: > "$LOG"

# wait on the ARTEFACT, never on a /proc scan -- RESEARCH's w32 note, and w36g's own form
echo "waiting for the w36g ARM 197 control to exit..." | tee -a "$LOG"
while ! grep -q 'w36g done' experiments/w36g_build.log 2>/dev/null; do sleep 60; done
echo "w36g clear at $(date -u)" | tee -a "$LOG"
sleep 15

echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w38_ad202stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w38d done $(date -u). Bar: w36_ad199stdcorr 0.9701400060 (leader, unsent)." >> "$LOG"
echo "    WANTED moves to ARM 202 only above 0.9701440 (leader + the 4e-6 rebuild floor). ===" >> "$LOG"
