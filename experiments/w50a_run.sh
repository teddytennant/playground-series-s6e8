#!/usr/bin/env bash
# w50a -- ARM 216: ARM 217's pack MINUS `hboyang_mix`, and nothing else changed.
#
# This is `w42e_run.sh` with one member appended to --drop. ARM 217 on disk is therefore a
# matched control differing by EXACTLY ONE COLUMN, and ARM 211 on disk is a matched control
# differing by exactly the other five members of ext_members16. The two deltas decompose
# ARM 217's +41.71e-6 into "the suspect vector" and "the other five".
#
# The registered call, the decision rules R1-R4 and the WANTED-ineligibility carry-over are
# fixed in experiments/w50_prereg.txt, committed BEFORE this script was run.
#
# ⛔ ARM 216 INHERITS w42b's WANTED-INELIGIBILITY. Five of the six imported members still
# have unread es-on-val status; dropping the sixth does not discharge that clause. Queue it,
# send it, read it -- but it is not a deadline pick on CV.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w50a_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12,ext_members14,ext_members15,ext_members16
NAME=w50_ad216std
: > "$LOG"

# the control must be on disk before the arm means anything -- w42b's ORDERING clause.
for f in experiments/w40f_build.log experiments/w42e_build.log; do
  grep -q 'OOF AUC' "$f" 2>/dev/null || { echo "ABORT: $f has no base AUC" | tee -a "$LOG"; exit 3; }
done
[ -f data/ext_members16/oof_hboyang_mix.npy ] || { echo "ABORT: ext_members16 missing" | tee -a "$LOG"; exit 4; }

echo "=== $NAME  = ARM 217 minus hboyang_mix  $(date -u) ===" | tee -a "$LOG"
echo "    control ARM 211 h3 base 0.9701331846 (w40f_build.log)" | tee -a "$LOG"
echo "    control ARM 217 h3 base 0.9701748950 (w42e_build.log)" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat,hboyang_mix" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w50_ad216stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w50a done $(date -u). Read against experiments/w50_prereg.txt R1-R4. ===" >> "$LOG"
