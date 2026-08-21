#!/usr/bin/env bash
# w42e -- ARM 217: ARM 211's pack plus the six streams the w42b_prereg rule selected out of
# the 52 previously-unmined CANDIDATE kernel outputs. `w40f_run.sh` with ext_members16
# appended to --extra-dirs and NOTHING else changed, so ARM 211 on disk is a matched control
# and the 217-211 delta is attributable to the six added streams AS A GROUP.
#
# The rule, the expectation and the ⛔ WANTED-ineligibility clause are fixed in
# experiments/w42b_prereg.txt, committed BEFORE any AUC for these streams existed.
#
# ⛔ ARM 217 IS NOT ELIGIBLE FOR check_selection.WANTED whatever its CV. The six come from
# six separate authors' notebooks whose es-on-val status is unread, and w41 §4 withdrew the
# single deflation constant that might otherwise have priced it. Queue it, send it -- a send
# is free and its LB reading is the discriminator -- but it is not a deadline pick on CV.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w42e_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12,ext_members14,ext_members15,ext_members16
NAME=w42_ad217std
: > "$LOG"

# wait on the ARTEFACT, never on a /proc scan -- RESEARCH's w32 note.
echo "waiting for the w40f ARM 211 build to exit..." | tee -a "$LOG"
for i in $(seq 1 900); do
  grep -q 'w40f done' experiments/w40f_build.log 2>/dev/null && break
  sleep 60
done
if ! grep -q 'w40f done' experiments/w40f_build.log 2>/dev/null; then
  echo "ABORT: ARM 211 never printed 'w40f done'. An arm without its matched control" | tee -a "$LOG"
  echo "       measures nothing, so this build does not run. (w42b_prereg, ORDERING.)" | tee -a "$LOG"
  exit 3
fi
echo "w40f clear at $(date -u)" | tee -a "$LOG"
sleep 15

# the import is re-asserted here rather than assumed: it refuses to write if the rule stops
# yielding the pinned six, so a drifted vet cannot reach the build silently.
[ -f data/ext_members16/oof_hboyang_mix.npy ] || { echo "ABORT: ext_members16 missing" | tee -a "$LOG"; exit 4; }

echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w42_ad217stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w42e done $(date -u). Control: ARM 211 (see experiments/w40f_build.log)." >> "$LOG"
echo "    ⛔ NOT ELIGIBLE for check_selection.WANTED on CV alone -- w42b_prereg. ===" >> "$LOG"
