#!/usr/bin/env bash
# w40f -- ARM 211: ARM 202's pack plus the nine `yadoy666` union94 streams selected by the
# w40d_prereg rule. `w38d_run.sh` with ext_members15 appended to --extra-dirs and NOTHING
# else changed, so ARM 202 on disk is a matched control and the 211-202 delta is attributable
# to the nine added streams AS A GROUP.
#
# The pack, the rule that chose it, the expectation and the ⛔ WANTED-ineligibility clause are
# all fixed in experiments/w40d_prereg.txt, committed BEFORE this ran. Nothing here is chosen
# from a number.
#
# ⛔ ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED whatever its CV -- the nine streams
# come from an aggregator and their es-on-val status is UNKNOWN. It may be queued and sent;
# it may not become a deadline pick. See the prereg.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w40f_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12,ext_members14,ext_members15
NAME=w40_ad211std
: > "$LOG"

# wait on the ARTEFACT, never on a /proc scan -- RESEARCH's w32 note.
echo "waiting for the w38d ARM 202 build to exit..." | tee -a "$LOG"
for i in $(seq 1 240); do
  grep -q 'w38d done' experiments/w38d_build.log 2>/dev/null && break
  sleep 60
done
if ! grep -q 'w38d done' experiments/w38d_build.log 2>/dev/null; then
  echo "ABORT: ARM 202 never printed 'w38d done'. An arm without its matched control" | tee -a "$LOG"
  echo "       measures nothing, so this build does not run. (w40d_prereg, ORDERING.)" | tee -a "$LOG"
  exit 3
fi
echo "w38d clear at $(date -u)" | tee -a "$LOG"
sleep 15

echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w40_ad211stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w40f done $(date -u). Control: ARM 202 (see experiments/w38d_build.log)." >> "$LOG"
echo "    ⛔ NOT ELIGIBLE for check_selection.WANTED on CV alone -- w40d_prereg. ===" >> "$LOG"
