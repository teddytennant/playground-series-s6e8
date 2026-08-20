#!/usr/bin/env bash
# w36g -- ARM 197: the same build with the ravi pair REMOVED. `w36b_run.sh` with
# ext_members11 dropped from --extra-dirs and nothing else changed, so w36_ad199std* on disk
# is a matched control and the 199-197 difference is the ravi pair's value on the SHIPPING
# instrument rather than on the 50/50 paired one.
#
# WHY THIS EXISTS: the pre-registered rule fired 199 on a 4/5 sign gate; w26i's own house
# criterion is 5/5 and calls the ravi pair a NULL (rep 3 was -1e-6). See the addendum in
# w36b_prereg.txt. The selection rule between the two arms is registered there, before either
# number existed. ARM 197 is a CONTROL. It is not promoted just for landing higher.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w36g_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members12
NAME=w36_ad197std
: > "$LOG"

echo "waiting for the w36a ram measurement to exit..." | tee -a "$LOG"
while ! grep -q 'w36a done' experiments/w36a_value.log 2>/dev/null; do sleep 60; done
echo "w36a clear at $(date -u)" | tee -a "$LOG"
sleep 15

echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"
nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="w36_ad197stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w36g done $(date -u). Bars: w34_ad195std_h3 0.9701205753 / w34_ad195stdcorr 0.9701247949" >> "$LOG"
echo "    WANTED moves only above 0.9701288, and ARM 197 is a CONTROL (prereg addendum) ===" >> "$LOG"
