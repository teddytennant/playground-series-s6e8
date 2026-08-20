#!/usr/bin/env bash
# w36b -- the four-clean-member build. `w34a_run.sh` VERBATIM with the --extra-dirs list
# extended and NOTHING else changed, so submissions/w34_ad195std*.csv on disk are a matched
# control and the delta is attributable to the added members as a group.
#
# The pack is chosen by the rule pre-registered in experiments/w36b_prereg.txt, committed
# BEFORE experiments/w34c_value.csv existed (git 2d1eff1). This script evaluates that rule
# itself; no human read of the table sits between the number and the pack.
#
# ONE ARM ONLY. w34a took ~6.5 h per arm and the 08-21 send day opens at 00:00 UTC.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w36b_build.log
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400
: > "$LOG"

echo "waiting for w34c_value to exit..." | tee -a "$LOG"
while ! grep -q 'w34c done' experiments/w34c_value.log 2>/dev/null; do sleep 30; done
echo "w34c clear at $(date -u)" | tee -a "$LOG"
sleep 15

# ---- the pre-registered rule, evaluated in code ------------------------------------------
DEC=$("$P" - <<'PY'
import os, sys
import pandas as pd
p = "experiments/w34c_value.csv"
if not os.path.exists(p):
    print("197 no-w34c-csv"); sys.exit()
d = pd.read_csv(p)
gate = float((d["gate_cat4"] - d["base165"]).mean())
pos  = int((d["both"] > d["pack"]).sum()); n = len(d)
ok_a = 3.6e-5 <= gate <= 4.6e-5
ok_b = pos >= 4 and n >= 5
print(("199" if (ok_a and ok_b) else "197") +
      f" gate={gate:.3e} ok_a={ok_a} both_positive={pos}/{n} ok_b={ok_b}")
PY
)
ARM=${DEC%% *}
echo "PREREG RULE -> ARM $ARM   [$DEC]" | tee -a "$LOG"

if [ "$ARM" = "199" ]; then
  X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members11,ext_members12
  NAME=w36_ad199std
else
  X=ext_members3,ext_members4,ext_members6,ext_members7pin,ext_members8,ext_members10,ext_members12
  NAME=w36_ad197std
fi
echo "=== $NAME  extra-dirs $X  $(date -u) ===" | tee -a "$LOG"

nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name "$NAME" \
    --standardize --extra-dirs "$X" --drop "$D,om_cat" >> "$LOG" 2>&1 || exit 1
nice -n 15 "$P" experiments/make_h3.py "$NAME" >> "$LOG" 2>&1 || exit 1
W21A_BASE="${NAME}_h3" W21A_TAG="${NAME%std}stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w36b done $(date -u). Bars: w34_ad195std_h3 0.9701205753 / w34_ad195stdcorr 0.9701247949" >> "$LOG"
echo "    WANTED moves only above 0.9701288 (leader + the 4e-6 rebuild floor, prereg) ===" >> "$LOG"
