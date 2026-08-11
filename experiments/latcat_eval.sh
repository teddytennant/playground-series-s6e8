#!/usr/bin/env bash
# Evaluate the lat / latcat one-factor pair once both 5-fold runs have landed.
#
#   1. rebuild the pack matrix at the 156-member set -- the two members under test are
#      DROPPED from it, otherwise each would read maxcorr 1.0000 against itself
#   2. profile both against that pack (solo AUC, maxcorr, median corr)
#   3. build blend158 and its four transform stacks, cross-fitted on the frozen folds,
#      directly comparable to the blend156 numbers already in the journal
#
# Bracket classes in the pgrep patterns stop the waiter matching its own command line.
set -u
cd "$(dirname "$0")/.."

DROP5=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,bolt_xgb_d7_alt2
PY=.venv/bin/python

until ! pgrep -f "run_xgb.py --name xgb_[l]at" >/dev/null \
   && ! pgrep -f "run_xgb.py --name xgb_[l]atcat" >/dev/null; do sleep 30; done

for m in xgb_lat xgb_latcat; do
  if [ ! -f "oof/oof_${m}.npy" ] || [ ! -f "oof/test_${m}.npy" ]; then
    echo "!! ${m} did not produce a member -- stopping, nothing downstream is valid"
    exit 1
  fi
done
echo "=== both members present ==="

echo "=== 1. pack matrix at 156 members (the pair excluded) ==="
nice -n 15 $PY experiments/stack_lab.py --transform hybrid --refresh --reps 0 \
  --drop "${DROP5},xgb_lat,xgb_latcat" || exit 1

echo "=== 2. member profile vs the 156-member pack ==="
nice -n 15 $PY experiments/member_profile.py xgb_lat xgb_latcat || exit 1

echo "=== 3. blend158 (156 + the pair), four transforms + rank-ensemble ==="
nice -n 15 $PY experiments/blend_lab.py --build --reps 0 --C 1.0 \
  --drop "${DROP5}" --submit-name blend158 || exit 1

echo "=== latcat_eval complete ==="
