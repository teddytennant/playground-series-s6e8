#!/usr/bin/env bash
# w35b -- re-run the w34c measurement that was killed when the w34 session ended.
#
# The two CLEAN members from the w34 bibliography scan, priced with the same paired 50/50
# instrument that priced om_ftt at +7.5e-6 in w33b:
#   ravi_xgb1c    solo 0.964201   maxcorr 0.99278
#   ravi_lgbm1c   solo 0.964173   maxcorr 0.99224
#
# The w34 registered expectation stands VERBATIM and is not restated here so it cannot be
# quietly edited: see experiments/w34c_run.sh and the w34 journal entry §6. Short form --
# these are NOT the om_ftt profile (below-median solo, ~0.992 maxcorr, another GBDT into a
# stack that prices another GBDT at ~4e-7), so the honest prior is 0 to +2e-6 each, a null.
#
# ⚠ The cat4 reproduction gate must return ~+4.1e-5 or NO ROW IN THE OUTPUT IS COMPARABLE
# to anything in RESEARCH.md. Check it before reading the value column.
cd "$(dirname "$0")/.."

echo "waiting for w35a requeue to finish..."
while ! grep -q 'w35a done' experiments/w35a_requeue.log 2>/dev/null; do sleep 20; done
echo "w35a clear at $(date -u)"

echo "########## w35b_value  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w26i_value.py \
  --new-dir "$PWD/data/ext_members11" \
  --new-names ravi_xgb1c,ravi_lgbm1c \
  --reps 5 --out w35b_value
echo "########## w35b done  $(date -u) ##########"
