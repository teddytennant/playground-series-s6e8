#!/usr/bin/env bash
# w96d -- the w96b prereg, executed end to end with no human read in the middle.
#
#   1. replicate w96a at the TUNED vector (what the blend members were really trained at)
#   2. apply the addendum's rule to the replication: A and B only, WITHDRAW-ONLY
#   3. build the member pair ONLY if the replication confirms
#
# The order is the point. The replication runs BEFORE the build so it can still cost us the
# build, which is the only direction the addendum allows it to act in. Nothing here writes a
# submission, a blend weight or a registration; step 3 writes two names into oof/ and stops.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python

echo "== 1. replication at the tuned vector =="
"$P" -u experiments/w96a_teprior_folds.py --tuned --folds 0,1,2,3,4 || {
  echo "REPLICATION CRASHED -- not building. The control-vector BUILD is untouched but"
  echo "unconfirmed; re-run this script rather than launching w96c by hand."; exit 1; }

echo
echo "== 2. the addendum's rule, applied to the replication =="
if "$P" experiments/w96b_gate.py --json experiments/w96a_teprior_folds_tuned.json --confirm; then
  echo
  echo "== 3. build =="
  "$P" -u experiments/w96c_build_teprior_member.py
else
  echo
  echo "WITHDRAWN. The control-vector rule said BUILD and the tuned vector did not"
  echo "reproduce it. Per the addendum that is the end of the line -- do NOT build, and"
  echo "do not go looking for a third operating point that agrees with the first."
fi
echo "w96d done"
