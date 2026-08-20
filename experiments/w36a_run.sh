#!/usr/bin/env bash
# w36a -- price the two CLEAN members the w35 kernel-output sweep imported.
#
#   ram_hgb   redamountassir/s6e8-histgradientboosting-lb-0-96945   solo 0.968026  maxcorr 0.99474
#   ram_lgb   redamountassir/s6e8-lgbm-lb-0-96965                   solo 0.968259  maxcorr 0.99506
#
# ⚠ PROFILE, AND WHY IT IS NOT THE ravi PROFILE. The pack median solo is 0.966319 and the
# pack median maxcorr is ~0.9946. These two are +0.0017/+0.0019 ABOVE the median solo at
# ESSENTIALLY THE MEDIAN maxcorr -- the mirror image of ravi_xgb1c/ravi_lgbm1c (0.9642 solo,
# below median, at 0.992 maxcorr, i.e. below-median strength bought with decorrelation).
# `ram_hgb` is also a HistGradientBoosting, a sklearn implementation the pack does not hold.
#
# REGISTERED EXPECTATION, written before the instrument runs. w29's acceptance gate wants
# decorrelated AND near-median solo; these clear on strength, not on decorrelation, and the
# only member that ever paid big here (om_ftt, +7.5e-6) paid on being an absent FUNCTION
# CLASS at maxcorr 0.9845. At 0.995 maxcorr these two are inside the crowd. Honest prior:
# 0 to +4e-6 each, with ram_hgb the likelier of the two on the sklearn-implementation
# argument. A sign that flips across reps is a null whatever the mean (w20d's `nn`).
#
# ⚠ The cat4 reproduction gate must return ~+4.1e-5 or NO ROW IS COMPARABLE to RESEARCH.md.
# ⚠ SERIALISED ON PURPOSE. w35b was OOM-Killed at rep 0 because it ran concurrently with the
# identical w34c job; 32 GB does not hold two copies of a 189-member design matrix.
cd "$(dirname "$0")/.."

echo "waiting for the w36b BUILD to exit (it holds the memory)..."
while ! grep -q "w36b done" experiments/w36b_build.log 2>/dev/null; do sleep 60; done
echo "w36b clear at $(date -u)"
sleep 20

echo "########## w36a_value  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w26i_value.py \
  --new-dir "$PWD/data/ext_members12" \
  --new-names ram_hgb,ram_lgb \
  --reps 5 --out w36a_value
echo "########## w36a done  $(date -u) ##########"
