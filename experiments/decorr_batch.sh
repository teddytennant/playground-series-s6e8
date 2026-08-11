#!/usr/bin/env bash
# Members tuned for DECORRELATION, not solo AUC.
#
# Slot 1 measured solo-AUC tuning of LightGBM at +0.00003 (null). Every measurement
# since says a member's worth to the stack is set by how differently it is wrong, not
# by how good it is alone: bolt_extratrees_support sits at maxcorr 0.811 and 35 ordinary
# GBDTs from one independent pipeline were the single largest share of slot 3's gain.
#
# So the hyperparameters here are chosen to move the FUNCTION SHAPE away from the pack's
# hand-set (leaves 96, depth 7) vector, accepting a worse solo AUC as the price.
# All fixed-schedule (--stopping 0): the round count must not see the held-out labels.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run () {  # name cores rounds params
  taskset -c "$2" nice -n 15 $PY agent/run_lgbm.py --name "$1" --frac \
    --stopping 0 --n-estimators "$3" --params "$4" > "logs/$1.log" 2>&1
  echo "done $1"
}

mkdir -p logs

# 1. STUMP -- depth 3 / 8 leaves. Only low-order interactions. The pack is all depth 7+,
#    so this is the largest available move in function shape at trivial cost per tree.
run lgbm_stump_lat_frac 0-3 4000 \
 '{"learning_rate":0.05,"num_leaves":8,"max_depth":3,"min_child_samples":100,"subsample":0.9,"subsample_freq":1,"colsample_bytree":0.6,"reg_lambda":5.0}' &

# 2. GOSS + extra_trees + thin colsample. Different sampling (gradient-based one-side),
#    randomized split thresholds, and each tree sees 20% of columns. extra_trees measured
#    -0.00037 SOLO in slot 1 -- that is the trade being made deliberately.
#
#    ABANDONED 2026-08-11. Killed after 48 minutes without completing fold 0, against
#    20 min/fold for the stump above. A peer session took the box to load 47 on 16 cores
#    and this config is the expensive one: GOSS reweights every row each iteration and
#    extra_trees gives up the histogram short-circuit, so it does not amortise the way
#    plain boosting does. If it is retried, do it on an idle box and drop to ~1200 rounds.
run lgbm_goss_xt_lat_frac 4-7 2500 \
 '{"boosting_type":"goss","learning_rate":0.04,"num_leaves":63,"max_depth":7,"min_child_samples":60,"colsample_bytree":0.2,"extra_trees":true,"reg_lambda":20.0}' &

wait
echo "batch done"
