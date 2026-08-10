#!/usr/bin/env bash
# A batch of members from OUR pipeline, every one on a FIXED iteration schedule.
#
# Two reasons for the fixed schedule. It removes the early-stopping optimism that
# agent/run_lgbm.py's --stopping path carries (it stops on the rows that become the
# member's own OOF). And it is what the strongest independent public library does --
# beicicc names its datasets "fixed900" / "fixed1500" / "fixed4000" for exactly this.
#
# Diversity is by MECHANISM, not by seed: random split thresholds, a squared-error loss
# instead of logloss, and a second implementation (XGBoost) of the same idea. The slot-3
# attribution says an independent pipeline's members are worth ~5-9e-6 each even when the
# model family is one already held.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

# random split thresholds -- the boosted analogue of what makes ExtraTrees disagree
$PY agent/run_lgbm.py --name lgb_xt_lat_frac --frac --stopping 0 --n-estimators 2200 \
  --seeds 7 --params '{"learning_rate":0.025,"num_leaves":63,"max_depth":7,"max_bin":511,
   "min_child_samples":250,"subsample":0.9,"subsample_freq":1,"colsample_bytree":0.6,
   "reg_lambda":80.0,"extra_trees":true}'

# squared-error loss on the 0/1 label: a different estimator, not a differently-tuned one
$PY agent/run_lgbm.py --name lgb_l2_lat_frac --frac --stopping 0 --n-estimators 1900 \
  --seeds 11 --params '{"objective":"regression","learning_rate":0.025,"num_leaves":63,
   "max_depth":7,"max_bin":511,"min_child_samples":250,"subsample":0.9,"subsample_freq":1,
   "colsample_bytree":0.6,"reg_lambda":80.0}'

# a second implementation of the boosted-tree idea on our features
$PY experiments/run_xgb.py --name xgb_lat_frac --frac --rounds 2000 --seed 13

# and a deeper/wider LightGBM, fixed schedule, different column sampling
$PY agent/run_lgbm.py --name lgb_wide_lat_frac --frac --stopping 0 --n-estimators 1700 \
  --seeds 23 --params '{"learning_rate":0.03,"num_leaves":160,"max_depth":9,"max_bin":255,
   "min_child_samples":120,"subsample":0.8,"subsample_freq":1,"colsample_bytree":0.4,
   "reg_lambda":20.0}'
