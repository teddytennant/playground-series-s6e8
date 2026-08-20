#!/usr/bin/env bash
# w34c -- price the two CLEAN members the w34 bibliography scan turned up, with the same
# paired 50/50 instrument that priced om_ftt at +7.5e-6 in w33b.
#
#   ravi_xgb1c    solo 0.964201   maxcorr 0.99278 vs bei_xgb_identity_digit_raw12
#   ravi_lgbm1c   solo 0.964173   maxcorr 0.99224 vs bei_xgb_identity_digit_raw12
#
# ⚠ REGISTERED EXPECTATION, written before the run so the table is read against something.
# These are NOT the om_ftt profile. om_ftt paid +7.5e-6 because it is a pipeline class the
# pack did not hold (FT-Transformer, maxcorr 0.9845). These two are an XGBoost and a
# LightGBM at solo 0.9642 -- BELOW the pack median 0.966 -- and maxcorr ~0.992 against a
# pack whose median is 0.9946. They clear w29's acceptance gate (decorrelated AND within
# 0.005 of the pack median solo) but only just, and RESEARCH prices another tuned GBDT into
# this stack at ~4e-7. The honest prior is 0 to +2e-6 each, i.e. probably a null. They are
# measured anyway because they are honest, free, and the measurement is the only thing that
# can distinguish "another GBDT" from "the one that happened to matter".
#
# The four es-on-val members from the same scan (dm_cat, dm_lgb, ravi_cb1c, ravi_realmlp1c)
# are in data/ext_members11es/ and are NOT on this or any --extra-dirs list.
#
# ONE JOB AT A TIME: waits on the SHELL SCRIPT w34a_run.sh, not on a python process, since a
# queued stage's python does not exist yet and the wait would return instantly.
cd "$(dirname "$0")/.."

wait_for () {
  echo "waiting for $1 to exit..."
  while :; do
    alive=0
    for p in /proc/[0-9]*/cmdline; do
      case "$( { tr "\0" " " < "$p"; } 2>/dev/null )" in *"$1"*) alive=1;; esac
    done
    [ "$alive" = 0 ] && break
    sleep 30
  done
  echo "$1 clear at $(date -u)"
}

wait_for w34a_run.sh

echo "########## w34c_value  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w26i_value.py \
  --new-dir "$PWD/data/ext_members11" \
  --new-names ravi_xgb1c,ravi_lgbm1c \
  --reps 5 --out w34c_value
echo "########## w34c done  $(date -u) ##########"
