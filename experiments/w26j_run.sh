#!/usr/bin/env bash
# w26j -- the XGBoost arm of the same question w26i asks of CatBoost, pre-registered at
# experiments/w26_prereg.txt §F before anything was built.
#
# THE QUESTION. w24c prices a FOREIGN-pipeline XGBoost at +6.24e-6/member into this stack
# and a foreign CatBoost at +9.08e-6. Every "another GBDT is worth nothing" measurement on
# file was made on a GBDT WE built on OUR features. w26i tests whether an ours-built
# CatBoost is worth anything; this tests the same on XGBoost, so the two together form a
# 2x2 (family x pipeline-origin) instead of two unrelated single-family readings. Neither
# arm alone can separate "ours-built members are worth nothing" from "this family is worth
# nothing"; the pair can.
#
# WHY THIS IS NOT THE CLOSED QUESTION. RESEARCH closed "tuning any GBDT is worth ~4e-7"
# on 2026-08-13. That closure is over knobs inside a fixed loss and a fixed function class.
# Neither member here changes a knob: F1 changes the LOSS and F2 changes the ADDITIVE
# EXPANSION, which is the dimension the same file says does move a member. Round count,
# depth, eta, subsample, colsample, min_child_weight, lambda, max_bin, max_cat_threshold
# and seed are all held at xgb_latcat's exact values, so each member is a ONE-VARIABLE
# contrast against a control already on disk (xgb_latcat, CV 0.967696).
#
# ONE JOB AT A TIME. Waits on the w26i SHELL SCRIPT, not on any python process -- at
# launch a queued stage's python does not exist yet and the wait would return instantly,
# which is the w25 §8 triple-booking failure that turned 123s fits into 270s ones.
#
# No `set -e`: a failure in one member must not cost the other, and run_xgb.py now
# checkpoints per fold (cache/xgbckpt/), so a kill costs the current fold and nothing more.
cd "$(dirname "$0")/.."

NEWDIR="$PWD/data/ext_members5"
mkdir -p "$NEWDIR"

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

wait_for w26i_run.sh
wait_for blend_lab.py          # belt and braces: w26i's last build could outlive its shell

# ---------------------------------------------------------------------------------
# 0. TIMING PROBES. Diagnostic only -- `--probe` carves its holdout out of fold 0's
#    TRAINING rows and saves nothing. These do NOT pick the round count: rounds stay
#    frozen at xgb_latcat's 3200 for both members, because moving them would make each
#    member a two-variable contrast and the whole design is one variable. What the
#    probes are for is (a) pricing the builds before 16 cores are committed to them,
#    which the record says to do and slots often do not, and (b) an early warning if
#    DART collapses -- a member that is merely WORSE and therefore decorrelated is the
#    documented null pattern here (xgb_cat_lattice, 0.9611 solo, bought nothing).
# ---------------------------------------------------------------------------------
echo "########## PROBE F1 xgb_latcat_l2  $(date -u) ##########"
timeout 7200 .venv/bin/python experiments/run_xgb.py \
  --name probe_l2 --mode latcat --objective reg:squarederror \
  --probe 0.08 --rounds 400 --probe-stopping 400 --threads 16

echo "########## PROBE F2 xgb_latcat_dart  $(date -u) ##########"
timeout 7200 .venv/bin/python experiments/run_xgb.py \
  --name probe_dart --mode latcat --rate-drop 0.10 --skip-drop 0.5 --one-drop \
  --probe 0.08 --rounds 400 --probe-stopping 400 --threads 16

# ---------------------------------------------------------------------------------
# 1. THE TWO MEMBERS. --outdir keeps them OUT of oof/: oof/ is in load_members' default
#    scan set, so saving there moves the pack under blend_lab and every gate stated
#    against a member COUNT, all at once and silently. See RESEARCH, w26 slot 5.
#    Every flag below except the one named on each line is xgb_latcat's exact value.
# ---------------------------------------------------------------------------------
echo "########## F1 xgb_latcat_l2  $(date -u) ##########"
.venv/bin/python experiments/run_xgb.py \
  --name xgb_latcat_l2 --mode latcat --objective reg:squarederror \
  --rounds 3200 --eta 0.03 --depth 7 --subsample 0.9 --colsample 0.5 \
  --min-child-weight 60 --reg-lambda 40 --max-bin 512 --max-cat-threshold 64 \
  --seed 13 --threads 16 --outdir "$NEWDIR"

# ⚠ THE PROBE UNDER-PRICES DART, AND QUADRATICALLY. Each DART round must undo the trees
# it drops, so the per-round cost grows with the number of trees already built and the
# TOTAL cost goes as rounds^2. The 400-round probe therefore sees ~1.6% of the 3200-round
# overhead, not 12.5%. `timeout` is the guard, and it is safe to hit one: run_xgb.py
# checkpoints per fold, so a timeout costs the fold in flight and re-running the identical
# command resumes. If this stage times out, RE-RUN IT -- do not conclude DART failed.
echo "########## F2 xgb_latcat_dart  $(date -u) ##########"
timeout 28800 .venv/bin/python experiments/run_xgb.py \
  --name xgb_latcat_dart --mode latcat --rate-drop 0.10 --skip-drop 0.5 --one-drop \
  --rounds 3200 --eta 0.03 --depth 7 --subsample 0.9 --colsample 0.5 \
  --min-child-weight 60 --reg-lambda 40 --max-bin 512 --max-cat-threshold 64 \
  --seed 13 --threads 16 --outdir "$NEWDIR"

# ---------------------------------------------------------------------------------
# 2. WHAT ARE THEY WORTH. Same instrument as w20d / w21b / w26i, deliberately unchanged:
#    paired 50/50 stratified splits, 5 reps, hybrid, C=1.0, the same rows with and
#    without the member. The maxcorr screen runs first, and w20d's `cat4` cell runs as a
#    REPRODUCTION GATE -- a miss invalidates the whole table and the script says so
#    itself. The pack it measures against now INCLUDES w26i's two CatBoosts, which is the
#    honest base: measuring against a pack missing them would overstate these.
# ---------------------------------------------------------------------------------
echo "########## w26j_value  $(date -u) ##########"
.venv/bin/python experiments/w26i_value.py \
  --new-dir "$NEWDIR" --new-names xgb_latcat_l2,xgb_latcat_dart \
  --reps 5 --out w26j_value \
  --prereg-note "PREREG §F3 said: each member 0 to +3e-6, modal +1e-6; the pair together
0 to +5e-6 on the combiner, modal +1.5e-6 -- BELOW the +3.1e-6 that w26d prices as an
even-money shot at the 0.97118 board record. The modal outcome of this build does not
produce a record file and must not be written up later as if it were expected to."

# ---------------------------------------------------------------------------------
# 3. THE BUILDS. No fresh 187 control here: w26i runs one immediately before this stage
#    on the identical code path, so the comparison base is that run's output rather than
#    a number recorded a wave ago. Adding a second control would cost ~40 minutes to
#    reproduce a number that will be an hour old.
#    ⚠ If w26i's members did not land, ext_members5 is added to a 187 pack and these are
#    189-member builds, not 191. Read the member count blend_lab prints; do not assume it.
# ---------------------------------------------------------------------------------
echo "########## xgb-augmented pack, h3  $(date -u) ##########"
.venv/bin/python experiments/blend_lab.py --build --reps 0 --standardize \
  --kinds hybrid,rankraw,rescale --extra-dirs ext_members3,ext_members4,ext_members5 \
  --submit-name w26j_xgb2std_h3

echo "########## xgb-augmented pack, ens4  $(date -u) ##########"
.venv/bin/python experiments/blend_lab.py --build --reps 0 --standardize \
  --extra-dirs ext_members3,ext_members4,ext_members5 \
  --submit-name w26j_xgb2std

# ---------------------------------------------------------------------------------
# 4. Requeue and reprice, so the next send day finds a correct queue without thinking.
# ---------------------------------------------------------------------------------
echo "########## requeue  $(date -u) ##########"
.venv/bin/python experiments/w23b_sendqueue.py
.venv/bin/python experiments/w26d_queueprice.py
echo "########## w26j done  $(date -u) ##########"
