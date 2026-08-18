#!/usr/bin/env bash
# w26i -- two new CatBoost members, then what they are worth, then the 189-member builds.
# Pre-registered at experiments/w26_prereg.txt §E before any of it was started.
#
# ONE JOB AT A TIME. This waits for the w26h chain to exit before it starts anything.
# w25 §8 triple-booked 16 cores and turned 123s fits into 270s ones, and slot 4 repeated
# the lesson: wait on the SHELL SCRIPT, not on the python process, because at launch the
# python process of a queued stage does not exist yet and the wait would return instantly.
#
# No `set -e`: a failure in one member must not cost the other, and every stage resumes.
cd "$(dirname "$0")/.."

NEWDIR="$PWD/data/ext_members4"
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

wait_for w26h_run.sh
wait_for w26f_csweep          # belt and braces: w26h's own waiter could have raced

# ---------------------------------------------------------------------------------
# 1. THE TWO MEMBERS. Each is a ONE-VARIABLE contrast against a member already on disk,
#    and each is checkpointed per fold, so a kill costs the current fold and nothing more.
#    --outdir keeps them OUT of oof/: every reproduction gate in this repo is stated
#    against a fixed member count, and dropping these into oof/ would move the pack under
#    w26f, w26h and blend_lab simultaneously.
# ---------------------------------------------------------------------------------

# E1: cat_native at max_ctr_complexity 2. Identical to oof/summary_cat_native.json in every
# other parameter (mode native, lr 0.06, depth 6, l2 6.0, seed 7, border 254, 5000 iters).
echo "########## E1 cat_native_ctr2  $(date -u) ##########"
.venv/bin/python experiments/run_catboost.py \
  --name cat_native_ctr2 --mode native --ctr-complexity 2 \
  --iterations 5000 --lr 0.06 --depth 6 --l2 6.0 --seed 7 --border-count 254 \
  --threads 16 --outdir "$NEWDIR"

# E2: cat_lat's settings on the union representation. Identical to
# oof/summary_cat_lat.json except mode (lr 0.05, depth 7, l2 6.0, seed 7, 5000 iters).
echo "########## E2 cat_natlat  $(date -u) ##########"
.venv/bin/python experiments/run_catboost.py \
  --name cat_natlat --mode natlat \
  --iterations 5000 --lr 0.05 --depth 7 --l2 6.0 --seed 7 --border-count 254 \
  --threads 16 --outdir "$NEWDIR"

# ---------------------------------------------------------------------------------
# 2. WHAT ARE THEY WORTH. maxcorr screen first (R-E3), then the paired 50/50 instrument
#    with w20d's cat4 cell as the reproduction gate (R-E2). If the gate fails, the table
#    is not comparable to RESEARCH.md and the script says so in its own output.
# ---------------------------------------------------------------------------------
echo "########## w26i_value  $(date -u) ##########"
.venv/bin/python experiments/w26i_value.py --new-dir "$NEWDIR" --reps 5

# ---------------------------------------------------------------------------------
# 3. THE BUILDS. The 187-member control runs FIRST and must reproduce the numbers
#    logs_w23f_stdbuild4.txt recorded -- hybrid 0.970098 / rankraw 0.970092 /
#    rescale 0.970094 -- on this same code path. If it does not, the 189-vs-187
#    comparison is across code paths and means nothing, which is the failure this
#    workspace keeps rediscovering. Control first, so it exists even if the chain is cut.
# ---------------------------------------------------------------------------------
echo "########## CONTROL: 187-member h3, must reproduce w23f  $(date -u) ##########"
.venv/bin/python experiments/blend_lab.py --build --reps 0 --standardize \
  --kinds hybrid,rankraw,rescale --extra-dirs ext_members3 \
  --submit-name w26i_ctrl187_h3

echo "########## 189-member h3  $(date -u) ##########"
.venv/bin/python experiments/blend_lab.py --build --reps 0 --standardize \
  --kinds hybrid,rankraw,rescale --extra-dirs ext_members3,ext_members4 \
  --submit-name w26i_ad189std_h3

echo "########## 189-member ens4  $(date -u) ##########"
.venv/bin/python experiments/blend_lab.py --build --reps 0 --standardize \
  --extra-dirs ext_members3,ext_members4 \
  --submit-name w26i_ad189std

# ---------------------------------------------------------------------------------
# 4. Requeue and reprice, so the next send day finds a correct queue without thinking.
# ---------------------------------------------------------------------------------
echo "########## requeue  $(date -u) ##########"
.venv/bin/python experiments/w23b_sendqueue.py
.venv/bin/python experiments/w26d_queueprice.py
echo "########## w26i done  $(date -u) ##########"
