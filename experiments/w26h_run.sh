#!/usr/bin/env bash
# Waits for the WHOLE w26f chain (w26f_run.sh, which itself waits on w26a) to exit, then
# builds the 28 C-grid candidate files. One job at a time -- w25 §8 triple-booked 16 cores.
#
# It waits on w26f_run.sh rather than on w26f_csweep, because at launch time w26f_csweep is
# not running yet: w26f_run.sh is still in its own wait loop behind w26a. Checking for the
# python process would let this start immediately and triple-book the box.
#
# w26h_cbuild.py refuses to build anything unless w26f's --cv C=1.0 cells reproduce the
# global-scale build in logs_w23f_stdbuild4.txt, so a broken or incomplete chain upstream
# ends this quietly instead of writing 28 files defended by the wrong number.
cd "$(dirname "$0")/.."

echo "waiting for the w26f chain to exit..."
while :; do
  alive=0
  for p in /proc/[0-9]*/cmdline; do
    case "$( { tr "\0" " " < "$p"; } 2>/dev/null )" in
      *w26f_run.sh*|*w26f_csweep*) alive=1;;
    esac
  done
  [ "$alive" = 0 ] && break
  sleep 30
done
echo "w26f chain clear at $(date -u), starting w26h"

.venv/bin/python experiments/w26h_cbuild.py --dry || exit 1
for k in hybrid rankraw rescale; do
  echo "########## $k ##########"
  .venv/bin/python experiments/w26h_cbuild.py --kind "$k" || exit 1
done
echo "########## mix ##########"
.venv/bin/python experiments/w26h_cbuild.py --mix

echo "########## requeue ##########"
.venv/bin/python experiments/w23b_sendqueue.py
.venv/bin/python experiments/w26d_queueprice.py
