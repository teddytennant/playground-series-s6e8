#!/usr/bin/env bash
# w34d -- validate the w34 builds and rebuild the send queue, so the 08-21 UTC day opens on
# a correct queue with `w34_ad195stdcorr` pinned at its head by the WANTED priority column.
#
# ⚠ THIS IS THE STEP THAT ACTUALLY MATTERS. check_selection.py's own comment records that
# the slot-1 pick sat UNSENT FOR FOUR SLOTS while three journal entries said "send it
# first", and the cause was mechanical: w26g_send.py sends the top of w26d_queueprice.csv,
# and a stale queue could not see files that did not exist when it was built. The queue goes
# stale the moment anything is built, and this slot built seven new CSVs.
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

echo "########## w28b_verify  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w28b_verify.py
echo "########## requeue  $(date -u) ##########"
nice -n 15 .venv/bin/python experiments/w23b_sendqueue.py
nice -n 15 .venv/bin/python experiments/w26d_queueprice.py
echo "########## w34d done  $(date -u) ##########"
