#!/usr/bin/env bash
# Waits for w25d to exit, THEN runs w26a. Never concurrently: w25 §8 triple-booked 16 cores
# and turned 123s fits into 270s ones while the log looked frozen. One job at a time.
# No `set -e`: a kill in one mode must not cost the others, and every mode resumes from the
# rows already in w26a_sensitivity.csv.
cd "$(dirname "$0")/.."

echo "waiting for w25d to exit..."
while :; do
  alive=0
  for p in /proc/[0-9]*/cmdline; do
    case "$( { tr "\0" " " < "$p"; } 2>/dev/null )" in *w25d_stdholdout*) alive=1;; esac
  done
  [ "$alive" = 0 ] && break
  sleep 20
done
echo "w25d clear at $(date -u), starting w26a"

# ORDER MATTERS. subset and logitnoise are the two arms Q1 is read from (w26_prereg §3);
# rawnoise is the confounded arm kept only to MEASURE the confound for Q4. If this run is
# cut short, the arms that decide the question are the ones already on disk.
for m in subset logitnoise rawnoise; do
  echo "########## $m ##########"
  .venv/bin/python experiments/w26a_sensitivity.py --mode "$m"
done
echo "########## report ##########"
.venv/bin/python experiments/w26a_sensitivity.py --report
