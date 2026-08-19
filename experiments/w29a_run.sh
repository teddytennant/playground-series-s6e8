#!/usr/bin/env bash
# w29a -- build the four M14 arms on the full frozen folds, into data/ext_members8.
# Hyperparameters were fixed by a fold-0 / 200k-row probe (logged in w29a_probe.log);
# nothing here is tuned against the gate, which is maxcorr, not solo AUC.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w29a_build.log
: > "$LOG"
nice -n 10 $P experiments/w29a_newclass.py --arm qda_raw   --reg 0.05                >> "$LOG" 2>&1
nice -n 10 $P experiments/w29a_newclass.py --arm rff_raw   --dim 1024 --gamma 0.0125 >> "$LOG" 2>&1
nice -n 10 $P experiments/w29a_newclass.py --arm rff_lat   --dim 1024 --gamma 0.001  >> "$LOG" 2>&1
nice -n 10 $P experiments/w29a_newclass.py --arm poly2_raw                            >> "$LOG" 2>&1
echo "=== w29a done ===" >> "$LOG"
