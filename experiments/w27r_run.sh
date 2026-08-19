#!/usr/bin/env bash
# w27r -- the feature-block ablation ladder. Pre-registered at w27_prereg_slot5.txt §M7.
# Cheapest arm first (§M7(e)): encdrop 40 cols, tedrop 112, rawdrop 144. A kill at any point
# therefore costs the LEAST informative arm, not the most.
# Per-(arm,fold) checkpointed; an identical re-run resumes and refits nothing.
# ⚠ Never edit this file while it is running -- bash re-reads from the current byte offset
# (journal 2026-08-19 §12 rule 2). Write a new file instead.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/w27r_blockdrop.log
: > "$LOG"
for arm in encdrop tedrop rawdrop; do
  echo "=== arm=$arm ===" >> "$LOG"
  nice -n 15 .venv/bin/python experiments/w27r_blockdrop.py \
      --arm "$arm" --rounds 400 --seed 42 --jobs 3 >> "$LOG" 2>&1
done
echo "=== all arms done ===" >> "$LOG"
