#!/usr/bin/env bash
# w27m -- temporarily yield CPU to the critical path, then put it back.
#
# The box is at load ~47 on 16 cores with six resident jobs, and w27j was getting 11% of one
# core. Two of the six are long-horizon and checkpointed at a granularity that makes a pause
# free: w27c_ctdrop (per FOLD) and w27g_tunect (per CONFIG). SIGSTOP costs them nothing but
# wall-clock and loses no work at all -- unlike a kill, which would discard the partial fold.
#
# The resume is armed HERE rather than left to a later human step, because a job left stopped
# looks identical to a job that finished and would quietly stall the whole CT thread.
set -u
STOP="$1"          # comma-separated pids to pause
WAIT_PAT="$2"      # resume once no process matches this pgrep -f pattern
IFS=',' read -ra P <<< "$STOP"
for p in "${P[@]}"; do kill -STOP "$p" 2>/dev/null && echo "paused $p"; done
trap 'for p in "${P[@]}"; do kill -CONT "$p" 2>/dev/null; done; echo "resumed on trap"' EXIT INT TERM
# HARD DEADLINE as well as the wait condition. If the critical-path job dies in a way that
# leaves a matching process behind, or the pattern is wrong, the pause must still end by
# itself -- a stopped job is indistinguishable from a finished one and would stall the thread
# silently for however long nobody looks.
MAXW="${3:-7200}"
t0=$SECONDS
until ! pgrep -f "$WAIT_PAT" >/dev/null || (( SECONDS - t0 > MAXW )); do sleep 20; done
if (( SECONDS - t0 > MAXW )); then echo "DEADLINE ${MAXW}s hit -- resuming anyway"; else
  echo "critical path clear -- resuming"; fi
