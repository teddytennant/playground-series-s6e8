#!/usr/bin/env bash
# w27w -- the M11 partition replication. Pre-registered at
# experiments/w27_prereg_slot6.txt §M11, appended before w27w_partition.py was written.
#
# ⚠ CHAINED, not concurrent. Two reasons, both measured rather than assumed:
#   1. MEMORY. 12 GB available with 6 GB of swap already in use, and w27t is holding a
#      190x691k x 4-transform load. w25d's first version was OOM-killed by the OS partway
#      through transform 1 with NO traceback; that is the failure mode this avoids.
#   2. The box is at load ~30 on 16 cores with w27r_blockdrop, w27j_ctclass, w26i_value
#      and two foreign projects already running.
#
# ⚠ NEW FILE rather than an edit of any running script: bash re-reads a running script
# from its current byte offset, so editing one switches the live job mid-flight
# (w27 slot 4 §12).
#
# Waits on the w27t LAUNCHER pid, which is also what w27u_resume.sh watches, so arm 2 of
# w27s comes back at the same moment. That is deliberate: arm 2 is ~300s/fit and mostly
# blocked on lbfgs iterations, w27w is ~20s/fit, and they interleave rather than contend.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/w27w_partition.log
: > "$LOG"

WAIT=${1:-}
if [ -n "$WAIT" ]; then
  echo "waiting on pid $WAIT" >> "$LOG"
  while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
fi
echo "=== w27t finished, starting the M11 partition replication ===" >> "$LOG"

# 3 BLAS threads: polite on a loaded box, and the fits are lbfgs-iteration-bound rather
# than BLAS-bound at 190 columns, so more threads buy very little.
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
  nice -n 15 .venv/bin/python experiments/w27w_partition.py --seeds 42,101,13,7 \
  >> "$LOG" 2>&1

echo "=== w27w done ===" >> "$LOG"
