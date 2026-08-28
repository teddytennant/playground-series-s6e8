#!/usr/bin/env bash
# w109a -- the duplicate-column census + the five-arm ablation. One process, one load,
# so all five arms share the w28 BLAS threading floor and the deltas are attributable.
export PATH=/run/wrappers/bin:/run/current-system/sw/bin:/usr/bin:/bin:$PATH
set -u
cd /home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8 || exit 1
[ -x .venv/bin/python ] || { echo "no venv"; exit 1; }
exec > experiments/w109a_run.log 2>&1
echo "=== w109a start $(date -u) ==="
nice -n 15 .venv/bin/python experiments/w109a_dupscan.py --scan --arms
echo "=== w109a done $(date -u) rc=$? ==="
