#!/usr/bin/env bash
# w102a -- resume the w96 build and price it, end to end, with no human read in the middle.
#
# w96d's replication gate already CONFIRMED at the tuned vector (logs_w96d_replicate_then_build
# .txt, A/B/C all PASS) and its build step was still running when the previous session ended,
# which killed it mid-fold with oof_w96/ empty. So this re-runs step 3 ALONE. It deliberately
# does not re-run the gate: the gate is withdraw-only and it has already been applied to a
# complete five-fold replication, so running it again could only ever confirm the same file.
set -u

# Hardcode the workspace rather than deriving it with `dirname`. This script runs as a
# transient systemd user unit, whose PATH holds almost nothing -- `dirname` is not on it, so
# `cd "$(dirname "$0")/.."` silently became `cd /`, and the build then failed on a missing
# .venv from the wrong directory. Give it a usable PATH too, for the same reason.
export PATH="/run/current-system/sw/bin:/usr/bin:/bin:$PATH"
cd /home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8 || exit 1
P=.venv/bin/python

# Fail loudly instead of running the build from somewhere unexpected. The failure this
# guards against printed "No such file or directory" from 30 lines further down, which reads
# like a broken build and is really a broken cd.
[ -x "$P" ] || { echo "NOT IN THE WORKSPACE: $P missing from $PWD"; exit 1; }

# Own the log file rather than inheriting a caller's redirect. This script is launched as a
# transient systemd user unit so it survives the session that starts it -- the previous
# attempt at this build was killed mid-fold when its session ended -- and a unit's stdout
# goes to the journal, which is not where any run here looks. So point it at a file.
exec > logs_w102a_build_then_price.txt 2>&1

echo "== build (w96d step 3, resumed) =="
"$P" -u experiments/w96c_build_teprior_member.py || {
  echo "BUILD CRASHED -- nothing is decided. oof_w96/ may hold a partial arm; delete it"
  echo "before re-running so a half-written arm cannot be priced as a whole one."; exit 1; }

echo
bash experiments/w97b_value_then_gate.sh none
