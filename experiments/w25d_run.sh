#!/usr/bin/env bash
# no set -e: a kill in one transform must not cost the other two.
# Every rep is checkpointed to w25d_{hold,arms}_<kind> as it finishes, so re-running this
# script after a kill resumes rather than restarting. w25d was killed mid-run twice before
# that was true and lost every completed rep both times.
cd "$(dirname "$0")/.."
for k in hybrid rankraw rescale; do
  echo "########## $k ##########"
  .venv/bin/python experiments/w25d_stdholdout.py --kind "$k" --reps 5
done
echo "########## combine ##########"
.venv/bin/python experiments/w25d_stdholdout.py --combine --reps 5
