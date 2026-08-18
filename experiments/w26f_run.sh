#!/usr/bin/env bash
# One job at a time -- w25 §8 triple-booked 16 cores and turned 123s fits into 270s ones.
# Every cell is checkpointed, so re-running this after a kill resumes rather than restarts.
cd "$(dirname "$0")/.."
for k in hybrid rankraw rescale; do
  echo "########## $k ##########"
  .venv/bin/python experiments/w26f_csweep.py --kind "$k"
done
echo "########## report ##########"
.venv/bin/python experiments/w26f_csweep.py --report
