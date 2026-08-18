#!/usr/bin/env bash
# no set -e: a kill in one transform must not cost the other two
cd "$(dirname "$0")/.."
for k in hybrid rankraw rescale; do
  echo "########## $k ##########"
  .venv/bin/python experiments/w25d_stdholdout.py --kind "$k" --reps 5
done
echo "########## combine ##########"
.venv/bin/python experiments/w25d_stdholdout.py --combine --reps 5
