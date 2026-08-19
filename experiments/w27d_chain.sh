#!/usr/bin/env bash
# w27d -- chain the 2000-round pack valuation behind the two jobs it depends on.
# Waits for w26k to pool (its _oof.npy appearing is the completion signal) and for the
# 400-round valuation to finish, then exports the 2000-round pair and measures it with the
# identical w26i_value.py instrument. Registered at w26_prereg.txt SS I2(a).
set -u
W=/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
cd "$W"
P="$W/.venv/bin/python"

while pgrep -f "w26k_ctscale.py --name ctscale" >/dev/null; do sleep 30; done
if [ ! -f experiments/w26k_ctscale_oof.npy ]; then
  echo "[w27d] w26k exited without pooling -- re-run it, it resumes per fold. STOP."
  exit 1
fi
echo "[w27d] w26k pooled."

while pgrep -f "w26i_value.py --new-dir data/ext_members6" >/dev/null; do sleep 30; done
echo "[w27d] w27b finished."

mkdir -p data/ext_members7
"$P" experiments/w26m_export.py --src experiments/w26k_ctscale --arm ct1.3333 \
     --name lat_ctfix2000 --out data/ext_members7 || exit 1
"$P" experiments/w26m_export.py --src experiments/w26k_ctscale --arm ct1.0000 \
     --name lat_ctraw2000 --out data/ext_members7 || exit 1

exec "$P" experiments/w26i_value.py --new-dir data/ext_members7 \
     --new-names lat_ctfix2000,lat_ctraw2000 --reps 5 --out w27d_ctvalue2000 \
     --prereg-note "PREREG §I2(a): Delta_2000 = value(pack+ctfix2000) - value(pack+ctraw2000) registered 0 to +5e-6, modal +1e-6. R-I3: a Delta of +1e-6 is a real mechanism and still NOT worth a file; the ship bar is cross-fitted CV 0.9701181879."
