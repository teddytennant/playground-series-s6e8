#!/usr/bin/env bash
# w27h -- the 188-member standardised build: the 187-member pack plus lat_ctfix_r400 only.
# Registered at w26_prereg.txt §L. w23f's chain verbatim, one member wider, float32 + C=1.0
# so w23_ad187std_h3 (0.9701092751) on disk is a like-for-like control needing no refit.
set -u
W=/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
cd "$W"
P="$W/.venv/bin/python"
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400

"$P" experiments/blend_lab.py --reps 0 --build --submit-name w27_ad188std --standardize \
     --extra-dirs ext_members3,ext_members4,ext_members6 --drop "$D" || exit 1
"$P" experiments/make_h3.py w27_ad188std || exit 1
W21A_BASE=w27_ad188std_h3 W21A_TAG=w27_ad188stdcorr exec "$P" experiments/w21a_ad187corr.py
