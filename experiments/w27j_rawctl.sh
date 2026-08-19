#!/usr/bin/env bash
# w27j -- THE CONTROL that makes w27h's +2.20e-6 interpretable.
#
# w27h built 187 + lat_ctfix_r400 (the CT-corrected lattice LightGBM) and got h3
# 0.9701114720 against the 187-member 0.9701092751, i.e. +2.197e-6. That number is the value
# of adding ONE MORE LATTICE LIGHTGBM, not the value of the correction: w20d already prices a
# marginal lgb member at ~2.34e-6, and w27b's paired instrument gives the SKEWED member the
# larger marginal value in rep 1 (+5.4e-6 vs +3.0e-6).
#
# This builds the identical object with lat_ctraw_r400 -- the same member, same cache, same
# config, same fold split, differing ONLY in the 4/3. Then
#
#     w27_ad188std_h3  -  w27_ad188raw_h3
#
# is the pack-level price of the CT fix, paired on everything else. Registered expectation
# before it runs: ~0, on the same grounds as prereg §L4 (§H2's closure -- the 187-member span
# already contains what the 12 columns can say) and on w27b's two reps.
#
# ⚠ blend_lab's default extra_dirs is (ext_members, ext_members2) ONLY. The 187-pack is
# ext_members{,2,3,4}. Read the "N members" line: it must say 188.
#
# ~20 min of fitting on an idle box; it took 998s under load ~59 for w27h.
set -u
W=/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
cd "$W"
P="$W/.venv/bin/python"
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctfix_r400,lat_ctfixte_r400

"$P" experiments/blend_lab.py --reps 0 --build --submit-name w27_ad188raw --standardize \
     --extra-dirs ext_members3,ext_members4,ext_members6 --drop "$D" || exit 1
exec "$P" experiments/make_h3.py w27_ad188raw
