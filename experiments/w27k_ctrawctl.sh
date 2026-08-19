#!/usr/bin/env bash
# w27k -- THE MISSING CONTROL. Journal w27 slot-2 §9 named this as the first thing the next
# slot should build, and it is one blend_lab invocation.
#
# The 188-member build (w27h) added `lat_ctfix_r400` and took +2.20e-6 of cross-fitted CV over
# the identical 187-member pack. That number is the value of ADDING A MEMBER; it says nothing
# about the CT correction, because no build exists in which the SAME member is present in its
# uncorrected form. This is that build: 187 + `lat_ctraw_r400`, same cache, same config, same
# frozen folds, differing from w27_ad188std by exactly one column's 4/3.
#
#     w27_ad188std     = 187 + lat_ctfix_r400   CV 0.9701093456 (ens4) / 0.9701114720 (h3)
#     w27_ad188raw     = 187 + lat_ctraw_r400   <- this
#     w23_ad187std     = 187                    CV 0.9700343    (ens4) / 0.9701092751 (h3)
#
# Then d(fix) - d(raw) is the correction's value on the SAME instrument that produced the
# +2.20e-6, rather than on the paired 50/50 instrument (w27b) whose two reps disagree in sign.
#
# ⚠ REGISTERED BEFORE THE RUN, and it is not a hopeful one. Two instruments already say
# `ctfix - ctraw` is a wash inside a 187-member pack. The registered expectation is
# **-2 to +3e-6, modal +0.5e-6** — i.e. the raw control should land close to the fix, and the
# +2.20e-6 should turn out to be per-member value rather than correction value. A delta above
# +3e-6 would be the surprise and would be the first pack-level evidence for the correction.
#
# ⚠ NOTHING IS SHIPPED FROM THE ARM COMPARISON. Choosing between w27_ad188std and
# w27_ad188raw on their own cross-fitted CVs is exactly the arm-selection optimism R-J2
# forbids; the ship candidate was fixed as the `ctfix` build in prereg §L before either
# number existed and does not move on this.
#
# Chained rather than launched: five heavy jobs are already resident and the box is at load
# ~35 on 16 cores. Bracket class in the pgrep pattern so the waiter cannot match itself.
set -u
W=/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8
cd "$W"
P="$W/.venv/bin/python"

# ⚠ WAIT ON MEMORY, NOT JUST ON A PID. This build peaks at ~5.7 GB and the first attempt
# fired while w21a was still running, drove a 31 GB box with a full 16 GB swap to ~1 GB free,
# and stalled BOTH jobs -- w21a's CPU time moved 40s in ten minutes of wall clock. Thrashing
# looks exactly like slowness and cost nearly an hour before anyone looked at `free`.
until ! pgrep -f "w21a_[a]d187corr.py|w27j_[c]tclass.py" >/dev/null; do sleep 60; done
until [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -gt 11000 ]; do
  echo "waiting for 11 GB available (have $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MB)"
  sleep 60
done
echo "=== critical path clear and memory available, starting the ctraw control ==="

# w27h's drop list with the CT arms swapped: keep ctraw, drop ctfix and ctfixte.
D=golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctfix_r400,lat_ctfixte_r400

nice -n 15 "$P" experiments/blend_lab.py --reps 0 --build --submit-name w27_ad188raw \
     --standardize --extra-dirs ext_members3,ext_members4,ext_members6 --drop "$D" || exit 1
nice -n 15 "$P" experiments/make_h3.py w27_ad188raw || exit 1
echo "=== w27k complete: compare w27_ad188raw_h3 against w27_ad188std_h3 0.9701114720 ==="
