#!/usr/bin/env bash
# w41a -- RESTART of the w36g -> w38d -> w40f chain, which died at 18:44 UTC on 08-20 when the
# session that launched it ended. All three stopped at once; w38d and w40f never started at all
# (both were still printing "waiting"), so ARM 202 and ARM 211 do not exist.
#
# ⚠ THE FIX THAT MATTERS: this is launched with `setsid` so it detaches from the session's
# process group. The previous launch did not, and ~7 hours of build time was lost silently.
#
# ARM 197's base and h3 COMPLETED before the kill (submissions/w36_ad197std{,_h3}.csv, ens4
# CV 0.970121). Only the w21a correction stage was interrupted, so stage 1 resumes exactly
# there rather than rebuilding. w21a refits all five arms from scratch and is deterministic,
# so a restart is a clean rerun, not a partial resume.
#
# Stages 2 and 3 invoke the PRE-REGISTERED runners verbatim -- nothing about ARM 202 or ARM 211
# is redefined here. Their internal wait-loops clear immediately because the predecessor's log
# carries its `done` marker by then, which also keeps the original artefact-not-/proc gating.
set -u
cd "$(dirname "$0")/.."
P=.venv/bin/python
LOG=experiments/w36g_build.log

echo "=== w41a chain restart $(date -u): resuming w36g at the w21a stage ===" >> "$LOG"
W21A_BASE="w36_ad197std_h3" W21A_TAG="w36_ad197stdcorr" \
    nice -n 15 "$P" experiments/w21a_ad187corr.py >> "$LOG" 2>&1

echo "=== w36g done $(date -u). Bars: w34_ad195std_h3 0.9701205753 / w34_ad195stdcorr 0.9701247949" >> "$LOG"
echo "    WANTED moves only above 0.9701288, and ARM 197 is a CONTROL (prereg addendum) ===" >> "$LOG"

bash experiments/w38d_run.sh
bash experiments/w40f_run.sh
echo "=== w41a chain complete $(date -u) ===" >> experiments/w41a_chain.log
