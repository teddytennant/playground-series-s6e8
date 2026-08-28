#!/usr/bin/env bash
# w103b — DOES `setsid` SURVIVE A SESSION TEARDOWN, OR ONLY A TOOL CALL?
#
# WHY THIS EXISTS. RESEARCH long claimed `setsid` was not installed here (w103: false — it is at
# /run/current-system/sw/bin, invisible to the agent shell's curated PATH). So the systemd-user
# launch recipe was built to route around a tool that exists, and `setsid` itself was never once
# tried. It is UNTESTED, not known-bad.
#
# ⚠ THE TRAP THIS SCRIPT EXISTS TO AVOID. w103's first probe ran `setsid ... sleep 90` and watched
# it survive three Bash tool calls. That proves NOTHING: w102 recorded that plain `nohup` also
# "runs for as long as the session lives". Both survive a tool call; they differ only at SESSION
# TEARDOWN, which a session cannot trigger on itself. ⟹ the discriminating observation can only be
# made by a LATER RUN, so the probe has to outlive the run that starts it and leave a timestamped
# trail behind.
#
# HOW A FUTURE RUN READS IT:
#   tail -3 experiments/w103b_heartbeat.txt        # last beat, UTC epoch + human
#   git log -1 --format=%cI                        # when the starting run last committed
# If beats continue well past the starting session's end -> setsid DETACHES PROPERLY, and the
# four-attempt systemd dance in RESEARCH can be replaced by one line. If they stop at the session
# boundary -> setsid is no better than nohup, and the systemd recipe is load-bearing. Record the
# verdict in RESEARCH either way and delete this probe.
#
# Cost: one sleeping shell, ~0% CPU, 8 hours max, then it exits on its own.
OUT="$(cd "$(dirname "$0")" && pwd)/w103b_heartbeat.txt"
for _ in $(seq 1 960); do
    printf '%s  %s\n' "$(date -u +%s)" "$(date -u +%FT%TZ)" >> "$OUT"
    sleep 30
done
