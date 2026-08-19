#!/usr/bin/env bash
# w27u -- resume the SUSPENDED ridge_sweep arm once w27t has finished with the cores.
#
# w27 slot 6 suspended the M8(b) unstandardised control arm with `kill -STOP` (RESEARCH:
# "Yield cores with kill -STOP / kill -CONT, do not kill long jobs") so the 190-member
# build could finish inside the slot. A suspended process nobody resumes is a SILENT
# failure -- worse than a slow one, because no log line ever says it is stuck. This
# watchdog removes that failure mode: it resumes the arm when w27t exits, or after a
# 3-hour deadline, whichever comes first.
#
# $1 = pid to resume   $2 = pid to wait on
set -u
cd "$(dirname "$0")/.."
LOG=experiments/w27u_resume.log
STOPPED=$1
WAITON=$2
DEADLINE=$(( $(date +%s) + 10800 ))

echo "$(date -u +%FT%TZ) watchdog up: will CONT $STOPPED when $WAITON exits (deadline 3h)" > "$LOG"
while kill -0 "$WAITON" 2>/dev/null && [ "$(date +%s)" -lt "$DEADLINE" ]; do sleep 30; done

if kill -0 "$STOPPED" 2>/dev/null; then
    kill -CONT "$STOPPED"
    echo "$(date -u +%FT%TZ) CONT sent to $STOPPED; state now $(ps -o stat= -p "$STOPPED")" >> "$LOG"
else
    echo "$(date -u +%FT%TZ) pid $STOPPED no longer exists -- nothing to resume" >> "$LOG"
fi
