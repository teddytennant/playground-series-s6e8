#!/usr/bin/env bash
# w27p -- the xgb and cat legs of the pre-registered CT trio (prereg slot4 SS M5).
#
# CHAINED, not launched alongside w27o. w27o's xgb/cat cells are the SAME fits with the same
# params/seed/matrix, and the box is already five jobs deep at load 41. The marker
# "classes=lgb fold=0" is w27o's log line for "every xgb/cat cell is done".
#
# Also gates on memory: kill -STOP does not free memory, and a 5.7 GB blend job put swap at
# 100% on 08-18 while `uptime` still read like ordinary busyness (see JOURNAL 08-19 SS5).
# Diagnostic order for a stall is free -m, then ps --sort=-rss, then vmstat.
set -u
cd "$(dirname "$0")/.."
LOG=experiments/w27p_serveclass.log
MARK=experiments/w27o_ctclass5.log
: > "$LOG"

wait_for () {
  local waited=0
  while [ "$waited" -lt 43200 ]; do
    if grep -q "classes=lgb fold=0" "$MARK" 2>/dev/null || \
       grep -q "all cells done" "$MARK" 2>/dev/null; then
      local avail
      avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      if [ "${avail:-0}" -gt 6000 ]; then
        echo "=== w27o xgb/cat cells done, MemAvailable ${avail} MB, starting ===" >> "$LOG"
        return 0
      fi
      echo "=== waiting on memory: MemAvailable ${avail} MB (need >6000) ===" >> "$LOG"
    fi
    sleep 120; waited=$((waited + 120))
  done
  echo "=== gave up waiting after 12h ===" >> "$LOG"; return 1
}

wait_for || exit 1
for c in cat xgb; do          # cat first: ~37s/fit against xgb's ~57s
  echo "=== leg $c ===" >> "$LOG"
  nice -n 15 .venv/bin/python experiments/w27p_serveclass.py \
      --cls "$c" --rounds 400 --seed 42 --threads 3 >> "$LOG" 2>&1
done
echo "=== both legs done ===" >> "$LOG"
