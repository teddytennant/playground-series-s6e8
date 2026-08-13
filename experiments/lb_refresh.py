"""Turn `kaggle competitions submissions -v` into experiments/lb_scores.json.

Exists because `audit.py`'s hand-maintained LB dict went ten points stale on 2026-08-13,
after which the audit listed six already-submitted files as "never submitted". In a
workspace where the only reason to spend a daily slot is that the file is genuinely
different, that error spends one on a duplicate.

The API is the authority for what has been scored; nothing here should ever be typed by
hand again. Run it at the START of a run, before reading the audit's queue:

    kaggle competitions submissions -c playground-series-s6e8 -v \
      | python experiments/lb_refresh.py

Only COMPLETE submissions with a public score are written. Duplicate file names (the same
name submitted twice) keep the FIRST occurrence, which is the most recent -- the CLI
returns newest-first -- and warn, because a repeated name means two different builds have
been shipped under one identity and the CV attached to that name is ambiguous.
"""
from __future__ import annotations

import csv
import json
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb_scores.json")


def main():
    rows = list(csv.DictReader(sys.stdin))
    if not rows:
        raise SystemExit("no rows on stdin -- pipe `kaggle competitions submissions -v`")

    scores, dates, dupes = {}, {}, []
    for r in rows:
        name, sc = r.get("fileName", ""), (r.get("publicScore") or "").strip()
        if not name or not sc or "COMPLETE" not in r.get("status", ""):
            continue
        if name in scores:
            dupes.append((name, dates[name], r["date"][:19], scores[name], sc))
            continue
        scores[name] = float(sc)
        dates[name] = r["date"][:19]

    with open(OUT, "w") as f:
        json.dump(dict(sorted(scores.items())), f, indent=1)

    print(f"wrote {OUT}: {len(scores)} scored files from {len(rows)} submissions")
    for name, keep, drop, s_keep, s_drop in dupes:
        print(f"  ! duplicate file name {name}: kept {keep} ({s_keep}), "
              f"ignored {drop} ({s_drop}) -- two builds under one name")
    best = max(scores.values())
    tied = sorted(n for n, v in scores.items() if v == best)
    print(f"  best public {best:.5f}, held by {len(tied)}: {', '.join(tied)}")
    if len(tied) > 1:
        print("  -> Kaggle's DEFAULT final selection breaks this tie for you. Choose "
              "explicitly on the website; see RESEARCH.md 'Final selection'.")


if __name__ == "__main__":
    main()
