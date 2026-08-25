"""w86a — every Kaggle-list call site must ask for MORE ROWS THAN THIS ACCOUNT WILL EVER HAVE.

THE DEFECT THIS WAS WRITTEN FOR (found 2026-08-25, w86).
The account is at 141 submissions and sends 10 a day until 2026-08-31, so it finishes at
**201**. Twenty-two call sites in `experiments/` ask for `--page-size 200`. The CLI returns
exactly min(page_size, total) and says NOTHING — verified here, 140 -> 140 rows, 141 -> 141.
So on **the deadline day, and only then**, those twenty-two silently lose the oldest row.

⚠ ONE OF THEM IS `w84a_pickargmax.py`, THE GUARD THE DEADLINE PICK RESTS ON. It runs on
08-31, after the sends, when the list is 201 long. That is the single day it has to be right.

⚠⚠ AND ITS OWN ANTI-TRUNCATION CONTROL PASSES ANYWAY. w84a's G2 compares the paginated fetch
against the un-paginated one and passes when paginated > capped. At 201 rows that reads
200 > 50 and prints "✅ pagination exercised" while the fetch it just blessed is truncated.
**G2 tests that the cap beats 50. It never tested that the cap beats the history.** w84 wrote
down the exact lesson -- "a sample-size floor cannot detect truncation, only a comparison
against the un-capped call can" -- and then built the replacement with the same blind spot one
level up. w84a's `MIN_SAMPLE = 60` is that floor, and it cannot see this either.

THE ONLY TEST THAT DETECTS ITS OWN TRUNCATION IS SELF-REFERENTIAL: `len(rows) >= PAGE` ->
raise. `w55a_unpriced.py` has had it all along. G4 requires it of the deadline-day scripts.

GUARDS
  G1    every Kaggle list call site in experiments/ passes a page size at all.
  G2    every page size EXCEEDS the projected final row count, taken live, with margin.
  G3 -  a planted call site at the old 200 MUST fail G2 -- the bar is fired, not just stated.
  G4    the scripts that run on the deadline day compare their row count against THEIR OWN
        page size (either `>= PAGE -> raise` or `assert len < <that size>`; both detect it).
  G5 +- the silent-truncation premise is re-measured against the live API, not asserted.

⛔ DO NOT satisfy G2 by lowering the projection. The projection is (live rows) + (slots that
   remain before the deadline) and both come from outside this file.
⛔ DO NOT count this as covering row-count DRIFT generally. It bounds one failure mode: a
   fixed cap outliving the history it reads.

    .venv/bin/python experiments/w86a_pagecap.py          # 0 = ok, 1 = a guard failed
"""
from __future__ import annotations

import datetime as dt
import io
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
OUT = os.path.join(HERE, "w86a_pagecap.json")

DEADLINE = dt.date(2026, 8, 31)      # last UTC day on which a submission can be sent
DAILY_CAP = 10                       # RESEARCH "Competition basics", confirmed twice
MARGIN = 1.20                        # a cap must clear the projection by 20%, not by 1 row
PAGE = 500                           # what this file requires call sites to use

# Scripts that are RUN ON THE DEADLINE DAY and whose truncation would be silent. G4's list.
DEADLINE_DAY = ["w84a_pickargmax.py", "w26g_send.py", "w23b_sendqueue.py",
                "w55a_unpriced.py", "w54a_vetoexpiry.py", "w85c_slotguard.py",
                "w74b_clickstaleguard.py"]

LIST_VERBS = ('"submissions"', '"leaderboard"')
SIZE_RE = re.compile(r'"--page-size",\s*(?:"(\d+)"|str\((\w+)\))')
CONST_RE = r"^{}\s*=\s*(\d+)\s*(?:#.*)?$"   # a trailing comment is normal


def api_rows(page: int) -> pd.DataFrame:
    env = dict(os.environ)
    env.setdefault("KAGGLE_CONFIG_DIR", os.path.expanduser("~/.kaggle"))
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                        "--page-size", str(page)],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        raise SystemExit(f"submissions API failed: {r.stderr.strip()[:300]}")
    df = pd.read_csv(io.StringIO(r.stdout))
    if len(df) >= page:                       # the self-referential guard, applied to itself
        raise SystemExit(f"w86a's own fetch returned {len(df)} >= page {page}; raise it")
    return df


def call_sites(text: str, name: str):
    """(line, page_size or None) for every kaggle list invocation in one file."""
    out = []
    for m in re.finditer(r'\[\s*"kaggle"[^\]]*\]', text, re.S):
        blob = m.group(0)
        if not any(v in blob for v in LIST_VERBS):
            continue
        line = text[:m.start()].count("\n") + 1
        sm = SIZE_RE.search(blob)
        if sm is None:
            out.append((line, None))
            continue
        if sm.group(1) is not None:
            out.append((line, int(sm.group(1))))
        else:                                  # str(PAGE) -> resolve the module constant
            cm = re.search(CONST_RE.format(re.escape(sm.group(2))), text, re.M)
            out.append((line, int(cm.group(1)) if cm else None))
    return out


def main() -> int:
    bad, today = [], dt.datetime.now(dt.timezone.utc).date()

    # ---- the projection, both terms taken from outside this file -----------------------
    df = api_rows(PAGE)
    sent_today = int((df.date.str[:10] == today.isoformat()).sum())
    days_after_today = max(0, (DEADLINE - today).days)
    slots_left = (DAILY_CAP - sent_today) + DAILY_CAP * days_after_today
    projected = len(df) + slots_left
    bar = int(projected * MARGIN)
    print(f"live rows {len(df)}   sent today {sent_today}   whole days after today "
          f"{days_after_today}\nslots left {slots_left}  ->  PROJECTED FINAL {projected}"
          f"   bar = {MARGIN:g}x = {bar}\n")

    # ---- G5: re-measure the premise rather than asserting it ---------------------------
    n = len(df)
    short = api_rows_raw = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", str(n - 1)],
        capture_output=True, text=True, timeout=600).stdout
    got = len(pd.read_csv(io.StringIO(short)))
    if got != n - 1:
        print(f"  ⚠ G5 page-size {n-1} returned {got}, expected {n-1} -- premise CHANGED")
        bad.append(f"G5 truncation premise no longer holds ({got} != {n-1})")
    else:
        print(f"  ✅ G5 cap is silent and exact: page-size {n-1} -> {got} rows, no warning")

    # ---- G1 / G2 over every call site --------------------------------------------------
    sites, under = [], []
    for p in sorted(os.listdir(HERE)):
        if not p.endswith(".py") or p == os.path.basename(__file__):
            continue
        text = open(os.path.join(HERE, p)).read()
        for line, size in call_sites(text, p):
            sites.append((p, line, size))
            if size is None:
                bad.append(f"G1 {p}:{line} kaggle list call with NO page size (cap 50)")
            elif size < bar:
                under.append((p, line, size))
    for p, line, size in under:
        bad.append(f"G2 {p}:{line} page-size {size} < bar {bar}")
    print(f"  {'✅' if not under else '❌'} G1/G2 {len(sites)} call sites, "
          f"{len(under)} under the bar")
    for p, line, size in under[:25]:
        print(f"       {p}:{line}  page-size {size}")

    # ---- G3: negative control -- the old 200 must be rejected by this bar ---------------
    # ⚠ THE PLANTED SIZE IS ASSEMBLED, NOT WRITTEN OUT. w86's own repo-wide
    # `sed 's/"--page-size", "200"/"--page-size", "500"/'` rewrote this control's literal and
    # G3 caught its own disarming. A negative control that matches the pattern being swept
    # will be swept. Keep it un-greppable.
    OLD_CAP = 200
    planted = ('x = ["kaggle", "competitions", "submissions", "-c", C, "-v", '
               f'"--page-size", "{OLD_CAP}"]')
    psites = call_sites(planted, "<planted>")
    if len(psites) != 1 or psites[0][1] != OLD_CAP:
        bad.append(f"G3 the parser did not read the planted site: {psites}")
    elif OLD_CAP >= bar:
        bad.append(f"G3 bar {bar} does not reject the old {OLD_CAP} -- the guard is inert")
    else:
        print(f"  ✅ G3 planted page-size {OLD_CAP} site is parsed and rejected by bar {bar}")

    # ---- G4: deadline-day scripts carry the self-referential guard ----------------------
    missing = []
    for name in DEADLINE_DAY:
        fp = os.path.join(HERE, name)
        if not os.path.exists(fp):
            missing.append(f"{name} (absent)")
            continue
        t = open(fp).read()
        # Two idioms are equally sound and both appear on disk: `len(rows) >= PAGE -> raise`
        # (w55a, w54a) and `assert len(sub) < 500` (w74b). What makes either work is that the
        # bound IS THE FILE'S OWN PAGE SIZE, so accept the symbol or that exact literal --
        # and nothing else. A comparison against some other number is not this guard.
        bounds = {"PAGE"} | {str(sz) for _, sz in call_sites(t, name) if sz is not None}
        alt = "|".join(sorted(map(re.escape, bounds), key=len, reverse=True))
        if not re.search(rf"len\(\s*\w+\s*\)[^\n]{{0,16}}(>=|<)\s*({alt})\b", t):
            missing.append(name)
    for name in missing:
        bad.append(f"G4 {name} has no self-referential len(rows) >= PAGE truncation guard")
    print(f"  {'✅' if not missing else '❌'} G4 {len(DEADLINE_DAY) - len(missing)}"
          f"/{len(DEADLINE_DAY)} deadline-day scripts carry the >= PAGE guard")

    import json
    json.dump({"live_rows": len(df), "sent_today": sent_today, "slots_left": slots_left,
               "projected_final": projected, "bar": bar, "call_sites": len(sites),
               "under_bar": len(under), "g4_missing": missing, "failures": bad},
              open(OUT, "w"), indent=2)
    print(f"\nFAILURES {len(bad)}")
    for b in bad:
        print("  ❌", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
