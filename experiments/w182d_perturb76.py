"""w182d -- perturbs #76 w182c_elapsedguard against the live corpus, both directions.

Each probe rewrites JOURNAL.md in place for the length of one subprocess and restores it. The
probe helper REFUSES to score an edit that changed nothing -- w182b's lesson, applied to the
battery written an hour after it.
"""
from __future__ import annotations

import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
JOURNAL = os.path.join(ROOT, "JOURNAL.md")
GUARD = os.path.join(HERE, "w182c_elapsedguard.py")
ORIG = open(JOURNAL, encoding="utf-8").read()
FAILS = []

NEW_ENTRY = ("\n\n# 2026-09-06 — w9001 — CLOSED. ANGLE: probe.\n\n"
             "`kaggle competitions list` at **2026-09-06 10:00 UTC** returns deadline "
             "**2026-08-31 23:59**, past by ~{fig} days, `userRank 319`.\n")


def probe(name, text, want_rc, args=(), baseline=False, needle=""):
    if not baseline and text == ORIG:
        print(f"  FAIL {name:<58} NO-OP -- the edit matched nothing")
        FAILS.append(name + " (no-op)")
        return
    try:
        open(JOURNAL, "w", encoding="utf-8").write(text)
        p = subprocess.run([PY, GUARD, *args], capture_output=True, text=True, timeout=300)
    finally:
        open(JOURNAL, "w", encoding="utf-8").write(ORIG)
    ok = p.returncode == want_rc and (needle in p.stdout if needle else True)
    print(f"  {'OK  ' if ok else 'FAIL'} {name:<58} rc={p.returncode} want={want_rc}")
    if not ok:
        FAILS.append(name)
        for ln in p.stdout.split("\n"):
            if "FAIL" in ln:
                print("        " + ln.strip())


def main():
    print("w182d -- perturbing #76 w182c_elapsedguard\n")
    print("the shipped corpus")
    probe("shipped", ORIG, 0, baseline=True)
    probe("--control still fires", ORIG, 0, ("--control",), baseline=True)

    print("\na new run gets the figure right -> green")
    # 2026-09-06 10:00 is 5.42 d after 08-31 23:59
    probe("w9001 publishes 5.4 (the clock)", ORIG + NEW_ENTRY.format(fig="5.4"), 0)

    print("\na new run repeats the defect -> red")
    probe("w9001 publishes 6.4 (the calendar-day difference)",
          ORIG + NEW_ENTRY.format(fig="6.4"), 1, needle="the clock says")
    probe("w9001 publishes 5.9 (just outside TOL)", ORIG + NEW_ENTRY.format(fig="5.9"), 1)
    probe("w9001 publishes 5.5 (inside TOL, 0.08 off)", ORIG + NEW_ENTRY.format(fig="5.5"), 0)

    print("\nthe append-only arms")
    probe("a frozen claim REWRITTEN in place -> C3 red",
          ORIG.replace("past by ~5.6 days, `userRank 319` of `teamCount 3531`. Handed angle\nwas slot 5",
                       "past by ~4.6 days, `userRank 319` of `teamCount 3531`. Handed angle\nwas slot 5"),
          1, needle="REWRITTEN in place")
    probe("this run's repayment addendum deleted -> C4 red",
          ORIG.replace("past by **4.66** days", "past by roughly five days"), 1,
          needle="not repaid")

    print("\nthe diagnosis is checked, not asserted")
    # Move the deadline the entries name to midnight: then the calendar column IS the elapsed
    # column, the two stop separating, and C4 must say so rather than keep printing OK.
    probe("entries name a MIDNIGHT deadline -> the two columns merge, C4 red",
          ORIG.replace("deadline **2026-08-31 23:59**", "deadline **2026-08-31 00:00**"), 1,
          needle="no longer separate")

    print("\nthe reader tracks the text")
    probe("every claim sentence removed -> C1 red (not a silent pass)",
          ORIG.replace("past by ~", "past by about "), 1, needle="no longer reading it")

    print(f"\nFAILURES: {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
