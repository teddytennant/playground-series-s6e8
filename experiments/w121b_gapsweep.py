"""w121b -- MEASUREMENT ONLY. Sweep #50's label-to-quote window and diff the whole census.

`ANGLE_Q` binds an angle label to the quote that follows it, but only within 30 characters.
w121a measured that the three unread Foundation runs sit at gaps 48-58, because each narrates
its REFUSAL between the label and the quote. The window is therefore a tunable, and this picks
its value by measuring the full-corpus effect of every candidate rather than by taste.

Nothing is written. Same manoeuvre as w119b: monkeypatch the reader, diff per run.
"""
from __future__ import annotations
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w117a_handcount as W                                     # noqa: E402

ORIG = W.ANGLE_Q
base = {r["line"]: (r["row"], r["how"], r["genus"]) for r in W.census()}


def census_with(gap: int):
    W.ANGLE_Q = re.compile(rf"\b(?:angle|issued)\b.{{0,{gap}}}?[“\"]", re.I | re.S)
    try:
        return {r["line"]: (r["row"], r["how"], r["genus"]) for r in W.census()}
    finally:
        W.ANGLE_Q = ORIG


def counts(d):
    c = {}
    for row, _, _ in d.values():
        if row:
            c[row] = c.get(row, 0) + 1
    return c


TARGETS = {17388: "w40", 21007: "w58", 25187: "w76"}
print(f"corpus: {len(base)} runs, {sum(counts(base).values())} resolved at the shipped gap=30\n")
print(" gap  resolved  newly-resolved  re-assigned  targets-recovered")
for gap in (30, 40, 50, 55, 60, 70, 80, 100, 140, 200):
    new = census_with(gap)
    assert base.keys() == new.keys(), "censuses disagree on the run headers themselves"
    newly = [ln for ln in base if base[ln][0] is None and new[ln][0] is not None]
    reassigned = [ln for ln in base
                  if base[ln][0] is not None and new[ln][0] is not None
                  and base[ln][0] != new[ln][0]]
    lost = [ln for ln in base if base[ln][0] is not None and new[ln][0] is None]
    got = sum(1 for ln in TARGETS if new[ln][0] == 8)
    flag = "  <-- LOST %s" % lost if lost else ""
    print(f" {gap:4d}  {sum(counts(new).values()):8d}  {len(newly):14d}  "
          f"{len(reassigned):11d}  {got}/3{flag}")

print()
for gap in (50, 60, 200):
    new = census_with(gap)
    print(f"--- gap={gap}: every run whose assignment moves ---")
    for ln in sorted(base):
        if base[ln][:1] != new[ln][:1] or base[ln][1] != new[ln][1]:
            tag = TARGETS.get(ln, "")
            print(f"  L{ln:<6} {tag:4s} row {str(base[ln][0]):4s} ({base[ln][1]:11s})"
                  f" -> row {str(new[ln][0]):4s} ({new[ln][1]:11s})  genus={new[ln][2][:46]!r}")
    cb, cn = counts(base), counts(new)
    moved = [r for r in sorted(W.GENERA) if cb.get(r, 0) != cn.get(r, 0)]
    print(f"  row counts that move: {[(r, cb.get(r,0), cn.get(r,0)) for r in moved]}\n")
