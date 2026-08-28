"""w101a — STANDING GUARD: EVERY ANCHOR IN THE ANGLE INDEX MUST STILL RESOLVE.

WHAT CLAIM THIS COVERS, AND WHY NOTHING ELSE COVERS IT. RESEARCH.md's ANGLE INDEX (w101) is a
navigation table: seven closed modelling angles, each pointing at the section that holds its
number. Its whole value is that a handed ANGLE costs one grep instead of half a run. That value
is entirely contingent on the pointers resolving, and **the pointers rot silently**:

  * RESEARCH.md gets a new section prepended at the TOP every single run, so anything anchored
    to a LINE NUMBER is wrong by the next day. That is why the index anchors on header TEXT --
    and header text gets retitled too.
  * A run that retitles `# ⛔ CATBOOST TUNING IS CLOSED` breaks the index and gets no warning,
    because nothing reads the index except a human under time pressure.

⚠⚠ THE FAILURE MODE IS THE EXPENSIVE DIRECTION. A future run greps a dead anchor, gets nothing
back, concludes the closure was never written down, and re-runs a closed angle -- which is the
precise cost the index was built to prevent. A broken index is WORSE than no index, because no
index at least sends you to `grep -i` over the whole document.

🎯 THE BUG THIS CHECK EXISTS BECAUSE OF, AND IT WAS IN THE INDEX'S OWN FIRST DRAFT. w101 wrote
the table, then hand-verified the anchors with `grep -c`. Six came back with 2+ hits and one --
`THE ORIGINAL DATASET IS` -- came back with exactly **1**. One hit is not a pass: it is the
index's OWN copy of the string, and the target did not exist. The real header is
`## The original dataset — CLOSED, both routes measured here`. ⟹ **AN ANCHOR MUST BE COUNTED
OUTSIDE THE BLOCK THAT QUOTES IT.** A naive `grep -c >= 1` self-satisfies on every row, which
is the same shape as w100a's "a check whose universe is a filtered artefact measures the
filter": here, a check whose corpus includes the claim measures the claim.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +   every anchor in the index resolves at least once OUTSIDE the index block.
  C2 +-  FIRED BOTH WAYS. A planted anchor that exists nowhere must be reported DEAD, and the
         same corpus without it must come back clean. If neither moves, C1 is not testing.
  C3 +-  THE SELF-EXCLUSION REGRESSION, i.e. the actual w101 bug, fired in code rather than
         quoted as prose: an anchor that occurs ONLY inside the index must be DEAD under the
         live reader and would be ALIVE under the naive `count >= 1` reader. Both are computed.
  C4 +   NOT VACUOUS. The block must be found and must yield at least MIN_ROWS rows with at
         least MIN_ANCHORS anchors. A missing or emptied index goes RED, never green-over-
         nothing -- the vacuous-pass trap.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
JOURNAL = os.path.join(ROOT, "JOURNAL.md")
OUT = os.path.join(HERE, "w101a_angleguard.json")

BLOCK_HEAD = "# 📇 THE ANGLE INDEX"
# The index is deliberately small and fixed; if it shrinks, that is a regression, not a tidy-up.
MIN_ROWS, MIN_ANCHORS = 10, 12   # w110 added rows 9 (error analysis) and 10 (consolidation)

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def block_span(txt: str):
    """(start, end) of the ANGLE INDEX block. end = the next top-level '# ' header, or EOF.

    ⚠ RESEARCH.md's house style puts the provenance on a SECOND '# ' line under the title
    (`# 📇 THE ANGLE INDEX ...` / `# (w101, 2026-08-27) ...`). A naive "next \n# " therefore
    ends the block on its own subtitle and hands the parser one header and zero rows -- which
    is what the first draft of this file did, and C4 caught it on the first run. The leading
    run of '# ' lines is part of the header, so consume it before looking for the end.
    """
    i = txt.find(BLOCK_HEAD)
    if i < 0:
        return None
    k = i
    while True:
        eol = txt.find("\n", k)
        if eol < 0:
            return (i, len(txt))
        if not txt.startswith("# ", eol + 1):
            break
        k = eol + 1
    j = txt.find("\n# ", k)
    return (i, len(txt) if j < 0 else j)


def parse_rows(block: str):
    """The table rows, as (row_number, target_file, [anchors]).

    Anchors are the backticked strings in the LAST column only. Earlier columns carry prose
    and prices that happen to contain backticks, and sweeping those in would manufacture
    anchors nobody wrote -- a MENTION is not a POINTER (w91b's bug, in a different file).
    """
    rows = []
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 5 or not re.fullmatch(r"\d+", cells[0]):
            continue
        anchors = re.findall(r"`([^`]+)`", cells[-1])
        # A row naming JOURNAL sends its anchors there; everything else lives in RESEARCH.
        tgt = JOURNAL if "JOURNAL" in cells[-1] else RESEARCH
        rows.append((int(cells[0]), tgt, anchors))
    return rows


def resolve(anchor: str, target: str, research: str, journal: str, span) -> tuple[int, int]:
    """(hits_outside_the_index, hits_naive). The pair is the point -- C3 reads both."""
    txt = research if target == RESEARCH else journal
    naive = txt.count(anchor)
    if target != RESEARCH:
        return naive, naive          # the index is not in JOURNAL; nothing to exclude
    inside = research[span[0]:span[1]].count(anchor)
    return naive - inside, naive


def audit(research: str, journal: str, label: str):
    """Returns (rows, dead, n_anchors). Pure over its inputs so the controls can feed it
    doctored corpora instead of writing to disk."""
    span = block_span(research)
    if span is None:
        return None, [f"{label}: ANGLE INDEX block not found"], 0
    rows = parse_rows(research[span[0]:span[1]])
    dead, n = [], 0
    for num, tgt, anchors in rows:
        for a in anchors:
            n += 1
            outside, naive = resolve(a, tgt, research, journal, span)
            if outside < 1:
                dead.append(f"row {num}: {a!r} -> {outside} outside the index "
                            f"({naive} naive) in {os.path.basename(tgt)}")
    return rows, dead, n


def main() -> int:
    research = open(RESEARCH, encoding="utf-8").read()
    journal = open(JOURNAL, encoding="utf-8").read()

    print("w101a — the ANGLE INDEX's pointers must resolve outside the block that quotes them\n")

    # ---- C4: not vacuous. Do this FIRST; every other control is meaningless over an empty set.
    rows, dead, n_anchors = audit(research, journal, "live")
    if rows is None:
        fail(dead[0] + " -- the index is gone. This is a RED, not an empty pass.")
        json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                       failures=FAILS, rows=0, anchors=0), open(OUT, "w"), indent=1)
        print(f"\nFAILURES {FAILS}")
        return 1
    print(f"C4 not vacuous: {len(rows)} rows, {n_anchors} anchors "
          f"(floor {MIN_ROWS}/{MIN_ANCHORS})")
    if len(rows) < MIN_ROWS or n_anchors < MIN_ANCHORS:
        fail(f"index shrank to {len(rows)} rows / {n_anchors} anchors -- below the floor. "
             f"An angle row is only removed when the angle RE-OPENS, which has never happened.")

    # ---- C1: every anchor resolves outside the index.
    for d in dead:
        fail(d)
    print(f"C1 anchors resolving outside the index: {n_anchors - len(dead)}/{n_anchors}")

    # ---- C2 +-: a planted nowhere-anchor must be caught, and its removal must restore clean.
    ghost = "ZZ_NO_SUCH_HEADER_W101A_ZZ"
    doctored = research.replace(
        "| 7 |", f"| 8 | *control* | — | — | `{ghost}` |\n| 7 |", 1)
    _, dead_plus, _ = audit(doctored, journal, "C2+")
    _, dead_base, _ = audit(research, journal, "C2-")
    caught = any(ghost in d for d in dead_plus)
    print(f"C2 planted nowhere-anchor  caught={caught}  "
          f"dead {len(dead_base)} -> {len(dead_plus)}")
    if not caught:
        fail("C2: a planted anchor that exists nowhere was NOT reported dead -- C1 is not testing")
    if len(dead_plus) != len(dead_base) + 1:
        fail(f"C2: planting one ghost moved the dead count by "
             f"{len(dead_plus) - len(dead_base)}, expected 1")

    # ---- C3 +-: THE w101 BUG. An anchor present ONLY in the index must be dead, and the
    # naive reader must disagree. If the two readers agree here, the exclusion is not wired.
    self_only = "ZZ_ONLY_INSIDE_THE_INDEX_W101A_ZZ"
    doctored3 = research.replace(
        "| 7 |", f"| 8 | *control* | — | — | `{self_only}` |\n| 7 |", 1)
    span3 = block_span(doctored3)
    live_hits, naive_hits = resolve(self_only, RESEARCH, doctored3, journal, span3)
    print(f"C3 self-only anchor  live={live_hits} (want 0)  naive={naive_hits} (want >=1)")
    if live_hits != 0:
        fail("C3: an anchor occurring only inside the index was not excluded -- "
             "this is exactly the w101 `THE ORIGINAL DATASET IS` bug, live again")
    if naive_hits < 1:
        fail("C3: the naive reader did not see the self-only anchor, so the two readers are "
             "not being compared and C3 proves nothing")

    json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                   rows=len(rows), anchors=n_anchors, dead=dead,
                   c2_caught=caught, c3_live=live_hits, c3_naive=naive_hits,
                   failures=FAILS), open(OUT, "w"), indent=1)

    print(f"\nFAILURES {FAILS}")
    if FAILS == 0:
        print("OK: every ANGLE INDEX pointer resolves. A handed angle still costs one grep.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
