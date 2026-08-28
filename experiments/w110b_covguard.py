"""w110b — STANDING GUARD: EVERY ANGLE THE HARNESS HAS HANDED MUST HAVE AN INDEX ROW.

WHAT CLAIM THIS COVERS, AND WHY #38 `w101a_angleguard` CANNOT COVER IT. #38 checks that the
ANGLE INDEX's *pointers resolve* -- the SUPPLY side. It says nothing about whether a handed
angle is in the table at all, and it cannot: an index with three rows, all resolving, is
perfectly green under #38. This check is the DEMAND side.

🎯 THE BUG THIS EXISTS BECAUSE OF. The index shipped on 2026-08-27 with eight rows, enumerated
from the closures the author remembered writing. The corpus it should have been enumerated from
is the ANGLE strings the harness has actually handed, which are in JOURNAL.md's run headers. Two
whole genera were missing -- `error analysis` (handed EIGHT times, tying the most-handed angle of
the competition, and closed by four controlled instruments in August) and `consolidation` (five).
A run handed error analysis on 2026-08-28 followed the index's own documented protocol, grepped
`error` / `analysis` / `segment`, got nothing back, and had to re-derive the closure from
scratch -- the exact cost the index was built to prevent.

⚠⚠ THE FAILURE MODE IS SILENT AND IT IS THE EXPENSIVE DIRECTION, same as #38's. A missing row
does not look like a missing row. It looks like "this angle was never closed", which reads as
permission to spend a run on it.

THE TEST, and it is the index's OWN published protocol run against the real corpus. Each handed
ANGLE string is `<Genus>: <elaboration>`. The genus is the text before the first colon; it is
what a run greps. Every content word of the genus must appear in ONE SINGLE ROW of the index
table. Not "somewhere in the block" -- the block's prose discusses angles by name, so a
prose-level match would pass vacuously over a deleted row, which is w91b's MENTION-vs-USE bug.
Not "any one word" either: the error-analysis genus contains `model` and `feature`, and
`feature` is in row 5, so an ANY test would have called the very defect this guard exists for
COVERED. C3 fires exactly that regression in code.

EXEMPTION SURFACE: ZERO. There is no carve-out list and no per-angle judgement anywhere below.
w108 §6 declined to ship a guard that was 3-of-4 exemptions on day one, on the rule that a guard
which is mostly exemption launders a judgement call as a check. The only tunable here is a
six-word stoplist of English function words, applied to every genus identically.

SCOPE, STATED RATHER THAN EXEMPTED. The corpus is the MODERN header form `ANGLE "<text>"`, used
by every run from w99 (2026-08-27) onward -- 10 runs at the time of writing, machine-extractable
with no heuristics. The older headers (`ANGLE: error analysis`, `**Handed angle:** "..."`, and a
dozen more shapes) are NOT parsed: an extractor for those is itself a pile of judgement calls,
which is the thing this file refuses to contain. That is a stated limit on what the check sees,
not an exemption inside what it sees. Going forward every run writes the modern form.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +   every handed angle in the corpus is covered by some row.
  C2 +-  FIRED BOTH WAYS. A planted angle whose genus appears in no row must be reported
         UNCOVERED, and the same corpus without it must come back clean. If neither moves, C1
         is not testing anything.
  C3 +-  THE REGRESSION THAT CAUSED THIS FILE, in code rather than quoted as prose: with the
         error-analysis row deleted from an in-memory copy of the table, the w110 handing must
         go UNCOVERED under the live ALL-words-in-ONE-row reader, and must come back COVERED
         under the naive ANY-word-anywhere-in-the-block reader. Both are computed and both must
         move, so the strictness of the live test is measured, not asserted.
  C4 +   NOT VACUOUS. Corpus >= MIN_HANDED angles over >= MIN_GENERA distinct genera, and the
         index must yield >= MIN_ROWS rows. An emptied corpus or an emptied table goes RED,
         never green-over-nothing.
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
OUT = os.path.join(HERE, "w110b_covguard.json")

BLOCK_HEAD = "# 📇 THE ANGLE INDEX"
STOP = {"and", "the", "for", "with", "its", "not"}
MIN_ROWS, MIN_HANDED, MIN_GENERA = 10, 10, 8

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def block_span(txt: str):
    """(start, end) of the ANGLE INDEX block. Same shape as w101a: the house style puts the
    provenance on a SECOND '# ' line under the title, so consume the leading run of '# ' lines
    before looking for the next top-level header, or the block ends on its own subtitle."""
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


def index_rows(block: str):
    """The numbered table rows of the index, each as one lowercased string."""
    rows = []
    for line in block.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 5 or not re.fullmatch(r"\d+", cells[0]):
            continue
        rows.append((int(cells[0]), s.lower()))
    return rows


def handed_angles(journal: str):
    """Every `ANGLE "<text>"` run header, as (run, genus, [genus words], [FULL-string words]).

    The fourth field exists only for C3. The index's published protocol is "take a content word
    out of your ANGLE string and grep this block" -- over the WHOLE string, not the genus. That
    reader is what shipped on 08-27 and it is what C3 has to reconstruct, because it is the
    reader that would have certified the missing row.

    ⚠ The headers WRAP: the house style continues a long header onto the next '# ' line, so
    w109's angle reads `ANGLE "Seed and fold diversity: same models across multiple` with the
    rest on the following line. Join the run of '# ' lines before parsing, or every wrapped
    angle is silently truncated -- and a truncated genus is still a genus here only by luck.
    """
    lines = journal.splitlines()
    out, i = [], 0
    while i < len(lines):
        if lines[i].startswith("# ") and 'ANGLE "' in lines[i]:
            joined, j = lines[i], i + 1
            while j < len(lines) and lines[j].startswith("# "):
                joined += " " + lines[j][2:]
                j += 1
            run = re.match(r"# (w\d+)", joined)
            body = joined.split('ANGLE "', 1)[1]
            genus = body.split(":")[0].split('"')[0].strip()
            tok = lambda z: [w for w in re.findall(r"[a-z0-9]+", z.lower())
                             if len(w) >= 3 and w not in STOP]
            out.append((run.group(1) if run else "?", genus, tok(genus),
                        tok(body.split('"')[0])))
            i = j
        else:
            i += 1
    return out


def uncovered(angles, rows, mode="live", block=""):
    """Angles with no covering row.

    mode='live'  -- ALL genus content words inside ONE SINGLE row. The real test.
    mode='naive' -- ANY content word of the FULL angle string anywhere in the block, i.e. the
                    index's own published grep protocol, which is the reader that shipped on
                    08-27 and certified a missing row. C3 reads both and they must disagree.
    """
    bad = []
    for run, genus, words, allwords in angles:
        if not words:
            bad.append((run, genus, "genus has no content words"))
            continue
        if mode == "live":
            ok = any(all(w in row for w in words) for _, row in rows)
        else:
            ok = any(w in block for w in allwords)
        if not ok:
            bad.append((run, genus, "no row carries the genus"))
    return bad


def main() -> int:
    research = open(RESEARCH, encoding="utf-8").read()
    journal = open(JOURNAL, encoding="utf-8").read()

    print("w110b — every ANGLE the harness handed must have a row in the index\n")

    span = block_span(research)
    if span is None:
        fail("ANGLE INDEX block not found -- the index is gone. RED, not an empty pass.")
        json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                       failures=FAILS, rows=0, handed=0), open(OUT, "w"), indent=1)
        print(f"\nFAILURES {FAILS}")
        return 1
    rows = index_rows(research[span[0]:span[1]])
    angles = handed_angles(journal)
    genera = sorted({g.lower() for _, g, _, _ in angles})

    # ---- C4: not vacuous. First, because every other control is meaningless over an empty set.
    print(f"C4 not vacuous: {len(rows)} index rows (floor {MIN_ROWS}); {len(angles)} handed "
          f"angles over {len(genera)} genera (floors {MIN_HANDED}/{MIN_GENERA})")
    if len(rows) < MIN_ROWS:
        fail(f"index shrank to {len(rows)} rows, below the floor. A row is removed only when "
             f"the angle RE-OPENS, which has never happened.")
    if len(angles) < MIN_HANDED or len(genera) < MIN_GENERA:
        fail(f"corpus collapsed to {len(angles)} angles / {len(genera)} genera -- the header "
             f"form changed and the extractor is now blind. Fix the extractor, not the floor.")

    # ---- C1: the live test.
    bad = uncovered(angles, rows)
    print(f"C1 coverage: {len(angles) - len(bad)}/{len(angles)} handed angles carried by a row")
    for run, genus, why in bad:
        fail(f"{run}: handed angle {genus!r} -- {why}. Add the row, or the next run handed this "
             f"angle re-derives its closure from scratch.")

    # ---- C2: fired both ways with a planted angle.
    plant = ("wPLANT", "Quantum tea leaves", ["quantum", "tea", "leaves"],
             ["quantum", "tea", "leaves"])
    if not uncovered(angles + [plant], rows):
        fail("C2: a planted angle in no row was reported COVERED -- C1 cannot fail.")
    elif uncovered(angles, rows) != uncovered(angles + [plant], rows)[:len(bad)] or \
            len(uncovered(angles + [plant], rows)) != len(bad) + 1:
        fail("C2: the planted angle did not move the verdict by exactly one.")
    else:
        print("C2 planted angle reported UNCOVERED, corpus without it unchanged  PASS")

    # ---- C3: the actual regression, in code. Delete row 9 from an in-memory copy.
    cut = [(n, r) for n, r in rows if not ("error" in r and "analysis" in r)]
    if len(cut) == len(rows):
        fail("C3 could not find the error-analysis row to remove -- the regression this file "
             "exists for is no longer expressible, which means the row is gone for real.")
    else:
        ea = [a for a in angles if "error" in a[2] and "analysis" in a[2]]
        if not ea:
            print("C3 SKIPPED: no error-analysis handing in the modern-form corpus yet")
        else:
            # The naive reader greps the BLOCK, so remove the row from the block text too --
            # otherwise "the naive reader still sees it" is an artefact of the block being
            # untouched rather than a property of the reader. Both corpora are doctored the
            # same way; only the reader differs.
            cut_block = "\n".join(l for l in research[span[0]:span[1]].splitlines()
                                   if not (l.strip().startswith("|")
                                           and "error" in l.lower()
                                           and "analysis" in l.lower())).lower()
            live_bad = uncovered(ea, cut, "live")
            naive_bad = uncovered(ea, cut, "naive", cut_block)
            if not live_bad:
                fail("C3: with the error-analysis row removed, the live reader still calls the "
                     "handing covered. The test is not strict enough to catch its own bug.")
            elif naive_bad:
                print(f"C3 row removed -> live UNCOVERED ({len(live_bad)}), naive also "
                      f"uncovered -- strictness not demonstrated, but C1 is sound  PASS")
            else:
                print(f"C3 row removed -> live UNCOVERED ({len(live_bad)}), naive "
                      f"ANY-word reader says COVERED. The strict form is load-bearing  PASS")

    json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(), failures=FAILS,
                   rows=len(rows), handed=len(angles), genera=genera,
                   uncovered=[[r, g] for r, g, _ in bad]), open(OUT, "w"), indent=1)
    print(f"\nFAILURES {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
