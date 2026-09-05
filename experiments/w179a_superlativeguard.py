"""w179a -- A CENSUS SUPERLATIVE IN THE ANGLE INDEX MUST AGREE WITH THE CENSUS (2026-09-05).

🎯 THE DEFECT THIS EXISTS BECAUSE OF. Row 5's handing trail ended, for six days, with

    ... → w171 09-04 (closed) — the most-handed row** (count from `w117a_handcount`, ...)

`w117a_handcount` (#50) reads the SAME cell, parses the `×N` out of it, and compares N against
the corpus. It has been green throughout. It never looked at the clause after the count, and
that clause is a second, stronger claim about the same census: not "row 5 was handed 20 times"
but "no row was handed more". The first is checked every run; the second had never been checked
at all, and it was FALSE.

    census at the time of writing   rows 1,2,3,4,5,9,10 all ×20  -- a SEVEN-WAY tie at the top

⚠ AND IT WAS TRUE WHEN IT WAS TYPED. Reconstructed from git over every commit that touched
RESEARCH.md (see FROZEN_TIMELINE):

    115f3e9  08-30  row 5 unique max at ×16   clause introduced   <- TRUE
    58166b6  08-31  row 3 catches up, tie [3,5]                   <- FALSE, one day later
    ...      through 12 further commits, the tie growing to seven rows

🎯 THE MECHANISM IS THE ROTATION ITSELF, WHICH IS WHY THIS IS A CLASS AND NOT A TYPO. The
launcher hands the ten rows round-robin, so the counts are driven toward equality by design. A
uniqueness superlative over a round-robin census is a claim the scheduler actively destroys: it
can be true only in the brief window where one row is a single hand ahead, and the next pass
takes it away. Nothing that decays on its own should be written where only its neighbours are
audited. The `×N` beside it is checked every run precisely because it is expected to change;
the superlative was not, so it rotted in place.

⚠ THE SAME FILE ALREADY CONTRADICTED ITSELF AND NOBODY READ BOTH LINES. An archived section
says `error analysis ties feature engineering as the most-handed angle of the whole competition`
-- the tie, stated plainly, while the index cell three thousand lines below still called row 5
the unique maximum. Two statements of one census fact, and no reader compared them. That is
w178's shape (#72) at a different pair of surfaces, which is the reason to guard the class
rather than delete the sentence.

SCOPE, and why it is this narrow. Only the ten ANGLE INDEX trail cells are enforced. Prose
elsewhere in RESEARCH.md is HISTORICAL NARRATIVE -- "w116 found that row 5 was the most-handed
angle" is a true statement about 08-29 and must stay true, exactly like #72's frozen four. The
index is the operative table a run reads to learn the state NOW, so it is the one surface where
a decayed claim misleads. C4 checks the scope discriminates rather than merely suppresses.

    .venv/bin/python experiments/w179a_superlativeguard.py     # rc 0 = green

Deterministic: reads RESEARCH.md, JOURNAL.md and git. No API call, no model fit.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# C5: the census is IMPORTED, never re-implemented, so #73 and #50 cannot disagree about what a
# row is or how many times it was handed. Same rule #72 adopted for `classify`.
from w117a_handcount import BLOCK_HEAD, GENERA, census, check, claimed_counts  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"

# Answerability floors. #70's rule: a check with too little evidence FAILS rather than passing
# blind, because "nothing to check" and "everything checks out" print the same green otherwise.
MIN_RUNS, MIN_ROWS = 100, 10

# The superlative vocabulary, and which end of the census each phrase claims. A phrase is
# enforced only when it appears INSIDE one of the ten trail cells.
SUPERLATIVES = [
    (re.compile(r"most[- ]handed", re.I), "max"),
    (re.compile(r"least[- ]handed", re.I), "min"),
    (re.compile(r"fewest hand", re.I), "min"),
    (re.compile(r"handed (?:the )?most", re.I), "max"),
    (re.compile(r"handed (?:the )?least|handed fewest", re.I), "min"),
]

# The reconstructed life of the defect, frozen. Each entry is (commit, date, row-5 count, the
# rows tied at the maximum, whether the clause was in the file). C3 replays these against git,
# so the claim "it was true when written and false the next day" is evidence rather than a
# story -- and it cannot rot, because these commits are immutable.
FROZEN_TIMELINE = [
    ("115f3e9", "2026-08-30", 16, [5], True),        # introduced, and TRUE
    ("58166b6", "2026-08-31", 16, [3, 5], True),     # row 3 catches up -- FALSE from here
    ("6361962", "2026-09-03", 19, [4, 5, 9, 10], True),
]

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  {msg}. FAIL")


def trail_cells(text: str) -> dict[int, str]:
    """The ten ANGLE INDEX trail cells, keyed by row.

    Same block-identification rule as #50/#65/#66/#67: the ten-row table that follows
    BLOCK_HEAD. Column 2 is the trail -- the cell #50 parses `×N` out of, so #73 enforces the
    superlative on exactly the text #50 already reads the count from, and nowhere else.
    """
    out: dict[int, str] = {}
    start = text.index(BLOCK_HEAD)
    for line in text[start:start + 20000].split("\n"):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        row = int(m.group(1))
        if row in GENERA and row not in out:
            out[row] = cells[2]
    return out


def measured_counts(claims, runs) -> tuple[dict[int, int], list]:
    """The census counts EXACTLY as #50 computes them, adjustment and all.

    Not a re-count. #50's `check` already carries the current-run adjustment -- during a run
    the entry is not in JOURNAL.md yet, so the corpus is one short on the row being handed --
    and returns the same `measured` map it judges the index against. Taking that map instead of
    summing `census()` here is what makes C5's claim ("no disagreement with the census") true by
    construction rather than by two implementations happening to agree.
    """
    bad, measured = check(claims, runs, "w179a", quiet=True)
    return {r: measured.get(r, 0) for r in GENERA}, bad


def violations(cells: dict[int, str], counts: dict[int, int]):
    """Every superlative in a trail cell, paired with whether the census supports it.

    A `max` claim on row R holds iff R is the UNIQUE maximum. A tie is a failure: "the
    most-handed row" asserts that no other row equals it, which is what makes the claim worth
    more than the `×N` sitting beside it.
    """
    out = []
    hi, lo = max(counts.values()), min(counts.values())
    for row, cell in sorted(cells.items()):
        for pat, end in SUPERLATIVES:
            m = pat.search(cell)
            if not m:
                continue
            want = hi if end == "max" else lo
            tied = sorted(r for r, v in counts.items() if v == want)
            ok = tied == [row]
            out.append((row, m.group(0), end, counts[row], tied, ok))
    return out


def git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT,
                          capture_output=True, text=True).stdout


def main() -> int:
    text = RESEARCH.read_text(encoding="utf-8")
    runs = census()
    cells = trail_cells(text)
    claims = claimed_counts(text)
    counts, drift = measured_counts(claims, runs)

    print("C1 the corpus and the table are answerable")
    if len(runs) < MIN_RUNS:
        fail(f"only {len(runs)} run headers, under the {MIN_RUNS} floor")
    if len(cells) < MIN_ROWS:
        fail(f"only {len(cells)} trail cells parsed, expected {MIN_ROWS}")
    hi = max(counts.values())
    tied_hi = sorted(r for r, v in counts.items() if v == hi)
    print(f"  {len(runs)} run headers, {len(cells)} trail cells, max ×{hi} held by {tied_hi}")

    print("C2 LIVE: every superlative in a trail cell agrees with the census")
    vs = violations(cells, counts)
    if not vs:
        print("  no trail cell claims a superlative  OK")
    for row, phrase, end, n, tied, ok in vs:
        if ok:
            print(f"  row {row} {phrase!r}: ×{n}, unique {end}  OK")
        else:
            fail(f"row {row} claims {phrase!r} at ×{n}, but the {end} is held by {tied}")

    print("C3 FROZEN: the defect was true when written and false the next day")
    for commit, date, want5, want_tied, want_clause in FROZEN_TIMELINE:
        blob = git_show(commit, "RESEARCH.md")
        if not blob:
            fail(f"{commit} ({date}) unreadable")
            continue
        claims = claimed_counts(blob)
        got = {r: v for r, v in claims.items() if r in GENERA and v is not None}
        if len(got) != MIN_ROWS:
            fail(f"{commit}: parsed {len(got)} rows, expected {MIN_ROWS}")
            continue
        mx = max(got.values())
        tied = sorted(r for r, v in got.items() if v == mx)
        clause = "most-handed row" in blob
        state, want = (got[5], tied, clause), (want5, want_tied, want_clause)
        verdict = "OK" if state == want else "MISMATCH"
        if state != want:
            fail(f"{commit} ({date}): {state} != frozen {want}")
        note = "TRUE" if tied == [5] else "FALSE"
        print(f"  {commit} {date}  row5 ×{got[5]}  tied={tied}  clause={clause}  "
              f"claim was {note}  {verdict}")

    print("C4 the reader tracks its evidence, and its scope discriminates")
    probe_counts = dict(counts)
    # (a) a superlative on a row that is not the unique max -> caught
    not_max = min(counts, key=lambda r: counts[r])
    probe = {not_max: f"**×{counts[not_max]}, from 08-10 — the most-handed row**"}
    n = sum(1 for *_, ok in violations(probe, probe_counts) if not ok)
    print(f"  a superlative on the LEAST-handed row fires: {n} violation(s)  "
          f"{'OK' if n == 1 else 'BROKEN'}")
    if n != 1:
        fail("the reader does not catch a false superlative")
    # (b) the same phrase on a row that IS the unique max -> green, so it is not a blanket ban
    synth = {r: (99 if r == 5 else 1) for r in GENERA}
    n_ok = sum(1 for *_, ok in violations({5: "— the most-handed row"}, synth) if ok)
    print(f"  the same phrase where the census SUPPORTS it passes: {n_ok} upheld  "
          f"{'OK' if n_ok == 1 else 'BROKEN'}")
    if n_ok != 1:
        fail("the reader rejects a superlative the census supports")
    # (c) historical narrative outside the index is out of scope
    outside = "w116 found that row 5 was the most-handed angle and drew the lesson"
    in_cells = any(pat.search(c) for c in cells.values() for pat, _ in SUPERLATIVES)
    print(f"  narrative prose outside the ten cells is not covered: "
          f"{'covered' if outside in str(cells.values()) else 'not covered'}  OK")
    hist = len(re.findall(r"most[- ]handed", text, re.I))
    print(f"  {hist} 'most-handed' mention(s) in the file, {sum(1 for _ in vs)} inside a "
          f"trail cell -- the rest are history and stay")
    # (d) a cell with no superlative is silent
    n_quiet = len(violations({1: "**×20, from 08-10 → w176 09-05 (closed)**"}, probe_counts))
    print(f"  a cell claiming no superlative is not covered: {n_quiet} covered  "
          f"{'OK' if n_quiet == 0 else 'BROKEN'}")
    if n_quiet:
        fail("the reader covers a cell that makes no superlative claim")
    if in_cells and not vs:
        fail("a trail cell matches a superlative pattern that violations() did not report")

    print("C5 no disagreement with the census about what a row is")
    print("  census, GENERA and check() imported from w117a_handcount -- one implementation")
    if drift:
        fail(f"claimed vs measured drift on {[b[0] for b in drift]} -- #50 should be red too")
    else:
        print(f"  claimed == measured on all {len(GENERA)} rows  OK")

    print(f"\nFAILURES: {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
