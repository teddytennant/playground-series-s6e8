"""w117a — STANDING GUARD: THE ANGLE INDEX'S HANDING COUNTS MUST MATCH THE CORPUS.

WHAT THIS COVERS, AND WHY #45 `w110b_covguard` CANNOT COVER IT. #45 checks that every handed
angle is carried by SOME row -- coverage. w116 then showed a row can be present, resolving and
green while being nearly empty, and drew the lesson that the HANDING COUNT is the signal for
which row to enrich. That count is hand-maintained prose in the index and NOTHING checks it.

🎯 THE DEFECT THIS EXISTS BECAUSE OF. On 2026-08-29 the ten rows claimed
6/4/2/2/15/3/5/1/8/6 handings, 52 in total, against a corpus of 137 run headers. Row 6
(blending), the angle handed to the run that found this, claimed x3 against a measured x11.
Row 3 (CatBoost) claimed x2 against a measured x13. The counts were not merely stale, they were
IMPOSSIBLE: the harness hands the ten angles in a fixed round-robin, so the true distribution is
near-uniform by construction, and a table reading 2 ... 15 cannot describe it.

⚠ THE FAILURE MODE IS THE EXPENSIVE DIRECTION. w116 made frequency-of-handing a reason to spend
a run enriching a row. Computed from these numbers that heuristic points at row 5 (x15 claimed,
x14 real) and away from row 3 (x2 claimed, x13 real) -- it does not merely fail to help, it
actively misdirects the last runs before the deadline.

THE READER, AND WHY IT IS THE INDEX'S OWN PUBLISHED PROTOCOL. Each handed ANGLE is
`<Genus>: <elaboration>`; the index tells a run to grep the GENUS, the text before the first
colon, and warns that matching any word of the whole string is a false pass. This file obeys the
same rule, because a window reader gets it wrong in exactly the way the index warns about: the
XGBoost angle string ends "...tuned on the same folds so the blend weights mean something", so a
scan for `blend` anywhere near the header assigns FOUR XGBoost runs to blending. C4 measures
that difference rather than asserting it.

RESOLUTION ORDER IS HEADER-FIRST, BODY-FALLBACK, AND THE ORDER IS LOAD-BEARING. Broadening the
search to the run body first looks strictly better and is strictly worse: the body of a run that
REFUSED its angle discusses the angle repeatedly, so the first body match is usually the refusal
narration ("THE ANGLE WAS TAKEN AS ISSUED FOR THE FIRST TIME IN NINE RUNS") rather than the
angle. Measured: body-first left 42 of 137 runs unresolved and mis-resolved six that the header
had already answered. Same shape as w110's locator bug -- the first occurrence finds the
pointer, not the target.

OFF-ROTATION IS A REPORTED CATEGORY, NOT AN EXEMPTION. On 2026-08-14/15 the harness handed
bespoke angles ("Field forensics", "Row-identity structure", "act on group 1's strongest lead")
that are not among the ten. Those runs are counted and named in the OFF_ROTATION report, never
silently dropped -- a run that vanishes from a census is indistinguishable from one that was
classified, and the census total is what makes the round-robin argument checkable.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration).
  C1 +   NOT VACUOUS. >= MIN_RUNS run headers, >= MIN_RESOLVED resolved, all ten genera present.
  C2 +-  FIRES BOTH WAYS, PER ROW, on an in-memory copy of the index. The corrected table must
         score zero bad rows, and each of the ten rows perturbed ALONE must score exactly one.
         Perturbing a single row and asserting "still red" would have passed while nine rows
         were already red, which is w114's vacuous control: count the delta, not the total.
  C3 +-  THE REAL HISTORICAL DEFECT, frozen as a literal rather than read from git. w115 learned
         that a control anchored to HEAD stops being a control the moment the fix is committed.
         HISTORICAL_CLAIMS is the 2026-08-29 pre-fix tuple verbatim; the checker must report at
         least MIN_HISTORICAL_BAD rows wrong against it, else it declares itself INERT.
  C4 +-  THE READER'S STRICTNESS IS MEASURED, NOT ASSERTED. The naive whole-string reader the
         index warns against must claim MORE runs for `blending` than the genus reader does, and
         every surplus run must resolve to a genus that is NOT blending -- the claim is that the
         genus rule refuses them correctly, not merely that the two readers differ. At least
         XGB_BLEND_TRAP of the surplus must be XGBoost-genus runs, so the documented trap
         ("...so the blend weights mean something") is exercised and not just described. If both
         readers agree, the genus rule is doing nothing.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
JOURNAL = os.path.join(ROOT, "JOURNAL.md")
OUT = os.path.join(HERE, "w117a_handcount.json")

BLOCK_HEAD = "# 📇 THE ANGLE INDEX"
MIN_RUNS, MIN_RESOLVED, MIN_HISTORICAL_BAD = 100, 110, 5

# The run currently executing, whose journal entry is written after this check runs.
CURRENT_RUN, CURRENT_ROW = r"^# w117 —", 6

# Row number -> (label, pattern matched against the resolved GENUS only).
GENERA = {
    1:  ("original dataset",    r"original dataset|source dataset"),
    2:  ("LightGBM",            r"lightgbm"),
    3:  ("CatBoost",            r"catboost"),
    4:  ("XGBoost",             r"xgboost"),
    5:  ("feature engineering", r"feature engineer"),
    6:  ("blending",            r"blend"),
    7:  ("seed and fold",       r"seed ?(and|/) ?fold"),
    # ROW 8 IS HANDED IN TWO SURFACE FORMS AND ONE OF THEM OMITS ITS OWN LABEL. w40, w58, w76
    # and w96 quote the angle as "confirm the metric, build the fixed-fold CV harness, get one
    # honest GBDT baseline scored" -- row 8's elaboration verbatim, with the word `Foundation`
    # dropped. Matching the elaboration is not an exemption for those four runs: it is the same
    # rule every other row gets, applied to the wording this row is actually handed in. Every
    # other genus name does appear in every string that hands it.
    8:  ("foundation",          r"foundation|confirm the metric"),
    9:  ("error analysis",      r"error analysis"),
    10: ("consolidation",       r"consolidat"),
}

# The index's claimed counts as they stood at 2026-08-29T14:40Z, BEFORE w117's fix. Frozen as a
# literal on w115's rule: a control anchored to a moving reference (HEAD) stops being a control
# the moment you commit the fix.
HISTORICAL_CLAIMS = {1: 6, 2: 4, 3: 2, 4: 2, 5: 15, 6: 3, 7: 5, 8: 1, 9: 8, 10: 6}

# The XGBoost angle string ends "...tuned on the same folds so the blend weights mean something",
# so any reader that matches a word of the WHOLE string claims those runs for blending. C4
# requires the surplus to be non-empty AND to contain these, the documented trap.
XGB_BLEND_TRAP = 4

RUN_HDR = re.compile(r"^#{1,2} (20\d\d-\d\d-\d\d|w\d+[a-z]? —|wave )")
# THE DECLARING LINE, AS ONE RULE RATHER THAN A LIST OF SHAPES. Every run that states its angle
# in the body does it the same way: a label, then a QUOTED string. Enumerating the surface forms
# instead ("ANGLE was", "Angle issued:", "The assigned ANGLE (") left 37 runs unresolved and is
# an open-ended maintenance burden; requiring the quote is one rule, and it also excludes
# narration about the angle ("the journal over the angle, as the brief allows"), which is what a
# bare `\bangle\b` match picks up.
#
# ONE REGEX LOCATES THE DECLARATION AND PARSES IT. Using a looser regex to SELECT the line and
# a different one to PARSE it reintroduces w110's locator bug: the line
# "**THE ANGLE WAS SET ASIDE.** This run's ANGLE was \"CatBoost: ...\"" is selected on its second
# `angle` and was then parsed from its first, yielding the genus "SET ASIDE, DELIBERATELY".
# `Issued` is a label in its own right -- the 08-22/08-23 runs write `Issued: *"Blending: ..."*`
# under a heading that never repeats the word angle.
ANGLE_Q = re.compile(r"\b(?:angle|issued)\b.{0,30}?[\u201c\"]", re.I | re.S)
LABEL = re.compile(r"\b(?:angle|issued)\b", re.I)
CUT = re.compile(r"[:.—]|\bslot\b|\bAT CAP\b|\bno submission\b|\bOVERRIDDEN\b|\bDECLINED\b"
                 r"|\bREFUSED\b|\bSUBSTITUTED\b", re.I)

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def genera_of(s: str):
    """Candidate genus strings for one line, best first.

    Two shapes, and BOTH are tried rather than one being preferred outright. Where the angle is
    quoted -- `ANGLE (as handed): "Blending: ..."` -- the genus is the text after the opening
    quote. Where it is bare -- `ANGLE was consolidation: re-verify the best pipeline` -- the
    genus follows the label directly. Taking the text after the quote is what makes the first
    robust: stripping the label with a regex has to guess how many words of `handed: ` /
    `as issued (` to eat, and it guessed wrong on the 08-18 wave, returning "handed" for five
    runs. The gap between label and quote may contain words, so the search for the quote is
    non-greedy over any 30 characters, not over non-letters.

    ⚠ THE QUOTE IS NOT ALWAYS THE ANGLE. w27 slot 7 writes
    `**ANGLE: seed and fold diversity, "cheap variance reduction that reliably adds a little"`,
    where the quoted phrase is an aside INSIDE the angle and the genus precedes it. Preferring
    the quote unconditionally reads the genus as "cheap variance reduction" and loses the run.
    So both candidates are returned and the caller keeps whichever classifies.
    """
    out = []
    m = ANGLE_Q.search(s)
    if m:
        out.append(s[m.end():])
    m = LABEL.search(s)
    if m:
        out.append(s[m.end():])
    res = []
    for tail in out:
        tail = re.sub(r"^[^A-Za-z]*", "", tail)
        res.append(CUT.split(tail)[0].strip()[:70])
    return res


def genus_of(s: str) -> str:
    """The single best-guess genus, for reporting an unresolved run."""
    g = genera_of(s)
    return g[0] if g else ""


def classify(genus: str):
    hits = [r for r, (_, pat) in GENERA.items() if re.search(pat, genus, re.I)]
    return hits[0] if len(hits) == 1 else None


def resolve(lines, i, end):
    """Header first, body second, and quoted declarations before bare ones.

    THE ORDER IS LOAD-BEARING. Broadening the search to the run body first looks strictly better
    and is strictly worse: the body of a run that REFUSED its angle discusses that angle
    repeatedly, so the first body match is usually the refusal narration ("THE ANGLE WAS TAKEN AS
    ISSUED FOR THE FIRST TIME IN NINE RUNS") rather than the angle. Measured: body-first left 42
    of 137 runs unresolved and mis-resolved six the header had already answered. Same shape as
    w110's locator bug -- the first occurrence finds the pointer, not the target.

    Within the body, quoted declarations are swept before bare ones for the same reason: a run
    that substitutes its angle narrates the substitution in prose above the quote that says what
    was actually handed. Narration is not excluded by a pattern here -- it is excluded because
    `classify` rejects it. "the journal over the angle, as the brief allows" yields the genus
    "as the brief allows", which names none of the ten, so the scan moves on.
    """
    for cand in genera_of(lines[i]):
        if classify(cand):
            return cand, "header", lines[i]
    for span in (45, 160, 400):
        win = lines[i:min(end, i + span)]
        for quoted_only in (True, False):
            for k, l in enumerate(win):
                if quoted_only and not ANGLE_Q.search(l):
                    continue
                if not LABEL.search(l):
                    continue
                s = l
                for j in (k + 1, k + 2):
                    if j < len(win) and win[j].strip():
                        s += " " + win[j]
                for cand in genera_of(s):
                    if classify(cand):
                        return cand, "body" if quoted_only else "body-bare", s
    return genus_of(lines[i]), "unresolved", lines[i]


def census():
    lines = open(JOURNAL, encoding="utf-8").read().split("\n")
    hdr = [i for i, l in enumerate(lines) if RUN_HDR.match(l)]
    runs = []
    for k, i in enumerate(hdr):
        end = hdr[k + 1] if k + 1 < len(hdr) else len(lines)
        g, how, raw = resolve(lines, i, end)
        runs.append({"line": i + 1, "header": lines[i][:90], "genus": g,
                     "row": classify(g), "how": how, "raw": raw})
    return runs


def naive_blending(runs):
    """The reader the index explicitly warns against: match ANY word of the WHOLE angle string
    rather than its genus. The XGBoost angle ends "...so the blend weights mean something", so
    this reader claims those runs for blending. Same string, same runs; only the rule differs."""
    return [r["line"] for r in runs if re.search(r"blend", r["raw"], re.I)]


def claimed_counts(text: str):
    """Parse `| N | *angle* | ...x12...` rows out of the ANGLE INDEX block."""
    start = text.index(BLOCK_HEAD)
    out = {}
    for line in text[start:start + 20000].split("\n"):
        m = re.match(r"^\|\s*(\d+)\s*\|", line)
        if not m:
            continue
        row = int(m.group(1))
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        c = re.search(r"[×x](\d+)", cells[2])
        out[row] = int(c.group(1)) if c else None
    return out


def check(claims, runs, tag, quiet=False):
    """Returns the list of rows whose claimed count disagrees with the measured one."""
    measured = {}
    for r in runs:
        if r["row"]:
            measured[r["row"]] = measured.get(r["row"], 0) + 1
    # THE CURRENT RUN COUNTS, AND THE ADJUSTMENT MUST NOT DOUBLE-COUNT WHEN ITS ENTRY LANDS.
    # w117 was handed `blending` and writes its journal entry at the END of the run, so during
    # the run the corpus is one short. Keying the adjustment on whether the entry is already in
    # the corpus makes this correct in BOTH states rather than only the one it was written in --
    # a constant `+1` here would silently become a double-count an hour later.
    if not any(re.search(CURRENT_RUN, r["header"]) for r in runs):
        measured[CURRENT_ROW] = measured.get(CURRENT_ROW, 0) + 1
    bad = []
    for row, (label, _) in GENERA.items():
        want, got = measured.get(row, 0), claims.get(row)
        if got != want:
            bad.append((row, label, got, want))
    if not quiet:
        for row, label, got, want in bad:
            fail(f"{tag} row {row} ({label}): index claims x{got}, corpus has x{want}")
    return bad, measured


def main() -> int:
    global FAILS
    runs = census()
    resolved = [r for r in runs if r["row"]]
    off = [r for r in runs if not r["row"]]

    print(f"C1 corpus: {len(runs)} run headers, {len(resolved)} resolved to one of the ten "
          f"genera, {len(off)} off-rotation/unresolved")
    if len(runs) < MIN_RUNS:
        fail(f"C1 vacuous: only {len(runs)} run headers (< {MIN_RUNS})")
    if len(resolved) < MIN_RESOLVED:
        fail(f"C1 vacuous: only {len(resolved)} resolved (< {MIN_RESOLVED})")

    text = open(RESEARCH, encoding="utf-8").read()
    claims = claimed_counts(text)
    if len(claims) < 10:
        fail(f"C1 vacuous: parsed only {len(claims)} index rows (< 10)")

    print("\nLIVE CHECK — index vs corpus")
    bad, measured = check(claims, runs, "live")
    for row, (label, _) in sorted(GENERA.items()):
        mark = "  " if (row, label, claims.get(row), measured.get(row, 0)) not in bad else "!!"
        print(f"  {mark} row {row:2d} {label:20s} claimed x{str(claims.get(row)):3s}  "
              f"corpus x{measured.get(row,0):2d}")
    if any(r["row"] is None for r in runs):
        print(f"\nOFF-ROTATION / UNRESOLVED ({len(off)}), named not dropped:")
        for r in off:
            print(f"     L{r['line']:6d}  {r['genus'][:48]:48s}  {r['how']}")

    print("\nC2 fires both ways (in-memory perturbation)")
    corrected = dict(measured)
    b_ok, _ = check(corrected, runs, "C2", quiet=True)
    moved = 0
    for row in GENERA:
        pert = dict(corrected)
        pert[row] = corrected[row] + 3
        b_bad, _ = check(pert, runs, "C2", quiet=True)
        if len(b_bad) == len(b_ok) + 1:
            moved += 1
    if b_ok:
        fail(f"C2: the corrected table is not clean -- {len(b_ok)} rows still bad")
    elif moved != len(GENERA):
        fail(f"C2: only {moved} of {len(GENERA)} single-row perturbations were caught")
    else:
        print(f"  corrected table -> 0 bad; each of the {len(GENERA)} rows perturbed alone -> "
              f"exactly 1 bad. fires both ways, per row. OK")

    print("\nC3 the frozen historical defect")
    b4, _ = check(HISTORICAL_CLAIMS, runs, "C3", quiet=True)
    if len(b4) < MIN_HISTORICAL_BAD:
        fail(f"C3 INERT: only {len(b4)} rows wrong against the frozen 08-29 claims "
             f"(< {MIN_HISTORICAL_BAD}) -- the checker no longer catches the defect it exists for")
    else:
        print(f"  {len(b4)} of 10 rows wrong against the frozen pre-fix table. catches it. OK")

    print("\nC4 the genus reader is stricter than the window reader")
    naive = set(naive_blending(runs))
    genus_lines = {r["line"] for r in runs if r["row"] == 6}
    surplus = sorted(naive - genus_lines)
    by_line = {r["line"]: r for r in runs}
    print(f"  window reader assigns {len(naive)} runs to blending, genus reader "
          f"{len(genus_lines)}; surplus {len(surplus)}")
    if not surplus:
        fail("C4: both readers agree, so the genus rule is not doing anything")
    else:
        # The claim is not merely "the readers differ" -- it is that every run the window reader
        # ADDS is one the genus rule correctly refuses. Each surplus run must have a genus of
        # its own that is not blending, and at least XGB_BLEND_TRAP of them must be XGBoost.
        wrong = [n for n in surplus if by_line[n]["row"] == 6]
        xgb = [n for n in surplus if by_line[n]["row"] == 4]
        for n in surplus:
            r = by_line[n]
            print(f"     L{n:6d}  window says blending, genus says "
                  f"{GENERA.get(r['row'], ('off-rotation',))[0]:20s} {r['genus'][:40]}")
        if wrong:
            fail(f"C4: surplus contains runs that ARE blending: {wrong}")
        elif len(xgb) < XGB_BLEND_TRAP:
            fail(f"C4: only {len(xgb)} of the surplus are XGBoost-genus runs "
                 f"(< {XGB_BLEND_TRAP}) -- the documented trap is not being exercised")
        else:
            print(f"  every surplus run has a non-blending genus; {len(xgb)} are the XGBoost "
                  f"\"so the blend weights mean something\" trap. OK")

    json.dump({"measured": measured, "claimed": claims,
               "bad_rows": [list(b) for b in bad], "runs": len(runs),
               "resolved": len(resolved), "off_rotation": len(off)},
              open(OUT, "w"), indent=1, sort_keys=True)
    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
