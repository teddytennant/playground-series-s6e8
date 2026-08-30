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
  C5 +-  THE AMBIGUITY RULE IS POSITIONAL AND FIRES BOTH WAYS. `classify` used to return None
         when a genus named two of the ten, which lost w14d out of row 9 and into OFF_ROTATION.
         It now takes the one named FIRST. At least MIN_AMBIGUOUS resolved runs must actually
         name two genera -- else the rule is INERT and says so -- and the negative direction is
         the mirrored probe: the same two genera in the opposite order must resolve the
         opposite way, which is what separates "positional" from "prefers the lower row".
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
MIN_RUNS, MIN_RESOLVED, MIN_HISTORICAL_BAD, MIN_AMBIGUOUS = 100, 110, 5, 1

# The run currently executing, whose journal entry is written after this check runs.
CURRENT_RUN, CURRENT_ROW = r"^# w126 —", 6

# Row number -> (label, pattern matched against the resolved GENUS only).
GENERA = {
    1:  ("original dataset",    r"original dataset|source dataset"),
    2:  ("LightGBM",            r"lightgbm"),
    3:  ("CatBoost",            r"catboost"),
    4:  ("XGBoost",             r"xgboost"),
    5:  ("feature engineering", r"feature engineer"),
    6:  ("blending",            r"blend"),
    7:  ("seed and fold",       r"seed ?(and|/) ?fold"),
    # ROW 8 IS HANDED IN TWO SURFACE FORMS AND ONE OF THEM OMITS ITS OWN LABEL. w40, w76 and
    # w96 quote the angle as "confirm the metric, build the fixed-fold CV harness, get one
    # honest GBDT baseline scored" -- row 8's elaboration verbatim, with the word `Foundation`
    # dropped. Matching the elaboration is not an exemption for those runs: it is the same rule
    # every other row gets, applied to the wording this row is actually handed in. Every other
    # genus name does appear in every string that hands it.
    #
    # This comment named w58 here too, and used the missing word to explain why w40/w58/w76 went
    # unread. That was wrong twice over: w58 quotes `*"Foundation: confirm the metric,` WITH the
    # word, and w96, which genuinely lacks it, was being read correctly all along. The three were
    # invisible for an unrelated reason measured in w121 -- the label-to-quote window, see
    # ANGLE_Q. A stated cause that nothing checks is how a row keeps a hand-written `+3`.
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

# C6's frozen corpus: the three Foundation-angle runs whose refusal narration sat between
# the label and the quote. Verbatim from JOURNAL.md, which is append-only.
REFUSAL_DECLS = [
    ("w40", '**The ANGLE as issued is stale and I did not follow it.** It asks to '
            '"confirm the metric, build the fixed-fold CV harness, and get one honest '
            'GBDT baseline scored" \u2014 day-one work, finished t'),
    ("w58", '**\u26a0 THE ANGLE IS STALE AND I AM SUBSTITUTING, ON THE RECORD.** '
            '*"Foundation: confirm the metric, build the fixed-fold CV harness, and get '
            'one honest GBDT baseline scored"* \u2014 the metric is AU'),
    ("w76", '\u26a0 **THE PROMPT\'S ANGLE IS OBSOLETE AND I DID NOT FOLLOW IT.** It reads '
            '*"confirm the metric, build the fixed-fold CV harness, get one honest GBDT '
            'baseline scored"* \u2014 that is the w1\u2013w5 brief '),
]
# At least this many resolved runs must need a gap > 30, else C6 declares itself inert.
MIN_WIDE_GAP = 3

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
# THE LABEL-TO-QUOTE WINDOW IS 80, AND ITS VALUE IS NOT A TUNING CHOICE. At 30 it silently
# dropped the three runs that REFUSED the Foundation angle: each narrates the refusal
# BETWEEN the label and the quote ("...ANGLE as issued is stale and I did not follow it.**
# It asks to \"confirm the metric..."), which pushes the quote to a gap of 48-58. So the
# window was selecting on whether a run OBEYED its angle -- a bias aimed straight at the
# rows that get refused most. `w121b_gapsweep.py` swept 30..5000 over the whole corpus:
# every value >= 55 yields the IDENTICAL census (130 resolved, 3 recovered, 0 re-assigned,
# 0 lost) and nothing changes again out to 5000, past the length of any joined declaration.
# 80 sits inside that flat region rather than on its edge. C6 keeps both directions.
ANGLE_Q = re.compile(r"\b(?:angle|issued)\b.{0,80}?[\u201c\"]", re.I | re.S)
# The pre-fix window, frozen as a literal on w115's rule so C6 keeps a fixed reference.
ANGLE_Q_NARROW = re.compile(r"\b(?:angle|issued)\b.{0,30}?[\u201c\"]", re.I | re.S)
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
    """The genus a string names, resolved POSITIONALLY: the pattern that matches earliest.

    ⚠ THIS RETURNED None ON AMBIGUITY UNTIL 2026-08-29, AND THAT SILENTLY LOST A RUN. w14d's
    header is `ANGLE: error analysis on the best blend's OOF, the generator's coin-flip band`.
    The angle string carries no colon, so `CUT` never trims it and the whole clause becomes the
    genus -- and that clause names `error analysis` at offset 0 and `blend` at offset 27. Two
    hits, so the exactly-one rule refused it; the body scan found no quoted declaration; and a
    ROTATION run landed in OFF_ROTATION. The census read row 9 as x11 against a true x12, and
    the miss was sitting in this guard's own report the whole time, labelled "not one of the
    ten" -- which is the one thing it was not.

    🎯 THE INDEX'S PUBLISHED PROTOCOL IS ALREADY POSITIONAL: "take the GENUS of your ANGLE
    string -- everything before its first colon". A genus is the LEADING phrase. When the
    string carries no colon for `CUT` to cut on, earliest-match is that same rule applied to
    the wording the run was actually handed in, not a relaxation of it. And it is not "prefer
    the lowest row number": C5 hands it the mirrored string and requires the mirrored answer.

    Measured over all 139 run headers before shipping (`w119b_diff.py`): exactly ONE assignment
    changes, L4077 unresolved -> row 9; no run changes its resolution path; no other row moves.
    """
    hits = sorted((m.start(), r) for r, (_, pat) in GENERA.items()
                  for m in [re.search(pat, genus, re.I)] if m)
    return hits[0][1] if hits else None


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

    print("\nC5 the ambiguity rule is positional, and it fires both ways")
    amb = [r for r in runs if r["row"] and
           sum(1 for _, pat in GENERA.values() if re.search(pat, r["genus"], re.I)) > 1]
    # The mirrored pair. Probe A is w14d's own genus verbatim; probe B names the same two
    # genera in the opposite order. A rule that merely preferred the lower row number would
    # answer 6 to both; a positional one answers 9 and then 6.
    probe_a = "error analysis on the best blend's OOF, the generator's coin-flip band"
    probe_b = "blending: rank-average the members, then error analysis on what is left"
    ga, gb = classify(probe_a), classify(probe_b)
    if len(amb) < MIN_AMBIGUOUS:
        fail(f"C5 INERT: only {len(amb)} resolved run(s) name two of the ten genera "
             f"(< {MIN_AMBIGUOUS}) -- nothing in the corpus exercises the rule")
    elif (ga, gb) != (9, 6):
        fail(f"C5: the rule is not positional -- probe A -> {ga} (want 9), "
             f"probe B -> {gb} (want 6)")
    else:
        print(f"  mirrored probes resolve to {ga} then {gb}, so the rule is positional and not "
              f"lowest-row. {len(amb)} corpus run(s) name two genera:")
        for r in amb:
            print(f"     L{r['line']:6d}  row {r['row']:2d}  {r['genus'][:56]}")

    print("\nC6 the label-to-quote window, both directions")
    # THE FROZEN DEFECT. These are the three declaring strings verbatim, as they stand in
    # JOURNAL.md, joined the way `resolve` joins them. Frozen as literals on w115's rule: a
    # control anchored to the live corpus stops being a control the moment the corpus moves,
    # and JOURNAL.md is append-only so these three can never change again.
    #
    # THE POSITIVE DIRECTION is that the shipped window reads all three as row 8. THE NEGATIVE
    # DIRECTION is that the pre-fix window reads NONE of them -- without it this passes just as
    # happily with the window reverted to 30, because `resolve` would still be finding these
    # three somewhere else. The pairing is the control; either half alone is decoration.
    for label, decl in REFUSAL_DECLS:
        got = next((classify(g) for g in genera_of(decl) if classify(g)), None)
        old_q, globals()["ANGLE_Q"] = ANGLE_Q, ANGLE_Q_NARROW
        try:
            was = next((classify(g) for g in genera_of(decl) if classify(g)), None)
        finally:
            globals()["ANGLE_Q"] = old_q
        if got != 8:
            fail(f"C6 {label}: the refusal-narrated declaration resolves to {got}, want row 8")
        elif was is not None:
            fail(f"C6 {label} INERT: the pre-fix 30-char window already resolved this to "
                 f"row {was}, so the widening is not what recovers it")
        else:
            print(f"  {label}: narrow window -> unresolved, shipped window -> row 8. OK")

    # AND THE WIDENING MUST STILL BE LOAD-BEARING IN THE LIVE CENSUS, not only on the literals.
    wide = [r for r in runs if r["row"] and ANGLE_Q.search(r["raw"])
            and not ANGLE_Q_NARROW.search(r["raw"])]
    if len(wide) < MIN_WIDE_GAP:
        fail(f"C6 INERT: only {len(wide)} resolved run(s) in the corpus need a gap > 30 "
             f"(< {MIN_WIDE_GAP}) -- the window is doing nothing and should be re-measured")
    else:
        print(f"  {len(wide)} resolved corpus run(s) need a gap > 30: "
              f"{[r['line'] for r in wide]}")

    json.dump({"measured": measured, "claimed": claims, "ambiguous": len(amb),
               "bad_rows": [list(b) for b in bad], "runs": len(runs),
               "resolved": len(resolved), "off_rotation": len(off)},
              open(OUT, "w"), indent=1, sort_keys=True)
    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
