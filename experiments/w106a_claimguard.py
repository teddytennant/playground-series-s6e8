r"""w106a — STANDING GUARD: A POSSESSION CLAIM IN RESEARCH.md MUST NAME MEMBERS THAT EXIST.

WHAT CLAIM THIS COVERS. w105 found that the CatBoost closure supported itself with *"the
CatBoost function class is already in the pack four ways (`cat_lat`, `cat_native`,
`cat_native_ctr2`, `cat_natlat`)"* and that THE LAST TWO WERE NEVER BUILT -- no such file
exists anywhere on disk. The closure survived on other evidence, so nothing was lost that
day. What was lost is the assumption that a closure's citations are real.

🎯 THE SHAPE, and it is why `w101a_angleguard` cannot cover this. w101a proves the ANGLE
INDEX's anchors RESOLVE -- that the grep string still finds its section. It does, greenly, at
a paragraph whose contents are false. **A WORKING POINTER TO A WRONG CLAIM IS THE INDEX'S
BLIND SPOT.** The pointer and the claim are different objects and need different checks:
w101a checks the sentence is reachable, w106a checks the artefact is there. w105 said it as
"check the artefact, not the sentence"; this is that sentence made executable.

WHY IT MATTERS BEYOND TIDINESS. Every modelling angle handed to this workspace is refused by
citing a measurement. A run that greps the ANGLE INDEX, lands on a closure and reads a list of
members that "we already hold" has no cheap way to know the list is fiction -- and the whole
point of the index is that the refusal costs ONE grep, not half a run. A closure standing on
files that were never built is a refusal the next run cannot audit.

HOW IT DECIDES, and the one design choice that makes it trustworthy:

  INVENTORY IS A DELIBERATE SUPERSET. Any `oof_<name>.npy` ANYWHERE under the workspace
  counts -- the live pack, every `data/ext_members*`, every pulled notebook, `oof_rejected/`.
  This guard therefore CANNOT fire on "held, but not in this pack", only on "exists nowhere
  at all". A narrower inventory would be a more sensitive instrument and a less believable
  one, and an unbelievable guard gets switched off.

  THE SIBLING RULE. A backticked, member-shaped token is only reported when at least one of
  its SIBLINGS in the same claim IS in the inventory. Rationale: if a list names four things
  and two of them are our members, the list is speaking our vocabulary and an absent entry is
  a defect. If NONE of them are ours, the sentence is about somebody else's artefacts and the
  absence means nothing. Measured on the live document (w106): the sibling rule turns 12
  flagged tokens into 1 true finding and 0 false positives. Every one of the 11 it suppresses
  is a real non-member -- Kaggle usernames (`amanatar`, `lavanyabacche`), an md5, a transform
  bit (`drop_worst`), a run label (`w38c`), a correction object (`c_avg`), and the FOREIGN
  names of rejected external streams (`omidbag_oof`, `mohankr_realmlp`) whose sentence says
  they duplicate members we hold, i.e. the claim's subject is the other side of the sentence.

  ⚠ THE PRICE OF THAT RULE, STATED SO NOBODY REDISCOVERS IT AS A SURPRISE: a claim in which
  EVERY named member is absent is invisible to this guard. That is a real hole and it is
  chosen. A guard that cries wolf on usernames would be muted within two runs, and a muted
  guard is worth less than a narrow one. If a wholly-fabricated list ever turns up, widen it
  then, with that instance as the control.

WHY RESEARCH.md ONLY, AND NOT JOURNAL.md. JOURNAL is append-only history: a claim there is a
record of what a run believed on a given day, and correcting it would be falsifying the log. It
is also never the document a later run greps to make a decision. RESEARCH is -- it is the live
navigational aid, the thing the ANGLE INDEX points into, and the only one where a false citation
can still cost a run half a day. Scope the guard to the document whose claims are load-bearing.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C0  +   INVENTORY FLOOR. >= 100 member names must be found. This is the arm that stops the
          guard from being one that can only PASS: if the walk broke and the inventory came
          back empty, the sibling rule would suppress EVERY finding and the guard would go
          green on a document full of fiction. A silent zero is the failure mode of an
          instrument that reports absence.
  C1  -   THE LIVE DOCUMENT IS CLEAN. RESEARCH.md as it stands must produce no finding. w105
          corrected the w61 bullet in place, so a finding here means a NEW false citation has
          entered the document since.
  C2  +   THE HISTORICAL TRUE POSITIVE, IN CODE. A doctored copy that restores w61's original
          "four ways" sentence must fire, and must name EXACTLY `cat_native_ctr2` and
          `cat_natlat`. This is the guard reproducing the finding that caused it, on the real
          text, which is the only evidence that it would have caught it in time.
  C3  +-  THE SIBLING RULE IS DOING THE WORK, NOT THE EXEMPTION. Scanned with exemptions
          DISABLED, the live document must yield exactly the two tokens quoted inside w105's
          own correction section -- no more. More than those two means the sibling rule has
          started admitting noise and the guard is on its way to being ignored.
  C4  +   NO STALE EXEMPTION. Every section named in EXEMPT_SECTIONS must still be findable in
          RESEARCH.md. An exemption whose section has been deleted or retitled is a hole that
          nothing reports, so it is fatal here rather than silently widening coverage. This is
          w105's C4 span-scoping rule applied to a second document region.
  C5  +   NOT VACUOUS. The scan must find several possession claims and several tokens that
          resolve, or the regex has drifted off the document's vocabulary and the green is
          meaningless.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
OUT = os.path.join(ROOT, "experiments", "w106a_claimguard.json")

# Sections that legitimately QUOTE a false possession claim in order to correct it. Scoped to
# the section's own span so the exemption dies with the section (C4 makes a stale one fatal).
EXEMPT_SECTIONS = [
    ("## ⚰️ AND A CITATION IN THE CATBOOST CLOSURE THAT POINTS AT NOTHING",
     "w105's correction: quotes the never-built pair verbatim as the finding's evidence"),
    ("# 🔻 A CLOSURE'S CITATIONS ARE NOW MACHINE-CHECKED — `w106a_claimguard`, standing check #42",
     "this guard's own write-up: quotes the same claim to state what it catches"),
]

# ⚠ THE WIDENING HAZARD, RECORDED BECAUSE THE LIST GREW ON DAY ONE. Both entries above exist for
# the same reason -- a section that QUOTES the false claim in order to correct it -- and every
# future write-up of this defect would want a third. That is how a guard's coverage erodes: not
# by anyone switching it off, but by each individual exemption being obviously reasonable. Two is
# fine. If this list reaches FOUR, stop adding to it and fix the root cause instead: quote broken
# citations WITHOUT a possession verb on the same line ("the bullet named cat_natlat", not "we
# already hold cat_natlat"), which costs nothing and needs no exemption at all.

CLAIM = re.compile(
    r"already (?:in the pack|hold|holds|have|has|ships|shipped)"
    r"|pack already (?:holds|has|contains)"
    r"|we (?:already )?(?:hold|have)"
    r"|is in the pack|are in the pack",
    re.I,
)
TOK = re.compile(r"`([^`]+)`")
# A member name as this workspace writes them: lowercase, >=3 chars, optional trailing glob.
SHAPE = re.compile(r"^[a-z][a-z0-9_]{2,}\*?$")
NEVER_BUILT = {"cat_native_ctr2", "cat_natlat"}  # w26 E1/E2: pre-registered, never built
WINDOW = 3  # a claim's names may spill onto the next couple of lines
# ⚠ BUT NOT ACROSS A TABLE ROW (w127). A markdown table row is a self-contained cell: the
# ANGLE INDEX puts rows 5, 6 and 7 on three consecutive lines, and a 3-line window opened on
# row 5's "pack already holds" swallowed rows 6 and 7 whole. That fires BOTH ways and the
# quiet direction is the dangerous one -- a broken citation on row 5 can be RESCUED by a
# valid member name three rows down, and the guard goes green on a claim it never checked.
# w127 hit the loud direction: adding `xgb_latcat` to row 7 supplied the "present" sibling
# that row 5's window needed, and five non-member tokens from rows 5/6 became findings.
# Same genus as w121's identifier-boundary rule and w119's positional `classify`: a reader
# whose window crosses a boundary it does not know about.
TABLE_ROW = re.compile(r"^\s*\|")

FAILS = 0
NOTES: dict = {}


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"FAIL {msg}")


def inventory(root: str) -> set:
    """Every member name with an oof_*.npy anywhere under root. Deliberately a superset."""
    names = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__")]
        for fn in filenames:
            if fn.startswith("oof_") and fn.endswith(".npy"):
                names.add(fn[4:-4])
    return names


def exempt_spans(lines: list) -> list:
    """(lo, hi, header, found) for each EXEMPT_SECTIONS entry; hi is the next '#' header."""
    spans = []
    for header, why in EXEMPT_SECTIONS:
        lo = next((i for i, l in enumerate(lines) if l.strip() == header.strip()), None)
        if lo is None:
            spans.append(dict(header=header, why=why, found=False, lo=None, hi=None))
            continue
        # Skip the CONTIGUOUS header block first: this document routinely wraps a title over
        # two or three "#" lines, and closing the span on the second of them would exempt the
        # title and nothing else -- a stale exemption that still reports found=True.
        body = lo + 1
        while body < len(lines) and lines[body].startswith("#"):
            body += 1
        hi = next((j for j in range(body, len(lines)) if lines[j].startswith("#")), len(lines))
        spans.append(dict(header=header, why=why, found=True, lo=lo, hi=hi))
    return spans


def scan(text: str, inv: set, use_exempt: bool = True) -> dict:
    """Findings, plus the bookkeeping C5 needs to prove the scan was not asleep."""
    lines = text.split("\n")
    spans = exempt_spans(lines)
    skip = set()
    if use_exempt:
        for s in spans:
            if s["found"]:
                skip.update(range(s["lo"], s["hi"]))

    findings, n_claims, n_present = [], 0, 0
    for i, line in enumerate(lines):
        if not CLAIM.search(line) or i in skip:
            continue
        n_claims += 1
        # The window stops at the next table row, and is a single line when the claim is
        # itself in one. See TABLE_ROW above for why; C2/C3 check the change did not
        # also stop the guard finding the defect it exists for.
        if TABLE_ROW.match(line):
            win = [line]
        else:
            win = []
            for l in lines[i:i + WINDOW]:
                if win and TABLE_ROW.match(l):
                    break
                win.append(l)
        group = [t for t in TOK.findall(" ".join(win)) if SHAPE.match(t)]
        present = [t for t in group
                   if (fnmatch.filter(inv, t) if t.endswith("*") else t in inv)]
        absent = [t for t in group if t not in present]
        n_present += len(present)
        if present and absent:  # THE SIBLING RULE
            findings.append(dict(line=i + 1, absent=absent, present=present,
                                 text=line.strip()[:160]))
    return dict(findings=findings, n_claims=n_claims, n_present=n_present, spans=spans)


def main() -> int:
    inv = inventory(ROOT)
    text = open(RESEARCH, encoding="utf-8").read()

    # ---- C0: inventory floor -- the arm that stops this being a guard that can only pass ----
    NOTES["c0"] = {"inventory": len(inv)}
    if len(inv) < 100:
        fail(f"C0 inventory has only {len(inv)} member names; the walk is broken and every "
             f"finding would be suppressed by the sibling rule -- this guard would go green "
             f"on a document full of fiction")

    # ---- C1: the live document is clean ------------------------------------------------------
    live = scan(text, inv, use_exempt=True)
    NOTES["c1"] = {"findings": live["findings"]}
    for f in live["findings"]:
        fail(f"C1 RESEARCH.md:{f['line']} claims to hold {f['absent']} -- no oof_*.npy for "
             f"those anywhere on disk (siblings that DO exist: {f['present']}). "
             f"Line: {f['text']}")

    # ---- C2: the historical true positive, in code -------------------------------------------
    original = ("* the CatBoost function class is already in the pack four ways (`cat_lat`, "
                "`cat_native`, `cat_native_ctr2`, `cat_natlat`) -- and `run_catboost.py "
                "--inner` is the honest tuning path, built and run;")
    doctored = scan(text + "\n\n## w106a C2 DOCTORED CONTROL\n" + original + "\n", inv,
                    use_exempt=True)
    got = {t for f in doctored["findings"] for t in f["absent"]}
    NOTES["c2"] = {"absent_reported": sorted(got), "n_findings": len(doctored["findings"])}
    if got != NEVER_BUILT:
        fail(f"C2 restoring w61's original sentence did not reproduce the w105 finding: "
             f"expected {sorted(NEVER_BUILT)}, got {sorted(got)}")

    # ---- C3: the sibling rule, not the exemption, is doing the work ---------------------------
    unexempt = scan(text, inv, use_exempt=False)
    got3 = {t for f in unexempt["findings"] for t in f["absent"]}
    NOTES["c3"] = {"absent_reported": sorted(got3), "n_findings": len(unexempt["findings"])}
    if got3 != NEVER_BUILT:
        fail(f"C3 with exemptions off the document should yield exactly the pair quoted inside "
             f"w105's correction; got {sorted(got3)}. More than that means the sibling rule has started "
             f"admitting noise")

    # ---- C4: no stale exemption --------------------------------------------------------------
    NOTES["c4"] = {"spans": [{k: s[k] for k in ("header", "found", "lo", "hi")}
                             for s in live["spans"]]}
    for s in live["spans"]:
        if not s["found"]:
            fail(f"C4 exempt section not found in RESEARCH.md, so the exemption is stale and "
                 f"silently covers nothing: {s['header']!r} ({s['why']})")

    # ---- C5: not vacuous ---------------------------------------------------------------------
    NOTES["c5"] = {"claims": live["n_claims"], "resolved_tokens": live["n_present"]}
    if live["n_claims"] < 5:
        fail(f"C5 only {live['n_claims']} possession claims matched; the regex has drifted off "
             f"the document's vocabulary and the green means nothing")
    if live["n_present"] < 8:
        fail(f"C5 only {live['n_present']} cited tokens resolved to real members; the shape "
             f"filter or the inventory has drifted")

    NOTES["utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    NOTES["failures"] = FAILS
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(NOTES, fh, indent=2, sort_keys=True)
    print(f"FAILURES {FAILS}   -> {os.path.relpath(OUT, ROOT)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
