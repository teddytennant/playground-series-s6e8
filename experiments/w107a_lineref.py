#!/usr/bin/env python3
"""Standing check #43 — line-number citations in RESEARCH.md must not point into RESEARCH.md.

WHY (w107, 2026-08-28, measured — not asserted):
  RESEARCH.md is PREPENDED every run. The last four commits touching it each open with a
  hunk at `@@ -1,3 +1,N @@`, N = 69..133. So every internal line number decays by ~100
  lines per run, deterministically, and nothing ever notices: the pointer still resolves,
  it just lands on an unrelated paragraph that reads like prose because it IS prose.

  Measured on the live document, every citation class:
      into RESEARCH.md (prepended)   7 refs   0 live   7 stale
      into JOURNAL.md  (appended)    1 ref    1 live   0 stale
      into source files              3 refs   2 live   1 drifted (file was edited)
  Four of the seven were stale by ~4,600 lines in the same direction — which is exactly
  the accumulated prepend since the w26-era runs that wrote them.

  The ANGLE INDEX already says "anchors are section-header text, not line numbers,
  RESEARCH.md is edited at the top every run". The closure body of the very row that
  sentence sits above cited three line numbers. Distance in a grep-navigated file is the
  same as absence (w105), so the rule is executable here instead of written down again.

THE RULE
  - a citation into RESEARCH.md itself  -> FAIL. Not "check it", FAIL: it is structurally
    unmaintainable. Replace it with a grep-able text anchor.
  - a citation naming any other file    -> RESOLVE (file exists, line in range).

⚠ THE PRICE, STATED UP FRONT (w106 house style):
  1. For external refs this checks existence and range ONLY, never semantics. `w25a_cvlb_full.py
     (import line 80, call line 34)` passes here and line 34 is not the call it claims. A
     range check cannot see that; do not read a green as "the citation is true".
  2. JOURNAL.md is NOT scanned. It is append-only by the playbook's hard rules, so a guard
     failing on its history could never be made green, and a guard that cannot go green gets
     switched off. Chosen, not overlooked.
  3. Fenced code blocks are skipped, so a citation inside ``` is invisible. That exclusion is
     STRUCTURAL, not per-instance: it is what lets a run quote a broken pointer in its own
     write-up without adding an exemption. w106 §5 found exemption lists erode a guard by
     each addition being individually reasonable; this one cannot grow.
"""
from __future__ import annotations

import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
DOC = os.environ.get("W107A_DOC", "RESEARCH.md")   # overridable so the arms can doctor a copy

# "line 2026-08-11" / "line 08-27" are dates, not citations.
# (?!\d) pins the whole number: without it the regex backtracks and reads the "202" out
# of "line 2026-08-11", turning a date into a citation.
CITE = re.compile(r"\blines?\s*~?\s*(\d{2,6})(?!\d)(?!\s*[-–]\s*\d{1,2}\b)")
FILEREF = re.compile(r"\b[\w./-]*\w+\.(?:py|sh|md|csv|txt|json|npy|npz|yml|yaml)\b")
BAREDOC = re.compile(r"\bJOURNAL\b(?!\.)")


def strip_fences(lines):
    """Blank out fenced code blocks, preserving line numbering."""
    out, infence = [], False
    for ln in lines:
        if ln.lstrip().startswith("```"):
            infence = not infence
            out.append("")
            continue
        out.append("" if infence else ln)
    return out


def scan(doc_path):
    raw = open(doc_path, encoding="utf-8").read().splitlines()
    body = strip_fences(raw)
    nlines = len(raw)
    cites = []
    for i, ln in enumerate(body):
        for m in CITE.finditer(ln):
            num = int(m.group(1))
            # context = this line plus the one before (citations wrap across lines)
            ctx = (body[i - 1] if i else "") + " " + ln
            named = FILEREF.findall(ctx)
            # "JOURNAL line 19481" names a file without writing the extension.
            if BAREDOC.search(ctx):
                named.append("JOURNAL.md")
            non_self = [f for f in named if os.path.basename(f).lower() != "research.md"]
            # INTERNAL = points into RESEARCH.md: names it explicitly, or names no file at all.
            internal = (not non_self) or bool(re.search(r"RESEARCH(?:\.md)?\s*(?:line|\()", ctx))
            cites.append(
                dict(line=i + 1, num=num, text=ln.strip()[:110], internal=internal, files=non_self)
            )
    return cites, nlines


def _candidates(f):
    """Where a cited path might live: as written, under experiments/, or anywhere in the repo."""
    yield os.path.join(ROOT, f)
    yield os.path.join(ROOT, "experiments", f)
    base = os.path.basename(f)
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in (".git", ".venv", "__pycache__")]
        if base in filenames:
            yield os.path.join(dirpath, base)


def resolve(c):
    """For an external citation: does the named file exist and contain that line?"""
    for f in c["files"]:
        for cand in _candidates(f):
            if os.path.isfile(cand):
                n = sum(1 for _ in open(cand, encoding="utf-8", errors="replace"))
                if c["num"] <= n:
                    return True, f"{f} ({n} lines)"
                return False, f"{f} has only {n} lines, cited {c['num']}"
    return False, "named file(s) not found on disk: " + ", ".join(c["files"])


def main():
    doc = os.path.join(ROOT, DOC)
    cites, nlines = scan(doc)
    fails, notes = [], []

    internal = [c for c in cites if c["internal"]]
    external = [c for c in cites if not c["internal"]]

    for c in internal:
        fails.append(
            f"{DOC}:{c['line']} cites line {c['num']} of the document itself — RESEARCH.md is "
            f"prepended ~100 lines/run, so this pointer is already wrong or soon will be. "
            f"Use a grep-able text anchor.  >> {c['text']}"
        )

    for c in external:
        ok, why = resolve(c)
        if ok:
            notes.append(f"  ok   {DOC}:{c['line']} -> {why}")
        else:
            fails.append(f"{DOC}:{c['line']} cites line {c['num']} but {why}.  >> {c['text']}")

    # C0 NON-VACUITY. An instrument that reports absence must prove it can still see (w106 §3).
    if len(cites) == 0:
        fails.append("C0: scanner found NO line citations at all in a 14k-line document — "
                     "the regex or the fence-stripper is broken, not the document clean.")
    if len(external) == 0:
        fails.append("C0: scanner resolved NO external citations — it has never exercised the "
                     "resolve path, so a green says nothing about it.")

    print(f"#43 lineref — {DOC}: {nlines} lines, {len(cites)} citation(s) "
          f"({len(internal)} internal, {len(external)} external)")
    for n in notes:
        print(n)
    for f in fails:
        print("FAIL " + f)
    print(f"FAILURES {len(fails)}   -> experiments/w107a_lineref.json")

    json.dump(
        dict(doc_lines=nlines, n_cites=len(cites), n_internal=len(internal),
             n_external=len(external), failures=fails, ok=notes),
        open(os.path.join(ROOT, "experiments", "w107a_lineref.json"), "w"), indent=1)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
