"""w115a — STANDING GUARD #49: the HUMAN-READ DOCS must not name a non-WANTED file as the pick.

WHY THIS EXISTS, AND WHY #48 DOES NOT ALREADY COVER IT.
w114 §2 found the largest reachable error in this workspace: `check_selection.py` printed prose
naming four superseded files as "current WANTED". The constant was right the whole time; the
PROSE next to it had rotted, and no guard looked at printed prose. `w114b_selectguard` (#48)
closed that -- but it walks exactly one file, `check_selection.py`, and only the string literals
`main()` prints.

A human does not only run the script. The playbook sends them to `RESEARCH.md` ("distilled
durable facts", read as CURRENT) and to `SELECT_THESE.md`. On 2026-08-29 `RESEARCH.md` still
carried five assertion-shaped lines naming `blend159av_h3` / `w21_ad187corr` as "the CV pick" --
the two files w114a priced at +81.92e-6 and +35.17e-6 against a +4.52e-6 no-click cost. Each was
true when written and sat in a dated section; none said so on the line itself. That is w114 §2's
lesson exactly ("a ✅ and a date are not a freshness claim"), one document over.

⚠ SEVERITY, STATED HONESTLY. This is the same CLASS as w114 §2 but not the same MAGNITUDE. The
reader's real entry points -- `SELECT_THESE.md` and `check_selection.py` -- were both already
clean and loud, and RESEARCH.md's top section carries the correct warning. The stale lines sat
7,000+ lines down. Reachable, not default. The guard is cheap; the finding was not a second w114.

WHAT IS SCANNED, AND WHAT IS DELIBERATELY NOT.
  scanned      RESEARCH.md, SELECT_THESE.md -- both are read as statements about NOW.
  NOT scanned  JOURNAL.md. It is append-only dated history; an 08-17 entry saying "the pick is
               now w16i_schemeavg" was TRUE on 08-17 and rewriting it would destroy the record.
               This is the same exemption w114 gave `CLICK_HISTORY` and the module docstring.

A line is a FINDING when it is assertion-shaped (ASSERT_RE: "the CV pick", "the pick is",
"current WANTED", ...) AND names a stem that is SELECTABLE (`submissions/<stem>.csv` exists) and
is not in `check_selection.WANTED`. A line is exonerated by either:
  * carrying a STALE marker (STALE_RE) -- the fix w115 applied, which keeps the history readable
    while removing the false currency; or
  * an ALLOWED entry, which pairs a stem with a distinctive substring AND a written reason.
    Keyed on substring, never on line number, so it survives the prepends this file sees weekly.

CONTROLS (w72 §5.3: a control that can only fail is not a control).
  C1 +   NOT VACUOUS. The stem set and the candidate-line set must both clear a floor, so a
         guard that silently scanned nothing cannot report green.
  C2 +-  THE REAL HISTORICAL ARTEFACT, not a plant (w114 §3's strongest control). Scans
         `git show HEAD:RESEARCH.md` -- the file before w115 annotated it -- and requires it to
         be RED. If the pre-fix file also passes, this guard is inert and says so.
  C3 +   COUNTS THE DELTA, NOT THE TOTAL (w114 §3's mistake, not repeated). The set difference
         HEAD-minus-current must be exactly the stems w115 annotated; the guard fails if the
         improvement it claims is not the improvement it can measure.
  C4 +-  FIRES BOTH WAYS on a synthetic pair: a bare stale assertion must trip, and the same
         line carrying a STALE marker must not.
  C5 +   BOUNDARY. A stem embedded in a longer identifier is not that stem (w114 §3: `blend158`
         inside `blend158_logit`).

    .venv/bin/python experiments/w115a_docselectguard.py            # 0 = ok, 1 = a guard failed
    .venv/bin/python experiments/w115a_docselectguard.py --list     # show findings, no exit code
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUBS = os.path.join(ROOT, "submissions")
DOCS = ["RESEARCH.md", "SELECT_THESE.md"]
OUT = os.path.join(HERE, "w115a_docselectguard.json")

# Language that asserts a file IS the current pick / IS wanted.
ASSERT_RE = re.compile(
    r"(current WANTED|WANTED is|the CV pick|the pick is|CV-preferred|is the argmax)", re.I)
# The annotation w115 applied. Its presence means the line is explicitly dated as past.
STALE_RE = re.compile(r"(STALE NAME\s*—|\[STALE\s)", re.I)

MIN_STEMS = 50          # C1 floor: the submissions dir really was read
MIN_CANDIDATES = 8      # C1 floor: assertion-shaped lines really were found

# ── C2's fixture: the REAL defect, frozen ────────────────────────────────────────────────────
# These are the five lines verbatim from RESEARCH.md at commit 0f8c398 -- the state that stood
# for twelve days, naming the two mis-click pairs as "the pick". NOT a synthetic plant.
#
# ⚠ WHY THIS IS A FROZEN LITERAL AND NOT `git show HEAD:RESEARCH.md`, WHICH IS WHAT IT WAS.
# The guard passed while the fix was uncommitted and went RED the instant the fix landed, because
# HEAD had become the FIXED file and the control had nothing left to catch. It failed CLOSED and
# said "INERT", which is the right direction -- but the lesson is general:
# 🎯 A CONTROL ANCHORED TO A MOVING REFERENCE STOPS BEING A CONTROL THE MOMENT YOU COMMIT THE FIX.
# A pinned SHA would also rot if history is ever rewritten. The artefact itself is the only
# reference that cannot drift, so it is checked in here.
HISTORICAL_DEFECT = [
    "all along. The pick is `blend159av_h3`.",
    "> any remaining modelling. The spread between the CV pick (`blend159av_h3`, 0.970049) and",
    "Note **none of the four is the CV pick**: `blend159av_h3` (0.970049) and `blend158_h3`",
    "whose base is the CV pick `blend159av_h3` and whose CV (0.9700528) is the best held here, so the",
    "auto-slot 1: public 0.97117, 1-way — w21_ad187corr   <- IDENTICAL to the CV pick",
]
MIN_HISTORICAL = 5      # every line above must still be caught

# (stem, distinctive substring on the line, reason). Substring-keyed so line shifts don't break it.
ALLOWED = [
    ("w21_ad187corr", "WANTED is now {w21_ad187corr.csv",
     "w114's own section quoting the removed defect verbatim. Naming it IS the correction."),
    ("w20_ad187_h3", "WANTED is now {w21_ad187corr.csv",
     "same line as above; quoted as part of the defect being described."),
    ("w16i_schemeavg", "annotated `w16i_schemeavg`",
     "w114's section describing the rot it removed, not asserting it."),
    ("blend159av_h3", "annotated `w16i_schemeavg`",
     "same line; named as the defect, not as the pick."),
    ("w36_ad199std_h3", "CV pick's base is",
     "named as the BASE artefact of the pick, not the pick. The pick itself is WANTED."),
    ("w36_ad199std_h3", "| `w36_ad199stdcorr` (the CV pick) |",
     "reproduction table; the WANTED pick is named correctly on the same line."),
    ("blend158_logit", "private gap against the pick",
     "named as the file measured AGAINST the pick, i.e. explicitly not the pick."),
    ("w16i_schemeavg", "was true on 08-17",
     "w115's section quoting a JOURNAL line as the example of correct-as-history prose. "
     "The quote is the point; the same sentence dates it. (Caught by this guard on its "
     "first run against w115's own new text -- which is the guard working.)"),
]


def stems():
    if not os.path.isdir(SUBS):
        return set()
    return {f[:-4] for f in os.listdir(SUBS) if f.endswith(".csv")}


def wanted():
    sys.path.insert(0, HERE)
    import check_selection as cs
    return {w[:-4] if w.endswith(".csv") else w for w in cs.WANTED}


def names_in(line, pool):
    """Stems named in `line`, with identifier boundaries (C5)."""
    return {s for s in pool
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", line)}


def allowed_for(line, stem):
    return any(s == stem and sub in line for s, sub, _ in ALLOWED)


def scan_text(text, pool, want, doc="<text>"):
    """-> (findings, n_candidate_lines). A finding is (doc, lineno, stem, line)."""
    found, cand = [], 0
    for i, line in enumerate(text.split("\n"), 1):
        if not ASSERT_RE.search(line):
            continue
        cand += 1
        if STALE_RE.search(line):
            continue
        for stem in sorted(names_in(line, pool) - want):
            if allowed_for(line, stem):
                continue
            found.append((doc, i, stem, line.strip()[:140]))
    return found, cand


def scan_docs(pool, want):
    findings, cand = [], 0
    for doc in DOCS:
        p = os.path.join(ROOT, doc)
        if not os.path.exists(p):
            continue
        f, c = scan_text(open(p, encoding="utf-8", errors="replace").read(), pool, want, doc)
        findings += f
        cand += c
    return findings, cand


def git_head(path):
    r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    return r.stdout if r.returncode == 0 else None


def main():
    fails = []
    pool, want = stems(), wanted()
    print(f"pool {len(pool)} selectable stem(s); WANTED {sorted(want)}")

    findings, cand = scan_docs(pool, want)

    # ---- C1: not vacuous ------------------------------------------------------------------
    print(f"C1  {len(pool)} stems (floor {MIN_STEMS}), {cand} assertion-shaped "
          f"line(s) (floor {MIN_CANDIDATES})")
    if len(pool) < MIN_STEMS or cand < MIN_CANDIDATES:
        fails.append(f"C1: vacuous scan — {len(pool)} stems, {cand} candidate lines")

    # ---- the finding itself ---------------------------------------------------------------
    print(f"MAIN {len(findings)} unexonerated finding(s) across {DOCS}")
    for doc, i, stem, line in findings:
        print(f"  {doc}:{i}  {stem}\n      {line}")
        fails.append(f"{doc}:{i} names non-WANTED selectable `{stem}` as the pick")

    # ---- C2: the real historical artefact, frozen so it cannot drift -----------------------
    h_find, _ = scan_text("\n".join(HISTORICAL_DEFECT), pool, want, "HISTORICAL_DEFECT")
    h_stems = {f[2] for f in h_find}
    print(f"C2  frozen historical defect ({len(HISTORICAL_DEFECT)} real lines from 0f8c398) -> "
          f"{len(h_find)} finding(s) over {len(h_stems)} stem(s): {sorted(h_stems)}")
    if len(h_find) < MIN_HISTORICAL:
        fails.append(f"C2: INERT — the guard catches only {len(h_find)} of "
                     f"{MIN_HISTORICAL} lines of the REAL historical defect, so a green "
                     f"result proves nothing about the failure it exists to prevent")

    # ---- C3: the live docs must be clean of exactly that defect ----------------------------
    # Counts the DELTA (w114 §3): the defect stems must be caught in the fixture AND absent
    # from the live docs. A guard that cannot demonstrate both is not measuring an improvement.
    live_bad = {f[2] for f in findings}
    still = h_stems & live_bad
    print(f"C3  defect stems still live in the docs: {sorted(still) or 'none'}")
    if still:
        fails.append(f"C3: the historical defect is still present live for {sorted(still)}")

    # ---- C4: fires both ways --------------------------------------------------------------
    probe_stem = sorted(pool - want)[0] if (pool - want) else None
    if probe_stem is None:
        fails.append("C4: no non-WANTED stem available to probe with")
    else:
        dirty = f"The pick is `{probe_stem}` and nothing else."
        clean = dirty + " ⚠ STALE NAME — the pick moved."
        d_f, _ = scan_text(dirty, pool, want)
        c_f, _ = scan_text(clean, pool, want)
        print(f"C4  planted assertion -> {len(d_f)} finding(s); same line marked STALE -> "
              f"{len(c_f)} finding(s)")
        if len(d_f) != 1:
            fails.append(f"C4: a bare stale assertion must be caught, got {len(d_f)}")
        if len(c_f) != 0:
            fails.append(f"C4: a STALE-marked line must be exonerated, got {len(c_f)}")

    # ---- C5: identifier boundary ----------------------------------------------------------
    long_pool = {"blend158"} & pool
    if long_pool:
        n = names_in("the pick is `blend158_logit` today", {"blend158"})
        print(f"C5  `blend158` matched inside `blend158_logit`: {bool(n)} (must be False)")
        if n:
            fails.append("C5: stem matched without an identifier boundary")
    else:
        print("C5  SKIP — `blend158` not on disk to probe the boundary with")

    json.dump({"findings": [list(f) for f in findings], "candidates": cand,
               "stems": len(pool), "failures": len(fails)},
              open(OUT, "w"), indent=1)

    print()
    if fails:
        print(f"FAIL {len(fails)}")
        for f in fails:
            print("  -", f)
        return 1
    print("✅ CLEAN — no human-read doc names a non-WANTED selectable file as the pick")
    return 0


if __name__ == "__main__":
    if "--list" in sys.argv:
        p, w = stems(), wanted()
        fs, c = scan_docs(p, w)
        for doc, i, stem, line in fs:
            print(f"{doc}:{i}  {stem}  {line}")
        print(f"({len(fs)} finding(s), {c} candidate line(s))")
        sys.exit(0)
    sys.exit(main())
