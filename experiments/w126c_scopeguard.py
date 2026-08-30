"""w126c -- STANDING CHECK #55. A SEARCH PRICE IS A NUMBER, A QUANTITY, A LAYER, AND A SCOPE.
#53 gave the ANGLE INDEX's price column its QUANTITIES. #54 gave row 5 its LAYER. Neither can
see a price whose SCOPE -- which search, over how many parameters -- belongs to a different
object than its magnitude. (w126, 2026-08-30)

THE DEFECT THIS EXISTS BECAUSE OF. Row 6 read, in full:

    a **SEARCH** price (re-weighting members already in) -- **-1.07e-6**

-1.07e-6 is w36d's cross-arm number over k=4 TRANSFORM arms (clause 3 of the handed string).
"re-weighting members already in" is clause 1, which w108 had already recorded as ALREADY THE
SHIPPED ARCHITECTURE -- 104+ MEMBER weights fitted inside the frozen folds by agent/stack.py.
w126a measured that second search: it beats equal weights at 13/13 rungs of a nested k ladder,
by +2,343e-6 at k=104. Opposite sign, ~2,000x the magnitude, under one label.

AND IT DECIDES A CLOSURE. The workspace's search-cost rule is `optimism ~ 0.55(k-1) e-6`.
Read the parenthetical literally at k=104 it gives +57e-6, ABOVE the 50e-6 floor; read the
number's real scope, k=4, it gives +1.65e-6. The mislabelled word picks the side.

WHY #53 AND #54 ARE BOTH BLIND TO IT. #53's C1 asks only that a magnitude name a quantity from
{TUNING, ENROLMENT, CONCAT, SEARCH}; row 6 says SEARCH, so it passed. #54's C1 is scoped to
row 5 by construction, because row 5 was the only cell whose evidence was member-layer. Seventh
run of this genus (w120 priority / w121 a prose cause / w122 slot for tier / w123 a family label
on a residual / w124 no unit / w125 no layer / w126 no scope), and each fix has been blind to
the next one in the same way.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration).
  C1 +   SCOPE, NOT VACUOUS. Every ANGLE INDEX price cell carrying an e-6/e-7 magnitude AND the
         word SEARCH must name a LAYER word and a `k=`. Vacuous if no such cell exists.
  C2 +   ROW 6 PUBLISHES BOTH SEARCHES -- both layers, both k, and the word INCUMBENT, so the
         positive member-layer number can never again be read as an open door.
  C3 +   THE RULE'S RANGE. Every site stating the search-cost rate as a PROPERTY must disclose
         the k it was fitted at. BOTH phrasings are covered: `0.45k e-6` (the /k form) and
         `+0.45e-6|+0.55e-6 of optimism per free parameter` (the per-parameter form). The
         second phrasing is in scope because that is where the rate went stale for 22 runs --
         w108 corrected the deriving document and not the one that repeats it. A match inside
         a `*"..."*` span is a QUOTATION of superseded text, not a definition, and is skipped;
         C5 checks that exemption in the other direction, since the pre-fix w63 line carries
         no quote markers and must still fire.
  C4 +-  THE ARTEFACT STILL SAYS IT. w36d's eight published cells and the cross-arm -1.0710e-6
         are compared against w126a_row6.json, which measured them from the OOF arrays. Frozen
         literals here, so this is a COMPARISON. If the json is absent the check reports itself
         INERT rather than green -- a missing artefact is not a pass.
  C5 +-  --control runs C1/C2/C3 over the FROZEN PRE-FIX text, inlined as literals rather than
         read from HEAD, because a control anchored to HEAD stops being a control the moment
         the fix is committed (w115). Pre-fix must FIRE on all three; shipped must be silent.

WHAT C1 IS BLIND TO, WRITTEN DOWN BECAUSE #53's UNWRITTEN EXEMPTION IS WHERE #54's DEFECT LIVED
AND #54's IS WHERE THIS ONE DID. C1 fires only on cells carrying BOTH a magnitude and the word
SEARCH. Today that is one cell, row 6. A cell whose quantity is ENROLMENT or TUNING can be
scoped just as wrongly -- enrolling into base104 and enrolling into a 12-member pool are not the
same price -- and C1 will not look at it. Rows 3, 4 and 5 all disclose `base104` in prose and
NOTHING CHECKS THAT THEY KEEP DOING SO.

    .venv/bin/python experiments/w126c_scopeguard.py            # rc 0 = clean
    .venv/bin/python experiments/w126c_scopeguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from w128b_pricedguard import claims  # noqa: E402  -- the negation-aware reader, #57
ROOT = os.path.dirname(HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
ROW6_JSON = os.path.join(HERE, "w126a_row6.json")
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"

# C1. Same magnitude predicate as #53, so the two guards agree on what "priced" means.
MAGNITUDE = re.compile(r"\d\s*e-[67]|\d,\d{3}e-[67]")
SEARCH_WORD = "SEARCH"
LAYER_WORDS = ("TOP-LEVEL", "TRANSFORM", "MEMBER", "STACK")
K_TOKEN = re.compile(r"\bk=\d+")

# C2. Row 6 carries two searches; both must be legible from the cell alone.
ROW6 = 6
ROW6_TOKENS = ("TOP-LEVEL", "MEMBER", "k=4", "k=104", "INCUMBENT")

# C3. The two phrasings the rate is stated in, and the disclosure a definition must carry.
RATE_DEFINITION = re.compile(
    r"0\.45k\s+e-6|\+0\.(?:45|55)e-6 of optimism per free parameter")
RATE_RANGE_TOKENS = ("k=3 AND k=4", "k=3 and k=4", "fitted at k=3")
RATE_WINDOW = 8
# A DEFINITION site issues the licence; a QUOTATION of prior text does not, and this document
# quotes its own superseded numbers constantly when recording a correction. The workspace's
# convention for that is `*"..."*`, so a match falling inside such a span is not a definition.
# ⚠ WHAT THIS IS BLIND TO, written down: a live claim wrapped in quote markers is exempted by
# this rule. The exemption is checked in the other direction by C5 -- the pre-fix w63 line is
# NOT quoted, so it must still fire -- but nothing stops a future run from hiding a real
# definition inside `*"..."*`.
QUOTED_SPAN = re.compile(r'\*"[^"]*"\*')

# C4. w36d_wsearch.json, verbatim, and the cross-arm number row 6 publishes. Frozen HERE by
# hand so this file compares rather than recomputes something that agrees with itself.
W36D = {
    "h3":   {"equal": 0.9701205752950482, "insample": 0.9701208766029016,
             "xfit": 0.9701196117472624, "optimism": 1.2648556392269583e-06},
    "all4": {"equal": 0.9701181652381732, "insample": 0.9701213202375878,
             "xfit": 0.9701195043323168, "optimism": 1.8159052710409185e-06},
}
CROSS_ARM = -1.0710e-6
W36D_TOL = 1e-9
CROSS_TOL = 1e-9

# C5. The text exactly as it stood before this run's edit.
PREFIX = {
    "row6_cell": "a **SEARCH** price (re-weighting members already in) — **−1.07e-6**",
    # the w63 restatement, which had carried the superseded /k rate since w108 corrected it
    "rate_definition": "WORSE**, at a measured **+0.45e-6 of optimism per free parameter**. "
                       "Hill-climbing is closed",
}


def index_rows(txt):
    """{row number -> price cell} from the ANGLE INDEX table. Anchored on the block head so a
    `| 6 |` line elsewhere in the document cannot be picked up (w110's first-occurrence trap)."""
    k = txt.find(ANGLE_HEAD)
    if k < 0:
        return {}
    rows = {}
    for ln in txt[k:].splitlines():
        m = re.match(r"^\| (\d+) \|", ln)
        if m:
            cells = [c.strip() for c in ln.split(" | ")]
            if len(cells) > 3:
                rows[int(m.group(1))] = cells[3]
        if ln.startswith("| 10 |"):
            break
    return rows


def unscoped(rows):
    """C1 predicate: priced SEARCH cells that name no layer, or no k, or neither."""
    bad = []
    for n, c in sorted(rows.items()):
        if not (MAGNITUDE.search(c) and claims(c, SEARCH_WORD)):
            continue
        miss = []
        if not any(w in c for w in LAYER_WORDS):
            miss.append("a LAYER word " + str(list(LAYER_WORDS)))
        if not K_TOKEN.search(c):
            miss.append("a `k=`")
        if miss:
            bad.append((n, miss))
    return bad


def missing_row6(cell):
    """C2 predicate: which of row 6's two searches the cell does not publish."""
    return [t for t in ROW6_TOKENS if t not in cell]


def unranged_rate(txt):
    """C3 predicate: rate DEFINITION sites that do not disclose the k they were fitted at."""
    lines = txt.splitlines()
    bad = []
    for i, ln in enumerate(lines):
        m = RATE_DEFINITION.search(ln)
        if not m:
            continue
        if any(q.start() <= m.start() and m.end() <= q.end()
               for q in QUOTED_SPAN.finditer(ln)):
            continue                      # a quotation of superseded text, not a definition
        win = "\n".join(lines[i:i + RATE_WINDOW])
        if not any(t in win for t in RATE_RANGE_TOKENS):
            bad.append((i + 1, ln.strip()[:110]))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1/C2/C3 over the frozen PRE-FIX text; all three must fire")
    a = ap.parse_args()

    txt = open(RESEARCH).read()
    rows = index_rows(txt)

    if a.control:
        print("w126c --control: C1/C2/C3 against the frozen PRE-FIX text\n")
        pre_rows = dict(rows)
        pre_rows[ROW6] = PREFIX["row6_cell"]
        c1 = unscoped(pre_rows)
        c2 = missing_row6(PREFIX["row6_cell"])
        # Restore the w63 restatement to the frozen pre-fix literal -- the whole line, so the
        # inline range annotation goes with it. Blanket-stripping every line carrying the
        # range token would delete this line entirely and the control would silently lose its
        # most important site (the one that had been stale for 22 runs).
        shipped_w63 = next((l for l in txt.splitlines()
                            if "of optimism per free parameter" in l), None)
        pre_txt = txt.replace(shipped_w63, PREFIX["rate_definition"]) if shipped_w63 else txt
        # the other three sites got the range on a line of their own; drop those
        pre_txt = "\n".join(l for l in pre_txt.splitlines()
                            if l.strip() and not l.lstrip().startswith("⚠ **FITTED AT k=3"))
        c3 = unranged_rate(pre_txt)
        print(f"  C1 pre-fix row {ROW6} cell {PREFIX['row6_cell']!r}")
        print(f"     -> {len(c1)} unscoped priced SEARCH cell(s)   "
              f"{'FIRES' if c1 else 'silent -- INERT'}")
        for n, miss in c1:
            print(f"        row {n} names neither " + " nor ".join(miss))
        print(f"  C2 -> row {ROW6} does not publish {c2}   "
              f"{'FIRES' if c2 else 'silent -- INERT'}")
        print(f"  C3 -> {len(c3)} unranged rate definition site(s)   "
              f"{'FIRES' if c3 else 'silent -- INERT'}")
        for lineno, ln in c3:
            # NOT a live line number: these index the synthetic pre-fix document this arm
            # builds in memory. Labelling them `RESEARCH.md:` would invite a grep at a line
            # that does not hold that text -- the exact trap w99/w101 recorded about line
            # numbers in this file.
            print(f"        (pre-fix doc):{lineno}  {ln}")

        s1, s2, s3 = unscoped(rows), missing_row6(rows.get(ROW6, "")), unranged_rate(txt)
        print(f"\n  shipped: C1 {len(s1)} unscoped · C2 missing {s2} · C3 {len(s3)} unranged")
        inert = [n for n, fired in (("C1", c1), ("C2", c2), ("C3", c3)) if not fired]
        if inert:
            print(f"\n  ⛔ INERT: {inert} did not fire on the pre-fix text. A control that "
                  f"cannot separate is decoration, not a control.")
        else:
            print("\n  ✅ all three fire on the pre-fix text and are silent on the shipped text")
        return 0

    fails = []

    def fail(m):
        fails.append(m)
        print("  FAIL " + m)

    print("w126c -- #55: a SEARCH price must name its SCOPE\n")

    # C1 ------------------------------------------------------------------
    priced_search = [n for n, c in sorted(rows.items())
                     if MAGNITUDE.search(c) and SEARCH_WORD in c]
    if not priced_search:
        fail("C1 is VACUOUS: no ANGLE INDEX price cell carries both a magnitude and "
             f"`{SEARCH_WORD}`. Either the index moved or the predicate stopped matching it.")
    else:
        bad = unscoped(rows)
        for n, miss in bad:
            fail(f"C1 ANGLE INDEX row {n} prices a {SEARCH_WORD} and names " +
                 " nor ".join(miss) + ". -1.07e-6 is a k=4 top-level number and the member-"
                 "layer search it sits next to is +2,343e-6 at k=104 (w126a).")
        if not bad:
            print(f"  C1 OK all {len(priced_search)} priced {SEARCH_WORD} cell(s) "
                  f"{priced_search} name a layer and a k")

    # C2 ------------------------------------------------------------------
    miss = missing_row6(rows.get(ROW6, ""))
    if miss:
        fail(f"C2 ANGLE INDEX row {ROW6} does not publish {miss}. It carries two searches "
             f"whose prices differ in sign and by ~2,000x; the cell must name both.")
    else:
        print(f"  C2 OK row {ROW6} publishes both searches {list(ROW6_TOKENS)}")

    # C3 ------------------------------------------------------------------
    sites = [i + 1 for i, ln in enumerate(txt.splitlines()) if RATE_DEFINITION.search(ln)]
    if not sites:
        fail("C3 is VACUOUS: the search-cost rate is stated nowhere in RESEARCH.md under "
             "either phrasing. The predicate has stopped matching the document.")
    bad = unranged_rate(txt)
    for lineno, ln in bad:
        fail(f"C3 RESEARCH.md:{lineno} states the search-cost rate as a property and does not "
             f"disclose the k it was fitted at (k=3 and k=4, two points): {ln}")
    if sites and not bad:
        print(f"  C3 OK all {len(sites)} rate definition site(s) disclose their fitted k")

    # C4 ------------------------------------------------------------------
    if not os.path.exists(ROW6_JSON):
        fail(f"C4 is INERT: {os.path.basename(ROW6_JSON)} is not on disk, so row 6's price is "
             f"unbacked. A missing artefact is not a pass.")
    else:
        d = json.load(open(ROW6_JSON))
        r2 = d.get("r2", {})
        n_ok = 0
        for tag, cells in W36D.items():
            got = r2.get(tag)
            if not got:
                fail(f"C4 w126a_row6.json has no `{tag}` arm; row 6's evidence is unbacked")
                continue
            for f, want in cells.items():
                if abs(got.get(f, float('nan')) - want) > W36D_TOL:
                    fail(f"C4 w36d {tag}.{f} measures {got.get(f)!r} against the published "
                         f"{want!r}")
                else:
                    n_ok += 1
        cross = d.get("cross_arm")
        if cross is None or abs(cross - CROSS_ARM) > CROSS_TOL:
            fail(f"C4 the cross-arm number row 6 publishes measures {cross!r}, "
                 f"published {CROSS_ARM!r}")
        else:
            print(f"  C4 OK {n_ok}/8 w36d cells and the cross-arm {cross*1e6:+.4f}e-6 "
                  f"reproduce from the OOF arrays to {W36D_TOL:g}")

    print(f"\nFAILURES: {len(fails)}")
    if not fails:
        print("✅ CLEAN — every SEARCH price names which search, at which layer, over how many "
              "parameters")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
