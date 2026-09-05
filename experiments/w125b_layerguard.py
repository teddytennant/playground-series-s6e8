"""w125b -- STANDING CHECK #54. A PRICE IS A NUMBER, A QUANTITY, AND A LAYER. #53 GAVE THE
ANGLE INDEX'S PRICE COLUMN ITS QUANTITIES. NONE OF THEM CARRIES A LAYER. (w125, 2026-08-30)

WHAT WENT WRONG. Nine of the ANGLE INDEX's ten price cells hold STACK-layer numbers -- e-6 of
blend CV, which is the thing the workspace ships. Row 5's price cell reads `negative`, and the
numbers carrying it are MEMBER-layer:

    -19.26e-6 (xgb)   -82.68e-6 (cat)   solo fold AUC, on top of the LightGBM null

Same column, same heading, different layer, and nothing says so. The workspace's own standing
rule says these are not interchangeable, in terms: *"member-level AUC is not evidence about
stack value ... the sign is not even guaranteed"* (w106) and *"quote a member's fold AUC as a
member number and never as a blend number"*. Row 5 is the one cell that does exactly that.

🎯 SAME DEFECT GENUS, SIXTH PLACE. w120 §4 `priority` naming an input, w121 §3 a prose cause
beside a derived count, w122 §2 `slot` printed for `tier`, w123 §3 a family label on a residual
group, w124 §3 a magnitude with no unit -- and now a magnitude with a unit but no layer. #53
closed the QUANTITY hole and could not see this one: `negative` carries no `e-6`, so #53's C1
exempts it by design.

AND THE HOLE HAS A SECOND MOUTH. The `1.4%` solo->stack pass-through is quoted in thirteen
places, and row 4's entire `+4e-7` price cell is DERIVED by multiplying through it. It was
fitted on ONE point: seed-averaging `xgb_latcat`, +138e-6 solo -> +2e-6 stack. Every in-range
use (3e-5, 84e-6) is defensible. Nothing in the document says what the range is, so a later run
reading "pass-through is 1.4%" has no way to know it is extrapolating when it multiplies a
member-level effect two orders of magnitude further out. w125a measured such an effect -- the
whole TE_+CT_ encoding channel, +13,252e-6 at member level -- and it does not convert at 1.4%.

WHAT THIS GUARD ENFORCES.

  C1  LAYER. The ANGLE INDEX row 5 price cell must name both layers its evidence sits at:
      MEMBER (the -19.26e-6/-82.68e-6 solo readings the closure was argued from) and STACK
      (the enrolment price w125a measured on base104). A reader comparing row 5 to row 6 is
      comparing two numbers and must be able to see whether they are the same kind of number.
  C2  THE LADDER'S PROSE CELL. The feature-block ablation ladder publishes `tedrop`'s pooled
      OOF as "*below `encdrop`*" -- prose in a column of numbers, in two tables. That is why
      the CT_-on-raw marginal is published as "≈ -2,700e-6" when the arrays give it exactly.
      The cell must carry a number and it must match the array on disk.
  C3  THE PASS-THROUGH'S RANGE. Wherever RESEARCH.md states the pass-through as a PROPERTY
      ("solo→stack pass-through is 1.4%"), the statement must disclose the range it was fitted
      over. Applying it is fine; publishing it as a bare constant is what licenses the
      extrapolation. This is a check on the DEFINITION sites, not on every use.
  C4  THE ARRAYS STILL SAY IT. The ladder's four published pooled-OOF figures are recomputed
      from the .npy files and must reproduce to 1e-9. A table of measured numbers with no
      re-derivation is the w124 §6 defect; this one is cheap enough to check every run.
  C5  --control: the C1/C2/C3 predicates over the FROZEN pre-fix text, which must fire. If
      they do not, this guard reports INERT rather than green -- a fix with no alternative
      behaviour to select from is not a fix (w121 §5, w122 §5).

⚠ JOURNAL.md is append-only and NOT in scope. Only RESEARCH.md, the live document, is checked.

    .venv/bin/python experiments/w125b_layerguard.py            # rc 0 = clean
    .venv/bin/python experiments/w125b_layerguard.py --control  # exits 0, having fired

Deterministic: reads one document and four .npy arrays. No fit, no API call.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, TARGET, load_raw   # noqa: E402

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"

# C1. Row 5 is the only cell whose evidence is member-layer, so it is the only cell required to
# name a layer. Broadening this to "every row must say STACK" would be noise: TUNING, ENROLMENT,
# CONCAT and SEARCH are stack-layer by definition and #53 already pins them.
LAYER_ROW = 5
LAYER_TOKENS = ("MEMBER", "STACK")

# C2. The prose cell, in both tables that carry it.
LADDER_PROSE = re.compile(r"\*below\s+`?encdrop`?\*", re.I)

# C3. A DEFINITION site states the pass-through as a property of the workspace. A USE site
# multiplies by it. Only definition sites need the range -- that is where the licence is issued.
# ⚠ (w181) `solo[-→>]+stack` did not admit the spelled-out "Solo-TO-stack", and the whole
# pattern did not admit the "X HAS ~1.4% pass-through" word order. Both forms appear in
# RESEARCH.md stating the rate as a property of the workspace, and both were invisible.
PT_DEFINITION = re.compile(
    r"(?:solo\s*(?:[-→>]+|-to-)\s*stack\s+)?pass-through\s+(?:here\s+)?is\s+(?:~\s*)?\*{0,2}1\.4%"
    r"|has\s+(?:~\s*)?\*{0,2}1\.4%\*{0,2}\s+pass-through",
    re.I)
# the range disclosure the definition must carry, within its own line or the next two
PT_RANGE_TOKENS = ("138e-6", "fitted", "one point", "not a constant", "range")
# ⚠ (w181) #55 has had this exemption since w126; #54 never did, and it showed. w181's own entry
# quotes the matched sentence as `*"Solo→stack pass-through here is ~1.4%"*` to compare it against
# the wrapped twin -- a MENTION -- and it passed only because the following prose contains the
# word "ranged", whose substring "range" is a disclosure token. A site that passes by accident is
# indistinguishable from one that passes on purpose, so quotations are now exempted outright.
QUOTED_SPAN = re.compile(r'\*"[^"]*"\*')
PT_WINDOW = 3

# C4. The ladder's published pooled-OOF column, frozen as literals so this is a COMPARISON.
LADDER = {
    "lat_ctraw_r400":   (os.path.join(DATA, "ext_members6"), 0.9654813306),
    "lat_ctfix_r400":   (os.path.join(DATA, "ext_members6"), 0.9657751945),
    "lat_ctdrop_r400":  (os.path.join(DATA, "ext_members7pin"), 0.9656895129),
    "lat_encdrop_r400": (os.path.join(DATA, "ext_members7pin"), 0.9522288823),
}
LADDER_TOL = 1e-9
# w125a, measured: `tedrop` is the cell the ladder carried as prose.
TEDROP = ("lat_tedrop_r400", os.path.join(DATA, "ext_members7"), 0.9496361971)

# C5. The text exactly as it stood before this run's edit, inlined so the control does not
# depend on a file a later run may overwrite (w115: freeze the control, do not read HEAD).
PREFIX = {
    "row5_cell": "**negative**",
    "ladder_cell": "| `tedrop` (drop `TE_`) | 112 | *below `encdrop`* | | | |",
    # The site that ALREADY named its fitted point ("seed-averaging `xgb_latcat` bought
    # +138e-6 solo") is not the control -- C3 is silent on it by design, both before and
    # after, which is the whole point of scoping C3 to sites that lack a range. The control
    # is one of the two that did lack one: RESEARCH.md:1957 as it stood before this run.
    "pt_definition": "value.** Solo→stack pass-through here is ~1.4% and the sign is not "
                     "even guaranteed.",
}


def index_rows(txt):
    """{row number -> price cell} from the ANGLE INDEX table. Anchored on the block head so a
    `| 5 |` line elsewhere in the document cannot be picked up (w110's first-occurrence trap)."""
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


def missing_layer(cell):
    """C1 predicate: which required layer words the row 5 price cell does not name."""
    return [t for t in LAYER_TOKENS if t not in cell]


def prose_cells(txt):
    """C2 predicate: lines of a pooled-OOF column that carry prose where a number belongs."""
    return [ln for ln in txt.splitlines() if LADDER_PROSE.search(ln)]


# ⚠ (w181) A SPECIMEN IS NOT A DEFINITION. This entry's own diagnosis quotes the defective sites
# verbatim inside indented blocks, and both C3 arms went red on that prose -- the same way #75 went
# red on w180's entry for quoting a counter. An indented block is the workspace's verbatim-specimen
# convention; measured over RESEARCH.md, every one of the 11 genuine rate/pass-through definitions
# is unindented and every specimen is indented, so the exemption separates them cleanly and loses
# no real site. Checked, not assumed -- see the w181 entry's site table.
def _is_specimen(line):
    """A verbatim specimen (indented block) MENTIONS a rule; it does not assert one."""
    return line.startswith("    ")

def _checked_sites(txt):
    """How many definition sites C3 actually ADJUDICATED. ⚠ (w181) this used to be
    `len(PT_DEFINITION.findall(txt))`, i.e. every raw match, so the guard printed "all N sites
    disclose their fitted range" about a set that included the specimens and quotations it had
    just skipped. A count that names a different set than the claim it supports is #53's defect
    genus at the guard's own output."""
    lines = txt.splitlines()
    n = 0
    for i, ln in enumerate(lines):
        joined = ln + " " + (lines[i + 1] if i + 1 < len(lines) else "")
        m = PT_DEFINITION.search(joined)
        if not m or m.start() >= len(ln) or _is_specimen(ln):
            continue
        if any(q.start() <= m.start() and m.end() <= q.end()
               for q in QUOTED_SPAN.finditer(joined)):
            continue
        n += 1
    return n


def unranged_definitions(txt):
    """C3 predicate: pass-through DEFINITION sites that do not disclose their fitted range."""
    # ⚠ (w181) THIS USED TO MATCH ONE LINE AT A TIME, and a definition that WRAPPED was
    # invisible. RESEARCH.md:19547 reads "Solo-to-stack pass-through here is\n~1.4%." -- the same
    # sentence as the site matched at 4810, differing only in the arrow spelling and a line
    # break, and it carried no range. Matching a claim that spans a wrap means matching against
    # the joined pair, then requiring the match to START on this line so a hit is not counted
    # twice.
    lines = txt.splitlines()
    bad = []
    for i, ln in enumerate(lines):
        joined = ln + " " + (lines[i + 1] if i + 1 < len(lines) else "")
        m = PT_DEFINITION.search(joined)
        if not m or m.start() >= len(ln):
            continue                      # no match, or it belongs to the next line
        if _is_specimen(ln):
            continue                      # a verbatim specimen, not a definition
        if any(q.start() <= m.start() and m.end() <= q.end()
               for q in QUOTED_SPAN.finditer(joined)):
            continue                      # a quotation of prior text, not a definition
        win = "\n".join(lines[i:i + PT_WINDOW])
        if not any(t in win for t in PT_RANGE_TOKENS):
            bad.append((i + 1, ln.strip()))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run the C1/C2/C3 predicates over the frozen PRE-FIX text; must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()

    if a.control:
        print("w125b --control: the C1/C2/C3 predicates against the frozen PRE-FIX text\n")
        c1 = missing_layer(PREFIX["row5_cell"])
        c2 = prose_cells(PREFIX["ladder_cell"])
        c3 = unranged_definitions(PREFIX["pt_definition"])
        print(f"  C1 pre-fix row 5 cell {PREFIX['row5_cell']!r}")
        print(f"       -> names no layer: missing {c1}  [{'FIRES' if c1 else 'silent'}]")
        print(f"  C2 pre-fix ladder cell -> {len(c2)} prose cell(s) in a numeric column  "
              f"[{'FIRES' if c2 else 'silent'}]")
        print(f"  C3 pre-fix definition  -> {len(c3)} unranged definition site(s)  "
              f"[{'FIRES' if c3 else 'silent'}]")
        rows = index_rows(txt)
        s1 = missing_layer(rows.get(LAYER_ROW, ""))
        s2 = prose_cells(txt)
        s3 = unranged_definitions(txt)
        print(f"\n  shipped: C1 missing {s1} · C2 {len(s2)} prose cell(s) · "
              f"C3 {len(s3)} unranged site(s)")
        fired = bool(c1) + bool(c2) + bool(c3)
        shipped = bool(s1) + bool(s2) + bool(s3)
        if not fired:
            print("\n  ** INERT ** the pre-fix predicates find nothing, so the guard has no "
                  "alternative behaviour to select from and is not evidence of a fix.")
        elif fired == shipped:
            print("\n  ** INERT ** both sides agree; the fix changed nothing observable.")
        else:
            print(f"\n  ✅ the pre-fix text fires {fired}/3 predicates and the shipped text "
                  f"fires {shipped}/3.")
        return 0

    fails, out = [], {}

    def fail(m):
        fails.append(m)
        print("  FAIL " + m)

    print("w125b -- a price is a number, a quantity AND a layer\n")

    rows = index_rows(txt)
    if not rows:
        fail("the ANGLE INDEX table did not parse; every check below is vacuous")
        print("\nFAILURES: 1")
        return 1

    # C1 ------------------------------------------------------------------
    cell = rows.get(LAYER_ROW, "")
    miss = missing_layer(cell)
    out["row5_cell"], out["row5_missing_layers"] = cell, miss
    if miss:
        fail(f"C1 ANGLE INDEX row {LAYER_ROW} does not name {miss}. Its closure is argued from "
             f"MEMBER-layer solo AUC (−19.26e-6 xgb, −82.68e-6 cat) while every other price in "
             f"that column is a STACK-layer number, and the cell does not say so. >> "
             f"{cell[:90]}")
    else:
        print(f"  C1 OK row {LAYER_ROW} names both layers {list(LAYER_TOKENS)}")

    # C2 ------------------------------------------------------------------
    prose = prose_cells(txt)
    out["prose_cells"] = prose
    if prose:
        for ln in prose:
            fail(f"C2 a pooled-OOF column still carries prose where a number belongs; the "
                 f"array is on disk and gives {TEDROP[2]:.10f}. >> {ln.strip()[:90]}")
    else:
        print("  C2 OK no prose cells left in the ablation ladder's numeric columns")

    # C3 ------------------------------------------------------------------
    unranged = unranged_definitions(txt)
    out["unranged_pt_definitions"] = unranged
    if unranged:
        for lineno, ln in unranged:
            fail(f"C3 RESEARCH.md:{lineno} states the solo→stack pass-through as a property "
                 f"with no fitted range. It was fitted on one point (+138e-6 solo → +2e-6 "
                 f"stack); the encoding channel is +13,252e-6 and does not convert at 1.4%. "
                 f">> {ln[:90]}")
    else:
        n = _checked_sites(txt)
        print(f"  C3 OK all {n} pass-through definition site(s) disclose their fitted range")

    # C4 ------------------------------------------------------------------
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    from sklearn.metrics import roc_auc_score
    recomputed = {}
    for nm, (d, pub) in list(LADDER.items()) + [(TEDROP[0], (TEDROP[1], TEDROP[2]))]:
        p = os.path.join(d, f"oof_{nm}.npy")
        if not os.path.exists(p):
            fail(f"C4 the ladder cites {nm}; {p} is not on disk")
            continue
        got = float(roc_auc_score(y, np.load(p)))
        recomputed[nm] = got
        if abs(got - pub) > LADDER_TOL:
            fail(f"C4 {nm} recomputes at {got:.10f}, the ladder publishes {pub:.10f}")
    if not any(m.startswith("C4") for m in fails):
        print(f"  C4 OK all {len(recomputed)} ladder arms reproduce from the arrays to "
              f"{LADDER_TOL:g}")
    out["ladder_recomputed"] = recomputed

    print(f"\nFAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    if not fails:
        print("✅ CLEAN — every published price names its layer, and the ladder has no prose "
              "left in its numeric columns")
    out["failures"] = fails
    with open(os.path.join(HERE, "w125b_layerguard.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
