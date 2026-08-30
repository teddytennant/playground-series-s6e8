"""w130b -- STANDING CHECK #59. A PRICE OF ZERO IS UNREADABLE WITHOUT ITS BASELINE, AND ZERO
IS THE ONE MAGNITUDE THE GUARD FAMILY ACCEPTS WITHOUT ONE. (w130, 2026-08-30)

THE DEFECT THIS EXISTS BECAUSE OF. Row 8 -- foundation, x13 handings -- read, in full:

    **NOT A PRICE** -- a foundation row, and the answer is **0**, and it holds on its own
    artefacts: the metric is a table row, the folds are frozen since w38 and verified
    against four public packs by #33, and the GBDT baselines are on disk

#57's C1 is satisfied by "a magnitude OR the bare `0`", so one character buys the green. w129
went further and cited this cell as the HONEST counterexample to row 10's opt-out ("row 8
passes on the `0`"). Five guards -- #53, #55, #56, #57, #58 -- are green on it, and C5 below
runs all five to show that is not an assertion.

WHAT THE `0` DOES NOT SAY IS WHAT IT IS MEASURED AGAINST. Every other priced row names its
baseline: `base104` without the member, the uncorrected stack, the shipped file's own bytes,
Kaggle's auto-selection by public score. Row 8's silent baseline is "the foundation ALREADY
EXISTS" -- built once, at w38, so re-doing it today buys nothing. That is a REPEAT price.
Priced against its own ABSENCE the same row is the LARGEST number in the table (w130a):

    A  confirm the metric        METRIC       FINAL-FILE   k=1    +63,263.9e-6
    B  the fixed-fold harness    MEASUREMENT  CV-ESTIMATE  per comparison, an sd
                                                  3.44e-6 paired vs 271.11e-6 unpaired
    C  one honest GBDT baseline  FOUNDATION   OOF          TOTAL +467,789.9e-6

⟹ A MAGNITUDE THAT SATISFIES EVERY CHECK IN THE FAMILY AND DENOTES NOTHING. The price column
is what a run reads when deciding where to spend a slot, and a `0` sitting in a column where
+10.04e-6 is published as a reason NOT to build says the foundation is the cheapest thing in
the table. It is 46,585x that bar.

⚠ AND THE WORD "baseline" IS NOT THE TEST. The pre-fix cell CONTAINS it -- "the GBDT baselines
are on disk", a noun for a model -- and contains "against" too -- "verified against four public
packs", a verification. A naive `"baseline" in cell` reader passes row 8. C6 requires this
file's reader to DISAGREE with that naive test, on row 8, in both directions. Same genus as
w129's citation defect: the token is present and means something else.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   A HEADLINE ZERO MUST NAME ITS BASELINE. A price cell whose zero stands before the
         cell's first non-zero e-6/e-7 magnitude -- or which carries no non-zero magnitude at
         all -- must contain a BASELINE ASSERTION. Rows 7, 9 and 10 carry zeros and pass
         because they name theirs; pre-fix rows 1 and 8 fire. Reports the row count and the
         number of headline-zero cells, so it cannot pass vacuously.
  C2 +   ROW 8 PUBLISHES ALL EIGHTEEN FACETS: three quantities, three layers, the k, the three
         baselines, the three magnitudes, and the REPEAT/ABSENCE split that says which baseline
         the `0` belonged to.
  C3 +-  THE ARTEFACT STILL SAYS IT -- `w130a_row8.json`, expected values frozen HERE as
         literals so this is a COMPARISON and not a recomputation that agrees with itself.
         INERT, not green, if the json is missing (#55's C4 rule).
  C4 +-  --control runs C1 over the FROZEN PRE-FIX rows 1 and 8, inlined as literals rather
         than read from HEAD (w115: a control anchored to HEAD stops being a control the moment
         the fix lands). Both must FIRE; the shipped table must be silent.
  C5 +-  THE SIBLINGS' BLINDNESS IS MEASURED, NOT ASSERTED. #53/#55/#56/#57/#58 are imported and
         their own C1 predicates run over a table whose row 8 is the pre-fix cell. All five must
         return ZERO findings for row 8 while this file's C1 fires; and each must still fire on
         ITS OWN frozen defect in the same call, else the demonstration is vacuous.
  C6 +-  THE BASELINE READER DISCRIMINATES. On the pre-fix row 8 cell the naive `"baseline" in
         cell` test must say GROUNDED and this file's reader must say NOT; on the shipped row 10
         cell both must say GROUNDED. If the two readers agree everywhere, the grammar is doing
         nothing and C1 is the naive test wearing a hat.

WHAT C1 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST EIGHT DEFECTS LIVED. It only
sees a HEADLINE zero -- a zero introduced late in a cell, after a real magnitude, is out of
scope by construction. Its baseline grammar is finite: a cell can name a baseline in a
construction I did not enumerate (`on base104`, `relative to h3`) and fire falsely, or dress a
decoy in one I did (`against a control at +10.04e-6`) and pass. And it reads only the price
COLUMN, like all five siblings; the `closed` column and the prose pool disclosures remain
unparsed by anything but `w117a_handcount`. 🎯 Six guards now share one column and five share
one vocabulary. If the next defect is in this genus it is in the `closed` column, or it is a
baseline that is named and WRONG -- nothing here checks that the stated baseline is the one the
number was actually measured against.

    .venv/bin/python experiments/w130b_zerobaselineguard.py            # rc 0 = clean
    .venv/bin/python experiments/w130b_zerobaselineguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
ROW8_JSON = os.path.join(HERE, "w130a_row8.json")

import w128b_pricedguard as g57  # noqa: E402 -- the shared reader and the shared table parser

index_rows = g57.index_rows

ROW8 = 8

# A magnitude that is not zero. Deliberately NOT g57.MAGNITUDE, which matches zeros too.
NONZERO = re.compile(r"[+−-]?(?!0(?:\.0+)?\s*e-[67])\d[\d,]*(?:\.\d+)?\s*e-[67]")
# A zero standing on its own: `0`, `**0**`, `+0.0000e-6`. Not `0.5`, not `1,000e-6`, not the
# `0` of `splits 0/1/2`, and not `50/50`.
ZERO = re.compile(r"(?<![\w.+\-])(?:\*\*)?[+−-]?0(?:\.0+\s*e-[67])?(?:\*\*)?(?!,\d)(?![\d.eE%/])")
# A BASELINE ASSERTION: the word used as "the thing this is measured against", not as a noun
# for a model ("the GBDT baselines ARE on disk"), and `against` followed by a thing rather than
# by a count ("verified against four public packs").
BASELINE_ASSERT = re.compile(
    r"\bbaselines?\b(?!\s+(?:are|is|was|were)\b)\s*[:,]?\s*\*{0,2}"
    r"(?:the|a|an|its|no|zero|chance|Kaggle)\b"
    r"|\b(?:measured|priced|read)\s+against\b"
    r"|\bagainst\s+\*{0,2}(?:the|a|an|its|chance|no|zero|Kaggle|size-matched)\b",
    re.I)
NAIVE = re.compile(r"baseline", re.I)          # C6: the reader this one must beat

# C2. Row 8's eighteen required tokens: three quantities, three layers, the k, three baselines,
# three magnitudes, the two-baseline split, and the non-addability.
ROW8_TOKENS = (
    "METRIC",                       # arm A quantity
    "MEASUREMENT",                  # arm B quantity
    "FOUNDATION",                   # arm C quantity
    "REPEAT",                       # the quantity the `0` itself was
    "FINAL-FILE layer",             # arm A layer
    "CV-ESTIMATE layer",            # arm B layer
    "OOF layer",                    # arm C layer
    "k=1",                          # arm A scope
    "+63,263.9e-6",                 # arm A magnitude
    "+122,243.5e-6",                # ...at the naive cut
    "271.11e-6",                    # arm B magnitude (the one that matters)
    "+467,789.9e-6",                # arm C magnitude
    "+1,325.4e-6",                  # ...and what everything else bought
    "thresholded to a hard class",  # arm A baseline
    "an unpaired harness",          # arm B baseline
    "a constant prediction",        # arm C baseline
    "ABSENCE",                      # the second baseline, named
    "NOT addable",                  # three currencies do not sum
)

# C3. Frozen literals -- w130a_row8.json, written by w130a from arrays on disk.
EXPECT = {
    "threshold_cost_e6": 63263.942081922876,
    "threshold_cost_naive_e6": 122243.48908409821,
    "sd_paired_e6": 3.4362624733717047,
    "sd_unpaired_e6": 271.10725772166234,
    "foundation_vs_chance_e6": 467789.9182813025,
    "stack_minus_best_member_e6": 1325.4171071855892,
    "auc_shipped": 0.9701400059625016,
    "best_single_member": 'oof_naji05',
    "rtol": 1e-9,
}

# C4/C5/C6. The two cells exactly as they stood before this run's edit.
PREFIX_ROW8 = ("**NOT A PRICE** — a foundation row, and the answer is **0**, and it holds on "
               "its own artefacts: the metric is a table row, the folds are frozen since w38 "
               "and verified against four public packs by #33, and the GBDT baselines are on "
               "disk")
PREFIX_ROW1 = ("a **CONCAT** price (extra training ROWS, not members). **0, and the usual "
               "Playground edge is INVERTED here: −58e-6 at 1× dose, −3,340e-6 at 50×; the "
               "best separate-estimator route is −1e-6 to −2e-6 in the stack**")


def headline_zeros(cell):
    """The zero tokens that stand before the cell's first non-zero magnitude. A cell with no
    non-zero magnitude at all has ALL of its zeros in headline position."""
    m = NONZERO.search(cell)
    cut = m.start() if m else len(cell)
    return [z.group(0) for z in ZERO.finditer(cell) if z.start() < cut]


def grounded(cell):
    """True if the cell names a baseline in a construction that asserts one."""
    return BASELINE_ASSERT.search(cell) is not None


def ungrounded(rows):
    """C1 predicate: price cells whose headline price is zero and that name no baseline."""
    bad = []
    for n, c in sorted(rows.items()):
        hz = headline_zeros(c)
        if hz and not grounded(c):
            bad.append((n, hz, c[:110]))
    return bad


def missing_row8(cell):
    """C2 predicate: which of row 8's required tokens the cell does not publish."""
    return [t for t in ROW8_TOKENS if t not in cell]


def sibling_predicates():
    """The five siblings' own C1 predicates, imported rather than reimplemented, each paired
    with the frozen pre-fix cell it was built for so C5 can show it still fires somewhere."""
    import w124b_priceunitguard as g53
    import w126c_scopeguard as g55
    import w127b_rateguard as g56
    import w129b_optoutguard as g58
    return [
        ("#53 w124b unpriced (quantity)", g53.unpriced, 4, g53.PREFIX_CELLS[4]),
        ("#55 w126c unscoped (scope)", g55.unscoped, 6, g55.PREFIX["row6_cell"]),
        ("#56 w127b undenominated (denominator)", g56.undenominated, 7, g56.PREFIX_ROW7),
        ("#57 w128b magnitudeless (silence)", g57.magnitudeless, 9, g57.PREFIX_ROW9),
        ("#58 w129b unpriced_claim (opt-out)", g58.unpriced_claim, 10, g58.PREFIX_ROW10),
    ]


def close(a, b, rtol):
    return abs(a - b) <= rtol * max(1.0, abs(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1 over the frozen PRE-FIX rows 1 and 8; both must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()
    rows = index_rows(txt)
    if not rows:
        print("w130b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w130b --control: C1 against the frozen PRE-FIX rows 1 and 8\n")
        pre = dict(rows)
        pre[1], pre[ROW8] = PREFIX_ROW1, PREFIX_ROW8
        hits = {n for n, _, _ in ungrounded(pre)}
        for n, cell in ((1, PREFIX_ROW1), (ROW8, PREFIX_ROW8)):
            print(f"  C1 pre-fix row {n}: zeros {headline_zeros(cell)}  "
                  f"baseline {'NAMED' if grounded(cell) else 'ABSENT'}  "
                  f"-> {'FIRES' if n in hits else 'SILENT (BAD)'}")
        c2 = missing_row8(PREFIX_ROW8)
        print(f"  C2 {'FIRES' if c2 else 'SILENT (BAD)'} -- pre-fix row 8 is missing "
              f"{len(c2)} of {len(ROW8_TOKENS)} tokens")
        s1 = ungrounded(rows)
        s2 = missing_row8(rows.get(ROW8, ""))
        print(f"\n  shipped C1 {'SILENT' if not s1 else 'FIRES (BAD): ' + str(s1)}")
        print(f"  shipped C2 {'SILENT' if not s2 else 'FIRES (BAD): missing ' + str(s2)}")
        works = {1, ROW8} <= hits and bool(c2) and not s1 and not s2
        print(f"\n  the control {'WORKS' if works else 'DOES NOT WORK'}: both pre-fix cells "
              f"fire, the shipped table is silent")
        return 0

    fails = []
    print("w130b -- #59: a price of zero must say what the zero is measured against\n")

    # ------------------------------------------------------------------------------ C1
    hz_cells = {n: headline_zeros(c) for n, c in rows.items() if headline_zeros(c)}
    bad = ungrounded(rows)
    print(f"C1 headline zeros must name a baseline")
    print(f"   {len(rows)} price cells parsed, {len(hz_cells)} carry a headline zero: "
          f"{sorted(hz_cells)}")
    for n in sorted(hz_cells):
        print(f"     row {n:2d}  zeros {hz_cells[n]}  "
              f"baseline {'NAMED' if grounded(rows[n]) else 'ABSENT'}")
    if len(rows) < 10:
        fails.append(f"C1 vacuous: only {len(rows)} price cells parsed")
    if not hz_cells:
        fails.append("C1 vacuous: no price cell carries a headline zero, so nothing was tested")
    if bad:
        fails.append(f"C1: {len(bad)} zero price(s) with no baseline: "
                     f"{[(n, z) for n, z, _ in bad]}")
    else:
        print("   OK -- every headline zero names its baseline")

    # ------------------------------------------------------------------------------ C2
    miss = missing_row8(rows.get(ROW8, ""))
    print(f"\nC2 row 8 publishes all {len(ROW8_TOKENS)} facets")
    if miss:
        fails.append(f"C2: row 8 does not publish {miss}")
    else:
        print("   OK -- three quantities, three layers, k=1, three baselines, three "
              "magnitudes, the REPEAT/ABSENCE split and the non-addability")

    # ------------------------------------------------------------------------------ C3
    print("\nC3 the artefact still says it")
    if not os.path.exists(ROW8_JSON):
        print(f"   INERT -- {os.path.basename(ROW8_JSON)} is missing; C3 asserts nothing")
    else:
        d = json.load(open(ROW8_JSON))
        got = {**d["arm_a"], **d["arm_b"], **d["arm_c"]}
        for k, v in EXPECT.items():
            if k == "rtol":
                continue
            g = got.get(k)
            if g is None:
                fails.append(f"C3: {k} absent from the artefact")
            elif isinstance(v, str):
                if g != v:
                    fails.append(f"C3: {k} is {g!r}, frozen as {v!r}")
            elif not close(float(g), v, EXPECT["rtol"]):
                fails.append(f"C3: {k} is {g}, frozen as {v}")
        if not any(f.startswith("C3") for f in fails):
            print(f"   OK -- {len(EXPECT) - 1} frozen values reproduce "
                  f"(A {EXPECT['threshold_cost_e6']:,.1f}e-6 · "
                  f"B {EXPECT['sd_paired_e6']:.2f}/{EXPECT['sd_unpaired_e6']:.2f}e-6 · "
                  f"C {EXPECT['foundation_vs_chance_e6']:,.1f}e-6)")
        if d.get("failures", 1) != 0:
            fails.append(f"C3: w130a reports {d.get('failures')} failure(s)")

    # ------------------------------------------------------------------------------ C5
    print("\nC5 the five siblings, run BLIND on the pre-fix row 8 cell")
    pre = dict(rows)
    pre[ROW8] = PREFIX_ROW8
    mine = [n for n, _, _ in ungrounded(pre)]
    for label, pred, own_row, own_cell in sibling_predicates():
        blind = [d for d in pred(pre) if d[0] == ROW8]
        probe = dict(rows)
        probe[own_row] = own_cell
        still = [d for d in pred(probe) if d[0] == own_row]
        print(f"   {label:42s} row 8: {'FINDS IT' if blind else 'blind'}   "
              f"own defect (row {own_row}): {'fires' if still else 'SILENT'}")
        if blind:
            fails.append(f"C5: {label} already covers row 8; #59 is redundant")
        if not still:
            fails.append(f"C5: {label} no longer fires on its own frozen defect -- vacuous")
    if ROW8 not in mine:
        fails.append("C5: this file's own C1 does NOT fire on the pre-fix row 8 cell")
    else:
        print(f"   {'#59 w130b ungrounded (baseline)':42s} row 8: FINDS IT")

    # ------------------------------------------------------------------------------ C6
    print("\nC6 the baseline reader must disagree with the naive `\"baseline\" in cell` test")
    probes = [
        ("pre-fix row 8", PREFIX_ROW8, True, False),      # naive passes, this reader must not
        ("shipped row 10", rows.get(10, ""), True, True),  # both must pass
    ]
    disagreements = 0
    for name, cell, want_naive, want_mine in probes:
        n_hit, m_hit = bool(NAIVE.search(cell)), grounded(cell)
        print(f"   {name:16s} naive={'GROUNDED' if n_hit else 'not'}   "
              f"#59={'GROUNDED' if m_hit else 'not'}")
        if n_hit != want_naive or m_hit != want_mine:
            fails.append(f"C6: {name} read as naive={n_hit}/mine={m_hit}, "
                         f"expected {want_naive}/{want_mine}")
        if n_hit != m_hit:
            disagreements += 1
    if disagreements == 0:
        fails.append("C6: the two readers agree on every probe -- the grammar does nothing")
    else:
        print(f"   OK -- they disagree on {disagreements} of {len(probes)} probes; the word "
              "`baselines` in \"the GBDT baselines are on disk\" is a noun, not a baseline")

    print()
    for f in fails:
        print(f"FAIL: {f}")
    print(f"FAILURES: {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
