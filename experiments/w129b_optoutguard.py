"""w129b -- STANDING CHECK #58. THE OPT-OUT IS NOT A PLACE TO MAKE A VALUE CLAIM.
(w129, 2026-08-30)

THE DEFECT THIS EXISTS BECAUSE OF. #57 (`w128b_pricedguard`) made silence unspendable: every
ANGLE INDEX price cell must carry an e-6/e-7 magnitude, or type the literal `NOT A PRICE`. Its
own blind-spot paragraph, written the same day, says what it cannot then do:

    "C1 checks for a magnitude token and accepts `NOT A PRICE` from anyone who types it."

Row 10 -- consolidation, x14 handings -- took the opt-out and then made a claim in the same cell:

    **NOT A PRICE -- not a modelling angle** -- it is the standing checklist, and it is the
    one angle that has ever PAID

"Has ever PAID" asserts a NON-ZERO POSITIVE magnitude, in the past tense, comparatively against
the other nine rows, with no number attached. Row 8's opt-out is honest by contrast: it is a
foundation row and its cell states that the answer is 0. 🎯 THE OPT-OUT WAS DESIGNED FOR CELLS
WITH NOTHING TO PRICE, AND ROW 10 HAD SOMETHING TO PRICE -- three arms, one per clause of its
own elaboration, all three measurable from artefacts already on disk (w129a_row10.py):

    A  confirm the picks         SELECTION layer,  k=2, per competition   +4.5228e-6
    B  re-verify the pipeline    PIPELINE layer,   per file rebuilt       +0.0000e-6 (a PASS)
    C  audit CV<->LB             PREDICTOR layer,  PREDICTED-LB units    -27.4266e-6

⟹ AN EXEMPTION GRANTED FOR "NOTHING TO MEASURE" GETS USED FOR "NOTHING MEASURED YET", AND THE
CELL THAT MOST NEEDS A NUMBER IS THE ONE ARGUING FOR ITS OWN ANGLE. That is what C1 catches.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   A CELL THAT OPTS OUT MUST NOT MAKE A VALUE CLAIM. Any price cell containing the
         literal `NOT A PRICE` and any of the VALUE words (PAID / paid / pays / worth / gain /
         value) must ALSO state a magnitude or the literal bare `0`. Row 8 passes on the `0`;
         pre-fix row 10 fires. Reports the row count so it cannot pass vacuously.
  C2 +   ROW 10 PUBLISHES ALL SIX FACETS AND THE REALISED/CONTINGENT SPLIT. Required tokens
         below: three quantities, three layers, a k, three denominators, three baselines, and
         the fact that the headline is contingent on a click nobody has made.
  C3 +-  THE ARTEFACT STILL SAYS IT -- `w129a_row10.json`, with the expected values frozen HERE
         as literals so this is a COMPARISON and not a recomputation that agrees with itself.
         INERT, not green, if the json is missing (#55's C4 rule).
  C4 +-  --control runs C1 and C2 over the FROZEN PRE-FIX row 10 cell, inlined as a literal
         rather than read from HEAD (w115: a control anchored to HEAD stops being a control the
         moment the fix lands). Pre-fix must FIRE on both; shipped must be silent on both.
  C5 +-  THE SIBLINGS' BLINDNESS IS MEASURED, NOT ASSERTED. #53/#55/#56/#57 are imported and
         their own C1 predicates run over a table whose row 10 is the pre-fix cell. All four
         must return ZERO findings for row 10 -- #57 included, because the opt-out satisfies it
         -- while this file's C1 fires; and each must still fire on ITS OWN frozen defect in
         the same call, else the demonstration is vacuous.
  C6 +-  THE CITATION READER DISCRIMINATES. w129 found the second reader defect in this family:
         row 10's rewritten cell cites "row 3's CatBoost ENROLMENT price", and `claims()` read
         that borrowed word as row 10's own quantity -- #53 went GREEN on a cell whose three
         real quantities were not in its vocabulary at all. Delete the six words of the citation
         and #53 fires. Fixed in the shared reader (`w128b.CITED`); C6 checks it in both
         directions and requires it to DISAGREE with the naive `word in cell` test.

WHAT C1 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST SEVEN DEFECTS LIVED. C1 has a
finite VALUE vocabulary; a cell can still assert worth in words this file does not know ("it is
the only row that has moved the board"). It reads only the price COLUMN, like the other four --
the `closed` column and the prose pool disclosures remain unparsed by anything but
`w117a_handcount`. And C1 now makes `NOT A PRICE` costly rather than impossible: a cell may still
opt out honestly, which is exactly what rows 8 and 10 did before anyone checked whether they
could. 🎯 If the next defect is in this genus, it is a value claim in a vocabulary I did not
enumerate, or it is in the closed column.

    .venv/bin/python experiments/w129b_optoutguard.py            # rc 0 = clean
    .venv/bin/python experiments/w129b_optoutguard.py --control  # exits 0, having fired
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
ROW10_JSON = os.path.join(HERE, "w129a_row10.json")

import w128b_pricedguard as g57  # noqa: E402  -- the shared reader and the shared table parser

claims, index_rows = g57.claims, g57.index_rows
OPT_OUT = g57.OPT_OUT
MAGNITUDE = g57.MAGNITUDE

ROW10 = 10
# C1. Words that assert worth. A cell may opt out of pricing; it may not opt out and then claim.
VALUE_WORDS = ("PAID", "paid", "pays", "worth", "gain", "value")
# ...unless it also states a magnitude, or the bare `0` row 8 states.
ZERO = re.compile(r"\bis\s+\*?\*?0\*?\*?\b|\banswer is \*\*0\*\*")

# C2. The six facets, plus the split that says the headline has not been collected.
ROW10_TOKENS = (
    "SELECTION",            # arm A's quantity
    "VERIFICATION",         # arm B's quantity
    "PREDICTOR",            # arm C's quantity
    "FINAL-FILE layer",     # arm A's layer
    "PIPELINE layer",       # arm B's layer
    "k=2",                  # arm A's scope
    "+4.5228e-6",           # arm A's magnitude
    "+3.0704e-6",           # ...and its upper-tau reading
    "+0.0000e-6",           # arm B's magnitude -- zero is the PASS
    "−27.4266e-6",          # arm C's magnitude, in PREDICTED-LB units
    "auto-selection",       # arm A's baseline
    "PREDICTED-LB units",   # arm C's units, so it cannot be read as AUC
    "NOT addable",          # the three currencies do not sum
    "CONTINGENT",           # the headline is unrealised
    "REALISED",             # ...and the realised total is stated
)

# C3. Frozen literals -- w129a_row10.json, written by w129a from artefacts on disk.
EXPECT = {
    "click_e6": 4.522768563707359,
    "click_tau_hi_e6": 3.070397390634753,
    "worst_misclick_e6": 81.92352750187274,
    "arm_b_e6": 0.0,
    "arm_c_e6": -27.426558533962574,
    "realised_e6": 0.0,
    "wanted_never_rebuilt": ["w23_ad187stdcorr"],
    "tol": 1e-9,
}

# C4. The cell exactly as it stood before this run's edit.
PREFIX_ROW10 = ("**NOT A PRICE — not a modelling angle** — it is the standing checklist, and "
                "it is the one angle that has ever PAID")

# C6. The citation probe, and the six words whose removal makes #53 fire.
CITATION_PROBE = "row 3's CatBoost ENROLMENT price of +10.04e-6/member"
OWN_PROBE = "an ENROLMENT price of +10.04e-6/member"


def unpriced_claim(rows):
    """C1 predicate: cells that opt out of pricing and then assert worth anyway."""
    bad = []
    for n, c in sorted(rows.items()):
        if OPT_OUT not in c:
            continue
        said = [w for w in VALUE_WORDS if claims(c, w)]
        if said and not MAGNITUDE.search(c) and not ZERO.search(c):
            bad.append((n, said, c[:110]))
    return bad


def missing_row10(cell):
    """C2 predicate: which of row 10's required tokens the cell does not publish."""
    return [t for t in ROW10_TOKENS if t not in cell]


def sibling_predicates():
    """The four siblings' own C1 predicates, imported rather than reimplemented, each paired
    with the frozen pre-fix cell it was built for so C5 can show it still fires somewhere."""
    import w124b_priceunitguard as g53
    import w126c_scopeguard as g55
    import w127b_rateguard as g56
    return [
        ("#53 w124b unpriced (quantity)", g53.unpriced, 4, g53.PREFIX_CELLS[4]),
        ("#55 w126c unscoped (scope)", g55.unscoped, 6, g55.PREFIX["row6_cell"]),
        ("#56 w127b undenominated (denominator)", g56.undenominated, 7, g56.PREFIX_ROW7),
        ("#57 w128b magnitudeless (silence)", g57.magnitudeless, 9, g57.PREFIX_ROW9),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1/C2 over the frozen PRE-FIX cell; both must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()
    rows = index_rows(txt)
    if not rows:
        print("w129b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w129b --control: C1/C2 against the frozen PRE-FIX row 10 cell\n")
        pre = dict(rows)
        pre[ROW10] = PREFIX_ROW10
        c1 = unpriced_claim(pre)
        c2 = missing_row10(PREFIX_ROW10)
        print(f"  C1 pre-fix row {ROW10} cell {PREFIX_ROW10[:70]!r}...")
        hit = [d for d in c1 if d[0] == ROW10]
        print(f"     {'FIRES' if hit else 'SILENT (BAD)'} -- value words claimed under the "
              f"opt-out: {hit[0][1] if hit else []}")
        print(f"  C2 {'FIRES' if c2 else 'SILENT (BAD)'} -- missing {len(c2)} of "
              f"{len(ROW10_TOKENS)} tokens")
        s1 = unpriced_claim(rows)
        s2 = missing_row10(rows.get(ROW10, ""))
        print(f"\n  shipped C1 {'SILENT' if not s1 else 'FIRES (BAD): ' + str(s1)}")
        print(f"  shipped C2 {'SILENT' if not s2 else 'FIRES (BAD): ' + str(s2)}")
        ok = bool(hit) and bool(c2) and not s1 and not s2
        print(f"\n  the control {'WORKS' if ok else 'DOES NOT WORK'}: pre-fix fires on both, "
              f"shipped is silent on both")
        return 0

    fails = []
    print("w129b -- #58: a cell that opts out of pricing must not then claim it PAID\n")

    # ---------------------------------------------------------------------------- C1
    opted = [n for n, c in sorted(rows.items()) if OPT_OUT in c]
    print(f"[C1] ANGLE INDEX price cells: {len(rows)} rows {sorted(rows)}; "
          f"opted out: {opted or 'none'}")
    if len(rows) < 10:
        fails.append(f"the ANGLE INDEX parsed to {len(rows)} rows; C1 cannot pass vacuously")
        print(f"     FAIL only {len(rows)} rows parsed")
    bad = unpriced_claim(rows)
    for n, said, c in bad:
        print(f"     FAIL row {n} opts out and then claims {said}: {c}")
        fails.append(f"ANGLE INDEX row {n}: `{OPT_OUT}` plus a value claim {said}, no magnitude")
    if not bad:
        print("     OK no cell opts out of pricing and asserts worth in the same breath")

    # ---------------------------------------------------------------------------- C2
    cell = rows.get(ROW10, "")
    miss = missing_row10(cell)
    print(f"\n[C2] row {ROW10} publishes three quantities, three layers, a k, three "
          f"denominators, three baselines and the realised/contingent split")
    if miss:
        print(f"     FAIL missing: {miss}")
        print(f"     cell: {cell[:220]}")
        fails.append(f"ANGLE INDEX row {ROW10} does not publish {miss}")
    else:
        print(f"     OK all {len(ROW10_TOKENS)} required tokens present")

    # ---------------------------------------------------------------------------- C3
    print(f"\n[C3] the artefact still says it -- w129a_row10.json vs literals frozen here")
    if not os.path.exists(ROW10_JSON):
        print("     INERT w129a_row10.json is not on disk; C3 asserts nothing this run")
    else:
        with open(ROW10_JSON) as fh:
            d = json.load(fh)
        checks = [
            ("click", d["arm_a"]["cost_no_click_e6"], EXPECT["click_e6"]),
            ("click tau_hi", d["arm_a"]["cost_no_click_tau_hi_e6"], EXPECT["click_tau_hi_e6"]),
            ("worst mis-click", d["arm_a"]["worst_misclick_e6"], EXPECT["worst_misclick_e6"]),
            ("arm B", d["arm_b"]["price_e6"], EXPECT["arm_b_e6"]),
            ("arm C", d["arm_c"]["coef_e6"], EXPECT["arm_c_e6"]),
            ("realised total", d["realised_total_e6"], EXPECT["realised_e6"]),
        ]
        for name, got, want in checks:
            ok = abs(got - want) <= EXPECT["tol"]
            print(f"     {'OK  ' if ok else 'FAIL'} {name:<16} {got:+.6f} vs {want:+.6f}")
            if not ok:
                fails.append(f"C3 {name}: artefact {got} != frozen {want}")
        got = d["arm_b"]["wanted_never_rebuilt"]
        ok = got == EXPECT["wanted_never_rebuilt"] or got == []
        print(f"     {'OK  ' if ok else 'FAIL'} WANTED never rebuilt: {got} "
              f"(frozen {EXPECT['wanted_never_rebuilt']}; [] once w129 closes it)")
        if not ok:
            fails.append(f"C3 coverage: {got} is neither the frozen gap nor closed")

    # ---------------------------------------------------------------------------- C5
    print(f"\n[C5] the siblings, imported, over a table whose row {ROW10} is the PRE-FIX cell")
    pre = dict(rows)
    pre[ROW10] = PREFIX_ROW10
    mine = [n for n, _, _ in unpriced_claim(pre)]
    print(f"     #58 (this file) on the same table: fires on rows {mine}")
    if ROW10 not in mine:
        fails.append("C5 is vacuous: #58 does not fire on the pre-fix row 10 cell")
    for label, pred, own_row, own_cell in sibling_predicates():
        hits = [n for n, *_ in pred(pre)]
        blind = ROW10 not in hits
        still = [n for n, *_ in pred({own_row: own_cell})]
        ok = blind and own_row in still
        print(f"     {'OK  ' if ok else 'FAIL'} {label:<38} row {ROW10}: "
              f"{'BLIND' if blind else 'FIRES'}   own row {own_row}: "
              f"{'fires' if own_row in still else 'SILENT -- importer broken'}")
        if not blind:
            fails.append(f"C5: {label} was not blind to the pre-fix row {ROW10} after all")
        if own_row not in still:
            fails.append(f"C5: {label} no longer fires on its own frozen defect")

    # ---------------------------------------------------------------------------- C6
    print("\n[C6] the CITATION reader, both directions, against the naive `word in cell` test")
    probes = [
        (CITATION_PROBE, "ENROLMENT", False, "a citation of ANOTHER row is not this row's claim"),
        (OWN_PROBE, "ENROLMENT", True, "the same word, unattributed, IS a claim"),
        ("not a SEARCH, a SELECTION price of +1e-6", "SEARCH", False, "the negation reader"),
        ("a SEARCH price of +1e-6", "SEARCH", True, "and it does not over-fire"),
    ]
    disagree = 0
    for cellp, word, want, why in probes:
        got, naive = claims(cellp, word), word in cellp
        ok = got == want
        disagree += got != naive
        print(f"     {'OK  ' if ok else 'FAIL'} claims={got!s:<5} naive={naive!s:<5} "
              f"want={want!s:<5} {why}")
        if not ok:
            fails.append(f"C6 probe {why!r}: claims()={got}, expected {want}")
    print(f"     the reader disagrees with the naive test on {disagree} of {len(probes)} probes")
    if disagree == 0:
        fails.append("C6: the reader agrees with `word in cell` everywhere -- it is decoration")

    print(f"\nFAILURES: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
