"""w131b -- STANDING CHECK #60. #59 MADE A ZERO NAME ITS BASELINE. IT NEVER ASKED THAT THE
BASELINE BE A CONTRAST -- AND THE FIRST CELL WRITTEN TO SATISFY #59 NAMED THE ARM ITSELF.
(w131, 2026-08-30)

THE DEFECT THIS EXISTS BECAUSE OF, AND IT IS ONE RUN OLD AND MINE. w130 registered #59 and,
in the same pass, hand-edited row 1's price cell to turn #59 green on it:

    a **CONCAT** price (extra training ROWS, not members). **0 measured against the same
    stack trained on `train.csv` alone (0× dose), and the usual Playground edge is INVERTED
    here: −58e-6 at 1× dose, −3,340e-6 at 50×; ...**

`0× dose` IS "the same stack trained on train.csv alone". `orig_concat.py` makes it literal:
its dose loop is `if w:`, so at w=0 no augmented frame is built at all, and w131a measured
that dose 0 appends exactly 0 rows and reproduces the training frame exactly. The headline
therefore compares a frame with ITSELF -- zero for any model, any metric, any seed, with ZERO
DEGREES OF FREEDOM, before anything is fitted. It is a CONSTRUCTION ZERO, not a measurement.

WHY THIS IS NOT #59 AGAIN. #59's `grounded()` says all three headline-zero rows -- 1, 8 and 9
-- NAME a baseline, and it is right about all three. The question #59 does not ask is whether
the named baseline DIFFERS from the arm. Rows 1 and 8 name IDENTITY baselines (`0× dose` /
`already exists`); row 9 names a CONTRAST (`the uncorrected stack`, `a size-matched
permuted-cell control`). C6 requires this file's reader to split those three the way #59
cannot, in both directions.

AND WHY POSITION, NOT PRESENCE, IS THE RULE. A construction zero is not a lie -- w130's row 8
carries one and is honest, because it opens `⚠ TWO PRICES, ... MEANS NOTHING WITHOUT ITS
BASELINE` and gets the reader past the `0` before the first real magnitude arrives. Row 1
carries the same kind of zero with the disclosure absent, so a reader scanning the price
column meets `0` and stops. The rule is therefore: the disclosure must stand BEFORE the
cell's first non-zero magnitude. C4 measures that the position is doing the work by moving
the disclosure clause behind that magnitude and requiring the guard to go red.

⟹ THE FOURTH PLACE A GREEN MEANS "I DID NOT LOOK", after w130's three (an exemption branch, a
vocabulary, a pass message): A CELL EDITED BY HAND TO SATISFY A GUARD. The only reader it was
ever tested against is the one it was written for.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   IDENTITY-BASELINE HEADLINE ZEROS MUST DISCLOSE, AND DISCLOSE FIRST. For every price
         cell whose headline is a zero (#59's reader, imported) and whose baseline clause
         names a NULL SETTING of the row's own manoeuvre, a CONSTRUCTION-ZERO disclosure must
         appear before the cell's first non-zero e-6/e-7 magnitude. Reports the row count, the
         headline-zero rows and which of them are IDENTITY, so it cannot pass vacuously.
  C2 +-  FIRES BOTH WAYS, PER ROW, on an in-memory copy. The shipped table must score zero
         bad rows, and each IDENTITY row with its disclosure deleted ALONE must score exactly
         one. Counting the total rather than the delta is w114's vacuous control.
  C3 +-  THE REAL HISTORICAL DEFECT, frozen as a literal rather than read from HEAD (w115: a
         control anchored to HEAD stops being a control the moment the fix lands). PREFIX_ROW1
         is w130's cell verbatim -- the one that passes #59 -- and it must FIRE here.
  C4 +-  POSITION IS MEASURED, NOT ASSERTED. The shipped row 1 cell with its disclosure clause
         MOVED to after the first non-zero magnitude must go red, while the same clause in
         place is green. If presence alone passed, C1 would be a keyword check wearing a hat.
  C5 +-  THE SIBLINGS' BLINDNESS IS MEASURED, NOT ASSERTED. #53/#55/#56/#57/#58/#59 are
         imported and their own C1 predicates run over a table whose row 1 is PREFIX_ROW1. All
         six must return ZERO findings for row 1 while this file's C1 fires; and each must
         still fire on ITS OWN frozen defect in the same call, else the demonstration is
         vacuous. ⚠ #59 is in that list on purpose: it is the guard the cell was written for.
  C6 +-  THE IDENTITY READER DISCRIMINATES. On the three headline-zero rows #59's `grounded`
         must say NAMED for all three, while this file's reader must say IDENTITY for rows 1
         and 8 and CONTRAST for row 9. If the two readers agree everywhere, C1 is #59 again.

WHAT C1 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST NINE DEFECTS LIVED. Its
NULL-SETTING vocabulary is finite and hand-written: a baseline can be an identity in a
construction I did not enumerate (`with the member removed and nothing put back`,
`the same file`) and pass, or wear one of my tokens innocently and fire falsely. It inherits
#59's headline reader, so a construction zero introduced LATE in a cell is out of scope by
construction. It cannot tell a true disclosure from a decoy carrying the word `CONSTRUCTION`.
And like all six siblings it reads only the price COLUMN. 🎯 Seven guards now share one
column. Every one of them checks the SHAPE of a claim; not one checks that a stated baseline
is the one the number was actually measured against, and that is still the open genus.

    .venv/bin/python experiments/w131b_selfbaselineguard.py            # rc 0 = clean
    .venv/bin/python experiments/w131b_selfbaselineguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")

import w128b_pricedguard as g57       # noqa: E402 -- the shared table parser
import w130b_zerobaselineguard as g59  # noqa: E402 -- the headline-zero reader, imported

index_rows = g57.index_rows
headline_zeros = g59.headline_zeros
NONZERO = g59.NONZERO

ROW1 = 1

# A NULL SETTING of the row's own manoeuvre: the arm's dial turned to zero, or the arm
# already done. `absence` is deliberately NOT here -- "priced against its own absence" is the
# opposite, a real contrast, and putting it in would make row 8's ABSENCE clause fire.
NULL_SETTING = re.compile(
    r"\b0\s*[×x]\s*dose\b|\bzero\s+(?:dose|rows|copies|members)\b"
    r"|\bno\s+extra\b|\bnone\s+(?:added|appended)\b"
    r"|\balready\s+(?:exists?|built|ships?)\b"
    r"|\balone\s*\(|\bwithout\s+(?:it|them|the\s+member)\b",
    re.I)

# The disclosure that a headline zero is true by construction. Must stand BEFORE the cell's
# first non-zero magnitude, so a reader cannot reach the real numbers without passing it.
DISCLOSURE = re.compile(
    r"\bCONSTRUCTION[ -]ZERO\b|\btrue\s+by\s+construction\b"
    r"|\bNOT A PRICE\b|\bTWO PRICES\b|\bMEANS NOTHING WITHOUT\b|\bREPEAT\b",
    re.I)

# C3/C4/C5/C6. Row 1 exactly as w130 left it -- the cell that passes #59.
PREFIX_ROW1 = ("a **CONCAT** price (extra training ROWS, not members). **0 measured against "
               "the same stack trained on `train.csv` alone (0× dose), and the usual "
               "Playground edge is INVERTED here: −58e-6 at 1× dose, −3,340e-6 at 50×; the "
               "best separate-estimator route is −1e-6 to −2e-6 in the stack**")


def baseline_kind(cell):
    """IDENTITY if the cell names a baseline that is the arm at a null setting, else CONTRAST.

    Only meaningful on a cell #59 already calls grounded; a cell that names no baseline at all
    is #59's finding, not this one's.
    """
    return "IDENTITY" if NULL_SETTING.search(cell) else "CONTRAST"


def head(cell):
    """The cell up to its first non-zero magnitude -- what a reader of the price column meets
    before any real number arrives. A cell with no non-zero magnitude is all head."""
    m = NONZERO.search(cell)
    return cell[:m.start()] if m else cell


def undisclosed(rows):
    """C1 predicate: headline-zero cells whose baseline is the arm itself and that do not say
    so before their first non-zero magnitude."""
    bad = []
    for n, c in sorted(rows.items()):
        if not headline_zeros(c):
            continue
        if baseline_kind(c) != "IDENTITY":
            continue
        if not DISCLOSURE.search(head(c)):
            bad.append((n, c[:110]))
    return bad


def strip_disclosure(cell):
    """C2: remove the disclosure tokens from a cell's head, leaving everything else."""
    h = head(cell)
    return DISCLOSURE.sub("...", h) + cell[len(h):]


def demote_disclosure(cell):
    """C4: move the head's disclosure sentence to the END of the cell, unchanged. The tokens
    are all still present; only their position relative to the first magnitude changes."""
    h = head(cell)
    m = DISCLOSURE.search(h)
    if not m:
        return cell
    # the sentence the disclosure sits in, delimited by the surrounding sentence boundaries
    start = max(h.rfind(". ", 0, m.start()) + 2, 0)
    end = h.find(". ", m.end())
    end = end + 2 if end != -1 else len(h)
    clause = h[start:end]
    return cell[:start] + cell[end:] + " " + clause


def sibling_predicates():
    """The six siblings' own C1 predicates, imported rather than reimplemented, each paired
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
        ("#59 w130b ungrounded (baseline)", g59.ungrounded, 8, g59.PREFIX_ROW8),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1 over the frozen pre-fix row 1; it must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()
    rows = index_rows(txt)
    if not rows:
        print("w131b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w131b --control: C1 against the frozen pre-fix row 1 (w130's cell)\n")
        pre = dict(rows)
        pre[ROW1] = PREFIX_ROW1
        hits = {n for n, _ in undisclosed(pre)}
        print(f"  C1 pre-fix row 1: zeros {headline_zeros(PREFIX_ROW1)}  "
              f"#59 says {'NAMED' if g59.grounded(PREFIX_ROW1) else 'ABSENT'}  "
              f"baseline {baseline_kind(PREFIX_ROW1)}  "
              f"disclosure {'PRESENT' if DISCLOSURE.search(head(PREFIX_ROW1)) else 'ABSENT'}  "
              f"-> {'FIRES' if ROW1 in hits else 'SILENT (BAD)'}")
        s = undisclosed(rows)
        print(f"  shipped C1 {'SILENT' if not s else 'FIRES (BAD): ' + str(s)}")
        works = ROW1 in hits and not s
        print(f"\n  the control {'WORKS' if works else 'DOES NOT WORK'}: the pre-fix cell "
              f"fires, the shipped table is silent")
        return 0

    fails = []
    print("w131b -- #60: a zero measured against the arm itself must say so, and say so first\n")

    # ------------------------------------------------------------------------------ C1
    hz = [n for n, c in sorted(rows.items()) if headline_zeros(c)]
    ident = [n for n in hz if baseline_kind(rows[n]) == "IDENTITY"]
    bad = undisclosed(rows)
    print("C1 an identity-baseline headline zero must disclose before its first magnitude")
    print(f"   {len(rows)} price cells parsed, {len(hz)} carry a headline zero: {hz}; "
          f"{len(ident)} of those name an IDENTITY baseline: {ident}")
    for n in hz:
        c = rows[n]
        print(f"     row {n:2d}  zeros {headline_zeros(c)}  baseline {baseline_kind(c)}  "
              f"disclosure-before-first-magnitude "
              f"{'PRESENT' if DISCLOSURE.search(head(c)) else 'ABSENT'}")
    if not hz:
        fails.append("C1 VACUOUS: no price cell carries a headline zero, so C1 checked nothing")
        print("   FAIL vacuous -- no headline zeros to check")
    if not ident:
        fails.append("C1 VACUOUS: no headline zero names an IDENTITY baseline; the reader is "
                     "not discriminating and C1 cannot fire on anything")
        print("   FAIL vacuous -- no identity baselines to check")
    for n, snippet in bad:
        fails.append(f"C1 row {n}: a headline zero measured against the arm itself, with no "
                     f"construction-zero disclosure before the first magnitude -- {snippet}")
        print(f"   FAIL row {n}: {snippet}")
    if hz and ident and not bad:
        print("   OK  every identity-baseline headline zero discloses, and discloses first")

    # ------------------------------------------------------------------------------ C2
    print("\nC2 fires both ways, per row (in-memory perturbation)")
    base_n = len(undisclosed(rows))
    print(f"   shipped table -> {base_n} bad row(s)")
    if base_n:
        fails.append("C2 cannot run: the shipped table is already bad, so a per-row delta is "
                     "not measurable")
    else:
        for n in ident:
            probe = dict(rows)
            probe[n] = strip_disclosure(rows[n])
            got = len(undisclosed(probe))
            mark = "OK " if got == 1 else "-> "
            print(f"   {mark}row {n:2d} disclosure deleted alone -> {got} bad row(s)")
            if got != 1:
                fails.append(f"C2 row {n}: deleting its disclosure gives {got} bad rows, not 1")

    # ------------------------------------------------------------------------------ C3
    print("\nC3 the frozen historical defect (w130's row 1, the cell that passes #59)")
    pre = dict(rows)
    pre[ROW1] = PREFIX_ROW1
    fired = ROW1 in {n for n, _ in undisclosed(pre)}
    print(f"   {'OK ' if fired else '-> '}pre-fix row 1 -> {'FIRES' if fired else 'SILENT'}")
    if not fired:
        fails.append("C3 INERT: the frozen pre-fix row 1 does not fire, so this guard would "
                     "not have caught the defect it was written for")

    # ------------------------------------------------------------------------------ C4
    print("\nC4 position is measured, not asserted")
    if ROW1 in ident and not bad:
        moved = dict(rows)
        moved[ROW1] = demote_disclosure(rows[ROW1])
        changed = moved[ROW1] != rows[ROW1]
        tokens_kept = bool(DISCLOSURE.search(moved[ROW1]))
        red = ROW1 in {n for n, _ in undisclosed(moved)}
        print(f"   the disclosure clause moved behind the first magnitude, tokens still "
              f"present: {tokens_kept}, cell changed: {changed}")
        print(f"   {'OK ' if red else '-> '}-> {'RED' if red else 'GREEN'} "
              f"(green would mean C1 is a keyword check, not a position check)")
        if not (changed and tokens_kept and red):
            fails.append("C4: moving the disclosure behind the first magnitude does not turn "
                         "C1 red, so position is not doing the work")
    else:
        print("   -> row 1 is not a clean identity+disclosed cell; C4 has nothing to move")
        fails.append("C4 cannot run: row 1 is not an identity-baseline cell that discloses")

    # ------------------------------------------------------------------------------ C5
    print("\nC5 the six siblings, run BLIND on the pre-fix row 1 cell")
    mine = ROW1 in {n for n, _ in undisclosed(pre)}
    for label, pred, own_row, own_cell in sibling_predicates():
        here = [f for f in pred(pre) if f[0] == ROW1]
        own = dict(rows)
        own[own_row] = own_cell
        still = [f for f in pred(own) if f[0] == own_row]
        ok = not here and still
        print(f"   {'OK ' if ok else '-> '}{label:<40s} on row 1: "
              f"{len(here)} finding(s); on its own frozen defect (row {own_row}): "
              f"{len(still)} finding(s)")
        if here:
            fails.append(f"C5 {label} also fires on row 1, so #60 is not a new check")
        if not still:
            fails.append(f"C5 {label} no longer fires on its own frozen defect; the blindness "
                         f"demonstration is vacuous")
    print(f"   {'OK ' if mine else '-> '}and #60 itself fires on row 1")
    if not mine:
        fails.append("C5 #60 does not fire on the cell it was written for")

    # ------------------------------------------------------------------------------ C6
    print("\nC6 the identity reader must split what #59 cannot")
    agree = True
    for n in hz:
        c = rows[n]
        g = "NAMED" if g59.grounded(c) else "ABSENT"
        k = baseline_kind(c)
        print(f"   row {n:2d}  #59 {g:<6s}  #60 {k}")
        if k == "CONTRAST":
            agree = False
    ok6 = all(g59.grounded(rows[n]) for n in hz) and not agree and bool(ident)
    print(f"   {'OK ' if ok6 else '-> '}#59 calls every headline zero grounded; #60 splits "
          f"them {len(ident)} IDENTITY / {len(hz) - len(ident)} CONTRAST")
    if not ok6:
        fails.append("C6: the two readers do not disagree anywhere, so #60's grammar is #59's "
                     "wearing a hat")

    print(f"\nFAILURES: {len(fails)}")
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
