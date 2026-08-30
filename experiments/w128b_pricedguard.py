"""w128b -- STANDING CHECK #57. A PRICE CELL WITH NO MAGNITUDE IS INVISIBLE TO #53, #55 AND #56.
(w128, 2026-08-30)

THE DEFECT THIS EXISTS BECAUSE OF. ANGLE INDEX row 9 -- error analysis, tied with row 5 as the
most-handed angle of the competition -- had a price cell that read, in full:

    **0 / negative**

Nine words shorter than any other row. It names no QUANTITY (rows 1-7 name CONCAT / TUNING /
ENROLMENT / SEARCH), no MAGNITUDE, no UNIT, no LAYER, no SCOPE, no DENOMINATOR -- and no
BASELINE, which is the facet this run adds to the list. Row 9's own second instrument
(`w14d_bandmap`, cross-fitted per-cell isotonic) published TWO signs for one manoeuvre:
-118e-6 against the uncorrected stack and +6e-6 against a size-matched permuted-cell control.
"negative" is true of the first reading and FALSE of the second, and the cell never said which.

🎯 AND THE STRUCTURAL POINT, WHICH IS WHY THIS IS A GUARD AND NOT A CORRECTION. #53
`w124b_priceunitguard` (quantity), #55 `w126c_scopeguard` (scope) and #56 `w127b_rateguard`
(denominator) all open with the same predicate: select cells carrying an `e-6`/`e-7` token.
Row 9 carries none, so ALL THREE SKIP IT IN SILENCE. The most under-specified cell in the table
is the one cell the entire guard family is structurally unable to see. C5 does not assert that
-- it imports the three siblings and runs their own predicates over the frozen pre-fix cell.

⚠ THE PREDICTION THAT WAS WRONG, AND WHY IT WAS WRONG IN AN INSTRUCTIVE WAY. w127 wrote down
what #56 was blind to and pointed the next run at rows 1 and 2. Both carry magnitudes. w127 was
searching the cells its own instrument could read, which is exactly the failure the guard family
keeps having. A CHECK THAT TRIGGERS ON A NUMBER CANNOT SEE A CELL THAT DECLINES TO GIVE ONE, and
neither can the person who wrote it.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   MAGNITUDE, NOT VACUOUS. Every ANGLE INDEX price cell must carry an e-6/e-7 magnitude,
         or opt out explicitly with the literal marker `NOT A PRICE`. The opt-out is a token a
         human has to type, not an inferred exemption, and it is the whole point: silence must
         not be spendable. Reports the row count so it cannot pass vacuously.
  C2 +   ROW 9 PUBLISHES ITS PRICE, ITS BASELINE AND ITS LAYER. Required tokens below.
  C3 +-  THE ARTEFACT STILL SAYS IT -- `w128a_row9.json`, with the expected values frozen HERE
         as literals so this is a COMPARISON, not a recomputation that agrees with itself.
         INERT, not green, if the json is missing (#55's C4 rule).
  C4 +-  --control runs C1 and C2 over the FROZEN PRE-FIX row 9 cell, inlined as a literal
         rather than read from HEAD (w115: a control anchored to HEAD stops being a control the
         moment the fix lands). Pre-fix must FIRE on both; shipped must be silent on both.
  C6 +-  THE NEGATION READER DISCRIMINATES. `claims()` is shared by all four guards now, so it
         is checked here in both directions on synthetic cells: a cell that only DENIES a
         quantity must not be read as claiming it, a cell that claims one must be, and the
         old `word in cell` test must disagree on the first -- else the fix changed nothing.
  C5 +-  THE SIBLINGS' BLINDNESS IS MEASURED, NOT ASSERTED. #53/#55/#56 are imported and their
         own C1 predicates are run over a table whose row 9 is the pre-fix cell. Each must
         return ZERO findings for row 9 while this file's C1 fires -- and each must still fire
         on ITS OWN frozen pre-fix cell in the same call, else the demonstration is vacuous
         (three broken importers would also return zero).

WHAT C1 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST SIX DEFECTS LIVED. C1 asks for
a magnitude TOKEN and for the opt-out marker; it cannot tell whether the magnitude is the right
one, and it accepts `NOT A PRICE` from anyone who types it. It also reads only the ANGLE INDEX's
price COLUMN -- the `closed` column carries handing counts and dates that nothing here checks,
and the fourth column's grep anchors are prose. Rows 3, 4 and 5 disclose their pool (`base104`)
in prose that no guard parses, and w127 nearly shipped a wrong one. If the next defect is in
this genus, look at the closed column or at a pool disclosure.

    .venv/bin/python experiments/w128b_pricedguard.py            # rc 0 = clean
    .venv/bin/python experiments/w128b_pricedguard.py --control  # exits 0, having fired
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
ROW9_JSON = os.path.join(HERE, "w128a_row9.json")
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"

# C1. The same magnitude predicate as #53/#55/#56, deliberately, so the four guards agree on
# what "carries a number" means -- and so the hole this one fills is exactly their hole.
MAGNITUDE = re.compile(r"\d\s*e-[67]|\d,\d{3}e-[67]")
OPT_OUT = "NOT A PRICE"

# C2. Row 9 carries one manoeuvre measured against TWO baselines; both must be legible from
# the cell alone, with the layer and the scope that make them readable.
ROW9 = 9
ROW9_TOKENS = (
    "CORRECTION",             # the quantity: not ENROLMENT, not SEARCH, not TUNING
    "STACK layer",            # the layer
    "k=1",                    # the scope
    "−118e-6",           # baseline 1: against the uncorrected stack
    "+6e-6",                  # baseline 2: against the size-matched permuted-cell control
    "permuted-cell control",  # ...named, so "negative" cannot be read without it
    "76.7%",                  # the structural cap: cross-cell pairs no feature can reach
)

# C3. Frozen literals -- w128a_row9.json. Sources: w110's I1/I2 as recomputed by w128a's R1.
EXPECT = {
    "global": 0.970048,
    "within": 0.006964,
    "cross": 0.022987,
    "share": 0.767,
    "hard_band_n": 250_188,
    "adjacent_n": 274_034,
    "auc_tol": 2e-5,
    "share_tol": 2e-3,
    "control_tol": 0.05e-6,
}

# C4. The cell exactly as it stood before this run's edit.
PREFIX_ROW9 = "**0 / negative**"


# 🔴 THE READER DEFECT #57 FOUND IN ITS OWN SIBLINGS, w128. Row 9's corrected cell says
# "not an ENROLMENT, not a SEARCH, not a TUNING" -- it NAMES the three quantities in order to
# DENY them. #53/#55/#56 all test `WORD in cell`, so all three read the denial as the claim.
# It cut both ways in one cell: #55 went RED demanding a scope for a SEARCH the cell disowns,
# and #53 went GREEN because it saw quantity words and stopped looking. A word-presence test
# cannot tell a label from its negation, and the fix belongs in the reader, not in the cell
# (w127 §11: without frozen controls, "reword the cell" and "fix the reader" look identical
# from the outside and the cheap one wins). Shared from here so the four guards cannot drift.
NEGATED = re.compile(r"\bnot\s+an?\s+$", re.I)


def claims(cell, word):
    """True if `word` appears in `cell` at least once as a CLAIM rather than a denial."""
    for m in re.finditer(re.escape(word), cell):
        if not NEGATED.search(cell[max(0, m.start() - 12):m.start()]):
            return True
    return False


def index_rows(txt):
    """{row number -> price cell} from the ANGLE INDEX table, anchored on the block head so a
    `| 9 |` line elsewhere in the document cannot be picked up (w110's first-occurrence trap)."""
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


def magnitudeless(rows):
    """C1 predicate: price cells that carry no magnitude and do not opt out in words."""
    return [(n, c[:110]) for n, c in sorted(rows.items())
            if not MAGNITUDE.search(c) and OPT_OUT not in c]


def missing_row9(cell):
    """C2 predicate: which of row 9's required tokens the cell does not publish."""
    return [t for t in ROW9_TOKENS if t not in cell]


def sibling_predicates():
    """#53, #55 and #56's own C1 predicates, imported rather than reimplemented, each paired
    with the frozen pre-fix cell it was built for so C5 can show it still fires somewhere."""
    import w124b_priceunitguard as g53
    import w126c_scopeguard as g55
    import w127b_rateguard as g56
    return [
        ("#53 w124b unpriced (quantity)", g53.unpriced, 4, g53.PREFIX_CELLS[4]),
        ("#55 w126c unscoped (scope)", g55.unscoped, 6, g55.PREFIX["row6_cell"]),
        ("#56 w127b undenominated (denominator)", g56.undenominated, 7, g56.PREFIX_ROW7),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1/C2 over the frozen PRE-FIX cell; both must fire")
    a = ap.parse_args()

    txt = open(RESEARCH).read()
    rows = index_rows(txt)
    if not rows:
        print("w128b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w128b --control: C1/C2 against the frozen PRE-FIX row 9 cell\n")
        pre = dict(rows)
        pre[ROW9] = PREFIX_ROW9
        c1 = magnitudeless(pre)
        c2 = missing_row9(PREFIX_ROW9)
        print(f"  C1 pre-fix row {ROW9} cell {PREFIX_ROW9!r}")
        print(f"     {'FIRES' if any(n == ROW9 for n, _ in c1) else 'SILENT (BAD)'} "
              f"-- magnitude-free rows: {[n for n, _ in c1]}")
        print(f"  C2 {'FIRES' if c2 else 'SILENT (BAD)'} -- missing {len(c2)} token(s): {c2}")
        s1 = magnitudeless(rows)
        s2 = missing_row9(rows.get(ROW9, ""))
        print(f"\n  shipped C1 {'SILENT' if not s1 else 'FIRES (BAD): ' + str(s1)}")
        print(f"  shipped C2 {'SILENT' if not s2 else 'FIRES (BAD): ' + str(s2)}")
        ok = (any(n == ROW9 for n, _ in c1) and c2 and not s1 and not s2)
        print(f"\n  the control {'WORKS' if ok else 'DOES NOT WORK'}: pre-fix fires on both, "
              f"shipped is silent on both")
        return 0

    fails = []
    print("w128b -- #57: a price cell with no MAGNITUDE is invisible to #53, #55 and #56\n")

    # ---------------------------------------------------------------------------- C1
    print(f"[C1] ANGLE INDEX price cells: {len(rows)} rows {sorted(rows)}")
    if len(rows) < 10:
        fails.append(f"the ANGLE INDEX parsed to {len(rows)} rows; C1 cannot pass vacuously")
        print(f"     FAIL only {len(rows)} rows parsed")
    bad = magnitudeless(rows)
    for n, c in bad:
        print(f"     FAIL row {n} states no magnitude and does not opt out: {c}")
        fails.append(f"ANGLE INDEX row {n}: a price cell with no e-6/e-7 and no `{OPT_OUT}`")
    if not bad:
        opted = [n for n, c in sorted(rows.items()) if OPT_OUT in c]
        print(f"     OK every price cell carries a magnitude or says `{OPT_OUT}` "
              f"(opted out: rows {opted})")

    # ---------------------------------------------------------------------------- C2
    cell = rows.get(ROW9, "")
    miss = missing_row9(cell)
    print(f"\n[C2] row {ROW9} publishes its price, both baselines, its layer and its scope")
    if miss:
        print(f"     FAIL missing: {miss}")
        print(f"     cell: {cell[:220]}")
        fails.append(f"ANGLE INDEX row {ROW9} does not publish {miss}")
    else:
        print("     OK the cell names CORRECTION, both baselines, the layer, k=1 and the "
              "76.7% cross-cell cap")

    # ---------------------------------------------------------------------------- C3
    print("\n[C3] the artefact still says it -- w128a_row9.json")
    if not os.path.exists(ROW9_JSON):
        print(f"     INERT {os.path.basename(ROW9_JSON)} is not on disk. This check is not "
              f"green, it is not running. Re-run experiments/w128a_row9.py.")
    else:
        j = json.load(open(ROW9_JSON))
        r1 = j.get("r1", {})
        for key, want, tol in (("global", EXPECT["global"], EXPECT["auc_tol"]),
                               ("within", EXPECT["within"], EXPECT["auc_tol"]),
                               ("cross", EXPECT["cross"], EXPECT["auc_tol"]),
                               ("share", EXPECT["share"], EXPECT["share_tol"])):
            got = r1.get(key)
            if got is None:
                fails.append(f"w128a_row9.json has no r1.{key}")
                print(f"     FAIL r1.{key} absent")
                continue
            ok = abs(got - want) <= tol
            print(f"     {'OK  ' if ok else 'FAIL'} r1.{key:<7s} {got:.6f}  (frozen "
                  f"{want:.6f})")
            if not ok:
                fails.append(f"w128a's r1.{key} is {got:.6f}, #57 froze {want:.6f}")
        for key, want in (("hard_band_n", EXPECT["hard_band_n"]),
                          ("adjacent_n", EXPECT["adjacent_n"])):
            got = r1.get(key)
            ok = got == want
            print(f"     {'OK  ' if ok else 'FAIL'} r1.{key:<11s} {got:,}  (frozen {want:,})"
                  if got is not None else f"     FAIL r1.{key} absent")
            if not ok:
                fails.append(f"w128a's r1.{key} is {got}, #57 froze {want}")
        ctl = j.get("control")
        if ctl is None:
            print("     note the paired arms were not run (--quick); the CatBoost control "
                  "gap is not checked")
        else:
            ok = abs(ctl["gap"]) <= EXPECT["control_tol"]
            print(f"     {'OK  ' if ok else 'FAIL'} the in-process CatBoost control gap "
                  f"{ctl['gap']*1e6:+.4f}e-6 (tol {EXPECT['control_tol']*1e6:.2f}e-6)")
            if not ok:
                fails.append(f"w128a's CatBoost control gap is {ctl['gap']*1e6:+.4f}e-6")
            if j.get("reopen"):
                fails.append("w128a reports row 9 RE-OPENS; the index still says closed")
                print("     FAIL w128a set reopen=true")

    # ---------------------------------------------------------------------------- C5
    print("\n[C5] the siblings' blindness, measured: their own predicates over the pre-fix cell")
    pre = dict(rows)
    pre[ROW9] = PREFIX_ROW9
    mine = [n for n, _ in magnitudeless(pre)]
    print(f"     #57 (this file) on the pre-fix table: fires on rows {mine}")
    if ROW9 not in mine:
        fails.append("#57 does not fire on its own pre-fix cell; C5 proves nothing")
        print("     FAIL #57 is silent on the cell it exists for")
    try:
        sibs = sibling_predicates()
    except Exception as exc:                                  # noqa: BLE001
        fails.append(f"C5 could not import the sibling guards: {exc}")
        print(f"     FAIL import: {exc}")
        sibs = []
    for label, pred, own_row, own_prefix in sibs:
        hits = [n for n, *_ in pred(pre)]
        blind = ROW9 not in hits
        # and the same predicate must still fire on ITS OWN pre-fix cell, else it is broken
        selfcheck = dict(rows)
        selfcheck[own_row] = own_prefix
        fires_own = own_row in [n for n, *_ in pred(selfcheck)]
        print(f"     {'OK  ' if blind and fires_own else 'FAIL'} {label:<38s} "
              f"row {ROW9}: {'BLIND' if blind else 'sees it'}   "
              f"own row {own_row}: {'fires' if fires_own else 'SILENT'}")
        if not blind:
            fails.append(f"{label} does see row {ROW9}; #57 is redundant and should be merged")
        if not fires_own:
            fails.append(f"{label} no longer fires on its own frozen pre-fix cell; C5's "
                         f"blindness reading is not trustworthy")

    # ---------------------------------------------------------------------------- C6
    print("\n[C6] the negation reader, both directions, on synthetic cells")
    probes = [
        ("a **SEARCH** price — **−1.07e-6**", "SEARCH", True),
        ("not a SEARCH, not an ENROLMENT — **+3.95e-6**", "SEARCH", False),
        ("not an ENROLMENT here, but an ENROLMENT there — **+7e-6**", "ENROLMENT", True),
        ("NOT A SEARCH at all — **+1e-6**", "SEARCH", False),
    ]
    for cell, word, want in probes:
        got = claims(cell, word)
        naive = word in cell
        mark = "OK  " if got == want else "FAIL"
        print(f"     {mark} claims({word}) = {got!s:<5s} (want {want!s:<5s})  naive "
              f"`in` = {naive!s:<5s}   {cell[:52]}")
        if got != want:
            fails.append(f"the negation reader returns {got} for {word} in {cell!r}")
    disagree = sum(1 for cell, word, _ in probes if claims(cell, word) != (word in cell))
    print(f"     {'OK  ' if disagree else 'FAIL'} the reader disagrees with the naive `in` "
          f"test on {disagree} of {len(probes)} probes")
    if not disagree:
        fails.append("the negation reader never disagrees with `word in cell`; it is "
                     "decoration and the three siblings gained nothing")

    print(f"\nFAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
