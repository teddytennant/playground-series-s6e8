"""w124b -- STANDING CHECK #53. A PRICE IS A NUMBER AND A QUANTITY. THE ANGLE INDEX'S PRICE
COLUMN PUBLISHED ONLY THE NUMBER. (w124, 2026-08-30)

WHAT WENT WRONG. The ANGLE INDEX has one column headed `price`. Six of its ten rows carry a
magnitude in that column, and they are not magnitudes of the same thing:

    row 2  +4e-7                     value of TUNING a GBDT the pack already holds
    row 4  +4e-7                     same, inherited from row 2
    row 3  5.9e-6 / 10.04e-6         value of ENROLLING one more member
    row 6  -1.07e-6                  value of a top-level weight SEARCH
    row 1  -58e-6 ... -3,340e-6      value of CONCATENATING extra training rows, by dose
    row 7  structural null / +2e-6   both -- and row 7 is the only row that says which

Set side by side under one heading a reader gets `5.9e-6` against `+4e-7` and concludes
CatBoost is ~15x XGBoost. Neither measurement supports that. They are different quantities:
one is what you get for RE-FITTING a model you have, the other is what you get for ADDING a
model you do not. Row 7 already writes `(stacker)` and `(member)` next to its two numbers, so
the fix is the index's own existing habit applied to the other five rows.

🎯 SAME DEFECT GENUS, FIFTH PLACE. w120 §4 `priority` naming an input, w121 §3 a prose cause
beside a derived count, w122 §2 `slot` printed for `tier`, w123 §3 a family label on a residual
group -- and now a column header that names a magnitude without its unit. Every one is a WORD
describing the input rather than the predicate. No numeric check reaches any of them: `+4e-7`
and `5.9e-6` are both correct.

WHAT THIS GUARD ENFORCES.

  C1  UNITS. Every ANGLE INDEX price cell carrying an `e-6`/`e-7` magnitude must name its
      quantity from a registered four-word vocabulary: TUNING, ENROLMENT, CONCAT, SEARCH.
      Rows whose price is a bare `0` / `negative` / `not a modelling angle` are exempt -- there
      is no magnitude to misread. This is a check on a WORD, which is the class of defect the
      last five runs kept finding.
  C2  ROW 4's ENROLMENT PRICE. Row 4 published a TUNING price and nothing else, so the index
      had no XGBoost enrolment number at all while row 3 had two CatBoost ones. w124a measured
      it on the same base104 pool, the same paired 50/50 procedure and the same three splits,
      with CatBoost re-measured in-process as a reproduction control. The row must name it.
  C3  THE DUPLICATE DENOMINATOR. `bolt_xgb_d7_alt1` == `bolt_xgb_d7_alt2` byte-identical
      (w109a_dupscan relation 1) and BOTH are inside the XGB subgroup, so any per-member XGB
      price divides by 11 distinct arrays, not 12 names. The disclosure must be in RESEARCH.md
      and the identity must still hold on the arrays.
  C4  THE FAMILY TABLE IS DATED. "Best single models by family" was measured once and never
      re-derived; two of its three GBDT rows are stale, and row 4's "XGBoost is not a missing
      leg" rests on it. The table must carry the re-derivation date and the margin.
  C5  --control: the C1 predicate over the FROZEN pre-fix cells, which must fire on all six
      magnitude-carrying rows. If it does not, this guard reports INERT rather than green -- a
      fix with no alternative behaviour to select from is not a fix (w121 §5, w122 §5).

⚠ JOURNAL.md is append-only and NOT in scope. Only RESEARCH.md, the live document, is checked.

    .venv/bin/python experiments/w124b_priceunitguard.py            # rc 0 = clean
    .venv/bin/python experiments/w124b_priceunitguard.py --control  # exits 0, having fired

Deterministic: reads one document and two .npy arrays. No fit, no API call.
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

from common import DATA          # noqa: E402

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
EXT2 = os.path.join(DATA, "ext_members2")
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"

# C1. The registered vocabulary. Four words, deliberately few: a price here is the value of
# tuning a model you hold, enrolling one you do not, concatenating rows, or searching weights.
UNITS = ("TUNING", "ENROLMENT", "CONCAT", "SEARCH")
MAGNITUDE = re.compile(r"\d\s*e-[67]|\d,\d{3}e-[67]")

# C2/C3. Frozen literals -- w124a_row4.json, base104, paired 50/50, 3 splits, C=1.0, hybrid.
# Written here by hand from the measurement so this file is a COMPARISON, not a recomputation
# that agrees with itself.
ROW4_TOKENS = ("ENROLMENT", "TUNING")
DUP_PAIR = ("bolt_xgb_d7_alt1", "bolt_xgb_d7_alt2")
DUP_DISCLOSURE = "11 distinct"        # must appear near the XGB enrolment price
FAMILY_TABLE_HEAD = "### Best single models by family"
FAMILY_TOKENS = ("re-derived", "w124")

# C5. The six magnitude-carrying price cells exactly as they stood before this run's edit,
# captured from RESEARCH.md at 2026-08-30 13:36Z into w124_prefix_cells.json and inlined here
# so the control does not depend on a file that a later run may overwrite.
PREFIX_CELLS = {
    1: "**0, and the usual Playground edge is INVERTED here: −58e-6 at 1× dose, −3,340e-6 at "
       "50×; the best separate-estimator route is −1e-6 to −2e-6 in the stack**",
    2: "**+4e-7**, and it holds on its own arrays: `lgbm_tuned_lat_frac` − "
       "`lgbm_fixed_lat_frac` re-measures at **+0.000031** against the published +3e-5",
    3: "⚠ **TWO PRICES, AND THE ROW USED TO PUBLISH ONLY THE LOWER ONE.** **5.9e-6/member** "
       "is the `rest`-group average, and `rest` is a **RESIDUAL** (8/35 CatBoost, 4 neural)",
    4: "**+4e-7**",
    6: "**−1.07e-6**",
    7: "structural null (stacker) · +2e-6 (member)",
}


def index_rows(txt):
    """{row number -> price cell} from the ANGLE INDEX table. Anchored on the block head, so
    a `| 3 |` line elsewhere in the document cannot be picked up (w110's first-occurrence trap
    in the other direction)."""
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


def unpriced(rows):
    """Price cells that carry a magnitude and name no quantity. The C1 predicate."""
    return [(n, c) for n, c in sorted(rows.items())
            if MAGNITUDE.search(c) and not any(u in c for u in UNITS)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run the C1 predicate over the frozen PRE-FIX cells; it must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()
    rows = index_rows(txt)

    if a.control:
        print("w124b --control: the C1 predicate against the frozen PRE-FIX price cells\n")
        fired = unpriced(PREFIX_CELLS)
        for n, c in sorted(PREFIX_CELLS.items()):
            hit = any(x[0] == n for x in fired)
            print(f"  row {n:2d}  {'FIRES ' if hit else 'silent'}  {c[:78]}")
        shipped = unpriced(rows)
        print(f"\n  pre-fix cells: {len(fired)} magnitude(s) published with no quantity")
        print(f"  shipped cells: {len(shipped)} magnitude(s) published with no quantity")
        if not fired:
            print("\n  ** INERT ** the pre-fix predicate finds nothing, so the guard has no "
                  "alternative behaviour to select from and is not evidence of a fix.")
        elif len(fired) == len(shipped):
            print("\n  ** INERT ** both predicates agree; the fix changed nothing observable.")
        else:
            print(f"\n  ✅ the pre-fix column left {len(fired)} magnitude(s) without a "
                  f"quantity and the shipped one leaves {len(shipped)}.")
        return 0

    fails, out = [], {}

    def fail(m):
        fails.append(m)
        print("  FAIL " + m)

    print("w124b -- a price is a number AND a quantity\n")

    if not rows:
        fail("the ANGLE INDEX table did not parse; every check below is vacuous")
        print("\nFAILURES: 1")
        return 1

    # C1 ------------------------------------------------------------------
    priced = [n for n, c in sorted(rows.items()) if MAGNITUDE.search(c)]
    bad = unpriced(rows)
    out["magnitude_rows"], out["unpriced"] = priced, [n for n, _ in bad]
    if bad:
        for n, c in bad:
            fail(f"C1 ANGLE INDEX row {n} publishes a magnitude with no quantity. A price is "
                 f"the value of {'/'.join(UNITS)}, and the reader cannot tell which from the "
                 f"number. >> {c[:80]}")
    else:
        print(f"  C1 OK all {len(priced)} magnitude-carrying price cell(s) name their "
              f"quantity (rows {priced})")

    # C2 ------------------------------------------------------------------
    cell4 = rows.get(4, "")
    miss = [t for t in ROW4_TOKENS if t not in cell4]
    if miss:
        fail(f"C2 ANGLE INDEX row 4 omits {miss}. It published a TUNING price and no "
             f"ENROLMENT price at all, while row 3 published two CatBoost enrolment prices "
             f"-- the exact side-by-side that invites the wrong ratio.")
    else:
        print(f"  C2 OK row 4 names both quantities {list(ROW4_TOKENS)}")
    out["row4_cell"] = cell4

    # C3 ------------------------------------------------------------------
    d1, d2 = DUP_PAIR
    try:
        o1 = np.load(os.path.join(EXT2, f"oof_{d1}.npy"))
        o2 = np.load(os.path.join(EXT2, f"oof_{d2}.npy"))
        same = bool(np.array_equal(o1, o2))
    except OSError as e:
        same = None
        fail(f"C3 could not read the duplicate pair off disk: {e}")
    if same is False:
        fail(f"C3 w109a_dupscan relation 1 says {d1} == {d2} byte-identical; they are not, so "
             f"the 11-distinct denominator is wrong")
    elif same:
        print(f"  C3 OK {d1} == {d2} still byte-identical on OOF")
    if DUP_DISCLOSURE not in txt:
        fail(f"C3 RESEARCH.md never says '{DUP_DISCLOSURE}': both halves of a byte-identical "
             f"pair are inside the 12-name XGB subgroup, so its per-member denominator is 11 "
             f"arrays. Publishing a per-member price over 12 names overstates the divisor.")
    else:
        print(f"  C3 OK the '{DUP_DISCLOSURE}' denominator is disclosed")
    out["dup_identical"] = same

    # C4 ------------------------------------------------------------------
    k = txt.find(FAMILY_TABLE_HEAD)
    if k < 0:
        fail(f"C4 '{FAMILY_TABLE_HEAD}' not found; row 4's 'not a missing leg' argument rests "
             f"on that table")
    else:
        span = txt[k:k + 2500]
        miss = [t for t in FAMILY_TOKENS if t not in span]
        if miss:
            fail(f"C4 the family table omits {miss}. It was measured once and never "
                 f"re-derived; two of its three GBDT rows are stale on today's arrays and the "
                 f"XGB-LGBM margin it implies is inside the noise floor.")
        else:
            print("  C4 OK the family table carries its re-derivation")
    out["family_table_found"] = k >= 0

    print(f"\nFAILURES: {len(fails)}")
    for m in fails:
        print("  - " + m)
    if not fails:
        print("✅ CLEAN — every published magnitude names the quantity it measures")
    out["failures"] = fails
    with open(os.path.join(HERE, "w124b_priceunitguard.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
