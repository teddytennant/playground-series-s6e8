"""w132b -- STANDING CHECK #61. `sign-consistent over 3 splits` IS NOT A SIGNIFICANCE TEST,
AND IT IS THE ONLY EVIDENCE STANDARD THE PRICE COLUMN HAS EVER USED. (w132, 2026-08-31)

WHAT THE SEVEN SIBLINGS COVER, AND WHY NONE OF THEM REACHES THIS. #53 gave the ANGLE INDEX's
price column its QUANTITIES, #54 row 5 its LAYER, #55 row 6 its SCOPE, #56 row 7 its
DENOMINATOR, #57 row 9 a MAGNITUDE at all, #58 row 10 the price of an OPT-OUT, #59 a headline
zero's BASELINE, #60 that the baseline be a CONTRAST. Every one of them checks the shape of
the POINT ESTIMATE. Not one asks what the number is worth against its own noise, and w131
wrote the hole down in terms:

    "Seven guards now share one column. Every one of them checks the SHAPE of a claim; not
     one checks that a stated baseline is the one the number was actually measured against."

THE DEFECT THIS EXISTS BECAUSE OF, AND IT DECIDES EVERY CLOSURE IN THE TABLE. Ten runs of
prices have been closed on a bracketed word: `[consistent]` / `consistent 3/3` when a number
is to be believed, `SIGN FLIPS` / `SIGN-FLIPPING` when it is to be dismissed. Sign-consistency
over 3 paired splits, under a symmetric null, occurs with probability 2*(1/2)^3 = 0.25. It is
a 25% false-positive rate wearing the costume of a test.

🎯 AND THE TABLE CARRIES ITS OWN COUNTEREXAMPLE, LIVE. Row 9's PERMUTED NULL -- a size-matched
permuted-cell control, a null BY CONSTRUCTION, the arm whose whole job is to be nothing --
is labelled `consistent` in w128a_row9.json. Its t is 1.66. The label passes the null it was
built to reject. Two more `consistent` arms sit under |t| = 3.

WHAT IT DOES NOT DO, WHICH IS THE POINT. t is scope-invariant: it is the same number computed at
group scope from (delta, sd) or at member scope from (per_member, sd/n), exactly (w132a R3, max
gap 3.6e-15 over 11 arms). So publishing t moves NO published verdict. The CatBoost control is
|t| = 8.87 and every arm it certifies as null is under 2.3. This is a REPORTING finding. Nothing
re-opens, and the operational rule -- all these rates are under the 50e-6 floor, so do not build
-- is unchanged: on 3 splits, df = 2, the 5% two-tailed critical t is 4.303 and the 1% is 9.925.

🔴 CORRECTED BY w158 (#68), AND THE CORRECTION FALSIFIES A SENTENCE THAT STOOD HERE FOR 26 RUNS.
This file shipped t as `delta/sd`. That is an effect size, not a t: `sd` is the sample standard
deviation of the REPS = 3 paired split deltas, so the statistic is `delta/(sd/sqrt(REPS))` and
every t this workspace published was understated by sqrt(3) = 1.732. The critical values above
belong to the CORRECTED scale, so the comparison was between two different objects. It ran in
the CONSERVATIVE direction for discovery and the ANTI-conservative one for dismissal, and no
arm changes side at 5% -- the largest corrected t among the arms this table calls null is 2.92
against 4.303. But the line that used to end this paragraph -- `NO single-family enrolment rate
in this table clears 1%` -- is FALSE on the corrected scale: +xgb_only reads 10.61 and
+xgb_dedup 10.75 against 9.925, and +lgb_only misses by 0.001 at 9.924.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   NOT VACUOUS. Every ANGLE INDEX price cell carrying BOTH an e-6/e-7 magnitude and a
         SIGN VERDICT must also carry a SIGNIFICANCE token (a t, or the criterion's own
         false-positive rate). Reports how many cells are in scope and which carry it, so it
         cannot pass by finding nothing to look at.
  C2 +-  FIRES BOTH WAYS, PER ROW, on an in-memory copy. The shipped table must score zero bad
         rows, and each in-scope row with its significance token deleted ALONE must score
         exactly one. Counting the total rather than the delta is w114's vacuous control.
  C3 +-  THE REAL HISTORICAL DEFECT, frozen as a literal rather than read from HEAD (w115).
         PREFIX_ROW3 is row 3's cell as it stood before this run -- the cell that passes all
         seven siblings -- and it must FIRE here.
  C4 +-  THE LABEL'S DISCRIMINATING POWER IS MEASURED, NOT ASSERTED. t is computed for every
         arm in the five run artefacts and split by the artefacts' OWN `sign` field. At least
         one `consistent` arm must land under WEAK_T and at least one `SIGN FLIPS` arm must
         exist; if `consistent` cleanly separated the label would be doing the job and this
         guard would be unnecessary, so that outcome is reported as INERT and fails. WEAK_T is
         1.5*sqrt(REPS), the w132 threshold carried onto w158's corrected scale unchanged in
         substance -- a bare 1.5 there would have silently tightened the rule by sqrt(3).
  C5 +-  THE SIBLINGS' BLINDNESS IS MEASURED, NOT ASSERTED. #53/#55/#56/#57/#58/#59/#60 are
         imported and their own C1 predicates run over a table whose row 3 is PREFIX_ROW3. All
         seven must return ZERO findings for row 3 while this file's C1 fires, and each must
         still fire on ITS OWN frozen defect in the same call, else the demonstration is
         vacuous.
  C6 +-  SCOPE-INVARIANCE, so the guard cannot be satisfied by rescaling. For every artefact
         arm, t from (delta, sd) must equal t from (per_member, sd/n) to 1e-9 -- and a cell
         quoting a GROUP sd beside a MEMBER price is therefore still checkable, which is why
         #61 asks for t rather than for a scope label.

WHAT #61 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST TEN DEFECTS LIVED. Its
SIGN-VERDICT and SIGNIFICANCE vocabularies are finite and hand-written: a cell can close on a
consistency argument phrased outside them (`the same sign every time I ran it`) and pass, or
wear a `t` innocently and pass without one being computed. It cannot check that a published t
is the RIGHT t -- it never re-derives one from arrays. It says nothing about df: a t of 5.12
on 2 degrees of freedom is a different object from a t of 5.12 on 200, and nothing here reads
the split count. And like all seven siblings it reads only the price COLUMN.
🎯 Eight guards now share one column. Every one of them checks a CLAIM ABOUT a number. Not one
re-derives the number itself from the arrays, so a cell that is well-formed in all eight
vocabularies and simply WRONG passes every check this workspace owns.

    .venv/bin/python experiments/w132b_significanceguard.py            # rc 0 = clean
    .venv/bin/python experiments/w132b_significanceguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")

import w128b_pricedguard as g57  # noqa: E402 -- the shared ANGLE INDEX table parser

index_rows = g57.index_rows

ROW3 = 3

MAGNITUDE = re.compile(r"[-+−]?\d[\d,.]*e-[67]")

# The bracketed word the column has closed on for ten runs, in every phrasing it has used.
SIGN_VERDICT = re.compile(
    r"\bsign[- ]?consistent\b|\bconsistent\s*3/3\b|\bsign[- ]?flipp?(?:ing|s)\b"
    r"|\bSIGN\s+FLIPS\b|\bsign[- ]?consistency\b",
    re.I)

# What it takes to turn that word into a claim about noise.
SIGNIFICANCE = re.compile(
    r"\bt\s*=\s*[-+−]?\d|\|t\|\s*[=<>]|\bt-stat|\bfalse[- ]positive rate\b|\bFPR\b"
    r"|\bcritical\s+t\b|\bdf\s*=\s*\d",
    re.I)

# C3/C5. Row 3 exactly as it stood before this run -- the cell that passes all seven siblings.
PREFIX_ROW3 = (
    "an **ENROLMENT** price (value of ADDING a member). ⚠ **TWO PRICES, AND THE ROW USED TO "
    "PUBLISH ONLY THE LOWER ONE.** **5.9e-6/member** is the `rest`-group average, and `rest` "
    "is a **RESIDUAL** (8/35 CatBoost, 4 neural nets), so it is not a CatBoost price; it "
    "re-measures **+5.59e-6/member** on today's base104. The **8 CatBoosts measured alone "
    "read +10.04e-6/member** (±0.000016 on the group delta, sign-consistent over 3 splits), "
    "which independently corroborates the only other pure-CatBoost measurement here — w20d's "
    "foreign `cat` group at **10.3e-6/member**. ⛔ Both are FOREIGN pipelines, so the "
    "operational rule is unchanged and reinforced: *prefer a pipeline we do not hold*, NOT "
    "*prefer CatBoost*")

ARTEFACTS = ("w123a_row3.json", "w124a_row4.json", "w127a_row7.json",
             "w128a_row9.json", "w131a_row1.json")

# The paired-split count every one of those five producers ran (`REPS = 3`). `sd` in the
# artefacts is std(ddof=1) over the REPS per-split deltas, so the standard error is sd/sqrt(REPS)
# and t is delta/(sd/sqrt(REPS)). #68 reads REPS out of the producers and out of this line, and
# goes red if they ever disagree -- do not edit one without the other.
REPS = 3
SQRT_REPS = math.sqrt(REPS)   # named for what it IS: sd/SQRT_REPS is the standard error
WEAK_T = 1.5 * SQRT_REPS   # w132's 1.5 carried onto w158's scale; see C4


def in_scope(cell):
    """A cell closes on a sign verdict and carries a magnitude, so its noise is load-bearing."""
    return bool(MAGNITUDE.search(cell) and SIGN_VERDICT.search(cell))


def insignificant(rows):
    """C1 predicate: cells that close on a sign verdict without ever saying what it is worth."""
    return [(n, c[:110]) for n, c in sorted(rows.items())
            if in_scope(c) and not SIGNIFICANCE.search(c)]


def strip_significance(cell):
    """C2: remove the significance tokens, leaving the sign verdict and every magnitude."""
    return SIGNIFICANCE.sub("...", cell)


def arm_table():
    """Every (delta, sd, n, sign) arm in the five run artefacts, with both t's.

    The classifier is the artefacts' own `sign` field, never this file's judgement.
    """
    out = []
    for nm in ARTEFACTS:
        with open(os.path.join(HERE, nm)) as fh:
            d = json.load(fh)
        for block in ("paired", "arms"):
            for k, v in (d.get(block) or {}).items():
                if not isinstance(v, dict) or "sd" not in v or "delta" not in v:
                    continue
                n = v.get("n") or (round(v["delta"] / v["per_member"])
                                   if v.get("per_member") else 1)
                pm = v.get("per_member", v["delta"])
                out.append({"src": nm[:5], "arm": k.strip()[:30], "n": n,
                            "per_member_e6": pm * 1e6, "sd_member_e6": v["sd"] / n * 1e6,
                            "t_group": v["delta"] / (v["sd"] / SQRT_REPS),
                            "t_member": pm / (v["sd"] / n / SQRT_REPS),
                            "sign": v.get("sign", "consistent")})
        ac = d.get("arm_c")
        if ac and "sd_e6" in ac:
            out.append({"src": nm[:5], "arm": "origmodel enrolment", "n": 1,
                        "per_member_e6": ac["mean_e6"], "sd_member_e6": ac["sd_e6"],
                        "t_group": ac["mean_e6"] / (ac["sd_e6"] / SQRT_REPS),
                        "t_member": ac["mean_e6"] / (ac["sd_e6"] / SQRT_REPS),
                        "sign": "SIGN FLIPS" if ac.get("sign_flipping") else "consistent"})
    return out


def sibling_predicates():
    """The seven siblings' own C1 predicates, imported rather than reimplemented."""
    import w124b_priceunitguard as g53
    import w126c_scopeguard as g55
    import w127b_rateguard as g56
    import w129b_optoutguard as g58
    import w130b_zerobaselineguard as g59
    import w131b_selfbaselineguard as g60
    return [
        ("#53 w124b unpriced (quantity)", g53.unpriced, 4, g53.PREFIX_CELLS[4]),
        ("#55 w126c unscoped (scope)", g55.unscoped, 6, g55.PREFIX["row6_cell"]),
        ("#56 w127b undenominated (denominator)", g56.undenominated, 7, g56.PREFIX_ROW7),
        ("#57 w128b magnitudeless (silence)", g57.magnitudeless, 9, g57.PREFIX_ROW9),
        ("#58 w129b unpriced_claim (opt-out)", g58.unpriced_claim, 10, g58.PREFIX_ROW10),
        ("#59 w130b ungrounded (baseline)", g59.ungrounded, 8, g59.PREFIX_ROW8),
        ("#60 w131b undisclosed (contrast)", g60.undisclosed, 1, g60.PREFIX_ROW1),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1 over the frozen pre-fix row 3; it must fire")
    a = ap.parse_args()

    txt = open(RESEARCH, encoding="utf-8").read()
    rows = index_rows(txt)
    if not rows:
        print("w132b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w132b --control: C1 against the frozen pre-fix row 3\n")
        pre = dict(rows)
        pre[ROW3] = PREFIX_ROW3
        hits = {n for n, _ in insignificant(pre)}
        print(f"  C1 pre-fix row 3: magnitude {bool(MAGNITUDE.search(PREFIX_ROW3))}  "
              f"sign verdict {bool(SIGN_VERDICT.search(PREFIX_ROW3))}  "
              f"significance {'PRESENT' if SIGNIFICANCE.search(PREFIX_ROW3) else 'ABSENT'}  "
              f"-> {'FIRES' if ROW3 in hits else 'SILENT (BAD)'}")
        s = insignificant(rows)
        print(f"  shipped C1 {'SILENT' if not s else 'FIRES (BAD): ' + str(s)}")
        works = ROW3 in hits and not s
        print(f"\n  the control {'WORKS' if works else 'DOES NOT WORK'}: the pre-fix cell "
              f"fires, the shipped table is silent")
        return 0

    fails = []
    print("w132b -- #61: a price closed on sign-consistency must say what that is worth\n")

    # ------------------------------------------------------------------------------ C1
    scope = [n for n, c in sorted(rows.items()) if in_scope(c)]
    bad = insignificant(rows)
    print("C1 every cell closing on a sign verdict must carry a significance token")
    print(f"   {len(rows)} price cells parsed, {len(scope)} close on a sign verdict: {scope}")
    for n in scope:
        c = rows[n]
        print(f"     row {n:2d}  sign verdict yes  significance "
              f"{'PRESENT' if SIGNIFICANCE.search(c) else 'ABSENT'}")
    if len(rows) < 10 or not scope:
        fails.append("C1 VACUOUS: fewer than 10 cells parsed, or none in scope")
    if bad:
        fails.append(f"C1: cell(s) close on a sign verdict with no significance: {bad}")
    print(f"   -> {len(bad)} finding(s)")

    # ------------------------------------------------------------------------------ C2
    print("\nC2 fires both ways, per row, on an in-memory copy")
    base_bad = len(insignificant(rows))
    ok = base_bad == 0
    detail = []
    for n in scope:
        pert = dict(rows)
        pert[n] = strip_significance(rows[n])
        k = len(insignificant(pert))
        detail.append(f"row {n}->{k - base_bad}")
        if k - base_bad != 1:
            ok = False
    print(f"   shipped table -> {base_bad} bad; each in-scope row stripped alone -> "
          f"{', '.join(detail)}")
    if not ok:
        fails.append("C2: the shipped table is not clean, or a stripped row does not add "
                     "exactly one finding")
    print(f"   -> {'fires both ways, per row. OK' if ok else 'BROKEN'}")

    # ------------------------------------------------------------------------------ C3
    print("\nC3 the real historical defect, frozen as a literal")
    pre = dict(rows)
    pre[ROW3] = PREFIX_ROW3
    c3 = ROW3 in {n for n, _ in insignificant(pre)}
    print(f"   pre-fix row 3 (passes all seven siblings) -> "
          f"{'FIRES here. OK' if c3 else 'SILENT -- INERT'}")
    if not c3:
        fails.append("C3: the frozen pre-fix row 3 does not fire; the guard is inert")

    # ------------------------------------------------------------------------------ C4
    print("\nC4 the label's discriminating power is measured, not asserted")
    arms = arm_table()
    if len(arms) < 8:
        fails.append(f"C4 VACUOUS: only {len(arms)} arms found in the artefacts")
    cons = [a_ for a_ in arms if a_["sign"] != "SIGN FLIPS"]
    flips = [a_ for a_ in arms if a_["sign"] == "SIGN FLIPS"]
    print(f"   {len(arms)} arms across {len(ARTEFACTS)} artefacts: "
          f"{len(cons)} `consistent`, {len(flips)} `SIGN FLIPS`")
    print(f"   {'src':<6} {'arm':<32} {'n':>2} {'/member':>10} {'sd/mem':>8} {'|t|':>5}  label")
    for a_ in sorted(arms, key=lambda x: -abs(x["t_member"])):
        print(f"   {a_['src']:<6} {a_['arm']:<32} {a_['n']:>2} "
              f"{a_['per_member_e6']:>+9.3f}e-6 {a_['sd_member_e6']:>7.3f} "
              f"{abs(a_['t_member']):>5.2f}  {a_['sign']}")
    weak = [a_ for a_ in cons if abs(a_["t_member"]) < WEAK_T]
    print(f"   `consistent` arms under |t| = {WEAK_T:.3f}: {len(weak)} "
          f"({', '.join(a_['arm'] for a_ in weak) or 'none'})")
    if not weak or not flips:
        fails.append("C4 INERT: `consistent` separates cleanly, so the label is doing the job "
                     "and #61 would be unnecessary")
    print(f"   sign-consistency FPR under a symmetric null over 3 splits: "
          f"2*(1/2)^3 = {2 * 0.5 ** 3:.2f}")
    print(f"   -> {'the label does not separate. OK' if (weak and flips) else 'INERT'}")

    # ------------------------------------------------------------------------------ C5
    print("\nC5 the seven siblings, imported and run blind on the pre-fix row 3")
    for name, pred, own_row, own_cell in sibling_predicates():
        blind = {h[0] for h in pred(pre)}
        own = dict(rows)
        own[own_row] = own_cell
        still = own_row in {h[0] for h in pred(own)}
        okk = (ROW3 not in blind) and still
        print(f"   {name:<40} row 3 {'SILENT' if ROW3 not in blind else 'FIRES (BAD)'}  "
              f"own defect {'FIRES' if still else 'SILENT (VACUOUS)'}  "
              f"{'OK' if okk else 'BROKEN'}")
        if not okk:
            fails.append(f"C5 {name}: blind-run or own-defect check failed")

    # ------------------------------------------------------------------------------ C6
    print("\nC6 scope-invariance: t cannot be changed by rescaling the sd")
    worst = max(abs(a_["t_group"] - a_["t_member"]) for a_ in arms)
    print(f"   max |t_group - t_member| over {len(arms)} arms: {worst:.3e}")
    if worst > 1e-9:
        fails.append(f"C6: t is not scale-invariant (max gap {worst:.3e})")
    print(f"   -> {'t is scale-invariant, so #61 asks for t and not for a scope label. OK' if worst <= 1e-9 else 'BROKEN'}")

    print(f"\nFAILURES: {len(fails)}")
    for f_ in fails:
        print("  " + f_)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
