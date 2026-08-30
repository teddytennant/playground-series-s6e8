"""w127b -- STANDING CHECK #56. AN ENROLMENT PRICE IS A RATE, SO IT NEEDS A DENOMINATOR.
#53 gave the ANGLE INDEX's price column its QUANTITIES. #54 gave row 5 its LAYER. #55 gave
row 6 its SCOPE -- and #55's docstring names, in terms, the hole this fills. (w127, 2026-08-30)

    "WHAT C1 IS BLIND TO ... C1 fires only on cells carrying BOTH a magnitude and the word
     SEARCH. A cell whose quantity is ENROLMENT or TUNING can be scoped just as wrongly ...
     and C1 will not look at it."

w126 wrote that one run before the harness handed row 7, whose price cell is exactly that
shape. Fifth consecutive run in which the previous run's written-down exemption is where the
next defect lived.

THE DEFECT THIS EXISTS BECAUSE OF. Row 7 read, in full:

    structural null (stacker) · **ENROLMENT** +2e-6 (member)

Rows 3 and 4 publish ENROLMENT prices as PER-MEMBER RATES -- `5.9e-6/member`,
`+10.04e-6/member`, `+7.38e-6/member`. Rows 5 and 6 put a LAYER word in that parenthetical --
`MEMBER layer`, `TOP-LEVEL layer, k=4`. Row 7's `(member)` is in the same position and is
NEITHER. The +2e-6 is a STACK-layer total for ONE member (the 2026-08-13 reading
`+138e-6 solo -> +2e-6 stack` for seed-averaging `xgb_latcat`, where the workspace's 1.4%
pass-through constant was fitted), i.e. k=1.

AND IT DECIDES A CLOSURE, THE SAME WAY ROW 6's DID. Read `+2e-6 (member)` with rows 3/4's
convention at the pack's k=104 and it multiplies out to +208e-6 -- FOUR TIMES the 50e-6 floor,
which re-opens the row on the last working day before the deadline. Read at its real scope,
k=1, it is +2e-6 and the row stays shut. w127a measured the arms the cell should have carried;
they are in `w127a_row7.json` and C3 compares against them.

A SECOND THING THE CELL GETS WRONG, and C2 covers it. ENROLMENT is defined by rows 3/4 as "the
value of ADDING a member the pack does not hold". The +2e-6 is w109's arm B -- three seed twins
REPLACED by their mean. Nothing is added; the pack loses two columns. That is a SUBSTITUTION
price wearing an ENROLMENT label, and w118 had already recorded that the configuration it was
measured in stopped shipping.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1 +   DENOMINATOR, NOT VACUOUS. Every ANGLE INDEX price cell carrying an e-6/e-7 magnitude
         AND the word ENROLMENT must state a denominator: a per-member rate token
         (`/member`, `per member`) or an explicit `k=`. Vacuous, and says so, if no such cell
         exists. Deliberately the SAME magnitude predicate as #53 and #55 so the three guards
         agree on what "priced" means.
  C2 +   ROW 7 PUBLISHES BOTH LAYERS AND ITS SCOPE. The cell must carry the MEMBER-layer
         number (+138e-6), the STACK-layer number (+2e-6), a `k=` for each, and the word
         SUBSTITUTION -- so no reader can take the stack number for a rate, and no reader can
         take the ENROLMENT label for "this was measured by adding something".
  C3 +-  THE ARTEFACT STILL SAYS IT. The member-arm gain and the in-process CatBoost control
         gap are compared against `w127a_row7.json`, with the expected values frozen HERE as
         literals so this is a COMPARISON, not a recomputation that agrees with itself. If the
         json is absent the check reports itself INERT rather than green -- a missing artefact
         is not a pass (#55's C4 rule).
  C4 +-  --control runs C1 and C2 over the FROZEN PRE-FIX row 7 cell, inlined as a literal
         rather than read from HEAD, because a control anchored to HEAD stops being a control
         the moment the fix is committed (w115). Pre-fix must FIRE on both; shipped must be
         silent on both.

WHAT C1 IS BLIND TO, WRITTEN DOWN, BECAUSE THAT IS WHERE THE LAST FIVE DEFECTS LIVED. C1 asks
for a denominator TOKEN; it cannot tell whether the denominator is the RIGHT one. Row 3's
`5.9e-6/member` divides by 35 names over 34 distinct arrays (w124 recorded the duplicate) and
C1 passes it. C1 also still ignores TUNING and CONCAT cells entirely: row 1's dose ladder
(`-58e-6 at 1x, -3,340e-6 at 50x`) is a rate in a unit no token here matches, and rows 2/4's
`+4e-7` is a per-tune total with no k at all. If the next defect is in this genus, look there.

    .venv/bin/python experiments/w127b_rateguard.py            # rc 0 = clean
    .venv/bin/python experiments/w127b_rateguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")
ROW7_JSON = os.path.join(HERE, "w127a_row7.json")
ANGLE_HEAD = "# 📇 THE ANGLE INDEX"

# C1. Same magnitude predicate as #53/#55.
MAGNITUDE = re.compile(r"\d\s*e-[67]|\d,\d{3}e-[67]")
ENROL_WORD = "ENROLMENT"
RATE_TOKEN = re.compile(r"/member\b|per member\b|per-member\b")
K_TOKEN = re.compile(r"\bk=\d+")

# C2. Row 7 carries two layers of one manoeuvre; both must be legible from the cell alone.
ROW7 = 7
ROW7_TOKENS = ("+138e-6", "+2e-6", "MEMBER layer", "STACK layer", "k=1", "SUBSTITUTION")

# C3. Frozen literals -- w127a_row7.json, base104, paired 50/50, splits 0/1/2, C=1.0, hybrid.
EXPECT = {
    "member_gain": 138.2e-6,      # w118's +138.2e-6, recomputed by w127a's R1
    "gain_tol": 1e-6,
    "control_tol": 0.05e-6,       # w123's CatBoost rate must come back in-process
}

# C4. The cell exactly as it stood before this run's edit.
PREFIX_ROW7 = "structural null (stacker) · **ENROLMENT** +2e-6 (member)"


def index_rows(txt):
    """{row number -> price cell} from the ANGLE INDEX table, anchored on the block head so a
    `| 7 |` line elsewhere in the document cannot be picked up (w110's first-occurrence trap)."""
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


def undenominated(rows):
    """C1 predicate: priced ENROLMENT cells that state no denominator."""
    bad = []
    for n, c in sorted(rows.items()):
        if not (MAGNITUDE.search(c) and ENROL_WORD in c):
            continue
        if not (RATE_TOKEN.search(c) or K_TOKEN.search(c)):
            bad.append((n, c[:120]))
    return bad


def missing_row7(cell):
    """C2 predicate: which of row 7's required tokens the cell does not publish."""
    return [t for t in ROW7_TOKENS if t not in cell]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run C1/C2 over the frozen PRE-FIX cell; both must fire")
    a = ap.parse_args()

    txt = open(RESEARCH).read()
    rows = index_rows(txt)
    if not rows:
        print("w127b: the ANGLE INDEX block was not found in RESEARCH.md")
        return 1

    if a.control:
        print("w127b --control: C1/C2 against the frozen PRE-FIX row 7 cell\n")
        pre = dict(rows)
        pre[ROW7] = PREFIX_ROW7
        c1 = undenominated(pre)
        c2 = missing_row7(PREFIX_ROW7)
        print(f"  C1 pre-fix row {ROW7} cell {PREFIX_ROW7!r}")
        print(f"     {'FIRES' if any(n == ROW7 for n, _ in c1) else 'SILENT (BAD)'} "
              f"-- undenominated ENROLMENT rows: {[n for n, _ in c1]}")
        print(f"  C2 {'FIRES' if c2 else 'SILENT (BAD)'} -- missing tokens: {c2}")
        s1 = undenominated(rows)
        s2 = missing_row7(rows.get(ROW7, ""))
        print(f"\n  shipped C1 {'SILENT' if not s1 else 'FIRES (BAD): ' + str(s1)}")
        print(f"  shipped C2 {'SILENT' if not s2 else 'FIRES (BAD): ' + str(s2)}")
        ok = (any(n == ROW7 for n, _ in c1) and c2 and not s1 and not s2)
        print(f"\n  the control {'WORKS' if ok else 'DOES NOT WORK'}: pre-fix fires on both, "
              f"shipped is silent on both")
        return 0

    fails = []
    print("w127b -- #56: an ENROLMENT price is a RATE, so it needs a DENOMINATOR\n")

    # ---------------------------------------------------------------------------- C1
    priced = [n for n, c in rows.items() if MAGNITUDE.search(c) and ENROL_WORD in c]
    print(f"[C1] priced ENROLMENT cells in the ANGLE INDEX: rows {sorted(priced)}")
    if not priced:
        print("     VACUOUS -- no priced ENROLMENT cell exists today. C1 asserts nothing.")
    bad = undenominated(rows)
    for n, c in bad:
        print(f"     FAIL row {n} prices an ENROLMENT with no denominator: {c}")
        fails.append(f"ANGLE INDEX row {n}: an ENROLMENT price with no `/member` and no `k=`")
    if priced and not bad:
        print("     OK every priced ENROLMENT cell states a denominator")

    # ---------------------------------------------------------------------------- C2
    cell = rows.get(ROW7, "")
    miss = missing_row7(cell)
    print(f"\n[C2] row {ROW7} publishes both layers and its scope")
    if miss:
        print(f"     FAIL missing: {miss}")
        print(f"     cell: {cell[:200]}")
        fails.append(f"ANGLE INDEX row {ROW7} does not publish {miss}")
    else:
        print("     OK the cell carries both layers, both scopes, and the word SUBSTITUTION")

    # ---------------------------------------------------------------------------- C3
    print("\n[C3] the artefact still says it -- w127a_row7.json")
    if not os.path.exists(ROW7_JSON):
        print(f"     INERT {os.path.basename(ROW7_JSON)} is not on disk. This check is not "
              f"green, it is not running. Re-run experiments/w127a_row7.py.")
    else:
        j = json.load(open(ROW7_JSON))
        ma = j.get("member_arm", {})
        got = ma.get("gain")
        if got is None:
            print("     FAIL the artefact holds no member-arm gain")
            fails.append("w127a_row7.json has no member_arm.gain")
        else:
            d = abs(got - EXPECT["member_gain"])
            mark = "OK " if d <= EXPECT["gain_tol"] else "FAIL "
            print(f"     {mark}member-layer gain {got*1e6:+.2f}e-6 against the frozen "
                  f"{EXPECT['member_gain']*1e6:+.1f}e-6")
            if d > EXPECT["gain_tol"]:
                fails.append(f"the member-layer +138.2e-6 recomputes at {got*1e6:+.2f}e-6")
        ctl = j.get("control", {})
        if "gap" in ctl:
            g = abs(ctl["gap"])
            mark = "OK " if g <= EXPECT["control_tol"] else "FAIL "
            print(f"     {mark}the in-process CatBoost control reproduces w123 to "
                  f"{ctl['gap']*1e6:+.4f}e-6")
            if g > EXPECT["control_tol"]:
                fails.append("w127a's arms are not in rows 3/4's units (control gap "
                             f"{ctl['gap']*1e6:+.4f}e-6)")
        else:
            print("     note the artefact was written with --quick; no control gap to check")
        if j.get("reopen"):
            print("     FAIL the artefact says row 7 RE-OPENS")
            fails.append("w127a_row7.json reports reopen=true and row 7 is still marked CLOSED")
        elif "reopen" in j:
            print("     OK the artefact says row 7 does not re-open")

    print(f"\nFAILURES: {len(fails)}")
    for f in fails:
        print("  - " + f)
    if not fails:
        print("✅ CLEAN — every ENROLMENT price says what it is divided by, and row 7 says "
              "which layer each of its two numbers lives at")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
