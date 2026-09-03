"""w158a -- STANDING CHECK #68. THE NUMBER THE PRICE COLUMN PUBLISHES AS `t` WAS `delta/sd`,
WHICH IS AN EFFECT SIZE AND NOT A t, AND IT WAS COMPARED AGAINST t-DISTRIBUTION CRITICAL
VALUES ON df = 2. (w158, 2026-09-03)

THE DEFECT. Five run artefacts price their arms as (delta, sd) over REPS = 3 paired 50/50
splits, with `sd = d.std(ddof=1)` taken over the three per-split deltas. The standard error of
that mean is sd/sqrt(REPS), so the statistic is

    t = delta / (sd / sqrt(REPS))          NOT     delta / sd

w132a introduced t to the ANGLE INDEX as `delta/sd` and #61 `w132b_significanceguard` shipped
the same expression twice (`t_group`, `t_member`). Every t this workspace has published since
is understated by sqrt(3) = 1.7321. The table then compares those numbers against 4.303 and
9.925 -- the two-tailed 5% and 1% critical values of Student's t on df = REPS - 1 = 2 -- which
belong to the CORRECTED scale. Two different objects, one comparison.

🎯 IT SURVIVED 26 RUNS BECAUSE THE ERROR IS IN THE SAFE DIRECTION FOR DISCOVERY. Understating t
can only make an arm look MORE null, never less, so nothing was ever wrongly promoted. What it
can do is wrongly DISMISS, and the honest report is that it did not. Over all 16 arms in the five
artefacts the 5% partition is IDENTICAL before and after -- the same 7 named arms clear 4.303
either way, and the largest corrected t among those that do not is 2.92. No row re-opens.

🔴 BUT IT FALSIFIES A PUBLISHED SENTENCE, IN BOTH THE TABLE AND #61's OWN DOCSTRING:
`NO single-family enrolment rate in this table clears 1%`. On the corrected scale +xgb_only
reads 10.61 and +xgb_dedup 10.75 against 9.925, and BOTH clear it. +lgb_only misses by 0.001
at 9.924. Row 3's `clears 5% with a thin margin` was also an artefact of the wrong scale: the
CatBoost control is 8.87, which is 2.06x the critical value, not a thin margin.

⚠ AND #61 NAMED THIS BLINDNESS ONE SENTENCE OFF. Its own docstring says "It says nothing about
df: a t of 5.12 on 2 degrees of freedom is a different object from a t of 5.12 on 200, and
nothing here reads the split count." The split count was not a missing CONTEXT for the number;
it was a missing FACTOR IN the number. #61 wrote down the exact input it needed, filed it under
scope, and shipped without it. Its closing line -- "a cell that is well-formed in all eight
vocabularies and simply WRONG passes every check this workspace owns" -- was about itself.

WHAT THIS DOES NOT FIX, SAID PLAINLY. Rescaling does not make three random 50/50 re-splits of
one dataset three independent observations. The df = 2 test was always an approximation and
still is; #68 checks that the number published is the statistic the table's own critical values
are drawn from, not that the test is the right test.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  C1  +   NOT VACUOUS. Every `t = <number>` token in the ANGLE INDEX price column, extracted per
          row and counted, so the check cannot pass by finding nothing to read.
  C1b +-  THE MATCHER TRACKS THE TEXT. Renumber one token and the extracted set must move;
          delete every token and the count must go to zero.
  C2  +-  THE DENOMINATOR IS READ, NOT ASSUMED. REPS is parsed out of all five artefact
          producers AND out of #61, all six must agree, and df = REPS - 1 must equal the df the
          table publishes. A producer that changes its split count turns this red instead of
          silently rescaling every t in the document.
  C3a +-  EVERY PUBLISHED t IS THE CORRECTED t. Each token must match some artefact arm's
          delta/(sd/sqrt(REPS)) to 0.005 (the rounding half-width of a 2dp figure) and must NOT
          match that same catalogue on the pre-w158 delta/sd scale.
  C3b +-  THE 1% VERDICT AGREES WITH THE ARITHMETIC. The single-family enrolment arms clearing
          9.925 are recomputed; if that set is non-empty the price column may not carry a
          negative-universal ("none clears 1%"), and if it is empty it may not claim one does.
  C4  +   ANCHORED IN #61. The three lines this guard's arithmetic depends on -- `SE`, the two
          t expressions and `WEAK_T` -- must still be there, so a revert there goes red HERE
          rather than quietly putting the document back on the wrong scale.
  C5  +-  --control. The 13 tokens and the two verdict sentences are frozen at their pre-w158
          values and substituted back into an in-memory copy of the live cells. Every
          substitution must apply exactly once or the control reports BROKEN rather than
          passing on noise (w157), and C3a must then fire on exactly 13 and C3b on exactly 1.
  C6  +   BLINDNESS, written down.

    .venv/bin/python experiments/w158a_tscaleguard.py            # rc 0 = clean
    .venv/bin/python experiments/w158a_tscaleguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

RESEARCH = os.path.join(ROOT, "RESEARCH.md")

import w128b_pricedguard as g57            # noqa: E402 -- the shared ANGLE INDEX table parser
import w132b_significanceguard as g61      # noqa: E402 -- the arm catalogue, and #68's subject

TOL = 0.005          # a 2dp figure is right to +-0.005; anything looser hides a sqrt(3)
CRIT5, CRIT1 = 4.303, 9.925                # two-tailed Student's t, df = 2

T_TOKEN = re.compile(r"\|?t\|?\s*=\s*([-+−]?\d+(?:\.\d+)?)")

# C2. Where REPS is written down. Parsed, never assumed.
REPS_SOURCES = ("w123a_row3.py", "w124a_row4.py", "w127a_row7.py", "w128a_row9.py",
                "w131a_row1.py", "w132b_significanceguard.py")
# ⚠ w131a writes it as a tuple unpack (`REPS, CVAL, TRANSFORM, DECORR_MAX = 3, 1.0, ...`), so a
# `^REPS = (\d+)` regex reads None there and C2 reports a disagreement that is really a parse
# miss. Split the targets and take REPS's position.
REPS_ASSIGN = re.compile(r"^(REPS\b[^=\n]*?)=\s*(.+)$", re.M)

# C3b. A negative universal about the 1% level, in the phrasings the column has used.
NEG_1PCT = re.compile(r"(?:no|none|NO|NONE)\b[^.|]{0,90}?clears?\s+1%", re.I)
POS_1PCT = re.compile(r"clears?\s+1%", re.I)

# C4. #61 lines this guard's arithmetic stands on.
ANCHORS_61 = (
    "SQRT_REPS = math.sqrt(REPS)",
    'v["delta"] / (v["sd"] / SQRT_REPS)',
    'pm / (v["sd"] / n / SQRT_REPS)',
    "WEAK_T = 1.5 * SQRT_REPS",
)

# C5. Frozen pre-w158 text: the 13 t tokens and the two verdict sentences, as they stood
# before this run. `new` is what is live now; substituting `old` back in rebuilds the specimen.
# ⚠ These are LITERALS on both sides on purpose. w156: a control that borrows live state stops
# working when the fix lands. w157: a control whose frozen state is narrower than its scope
# passes on noise. Here every substitution is required to apply EXACTLY ONCE, so a control that
# has stopped matching the document reports BROKEN instead of quietly testing nothing.
PREFIX_SUBS = [
    (1, "**t = 0.97**", "**t = 0.56**"),
    (3, "per-member sd 1.96e-6, t = 8.87**", "per-member sd 1.96e-6, t = 5.12**"),
    (3, "**t = 1.66**", "**t = 0.96**"),
    (4, "(t = 8.87) · XGBoost", "(t = 5.12) · XGBoost"),
    (4, "(t = 10.75)", "(t = 6.21)"),
    (4, "(t = 9.92)", "(t = 5.73)"),
    (7, "(t = 0.44)", "(t = 0.25)"),
    (7, "(t = 0.81)", "(t = 0.46)"),
    (7, "(t = 0.39)", "(t = 0.22)"),
    (9, "(t = 2.92)", "(t = 1.69)"),
    (9, "(t = 7.82)", "(t = 4.51)"),
    (9, "t = 0.33,", "t = 0.19,"),
    (9, "**t = 1.66**", "**t = 0.96**"),
]
PREFIX_VERDICTS = [
    (3,
     "this rate clears 5% at 2.06× the critical value, and **two of the four single-family "
     "enrolment rates DO clear 1%**",
     "this rate clears 5% with a thin margin and **no single-family enrolment rate in this "
     "table clears 1%**"),
    (4,
     "every one clears 5% and **XGBoost also clears 1%** at 10.75",
     "every one clears 5%, none clears 1%"),
]


def reps_table():
    """C2. REPS as each source writes it, keyed by file. Handles the tuple-unpack form."""
    out = {}
    for nm in REPS_SOURCES:
        val = None
        m = REPS_ASSIGN.search(open(os.path.join(HERE, nm), encoding="utf-8").read())
        if m:
            names = [s.strip() for s in m.group(1).split(",")]
            vals = [s.strip() for s in re.sub(r"\s+#.*$", "", m.group(2)).split(",")]
            if "REPS" in names and len(names) == len(vals):
                raw = vals[names.index("REPS")]
                val = int(raw) if raw.isdigit() else None
        out[nm] = val
    return out


def catalogue(reps):
    """Every artefact arm with both scales. t_corr comes from #61, so #61 is the anchor."""
    out = []
    for a in g61.arm_table():
        t_corr = abs(a["t_member"])
        out.append({"src": a["src"], "arm": a["arm"], "sign": a["sign"],
                    "per_member_e6": a["per_member_e6"],
                    "t_corr": t_corr, "t_pre": t_corr / math.sqrt(reps)})
    return out


def tokens(rows):
    """C1. (row, value) for every t the price column publishes."""
    out = []
    for n, cell in sorted(rows.items()):
        for v in T_TOKEN.findall(cell):
            out.append((n, float(v.replace("−", "-"))))
    return out


def classify(val, cat):
    """C3a. Which scale, if either, this published number sits on."""
    corr = [a for a in cat if abs(abs(val) - a["t_corr"]) <= TOL]
    pre = [a for a in cat if abs(abs(val) - a["t_pre"]) <= TOL]
    return corr, pre


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true")
    args = ap.parse_args()
    fails = []

    text = open(RESEARCH, encoding="utf-8").read()
    live = g57.index_rows(text)
    rows = dict(live)

    print("w158a -- #68 the price column's t is delta/(sd/sqrt(REPS)), not delta/sd")

    # -------------------------------------------------------------------------------- C2
    print("\nC2 REPS is read out of every source that fixes it, and they must agree")
    reps_by_file = reps_table()
    for nm, v in reps_by_file.items():
        print(f"   {nm:<32} REPS = {v}")
    vals = {v for v in reps_by_file.values()}
    if None in vals or len(vals) != 1:
        fails.append(f"C2: REPS disagrees or is unreadable across the sources: {reps_by_file}")
        reps = 3
    else:
        reps = vals.pop()
    df = reps - 1
    pub_df = sorted({int(m) for m in re.findall(r"df\s*=\s*(\d+)", " ".join(live.values()))})
    ok_df = pub_df == [df]
    print(f"   REPS = {reps}  ->  df = {df};  the table publishes df = {pub_df}  "
          f"{'OK' if ok_df else 'FAIL'}")
    if not ok_df:
        fails.append(f"C2: the table publishes df = {pub_df}, the producers imply {df}")
    print(f"   the standard-error factor this guard applies: sqrt({reps}) = {math.sqrt(reps):.4f}")

    cat = catalogue(reps)

    # -------------------------------------------------------------------------------- C5
    if args.control:
        print("\nC5 --control: the 13 pre-w158 tokens and 2 verdict sentences substituted back")
        applied = 0
        for n, new, old in PREFIX_SUBS + PREFIX_VERDICTS:
            hits = rows[n].count(new)
            if hits != 1:
                fails.append(f"C5 BROKEN: row {n} substitution {new!r} matched {hits} times, "
                             "not once -- the frozen text no longer tracks the document")
                continue
            rows[n] = rows[n].replace(new, old)
            applied += 1
        want = len(PREFIX_SUBS) + len(PREFIX_VERDICTS)
        print(f"   {applied}/{want} substitutions applied exactly once  "
              f"{'OK' if applied == want else 'BROKEN'}")

    where = "the frozen pre-w158 cells" if args.control else "RESEARCH.md"

    # -------------------------------------------------------------------------------- C1
    print(f"\nC1 every t the price column publishes, from {where}")
    toks = tokens(rows)
    per_row = {}
    for n, v in toks:
        per_row.setdefault(n, []).append(v)
    for n in sorted(per_row):
        print(f"   row {n:>2}:  " + ", ".join(f"t = {v}" for v in per_row[n]))
    print(f"   {len(toks)} tokens across {len(per_row)} rows")
    if len(toks) < 10:
        fails.append(f"C1 VACUOUS: only {len(toks)} t tokens found in the price column")

    # -------------------------------------------------------------------------------- C1b
    print("\nC1b the matcher tracks the text")
    bumped = dict(rows)
    n0 = sorted(per_row)[0]
    bumped[n0] = T_TOKEN.sub(lambda m: f"t = {float(m.group(1)) + 9:.2f}", bumped[n0], count=1)
    moved = tokens(bumped) != toks
    stripped = {n: T_TOKEN.sub("t = ?", c) for n, c in rows.items()}
    gone = len(tokens(stripped))
    print(f"   renumber one token -> extracted set {'moves' if moved else 'DOES NOT MOVE'}")
    print(f"   strip every token  -> {gone} found")
    if not moved or gone:
        fails.append("C1b: the matcher does not track the text")

    # -------------------------------------------------------------------------------- C3a
    print("\nC3a every published t is the CORRECTED t, and is not the pre-w158 one")
    print(f"   {'row':>3}  {'published':>9}  {'matches corrected':<34} {'pre-w158?':<10} verdict")
    bad = 0
    for n, v in toks:
        corr, pre = classify(v, cat)
        if corr:
            name, verdict = corr[0]["arm"].strip(), "OK"
        elif pre:
            name, verdict = pre[0]["arm"].strip(), "WRONG SCALE"
        else:
            name, verdict = "-", "UNMATCHED"
        if verdict != "OK":
            bad += 1
            fails.append(f"C3a row {n}: t = {v} is {verdict}"
                         + (f" (arm {name}, corrected {pre[0]['t_corr']:.2f})" if pre else ""))
        print(f"   {n:>3}  {v:>9}  {name[:34]:<34} {'yes' if pre else 'no':<10} {verdict}")
    print(f"   -> {bad} token(s) not on the corrected scale")

    # -------------------------------------------------------------------------------- C3b
    print("\nC3b the 1% verdict in the text agrees with the recomputed arithmetic")
    over = [a for a in cat
            if (a["arm"].startswith("+") and
                (a["arm"].endswith("_only") or a["arm"] == "+xgb_dedup"))
            and a["t_corr"] > CRIT1]
    print(f"   single-family enrolment arms with |t| > {CRIT1}: "
          + (", ".join(f"{a['arm'].strip()} {a['t_corr']:.2f}" for a in over) or "none"))
    joined = " ".join(rows[n] for n in sorted(rows))
    neg = NEG_1PCT.findall(joined)
    pos = POS_1PCT.findall(joined)
    if over and neg:
        fails.append(f"C3b: {len(over)} single-family rate(s) clear 1% but the column still "
                     f"says {neg[0]!r}")
    if not over and pos and not neg:
        fails.append("C3b: no single-family rate clears 1% but the column claims one does")
    print(f"   negative-universal phrases in the column: {len(neg)}  "
          f"{neg[0][:70] if neg else ''}")
    print(f"   -> {'FAIL' if (over and neg) else 'OK'}")

    # -------------------------------------------------------------------------------- C4
    print("\nC4 anchored in #61, whose arithmetic this guard imports")
    src61 = open(os.path.join(HERE, "w132b_significanceguard.py"), encoding="utf-8").read()
    for a in ANCHORS_61:
        hit = a in src61
        print(f"   {a:<34} {'present' if hit else 'MISSING'}")
        if not hit:
            fails.append(f"C4: #61 no longer contains {a!r}; #68's catalogue may be on the "
                         "pre-w158 scale again")

    # -------------------------------------------------------------------------------- C6
    print("\nC6 blindness, written down")
    print("   reads the price column only, like all eight siblings")
    print("   checks the SCALE of a t, not whether a t-test on 3 re-splits of one dataset is")
    print("     the right test -- df = 2 here was always an approximation and still is")
    print("   binds a published number to an arm by VALUE; two arms 0.005 apart would be")
    uniq = sorted({round(a["t_corr"], 6) for a in cat})   # +cat_only is in two artefacts
    print(f"     interchangeable to it (closest DISTINCT pair in the catalogue today: "
          f"{min(b - a for a, b in zip(uniq, uniq[1:])):.3f})")

    print(f"\nFAILURES: {len(fails)}")
    for f_ in fails:
        print("  " + f_)
    if args.control:
        print("\n--control: exits 0 having fired above" if fails
              else "\n--control: SILENT (BAD) -- the control tests nothing")
        return 0 if fails else 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
