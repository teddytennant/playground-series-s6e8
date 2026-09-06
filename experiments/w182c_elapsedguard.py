"""w182c (#76) -- "past by ~N days" in a closed-board entry is a DURATION, and 19 runs
published a CALENDAR-DAY DIFFERENCE under that name. Every one is overstated by ~1.0 day.

🔴 THE DEFECT. Since w164 (2026-09-04) every closed-board entry opens with the same sentence:

    returns deadline **2026-08-31 23:59**, past by ~5.6 days, `userRank 319` of 3531

The deadline sits at 23:59, which is the very END of 08-31. So the time elapsed from it to
2026-09-05 15:07 is 4.63 days. The difference between the two CALENDAR DATES is 5. The
published figure is neither of those by accident: it is the calendar difference plus the
fraction of today, which is what `(today - 08-31) + time-of-day` computes and what a reader
would call the elapsed time only if the deadline were at midnight.

    run    read at UTC     claimed   elapsed   calendar-day diff
    w164   09-04 12:44         4.6      3.53                4.53
    w181   09-05 15:07         5.6      4.63                5.63
    w182   09-05 15:45         5.7      4.66                5.66

⚠ ALL 19 AGREE WITH THE CALENDAR COLUMN TO WITHIN 0.07 AND DISAGREE WITH THE ELAPSED COLUMN BY
~1.00. That is what makes this a diagnosis rather than a guess: the error is not drift, not a
rounding habit and not one run's slip. It is one wrong subtraction, and C4 measures the claim
against BOTH columns rather than asserting which one it matches.

🎯 WHY IT SPREAD. Same mechanism as the consecutive-run counter #75 guards: the cheapest way to
write this number is to read the entry above and add the time since. That inherits the base. The
increments are all CORRECT -- 5.5 -> 5.6 -> 5.7 tracks 4.53 -> 4.56 -> 4.66 -- so a chain checked
against itself is perfectly consistent, and only a chain checked against the clock is not. #53's
genus (a column that names one quantity and measures another) meeting #75's spread.

⛔ THE 19 CANNOT BE CORRECTED. JOURNAL.md is append-only. So they are FROZEN, exactly as #72
freezes its four and #75 its twenty-four, and the live arm is the one a future run can act on.

  C1  ANSWERABLE. Enough entries parsed, and the deadline is re-derived from each entry's own
      text rather than hardcoded -- an entry that names a different deadline is judged against
      the one it names, and if the shared literal ever stops appearing this fails rather than
      passing on an empty set (#70's rule).
  C2  LIVE. Every figure in an entry after the frozen set must be within TOL of the elapsed
      time. 0 live today: w182's addendum is the repayment and C4 checks it landed.
  C3  FROZEN. The inherited copies are exactly these 19. A 20th is a new defect and fails; one
      DISAPPEARING is an append-only violation and also fails.
  C4  EVIDENCE, both directions. The claims match the calendar column and not the elapsed one;
      the offset is one day and constant; and w182's addendum publishes the corrected figure.
  C5  RUN_HDR is IMPORTED from w117a_handcount, never re-implemented, so #76 and #50 cannot
      disagree about what a run is.
  C6  BLINDNESS, measured. An entry publishing a figure with no readable timestamp cannot be
      judged; they are counted and named rather than passed.

    .venv/bin/python experiments/w182c_elapsedguard.py            # rc 0 = clean
    .venv/bin/python experiments/w182c_elapsedguard.py --control  # exits 0, having fired
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from w117a_handcount import RUN_HDR  # noqa: E402  -- C5: one implementation, not two

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
JOURNAL = ROOT / "JOURNAL.md"

# The figure and the reading time, as the entries write them.
CLAIM = re.compile(r"past by ~?([\d.]+) days?")
STAMP = re.compile(r"at \*\*(20\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d)")
DEADLINE = re.compile(r"deadline \*\*(20\d\d)-(\d\d)-(\d\d) (\d\d):(\d\d)\*\*")

# A claim is judged wrong when it misses the elapsed time by more than this. The 19 miss by
# ~1.00, the calendar column matches to 0.07, so any threshold in (0.1, 0.9) separates them;
# 0.15 leaves room for a run rounding to one decimal from a slightly different clock read.
TOL = 0.15

# ⛔ FROZEN -- the inherited copies, append-only and uncorrectable. Written as (run, claimed)
# so that C3 fails if one is edited in place as well as if one is removed.
FROZEN = {
    "w164": 4.6, "w165": 4.6, "w166": 4.6, "w167": 4.6, "w168": 4.6, "w169": 4.6,
    "w170": 4.6, "w171": 4.6, "w172": 4.6,
    "w173": 5.5, "w174": 5.5, "w175": 5.5, "w176": 5.5,
    "w177": 5.6, "w178": 5.6, "w179": 5.6, "w180": 5.6, "w181": 5.6,
    "w182": 5.7,
}

# w182's addendum repays the defect once, here. C4 requires it to be present and correct.
REPAID_BY = "w182"

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def claims(text: str) -> list[dict]:
    """Every published `past by ~N days`, with the run that wrote it and what it should be.

    The timestamp and the deadline are taken from the entry's OWN sentence -- a window of the
    three lines ending at the claim -- so a run that read the board at a different time, or
    against a different deadline, is judged against what it actually says.
    """
    lines = text.split("\n")
    run = None
    out = []
    for i, line in enumerate(lines):
        m = RUN_HDR.match(line)
        if m:
            w = re.search(r"\bw(\d+)\b", line)
            run = f"w{w.group(1)}" if w else None
        c = CLAIM.search(line)
        if not c or run is None:
            continue
        window = "\n".join(lines[max(0, i - 3):i + 1])
        st, dl = STAMP.search(window), DEADLINE.search(window)
        rec = {"run": run, "line": i + 1, "claimed": float(c.group(1)),
               "when": None, "deadline": None}
        if st and dl:
            rec["when"] = dt.datetime(*map(int, st.groups()))
            rec["deadline"] = dt.datetime(*map(int, dl.groups()))
            rec["elapsed"] = (rec["when"] - rec["deadline"]).total_seconds() / 86400
            # what the wrong subtraction produces: whole calendar days between the DATES,
            # plus the fraction of the reading day
            days = (rec["when"].date() - rec["deadline"].date()).days
            frac = (rec["when"] - dt.datetime(rec["when"].year, rec["when"].month,
                                              rec["when"].day)).total_seconds() / 86400
            rec["calendar"] = days + frac
        out.append(rec)
        run = None                      # one claim per entry; the opener is the one that counts
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="judge a synthetic entry that publishes the calendar figure")
    a = ap.parse_args()
    control = a.control

    print("w182c -- `past by ~N days` against the clock, not against the entry above")
    print()

    text = JOURNAL.read_text(encoding="utf-8")
    if control:
        # A future run repeating the defect: correct timestamp, calendar figure.
        text += ("\n\n# 2026-09-06 — w9001 — CLOSED. ANGLE: probe.\n\n"
                 "`kaggle competitions list` at **2026-09-06 10:00 UTC** returns deadline "
                 "**2026-08-31 23:59**, past by ~6.4 days, `userRank 319`.\n")
    cs = claims(text)

    print("C1 answerable")
    judgeable = [c for c in cs if c["when"] is not None]
    print(f"  {len(cs)} entr(ies) publish the figure, {len(judgeable)} with a readable "
          f"timestamp and deadline in the same sentence")
    if len(judgeable) < 15:
        fail(f"only {len(judgeable)} judgeable claims -- the sentence shape has moved and this "
             f"guard is no longer reading it")
    dls = {c["deadline"] for c in judgeable}
    print(f"  deadline(s) the entries name themselves: "
          f"{sorted(d.strftime('%Y-%m-%d %H:%M') for d in dls)}")
    if not dls:
        fail("no entry names a deadline; the figure cannot be checked against anything")

    print("C2 LIVE -- claims outside the frozen set must match the clock")
    live = [c for c in judgeable if c["run"] not in FROZEN]
    bad = [c for c in live if abs(c["claimed"] - c["elapsed"]) > TOL]
    for c in live:
        flag = "OK" if abs(c["claimed"] - c["elapsed"]) <= TOL else "WRONG"
        print(f"  {c['run']:<7} L{c['line']:<6} claims {c['claimed']:.1f}, "
              f"elapsed {c['elapsed']:.2f}  {flag}")
    if not live:
        print("  0 live claims -- every published figure is in the frozen set")
    if bad:
        for c in bad:
            fail(f"{c['run']} publishes {c['claimed']:.1f} days past the deadline it names; "
                 f"the clock says {c['elapsed']:.2f} "
                 f"(the calendar-day difference is {c['calendar']:.2f})")
    elif live:
        print(f"  {len(live)} live claim(s), all within {TOL} d of the clock  OK")

    print("C3 FROZEN -- the inherited copies are exactly these 19")
    seen = {c["run"]: c["claimed"] for c in judgeable if c["run"] in FROZEN}
    gone = sorted(set(FROZEN) - set(seen))
    moved = sorted(r for r, v in seen.items() if abs(v - FROZEN[r]) > 1e-9)
    print(f"  {len(seen)} of {len(FROZEN)} frozen claims present and unchanged")
    if gone:
        fail(f"frozen claim(s) {gone} have DISAPPEARED -- JOURNAL.md is append-only")
    if moved:
        fail(f"frozen claim(s) {moved} were REWRITTEN in place -- append-only was violated")
    new = sorted(c["run"] for c in bad)
    if new:
        print(f"  and {len(new)} claim(s) outside it repeat the defect: {new}")

    print("C4 EVIDENCE -- which column the claims actually match, both directions")
    fr = [c for c in judgeable if c["run"] in FROZEN]
    off_elapsed = [abs(c["claimed"] - c["elapsed"]) for c in fr]
    off_cal = [abs(c["claimed"] - c["calendar"]) for c in fr]
    print(f"  against ELAPSED   max miss {max(off_elapsed):.2f} d, min {min(off_elapsed):.2f} d")
    print(f"  against CALENDAR  max miss {max(off_cal):.2f} d, min {min(off_cal):.2f} d")
    if not (min(off_elapsed) > 0.8 and max(off_cal) < 0.1):
        fail("the frozen claims no longer separate the two columns; the diagnosis in this "
             "guard's docstring is not what the corpus shows")
    else:
        print("  the 19 match the calendar column and miss the elapsed one by ~1 day  OK")
    span = max(c["claimed"] - c["elapsed"] for c in fr) - \
        min(c["claimed"] - c["elapsed"] for c in fr)
    print(f"  the offset is CONSTANT to {span:.2f} d, so this is one wrong subtraction "
          f"inherited, not drift")
    if span > 0.3:
        fail(f"the offset varies by {span:.2f} d; that is drift, not a single bad base")
    # the repayment
    addendum = re.search(r"past by \*\*4\.6[0-9]?\*\* days", text) or \
        re.search(r"elapsed .{0,40}\b4\.6\d?\b", text)
    print(f"  {REPAID_BY}'s addendum publishes the corrected figure: "
          f"{'yes' if addendum else 'NO'}")
    if not addendum:
        fail(f"{REPAID_BY} froze its own wrong figure without publishing the right one; the "
             f"defect is recorded but not repaid")

    print("C5 no disagreement with #50 about what a run is")
    print("  RUN_HDR imported from w117a_handcount -- one implementation, not two")

    print("C6 blindness, measured rather than assumed")
    blind = [c for c in cs if c["when"] is None]
    print(f"  {len(blind)} claim(s) carry no readable timestamp+deadline and cannot be judged: "
          f"{[c['run'] for c in blind] or 'none'}")
    print("  it reads the OPENING claim of an entry only; a second figure later in the same")
    print("    entry is not judged.")
    print("  it does not read the competition; the deadline is whatever the entry names, so an")
    print("    entry naming the WRONG deadline and computing correctly from it passes here.")
    print("    #67 is the arm that checks the deadline against the live board.")

    print()
    print(f"FAILURES: {FAILS}")
    if control:
        ok = FAILS > 0
        print("--control: " + ("FIRES, as it must. OK" if ok else "DID NOT FIRE -- BAD"))
        return 0 if ok else 1
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
