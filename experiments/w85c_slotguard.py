"""w85c — STANDING GUARD: every remaining send slot must have a file to put in it.

WHAT CLAIM THIS COVERS, AND WHY IT NEEDED ITS OWN GUARD (w84 §3's lesson, applied).
The send path is densely guarded -- `w54a` prices the veto's expiry, `w55a` certifies the
unpriceable rows below the tier, `w56b` checks eligibility, `w59b`/`w62b` the hijack bar,
`w74b` the click price, `w84a` the argmax behind the deadline pick. Not one of them asserts
the plainest thing the brief asks for:

    submissions here never evict each other and the public board shows best-of-all,
    so an unused daily slot is pure waste -- use all ten, every day.

On 2026-08-25 there were 60 slots left and 42 sendable files. **18 slots were going to go
unused**, which is nearly two full days of the competition, and nothing said so. `w54a` came
closest and said 8, because it read a queue CSV written for a day whose ten had already landed
(fixed there this run, C1). `w85a_fillers.py` closed the gap; this file is what notices if it
re-opens -- files get vetoed, days pass, and the count only moves one way.

THE INVARIANT.  sendable >= slots remaining, where
    slots     = (cap - sent today) + whole days left before the deadline * cap   [live API]
    sendable  = rows in the written queue that `w26g_send.py` would actually plan  [live dry run]
The second is taken from the SENDER ITSELF, not recomputed here. A guard that re-derives the
number it is guarding tests its own arithmetic; this one runs `w26g_send.py --n <slots>` and
reads the plan length, so it is testing the code that will do the sending.

⚠ IT DOES NOT CHECK, AND MUST NOT CHECK, THAT THE FILES ARE ANY GOOD. A filler's whole
justification is that it is BELOW the auto-selection tier and therefore free (w54's rule,
certified per row by `w55a`). Quality is the deadline pick's problem and the deadline pick is
on CV. G3 checks the tier, not the CV.

CONTROLS (w72 §5.3: a control that can only fail is not a control).
  G1 +  sendable >= slots, from the live API and a live sender dry run.
  G2 +- EXERCISED BOTH WAYS: G1 is re-evaluated against the pre-w85 queue snapshot
        (`w26d_queueprice.csv.bak.w85`, 42 sendable) and MUST come out short. If a stale
        snapshot also passes, G1 is inert and this file says so instead of passing.
  G3 +  every row the sender would plan is either priced below the tier or certified below it
        by `w55a_unpriced.json` -- so closing the slot gap cannot have opened a tier gap.
  G4 +  `w54a_vetoexpiry.py`'s C1 freshness check is exercised on the same stale snapshot and
        must refuse it. That is the defect this run found; without G4 nothing re-tests the fix.

    .venv/bin/python experiments/w85c_slotguard.py          # 0 = ok, 1 = a guard failed
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads
COMP = "playground-series-s6e8"
QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
STALE = os.path.join(HERE, "w26d_queueprice.csv.bak.w85")
UNPRICED = os.path.join(HERE, "w55a_unpriced.json")
SENDER = os.path.join(HERE, "w26g_send.py")
W54A = os.path.join(HERE, "w54a_vetoexpiry.py")

DAILY_CAP = 10
DEADLINE = dt.date(2026, 8, 31)
TIER = 0.97119
PAGE = 200   # w140: AT the server cap, not above it. The endpoint caps a page at 200 and
             # returns a next_page_token the CLI never prints, so at 500 this file's own
             # `len(rows) >= PAGE` read `200 >= 500 -> False` and could NEVER fire.
             # Measured in w140c_pagetruth.py (T1/T2/T5).

PLAN_RE = re.compile(r"^plan, (\d+) file\(s\):", re.M)
# The sender's own words for "I would plan nothing", printed instead of a plan line.
DRAINED = "nothing to send: the queue is drained of everything sendable."


def api_rows():
    # ⚠ w140: paginated. This was a capped CLI read whose `>= PAGE` check could not fire
    # above the server's 200-row page cap. Returned in the CLI's own list-of-lists shape
    # (header first) so callers indexing rows[0] are untouched.
    rows = kaggle_list.submissions(COMP)
    cols = ["ref", "fileName", "date", "description", "status", "publicScore", "privateScore"]
    return [cols] + [[r[c] for c in cols] for r in rows]


def slots_left() -> tuple[int, int, int]:
    rows = api_rows()
    di = rows[0].index("date")
    today = dt.datetime.now(dt.timezone.utc).date()
    used = sum(1 for r in rows[1:]
               if len(r) > di and r[di][:10] == today.isoformat())
    days = (DEADLINE - today).days
    return max(DAILY_CAP - used, 0) + days * DAILY_CAP, used, days


def plan_len(n: int) -> int:
    """Ask the SENDER how many files it would actually plan. Dry run: it never sends.

    ⚠⚠ A DRAINED QUEUE PLANS ZERO AND PRINTS NO PLAN LINE (w112, 2026-08-29). `w26g_send.py`
    has a branch that prints DRAINED and returns before `plan, N file(s):` is ever emitted, so
    on a queue with nothing sendable this helper used to raise. Not a hypothetical: G2 hands it
    the pre-w85 control queue, and by 181 sends **all 48 of that queue's files were sent**,
    so the control stopped being readable at the exact moment it became maximally true.
    🎯 THE NEGATIVE CONTROL SUCCEEDED SO COMPLETELY THAT THE INSTRUMENT READING IT BROKE --
    G2 wants "the stale queue is SHORT", and a drained queue is short by every slot there is.
    ⛔ Zero is READ from the sender's own sentence, not inferred from a missing match: anything
    else unparseable still raises, because "refusing to guess" is the right default (w104).
    """
    out = subprocess.run([sys.executable, SENDER, "--n", str(n)],
                         capture_output=True, text=True, timeout=1800).stdout
    m = PLAN_RE.search(out)
    if not m:
        if DRAINED in out:
            return 0
        raise SystemExit("could not parse the sender's plan length -- refusing to guess")
    return int(m.group(1))


def main() -> int:
    fails = []
    slots, used, days = slots_left()
    print(f"UTC today {dt.datetime.now(dt.timezone.utc).date()}   deadline {DEADLINE}")
    print(f"sent today {used}/{DAILY_CAP}   whole days after today {days}   "
          f"SLOTS REMAINING {slots}\n")

    # ---- G1 ---------------------------------------------------------------------------
    # ⚠ `--n slots` is the exact invariant but it CANNOT report headroom: it returns `slots`
    # whether the queue has one file to spare or forty. Probe with a wide `--n` so the printed
    # spare count is the real one -- an invariant that is only ever "exactly met" gives no
    # warning before it breaks.
    total = plan_len(slots + 200)
    n = min(total, slots)
    gap = slots - n
    print(f"G1  sender can plan {total} file(s) against {slots} slot(s)  ->  "
          + (f"⛔ {gap} UNFILLED" if gap > 0 else f"OK, {total - slots} spare"))
    if gap > 0:
        fails.append(f"G1: {gap} send slot(s) have no file -- build fillers with w85a_fillers.py")

    # ---- G2: the same test against the pre-w85 queue MUST come out short ---------------
    if not os.path.exists(STALE):
        fails.append(f"G2: {os.path.basename(STALE)} is missing -- G1 cannot be exercised, so "
                     f"a passing G1 means nothing")
    else:
        keep = tempfile.mktemp(suffix=".csv")
        shutil.copy2(QUEUE, keep)
        try:
            shutil.copy2(STALE, QUEUE)
            n_stale = plan_len(slots)
        finally:
            shutil.copy2(keep, QUEUE)
            os.remove(keep)
        print(f"G2  same test on the pre-w85 queue: plans {n_stale} against {slots}  ->  "
              + (f"OK, short by {slots - n_stale} as expected"
                 if n_stale < slots else "⛔ ALSO PASSES"))
        if n_stale >= slots:
            fails.append("G2: the pre-w85 queue also fills every slot, so G1 is inert")

    # ---- G3: nothing the sender would plan sits above the tier -------------------------
    q = pd.read_csv(QUEUE)
    reg = json.load(open(UNPRICED))["rows"]
    over, uncert = [], []
    for r in q.itertuples():
        if getattr(r, "priority", 0) < 0:
            continue
        p = getattr(r, "pred_lb", float("nan"))
        if pd.notna(p):
            if p >= TIER:
                over.append((r.file, p))
        else:
            row = reg.get(r.file)
            if not (row and row.get("safe")):
                uncert.append(r.file)
    print(f"G3  {len(q)} queue row(s): {len(over)} priced at/above the {TIER} tier, "
          f"{len(uncert)} unpriced and uncertified")
    for f, p in over:
        print(f"      above tier {f} pred_lb {p:.6f}  (w58 blocks it in the sender)")
    if uncert:
        fails.append(f"G3: {len(uncert)} unpriced row(s) not certified by w55a: "
                     + ", ".join(uncert[:4]))

    # ---- G4: w54a's freshness refusal, exercised -------------------------------------
    if os.path.exists(STALE):
        keep = tempfile.mktemp(suffix=".csv")
        shutil.copy2(QUEUE, keep)
        try:
            shutil.copy2(STALE, QUEUE)
            rc_stale = subprocess.run([sys.executable, W54A],
                                      capture_output=True, text=True, timeout=900).returncode
        finally:
            shutil.copy2(keep, QUEUE)
            os.remove(keep)
        rc_live = subprocess.run([sys.executable, W54A],
                                 capture_output=True, text=True, timeout=900).returncode
        print(f"G4  w54a on the stale queue rc={rc_stale} (want 1), on the live queue "
              f"rc={rc_live} (want 0)")
        if rc_stale != 1:
            fails.append("G4: w54a accepted a queue whose rows are already sent -- its C1 "
                         "freshness check is not binding")
        if rc_live != 0:
            fails.append(f"G4: w54a rejects the LIVE queue (rc={rc_live})")

    for f in fails:
        print(f"  FAIL  {f}")
    print(f"\nFAILURES {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
