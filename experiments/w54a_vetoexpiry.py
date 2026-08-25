"""w54a -- THE VETO IS A DEFERRAL, NOT A BLOCK, AND THE ARITHMETIC SAYS WHEN IT EXPIRES.

`w48e_order.py` encodes the veto as `priority = -1`.  `w26g_send.py` then sorts
`["priority", "send_rank", "pred_lb"]` descending on priority -- so a vetoed file sorts to the
BACK of the plan and is otherwise treated exactly like any other row.  There is no filter on
`priority < 0` anywhere on the send path.

That is safe only while the queue is longer than the slots that remain.  It stops being safe the
moment the priority-0 tail drains, because from that point the sender's next-best row IS a vetoed
one and it will send it without comment.  This script computes the date that happens, from the
live submission list and the written queue, rather than asserting it.

Run with no arguments.  Prints a table and exits 1 if the veto expires before the deadline.

⚠⚠ IT READ A STALE QUEUE FOR A DAY AND THE SHORTFALL WAS WRONG (w85, 2026-08-25).
`w26d_queueprice.csv` is written by `w48e_order.py --day D --write`, i.e. it is the queue AS OF
day D.  Run this script after day D's ten have landed and every one of them is still in the CSV
marked unsent, so `len(q)` overstates the queue by ten and the printed "N slot(s) go UNFILLED"
comes back TEN TOO SMALL.  On 2026-08-25 it printed 8; the live number was 18.  Nothing was
malformed and nothing raised -- the same silent-and-flattering shape as the 50-row cap w84 §2
found in `w82a`.  C1 below now cross-checks the CSV against the live API and refuses to print
an arithmetic it cannot trust.
"""
import csv, io, os, subprocess, sys, datetime as dt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
DAILY_CAP = 10
DEADLINE = dt.date(2026, 8, 31)          # last UTC day on which a submission can land


def today_utc():
    return dt.datetime.now(dt.timezone.utc).date()


PAGE = 500   # ⚠ the API defaults to 50 and TRUNCATES SILENTLY -- w26g_send.py already passes an
             # explicit page size for this reason, and the first cut of this script did not.
             # A truncated list undercounts what has been sent and overstates the slack.


def api_rows(comp="playground-series-s6e8"):
    out = subprocess.run(["kaggle", "competitions", "submissions", "-c", comp, "-v",
                          "--page-size", str(PAGE)],
                         capture_output=True, text=True, timeout=300).stdout
    rows = list(csv.reader(io.StringIO(out)))
    if len(rows) < 2:
        raise SystemExit("could not read the submission list")
    if len(rows) - 1 >= PAGE:
        raise SystemExit(f"submission list came back at the page size ({len(rows)-1}); truncated.")
    return rows


def sent_today(comp="playground-series-s6e8"):
    """Count submissions already landed on the current UTC day, from the live API."""
    rows = api_rows(comp)
    hdr = rows[0]
    di = hdr.index("date")
    t = today_utc()
    n = 0
    for r in rows[1:]:
        if len(r) <= di or not r[0].strip().isdigit():
            continue
        try:
            d = dt.datetime.strptime(r[di][:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        if d == t:
            n += 1
    return n


def sent_files(comp="playground-series-s6e8"):
    """Every filename the live API has ever accepted, for the C1 freshness check."""
    rows = api_rows(comp)
    fi = rows[0].index("fileName")
    return {r[fi].strip() for r in rows[1:] if len(r) > fi}


def main():
    t = today_utc()
    q = pd.read_csv(QUEUE)

    # ---- C1: THE QUEUE MUST NOT CONTAIN ANYTHING THE API SAYS IS ALREADY SENT ----------
    # This is the freshness test, not a spelling test: it fails only when the CSV genuinely
    # describes a queue that no longer exists, and the remedy it names is the one command that
    # rebuilds it.  See the note in the docstring.
    stale = sorted(set(q.file) & sent_files()) if "file" in q else []
    if stale:
        print(f"⛔ STALE QUEUE -- {len(stale)} row(s) in {os.path.basename(QUEUE)} have already "
              f"been sent:\n     " + ", ".join(stale[:6])
              + (f" ... and {len(stale)-6} more" if len(stale) > 6 else ""))
        print("   Every number below would be computed on a queue that no longer exists, and the\n"
              "   unfilled-slot count would come out LOW BY EXACTLY THIS MANY. Refusing.\n"
              "   Rebuild first:  .venv/bin/python experiments/w23b_sendqueue.py\n"
              "                   .venv/bin/python experiments/w48e_order.py --day <UTC day> --write")
        return 1
    n_pin = int((q.priority == 1).sum())
    n_tail = int((q.priority == 0).sum())
    n_veto = int((q.priority == -1).sum())

    used = sent_today()
    left_today = max(DAILY_CAP - used, 0)
    days_after_today = (DEADLINE - t).days
    slots = left_today + days_after_today * DAILY_CAP

    # The auto-selection tier: while `check_selection` reports nothing selected, Kaggle picks the
    # account's best TWO submissions by PUBLIC score. So a file is only dangerous if it can reach
    # that tier -- and a file that cannot is a FREE slot, which is the whole basis on which the
    # w37 calibration sends were safe at 0.96813.
    rows = api_rows()
    hdr = rows[0]
    si, fi = hdr.index("publicScore"), hdr.index("fileName")
    sc = []
    for r in rows[1:]:
        if len(r) <= si:
            continue
        try:
            sc.append((float(r[si]), r[fi]))
        except ValueError:
            pass
    sc.sort(reverse=True)
    tier = sc[1][0] if len(sc) > 1 else None

    print(f"UTC today                {t}   deadline {DEADLINE}")
    print(f"sent today               {used} of {DAILY_CAP}  ->  {left_today} slot(s) left today")
    print(f"whole send days after    {days_after_today}  ->  {days_after_today * DAILY_CAP} slots")
    print(f"TOTAL SLOTS REMAINING    {slots}")
    print()
    print(f"queue (plan_day {q.plan_day.iloc[0] if 'plan_day' in q else '?'}): "
          f"{len(q)} unsent = {n_pin} pinned + {n_tail} tail + {n_veto} VETOED")
    print()

    # Walk the sender's own ordering forward, day by day, at the cap.
    order = ([("pin", 1)] * n_pin) + ([("tail", 0)] * n_tail) + ([("veto", -1)] * n_veto)
    day, cursor, first_veto_day, sent_veto = t, 0, None, 0
    budget = left_today
    while cursor < len(order) and day <= DEADLINE:
        take = min(budget, len(order) - cursor)
        for k in range(cursor, cursor + take):
            if order[k][0] == "veto":
                sent_veto += 1
                if first_veto_day is None:
                    first_veto_day = day
        cursor += take
        day += dt.timedelta(days=1)
        budget = DAILY_CAP

    sendable = n_pin + n_tail
    print(f"slack (slots - unsent)   {slots - len(q):+d}")
    _gap = slots - sendable
    print(f"SENDABLE if the veto binds {sendable}  ->  "
          + (f"{_gap} slot(s) go UNFILLED  ⛔ build fillers (w85a)" if _gap > 0
             else f"{-_gap} file(s) spare, every slot fills"))
    if tier is not None:
        print(f"auto-selection tier      {tier:.5f}  (2nd-best public of {len(sc)} scored)")
        print(f"  a filler is SAFE iff its predicted public score is < {tier:.5f}; below the tier "
              f"it\n  cannot be auto-selected, so it costs nothing. That is why w37's 0.96813 "
              f"calibration\n  sends were free. Fill the gap with files BELOW the tier, never by "
              f"retiring a veto.")

    if first_veto_day is None:
        print("\n  OK: the queue outlasts the calendar. No vetoed file is reached before the "
              "deadline.")
        return 0

    print(f"\n  ⛔ WITHOUT A FILTER, THE VETO EXPIRES {first_veto_day} -- a mechanical drain at "
          f"the cap")
    print(f"     reaches the first vetoed file that day and sends {sent_veto} of the {n_veto} "
          f"vetoed files")
    print( "     before the deadline. While `check_selection` reports nothing selected, Kaggle")
    print( "     auto-selects on best PUBLIC score: exactly the exposure the veto exists to "
           "close.")

    # Regression test, not prose. w54 put the filter in; this asserts it is still there, because
    # the failure mode is silent -- a sender without it plans a perfectly normal-looking ten.
    src = open(os.path.join(HERE, "w26g_send.py"), encoding="utf-8").read()
    guard = 'int(getattr(r, "priority", 0)) < 0 and not a.allow_vetoed'
    if guard not in src:
        print("\n  🔴 AND THE FILTER IS GONE FROM w26g_send.py. The date above is LIVE.")
        return 1
    print("\n  ✅ but `w26g_send.py` filters priority < 0 (w54), so the date above is neutralised.")
    print( "     Verify with:  .venv/bin/python experiments/w26g_send.py --n 90   (dry)")
    print(f"     -> it must plan {n_pin + n_tail} files, not {len(q)}, and report {n_veto} "
          f"blocked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
