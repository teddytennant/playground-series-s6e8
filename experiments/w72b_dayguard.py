"""w72b — THE REGISTRATION CHAIN IS SOUND: no file is ever SENT twice, and no day plans a
vetoed file or one that is already gone.

w72a lets any remaining window be registered from an artefact on disk, which turns a
one-file-per-day convention into a directory of JSONs that nothing cross-checks. The cost of a
mistake is a wasted slot: a re-send scores identically to the original (the brief: "resubmitting
an identical file is genuinely pointless") and the sender will not notice, because its own gates
are per-file, not per-plan.

⚠ w48e's own `bad = set(ORDER) & set(VETO)` assert only ever looks at ONE day — the day it was
invoked for. Nothing looked ACROSS days until this guard.

⚠⚠ THE TWO THINGS THE FIRST VERSION OF THIS GUARD GOT WRONG, both of which made it fire on a
sound chain. Recorded because each is a fact about the workspace, not a typo:

  1. **`w23b_sendqueue.csv` CONTAINS ONLY UNSENT FILES** — every row reads `sent = False`. The
     column is a filter that has already been applied, not a flag to test. Taking the sent set
     from it yields the EMPTY SET, which silently vacates every check that uses it while the
     checks that invert it fire on all 121 sent files. The authoritative sent list is the live
     API, via `w26g_send.api_submissions()`.
  2. **CROSS-DAY REGISTRATION OVERLAP IS LEGITIMATE.** `w38_ad202std_rescale` and
     `w40_ad211std_rescale` are registered for BOTH 08-23 and 08-24: the sender REFUSED them on
     08-23 at the P_MAX hijack gate, so they carried over and went out on 08-24 — exactly once
     each. A registration is an intention; only a send is an event. ⛔ So the guard is on the
     SENT history, and registration overlap is reported as information, not as a failure.

Also: `w48e_order.CAL_ROWS` are member-family BY DESIGN (a calibration file has no stack CV), so
the member check must exempt them or it fires on `w48_cal_hboyang_mix` every run.

⚠ WIDENED 2026-08-25 (w87). From 08-29 the supply IS members: w85's 25 fillers, each certified
below the auto-selection tier by `w55a_unpriced.json`. Check 4 exempted only CAL_ROWS and so
failed all three of those days the moment they were registered — an outdated rule, not a real
fault. It now reads `w72a.certified_members`, the same function the registrar admits on. A
member in neither set still fails, which is the case the check is for.

    .venv/bin/python experiments/w72b_dayguard.py     # rc=0 sound, rc=1 with the reason

⛔ Read `$?` from a command substitution or a redirect, never after a pipe (RESEARCH §9.1).
"""
from __future__ import annotations

import io, os, sys, contextlib
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from stdflag import is_member                                            # noqa: E402

FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def main() -> None:
    buf = io.StringIO()
    saved = sys.argv
    try:                                     # w48e reads sys.argv at import — see w72a._load
        sys.argv = [saved[0]]
        with contextlib.redirect_stdout(buf):
            import w48e_order as W
            import w26g_send as S
    finally:
        sys.argv = saved

    orders = {d: list(o) for d, o in W.ORDERS.items() if o}
    # ⚠ CAL_ROWS ALONE IS NO LONGER THE WHOLE LICENCE (w87). w85 built 25 raw-member fillers
    # and had every one certified below the auto-selection tier by `w55a_unpriced.json`; from
    # 08-29 they ARE the supply, and a member certified there is exactly as admissible as a
    # CAL_ROW. Read the ruling from `w72a.certified_members`, which is the same function the
    # registrar admits on, so this guard and the thing it guards cannot drift apart.
    # ⛔ This is a WIDENING, not a removal: a member in NEITHER set still fails below, which is
    # the case that matters — an uncertified member has no bound and could be auto-selected.
    import w72a_planday as P                                             # noqa: E402
    cal = P.certified_members(W)

    # THE AUTHORITATIVE SENT HISTORY — the live API, not the queue CSV (see the docstring).
    api = S.api_submissions()
    hist = Counter()
    for r in api:
        f = r.get("fileName") or r.get("file_name") or ""
        if f:
            hist[f[:-4] if f.endswith(".csv") else f] += 1
    unsent = set(pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
                 .file.str.replace(".csv", "", regex=False))

    print("=" * 88)
    print("  w72b  REGISTRATION-CHAIN GUARD")
    print("=" * 88)
    print(f"\n  {len(orders)} day(s) registered: {', '.join(sorted(orders))}")
    print(f"  {sum(hist.values())} submission(s) on record, {len(hist)} distinct file(s); "
          f"{len(unsent)} unsent\n")

    # 1. ten distinct files per day.
    for d, o in sorted(orders.items()):
        if len(o) != 10 or len(set(o)) != 10:
            fail(f"{d} registers {len(o)} files, {len(set(o))} distinct — expected ten distinct")

    # 2. ⛔ THE ONE THAT MATTERS: no file has been SENT more than once.
    for stem, n in sorted(hist.items()):
        if n > 1:
            fail(f"{stem} has been SENT {n} times — every re-send scores identically and "
                 f"burned a slot")

    # 3. no PENDING day plans a file that is already gone. A past day's ten ARE sent, so only
    #    days after today can fail this.
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    for d, o in sorted(orders.items()):
        if d <= today:
            continue
        clash = sorted(s for s in o if s in hist)
        if clash:
            fail(f"{d} plans already-sent file(s): {clash}")
        gone = sorted(s for s in o if s not in unsent)
        if gone:
            fail(f"{d} plans file(s) that are neither unsent nor sent — absent from the "
                 f"queue entirely: {gone}")

    # 4. no vetoed file on any day; no member-family file that nothing has certified.
    for d, o in sorted(orders.items()):
        if set(o) & set(W.VETO):
            fail(f"{d} plans VETOED file(s): {sorted(set(o) & set(W.VETO))}")
        mem = sorted(s for s in o if is_member(s) and s not in cal)
        if mem:
            fail(f"{d} plans member-family file(s) certified by neither w48e.CAL_ROWS nor "
                 f"w55a_unpriced.json — no stack CV and no bound below the tier: {mem}")

    for d, o in sorted(orders.items()):
        n_sent = sum(1 for s in o if s in hist)
        print(f"    {d}  {len(o):>2} registered  {n_sent:>2} of them sent  "
              f"{'past' if d <= today else 'pending'}")

    # INFORMATION, NOT A FAILURE — see docstring note 2.
    reg = Counter(s for o in orders.values() for s in o)
    carry = {s: sorted(d for d, o in orders.items() if s in o)
             for s, n in reg.items() if n > 1}
    if carry:
        print(f"\n  ⓘ {len(carry)} file(s) registered on more than one day — legitimate when the "
              f"sender refused\n    them on the earlier day and they carried over. Sent counts "
              f"are checked above.")
        for s, ds in sorted(carry.items()):
            print(f"      {s:<28} {ds}  sent {hist.get(s, 0)}x")

    print(f"\n  {sum(len(o) for o in orders.values())} registered slot(s), "
          f"{len(reg)} distinct file(s).   FAILURES {FAILURES}")
    if FAILURES:
        sys.exit(1)
    print("  ✅ nothing has been sent twice; no pending day plans a sent, missing or vetoed file.")


if __name__ == "__main__":
    main()
