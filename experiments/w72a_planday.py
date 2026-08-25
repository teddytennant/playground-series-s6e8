"""w72a — REGISTER ANY REMAINING SEND DAY, AND PRICE THE SUPPLY THAT HAS TO FILL THEM.

`w48e_order.py` exits 2 for a day that is not in its `ORDERS` dict, so an unregistered window
fails at step 2 of the four-command chain with nothing to fall back on. w70c fixed that for
2026-08-25 with a script hardcoded to that one day. This is the same derivation with the day as
an argument, so the remaining windows cost a flag instead of a file.

    .venv/bin/python experiments/w72a_planday.py --day 2026-08-26
    .venv/bin/python experiments/w72a_planday.py --day 2026-08-26 --rewrite   # only on drift
    .venv/bin/python experiments/w72a_planday.py --audit                      # supply only

THE FILTERS ARE READ FROM THE MODULE THAT OWNS EACH ONE:
  unsent (the LIVE API — see below) · has a stored OOF vector · not in `w48e_order.VETO` ·
  not registered for another day · and NOT one `w26g_send.py` will refuse, which is
  `above_tier_reason(r) is not None AND hijack_risk(pred_lb, tier) >= P_MAX` — both halves.
Family `member` is NO LONGER a blanket exclusion (w87): a member certified below the auto-
selection tier by `w55a_unpriced.json`, or listed in `w48e.CAL_ROWS`, is admissible and sorts
below every ranked row. See `certified_members` and `sort_key`.

⚠⚠ TWO BUGS FOUND IN THIS FILE ON 2026-08-25 (w87), BOTH SILENT, BOTH IN THE FLATTERING
DIRECTION, NEITHER EVER FIRED BY A TEST:
  1. the `sent` set was read from `w23b_sendqueue.csv`, whose `sent` column is False on every
     row, so it was ALWAYS EMPTY. On a stale queue the pool put ten already-sent files at the
     top — a whole day of duplicate sends, which score identically and waste the slot.
  2. the blanket `is_member` skip hid all 30 certified filler rows, leaving 4 sendable files
     for the 30 slots of 08-29..08-31. w54a said 67 sendable and w85c said the SENDER plans
     63; only this script said 4, and this script is the one that registers the day.
`w87a_registrarguard.py` now cross-checks this pool against the sender's own plannable set on
every run, which is the check that would have caught both.

⛔ SUPERSEDED BY w87 — KEPT FOR THE RECORD, DO NOT QUOTE THE NUMBERS. The shortfall below
was real when written and was closed by w85's 25 certified fillers plus the bug fix above; the
live reading on 2026-08-25 is 33 sendable against 30 slots. The three levers still stand.

⚠⚠ THE SUPPLY IS SHORT, AND THIS IS THE FIRST RUN TO MEASURE IT. After the 08-24 send there are
**34 sendable files** against **60 slots** (08-26..08-31, ten a day). The drain covers three
full days and **runs dry on 2026-08-29**. The brief's economics say an unused slot is pure
waste, so the shortfall is a real cost and it is not fixed by ranking the queue better — there
is nothing further down it. The three levers, in order of size:
  1. 19 unsent files sit in `w48e_order.VETO`, which is binding *while `check_selection` reports
     nothing selected*. The selection click dissolves the premise and returns ~2 days of supply.
     That click was already the largest item on the account; it is now also the supply fix.
  2. Build new files. Priced at ~0.03e-6 at pack level (w71 §5) — below every noise floor here.
  3. Send nothing on those days and record why.

⛔ DRIFT DOES NOT DEADLOCK THIS SCRIPT, BY DESIGN. w70c fails and then refuses to write, so a
registered day whose derivation has moved can never be corrected by re-running it — the same
shape as the w70 §10.3 deadlock, one layer up. Here drift is reported loudly and `--rewrite`
resolves it; every other failure still refuses to write, because those mean the derivation is
wrong rather than merely stale.
"""
from __future__ import annotations

import argparse, io, json, os, sys, contextlib

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from stdflag import is_member                                            # noqa: E402

N = 10
FAILURES = 0
DRIFT = False
_API = None          # the live submission list, memoised by `pool` for `_vetoed_sent`


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def artefact(day: str) -> str:
    return os.path.join(HERE, f"w72a_plan_{day}.json")


def _load():
    """Import the send chain quietly, and WITHOUT LETTING IT SEE OUR ARGV.

    ⚠⚠ `w48e_order.py` reads `sys.argv` AT MODULE SCOPE (line ~271): it scans for `--day`,
    adopts the value as the day it is ordering for, and `sys.exit(2)`s if that day is not in
    its `ORDERS`. So merely *importing* it from a script that itself takes `--day` hands it an
    argument meant for us, and it exits 2 — silently, because the pricing report it prints on
    the way is inside the stdout redirect below. This script's whole job is to register a day
    w48e does not know yet, so that collision is guaranteed, not incidental.

    ⛔ THE GENERAL RULE, and it is w70 §10.3's lesson in a second costume: A MODULE THAT READS
    `sys.argv` AT IMPORT IS NOT IMPORTABLE — it is a script wearing a module's clothes. We
    cannot restructure w48e without touching the send chain on a send day, so neutralise argv
    across the import instead, which is local and reversible.
    """
    buf = io.StringIO()
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        with contextlib.redirect_stdout(buf):
            import w26g_send as S
            import w48e_order as W
    finally:
        sys.argv = saved
    return S, W


def sort_key(cv, pred_lb, member: bool):
    """The one ordering used to SELECT a day's ten, and the one GATE 3 re-checks.

    Two populations, and they do not share a scale. A ranked row has a cross-fitted stack CV
    and is ordered on it. A certified member has no CV at all and is ordered on its `w55a`
    point estimate, which is the only quantity it owns. Every ranked row outranks every member
    — a member is a MEASUREMENT, not a candidate, so it fills a slot only after the candidates
    are gone. Returned as a tuple so the two never get compared numerically to each other.
    """
    return (0, float(pred_lb)) if member else (1, float(cv))


def certified_members(W):
    """Members the SEND PATH is willing to send. ⛔ DELEGATED — `w48e_order.certified_members`
    is the single owner of this ruling, because that module already holds `CAL_ROWS`, reads
    `w55a_unpriced.json` to price the rows, and VERIFIES the registered ten before writing.
    When this file re-implemented it, the registrar and the verifier disagreed within the hour:
    w72a registered 08-29..08-31 out of certified fillers and `w48e --write` then refused all
    thirty of them for having no OOF vector (w87). One function, three callers, no drift.

    ✅ A STALE `tier` IN `w55a_unpriced.json` CANNOT HARM US, and it is worth writing down why
    rather than adding a guard that would never fire: the tier is the second-best PUBLIC score
    this account holds, which is monotone non-decreasing, so a certification made below an older
    tier is still below today's. The staleness is conservative in the only direction it moves.
    """
    return W.certified_members()


def pool(S, W, day: str | None):
    """Everything still sendable, best CV first, excluding every OTHER day's registered ten."""
    global _API
    api = _API = S.api_submissions()
    bar, tier = S.hijack_cv_bar(api), S.auto_tier(api)

    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    # ⚠ `w26d_queueprice.csv` is written once a day by `w48e --write`, BEFORE that day's send,
    # so its `sent` column is stale the moment the window drains, and a run that registers a
    # day without re-writing the queue first would plan files that have already gone out.
    # ⚠⚠ THE OLD DEFENCE HERE WAS VACUOUS AND HAD BEEN SINCE THE DAY IT WAS WRITTEN. It read
    # `sent` from `w23b_sendqueue.csv` — a file that CONTAINS ONLY UNSENT ROWS, so every one
    # of its 86 `sent` values is False and the set came back EMPTY. `w72b_dayguard.py`'s own
    # docstring records this exact trap ("the column is a filter that has already been
    # applied, not a flag to test") after its first cut fell into it; w72a, written by the
    # same run, kept it. Demonstrated live (w87): pointed at the pre-w85 queue snapshot, the
    # old code put TEN ALREADY-SENT files at the top of a registrable pool. The authoritative
    # sent list is the live API, which this function has already fetched.
    sent = {r["fileName"].replace(".csv", "") for r in api if r.get("fileName")}

    # ⚠ SKIP OUR OWN DAY (w70c's lesson): once `day` is registered, its order is read back from
    # this script's own artefact, so excluding it here would empty the pool against itself.
    other = set()
    for d, o in W.ORDERS.items():
        if o and d != day:
            other |= set(o)

    cert = certified_members(W)

    ok, refused = [], []
    for r in q.itertuples():
        if r.stem in sent or r.stem in other or r.stem in W.VETO:
            continue
        mem = is_member(r.stem)
        # ⚠⚠ A BLANKET `is_member` SKIP HERE COST 26 OF THE LAST 30 SLOTS (w87). It predates
        # w85, which built 25 raw-member fillers for exactly these days and had them certified
        # below the tier by `w55a` and priced into the queue by `w48e`. The SENDER plans them
        # (w85c G1: 63 files for 60 slots); the REGISTRAR could not see them, and since w48e
        # exits 2 on a day it has no plan for, 08-29..08-31 were unregisterable — 30 slots with
        # 4 candidates. A member is admissible on exactly the sender's terms: certified by
        # `w55a_unpriced.json` or listed in `w48e.CAL_ROWS`, read from those modules, never
        # re-derived here.
        if mem and r.stem not in cert:
            continue
        # A member has NO cross-fitted stack CV and never will — that is what `member` MEANS.
        # NaN CV is disqualifying only for a row that is supposed to have one.
        if pd.isna(r.cv) and not mem:
            continue
        why = S.above_tier_reason(r, bar)
        risk = 1.0 if pd.isna(r.pred_lb) else S.hijack_risk(float(r.pred_lb), tier)
        (refused if (why and risk >= S.P_MAX) else ok).append(
            (r.stem, float(r.cv), risk, why, sort_key(r.cv, r.pred_lb, mem)))
    ok.sort(key=lambda t: t[4], reverse=True)
    refused.sort(key=lambda t: t[4], reverse=True)
    return ok, refused, tier, bar


def audit(S, W) -> None:
    ok, refused, tier, bar = pool(S, W, day=None)
    reg = {d: o for d, o in W.ORDERS.items() if o}
    days = [d for d in sorted(set(pd.date_range("2026-08-26", "2026-08-31").strftime("%Y-%m-%d")))]
    open_days = [d for d in days if d not in reg]
    slots = 10 * len(open_days)
    print("=" * 92)
    print("  SUPPLY AUDIT — what is left to send, against the slots left to fill")
    print("=" * 92)
    print(f"\n  tier {tier}   hijack CV bar {bar:.10f}   P_MAX {S.P_MAX}")
    print(f"  unregistered days 08-26..08-31 : {len(open_days)}  ->  {slots} slots")
    print(f"  SENDABLE files remaining       : {len(ok)}")
    print(f"  refused by the sender's own gate: {len(refused)}")
    print(f"  unsent but VETOED              : {len(set(W.VETO)) - len(_vetoed_sent(W))}"
          f"   (sendable again only if the veto's premise dissolves)")
    short = slots - len(ok)
    print(f"\n  SHORTFALL {short:+d} slot(s).  The drain covers {len(ok) // N} full day(s); "
          f"dry from {open_days[len(ok) // N] if len(ok) // N < len(open_days) else 'never'}.")
    if short > 0:
        print("  ⚠ An unused slot is pure waste under this brief. The shortfall is NOT fixable\n"
              "    by ranking the queue better — there is nothing further down it. See the\n"
              "    module docstring for the three levers; the selection click is the biggest.")
    for s, c, k, _, _ in refused:
        print(f"    refused {s:<28} cv {c:.10f}  hijack {k:.2e}")


def _vetoed_sent(W, api=None):
    """Vetoed stems that have nevertheless been sent. ⚠ Same fix as `pool`: this used to read
    `w23b_sendqueue.csv`, whose `sent` column is False on every row, so it always returned the
    empty set and the audit's "unsent but VETOED" line overstated by however many vetoed files
    had gone out. The live API is the only authority on what was sent."""
    api = api if api is not None else _API
    if api is None:
        S, _ = _load()
        api = S.api_submissions()
    return {r["fileName"].replace(".csv", "") for r in api if r.get("fileName")} & set(W.VETO)


def main() -> None:
    global DRIFT
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--rewrite", action="store_true",
                    help="permit overwriting an artefact whose derivation has DRIFTED")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()

    S, W = _load()
    if a.audit or not a.day:
        audit(S, W)
        if not a.day:
            return
        print()

    day = a.day
    ok, refused, tier, bar = pool(S, W, day)

    print("=" * 92)
    print(f"  w72a  THE REGISTERED TEN FOR {day}")
    print("=" * 92)
    if len(ok) < N:
        fail(f"only {len(ok)} sendable file(s) remain — cannot fill a ten for {day}. "
             f"The queue is dry; see the supply audit.")
        print(f"\n  ⛔ {FAILURES} failure(s) — nothing written.")
        sys.exit(1)

    # SELECT on descending CV, SEND in ascending CV so the day's best goes out LAST and wins any
    # public-score tie under w46b §5's latest-first tiebreak. Same convention as every prior day.
    plan = [s for s, _, _, _, _ in ok[:N]][::-1]

    # GATE 1 — ten distinct, none vetoed, none registered for another day.
    if len(set(plan)) != N:
        fail(f"the plan is not {N} distinct files: {plan}")
    if set(plan) & set(W.VETO):
        fail(f"the plan contains VETOED files: {sorted(set(plan) & set(W.VETO))}")
    for d, o in W.ORDERS.items():
        if o and d != day and set(plan) & set(o):
            fail(f"the plan re-sends files registered for {d}: {sorted(set(plan) & set(o))}")

    # GATE 2 — replay the sender's OWN two-part refusal test over the whole plan.
    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    for r in q[q.stem.isin(plan)].itertuples():
        why = S.above_tier_reason(r, bar)
        risk = S.hijack_risk(float(r.pred_lb), tier)
        if why and risk >= S.P_MAX:
            fail(f"{r.stem} would be REFUSED at the send: {str(why)[:80]} (hijack {risk:.2e})")

    # GATE 3 — ascending send order, on the SAME key `pool` selected with. ⚠ It used to read
    # `q.cv` directly, which is NaN on every certified member and makes `cvs != sorted(cvs)`
    # true for reasons that have nothing to do with the order (NaN != NaN). The key is the one
    # function so the gate cannot drift away from the selection it is checking.
    cert = certified_members(W)
    keys = [sort_key(q[q.stem == s].cv.iloc[0], q[q.stem == s].pred_lb.iloc[0],
                     is_member(s)) for s in plan]
    if keys != sorted(keys):
        fail(f"the plan is not in ascending send order: {keys}")

    # GATE 5 — every member in the plan is certified below the tier by the module that owns the
    # ruling. GATE 2 replays the sender's tier test, but a member's whole licence to be sent is
    # that certification, and a hand-edited plan is the case this catches.
    for s in plan:
        if is_member(s) and s not in cert:
            fail(f"{s} is family `member` and is NOT certified by w55a/CAL_ROWS — it has no "
                 f"stack CV and no certified bound, so nothing licenses sending it")

    # GATE 4 — drift. NOT a hard failure: a stale registration must stay fixable by re-running
    # this script, which is exactly what w70c's refuse-to-write made impossible.
    prev = W.ORDERS.get(day)
    if prev is not None and list(prev) != list(plan):
        DRIFT = True
        print(f"\n  ⚠ DRIFT: the registered {day} order differs from what the pricer now derives.")
        print(f"      registered: {list(prev)}")
        print(f"      derived:    {list(plan)}")
        if not a.rewrite:
            print("      Re-run with --rewrite to adopt the derived list, after reading WHY it moved.")

    print(f"\n  tier {tier}   bar {bar:.10f}   {len(ok)} sendable, {len(refused)} refused")
    print(f"  (SEND order: ascending on the selection key, the day's best LAST; a certified "
          f"member has no CV and sorts below every ranked row)")
    print(f"  {'#':>2} {'stem':<28} {'cv':>14} {'pred_lb':>9} {'hijack':>9}")
    for i, s in enumerate(plan, 1):
        r = q[q.stem == s].iloc[0]
        print(f"  {i:>2} {s:<28} {r.cv:.10f} {r.pred_lb:9.5f} "
              f"{S.hijack_risk(float(r.pred_lb), tier):9.2e}")

    if FAILURES:
        print(f"\n  ⛔ {FAILURES} failure(s) — REFUSING to write. The previous artefact (if any) "
              f"is untouched.")
        sys.exit(1)
    if DRIFT and not a.rewrite:
        print(f"\n  ⛔ drift unresolved — not writing. Use --rewrite.")
        sys.exit(1)
    with open(artefact(day), "w") as f:
        json.dump(dict(day=day, plan=plan, tier=tier, cv_bar=bar,
                       sendable_remaining=len(ok),
                       refused=[dict(stem=s, cv=c, hijack=k, why=str(w)) for s, c, k, w, _ in refused],
                       failures=FAILURES), f, indent=1)
    print(f"\n  wrote {os.path.basename(artefact(day))}   FAILURES {FAILURES}")


if __name__ == "__main__":
    main()
