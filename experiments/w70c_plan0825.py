"""w70c — THE REGISTERED 2026-08-25 TEN, DERIVED FROM THE PRICER AND NEVER HAND-TYPED.

`w48e_order.py` refuses to write for a day that is not in its `ORDERS` dict and exits 2. Only
2026-08-22/23/24 were registered, so the 08-25 window would have failed at step 2 of the
four-command send chain with nothing built to fall back on. This registers it.

THE CONVENTION IT FOLLOWS (w63a, imported by w48e as `PLAN_0824`): the day's ten are DERIVED
here from the priced queue under stated filters and IMPORTED by `w48e_order.py`. A second
hand-typed copy in the order file is how the registered list and the priced list drift apart
without either noticing (w62 section 1).

THE FILTERS, all of them:
  unsent · has a stored OOF vector (so it has a CV to rank and defend on) · not in
  `w48e_order.VETO` · `stdflag.family() != "member"` · not already in the 08-24 ten ·
  AND NOT ONE THE SENDER WILL REFUSE (below).

⚠⚠ THE FILTER THAT IS NEW HERE, AND WHY A CV RANKING ALONE WOULD HAVE WASTED A SLOT.
`w26g_send.py:490` skips a row iff `above_tier_reason(r) is not None` AND
`hijack_risk(pred_lb, TIER) >= P_MAX`. Both halves matter and the previous plans only ever had
to satisfy the second, because no eligible file had ever tripped the first. ARM 208 does: w69
keyed `w69_ad208` in `check_selection.WANTED_INELIGIBLE`, and `above_tier_reason` prefix-matches
that dict, so ALL SEVEN ARM 208 files return a reason. Six of them survive anyway because their
hijack risk is under 2%; `w69_ad208stdcorr` — THE HIGHEST-CV ELIGIBLE UNSENT FILE IN THE QUEUE —
reads ~4.1e-2 and would be REFUSED at the send. Planning it is planning an unsent slot.

⛔ SO IT IS OMITTED, AND THE OMISSION IS CONDITIONAL, NOT PERMANENT. `w69_ad208`'s entry in
`WANTED_INELIGIBLE` is lifted by ARM 208's own (208 - 199) matched control landing inside ±4e-6
on all four criterion bases — that is w69's registered P8, computed by `w69a_factorial.py`. If a
later run lifts it, `above_tier_reason` returns None for every ARM 208 file, the first half of
the sender's test fails, `w69_ad208stdcorr` becomes sendable, and THIS DAY SHOULD BE RE-DERIVED
by re-running this script. The list below is the correct one while the bar stands.

    .venv/bin/python experiments/w70c_plan0825.py            # derive, verify, write the artefact
"""
from __future__ import annotations

import json, os, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from stdflag import is_member                                        # noqa: E402

DAY = "2026-08-25"
N = 10
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def derive(verbose: bool = True):
    """Return (plan, blocked, df). Imports the sender and the order file for their OWN
    constants -- P_MAX, TIER, hijack_risk, above_tier_reason and VETO are read, never restated."""
    import w26g_send as S
    import w48e_order as W

    # ⚠ `hijack_cv_bar` and `auto_tier` take the LIVE API SUBMISSION rows (dicts with a
    # `publicScore`), NOT the priced-queue rows. Passing the queue in raises deep inside
    # `live_tier1`. The tier and the bar are properties of THE BOARD, not of the queue.
    api = S.api_submissions()
    bar = S.hijack_cv_bar(api)
    tier = S.auto_tier(api)

    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q.file.str.replace(".csv", "", regex=False)

    un = q[~q.sent & q.cv.notna()].copy()
    elig = un[~un.stem.isin(W.VETO) & ~un.stem.map(is_member)].copy()
    elig = elig[~elig.stem.isin(set(W.ORDER_0824))]
    elig = elig.sort_values("cv", ascending=False)

    plan, blocked = [], []
    for r in elig.itertuples():
        why = S.above_tier_reason(r, bar)
        risk = 1.0 if pd.isna(getattr(r, "pred_lb", float("nan"))) \
            else S.hijack_risk(float(r.pred_lb), tier)
        if why and risk >= S.P_MAX:
            blocked.append((r.stem, float(r.cv), risk, why))
            continue
        plan.append(r.stem)
        if len(plan) == N:
            break

    # ⚠ SEND ORDER IS LIST ORDER -- `w48e_order.py:500` writes `send_rank = i` straight off the
    # enumeration and the sender drains in that order. The ten are SELECTED on descending CV and
    # SENT in ASCENDING CV, so the best of the day goes out LAST and wins any public-score tie
    # under w46b section 5's latest-first tiebreak. Free, and it is the convention every
    # registered day so far has followed by hand.
    plan.reverse()

    if verbose:
        print("=" * 96)
        print(f"w70c  THE REGISTERED TEN FOR {DAY}")
        print("=" * 96)
        print(f"\n  tier {tier}   hijack CV bar {bar}   P_MAX {S.P_MAX}")
        print(f"  {len(un)} unsent with a CV; {len(elig)} eligible after VETO / member / "
              f"the 08-24 ten.\n")
        print(f"  (listed in SEND order: ascending CV, best of the day LAST)")
        print(f"  {'#':>2} {'stem':<28} {'cv':>14} {'pred_lb':>9} {'hijack':>9}")
        for i, s in enumerate(plan, 1):
            r = elig[elig.stem == s].iloc[0]
            print(f"  {i:>2} {s:<28} {r.cv:.10f} {r.pred_lb:9.5f} "
                  f"{S.hijack_risk(float(r.pred_lb), tier):9.2e}")
        print(f"\n  ⛔ REFUSED BY THE SENDER and therefore not planned ({len(blocked)}):")
        for s, cv, risk, why in blocked:
            print(f"     {s:<28} cv {cv:.10f}  hijack {risk:.2e} >= {S.P_MAX}\n"
                  f"       {why[:110]}")
    return plan, blocked, elig


def main() -> None:
    plan, blocked, elig = derive()

    import w26g_send as S
    import w48e_order as W
    api = S.api_submissions()
    tier = S.auto_tier(api)

    # GATE 1 — ten distinct files, none vetoed, none already registered for another day.
    if len(plan) != N or len(set(plan)) != N:
        fail(f"the plan is not {N} distinct files: {plan}")
    if set(plan) & set(W.VETO):
        fail(f"the plan contains VETOED files: {sorted(set(plan) & set(W.VETO))}")
    # ⚠ SKIP OUR OWN DAY. `w48e.ORDERS["2026-08-25"]` is read back FROM THIS SCRIPT'S ARTEFACT,
    # so once the day is wired in, comparing the plan against every registered day compares it
    # against ITSELF and fires on all ten. The gate became self-referential the moment its own
    # output was plugged into the thing it checks. It is still worth running against the OTHER
    # days, which is the collision it was written for.
    for d, o in W.ORDERS.items():
        if d == DAY:
            continue
        if set(plan) & set(o):
            fail(f"the plan re-sends files already registered for {d}: {sorted(set(plan) & set(o))}")
    # ...and the self-comparison it replaces is still worth making, as a CONSISTENCY check:
    # if 08-25 is already registered, what is registered must EQUAL what we just derived.
    prev = W.ORDERS.get(DAY)
    if prev is not None and list(prev) != list(plan):
        fail(f"the registered {DAY} order has DRIFTED from what the pricer now derives.\n"
             f"       registered: {list(prev)}\n       derived:    {list(plan)}\n"
             f"       Re-run this script to rewrite the artefact, then re-check.")

    # GATE 2 — every planned file must actually survive the sender's own two-part test. This is
    # the check that would have caught `w69_ad208stdcorr`, and it is run over the WHOLE plan
    # rather than the one file this run happened to notice.
    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    bar = S.hijack_cv_bar(api)
    for r in q[q.stem.isin(plan)].itertuples():
        why = S.above_tier_reason(r, bar)
        risk = S.hijack_risk(float(r.pred_lb), tier)
        if why and risk >= S.P_MAX:
            fail(f"{r.stem} would be REFUSED at the send: {why[:80]} (hijack {risk:.2e})")

    # GATE 3 — the omitted file must really be the one the docstring names, and really be the
    # highest-CV eligible unsent file. A vacuity check on the whole rationale above.
    if not blocked:
        fail("no file was blocked — the docstring's rationale is vacuous; re-read it before "
             "trusting this list")
    elif blocked[0][0] != "w69_ad208stdcorr":
        fail(f"the highest-CV blocked file is {blocked[0][0]}, not w69_ad208stdcorr as "
             f"registered — the situation changed and this file must be re-read")
    # GATE 4 -- the plan must be in ASCENDING CV so the best file is sent last.
    cvs = [float(q[q.stem == s].cv.iloc[0]) for s in plan]
    if cvs != sorted(cvs):
        fail(f"the plan is not in ascending-CV send order: {cvs}")

    if blocked and blocked[0][1] <= max(q[q.stem.isin(plan)].cv):
        fail("the blocked file is NOT above the plan — omitting it costs nothing and the "
             "conditional re-derivation note is misleading")

    # ⛔ NEVER PERSIST A PLAN THAT FAILED ITS OWN GATES. A written-out failure is what deadlocked
    # this pair once already: w48e read it, refused, and w70c could not re-run to fix it because
    # w70c imports w48e. Leaving the previous artefact untouched is safe — w48e re-runs every one
    # of its own checks (veto, ten-distinct, md5, CV-reproduces) at send time.
    if FAILURES:
        print(f"\n  ⛔ {FAILURES} failure(s) — REFUSING to write the artefact. The previously "
              f"written one\n     (if any) is untouched; 2026-08-25 stays registered on it or "
              f"stays unregistered.")
        sys.exit(1)
    with open(os.path.join(HERE, "w70c_plan0825.json"), "w") as f:
        json.dump(dict(day=DAY, plan=plan, tier=tier, cv_bar=bar,
                       blocked=[dict(stem=s, cv=c, hijack=r, why=w) for s, c, r, w in blocked],
                       failures=FAILURES), f, indent=1)
    print(f"\n  wrote w70c_plan0825.json   FAILURES {FAILURES}")
    if FAILURES:
        sys.exit(1)


# ⚠ READ FROM THE ARTEFACT BY `w48e_order.py`, NOT IMPORTED AS A CONSTANT. w63a exposes
# `PLAN_0824` as a module-level constant and w48e imports it; the same shape is impossible here
# because THIS file has to import `w48e_order` for its VETO and `w26g_send` for its gates, and
# a module-level constant would make that a circular import that breaks whichever side is loaded
# first. w48e already reads the 08-22 order out of `w47b_probe.json`, so reading a JSON is the
# established second form of the same convention: the list is still DERIVED, never hand-typed,
# and w48e asserts the artefact's `failures == 0` and its `day` before using it.

if __name__ == "__main__":
    main()
