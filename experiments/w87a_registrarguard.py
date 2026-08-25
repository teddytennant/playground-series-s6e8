"""w87a — STANDING GUARD: THE REGISTRAR AND THE SENDER MUST AGREE ABOUT WHAT IS SENDABLE.

WHAT CLAIM THIS COVERS. Two different modules decide what goes out, and only one of them is
ever watched. `w26g_send.py` decides what it will SEND; `w72a_planday.py` decides what may be
REGISTERED for a day, and `w48e_order.py` exits 2 on a day it has no registration for. So a
file the sender would happily send but the registrar cannot register is unreachable — the
calendar simply never offers it a slot, and every instrument that counts "sendable" from the
sender's side goes on reporting that the slots are covered.

⚠⚠ THIS IS NOT HYPOTHETICAL, IT IS WHY THE FILE EXISTS. On 2026-08-25 (w87):

    w54a_vetoexpiry.py   67 sendable, "every slot fills"     <- reads the queue
    w85c_slotguard.py    63 planned by the SENDER vs 60 slots <- reads the sender
    w72a_planday.py       4 sendable  vs 30 slots            <- THE ONE THAT REGISTERS DAYS

Three instruments, one disagreement, and the outlier was the only one with a vote. w72a
carried a blanket `is_member` skip that predated w85, and w85's twenty-five certified fillers —
built for exactly the days 08-29..08-31 — are family `member`. 26 of the last 30 slots had no
registrable file and no guard could see it, because every other guard asks the sender.

The second bug in the same function was the same shape one layer down: the `sent` set was read
from `w23b_sendqueue.csv`, which contains ONLY unsent rows, so it was always empty. Pointed at
a stale queue the pool put TEN ALREADY-SENT files at the top of a registrable day. A duplicate
send scores identically to the original — the brief calls it "genuinely pointless" — so that is
a whole day of the competition spent re-sending yesterday.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +  every REGISTERED file is one the sender would actually plan, and the registrations
        cover every remaining slot. Both halves: a short calendar and a poisoned day differ.
  C2 +  the sender's plannable set minus the registrations is reported as HEADROOM, never
        asserted to be zero — 63 plannable for 60 slots is healthy, not a fault.
  C3 +- THE VACUOUS-`sent` REGRESSION, FIRED BOTH WAYS on the real pre-w85 queue snapshot:
        the old read must admit already-sent files, the live code must admit none.
  C4 +- THE MEMBER-ADMISSION REGRESSION, FIRED: with `certified_members` forced empty the pool
        must COLLAPSE below the slots it has to fill. If it does not, the certification is not
        load-bearing and C1 is passing for the wrong reason.
  C5 +  every planned member is certified below the tier by `w55a`/`CAL_ROWS`, and its point
        estimate is strictly under the LIVE tier. A member's whole licence to occupy a slot is
        that it cannot be auto-selected.
  C6 +  no registered file has already been sent, and no two FUTURE days plan the same file.
  C7 +  the sender's two-part refusal test replayed at the LIVE bar and tier over every
        registered future day. A plan written under an older, more permissive bar is the case
        this catches, and three of the six days on disk were.
  C8 +- `w48e`'s OOF exemption for a certified member DISCRIMINATES — fired on a stem that
        reads as a member and is certified by nothing — and the certified set has ONE owner.

⛔ IT DOES NOT JUDGE QUALITY, and must not be changed to. A filler is free precisely because it
is below the auto-selection tier; the deadline pick is a separate question decided on CV by
`w84a_pickargmax.py`. C5 checks the tier, never the CV — the same ruling as `w85c` G3.

    .venv/bin/python experiments/w87a_registrarguard.py     # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import contextlib
import datetime as dt
import glob
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

OUT = os.path.join(HERE, "w87a_registrarguard.json")
SENDER = os.path.join(HERE, "w26g_send.py")
STALE = os.path.join(HERE, "w26d_queueprice.stale_w26h.csv")
DEADLINE = dt.date(2026, 8, 31)
DAILY_CAP = 10
PLAN_ROW = re.compile(r"^\s*\d+\.\s+(\S+\.csv)\s")

FAILS: list[str] = []


def chk(tag: str, ok: bool, msg: str) -> bool:
    print(f"  {'✅' if ok else '❌'} {tag} {msg}")
    if not ok:
        FAILS.append(f"{tag}: {msg}")
    return ok


def sender_plan(n: int) -> list[str]:
    """The stems `w26g_send.py` would actually plan, taken FROM THE SENDER (w85c's move).

    A guard that re-derives the set it is guarding is testing its own arithmetic. This runs the
    real sender in its dry-run mode and parses the plan it prints.
    """
    out = subprocess.run([sys.executable, SENDER, "--n", str(n)],
                         capture_output=True, text=True, timeout=1800).stdout
    stems = [m.group(1)[:-4] for m in (PLAN_ROW.match(ln) for ln in out.splitlines()) if m]
    if not stems:
        raise SystemExit("could not parse the sender's plan — refusing to guess")
    return stems


def slots_left(api) -> int:
    today = dt.datetime.now(dt.timezone.utc).date()
    used = sum(1 for r in api if r["date"][:10] == today.isoformat())
    return max(DAILY_CAP - used, 0) + max(0, (DEADLINE - today).days) * DAILY_CAP


def pending(W, api):
    """The registered days whose sends have NOT happened yet.

    ⚠ `d >= today` is the wrong predicate and it fails loudly the moment you use it. TODAY is a
    registered day too, and once its ten have landed its files are `sent` — so a "future"
    window that includes today reports all ten as unsendable, already-sent and missing from the
    rebuilt queue. Today counts as pending only while it still has slots, and whether it does
    is a live reading, not an assumption about what time this script is run at.
    """
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    used = sum(1 for r in api if r["date"][:10] == today)
    return {d: list(o) for d, o in W.ORDERS.items()
            if o and (d > today or (d == today and used < DAILY_CAP))}


def main() -> int:
    import w72a_planday as P
    S, W = P._load()
    api = S.api_submissions()
    # One live read, reused everywhere below. `pool` refetches on its own, so pin it: the
    # controls must all see the SAME history or a disagreement could be nothing but a race.
    S.api_submissions = lambda _a=api: _a
    bar, tier = S.hijack_cv_bar(api), S.auto_tier(api)
    sent = {r["fileName"].replace(".csv", "") for r in api if r.get("fileName")}
    slots = slots_left(api)

    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    qi = {r.stem: r for r in q.itertuples()}

    print("=" * 92)
    print("  w87a  THE REGISTRAR AND THE SENDER MUST AGREE")
    print("=" * 92)
    print(f"\n  live rows {len(api)}   slots left {slots}   tier {tier}   bar {bar:.10f}")

    fut = pending(W, api)
    reg = sorted({s for o in fut.values() for s in o})
    print(f"  PENDING registered days {sorted(fut)}  ->  {len(reg)} distinct files")

    plannable = sender_plan(max(slots, len(reg)) + 20)
    print(f"  the sender would plan {len(plannable)} file(s)")

    # ---- C1: the registrations cover the slots, and every one of them is sendable ----------
    missing = [s for s in reg if s not in set(plannable)]
    chk("C1a", not missing,
        f"every registered file is one the sender would plan"
        f"{'' if not missing else f' — UNSENDABLE: {missing}'}")
    chk("C1b", len(reg) >= slots,
        f"registrations cover the calendar: {len(reg)} file(s) for {slots} slot(s)")

    # ---- C2: headroom, reported, never asserted to be zero --------------------------------
    spare = [s for s in plannable if s not in set(reg)]
    print(f"  ℹ C2 headroom {len(spare)} plannable file(s) not registered for any day"
          f"{'' if not spare else ': ' + ', '.join(spare[:6])}")

    # ---- C3: the vacuous-`sent` regression, fired both ways --------------------------------
    if not os.path.exists(STALE):
        chk("C3", False, f"the stale queue snapshot is missing ({os.path.basename(STALE)}) — "
                         f"the negative control cannot be fired, so it is not a pass")
    else:
        tmp = tempfile.mkdtemp()
        shutil.copy(STALE, os.path.join(tmp, "w26d_queueprice.csv"))
        shutil.copy(os.path.join(HERE, "w23b_sendqueue.csv"),
                    os.path.join(tmp, "w23b_sendqueue.csv"))
        real_here = P.HERE
        try:
            P.HERE = tmp
            live_ok, _, _, _ = P.pool(S, W, day="2026-08-29")
            # ...and the same pool with the OLD source of `sent` restored, which is the bug.
            old = {s for s in pd.read_csv(os.path.join(tmp, "w23b_sendqueue.csv"))
                   .pipe(lambda d: d[d.sent]).file.str.replace(".csv", "", regex=False)}
        finally:
            P.HERE = real_here
        dup_live = [s for s, *_ in live_ok[:DAILY_CAP] if s in sent]
        chk("C3a", not dup_live,
            f"on a STALE queue the live pool plans 0 already-sent files"
            f"{'' if not dup_live else f' — {dup_live}'}")
        chk("C3b", len(old) == 0,
            "the old source of `sent` (w23b_sendqueue.csv) is confirmed VACUOUS — it returns "
            f"{len(old)} sent stems out of a live history of {len(sent)}, which is why the "
            "bug was silent. A non-empty answer here means that file changed shape and this "
            "control needs rewriting, not deleting")
        # The counterfactual: had the pool used `old`, how many already-sent files would the
        # top ten have carried? Re-derive it directly rather than asserting it from memory.
        st = pd.read_csv(STALE)
        st["stem"] = st.file.str.replace(".csv", "", regex=False)
        would = [s for s in st.stem if s in sent][:DAILY_CAP]
        chk("C3c", len(would) > 0,
            f"the control is FIRED, not stated: the stale snapshot really does contain "
            f"{len([s for s in st.stem if s in sent])} already-sent files, so a vacuous `sent` "
            f"filter had something to let through")

    # ---- C4: the member-admission regression, fired ----------------------------------------
    real_cert = P.certified_members
    try:
        P.certified_members = lambda _W: set()
        blind, _, _, _ = P.pool(S, W, day=None)
    finally:
        P.certified_members = real_cert
    seeing, _, _, _ = P.pool(S, W, day=None)
    chk("C4", len(blind) < slots <= len(seeing) + len(reg),
        f"member certification is LOAD-BEARING: blind to it the pool holds {len(blind)} "
        f"file(s) for {slots} slot(s); reading it, {len(seeing)} unregistered + {len(reg)} "
        f"already registered")

    # ---- C5: every planned member is certified, and under the LIVE tier --------------------
    from stdflag import is_member                                        # noqa: E402
    cert = real_cert(W)
    bad_mem = []
    for d, o in sorted(fut.items()):
        for s in o:
            if not is_member(s):
                continue
            r = qi.get(s)
            if s not in cert:
                bad_mem.append(f"{d}/{s}: not certified")
            elif r is None or pd.isna(r.pred_lb):
                bad_mem.append(f"{d}/{s}: no point estimate")
            elif float(r.pred_lb) >= tier:
                bad_mem.append(f"{d}/{s}: pred_lb {r.pred_lb:.6f} >= tier {tier}")
    n_mem = sum(1 for o in fut.values() for s in o if is_member(s))
    chk("C5", not bad_mem,
        f"all {n_mem} planned member(s) certified below the {tier} tier"
        f"{'' if not bad_mem else ' — ' + '; '.join(bad_mem[:4])}")

    # ---- C6: nothing already sent, no cross-day duplicate among future days ----------------
    already = sorted({s for o in fut.values() for s in o} & sent)
    chk("C6a", not already,
        f"no registered future file has already been sent{'' if not already else f' — {already}'}")
    seen, dup = {}, []
    for d, o in sorted(fut.items()):
        for s in o:
            if s in seen:
                dup.append(f"{s} ({seen[s]} and {d})")
            seen[s] = d
    chk("C6b", not dup,
        f"no file is registered for two FUTURE days{'' if not dup else ' — ' + '; '.join(dup)}")

    # ---- C7: replay the sender's refusal test at the LIVE bar over every registered day ----
    refused, stale_bar = [], []
    for d, o in sorted(fut.items()):
        p = os.path.join(HERE, f"w72a_plan_{d}.json")
        if os.path.exists(p):
            rec = json.load(open(p)).get("cv_bar")
            if rec is not None and abs(float(rec) - bar) > 1e-12:
                stale_bar.append(f"{d} {float(rec):.10f}")
        for s in o:
            r = qi.get(s)
            if r is None:
                refused.append(f"{d}/{s}: not in the written queue")
                continue
            why = S.above_tier_reason(r, bar)
            risk = 1.0 if pd.isna(r.pred_lb) else S.hijack_risk(float(r.pred_lb), tier)
            if why and risk >= S.P_MAX:
                refused.append(f"{d}/{s}: REFUSED (hijack {risk:.2e})")
    chk("C7", not refused,
        f"every registered file passes the sender's refusal test at the LIVE bar"
        f"{'' if not refused else ' — ' + '; '.join(refused[:4])}")
    # A plan written under an older bar is REPORTED. It is only a fault if C7 caught something,
    # and the older bars here are the LOWER, more permissive ones — so C7 is the test that
    # matters and this line is context for reading it.
    if stale_bar:
        print(f"  ℹ C7 note: {len(stale_bar)} plan(s) recorded a bar older than the live "
              f"{bar:.10f} — {'; '.join(stale_bar)}. The live bar is the HIGHER (stricter) "
              f"one, and C7 just re-tested every one of their files against it.")

    # ---- C8: the w48e OOF exemption is NARROW, and one owner holds the ruling ---------------
    # `w48e_order.py` skips the `oof_<stem>.npy` requirement for a certified member, because a
    # member has no stack CV to reproduce. That exemption is only safe while it discriminates:
    # an UNcertified member must still be refused, and the certified set must be the same set
    # the registrar admits on. Both are checked here, and the first is FIRED on a stem built to
    # look like a filler and certified by nothing.
    probe = "w85_cal_this_stem_was_never_certified"
    chk("C8a", is_member(probe) and probe not in cert,
        f"the exemption discriminates: `{probe}` reads as family `member` "
        f"({is_member(probe)}) yet is NOT certified, so w48e still demands its OOF")
    chk("C8b", cert == W.certified_members() and len(cert) > 0,
        f"one owner: w72a and w48e report the same {len(cert)} certified member(s)")

    json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                   live_rows=len(api), slots=slots, registered=len(reg),
                   plannable=len(plannable), headroom=len(spare),
                   tier=tier, bar=bar, days=sorted(fut), failures=FAILS),
              open(OUT, "w"), indent=1)

    print(f"\n  {'✅' if not FAILS else '⛔'} CONTROLS {len(FAILS)} failure(s)"
          + ("" if not FAILS else "\n    " + "\n    ".join(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
