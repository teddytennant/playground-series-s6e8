"""w26g — drain the priced send queue safely, so a Kaggle day costs a slot almost nothing.

WHY THIS EXISTS
---------------
The brief's economics here are the opposite of the simulation competitions: submissions do
not evict each other, the public board shows best-of-all, so an unused daily slot is pure
waste. Ten slots a day need ten files a day. w23b enumerated the queue and w26d priced it
(P(any of the ten beats the account best) = 6.3e-4 -- draining it is free, not promising).
What has never existed is the last mile: a run still had to hand-pick filenames, hand-write
ten messages, and hand-check nothing had already been sent. This does that part.

The two ways a slot can be WASTED rather than merely unproductive, both guarded here:

  1. Re-sending a file that is already on the board. "Scores are deterministic ... resubmitting
     an identical file is genuinely pointless." Guarded by filename AND by md5, because the
     same predictions have been written under more than one stem in this workspace.
  2. Reading a TRUNCATED submission list. The CLI's default page size is 50 and this account
     passed 50 submissions on 2026-08-17; w17 slot 2 found two files hidden from every
     API-reading script here by exactly that. This asks for --page-size 500 and refuses to
     run if the list comes back exactly at the page size.

Sends NOTHING without --go. Refuses to exceed the 10/day cap, counted live from the API in
UTC, which is the day boundary Kaggle uses.

    .venv/bin/python experiments/w26g_send.py                 # plan only, sends nothing
    .venv/bin/python experiments/w26g_send.py --go --n 10     # actually send
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import SUB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
DAILY_CAP = 10
N_TEST = 296_302
PAGE = 500
QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
LOG = os.path.join(HERE, "w26g_sent.csv")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def api_submissions():
    """The live list, with an explicit page size and a truncation guard."""
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                        "--page-size", str(PAGE)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or "id,fileName" not in r.stdout and "ref,fileName" not in r.stdout:
        raise SystemExit(f"kaggle submissions failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    if len(rows) >= PAGE:
        raise SystemExit(f"submission list came back at exactly the page size ({len(rows)}); "
                         f"it is truncated. Raise PAGE before trusting anything below.")
    return rows


def sent_today(rows):
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return [r for r in rows if r["date"][:10] == today], today


def check_file(path):
    """A submission that is malformed scores zero and burns the slot. Cheap to check."""
    if not os.path.exists(path):
        return f"missing: {path}"
    d = pd.read_csv(path)
    if list(d.columns) != ["id", "addicted_label"]:
        return f"columns {list(d.columns)} != ['id','addicted_label']"
    if len(d) != N_TEST:
        return f"{len(d):,} rows != {N_TEST:,}"
    v = d["addicted_label"].to_numpy()
    if not pd.Series(v).notna().all():
        return "non-finite values"
    if d["id"].duplicated().any():
        return "duplicate ids"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually submit. Without it, plan only.")
    ap.add_argument("--n", type=int, default=DAILY_CAP, help="max files to send this run")
    ap.add_argument("--tag", default="w26g", help="prefix for the submission message")
    # ⛔ THE VETO IS A DEFERRAL UNTIL SOMETHING MAKES IT A BLOCK (w54). `w48e_order.py` encodes
    # the veto as `priority = -1` and this sender sorts on priority descending, so a vetoed file
    # merely goes to the BACK of the plan. That is safe only while the queue is longer than the
    # slots that remain, and w54a computed the date it stops being true: on 2026-08-22 there were
    # 82 unsent files and 90 slots left before the 08-31 deadline, so a mechanical drain at the
    # cap reaches the first vetoed file on 2026-08-29 and sends all 19 of them by the deadline --
    # including `w50_ad216std_logit`, the logit family term on the highest CV base on disk, which
    # `w48e` calls "the single most dangerous file in submissions/". While `check_selection`
    # reports nothing selected, Kaggle auto-selects on best PUBLIC score, which is precisely the
    # exposure the veto exists to close.
    #
    # So the filter is here, it defaults to ON, and it is deliberately NOT conditional on a live
    # `check_selection` read: an API hiccup must not silently un-veto the queue. Fail safe.
    # Retiring a veto is an evidence decision and belongs in `w48e.VETO`, one entry at a time,
    # with the reason written down -- not in a flag on the send line.
    ap.add_argument("--allow-vetoed", action="store_true",
                    help="send priority<0 files anyway. Only defensible once selection is "
                         "confirmed made; retire the VETO entry in w48e_order.py instead.")
    # Same reasoning one level up, and it is the SAME BUG (w55). w54 established the number
    # that makes a spare slot free or a liability -- "a filler is SAFE iff its predicted public
    # score is < the 0.97118 auto-selection tier" -- and wrote it into RESEARCH.md and into
    # this file's own unfilled-slots message. Nothing enforced it, and four queue rows carry
    # `pred_lb = NaN`, so it could not have been enforced on them anyway. w48e's comment
    # defended those rows with "they keep priority 0 and sort last", which is exactly the
    # argument w54 refuted for the veto: last is reached on ~2026-08-29. Certification belongs
    # in `w55a_unpriced.py`, on evidence, not in a flag on the send line.
    ap.add_argument("--allow-unpriced", action="store_true",
                    help="send rows with no pred_lb anyway. The wrong tool: certify them with "
                         "experiments/w55a_unpriced.py and re-run w48e_order.py --write.")
    a = ap.parse_args()

    rows = api_submissions()
    today, daystr = sent_today(rows)
    left = DAILY_CAP - len(today)
    print(f"{len(rows)} submissions on record; {len(today)} already sent on {daystr} (UTC); "
          f"{left} of {DAILY_CAP} slots left today")

    sent_names = {r["fileName"] for r in rows}
    sent_md5 = set()
    for n in sent_names:
        p = os.path.join(SUB, n)
        if os.path.exists(p):
            sent_md5.add(md5(p))
    print(f"{len(sent_names)} distinct filenames sent, {len(sent_md5)} of them still on disk "
          f"and fingerprinted")

    q = pd.read_csv(QUEUE)
    # ⚠ THE QUEUE MUST BE KEYED TO TODAY (w53). w49 keyed `w48e_order.py`, the WRITER, to the UTC
    # day so a registered list could never be written for the wrong one. Nothing keyed the READER,
    # and the gap is not theoretical: w53 dry-ran this sender against the 08-22 CSV on the evening
    # of 08-22 and it planned a ten that (a) omitted `w48_cal_hboyang_mix`, slot 1 of the
    # registered 08-23 list, and (b) included two `logit`-family files. Both follow from the same
    # mechanism -- every priority-1 row was already sent, so the plan fell through to the
    # priority-0 tail, where no veto applies because `w48e` enforces the veto as priority -1 and
    # DROPS the `vetoed` column before writing. A stale queue is an UNVETOED queue.
    #
    # A dry run is allowed to look at a queue for another day -- that is how a run at the cap
    # inspects tomorrow's plan -- but it says so loudly. A real send refuses.
    _day = str(q["plan_day"].iloc[0]) if "plan_day" in q.columns and len(q) else None
    if _day != daystr:
        _msg = (f"queue was written for {_day or 'AN UNSTAMPED DAY (pre-w53 artefact)'}, "
                f"not today ({daystr})")
        if a.go:
            print(f"\n⛔ REFUSING TO SEND: {_msg}.\n"
                  f"   Run `w48e_order.py --day {daystr} --write` first. Do not send off a stale "
                  f"queue: its priority-1 band is another day's list, and the files behind it "
                  f"carry a stale day's veto.")
            return 1
        # ⚠ NOT "carries no veto" any more (w54). The priority<0 filter above is unconditional, so
        # a stale CSV's vetoed rows are still blocked. What a stale CSV loses is everything the
        # veto has learned SINCE it was written -- `w48e.VETO` is applied at WRITE time and the
        # `vetoed` column is dropped, so a row vetoed today reads as priority 0 in yesterday's
        # artefact and sails through. Re-write, don't reason about it.
        print(f"\n⚠ DRY RUN AGAINST A QUEUE FOR ANOTHER DAY: {_msg}. The plan below is NOT the "
              f"registered list for today, and it carries only the veto as it stood on "
              f"{_day or 'the day it was written'}.")
    # `priority` pins `check_selection.WANTED` to the head (w28). A deadline pick that is never
    # submitted cannot be selected, and that outranks any public-LB ordering.
    if "priority" not in q.columns:
        q["priority"] = 0
    # `send_rank` (w37) overrides the pred_lb tiebreak inside a priority band. It exists for
    # files whose send ORDER carries information that their predicted LB does not: the w37
    # es-bias calibration sends are deliberately low-scoring member vectors, and the CLEAN
    # anchor among them must land before the DIRTY readings it is there to make interpretable.
    # Blank/absent = default behaviour, so every pre-w37 row is unaffected.
    if "send_rank" not in q.columns:
        q["send_rank"] = float("nan")
    q["send_rank"] = pd.to_numeric(q["send_rank"], errors="coerce").fillna(1e9)
    q = q.sort_values(["priority", "send_rank", "pred_lb"], ascending=[False, True, False])
    _pin = q[q.priority == 1].file.tolist()
    if _pin:
        # Priority 1 is no longer WANTED-only: w37 pins calibration sends into the same band.
        # Label each one, so the plan never implies a measurement file is a deadline pick.
        try:
            from check_selection import WANTED as _W
        except Exception:
            _W = set()
        print("PINNED first (priority 1):")
        for f in _pin:
            print(f"  {f:34s} {'check_selection.WANTED, unsent' if f in _W else 'pinned, NOT a deadline pick'}")
    # A dry run plans the full --n regardless of slots left, so a slot at the cap can still
    # SEE tomorrow's queue and check it is sane. Only a real send is clamped by `left`.
    cap = a.n if not a.go else min(a.n, max(left, 0))
    plan, seen_md5, blocked, unpriced = [], set(), [], []
    for r in q.itertuples():
        if len(plan) >= cap:
            break
        if r.file in sent_names:
            continue
        if int(getattr(r, "priority", 0)) < 0 and not a.allow_vetoed:
            # Not "skip": BLOCKED. Collected and reported after the plan so it cannot scroll off.
            blocked.append(r.file)
            continue
        # w55: unpriceable. A row with no `pred_lb` is a row the auto-selection tier rule
        # cannot be evaluated on, and its submission message renders as "predicted LB nan ...
        # P(beat) nan", destroying the provenance trail every later run reads back. `w48e`
        # prices such rows from `w55a_unpriced.json` once w55a has CERTIFIED them below the
        # tier on a calibrated instrument; a row that reaches here still NaN has been certified
        # by nothing. Unconditional and default-on, exactly like the veto filter above.
        # NaN `cv` is tolerated ONLY where the row has renounced being a candidate by declaring
        # fam == "member" (w53: a member's published OOF is not a cross-fitted stack CV).
        _why = None
        if pd.isna(getattr(r, "pred_lb", None)):
            _why = "no pred_lb — never certified against the auto-selection tier"
        elif pd.isna(getattr(r, "cv", None)) and str(getattr(r, "fam", "")) != "member":
            _why = "no cv and fam != member — undeclared candidate carrying no CV"
        if _why and not a.allow_unpriced:
            unpriced.append((r.file, _why))
            continue
        p = os.path.join(SUB, r.file)
        m = r.md5 if isinstance(getattr(r, "md5", None), str) else None
        if m is None and os.path.exists(p):
            m = md5(p)
        if m in sent_md5:
            print(f"  skip {r.file}: identical predictions already sent (md5 {m[:8]})")
            continue
        if m in seen_md5:
            print(f"  skip {r.file}: duplicate of another file already in this plan")
            continue
        why = check_file(p)
        if why:
            print(f"  skip {r.file}: {why}")
            continue
        # the md5 on the queue CSV is a claim; verify it rather than trust it
        real = md5(p)
        if m and real != m:
            print(f"  skip {r.file}: md5 on disk {real[:8]} != queue's {m[:8]}, file changed")
            continue
        seen_md5.add(real)
        plan.append(r)

    if blocked:
        print(f"\n  ⛔ {len(blocked)} VETOED file(s) skipped, not sent (w54). Reasons are in "
              f"`w48e_order.py`'s VETO dict:")
        for f in blocked[:8]:
            print(f"       {f}")
        if len(blocked) > 8:
            print(f"       ... and {len(blocked) - 8} more")
        print("     A vetoed file is unsendable while `check_selection` reports nothing "
              "selected.")
        print("     Retire the entry in w48e_order.py on evidence; do NOT pass --allow-vetoed.")

    if unpriced:
        print(f"\n  ⛔ {len(unpriced)} UNPRICEABLE file(s) skipped, not sent (w55). The "
              f"auto-selection tier\n     rule cannot be evaluated on a row with no price, so "
              f"the row is not sendable:")
        for f, why in unpriced[:8]:
            print(f"       {f:34s} {why}")
        if len(unpriced) > 8:
            print(f"       ... and {len(unpriced) - 8} more")
        print("     Certify with `.venv/bin/python experiments/w55a_unpriced.py`, then re-run"
              "\n     `w48e_order.py --day <today> --write`. Do NOT pass --allow-unpriced.")

    if not plan:
        print("\nnothing to send: the queue is drained of everything sendable.")
        if blocked:
            print("  ⚠ SLOTS WILL GO UNFILLED. That is the intended trade: the brief's "
                  "\"an extra\n    submission can never hurt\" is FALSE on this account while "
                  "nothing is selected,\n    because Kaggle then auto-selects on best PUBLIC "
                  "score. Build something new and\n    non-vetoed to fill them; do not reach "
                  "for the veto list.")
        return 0

    print(f"\nplan, {len(plan)} file(s):")
    for i, r in enumerate(plan, 1):
        print(f" {i:2d}. {r.file:34s} CV {r.cv:.10f}  pred_lb {r.pred_lb:.6f}  "
              f"P(beat best) {r.p_beat:.2e}")

    if not a.go:
        print("\nDRY RUN. Re-run with --go to send.")
        return 0
    if left <= 0:
        print("\nat the daily cap; refusing to send.")
        return 1

    for r in plan:
        # A `msg` on the queue row replaces the queue-drain boilerplate. Without this, a w37
        # calibration send would be described as a queue-drain attempt and a future run reading
        # the submission history would misread its ~0.958 public score as a huge regression.
        override = getattr(r, "msg", None)
        # ⚠ NaN-safe (w55). The default template formats `cv` and `p_beat`; a `member` row has
        # neither, and would ship "CV nan ... P(beat) nan". `w48e` writes a proper `msg` for
        # every row it certifies, so this branch should never fire -- it is the fail-safe for
        # a certified row that somehow arrives without one, and it refuses rather than lie.
        _has_msg = isinstance(override, str) and override.strip()
        if not _has_msg and pd.isna(getattr(r, "cv", None)):
            print(f"  skip {r.file}: fam={r.fam} row has no cv and no registered msg; "
                  f"re-run w48e_order.py --write so it is described honestly")
            continue
        msg = (f"{a.tag} queue-drain {r.stem} — CV {r.cv:.10f}, family {r.fam}, "
               f"w26d predicted LB {r.pred_lb:.6f} with P(beats the 0.97118 account best) "
               f"{r.p_beat:.2e}. Sent because the brief's economics make an unused slot pure "
               f"waste, NOT because it is expected to move anything: every unsent file is "
               f"below the best already-sent CV and w26d prices the whole queue at 6.3e-4 of "
               f"beating the board. Not a deadline candidate — selection is on CV and this "
               f"is {(0.9701150809 - r.cv)*1e6:.1f}e-6 below the best sent CV.")
        if isinstance(override, str) and override.strip():
            msg = override.strip()
        p = os.path.join(SUB, r.file)
        print(f"\n--> {r.file}", flush=True)
        out = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", p,
                              "-m", msg], capture_output=True, text=True, timeout=1200)
        print((out.stdout + out.stderr).strip()[-600:])
        fresh = not os.path.exists(LOG)
        with open(LOG, "a") as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(["utc", "file", "cv", "pred_lb", "p_beat", "md5", "rc"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        r.file, r.cv, r.pred_lb, r.p_beat, md5(p), out.returncode])

    # `submit` has returned 400 after a 100% upload with nothing registering (RESEARCH.md).
    # Never trust the exit status; re-read the list.
    after, _ = sent_today(api_submissions())
    print(f"\nconfirmed from the API: {len(after)} submissions today (was {len(today)}). "
          f"{DAILY_CAP - len(after)} slots left.")
    for r in after[:len(plan)]:
        print(f"  {r['date']}  {r['fileName']:34s} {r.get('publicScore','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
