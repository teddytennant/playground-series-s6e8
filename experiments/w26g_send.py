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

    q = pd.read_csv(QUEUE).sort_values("pred_lb", ascending=False)
    # A dry run plans the full --n regardless of slots left, so a slot at the cap can still
    # SEE tomorrow's queue and check it is sane. Only a real send is clamped by `left`.
    cap = a.n if not a.go else min(a.n, max(left, 0))
    plan, seen_md5 = [], set()
    for r in q.itertuples():
        if len(plan) >= cap:
            break
        if r.file in sent_names:
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

    if not plan:
        print("\nnothing to send: the queue is drained.")
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
        msg = (f"{a.tag} queue-drain {r.stem} — CV {r.cv:.10f}, family {r.fam}, "
               f"w26d predicted LB {r.pred_lb:.6f} with P(beats the 0.97118 account best) "
               f"{r.p_beat:.2e}. Sent because the brief's economics make an unused slot pure "
               f"waste, NOT because it is expected to move anything: every unsent file is "
               f"below the best already-sent CV and w26d prices the whole queue at 6.3e-4 of "
               f"beating the board. Not a deadline candidate — selection is on CV and this "
               f"is {(0.9701150809 - r.cv)*1e6:.1f}e-6 below the best sent CV.")
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
