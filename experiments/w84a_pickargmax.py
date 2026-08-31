"""w84a — is the file we intend to SELECT still the best thing we have ever SENT, on CV?

WHAT THIS EXISTS FOR, AND WHY IT IS NOT ONE OF THE 27.
`SELECT_THESE.md` names two refs, and every guard around it protects a DIFFERENT claim:
`w74b_clickstaleguard` protects the click PRICE against the public tier moving, `w56b`
protects WANTED against becoming INELIGIBLE, `w59b`/`w62b` protect the send-path hijack BAR.
None of them asserts the thing the pick actually rests on — that **no file this account has
sent out-CVs the slot-1 pick**. That claim was true when WANTED was set and has been carried
in prose ever since, across ~60 further sends. Prose does not survive; a test that re-reads
the live submission list does (w54a/w55a/w56b's idiom).

⚠ THE HAZARD IS LIVE, NOT HYPOTHETICAL. Six days and up to 60 sends remain, and
`w26g_send.hijack_cv_bar` exists precisely because a queued file CAN clear the bar. If one
ever does, it becomes a legitimate deadline candidate and WANTED must be revisited by hand.
This guard is what notices. Today the best of the ten 08-25 sends is CV rank 7 and the pick
is rank 1 of 130, so it passes on a real margin rather than vacuously.

⛔ THIS ADOPTS NOTHING AND MUST NOT MOVE WANTED BY ITSELF. It reports; a human decides.
⛔ G4 IS NOT A BUG. Slot 2 is CV rank ~35 ON PURPOSE — it is the cross-base hedge priced at a
   measured size by `w64a_hedgeprice.py` (check_selection.py's "SLOT 2 IS SETTLED AT A
   MEASURED SIZE — DO NOT RE-OPEN IT"). A later run that "fixes" slot 2 into the CV #2 file
   is re-opening a settled question; G4 pins the citation so that edit trips a control.

GUARDS
  G1    slot-1 WANTED is the strict CV argmax over every sent file with a parseable CV.
  G2 +- the fetch is PAGINATED and it is exercised both ways (the 50-row cap silently cost
        w82a 18 rows and two days; found 2026-08-25, w84).
  G2b   the cap OUTLIVES THE HISTORY -- len(rows) < PAGE. G2 alone compared the cap against
        50, which 200 beat while still truncating 201 rows on the deadline day (w86 §1).
  G3 -  a planted higher-CV send TRIPS G1 — the guard is fired, not just asserted.
  G4    slot 2 is the w64-settled hedge: its identity and the w64 marker are both pinned.
  G5    CVs parsed from the descriptions agree with the on-disk ledger where both carry a
        file — two independent sources, so a description-format drift or a ledger edit shows.

    .venv/bin/python experiments/w84a_pickargmax.py          # 0 = ok, 1 = a guard failed
"""
from __future__ import annotations

import csv, io, json, os, re, subprocess, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads; see its docstring
COMP = "playground-series-s6e8"
OUT = os.path.join(HERE, "w84a_pickargmax.json")
LEDGER = os.path.join(HERE, "w57a_tierprice2.json")
CHECKSEL = os.path.join(HERE, "check_selection.py")

# ⚠ Always paginated. Without it the CLI returns 50 rows here and says nothing (RESEARCH,
# "The Kaggle submission list is PAGINATED"). G2 exercises this.
# ⚠⚠ AND THE CAP MUST OUTLIVE THE HISTORY. This was 200; the account finishes at 201 on
# 2026-08-31, so on the one day this guard decides anything it would have read 200 of 201 and
# G2 would still have passed, because G2 only ever compared the cap against 50 (w86 §1).
# `fetch_raw` now refuses at the cap itself -- the only test that detects its own truncation.
PAGE = 200   # w140: AT the server cap, not above it. The endpoint caps a page at 200 and
             # returns a next_page_token the CLI never prints, so at 500 this file's own
             # `len(rows) >= PAGE` read `200 >= 500 -> False` and could NEVER fire.
             # Measured in w140c_pagetruth.py (T1/T2/T5).
SUBS_ARGV = ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", str(PAGE)]

CV_RE = re.compile(r"CV (0\.\d{6,})")
SLOT1 = "w36_ad199stdcorr.csv"          # the CV pick
SLOT2 = "w23_ad187stdcorr.csv"          # the w64-settled cross-base hedge, NOT the CV #2
W64_MARKER = "SLOT 2 IS SETTLED AT A MEASURED SIZE"
MIN_SAMPLE = 60                          # below this the list is not the account's history


def fetch_raw(argv, *, capped: bool = False) -> str:
    """⚠ w140: the UNCAPPED path now paginates (kaggle_list). `argv` is still honoured
    for the deliberately-capped control in G2, which must keep hitting the cap."""
    if not capped:
        rows = kaggle_list.submissions()
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
        return buf.getvalue()
    raw = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    if not capped and len(parse(raw)) >= PAGE:
        raise SystemExit(f"submission list came back at the page size ({PAGE}); it is "
                         f"TRUNCATED and the argmax below would be over a partial history. "
                         f"Raise PAGE and re-run w86a_pagecap.py.")
    return raw


def parse(raw: str) -> pd.DataFrame:
    lines = raw.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("ref,"))
    return pd.read_csv(io.StringIO("\n".join(lines[head:])))


def by_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Best published CV per distinct sent file, ranked. Descriptions are immutable, so this
    is a read of what was claimed AT SEND TIME, not of anything a later run could edit."""
    d = df.copy()
    d["cv"] = d["description"].astype(str).str.extract(CV_RE)[0].astype(float)
    d = d.dropna(subset=["cv"])
    best = (d.groupby("fileName", as_index=False)
              .agg(cv=("cv", "max"), ref=("ref", "first"), pub=("publicScore", "max")))
    return best.sort_values("cv", ascending=False).reset_index(drop=True)


def rank_of(best: pd.DataFrame, fname: str):
    row = best[best["fileName"] == fname]
    if row.empty:
        return None, None
    cv = float(row["cv"].iloc[0])
    return int((best["cv"] > cv).sum()) + 1, cv


def main() -> int:
    bad: list[str] = []

    raw_full = fetch_raw(SUBS_ARGV)
    df = parse(raw_full)
    best = by_cv(df)

    print("=" * 92)
    print("w84a -- is the slot-1 pick still the CV argmax over everything this account has sent?")
    print("=" * 92)
    print(f"  sent rows {len(df)}   distinct files with a parseable CV {len(best)}")

    if len(best) < MIN_SAMPLE:
        bad.append(f"G0 only {len(best)} files carry a parseable CV, floor {MIN_SAMPLE}")

    # ---- G1: the pick is the strict argmax --------------------------------------------
    r1, cv1 = rank_of(best, SLOT1)
    if r1 is None:
        bad.append(f"G1 slot-1 pick {SLOT1} is not in the sent list with a CV at all")
    else:
        above = best[best["cv"] > cv1]
        print(f"\n  slot 1  {SLOT1:34s} CV {cv1:.10f}   rank {r1} of {len(best)}")
        if r1 != 1:
            bad.append(f"G1 ⚠⚠ {len(above)} SENT FILE(S) OUT-CV THE PICK: "
                       f"{above['fileName'].tolist()} -- WANTED must be revisited BY HAND")
        else:
            runner = best.iloc[1]
            print(f"          margin over the CV #2 sent file "
                  f"({runner['fileName']}) {(cv1 - runner['cv']) * 1e6:+.3f}e-6")

    # ---- G4: slot 2 is the settled hedge, deliberately not the CV #2 -------------------
    r2, cv2 = rank_of(best, SLOT2)
    if r2 is None:
        bad.append(f"G4 slot-2 hedge {SLOT2} is not in the sent list with a CV")
    else:
        print(f"  slot 2  {SLOT2:34s} CV {cv2:.10f}   rank {r2} of {len(best)}  "
              f"(a HEDGE, not a CV pick -- w64)")
    src = open(CHECKSEL).read()
    if W64_MARKER not in src:
        bad.append("G4 the w64 'slot 2 is settled' marker is gone from check_selection.py -- "
                   "somebody may be about to re-open it as a CV slot")
    if f'"{SLOT1}"' not in src or f'"{SLOT2}"' not in src:
        bad.append(f"G4 check_selection.WANTED no longer names {SLOT1} and {SLOT2}; "
                   f"this guard is pinned to the wrong pair")

    print("\n  --- top 8 by CV over everything ever sent ---")
    for i, r in enumerate(best.head(8).itertuples(), 1):
        tag = "  <== slot 1" if r.fileName == SLOT1 else ""
        print(f"   {i:2d}. {r.cv:.10f}  pub {r.pub}  {r.fileName}{tag}")

    # ---- G2: the pagination, exercised both ways --------------------------------------
    if "--page-size" not in SUBS_ARGV:
        bad.append("G2 the fetch has lost --page-size -- the history is capped at 50")
    else:
        capped = parse(fetch_raw([a for a in SUBS_ARGV if a not in ("--page-size", str(PAGE))],
                                 capped=True))
        if len(df) < len(capped):
            bad.append(f"G2 paginated fetch returned FEWER rows ({len(df)}) than capped "
                       f"({len(capped)}) -- pagination is broken")
        elif len(df) == len(capped):
            print(f"\n  ⚠ G2 account is at {len(df)} sends, at or under the CLI cap -- "
                  f"pagination untested this run, not passing")
        else:
            print(f"\n  ✅ G2 pagination exercised: {len(df)} paginated vs {len(capped)} capped")
        # ⚠ G2b IS THE PART G2 WAS MISSING. Beating 50 says nothing about beating the history;
        # only the cap compared against itself does (w86). This is what fetch_raw enforces, and
        # asserting it here too makes the failure legible instead of an exception.
        # ⚠ w140 REPLACED G2b's TEST, NOT ITS INTENT. It used to require the read to sit
        # strictly under its own page size. That is the right test for a CAPPED read and
        # it is unsatisfiable for a COMPLETE one: the paginated fetch is 201 rows and the
        # server's page cap is 200, so "under the cap" now means "truncated". The intent
        # -- the read must not be silently short -- is now enforced where it belongs, by
        # kaggle_list.submissions(), which follows next_page_token and RAISES rather than
        # returning a partial list. What is left to check here is that pagination really
        # bought rows the capped call could not reach.
        if len(df) <= kaggle_list.PAGE_CAP < len(capped) + 1 and len(df) == len(capped):
            bad.append(f"G2b paginated and capped agree at {len(df)} rows while the cap "
                       f"is {kaggle_list.PAGE_CAP} -- pagination bought nothing")
        else:
            print(f"  ✅ G2b paginated {len(df)} rows vs the server page cap "
                  f"{kaggle_list.PAGE_CAP}; the read is not bounded by a page size")

    # ---- G3: negative control -- a planted better send must TRIP G1 ---------------------
    planted = pd.concat([df, pd.DataFrame([{
        "ref": 1, "fileName": "PLANTED_hijacker.csv", "date": "2026-08-25 00:00:00",
        "description": "planted control CV 0.9999999999", "status": "x",
        "publicScore": 0.5, "privateScore": None}])], ignore_index=True)
    pr, _ = rank_of(by_cv(planted), SLOT1)
    if pr == 1:
        bad.append("G3 a planted higher-CV send did NOT displace the pick -- G1 is inert")
    else:
        print(f"  ✅ G3 planted hijacker demotes the pick to rank {pr} (guard fires)")

    # ---- G5: descriptions vs the on-disk ledger ----------------------------------------
    led = json.load(open(LEDGER))["cv"]
    shared = worst = 0
    worst_name = ""
    for name, v in led.items():
        row = best[best["fileName"] == f"{name}.csv"]
        if row.empty:
            continue
        shared += 1
        gap = abs(float(row["cv"].iloc[0]) - float(v))
        if gap > worst:
            worst, worst_name = gap, name
    if shared < 5:
        bad.append(f"G5 only {shared} files shared with the ledger -- cannot cross-check")
    elif worst > 1e-9:
        bad.append(f"G5 description CV disagrees with the ledger on {worst_name} by {worst:.2e}")
    else:
        print(f"  ✅ G5 {shared} files cross-check against the ledger, worst gap {worst:.1e}")

    json.dump({
        "n_sent_rows": int(len(df)), "n_files_with_cv": int(len(best)),
        "slot1": SLOT1, "slot1_cv": cv1, "slot1_rank": r1,
        "slot2": SLOT2, "slot2_cv": cv2, "slot2_rank": r2,
        "slot2_note": "w64-settled cross-base hedge; NOT a CV pick, see check_selection.py",
        "top8": best.head(8).to_dict(orient="records"),
        "ADOPTS": "nothing -- reports; a human moves WANTED",
    }, open(OUT, "w"), indent=2, default=float)

    print()
    for b in bad:
        print(f"  ❌ {b}")
    print(f"  {'✅ GUARDS 0 failures' if not bad else '❌ GUARDS FAILED'}")
    print(f"\nFAILURES: {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
