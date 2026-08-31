"""w23b: the send queue, ranked on cross-fitted OOF CV, so no future slot deliberates.

WHY THIS EXISTS
---------------
The brief's economics for this competition are the opposite of the simulation ones:
submissions do NOT evict each other, the public board shows best-of-all, so an unused
daily slot is pure waste and "vary something real each time" is the only constraint.
Ten slots a day therefore need ten files a day, and the workspace has been building
them faster than it sends them -- 92 CSVs built, 50 distinct filenames ever sent.

Every wave so far has rediscovered that queue by hand at the top of its slot. This
enumerates it once, scores each candidate on the SAME frozen-fold OOF vector that
every CV number in the journal comes from, and writes the ranking to disk. Nothing
here is a new model and nothing is chosen with reference to the public leaderboard --
it is bookkeeping that stops a slot being spent on bookkeeping.

    w23b_sendqueue.py
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import SUB, TARGET, load_raw  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads
N_TEST = 296_302


PAGE = 200   # w140: AT the server cap, not above it. The endpoint caps a page at 200 and
             # returns a next_page_token the CLI never prints, so at 500 this file's own
             # `len(rows) >= PAGE` read `200 >= 500 -> False` and could NEVER fire.
             # Measured in w140c_pagetruth.py (T1/T2/T5).


def sent_filenames():
    """Every fileName the API has ever accepted. Paginated -- the CLI prints a
    `Next Page Token` line ABOVE the header, which must be stripped (RESEARCH,
    w17 slot 2).

    ⚠ FIXED 2026-08-18 (w26 slot 4). The previous version passed no --page-size and
    reasoned "one page of 50 is enough while the account is under 100 sends". That is
    wrong twice over: the page IS 50, not 100, and the account passed 50 sends on
    2026-08-17. From then on this function silently returned only the most recent 50
    filenames, so **21 files that were already on the board were reported as unsent**
    and went into the queue this file exists to produce -- including blend158_h3, whose
    CV w26d then quoted as "the best unsent file". Four of the ten rows a send day would
    have drained were re-sends of byte-identical files, which the brief calls genuinely
    pointless. The `Next Page Token` guard below did not save it, so the guard is now a
    hard failure on a full page rather than a printed warning. Do not remove the
    --page-size argument, and do not trust any Kaggle list you did not ask a size for."""
    # ⚠ w140: the `Next Page Token` guard below never fired, and could not. The CLI does
    # not print that line at all (measured, w140c T3) while the server DOES return a
    # token, so `tok` was always None and `len(rows) >= PAGE` was False at PAGE=500.
    # This file was right that a token is the thing to check; it was reading for it in
    # the one place it never appears. kaggle_list reads it from the API.
    rows = kaggle_list.submissions()
    return {r["fileName"] for r in rows}, rows


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    sent, rows = sent_filenames()
    print(f"{len(rows)} submissions on record, {len(sent)} distinct filenames")

    have = sorted(f for f in os.listdir(SUB) if f.endswith(".csv"))
    recs = []
    for f in have:
        name = f[:-4]
        op = os.path.join(SUB, f"oof_{name}.npy")
        cv = np.nan
        if os.path.exists(op):
            o = np.load(op).astype(np.float64)
            if o.shape == (len(y),):
                cv = fast_auc(y, o)
        blob = open(os.path.join(SUB, f), "rb").read()
        nrow = blob.count(b"\n") - 1
        recs.append(dict(file=f, sent=f in sent, cv=cv, rows=nrow,
                         md5=hashlib.md5(blob).hexdigest()))

    df = pd.DataFrame(recs)
    # ⚠ "Scores are deterministic ... resubmitting an identical file is genuinely
    # pointless" (brief). Byte-identical twins DO occur here for a structural reason:
    # blend_lab writes the per-transform stacks on every build, so a `--kinds a,b,c`
    # run and a `--kinds a,b,c,d` run emit the same single-transform files under
    # different names. Collapse them, and mark a candidate DEAD if any of its twins has
    # already been sent -- that is a slot saved, not bookkeeping.
    twin_sent = df[df["sent"]].groupby("md5")["file"].first()
    df["twin_sent_as"] = df["md5"].map(twin_sent)
    df["dead"] = df["twin_sent_as"].notna() & ~df["sent"]
    dup = df[~df["sent"] & ~df["dead"]].duplicated("md5", keep="first")
    df.loc[dup[dup].index, "dead"] = True
    df.loc[dup[dup].index, "twin_sent_as"] = "(dup of a higher-ranked unsent file)"
    if df["dead"].any():
        print(f"\n{int(df['dead'].sum())} candidate(s) are byte-identical to another "
              f"file and would score identically -- excluded from the queue:")
        for _, r in df[df["dead"]].iterrows():
            print(f"  {r['file']:<34} == {r['twin_sent_as']}")
    bad = df[df["rows"] != N_TEST]
    if len(bad):
        print("\n⚠ WRONG ROW COUNT -- never send these:")
        print(bad.to_string(index=False))
    df = df[df["rows"] == N_TEST]

    q = (df[~df["sent"] & ~df["dead"]]
         .sort_values("cv", ascending=False, na_position="last")
         .drop(columns=["dead", "twin_sent_as"]).reset_index(drop=True))
    q.to_csv(os.path.join(HERE, "w23b_sendqueue.csv"), index=False)

    best_sent = df[df["sent"]].cv.max()
    print(f"\nbest CV among ALREADY-SENT files: {best_sent:.10f}")
    print(f"\nSEND QUEUE -- {len(q)} unsent valid files, ranked on cross-fitted OOF CV")
    print(q.head(20).to_string(index=False, float_format="%.10f"))
    nocv = q[q["cv"].isna()]
    if len(nocv):
        print(f"\n{len(nocv)} unsent files carry NO stored OOF vector and therefore "
              f"cannot be ranked or defended on CV:")
        print("  " + ", ".join(nocv["file"].tolist()))
    print(f"\nwrote {os.path.join(HERE, 'w23b_sendqueue.csv')}")


if __name__ == "__main__":
    main()
