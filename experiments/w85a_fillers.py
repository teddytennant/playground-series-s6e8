"""w85a — BUILD THE FILLERS. There are 18 more send slots than sendable files.

THE ARITHMETIC (w85, 2026-08-25, measured after the day's ten had landed):

    slots left before the deadline   6 whole days x 10          =  60
    unsent valid files (w23b, live)                             =  61
    of which VETOED above the hijack CV bar (w26g blocks them)  =  19
    SENDABLE                                                    =  42
    ------------------------------------------------------------------
    SLOTS THAT GO UNFILLED                                         18

`w54a_vetoexpiry.py` reports 8, not 18, because it reads `w26d_queueprice.csv` and that CSV
still carried the ten files sent at 12:40Z today. Same failure shape as w84 §2's 50-row cap:
a stale read that parses cleanly and errs in the flattering direction. Fixed there this run.

WHY THAT MATTERS. The brief is explicit that submissions here do not evict each other and the
public board shows best-of-all, so an unused slot is pure waste, and 18 of them is three days
of the competition thrown away. The workspace already knows the rule for spending them --
w54's tier rule, in `w54a`'s own output:

    a filler is SAFE iff its predicted public score is < the auto-selection tier (0.97119);
    below the tier it cannot be auto-selected, so it costs nothing.
    Fill the gap with files BELOW the tier, never by retiring a veto.

and it already has the precedent for what to put there: `w37b_calfiles.py`, which sent five raw
member test vectors and got back five exact public AUCs for individual models. This is that,
widened. Each file is one member's own test-prediction vector; its public score is a
MEASUREMENT of that member on the public slice, not an attempt on the board.

⛔ WHAT THIS DOES NOT DO. It does not re-open the es-on-val deflation constant (closed, w37e R3)
and it does not license a per-family or per-member correction to anything (w81 §6, w84 §2). The
readings are a record. Adoption would be choosing the population after seeing which way it went.

THE SAFETY BOUND, AND WHY IT IS NOT AN ASSERTION.
Over the five self-submitted raw member vectors that have landed, `public - oof_auc` is
    +4716.3, +1982.2, +1429.9, +1424.2, +1196.8  (e-6; max +0.0047163, median +0.0014299)
`w85b_prereg.py` C3 recomputes both from the live submission list and refuses to register if
they have moved. The registered prediction for a filler is `oof_auc + 0.0014299` (the MEDIAN
offset; w37e R2
killed the linear-in-AUC form at 6.28 sigma, so no line is fitted). `w55a_unpriced.py` then
certifies each row at `registered pred_lb + worst landed overshoot`, exactly as it already does
for the w37 rows -- the account's own out-of-sample residual, not a number invented here.

    FILLER_MAX_OOF = 0.9600  ->  registered pred_lb <= 0.96143
                             ->  w55a bound        <= 0.96615
                             ->  margin under the 0.97119 tier >= 5000e-6

CONTROLS (w72 §5.3: a control that can only fail is not a control).
  C1 +  every built file is re-read from disk and its md5 checked against every submission
        already on disk AND against the other fillers -- a byte-identical file scores
        identically and is a wasted slot, which is the one thing this script exists to stop.
        A duplicate is REMOVED and SKIPPED, not counted as a failure: catching one is C1
        working. It fired on the first run -- `mkt_realmlp` was already sent as
        `w37_cal_mkt_realmlp.csv`, and without C1 it would have gone out again for nothing.
        The fatal form of the same invariant is C5.
  ⚠ THE PRIOR-md5 MAP EXCLUDES THIS BUILDER'S OWN PREFIX. It did not on the first draft, and
        the second run therefore found every filler byte-identical to the copy it had written
        itself, deleted all 25, and reported 27 failures. A builder that is not idempotent
        destroys its own output the moment anyone re-runs it as a check.
  C5 -  the manifest is re-checked at the end: no two shipped fillers share an md5 and none
        matches a file already in `submissions/`. C1 can only skip; C5 is what fails.
  C2 +  the oof AUC is recomputed from the vector on disk, not taken from the survey CSV.
  C3 -  the tier bound is asserted per row, so raising FILLER_MAX_OOF without re-reading the
        offset table fails loudly instead of shipping a file that can be auto-selected.
  C4 +  row count and id order are checked against `data/test.csv` on every file.

    .venv/bin/python experiments/w85a_fillers.py            # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
SUB = os.path.join(W, "submissions")
TARGET = "addicted_label"

PREFIX = "w85_cal_"
FILLER_MAX_OOF = 0.9600      # registered; see the bound above
OFFSET_MEDIAN = 0.0014299     # median (public - oof_auc) over the five landed w37 readings
OFFSET_MAX = 0.0047163        # worst of the same five -- used only for the printed headroom
TIER = 0.97119               # auto-selection tier: 2nd-best public on the account
MARGIN_MIN = 1000e-6         # C3: refuse any row whose bound is not this far under the tier
WANT = 18                    # the measured shortfall

SURVEY = os.path.join(HERE, "w85_membersurvey.csv")
OUT = os.path.join(HERE, "w85a_fillers.csv")


def main() -> int:
    fails, skipped = [], []
    te = pd.read_csv(os.path.join(W, "data", "test.csv"), usecols=["id"])
    y = pd.read_csv(os.path.join(W, "data", "train.csv"),
                    usecols=[TARGET])[TARGET].to_numpy()

    df = pd.read_csv(SURVEY)
    df = df[df.oof_auc <= FILLER_MAX_OOF].copy()
    # one row per member: the same vector appears in ext_members7 and ext_members7pin
    df = df.sort_values(["member", "dir"]).drop_duplicates("member", keep="first")
    df = df.sort_values("oof_auc", ascending=False).reset_index(drop=True)

    # ⚠ this builder OVERWRITES its own output, so its own prefix must not count as a prior
    # file -- see the note above C5.
    seen = {}
    for f in sorted(os.listdir(SUB)):
        if f.endswith(".csv") and not f.startswith(PREFIX):
            seen[hashlib.md5(open(os.path.join(SUB, f), "rb").read()).hexdigest()] = f

    print(f"candidates at oof_auc <= {FILLER_MAX_OOF}: {len(df)}   shortfall to cover: {WANT}\n")
    print(f"{'member':26s} {'oof_auc':>12} {'pred_lb':>10} {'bound':>10} "
          f"{'margin e-6':>11} {'distinct':>9}  dup")

    rows = []
    for r in df.itertuples():
        oof = np.load(os.path.join(W, r.dir, f"oof_{r.member}.npy"))
        test = np.load(os.path.join(W, r.dir, f"test_{r.member}.npy"))
        auc = float(roc_auc_score(y, oof))                                   # C2
        if len(test) != len(te):                                             # C4
            fails.append(f"{r.member}: test len {len(test)} != {len(te)}")
            continue
        if not np.isfinite(test).all():
            fails.append(f"{r.member}: non-finite test values")
            continue
        if abs(auc - r.oof_auc) > 1e-9:
            fails.append(f"{r.member}: oof AUC {auc:.10f} != survey {r.oof_auc:.10f}")
            continue

        pred_lb = auc + OFFSET_MEDIAN
        bound = pred_lb + OFFSET_MAX
        margin = TIER - bound
        if margin < MARGIN_MIN:                                              # C3
            fails.append(f"{r.member}: bound {bound:.6f} only {margin*1e6:.1f}e-6 "
                         f"under the tier {TIER}")
            continue

        out = f"{PREFIX}{r.member}.csv"
        path = os.path.join(SUB, out)
        pd.DataFrame({"id": te["id"].to_numpy(), TARGET: test}).to_csv(path, index=False)
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()               # C1
        dup = seen.get(md5, "")
        ndist = int(np.unique(test).size)
        print(f"{r.member:26s} {auc:12.10f} {pred_lb:10.6f} {bound:10.6f} "
              f"{margin*1e6:11.1f} {ndist:9,}  {dup or '-'}")
        if dup:
            os.remove(path)
            skipped.append(f"{r.member}: byte-identical to {dup} -- removed, it would score "
                           f"identically and waste the slot")
            continue
        seen[md5] = out
        rows.append(dict(order=len(rows) + 1, file=out, member=r.member, src=r.dir,
                         oof_auc=auc, n_distinct=ndist, md5=md5,
                         pred_lb=pred_lb, bound=bound, margin_e6=margin * 1e6))

    man = pd.DataFrame(rows)
    man.to_csv(OUT, index=False)
    print(f"\nbuilt {len(man)} fillers -> {os.path.relpath(OUT, W)}")
    if len(man) < WANT:
        fails.append(f"built {len(man)} fillers, the measured shortfall is {WANT}")

    # C5 -- the fatal form of C1, re-derived from the files that actually shipped.
    prior = {h for h, f in seen.items() if not f.startswith(PREFIX)}
    if len(set(man.md5)) != len(man):
        fails.append("two shipped fillers share an md5")
    for r in man.itertuples():
        if r.md5 in prior:
            fails.append(f"{r.file}: md5 collides with a file already in submissions/")
        if hashlib.md5(open(os.path.join(SUB, r.file), "rb").read()).hexdigest() != r.md5:
            fails.append(f"{r.file}: on-disk md5 does not match the manifest")

    for s_ in skipped:
        print(f"  SKIP  {s_}")
    for f in fails:
        print(f"  FAIL  {f}")
    print(f"\nFAILURES {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
