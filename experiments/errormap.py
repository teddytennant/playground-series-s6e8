"""Descriptive half of the angle: WHERE is the stack wrong, and is that fixable?

`erroran.py`/`resid_boost2.py` ask whether anything is recoverable. This asks the prior
question the journal has never written down in one place: where does the AUC actually get
lost? Two decompositions.

1. WITHIN-SEGMENT AUC. Global AUC compares every row against every other row, including
   rows from segments of wildly different difficulty. Splitting it out says which
   populations the stack ranks well and which it does not.

2. WITHIN vs BETWEEN. Pooling only the pairs that fall INSIDE a segment gives the ranking
   quality the members are actually responsible for; the gap to global AUC is what the
   cross-segment scale is worth. If pooled-within is much higher than global, the loss is
   in regrading segments against each other -- a calibration problem, cheap to attack.
   If they are equal, the difficulty is inside the segments and no regrading helps.

The second is the one that decides whether a fix exists, and it is the one the journal's
missingness work (`iso_regime.py`, -0.000085) answered for exactly one segmentation.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, COMP, DAILY, NUM, SUB, TARGET, load_raw  # noqa: E402


def pooled_within(y, s, g):
    """AUC computed only over pairs inside the same segment, weighted by pair count."""
    num = den = 0.0
    for v in np.unique(g):
        m = g == v
        yy, ss = y[m], s[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(yy, ss)
        den += w
    return num / den, den


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    s = np.load(os.path.join(SUB, "oof_blend158_h3.npy"))
    g_auc = roc_auc_score(y, s)
    print(f"blend158_h3 global OOF AUC {g_auc:.6f}   n={len(y):,}\n")

    d = tr[DAILY].to_numpy("float64")
    comp = tr[COMP].to_numpy("float64")
    segs = {
        "n_missing_all": np.minimum(tr[NUM + CAT].isna().sum(1).to_numpy(), 5),
        "n_screen_missing": tr[[DAILY] + COMP].isna().sum(1).to_numpy(),
        "stress_level": tr["stress_level"].fillna("NA").to_numpy(),
        "academic_work_impact": tr["academic_work_impact"].fillna("NA").to_numpy(),
        "gender": tr["gender"].fillna("NA").to_numpy(),
        "daily_band": np.where(np.isfinite(d), np.clip(np.floor(d / 2), 0, 6), -1),
        "age_band": np.where(tr["age"].notna(),
                             np.clip(np.floor(tr["age"].to_numpy() / 10), 1, 6), -1),
        "other_screen_band": np.where(
            np.isfinite(d) & np.isfinite(comp).all(1),
            np.clip(np.floor(d - np.nansum(comp, axis=1)), 0, 5), -1),
    }

    print(f"{'segmentation':22s} {'pooled within':>13s} {'vs global':>10s}"
          f" {'pair share':>11s}")
    for name, g in segs.items():
        pw, den = pooled_within(y, s, np.asarray(g).astype(str))
        tot = int(y.sum()) * int((1 - y).sum())
        print(f"{name:22s} {pw:13.6f} {pw - g_auc:+10.6f} {den/tot:11.3f}")

    print("\nper-level detail (size, base rate, within-level AUC):")
    for name in ["n_missing_all", "n_screen_missing", "stress_level", "daily_band",
                 "other_screen_band"]:
        g = np.asarray(segs[name]).astype(str)
        print(f"\n  {name}")
        for v in sorted(np.unique(g)):
            m = g == v
            yy = y[m]
            if yy.sum() == 0 or (1 - yy).sum() == 0:
                continue
            print(f"    {v:>8s}  n={m.sum():7d} ({m.mean():5.1%})  base {yy.mean():.4f}"
                  f"  AUC {roc_auc_score(yy, s[m]):.6f}")


if __name__ == "__main__":
    main()
