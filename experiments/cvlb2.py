"""The corrected resolvability question, plus what the CV->LB residual is made of.

`cvlb.py` estimated "how small a gap can the public slice resolve" by bootstrapping ONE
file's AUC at the slice's size and reading off its sd (5.4e-4 at 20%). That is the wrong
null and it overstates the threshold by more than an order of magnitude. The public slice
is **fixed**: both candidates are scored on the identical rows, and the two candidates
correlate ~0.999. What matters is the sd of the *paired difference*

    AUC_a(slice) - AUC_b(slice)

which is tiny precisely because the shared slice noise cancels. Same mistake shape as the
chi2/df null in the 2026-08-11 error-analysis entry: an unmatched control.

Two measurements:

1.  **Paired slice bootstrap.** Draw subsamples at the public slice's size and take the sd
    of the paired AUC difference for real candidate pairs. This is the honest answer to
    "could the LB have separated these two files".
2.  **Residual structure.** Regress LB on CV over the 20 scored files and group the
    residual by transform family. If `logit` files sit systematically above the line, the
    CV->LB gap is not a constant offset plus noise, and any reasoning that treats LB as a
    monotone read-out of CV is wrong in a specific, nameable way.

    cvlb2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, SUB, TARGET  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302
REPS = 300

# (a, b) pairs spanning the CV gaps we actually have to decide between.
PAIRS = [("blend158_h3", "blend156"),          # 6e-6  -- the deadline decision
         ("blend158_h3", "blend156_h3"),       # 2e-6  -- the tightest
         ("blend158_h3", "blend150fx_logit"),  # 98e-6 -- across the logit anomaly
         ("blend156", "stack_pub74_logit")]    # 400e-6 -- the whole span


def fam(name):
    for k in ("_h3", "_hybrid", "_rankraw", "_rescale", "_logit"):
        if name.endswith(k):
            return k[1:]
    return "ens4"


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).to_numpy()
    n = len(y)

    print("=== 1. paired slice bootstrap: sd of AUC_a - AUC_b on a SHARED slice ===")
    print(f"{'pair':>44s} {'CV gap':>9s} {'20% sd':>9s} {'50% sd':>9s} {'20% P(order flips)':>19s}")
    for a, b in PAIRS:
        oa, ob = (np.load(os.path.join(SUB, f"oof_{x}.npy")) for x in (a, b))
        gap = roc_auc_score(y, oa) - roc_auc_score(y, ob)
        out = {}
        for frac in (0.20, 0.50):
            m = int(round(N_TEST * frac))
            rng = np.random.default_rng(7)
            d = np.empty(REPS)
            for i in range(REPS):
                idx = rng.choice(n, size=m, replace=False)
                yy = y[idx]
                d[i] = roc_auc_score(yy, oa[idx]) - roc_auc_score(yy, ob[idx])
            out[frac] = d
        flip = float((out[0.20] * np.sign(gap) < 0).mean())
        print(f"{a + ' vs ' + b:>44s} {gap:+9.6f} {out[0.20].std(ddof=1):9.6f} "
              f"{out[0.50].std(ddof=1):9.6f} {flip:19.1%}")
    print("  (cvlb.py's unpaired estimate was 5.4e-4 at 20% -- an unmatched null; the")
    print("   shared-slice noise cancels almost entirely between correlated candidates.)")

    print("\n=== 2. is the CV->LB gap a constant, or does the transform shift it? ===")
    df = pd.read_csv(os.path.join(HERE, "audit_results.csv")).dropna(subset=["cv", "lb"]).copy()
    df["fam"] = df.name.map(fam)
    df["gap"] = df.lb - df.cv
    b, a0 = np.polyfit(df.cv, df.lb, 1)
    df["resid"] = df.lb - (a0 + b * df.cv)
    print(f"{'family':>10s} {'n':>3s} {'mean LB-CV':>11s} {'mean resid':>11s} {'residuals':>34s}")
    for f, g in df.groupby("fam"):
        rs = " ".join(f"{v:+.5f}" for v in sorted(g.resid))
        print(f"{f:>10s} {len(g):3d} {g.gap.mean():+11.6f} {g.resid.mean():+11.6f} {rs:>34s}")
    print(f"\n  residual sd overall {df.resid.std(ddof=1):.6f}; LB quantisation 1e-5, so a")
    print(f"  residual of {df.resid.abs().max():.5f} is {df.resid.abs().max() / 1e-5:.1f} LB ticks -- real, not rounding.")

    df.sort_values("resid").to_csv(os.path.join(HERE, "cvlb_residuals.csv"), index=False)


if __name__ == "__main__":
    main()
