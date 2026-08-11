"""Does the public LB carry any information about CV, inside the regime we operate in?

`audit.py` reports pearson +0.958 over all 20 scored files. That number is worthless: it
is carried entirely by three early entries at CV 0.9696 sitting 30e-5 below the rest. The
question that matters is whether, among the 17 files in the dense cluster (CV >= 0.96995,
LB 0.97099-0.97106), the public slice can rank two candidates at all.

Three things are computed here, all of them cheap:

1.  Correlation restricted to the cluster, and the same statistic with the three outliers
    re-included, so the inflation is visible side by side.
2.  The **resolution argument**: the public LB is rounded to 5dp and is a fixed subsample
    of the 296,302-row test set. Its own sampling sd is estimated by bootstrapping the
    AUC of one submission's OOF vector at the public slice's size, which bounds how small
    a CV gap the LB could resolve even in principle.
3.  A **paired-rank test**: for every pair of scored files, does LB order them the same
    way as CV? Reported as a function of how large the CV gap is. If pairs separated by
    <2e-5 of CV are ordered at chance, then LB feedback on this workspace's top
    candidates is noise and must not be allowed to touch the deadline pick.

    cvlb.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, SUB, TARGET  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302
# Kaggle does not publish the public fraction for this competition. 20% is the Playground
# default; both 20% and 50% are reported so the conclusion does not hinge on the guess.
PUBLIC_FRACS = (0.20, 0.50)
CLUSTER = 0.96995


def main():
    df = pd.read_csv(os.path.join(HERE, "audit_results.csv")).dropna(subset=["cv", "lb"])
    print(f"{len(df)} scored files, CV {df.cv.min():.6f}-{df.cv.max():.6f}, "
          f"LB {df.lb.min():.5f}-{df.lb.max():.5f}")

    for label, sub in (("all scored", df),
                       (f"cluster CV>={CLUSTER}", df[df.cv >= CLUSTER])):
        c, l = sub.cv.to_numpy(), sub.lb.to_numpy()
        if len(sub) < 3:
            continue
        p = np.corrcoef(c, l)[0, 1]
        s = np.corrcoef(rankdata(c), rankdata(l))[0, 1]
        print(f"\n{label:24s} n={len(sub):2d}  pearson {p:+.3f}  spearman {s:+.3f}  "
              f"CV span {c.max() - c.min():.6f}  LB span {l.max() - l.min():.5f}")

    # ---- 2. what the public slice can resolve ----
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).to_numpy()
    o = np.load(os.path.join(SUB, "oof_blend158_h3.npy"))
    print("\n=== how small a gap could the public slice resolve? ===")
    rng = np.random.default_rng(42)
    for frac in PUBLIC_FRACS:
        n = int(round(N_TEST * frac))
        # Bootstrap AUC at the public slice's size using the OOF vector as a stand-in for
        # a submission's test-side behaviour: same model, same base rate, same n.
        aucs = np.empty(200)
        for i in range(200):
            idx = rng.choice(len(y), size=n, replace=False)
            aucs[i] = roc_auc_score(y[idx], o[idx])
        sd = aucs.std(ddof=1)
        print(f"  public = {frac:.0%} of test ({n:,} rows): AUC sd {sd:.6f}  "
              f"-> two files must differ by ~{2.8 * sd:.6f} to be separated at 95%")
    print(f"  our top four candidates span {df.cv.max() - df.cv.nlargest(4).min():.6f} of CV;"
          f" the LB print is quantised at 0.00001")

    # ---- 3. paired-rank agreement as a function of CV gap ----
    print("\n=== does LB order pairs the way CV does? ===")
    recs = []
    for (i, a), (j, b) in itertools.combinations(df.iterrows(), 2):
        gap = abs(a.cv - b.cv)
        hi, lo = (a, b) if a.cv > b.cv else (b, a)
        recs.append((gap, np.sign(hi.lb - lo.lb)))
    r = pd.DataFrame(recs, columns=["gap", "sign"])
    bins = [(0, 2e-5), (2e-5, 5e-5), (5e-5, 1e-4), (1e-4, 3e-4), (3e-4, 1.0)]
    print(f"{'CV gap':>18s} {'pairs':>6s} {'agree':>6s} {'tie':>5s} {'disagree':>9s} {'agree% of decided':>18s}")
    for lo, hi in bins:
        s = r[(r.gap >= lo) & (r.gap < hi)]["sign"]
        if not len(s):
            continue
        ag, ti, di = (s > 0).sum(), (s == 0).sum(), (s < 0).sum()
        dec = ag + di
        pct = f"{ag / dec:6.1%}" if dec else "     -"
        print(f"{lo:8.0e}-{hi:8.0e} {len(s):6d} {ag:6d} {ti:5d} {di:9d} {pct:>18s}")
    s = r["sign"]
    ag, di = (s > 0).sum(), (s < 0).sum()
    print(f"{'ALL':>18s} {len(s):6d} {ag:6d} {(s == 0).sum():5d} {di:9d} "
          f"{ag / (ag + di):17.1%}")


if __name__ == "__main__":
    main()
