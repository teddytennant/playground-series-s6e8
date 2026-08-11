"""How many members does the stack actually need, and how should they be chosen?

The pack's correlation matrix puts 95.39% of the variance in PC1 and leaves only ~55
directions above 1e-4 of lambda_max. If the stack's whole job lives in a thin tail, then
two questions have answers worth knowing and neither has ever been measured here:

1. **Does CV saturate in k?** A 149-member logistic stack on 691k rows is nowhere near
   overfitting on a naive n/p count, but the *effective* p is far below 149 and the
   columns are correlated 0.99+. If k=50 matches k=148, the extra members are decoration
   and the deadline pick should prefer the smaller, better-conditioned model.

2. **Which ordering wins at matched k -- solo AUC, or decorrelation?** Slot 3 measured
   this at GROUP level (decorrelated groups were worth ~1.5x more per member) but never
   at member level, and the number that finding leaned on turned out to be computed in
   the wrong space. This is the direct test:

     - `auc`    : greedily take the highest solo OOF AUC remaining.
     - `decorr` : greedily take the member whose max |corr| against those ALREADY CHOSEN
                  is smallest -- a diversity ordering that ignores solo strength.
     - `random` : the control, so "any k members" is separated from "these k members".

Everything is paired: both orderings are fitted and scored on identical rows, so split
noise cancels the same way it does in member_eval.py. Selection uses only the FIT half's
labels for solo AUC; the correlation ordering uses no labels at all.

    member_select.py --reps 2 --ks 10,25,50,100,148
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE  # noqa: E402

# byte-identical to bolt_xgb_d7_alt1 in both OOF and test (max|diff| == 0.0). An exactly
# collinear column pair; the smallest eigenvalue of the correlation matrix is 1.1e-16.
DUP = "bolt_xgb_d7_alt2"


def orderings(Z, y, fit_rows, names):
    """Return {name: index order}. Only fit_rows labels are ever touched."""
    n = len(names)
    Zc = (Z - Z.mean(0, keepdims=True)).astype(np.float64)
    Zc /= np.linalg.norm(Zc, axis=0, keepdims=True) + 1e-12
    C = np.abs(Zc.T @ Zc)

    auc = np.array([roc_auc_score(y[fit_rows], Z[fit_rows, j]) for j in range(n)])
    by_auc = list(np.argsort(-auc))

    # greedy max-diversity: seed on the strongest member, then repeatedly take whichever
    # remaining member is least correlated with the set already chosen.
    chosen = [int(np.argmax(auc))]
    remaining = set(range(n)) - set(chosen)
    while remaining:
        r = np.fromiter(remaining, int)
        worst = C[np.ix_(r, chosen)].max(1)
        chosen.append(int(r[int(np.argmin(worst))]))
        remaining.discard(chosen[-1])

    rng = np.random.default_rng(0)
    rnd = list(rng.permutation(n))
    return {"auc": by_auc, "decorr": chosen, "random": rnd}, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ks", default="10,25,50,100,148")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    ks = [int(k) for k in a.ks.split(",")]

    d = np.load(os.path.join(CACHE, "meta_hybrid.npz"), allow_pickle=True)
    names = list(d["names"])
    keep = [i for i, nm in enumerate(names) if nm != DUP]
    names = [names[i] for i in keep]
    Z, y = d["Z"][:, keep], d["y"]
    print(f"{len(names)} members after dropping {DUP} (exact duplicate)", flush=True)

    rows = []
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        ords, auc = orderings(Z, y, iA, names)
        if rep == 0:
            print("\nfirst 12 by each ordering")
            for k, o in ords.items():
                print(f"  {k:7s} " + ", ".join(names[i] for i in o[:12]))
            print()
        for kind, order in ords.items():
            for k in ks:
                cols = order[:k]
                t0 = time.time()
                m = LogisticRegression(max_iter=3000, C=a.C).fit(Z[np.ix_(iA, cols)], y[iA])
                s = roc_auc_score(y[iB], m.decision_function(Z[np.ix_(iB, cols)]))
                rows.append(dict(rep=rep, kind=kind, k=k, auc=s))
                print(f"  [rep {rep}] {kind:7s} k={k:3d}  {s:.6f}  ({time.time()-t0:.0f}s)",
                      flush=True)

    df = pd.DataFrame(rows)
    p = df.pivot_table(index="k", columns="kind", values="auc", aggfunc="mean")
    print("\nheld-out AUC, mean over reps (identical rows for every cell)")
    print(p.to_string(float_format="%.6f"))

    print("\npaired decorr - auc, per k")
    for k in ks:
        dd = (df[(df.k == k) & (df.kind == "decorr")].set_index("rep").auc
              - df[(df.k == k) & (df.kind == "auc")].set_index("rep").auc)
        ok = "consistent" if (dd > 0).all() or (dd < 0).all() else "SIGN FLIPS"
        print(f"  k={k:3d}  {dd.mean():+.6f}  [{ok}]")

    full = p.max().max()
    print(f"\nbest cell {full:.6f}")
    for k in ks:
        print(f"  k={k:3d} best ordering reaches {p.loc[k].max():.6f} "
              f"({p.loc[k].max()-full:+.6f} vs best cell)")
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "member_select_results.csv"), index=False)


if __name__ == "__main__":
    main()
