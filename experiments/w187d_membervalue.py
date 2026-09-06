"""Is the best single model this workspace ever built worth anything to the combiner?

w187_cb5 is +227.9e-6 of OOF AUC over the previous best own member and +420e-6 of public
score. w186's lesson is that neither fact predicts the STACK-layer price: a member that
improves by re-deriving what its ~104 correlated neighbours already say buys nothing once a
combiner sees them all, and w186's own +87.7e-6 solo gain priced at +0.86e-6/member, t=0.91.

⚠ THIS ONE IS NOT THE SAME SHAPE AND THE DIFFERENCE MATTERS. w186's treatment was nine extra
columns inside an ALREADY ENROLLED member, so a near-duplicate body control was needed to
subtract "one more correlated member". `w187_cb5` is a different library on a different frame
that no pool member has ever seen -- the labelled-original block and the exact-value
categoricals are both new -- so there is no body to subtract and the single arm IS the
measurement. The control here is the pool itself.

Paired 50/50 splits, same rows for every variant, so split noise (~2e-4, an order of magnitude
above the effects) cancels.
"""
from __future__ import annotations

import os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, ROOT, TARGET, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, to_logit  # noqa: E402

NEW = ["w187_cb5"]
REPS, C = 8, 1.0

tr, te = load_raw()
y = tr[TARGET].astype(int).to_numpy()
names, O, _ = load_members(y, len(te),
                           extra_dirs=(os.path.join(DATA, "ext_members"),
                                       os.path.join(ROOT, "oof_w187")),
                           drop=DEFAULT_DROP)
idx = {n: i for i, n in enumerate(names)}
missing = [n for n in NEW if n not in idx]
assert not missing, missing
pool = [n for n in names if n not in NEW]
print(f"pool {len(pool)} members;  candidate " +
      " ".join(f"{n}={roc_auc_score(y, O[:, idx[n]]):.7f}" for n in NEW), flush=True)

# how close is it to anything already in? a member that is worth nothing is usually a member
# the pool already holds a copy of, and that is checkable before the fit rather than after.
c = np.abs(np.corrcoef(np.column_stack([O[:, idx[n]] for n in NEW] +
                                       [O[:, idx[m]] for m in pool]).T)[0, 1:])
print(f"max |corr| to any pool member: {c.max():.6f}  ({pool[int(c.argmax())]})", flush=True)

Z = to_logit(O)
variants = {"pool": pool, "pool+cb5": pool + NEW}

rows = []
for rep in range(REPS):
    iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                  .split(np.zeros(len(y)), y))
    r = {}
    for k, mem in variants.items():
        cols = [idx[m] for m in mem]
        m = LogisticRegression(max_iter=3000, C=C).fit(Z[np.ix_(iA, cols)], y[iA])
        r[k] = roc_auc_score(y[iB], m.predict_proba(Z[np.ix_(iB, cols)])[:, 1])
    rows.append(r)
    print(f"  split {rep}: " + "  ".join(f"{k}={v:.6f}" for k, v in r.items()), flush=True)

df = pd.DataFrame(rows)
print("\nPAIRED DELTA vs pool (e-6)")
for col in df.columns:
    if col == "pool":
        continue
    d = (df[col] - df["pool"]) * 1e6
    sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    print(f"  {col:<12s} {d.mean():+8.2f} +/- {d.std(ddof=1):.2f}  [{sign}]  t={t:.2f}")
print("\nReference: the same instrument prices a foreign CatBoost at +10.04e-6/member "
      "(t=8.87) and w186's cdfd_* channel at +0.86e-6 (t=0.91).")
