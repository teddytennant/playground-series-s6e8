"""What is the cdfd_* member worth INSIDE a stack, and is it the channel or just a body?

w186a measured +87.7e-6 solo. A solo gain is not a stack gain: the pack already holds
~90 correlated members, and a member that improves by re-deriving what its neighbours
already say buys nothing once a combiner sees them all.

THE CONTROL THAT MAKES THIS READABLE. `w186_lgbmfrac_ctrl` is a near-duplicate of the
enrolled `lgbm_tuned_lat_frac` (spearman 0.999999994 on OOF, CV apart by 3.6e-8), so
adding it is adding a BODY and no channel. Adding `w186_lgbmfrac_cdf` is adding the same
body plus nine columns. The difference between those two paired deltas is the channel,
with the "one more correlated member" effect subtracted rather than assumed away.

Paired 50/50 splits, same rows for every variant, so split noise (~2e-4, an order of
magnitude above the effects) cancels.
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

NEW = ["w186_lgbmfrac_ctrl", "w186_lgbmfrac_cdf"]
REPS, C = 8, 1.0

tr, te = load_raw()
y = tr[TARGET].astype(int).to_numpy()
names, O, _ = load_members(y, len(te),
                           extra_dirs=(os.path.join(DATA, "ext_members"),
                                       os.path.join(ROOT, "oof_w186")),
                           drop=DEFAULT_DROP)
idx = {n: i for i, n in enumerate(names)}
missing = [n for n in NEW if n not in idx]
assert not missing, missing
pool = [n for n in names if n not in NEW]
print(f"pool {len(pool)} members;  candidates " +
      " ".join(f"{n}={roc_auc_score(y, O[:, idx[n]]):.7f}" for n in NEW))

Z = to_logit(O)
variants = {"pool": pool, "pool+ctrl": pool + NEW[:1], "pool+cdf": pool + NEW[1:],
            "pool+both": pool + NEW}

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
for c in df.columns:
    if c == "pool":
        continue
    d = (df[c] - df["pool"]) * 1e6
    sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
    print(f"  {c:<12s} {d.mean():+8.2f} +/- {d.std(ddof=1):.2f}  [{sign}]")

d = (df["pool+cdf"] - df["pool+ctrl"]) * 1e6
sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
print(f"\nTHE CHANNEL, body subtracted:  cdf - ctrl  {d.mean():+8.2f} +/- "
      f"{d.std(ddof=1):.2f} e-6  [{sign}]  t={d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):.2f}")
