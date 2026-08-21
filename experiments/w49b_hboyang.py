"""w49b -- CHARACTERISE `hboyang_mix` before it is sent. Characterisation only.

⚠ THIS DOES NOT TOUCH w48d'S REGISTERED RULE. That rule (predicted 0.97123, >=0.97116 HONEST,
<=0.97080 INFLATED) was registered before the artefact existed and stands exactly as written.
Nothing here re-prices it. The point is to know WHAT the file is before its score arrives, so
that tomorrow's reading is an interpretation rather than a guess.

WHY. w48d convicted this member on OUTLIER MAGNITUDE ALONE -- standalone OOF AUC 0.9701816,
+886e-6 clear of the best of the other 176 members, above our own 217-member stack -- and w48
was explicit that the direct leakage diagnostic did NOT convict. "Anomalously high AUC" has at
least three explanations and they imply different things:

  H-LEAK   its OOF is not honestly out-of-fold w.r.t. our split (target leaked).  -> drop it.
  H-BLEND  it is a public BLEND ("mix"), not a base model, whose weights were fitted on the
           full OOF set. Mildly optimistic, no leakage, and a perfectly ordinary reason to
           out-AUC any single base model.                                        -> not a scandal,
                                                                                    but still not
                                                                                    a base member.
  H-REAL   it is simply a very strong model.                                     -> the import
                                                                                    line closed early.

THE STRUCTURAL FACT THAT STARTED THIS. The OOF vector has 276,547 distinct values over 691,369
rows -- roughly 2.5 rows per distinct value -- while the TEST vector is fully distinct
(296,302/296,302). Both are rank-uniform (mean exactly 0.5000000). A blend of many continuous
models breaks ties almost surely, so heavy ties in the OOF are evidence AGAINST H-BLEND and
toward a model with a coarse, discrete score -- a lookup or aggregation over feature cells.
That matters because this dataset is heavily discretised: w15b found 722 `social` cells
covering 99.99% of rows, largest cell 133,995 rows.

⚠ AND NOTE THE ASYMMETRY, which is the single most suspicious thing on the page: the OOF is
tied and the TEST vector is not. A model applied identically to both partitions should tie in
both. Different tie structure across the two partitions means the two vectors were NOT produced
by the same procedure.

READINGS FIXED BEFORE LOOKING (this is characterisation, so these label rather than decide):
  T1  tie-size distribution. A few huge groups -> lookup/aggregation. Mostly pairs -> rounding.
  T2  reconstruction R^2 of hboyang's OOF from the other members we hold, in rank space.
      R^2 >= 0.99 supports H-BLEND; low R^2 leaves H-LEAK / H-REAL.
  T3  within-tie-group AUC. For an honest model, rows it CANNOT distinguish (identical score)
      must have labels unpredictable from that score -- by construction AUC is 0.5 inside a
      group. So instead: does the tie-group MEAN label track the group's score monotonically,
      and how much of the total AUC lives between groups vs within? Purely descriptive.
  T4  per-fold AUC on OUR frozen folds. A foreign-split OOF is still globally valid, so
      uniformity here is expected under ALL THREE hypotheses; a wild split is a red flag only.
"""
from __future__ import annotations

import glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, N_SPLITS, SEED                 # noqa: E402
from sklearn.model_selection import StratifiedKFold             # noqa: E402

M = "hboyang_mix"
y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
o = np.load(os.path.join(DATA, "ext_members16", f"oof_{M}.npy")).ravel()
t = np.load(os.path.join(DATA, "ext_members16", f"test_{M}.npy")).ravel()
out = {"member": M, "oof_auc": float(roc_auc_score(y, o))}

print("=" * 90); print(f"w49b  characterising `{M}`  (OOF AUC {out['oof_auc']:.10f})"); print("=" * 90)

# ---------------------------------------------------------------- T1  tie structure
u, inv, cnt = np.unique(o, return_inverse=True, return_counts=True)
ut, cntt = np.unique(t, return_counts=True)
out["oof_distinct"], out["test_distinct"] = int(len(u)), int(len(ut))
print(f"\nT1  TIE STRUCTURE")
print(f"    OOF  {len(o):,} rows -> {len(u):,} distinct   (max group {cnt.max():,}, "
      f"{(cnt>1).sum():,} groups have ties, {cnt[cnt>1].sum():,} rows tied)")
print(f"    TEST {len(t):,} rows -> {len(ut):,} distinct   (max group {cntt.max():,})")
for k in (1, 2, 3, 4, 5, 10, 100, 1000):
    print(f"      groups of size {k:>5}: {int((cnt==k).sum()):>9,}   "
          f"(>= {k}: {int((cnt>=k).sum()):>9,})")
out["oof_max_group"] = int(cnt.max()); out["test_max_group"] = int(cntt.max())

# ---------------------------------------------------------------- T4  per-fold AUC
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
fa = [float(roc_auc_score(y[v], o[v])) for _, v in skf.split(np.zeros(len(y)), y)]
print(f"\nT4  PER-FOLD AUC on our frozen folds: " + " ".join(f"{a:.6f}" for a in fa))
print(f"    spread {max(fa)-min(fa):.6f}  (expected small under ALL hypotheses; only a wild "
      f"split would be a flag)")
out["fold_auc"] = fa

# ---------------------------------------------------------------- T3  between vs within groups
gy = np.bincount(inv, weights=y, minlength=len(u))
gn = np.bincount(inv, minlength=len(u))
gmean = gy / gn
# AUC achievable from the group score alone == the actual AUC (score is constant in a group),
# so decompose the label variance instead: how much is explained BETWEEN groups?
tot = y.var()
between = float(np.average((gmean - y.mean()) ** 2, weights=gn))
print(f"\nT3  LABEL VARIANCE DECOMPOSITION over tie groups")
print(f"    total {tot:.6f}   between-group {between:.6f}   = {between/tot*100:.2f}% explained")
print(f"    monotone check: Spearman(group score, group mean label) = "
      f"{pd.Series(u).corr(pd.Series(gmean), method='spearman'):.6f}")
out["between_frac"] = between / tot

# ---------------------------------------------------------------- T2  reconstruction
print(f"\nT2  RECONSTRUCTION from the other members we hold (rank space)")
files = sorted(set(glob.glob(os.path.join(DATA, "ext_members*", "oof_*.npy"))))
files = [f for f in files if M not in os.path.basename(f)]
names, cols = [], []
for f in files:
    try:
        v = np.load(f).ravel()
    except Exception:
        continue
    if len(v) != len(y):
        continue
    n = os.path.basename(f)[4:-4]
    if n in names:
        continue
    names.append(n); cols.append(v.astype(np.float32))
print(f"    {len(names)} usable member OOF vectors")
X = np.column_stack(cols)
rng = np.random.default_rng(0)
fit = rng.choice(len(y), 200_000, replace=False)
hold = np.setdiff1d(np.arange(len(y)), fit)
def rk(a):
    return (pd.Series(a).rank().values / len(a)).astype(np.float32)
Xr = np.column_stack([rk(X[:, j]) for j in range(X.shape[1])])
yr = rk(o)
A = np.column_stack([Xr[fit], np.ones(len(fit), np.float32)])
beta, *_ = np.linalg.lstsq(A, yr[fit], rcond=None)
pred = np.column_stack([Xr[hold], np.ones(len(hold), np.float32)]) @ beta
r2 = 1 - ((yr[hold] - pred) ** 2).sum() / ((yr[hold] - yr[hold].mean()) ** 2).sum()
print(f"    held-out R^2 of hboyang's OOF on {len(names)} members: {r2:.6f}")
top = np.argsort(-np.abs(beta[:-1]))[:8]
print(f"    largest weights: " + ", ".join(f"{names[j]} {beta[j]:+.3f}" for j in top))
out["reconstruct_r2"] = float(r2)
out["reconstruct_n"] = len(names)

print(f"""
READING
  T2 R^2 = {r2:.4f} -- {'>= 0.99, supports H-BLEND' if r2 >= 0.99 else 'below 0.99; H-BLEND is NOT supported by reconstruction'}
  T1 max OOF tie group {cnt.max():,} vs max TEST tie group {cntt.max():,}.
  {'⚠ THE TWO PARTITIONS HAVE DIFFERENT TIE STRUCTURE -- they were not produced the same way.' if cnt.max() > 1 and cntt.max() == 1 else ''}
""")
json.dump(out, open(os.path.join(HERE, "w49b_hboyang.json"), "w"), indent=1)
