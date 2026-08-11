"""Does MISSINGNESS carry signal the stack has not extracted?

`erroran.py` at 2048 z-bins put the three columns whose "decimal digit" degenerates to a
pure NaN indicator (age, app_opens, notifications are integers) at the very top with
chi2/df ~2.1, far above the K=13 controls at ~1.06. That comparison is not admissible:
the controls have 13 x-bins and these have 2, and the chi2 bias from estimating the
z-bin rate is a function of cell sparsity, so a K=2 statistic cannot be judged against a
K=13 null.

This builds the matched null. For each of the 12 columns' NaN indicators it also draws
PERMUTED indicators with the IDENTICAL marginal missing rate, so real and control differ
in exactly one thing: whether the indicator knows which rows it marks.

RESEARCH.md records "NA-indicator features -0.00001, missingness is MCAR" from a
member-level ablation. That measured whether a GBM handed the indicator scores better.
This measures something the ablation could not: whether the 158-member STACK, which has
long since absorbed every member's own handling of NaNs, still mis-ranks rows by their
missingness pattern.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, NUM, SUB, TARGET, get_folds, load_raw  # noqa: E402
from erroran import SMOOTH  # noqa: E402

B = 2048


def stat(k, K, zb, y, folds, logitz, base):
    c = zb.astype("int64") * K + k.astype("int64")
    n = np.bincount(c, minlength=B * K).astype("float64").reshape(B, K)
    o = np.bincount(c, weights=y.astype("float64"), minlength=B * K).reshape(B, K)
    rn, ro = n.sum(1), o.sum(1)
    p = np.divide(ro, rn, out=np.full(B, y.mean()), where=rn > 0)
    e = n * p[:, None]
    v = n * (p * (1 - p))[:, None]
    live = v > 1e-9
    chi = float(((o - e)[live] ** 2 / v[live]).sum()) / max(int(live.sum() - B), 1)

    adj = np.zeros(len(y))
    for itr, iva in folds:
        nt = np.bincount(c[itr], minlength=B * K).astype("float64").reshape(B, K)
        ot = np.bincount(c[itr], weights=y[itr].astype("float64"),
                         minlength=B * K).reshape(B, K)
        pt = np.divide(ot.sum(1), nt.sum(1), out=np.full(B, y[itr].mean()),
                       where=nt.sum(1) > 0)
        pc = np.clip((ot + SMOOTH * pt[:, None]) / (nt + SMOOTH), 1e-6, 1 - 1e-6)
        pb = np.clip(pt, 1e-6, 1 - 1e-6)
        adj[iva] = (np.log(pc / (1 - pc)) - np.log(pb / (1 - pb))[:, None]).ravel()[c[iva]]
    return chi, roc_auc_score(y, logitz + adj) - base


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    z = np.load(os.path.join(SUB, "oof_blend158_h3.npy"))
    r = pd.Series(z).rank(method="average").to_numpy() / (len(z) + 1.0)
    logitz = np.log(r / (1 - r))
    zb = np.minimum((pd.Series(z).rank(method="first").to_numpy() - 1)
                    // (len(z) / B), B - 1).astype("int32")
    folds = get_folds(y)
    base = roc_auc_score(y, logitz)
    rng = np.random.default_rng(7)
    print(f"base OOF AUC {base:.6f}   {B} z-bins (~{len(y)//B} rows each)\n", flush=True)

    print(f"{'indicator':30s} {'rate':>7s} {'chi2/df':>8s} {'ctrl':>8s}"
          f" {'dAUC':>10s} {'ctrl dAUC':>10s}")
    rows = []
    for c in NUM + CAT:
        k = tr[c].isna().to_numpy().astype("int32")
        rate = float(k.mean())
        chi, da = stat(k, 2, zb, y, folds, logitz, base)
        cc, cd = [], []
        for _ in range(3):
            kk = (rng.random(len(y)) < rate).astype("int32")
            a, b = stat(kk, 2, zb, y, folds, logitz, base)
            cc.append(a)
            cd.append(b)
        print(f"{c:30s} {rate:7.4f} {chi:8.4f} {np.mean(cc):8.4f}"
              f" {da:+10.6f} {np.mean(cd):+10.6f}", flush=True)
        rows.append(dict(indicator=c, rate=rate, chi2_df=chi, ctrl_chi2_df=np.mean(cc),
                         dAUC=da, ctrl_dAUC=np.mean(cd)))

    # the joint pattern: how many columns are missing, and which BLOCK they fall in
    pat = tr[NUM + CAT].isna().to_numpy()
    joint = {
        "n_missing_all": np.minimum(pat.sum(1), 5).astype("int32"),
        "na_cat_count": np.minimum(tr[CAT].isna().sum(1).to_numpy(), 3).astype("int32"),
    }
    for name, k in joint.items():
        K = int(k.max()) + 1
        chi, da = stat(k, K, zb, y, folds, logitz, base)
        cc, cd = [], []
        for _ in range(3):
            kk = rng.permutation(k)
            a, b = stat(kk, K, zb, y, folds, logitz, base)
            cc.append(a)
            cd.append(b)
        print(f"{name:30s} K={K:<5d} {chi:8.4f} {np.mean(cc):8.4f}"
              f" {da:+10.6f} {np.mean(cd):+10.6f}", flush=True)
        rows.append(dict(indicator=name, rate=np.nan, chi2_df=chi,
                         ctrl_chi2_df=np.mean(cc), dAUC=da, ctrl_dAUC=np.mean(cd)))

    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "experiments", "erroran_na.csv"),
                              index=False)


if __name__ == "__main__":
    main()
