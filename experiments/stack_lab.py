"""Cheap bench for combiner experiments over the member matrix.

`agent/stack.py` rebuilds the member matrix from ~150 .npy pairs and re-applies the
`hybrid` transform on every invocation, which costs a couple of minutes before any
experiment starts. This caches the transformed matrix once (cache/meta_<transform>.npz)
so a sweep costs only the logistic regressions.

Two evaluations, both from stack.py's methodology:

* `--paired`  fit on half the rows, score on the other half, repeated on fixed splits.
  Variants are compared by the PAIRED difference on identical rows so split noise
  cancels. This is the selection instrument -- it resolves differences the cross-fit
  cannot.
* `--crossfit` the frozen 5 folds. The headline number, comparable to a member's OOF AUC,
  but ~5x the cost and it carries its own noise floor of ~5e-5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, transform  # noqa: E402


def build(kind, drop, refresh=False):
    """Member matrix on the stacker's scale, cached to disk."""
    p = os.path.join(CACHE, f"meta_{kind}.npz")
    if os.path.exists(p) and not refresh:
        d = np.load(p, allow_pickle=True)
        return list(d["names"]), d["Z"], d["Zt"], d["y"]
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = (os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(drop))
    print(f"{len(names)} members loaded", flush=True)
    Z, Zt = transform(O, T, kind)
    Z, Zt = Z.astype("float32"), Zt.astype("float32")
    np.savez(p, names=np.array(names, object), Z=Z, Zt=Zt, y=y)
    return names, Z, Zt, y


def paired(Z, y, cols, variants, reps=3):
    """{label: [held-out AUC per split]} for each variant, on identical row splits."""
    out = {k: [] for k in variants}
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        for label, (cs, C) in variants.items():
            idx = cols if cs is None else cs
            m = LogisticRegression(max_iter=3000, C=C).fit(Z[iA][:, idx], y[iA])
            out[label].append(roc_auc_score(y[iB], m.decision_function(Z[iB][:, idx])))
    return out


def crossfit(Z, y, C, idx=None):
    folds = get_folds(y)
    mo = np.zeros(len(y))
    X = Z if idx is None else Z[:, idx]
    for itr, iva in folds:
        m = LogisticRegression(max_iter=3000, C=C).fit(X[itr], y[itr])
        mo[iva] = m.decision_function(X[iva])
    return roc_auc_score(y, mo), mo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--drop", default=",".join(DEFAULT_DROP))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sweep-c", default="", help="comma-separated C values, paired eval")
    ap.add_argument("--crossfit", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    t0 = time.time()
    names, Z, Zt, y = build(a.transform, set(filter(None, a.drop.split(","))), a.refresh)
    print(f"matrix {Z.shape} ({time.time()-t0:.0f}s)", flush=True)

    if a.sweep_c:
        Cs = [float(x) for x in a.sweep_c.split(",")]
        cols = np.arange(Z.shape[1])
        res = paired(Z, y, cols, {f"C={C}": (None, C) for C in Cs}, a.reps)
        base = res[f"C={Cs[0]}"]
        print(f"\npaired 50/50, {a.reps} splits ({time.time()-t0:.0f}s)")
        for k, v in res.items():
            d = np.array(v) - np.array(base)
            print(f"  {k:>10s}  mean {np.mean(v):.6f}   vs {list(res)[0]}: "
                  f"{d.mean():+.6f} +/- {d.std(ddof=1):.6f}", flush=True)

    if a.crossfit:
        cv, _ = crossfit(Z, y, a.crossfit)
        print(f"\ncross-fitted C={a.crossfit}: {cv:.6f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
