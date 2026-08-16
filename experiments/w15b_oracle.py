"""w15b step 2b -- the nonparametric exact-cell oracle, split out of w15b_surface.py.

Section 3 of w15b_surface.py divided 0/0 at smoothing m=0 for cells with no training rows
in a given fold. Re-run standalone with the unseen-cell fallback made explicit, plus the
LEARNING CURVE the surface script did not have.

The mission asks: "what AUC would a model that knew the exact cell probabilities get?"
That number is the limit of this curve as the training half grows, so measure the curve
rather than one point. Cell rates are estimated from a fraction f of each fold's training
rows and scored on the held-out fold, f swept over a decade. If the curve has flattened by
f=1.0 the plug-in is already the oracle; if it is still climbing, the single-f number
understates the oracle and the extrapolation says by how much.

Rows whose cell was never seen in training fall back to the training prior -- that is what
an honest oracle must do, and at f=0.05 it happens often, which is exactly the estimation
cost the curve is measuring.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DAILY, SUB, TARGET, get_folds, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
WEEKEND = "weekend_screen_time"
BLEND = "blend159av_h3"
SEED = 20260815


def cell_key(df, cols):
    key = None
    for c in cols:
        v = df[c].to_numpy(np.float64)
        _, inv = np.unique(np.where(np.isnan(v), np.inf, v), return_inverse=True)
        inv = inv.astype(np.int64)
        key = inv if key is None else pd.factorize(key * (inv.max() + 1) + inv)[0]
    return np.asarray(key, dtype=np.int64)


def oracle_oof(key, y, folds, m, frac, rng):
    K = int(key.max()) + 1
    oof = np.empty(len(y))
    unseen = 0
    for tr_i, va_i in folds:
        if frac < 1.0:
            tr_i = rng.choice(tr_i, size=int(len(tr_i) * frac), replace=False)
        cnt = np.bincount(key[tr_i], minlength=K).astype(np.float64)
        tot = np.bincount(key[tr_i], weights=y[tr_i], minlength=K)
        prior = y[tr_i].mean()
        rate = (tot + m * prior) / (cnt + m)          # m > 0 => always defined
        v = rate[key[va_i]]
        miss = cnt[key[va_i]] == 0
        v[miss] = prior
        unseen += int(miss.sum())
        oof[va_i] = v
    return oof, unseen / len(y)


def main():
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    stack = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    soc = tr[SOCIAL].to_numpy(np.float64)
    day = tr[DAILY].to_numpy(np.float64)
    both = ~np.isnan(soc) & ~np.isnan(day)
    print(f"pack {BLEND}: AUC(all) {roc_auc_score(y, stack):.6f}   "
          f"AUC(both-observed) {roc_auc_score(y[both], stack[both]):.6f}")
    print(f"both drivers observed: {both.sum():,} rows\n", flush=True)

    out = {}
    for setname, cols in (("social+daily", [SOCIAL, DAILY]),
                          ("social+daily+weekend", [SOCIAL, DAILY, WEEKEND])):
        key = cell_key(tr, cols)
        print(f"=== {setname}: {key.max() + 1:,} cells ===", flush=True)
        for frac in (0.05, 0.125, 0.25, 0.5, 1.0):
            best = None
            for m in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
                rng = np.random.default_rng(SEED)
                o, miss = oracle_oof(key, y, folds, m, frac, rng)
                a_all = roc_auc_score(y, o)
                a_both = roc_auc_score(y[both], o[both])
                if best is None or a_both > best[1]:
                    best = (m, a_both, a_all, miss)
            m, a_both, a_all, miss = best
            print(f"  f={frac:5.3f}  n_train~{int(len(y) * 0.8 * frac):>7,}  best m={m:5.1f}"
                  f"  AUC(all) {a_all:.6f}  AUC(both-observed) {a_both:.6f}"
                  f"  unseen-cell rows {miss:.4f}", flush=True)
            out[f"{setname}|f{frac}"] = dict(frac=frac, m=m, auc_all=float(a_all),
                                             auc_both=float(a_both), unseen=float(miss))
        print()

    with open(os.path.join(ROOT, "experiments", "w15b_oracle.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote experiments/w15b_oracle.json")


if __name__ == "__main__":
    main()
