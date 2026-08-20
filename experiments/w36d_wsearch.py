"""w36d -- the handed angle, taken at the one place in this pipeline where a blend weight is
still UNFITTED.

WHERE THE FREE WEIGHTS ACTUALLY ARE
-----------------------------------
Everything below the top of this stack is already weight-searched on OOF: each transform
stack is a logistic regression over ~195 standardised member OOF columns, i.e. 195 weights
fitted out-of-fold. The angle's "search blend weights on OOF" is therefore already done, 195
times over, at level 2.

The exception is the LAST step. `make_h3.py` is an EQUAL rank-average of hybrid/rankraw/
rescale with `logit` dropped, and its docstring says so plainly: "the one blend decision here
with zero fitted parameters". That decision was made at 156 members (2026-08-11) and the pack
is now 195. So this is the one weight vector in the whole object that has never been fitted and
whose justifying measurement is 39 members stale.

THE INSTRUMENT, AND WHY THE OBVIOUS VERSION IS WRONG
----------------------------------------------------
The four per-transform OOF vectors are each cross-fitted, so an AUC computed on them is honest.
The WEIGHTS are not: fitting w on the full OOF and then scoring the same rows is in-sample, and
at 3-4 free parameters against effects of ~5e-6 that is exactly the kind of optimism this
workspace keeps finding after the fact.

So the weights are fitted INSIDE the frozen folds: for fold k, w is fitted on the other four
folds' rows and applied to fold k. The resulting vector is cross-fitted at the weight level and
directly comparable to the equal-weight h3 number the build prints.

BOTH numbers are reported, because their DIFFERENCE is the quantity of interest:
  w_insample  - what a naive weight search would have claimed
  w_xfit      - what it is actually worth
  the gap     - the selection optimism of a 3-4 parameter blend search at this scale, a number
                worth more than the blend itself since it prices every future top-level search.

A NULL IS THE EXPECTED RESULT and is reported as one. Equal weights over near-collinear
members are famously hard to beat; test rho between these transforms is ~0.999.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402

ALL4 = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")
# output suffixes are the ESTABLISHED family labels (`_wh3`, `_w`), not new ones: stdflag's
# suffix rule already classifies those and a novel suffix would silently fall through to `ens4`.


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def fast_auc(y, s):
    """AUC by rank sum. ~3x roc_auc_score here and this is called thousands of times.
    Verified against sklearn to 1e-12 in main() before any search runs."""
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_w(R, y, w0):
    """Maximise AUC over the simplex. SLSQP on -AUC; AUC is piecewise constant but over a
    dense 550k-row sample of near-collinear rankings it is smooth enough at this scale."""
    def neg(w):
        w = np.abs(w)
        s = w.sum()
        if s <= 0:
            return 0.0
        return -fast_auc(y, R @ (w / s))
    r = minimize(neg, w0, method="Nelder-Mead",
                 options=dict(xatol=2e-4, fatol=2e-10, maxiter=150, disp=False))
    w = np.abs(r.x)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="w34_ad195std")
    ap.add_argument("--out", default="w36d_wsearch")
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    _p = np.random.default_rng(0).random(len(y))
    assert abs(fast_auc(y, _p) - roc_auc_score(y, _p)) < 1e-12, "fast_auc disagrees with sklearn"

    O, T, ids = {}, {}, None
    for k in ALL4:
        O[k] = rk(np.load(os.path.join(SUB, f"oof_{a.base}_{k}.npy")))
        d = pd.read_csv(os.path.join(SUB, f"{a.base}_{k}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"{a.base}_{k}.csv id order differs -- refusing to average")
        T[k] = rk(d[TARGET].to_numpy())
        print(f"  {k:8s} solo cross-fitted AUC {roc_auc_score(y, O[k]):.9f}")

    rows = []
    for tag, keys in (("wh3", H3), ("w", ALL4)):
        R = np.column_stack([O[k] for k in keys])
        RT = np.column_stack([T[k] for k in keys])
        n = len(keys)
        w0 = np.full(n, 1.0 / n)

        eq = roc_auc_score(y, R @ w0)

        # in-sample: what a naive search would claim
        w_in = fit_w(R, y, w0)
        ins = roc_auc_score(y, R @ w_in)

        # cross-fitted at the weight level
        pred = np.empty(len(y))
        wk = []
        for tr_i, va_i in folds:
            w = fit_w(R[tr_i], y[tr_i], w0)
            pred[va_i] = R[va_i] @ w
            wk.append(w)
        xf = roc_auc_score(y, pred)
        wk = np.array(wk)

        print(f"\n--- {tag} ({', '.join(keys)}) ---")
        print(f"  equal weights          CV {eq:.10f}   w = {np.round(w0, 4)}")
        print(f"  fitted, IN-SAMPLE      CV {ins:.10f}   w = {np.round(w_in, 4)}"
              f"   (+{(ins - eq) * 1e6:+.3f}e-6, NOT honest)")
        print(f"  fitted, CROSS-FITTED   CV {xf:.10f}   ({(xf - eq) * 1e6:+.3f}e-6 vs equal)")
        print(f"  per-fold w  mean {np.round(wk.mean(0), 4)}  sd {np.round(wk.std(0), 5)}")
        print(f"  ** selection optimism of this search = {(ins - xf) * 1e6:+.3f}e-6 **")
        rows.append(dict(tag=tag, keys=list(keys), equal=eq, insample=ins, xfit=xf,
                         w_insample=[float(x) for x in w_in],
                         w_fold_mean=[float(x) for x in wk.mean(0)],
                         w_fold_sd=[float(x) for x in wk.std(0)],
                         d_xfit_vs_equal=xf - eq, optimism=ins - xf))

        # the shippable object uses the FULL-DATA weights (the honest estimate of their value
        # is xf, but the file itself should use every row -- same convention as the stacker)
        sub = os.path.join(SUB, f"{a.base}_{tag}w.csv")
        pd.DataFrame({"id": ids, TARGET: RT @ w_in}).to_csv(sub, index=False)
        np.save(os.path.join(SUB, f"oof_{a.base}_{tag}w.npy"), pred)
        print(f"  wrote {os.path.basename(sub)} (test uses full-data w; OOF stored is the xfit one)")

    json.dump(rows, open(os.path.join(HERE, f"{a.out}.json"), "w"), indent=1)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"{a.out}.csv"), index=False)
    print(f"\nwrote {a.out}.json")


if __name__ == "__main__":
    main()
