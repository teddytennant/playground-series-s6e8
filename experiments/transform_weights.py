"""Are the four transform stacks best combined with EQUAL weights?

`blend156` is a plain unweighted rank-average of four cross-fitted stacks whose own
cross-fitted CVs are NOT equal:

    logit 0.969962   hybrid 0.970023   rankraw 0.970036   rescale 0.970025

logit is 7e-5 behind the best and still receives a full quarter of the blend. Equal
weighting was never a measured choice -- it was the default, taken because it has no free
parameter. This asks the only question the angle actually leaves open at this level: does
searching the four weights on out-of-fold predictions beat not searching them?

The inputs are the cross-fitted stack OOF vectors already written by `blend_lab.py
--build`, so nothing is refitted here and the whole search costs a few hundred AUC
evaluations. That cheapness is the point: it makes an honest nested evaluation affordable
where refitting 156-member logistic regressions would not be.

HONESTY
-------
Searching 3 free weights on the same 691k OOF rows you then report is optimistic, even at
this ratio. Two instruments, and neither is the naive one:

* `--paired`  choose w on half the rows, score on the other half, compare against equal
  weights on those SAME rows. Repeated on fixed splits so split noise cancels. This
  answers "does searching help", which is the question.
* `--crossfit` choose w on 4 of the frozen 5 folds, apply to the held-out fold, assemble a
  full OOF vector. Directly comparable to the 0.970042 in the journal, and the number a
  deadline decision would be made on.

The per-member rank map rk() is label-free and monotone, so applying it to the full
vector before splitting leaks nothing.

    transform_weights.py --stack blend156 --paired --reps 3
    transform_weights.py --stack blend156 --crossfit --submit-name blend156w
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402

KINDS = ("logit", "hybrid", "rankraw", "rescale")


def rk(v):
    """Rank-normalise to (0,1); ties averaged. Same map blend_lab uses to ensemble."""
    return (rankdata(v) - 0.5) / len(v)


def simplex(k, step):
    """All weight vectors on the k-simplex with coordinates on a `step` grid.

    Stars and bars: k-1 cut positions chosen from 1..n+k-1 partition n units into k
    non-negative parts. The cut range was previously 1..n+k-2, one position short, so every
    row summed to (n-1)/n = 0.95 rather than 1. AUC is scale-invariant so the SCORES that
    grid produced were still valid -- it is a legitimate simplex at step 1/19 up to a
    constant factor -- but equal weights (0.25 each, summing to 1) were not on it, and
    neither was h3 = (0, 1/3, 1/3, 1/3). The search could not return either baseline it was
    being compared against.

    Equal weights are appended when they are still not a grid point. At k=4, step=0.05 they
    now are (5/20 each); at k=3 they are 1/3 and land between grid points regardless.
    """
    n = int(round(1.0 / step))
    out = []
    for cut in itertools.combinations(range(1, n + k), k - 1):
        prev, w = 0, []
        for c in cut:
            w.append(c - prev - 1)
            prev = c
        w.append(n + k - 1 - prev)
        out.append(np.array(w, np.float64) / n)
    g = np.array(out)
    eq = np.ones((1, k)) / k
    if not np.any(np.all(np.isclose(g, eq), axis=1)):
        g = np.vstack([g, eq])
    return g


def sel_auc(yy, s, npos, nneg):
    """AUC used only to CHOOSE weights. Ties are broken arbitrarily instead of averaged.

    ~8x faster than roc_auc_score, which matters at 1,540 grid points x 5 folds. The bias
    this introduces is confined to the selection step: every number this script REPORTS
    goes through roc_auc_score. A criterion that is a hair noisy is an honest thing to
    select on; a reported score that is would not be.
    """
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    return (ranks[yy].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def best_w(R, y, rows, grid, fast=True):
    """Weight vector on `grid` maximising AUC over `rows`. R is (n, k) rank columns."""
    A = np.ascontiguousarray(R[rows])
    yy = y[rows].astype(bool)
    if not fast:
        scores = np.array([roc_auc_score(yy, A @ w) for w in grid])
        return grid[int(np.argmax(scores))], float(scores.max())
    npos = int(yy.sum())
    nneg = len(yy) - npos
    scores = np.array([sel_auc(yy, A @ w, npos, nneg) for w in grid])
    w = grid[int(np.argmax(scores))]
    return w, float(roc_auc_score(yy, A @ w))


def load_stack(stack, kinds):
    """The cross-fitted OOF vector and the test vector for each transform stack."""
    oof, tst = [], []
    for k in kinds:
        op = os.path.join(SUB, f"oof_{stack}_{k}.npy")
        cp = os.path.join(SUB, f"{stack}_{k}.csv")
        if not os.path.exists(op) or not os.path.exists(cp):
            raise SystemExit(f"missing {op} or {cp} -- run blend_lab.py --build first")
        oof.append(np.load(op))
        tst.append(pd.read_csv(cp)[TARGET].to_numpy())
    return np.column_stack([rk(v) for v in oof]), np.column_stack([rk(v) for v in tst])


def named_weights(k, auc):
    """The unsearched baselines. Everything here is fixed before seeing held-out rows."""
    out = {"equal": np.ones(k) / k}
    if k == len(KINDS):
        # drop the transform whose own cross-fit is worst -- the cheapest possible
        # "weight by OOF performance", one bit of information instead of three floats
        w = np.ones(k)
        w[int(np.argmin(auc))] = 0.0
        out["drop_worst"] = w / w.sum()
    for p in (1, 2, 4, 8, 16, 32):
        w = np.maximum(auc - 0.5, 1e-12) ** p
        out[f"auc_p{p}"] = w / w.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="blend156")
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--crossfit", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--submit-name", default=None)
    a = ap.parse_args()

    kinds = [k for k in a.kinds.split(",") if k]
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    R, Rt = load_stack(a.stack, kinds)
    grid = simplex(len(kinds), a.step)
    solo = np.array([roc_auc_score(y, R[:, j]) for j in range(len(kinds))])
    print(f"{a.stack}: {len(kinds)} transform stacks, {len(grid)} grid points "
          f"({time.time()-t0:.0f}s)")
    for k, v in zip(kinds, solo):
        print(f"  {k:>8s}  cross-fitted OOF {v:.6f}")
    print(f"  {'equal':>8s}  cross-fitted OOF {roc_auc_score(y, R.mean(1)):.6f}  "
          f"<- the shipped blend", flush=True)

    if a.paired:
        rows = []
        for rep in range(a.reps):
            iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                          .split(np.zeros(len(y)), y))
            # weights that depend on data are chosen on iA ONLY, including the AUCs the
            # heuristic weightings are built from.
            aucA = np.array([roc_auc_score(y[iA], R[iA, j]) for j in range(len(kinds))])
            cand = named_weights(len(kinds), aucA)
            cand["searched"], _ = best_w(R, y, iA, grid)
            r = {"rep": rep}
            for nm, w in cand.items():
                r[nm] = roc_auc_score(y[iB], R[iB] @ w)
            r["w_searched"] = np.round(cand["searched"], 3).tolist()
            rows.append(r)
            print(f"  [rep {rep}] searched w={r['w_searched']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        df = pd.DataFrame(rows)
        print("\nheld-out AUC per 50/50 split")
        print(df.drop(columns=["w_searched"]).to_string(index=False, float_format="%.6f"))
        print("\nPAIRED vs equal weights (same rows, split noise cancels)")
        for c in df.columns:
            if c in ("rep", "equal", "w_searched"):
                continue
            d = df[c] - df["equal"]
            ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
            print(f"  {c:>10s}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")

    if a.crossfit:
        folds = get_folds(y)
        mo = np.zeros(len(y))
        ws = []
        for i, (itr, iva) in enumerate(folds):
            w, _ = best_w(R, y, itr, grid)
            ws.append(w)
            mo[iva] = R[iva] @ w
            print(f"  fold {i} w={np.round(w, 3).tolist()} ({time.time()-t0:.0f}s)",
                  flush=True)
        cv = roc_auc_score(y, mo)
        print(f"\ncross-fitted searched-weight blend: {cv:.6f}")
        print(f"equal-weight blend on the same rows: {roc_auc_score(y, R.mean(1)):.6f}")
        wbar = np.mean(ws, 0)
        print(f"mean fold weights {dict(zip(kinds, np.round(wbar, 3)))}  "
              f"spread {np.max(ws, 0) - np.min(ws, 0)}")

        if a.submit_name:
            # the shipped file uses weights fitted on ALL OOF rows; `cv` above is the
            # honest estimate of what that procedure scores, not of this exact vector.
            wfull, _ = best_w(R, y, np.arange(len(y)), grid)
            print(f"full-data weights {dict(zip(kinds, np.round(wfull, 3)))}")
            pred = Rt @ wfull
            pd.DataFrame({"id": te["id"].to_numpy(), TARGET: pred}).to_csv(
                os.path.join(SUB, f"{a.submit_name}.csv"), index=False)
            np.save(os.path.join(SUB, f"oof_{a.submit_name}.npy"), R @ wfull)
            print(f"wrote {os.path.join(SUB, a.submit_name)}.csv  "
                  f"rows={len(te):,}  cross-fitted CV {cv:.6f}")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
