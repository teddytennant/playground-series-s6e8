"""Paired row-bootstrap on saved OOF vectors: is a 4e-6 CV gap resolvable?

The deadline ranking in the journal separates `blend156w` (0.970047), `blend156_h3`
(0.970046) and `blend156` (0.970042) -- gaps of 1e-6 to 5e-6 on a single cross-fit over
the frozen folds. Each is a point estimate of a rank statistic on 691,369 rows, and the
journal has never priced its sampling noise.

This does that, and it costs nothing: every candidate is a saved cross-fitted OOF vector,
so no model is refitted. The rows are resampled with multinomial weights, the SAME weights
for every candidate in a rep, so what comes out is the noise of the PAIRED DIFFERENCE --
the quantity the ranking actually rests on -- not the far larger noise of either AUC alone.

Weighted AUC is computed from one global sort per candidate and an O(n) pass per rep, with
exact tie handling (rank-averaged ensembles do produce exact ties). No jitter, no
approximation.

Caveat worth stating: this prices ROW noise only. The fold assignment, the member models
and the stacker fits are all held fixed, so a bootstrap CI here is a lower bound on the
uncertainty of "which candidate is better", never an upper bound.

    auc_boot.py --reps 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import SUB, TARGET, load_raw  # noqa: E402


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


class WAuc:
    """Weighted AUC for one fixed score vector, O(n) per weight vector."""

    def __init__(self, s, y):
        o = np.argsort(s, kind="stable")
        self.y = y[o].astype(np.float64)
        ss = s[o]
        # group starts for runs of equal score
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        self.order = o

    def __call__(self, w):
        ws = w[self.order]
        wp, wn = ws * self.y, ws * (1.0 - self.y)
        gp = np.add.reduceat(wp, self.starts)
        gn = np.add.reduceat(wn, self.starts)
        before = np.r_[0.0, np.cumsum(gn)[:-1]]
        Wp, Wn = gp.sum(), gn.sum()
        return float((gp * (before + 0.5 * gn)).sum() / (Wp * Wn))


def load(name):
    p = os.path.join(SUB, f"oof_{name}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--ref", default="blend156")
    ap.add_argument("--names", default="", help="comma-separated oof_<name>.npy in submissions/")
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)

    if a.names:
        cand = {nm: load(nm) for nm in a.names.split(",") if nm}
    else:
        cand = {}
        for nm in ("blend156", "blend156w", "blend156_logit", "blend156_hybrid",
                   "blend156_rankraw", "blend156_rescale"):
            cand[nm] = load(nm)
        # h3 = the drop-logit ensemble, rebuilt from the same saved per-transform OOF
        cand["blend156_h3"] = np.mean([rk(cand[f"blend156_{k}"])
                                       for k in ("hybrid", "rankraw", "rescale")], 0)

    names = list(cand)
    t0 = time.time()
    scorers = {nm: WAuc(cand[nm], y) for nm in names}
    ones = np.ones(n)
    point = {nm: scorers[nm](ones) for nm in names}
    print(f"point estimates ({time.time()-t0:.0f}s to build scorers)")
    for nm in sorted(names, key=lambda z: -point[z]):
        print(f"  {nm:20s} {point[nm]:.6f}")

    rng = np.random.default_rng(0)
    boot = {nm: np.empty(a.reps) for nm in names}
    for r in range(a.reps):
        w = rng.poisson(1.0, n).astype(np.float64)   # Poisson bootstrap == multinomial to O(1/n)
        for nm in names:
            boot[nm][r] = scorers[nm](w)
        if (r + 1) % 50 == 0:
            print(f"  rep {r+1}/{a.reps}  {time.time()-t0:.0f}s", flush=True)

    print(f"\nMARGINAL bootstrap sd of each AUC (what a naive CI would use)")
    for nm in sorted(names, key=lambda z: -point[z]):
        print(f"  {nm:20s} {point[nm]:.6f}  sd {boot[nm].std(ddof=1):.6f}")

    print(f"\nPAIRED differences vs {a.ref} (same resampled rows)")
    ref = boot[a.ref]
    for nm in sorted(names, key=lambda z: -point[z]):
        if nm == a.ref:
            continue
        d = boot[nm] - ref
        obs = point[nm] - point[a.ref]
        frac = float((d > 0).mean())
        print(f"  {nm:20s} obs {obs:+.6f}   boot {d.mean():+.6f} +/- {d.std(ddof=1):.6f}"
              f"   P(better) {frac:.3f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
