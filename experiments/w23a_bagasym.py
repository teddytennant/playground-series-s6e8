"""w23a: the blender's OOF/test asymmetry, measured instead of inferred -- and the
penalty sweep it implies, at 187 members.

WHY THIS RUNS NOW
-----------------
w20 §5 observed dLB ~= 2x dCV on two wide pairs and named a mechanism for it: each
fold's meta-model is fitted on 4/5 of the rows (553,095) while the submitted test
column comes from a full-data fit (691,369), and a WIDER stack gains more from that
extra fifth than a narrow one does. It wrote "testable locally and for free next
slot". Three slots have passed without testing it.

Separately, the 08-13 journal recorded three loose ends in the blender that are all
the same loose end:

  (a) "C is not transferable across sample sizes ... one shared C regularises the
      submission model 2.0x harder than the instrument it was validated on."
      `blend_lab --lam` was written to fix it. "Not yet measured."
  (b) "The L2 penalty on unstandardised columns is not a neutral prior. Hybrid
      column sds run 1.81 to 27.58." `--standardize` exists. "Not yet measured."
  (c) "C=0.01 at +5e-6 +/- 3e-6 consistent" -- measured at 149 members and closed on
      performance grounds. The pack is now 187 and a variance-limited fit gets MORE
      from a penalty as K/n grows.

(a)-(c) are all "search the blend weights on OOF", and the K/n mechanism behind the
w20 observation is the SAME quantity that decides whether a penalty can help. One
harness answers both.

THE INSTRUMENT
--------------
One honest 20% stratified holdout H, never fitted on. P = the other 80% = 553,095
rows, which is EXACTLY the fold-fit n of the shipped pipeline, by construction. Every
number reported is roc_auc_score(y[H], decision_function(Z[H])) -- no cross-fitting,
no OOF re-use, no member selection, no leaderboard.

  PART A  learning surface over K (nested member subsets) x n (nested row subsets).
          The quantity of interest is the LAST STEP, D_K = AUC(n=|P|) - AUC(n=0.8|P|),
          a 1.25x row increase -- the same ratio as the pipeline's 553k -> 691k. It is
          an UPPER bound on the real step because learning curves flatten, so a small
          D_K is a safe conclusion and a large one is inflated.

  PART B  lam (L2 penalty PER ROW) x {unstandardised, standardised} at K=187, n=|P|.
          C = 1/(lam*n) per fit, so a winner transfers to the 691k full fit correctly
          -- defect (a) fixed rather than restated.

Hypotheses, gates, the registered point predictions and the ship rule are in
`experiments/w23_prereg.txt`, written before this file existed. Read that first; this
module deliberately contains no argument about what the numbers should be.

    w23a_bagasym.py --reps 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402
from blend_lab import HONEST_DROP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

MEMBER_SEED = 230817          # fixes the nested member ordering; same for every rep
K_GRID = (24, 47, 94, 187)
N_FRAC = (0.5, 0.625, 0.8, 1.0)
LAM_GRID = (0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
HOLD = 0.20

# GATE 1 (prereg §2): K=187, n=|P|, lam=0 must land within this of the shipped
# cross-fitted `w20_ad187_hybrid`. Different estimands; this is a loader/row-order
# sanity gate, not a precision test.
GATE1_REF = 0.970077
GATE1_TOL = 4e-4
# GATE 2: lam=1e-8 -> C=181, essentially unpenalised, must reproduce lam=0.
GATE2_TOL = 5e-6


def nested_strat(y, idx, rng):
    """A stratified permutation of `idx`, so prefixes of any length are stratified
    AND nested. Class proportions are preserved by interleaving the two class
    permutations on a common quantile grid rather than by concatenating them."""
    out = np.empty(len(idx), dtype=np.int64)
    pos = {}
    for c in (0, 1):
        m = idx[y[idx] == c]
        pos[c] = rng.permutation(m)
    # rank each class's members on [0,1) and merge by that rank -> every prefix holds
    # both classes in their population proportion.
    keys, vals = [], []
    for c in (0, 1):
        k = (np.arange(len(pos[c])) + 0.5) / len(pos[c])
        keys.append(k)
        vals.append(pos[c])
    keys = np.concatenate(keys)
    vals = np.concatenate(vals)
    out[:] = vals[np.argsort(keys, kind="stable")]
    return out


def fit_auc(Z, y, rows, hold, C, cols, std):
    """Fit the meta-logistic on `rows` x `cols` and score on the untouched holdout."""
    Ztr = Z[np.ix_(rows, cols)]
    Zho = Z[np.ix_(hold, cols)]
    if std:
        # scale taken from the FIT rows only -- the holdout never informs the scale
        s = Ztr.std(0)
        s[s <= 0] = 1.0
        Ztr = Ztr / s
        Zho = Zho / s
    m = LogisticRegression(max_iter=5000, C=C).fit(Ztr, y[rows])
    return roc_auc_score(y[hold], m.decision_function(Zho))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HERE, "w23a_bagasym"))
    a = ap.parse_args()

    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    print(f"{len(names)} members loaded in {time.time()-t0:.0f}s", flush=True)
    if len(names) != 187:
        raise SystemExit(f"expected the 187-member pack, got {len(names)}")

    t1 = time.time()
    Z, _Zt = transform(O, T, "hybrid")
    Z = Z.astype("float32")
    del O, T, _Zt
    print(f"hybrid transform {time.time()-t1:.0f}s   Z {Z.shape} "
          f"col sd {Z.std(0).min():.2f}..{Z.std(0).max():.2f}", flush=True)

    # nested member ordering: fixed across reps, so dK is never confounded with WHICH
    # members. K_GRID[-1] must be all of them.
    mrng = np.random.default_rng(MEMBER_SEED)
    order = mrng.permutation(len(names))
    cols = {K: np.sort(order[:K]) for K in K_GRID}

    rows_A, rows_B = [], []
    for rep in range(a.reps):
        ihold, ipool = None, None
        sss = StratifiedShuffleSplit(1, test_size=HOLD, random_state=1000 + rep)
        ipool, ihold = next(sss.split(np.zeros(n), y))
        ipool, ihold = np.sort(ipool), np.sort(ihold)
        rng = np.random.default_rng(230817 + rep)
        perm = nested_strat(y, ipool, rng)
        subs = {f: np.sort(perm[: int(round(f * len(perm)))]) for f in N_FRAC}
        print(f"\n=== rep {rep}   pool {len(ipool):,}  holdout {len(ihold):,}  "
              f"rate {y[ihold].mean():.6f} vs pool {y[ipool].mean():.6f}", flush=True)

        # ---- PART A: the (K, n) learning surface, lam = 0 (the shipped C = 1) ----
        for K in K_GRID:
            for f in N_FRAC:
                rr = subs[f]
                t = time.time()
                auc = fit_auc(Z, y, rr, ihold, 1.0, cols[K], False)
                rows_A.append(dict(rep=rep, K=K, frac=f, n_fit=len(rr), auc=auc))
                print(f"  A K={K:<4d} n={len(rr):>7,d}  {auc:.6f}  "
                      f"({time.time()-t:.0f}s)", flush=True)

        # ---- PART B: the penalty, at the full pack and the full pool ----
        rr = subs[1.0]
        for std in (False, True):
            for lam in LAM_GRID:
                C = 1.0 if lam == 0 else 1.0 / (lam * len(rr))
                t = time.time()
                auc = fit_auc(Z, y, rr, ihold, C, cols[187], std)
                rows_B.append(dict(rep=rep, std=std, lam=lam, C=C, auc=auc))
                print(f"  B std={int(std)} lam={lam:<8g} C={C:<10.4g} {auc:.6f}  "
                      f"({time.time()-t:.0f}s)", flush=True)

    dfA = pd.DataFrame(rows_A)
    dfB = pd.DataFrame(rows_B)
    dfA.to_csv(a.out + "_A.csv", index=False)
    dfB.to_csv(a.out + "_B.csv", index=False)

    # ------------------------------------------------------------------ gates
    top = dfA[(dfA.K == 187) & (dfA.frac == 1.0)].auc.mean()
    g1 = abs(top - GATE1_REF) <= GATE1_TOL
    b0 = dfB[(~dfB["std"]) & (dfB.lam == 0.0)].auc.to_numpy()
    b8 = dfB[(~dfB["std"]) & (dfB.lam == 1e-8)].auc.to_numpy()
    g2 = float(np.abs(b8 - b0).max()) <= GATE2_TOL
    print("\n" + "=" * 78)
    print(f"GATE 1  K=187 n=|P| lam=0 holdout AUC {top:.6f} vs shipped xfit "
          f"{GATE1_REF:.6f}  |d| {abs(top-GATE1_REF)*1e6:+.1f}e-6  tol "
          f"{GATE1_TOL*1e6:.0f}e-6  -> {'PASS' if g1 else 'FAIL'}")
    print(f"GATE 2  lam=1e-8 reproduces lam=0 to {np.abs(b8-b0).max()*1e6:.2f}e-6  "
          f"tol {GATE2_TOL*1e6:.0f}e-6  -> {'PASS' if g2 else 'FAIL'}")

    # ------------------------------------------------------- PART A read-out
    piv = dfA.pivot_table(index="K", columns="frac", values="auc", aggfunc="mean")
    print("\nPART A  mean holdout AUC over reps")
    print((piv * 1e6).round(1).to_string())

    print("\nD_K = AUC(n=|P|) - AUC(n=0.8|P|)   [the 1.25x step, paired within rep]")
    dk = {}
    for K in K_GRID:
        w = dfA[dfA.K == K].pivot_table(index="rep", columns="frac", values="auc")
        d = (w[1.0] - w[0.8]).to_numpy()
        dk[K] = d
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  K={K:<4d} {d.mean()*1e6:+8.2f}e-6  se {se*1e6:6.2f}  "
              f"reps {np.round(d*1e6,2).tolist()}  [{sign}]")
    # the whole curve, not just the last step -- slope in 1/n is the cleaner read
    print("\nfull-curve deficit vs n=|P|, by K   (e-6)")
    for K in K_GRID:
        w = dfA[dfA.K == K].pivot_table(index="rep", columns="frac", values="auc")
        print(f"  K={K:<4d} " + "  ".join(
            f"f={f}: {(w[f]-w[1.0]).mean()*1e6:+7.2f}" for f in N_FRAC))
    ratio = (dk[187].mean() / dk[24].mean()) if dk[24].mean() != 0 else np.nan
    mono = all(dk[K_GRID[i]].mean() <= dk[K_GRID[i + 1]].mean()
               for i in range(len(K_GRID) - 1))
    argmaxK = max(K_GRID, key=lambda K: dk[K].mean())
    print(f"\n  H2: D_187/D_24 = {ratio:.2f} (registered > 3);  monotone in K "
          f"{mono};  argmax K = {argmaxK} (registered 187)")

    # ------------------------------------------------------- PART B read-out
    print("\nPART B  penalty sweep, K=187, n=|P|, paired within rep against lam=0")
    ref = dfB[(~dfB["std"]) & (dfB.lam == 0.0)].set_index("rep").auc
    best = None
    for std in (False, True):
        for lam in LAM_GRID:
            s = dfB[(dfB["std"] == std) & (dfB.lam == lam)].set_index("rep").auc
            d = (s - ref).to_numpy()
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            sign = ("consistent" if (d > 0).all() or (d < 0).all()
                    else "SIGN FLIPS")
            print(f"  std={int(std)} lam={lam:<8g}  {d.mean()*1e6:+8.2f}e-6  "
                  f"se {se*1e6:6.2f}  [{sign}]")
            ok3 = (d > 0).all() and d.mean() >= 4e-6
            if not (std is False and lam == 0.0):
                if best is None or d.mean() > best["delta"]:
                    best = dict(std=bool(std), lam=float(lam),
                                delta=float(d.mean()), se=float(se),
                                consistent=bool((d > 0).all()), meets_H3=bool(ok3))
    print(f"\n  argmax cell: {best}")
    print("  prereg §4 ship rule: build ONLY if the argmax is consistent in sign "
          "across 3/3 reps AND >= +4e-6, and then halve the expected xfit gain.")
    ship = bool(best and best["consistent"] and best["delta"] >= 4e-6)
    print(f"  -> SHIP a --lam build: {ship}")

    out = dict(
        n_members=len(names), reps=a.reps, gate1=bool(g1), gate2=bool(g2),
        gate1_auc=float(top), gate1_ref=GATE1_REF,
        A_pivot={str(K): {str(f): float(piv.loc[K, f]) for f in N_FRAC}
                 for K in K_GRID},
        D_K={str(K): dict(mean=float(dk[K].mean()),
                          reps=[float(v) for v in dk[K]]) for K in K_GRID},
        H2_ratio=float(ratio), H2_monotone=bool(mono), H2_argmaxK=int(argmaxK),
        B_argmax=best, ship=ship,
    )
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out}_A.csv / _B.csv / .json    total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
