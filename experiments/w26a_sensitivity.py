"""w26a — is the `logit` transform the one that gains most from better member columns?

Pre-registered at experiments/w26_prereg.txt (which inherits and repairs w25_prereg
ADDENDUM A1) BEFORE this ran once. Q1-Q4 and T1-T3 are fixed there.

The question. Across 67 scored files the (LB - CV) gap is not constant by transform family:
logit sits +151.1e-6 above h3 (t +14.6), reproduced out of sample to 2.7e-6 by send 10 of
w25 at a CV 65e-6 above any logit file previously sent. 18x the entire simulated paired
slice budget, so it is not a slice draw. The live hypothesis H_L is a train->test effect:
OOF member columns come from 5-fold models and TEST member columns from full-data models, so
the test columns are strictly better inputs, and the unbounded logit transform is the most
sensitive of the four to member quality -- meaning cross-fitted CV SYSTEMATICALLY UNDERSTATES
every logit-containing mix. That is decision-relevant because h3 EXCLUDES logit, ens4
INCLUDES it, and both `WANTED` files are h3.

The test needs no leaderboard. If better columns are worth more to logit, worse columns must
COST logit more. So degrade the member matrix by a controlled amount and read the slope of
cross-fitted stack AUC against member quality, per transform.

Three degradation modes, because the registered one is confounded (w26_prereg §2): raw-space
Gaussian noise pushes member probabilities outside [0,1], `to_logit` pins every such row to a
constant, and `logit` eats that damage undiluted while `rankraw` is immune to it by
construction. That arm would show logit steepest even if H_L is false. So:

  subset      drop members at random (187 -> 140 -> 100 -> 70). Perturbs no value at all.
              Every transform is per-column, so subsetting COMMUTES with it and one
              transform pass serves every size.
  logitnoise  p' = sigmoid(logit(p) + d * sd_logit * eps). A real per-member quality change
              that stays strictly inside (0,1) and introduces no clipping. Carries Q3.
  rawnoise    the literally-registered arm, kept so the confound is MEASURED not argued.

Q1 requires logit to be steepest in BOTH `subset` and `logitnoise`. rawnoise alone does not
count and is reported for Q4.

One condition per row, appended to the CSV as it completes -- w25d's first attempt was
killed partway through with nothing on disk, and that is not repeated here.

  .venv/bin/python experiments/w26a_sensitivity.py --mode subset
  .venv/bin/python experiments/w26a_sensitivity.py --mode logitnoise
  .venv/bin/python experiments/w26a_sensitivity.py --mode rawnoise
  .venv/bin/python experiments/w26a_sensitivity.py --report
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members, to_logit, transform  # noqa: E402
from blend_lab import HONEST_DROP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "w26a_sensitivity.csv")
KINDS = ("logit", "hybrid", "rankraw", "rescale")
SIZES = (187, 140, 100, 70)
DRAWS = 2                       # per size; size 187 has only one distinct draw
DGRID = (0.0, 0.10, 0.25, 0.50)
# w25f family offsets over h3, e-6 -- fitted with NO knowledge of S_k. Q2 tests the ordering.
W25F_OFFSET = {"logit": 151.1, "rescale": 37.8, "rankraw": 15.1, "hybrid": 6.1}


def load_O(kinds_needed):
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_test = len(te)
    del tr, te
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, n_test, extra_dirs=extra, drop=set(HONEST_DROP))
    # T is only needed because transform() is a two-sided map; the test side is never
    # scored here. Keep it -- `rescale` derives its range from BOTH sides, so dropping it
    # would silently change the transform relative to every shipped build.
    return names, y, O, T


def cross_fit_auc(Z, y, folds):
    """Cross-fitted stack AUC on the frozen folds. Standardised design, C=1, float64.

    Standardised deliberately (w26_prereg §2): the unstandardised fit stops ~686 iterations
    short of convergence, and comparing SENSITIVITY across four transforms with unconverged
    fits measures optimiser progress on each transform's conditioning, not the transform's
    response to member quality.
    """
    s = Z.std(0)
    s[s <= 0] = 1.0
    Z = Z / s
    oof = np.zeros(len(y))
    it = []
    for itr, iva in folds:
        m = LogisticRegression(max_iter=5000, C=1.0, tol=1e-4).fit(Z[itr], y[itr])
        oof[iva] = m.decision_function(Z[iva])
        it.append(int(np.ravel(m.n_iter_)[0]))
    return roc_auc_score(y, oof), float(np.mean(it))


def append_row(rec):
    df = pd.DataFrame([rec])
    df.to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)


def done_keys():
    if not os.path.exists(OUT):
        return set()
    d = pd.read_csv(OUT)
    return {(r["mode"], r["kind"], float(r["level"]), int(r["draw"])) for _, r in d.iterrows()}


def clip_frac(O):
    return float(((O <= 0) | (O >= 1)).mean())


def run_subset(names, y, O, T, folds, have):
    """Column subsetting commutes with every per-column transform, so transform ONCE."""
    n = O.shape[1]
    for kind in KINDS:
        t0 = time.time()
        Z, _Zt = transform(O, T, kind)
        Z = np.ascontiguousarray(Z, dtype="float64")
        print(f"[subset] transform {kind:8s} {time.time()-t0:.0f}s", flush=True)
        for size in SIZES:
            ndraw = 1 if size == n else DRAWS
            for draw in range(ndraw):
                key = ("subset", kind, float(size), draw)
                if key in have:
                    print(f"  skip {key}", flush=True)
                    continue
                rng = np.random.default_rng(7000 + 13 * size + draw)
                cols = np.sort(rng.choice(n, size=size, replace=False)) if size < n \
                    else np.arange(n)
                t = time.time()
                auc, it = cross_fit_auc(Z[:, cols], y, folds)
                append_row(dict(mode="subset", kind=kind, level=float(size), draw=draw,
                                n_members=size, cv=auc, mean_iter=it,
                                clip_frac=clip_frac(O[:, cols]), secs=time.time() - t))
                print(f"  [subset] {kind:8s} m={size:3d} draw{draw} cv {auc:.10f} "
                      f"iter {it:.0f} {time.time()-t:.0f}s", flush=True)
        del Z, _Zt
        gc.collect()


def degrade(O, d, mode, seed):
    """Return a degraded copy of O. `d` is in units of the relevant per-column sd.

    ⚠ `logitnoise` applies its logit -> sigmoid ROUND TRIP AT EVERY LEVEL INCLUDING d=0,
    and that is not cosmetic. `to_logit` clips to [1e-15, 1-1e-15] and 0.79% of the raw
    cells sit outside the unit interval (naji03/naji05 emit -0.013..1.022), so the round
    trip pulls them inside. Applying it only at d>0 would make the d=0 baseline the ONLY
    level carrying that clipping damage, and since `logit` is the transform that eats
    clipping undiluted, S_logit would be measured against a baseline handicapped in a way
    no other level shares -- biasing the slope DOWNWARD and against Q1. Measured on
    synthetic columns before any real run: raw clip fraction 0.0079 at d=0 against 0.0000
    at d>0. Round-tripping everywhere makes noise the only thing that varies along the
    axis, which is the whole point of a slope.
    """
    rng = np.random.default_rng(seed)
    if mode == "rawnoise":
        # left alone at d=0 deliberately: this arm's ENTIRE purpose is to carry the
        # clipping confound so it can be measured (Q4), so it must not be repaired.
        if d == 0.0:
            return O
        return O + rng.standard_normal(O.shape) * (O.std(0) * d)
    if mode == "logitnoise":
        L = to_logit(O)
        if d > 0.0:
            L = L + rng.standard_normal(O.shape) * (L.std(0) * d)
        # back to probability space; strictly inside (0,1) so no NEW clipping is created.
        return 1.0 / (1.0 + np.exp(-np.clip(L, -30.0, 30.0)))
    raise SystemExit(mode)


def run_noise(mode, names, y, O, T, folds, have):
    for d in DGRID:
        ndraw = 1 if d == 0.0 else DRAWS
        for draw in range(ndraw):
            todo = [k for k in KINDS if (mode, k, float(d), draw) not in have]
            if not todo:
                print(f"  skip all kinds at d={d} draw{draw}", flush=True)
                continue
            Od = degrade(O, d, mode, seed=9000 + int(d * 1000) * 7 + draw)
            cf = clip_frac(Od)
            for kind in todo:
                t = time.time()
                Z, _Zt = transform(Od, T, kind)
                Z = np.ascontiguousarray(Z, dtype="float64")
                auc, it = cross_fit_auc(Z, y, folds)
                append_row(dict(mode=mode, kind=kind, level=float(d), draw=draw,
                                n_members=O.shape[1], cv=auc, mean_iter=it,
                                clip_frac=cf, secs=time.time() - t))
                print(f"  [{mode}] {kind:8s} d={d:.2f} draw{draw} cv {auc:.10f} "
                      f"iter {it:.0f} clip {cf:.4f} {time.time()-t:.0f}s", flush=True)
                del Z, _Zt
                gc.collect()
            if Od is not O:
                del Od
            gc.collect()


def slopes(df, mode, xcol):
    """S_k: OLS slope of cv (in e-6) against the degradation axis, per transform.

    Sign convention: S_k is reported as LOSS PER UNIT OF DEGRADATION and is positive when
    degrading hurts. For `subset` the axis is members-removed so more is worse; for the
    noise modes the axis is d.
    """
    out = {}
    sub = df[df["mode"] == mode]
    for kind in KINDS:
        k = sub[sub.kind == kind]
        if len(k) < 3:
            continue
        x = k[xcol].to_numpy(float)
        yv = (k.cv.to_numpy(float) - k.cv.max()) * 1e6
        b = np.polyfit(x, yv, 1)[0]
        base = k[k[xcol] == x.min()].cv.mean()
        out[kind] = dict(slope_e6=float(-b), n=len(k), base_cv=float(base))
    return out


def report():
    df = pd.read_csv(OUT).drop_duplicates(subset=["mode", "kind", "level", "draw"],
                                          keep="last")
    res = {}
    print("=" * 78)
    for mode in ("subset", "logitnoise", "rawnoise"):
        sub = df[df["mode"] == mode]
        if not len(sub):
            continue
        xcol = "removed" if mode == "subset" else "level"
        if mode == "subset":
            sub = sub.assign(removed=187 - sub.n_members)
        print(f"\n### mode {mode} — cross-fitted CV by level (mean over draws)")
        piv = sub.pivot_table(index=xcol, columns="kind", values="cv", aggfunc="mean")
        print((piv * 1e6).round(1).to_string())
        S = slopes(sub, mode, xcol)
        res[mode] = S
        print(f"\n  S_k (e-6 of CV lost per unit of {xcol}) — bigger = more sensitive")
        order = sorted(S, key=lambda k: -S[k]["slope_e6"])
        for k in order:
            print(f"    {k:8s} {S[k]['slope_e6']:+10.3f}   base cv {S[k]['base_cv']:.10f}")
        print(f"  ranking, most sensitive first: {' > '.join(order)}")
        res[mode + "_order"] = order

    print("\n" + "=" * 78)
    print("### Q1 — is logit the steepest in BOTH subset and logitnoise?")
    q1 = all(res.get(m + "_order", [None])[0] == "logit"
             for m in ("subset", "logitnoise") if m in res)
    have_both = "subset" in res and "logitnoise" in res
    for m in ("subset", "logitnoise", "rawnoise"):
        if m in res:
            print(f"    {m:11s} steepest = {res[m + '_order'][0]}")
    print(f"  Q1 {'PASS' if (q1 and have_both) else 'FAIL'}"
          f"{'' if have_both else '  (incomplete — both modes required)'}")

    print("\n### Q2 — does the S_k ordering match the w25f family offsets?")
    print(f"    w25f offsets over h3 (e-6): "
          f"{', '.join(f'{k} {v}' for k, v in W25F_OFFSET.items())}")
    for m in ("subset", "logitnoise", "rawnoise"):
        if m not in res:
            continue
        ks = [k for k in KINDS if k in res[m]]
        rho = spearmanr([res[m][k]["slope_e6"] for k in ks],
                        [W25F_OFFSET[k] for k in ks]).statistic
        exact = res[m + "_order"] == sorted(W25F_OFFSET, key=lambda k: -W25F_OFFSET[k])
        print(f"    {m:11s} rho {rho:+.3f}   exact-order-match {exact}")

    print("\n### Q4 — is rawnoise's logit gap inflated vs logitnoise (the clip confound)?")
    for m in ("logitnoise", "rawnoise"):
        if m in res and "logit" in res[m]:
            others = [res[m][k]["slope_e6"] for k in KINDS if k in res[m] and k != "logit"]
            print(f"    {m:11s} S_logit {res[m]['logit']['slope_e6']:+9.3f}   "
                  f"mean S_other {np.mean(others):+9.3f}   "
                  f"gap {res[m]['logit']['slope_e6'] - np.mean(others):+9.3f}")
    if "rawnoise" in res:
        cf = df[df["mode"] == "rawnoise"].groupby("level").clip_frac.mean()
        print(f"    clipped fraction of O by d: "
              f"{', '.join(f'{i:.2f}->{v:.4f}' for i, v in cf.items())}")

    json.dump(res, open(os.path.join(HERE, "w26a_sensitivity.json"), "w"), indent=1,
              default=float)
    print("\nwrote experiments/w26a_sensitivity.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("subset", "logitnoise", "rawnoise"))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    t0 = time.time()
    names, y, O, T = load_O(KINDS)
    folds = get_folds(y)
    print(f"{len(names)} members, O {O.shape}, {len(folds)} folds, "
          f"clip frac {clip_frac(O):.4f}, loaded {time.time()-t0:.0f}s", flush=True)
    have = done_keys()
    if a.mode == "subset":
        run_subset(names, y, O, T, folds, have)
    else:
        run_noise(a.mode, names, y, O, T, folds, have)
    print(f"mode {a.mode} done, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
