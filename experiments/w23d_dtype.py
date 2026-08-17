"""w23d: separate DTYPE from STANDARDISATION. w23c's first two arms say w23a attributed
the gain to the wrong axis.

WHAT FORCED THIS, and it is a correction to w23a rather than an extension of it
--------------------------------------------------------------------------------
w23a measured std=1 - std=0 as +19.21e-6 (se 2.85, consistent 3/3) at K=187 and called
standardisation the winner. w23c re-ran the same two cells on the same rows and got:

    std0_tol1e-4   686 iters  fit_logloss 0.200317800   hold_auc 0.970367
    std1_tol1e-4    65 iters  fit_logloss 0.200384753   hold_auc 0.970370

Two things are wrong with the w23a reading:

  1. std=1 has HIGHER fit log-loss. It is not finding a better point on the same
     surface, so w23c's registered mechanism (i) "under-convergence" is REFUTED as
     stated -- the standardised fit deliberately fits the training rows worse.
  2. The gap here is +3e-6, not +19e-6. The one thing that differs between w23a and
     w23c is DTYPE: w23a stores Z as float32 (copying `blend_lab.load_all`, which is
     what every shipped build has used), w23c casts to float64. In float64 the
     unstandardised fit reads 0.970367; in float32 w23a read 0.970360.

So the candidate explanation is not "standardising is a better prior". It is that the
SHIPPED cell -- float32 columns with sd 1.8..27.6 -- is numerically the worst of the
four available cells, and standardising rescues it by conditioning rather than by
regularising. If that is right, the free fix is a dtype, the effect should mostly
disappear in float64, and w23a's headline needs rewriting.

THE DESIGN — a 2x2, because w23a and w23c each varied both axes at once
----------------------------------------------------------------------
Same instrument as w23a: 20% stratified holdout, never fitted on, 3 reps, seeds
1000+rep, so every number here is paired with a number already on file. K=187, all
553,095 pool rows, C=1, sklearn's default tol.

    dtype in {float32, float64}  x  std in {0, 1}      = 4 cells x 3 reps

Everything is reported paired against the SHIPPED cell (float32, std=0). Fit log-loss is
always accumulated in float64 so the objective is comparable across cells.

REGISTERED PREDICTIONS, written before the run
---------------------------------------------
P1  (float32, std=0) is the WORST of the four cells in all 3 reps.
P2  (float64, std=0) - (float32, std=0) is POSITIVE and recovers most of w23a's
    +19.21e-6 -- registered range +8 to +20e-6.
P3  The std axis is a NULL once dtype is float64: |(float64,std=1) - (float64,std=0)|
    < 5e-6. This is the clause that decides whether the fix is a dtype or a prior.
P4  (float32, std=1) sits near (float64, std=1), i.e. standardising substitutes for
    precision. If instead float32+std=1 lags float64+std=1 badly, then BOTH fixes are
    needed and they are not interchangeable.

FALSIFICATION that matters: if P3 fails and the std axis is worth >5e-6 even in
float64, then standardisation is a real prior after all, w23c's fit-log-loss reading
stands, and the two effects ADD -- which would make the total larger, not smaller. Both
outcomes are actionable; they just point at different one-line changes to `blend_lab`.

    w23d_dtype.py
"""
from __future__ import annotations

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
HOLD = 0.20
REPS = 3
CELLS = [("f32_std0", "float32", False),   # THE SHIPPED CELL
         ("f32_std1", "float32", True),
         ("f64_std0", "float64", False),
         ("f64_std1", "float64", True)]


def logloss(y, z):
    z = np.asarray(z, np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def main():
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    Z64, _ = transform(O, T, "hybrid")
    Z64 = np.ascontiguousarray(Z64, dtype="float64")
    del O, T, _
    print(f"{len(names)} members, Z {Z64.shape}, loaded {time.time()-t0:.0f}s", flush=True)

    rows = []
    for rep in range(REPS):
        ip, ih = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=1000 + rep)
                      .split(np.zeros(len(y)), y))
        ip, ih = np.sort(ip), np.sort(ih)
        print(f"\n=== rep {rep}  pool {len(ip):,}  holdout {len(ih):,}", flush=True)
        for tag, dt, std in CELLS:
            A = Z64[ip].astype(dt)
            B = Z64[ih].astype(dt)
            if std:
                s = A.std(0)
                s[s <= 0] = 1
                A, B = (A / s).astype(dt), (B / s).astype(dt)
            t = time.time()
            m = LogisticRegression(max_iter=5000, C=1.0).fit(A, y[ip])
            el = time.time() - t
            zh = m.decision_function(B)
            zt = m.decision_function(A)
            rows.append(dict(rep=rep, cell=tag, dtype=dt, std=std,
                             n_iter=int(np.ravel(m.n_iter_)[0]), secs=round(el, 1),
                             fit_logloss=logloss(y[ip], zt),
                             auc=roc_auc_score(y[ih], zh)))
            print(f"  {tag:<9s} iters {rows[-1]['n_iter']:>5d} {el:6.1f}s  "
                  f"fit_logloss {rows[-1]['fit_logloss']:.9f}  "
                  f"auc {rows[-1]['auc']:.6f}", flush=True)
            del A, B

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "w23d_dtype.csv"), index=False)
    piv = df.pivot_table(index="cell", columns="rep", values="auc")
    print("\nholdout AUC by cell x rep")
    print((piv * 1e6).round(1).to_string())

    ref = df[df.cell == "f32_std0"].set_index("rep").auc
    print("\nPAIRED vs the SHIPPED cell f32_std0 (same holdout rows)")
    res = {}
    for tag, _, _ in CELLS:
        s = df[df.cell == tag].set_index("rep").auc
        d = (s - ref).to_numpy()
        se = d.std(ddof=1) / np.sqrt(len(d))
        sign = ("consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS")
        res[tag] = dict(mean=float(d.mean()), se=float(se),
                        reps=[float(v) for v in d], consistent=bool((d > 0).all()))
        print(f"  {tag:<9s} {d.mean()*1e6:+8.2f}e-6  se {se*1e6:5.2f}  "
              f"reps {np.round(d*1e6,2).tolist()}  [{sign}]")

    # ---- the registered readings, evaluated mechanically ----
    worst = piv.idxmin()
    p1 = bool((worst == "f32_std0").all())
    p2 = res["f64_std0"]["mean"]
    s64 = df[df.cell == "f64_std1"].set_index("rep").auc - \
        df[df.cell == "f64_std0"].set_index("rep").auc
    p3 = bool(abs(s64.mean()) < 5e-6)
    p4 = res["f32_std1"]["mean"] - res["f64_std1"]["mean"]
    print("\n" + "=" * 78)
    print(f"P1  shipped cell is worst in all 3 reps: {p1}   (per-rep worst: "
          f"{worst.to_dict()})")
    print(f"P2  f64 fixes the unstandardised fit: {p2*1e6:+.2f}e-6  "
          f"(registered +8..+20e-6 -> {'IN' if 8e-6 <= p2 <= 20e-6 else 'OUT OF'} range)")
    print(f"P3  std axis is a null in float64: {s64.mean()*1e6:+.2f}e-6  -> "
          f"{'NULL as registered' if p3 else 'NOT a null -- std is a real prior'}")
    print(f"P4  f32_std1 - f64_std1 = {p4*1e6:+.2f}e-6  "
          f"({'standardising substitutes for precision' if abs(p4) < 5e-6 else 'both fixes needed'})")
    print("\nfit log-loss by cell (lower = closer to the unpenalised MLE)")
    print(df.pivot_table(index="cell", values="fit_logloss").to_string(
        float_format="%.9f"))
    print("\nmean seconds and iterations by cell")
    print(df.pivot_table(index="cell", values=["secs", "n_iter"]).to_string(
        float_format="%.1f"))

    with open(os.path.join(HERE, "w23d_dtype.json"), "w") as f:
        json.dump(dict(paired=res, P1=p1, P2=float(p2), P3_std64=float(s64.mean()),
                       P3=p3, P4=float(p4),
                       fit_logloss={t: float(df[df.cell == t].fit_logloss.mean())
                                    for t, _, _ in CELLS}), f, indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
