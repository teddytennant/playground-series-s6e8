"""The stacker's L2 penalty has never actually been switched on.

`logs_csweep.txt` swept C over 0.03 .. 100 at 149 members and found every value
equivalent to within +/-6e-6, which the journal recorded as "C does not matter". That
reading is wrong, and the reason is arithmetic rather than empirical.

sklearn minimises

    0.5 * w'w  +  C * sum_i logloss_i

so the penalty a single row has to argue against is 1/(C*n). At n = 345,684 (the paired
fit half) and C = 0.03 that is **1e-4**. Every C in the swept range leaves the fit
effectively unpenalised, so the sweep did not measure regularisation -- it measured the
same unregularised solution eight times and correctly found no difference. To shrink 156
coefficients at this sample size C has to reach ~1e-5 and below, which nothing here has
tried.

Whether it should help is a real question with arguments both ways. The design is
pathological for an unpenalised fit -- 156 members at median pairwise correlation 0.98,
95.4% of the variance in PC1 -- and the fitted coefficients are correspondingly large and
opposing. But n/p is 4,400, which is enormous, and at that ratio even a badly conditioned
design can be estimated precisely. So this is a genuine test with a plausible null, not a
fix looking for a bug.

The instrument is the paired 50/50 split, which resolves ~4e-6, and every variant sees
identical rows. The coefficient norms are printed alongside so the answer comes with its
mechanism: if the AUC does not move while ||w|| falls by an order of magnitude, that is
the collinearity being harmless, stated with evidence.

    ridge_sweep.py --transform hybrid --reps 3
    ridge_sweep.py --transform hybrid --reps 0 --crossfit 1e-5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402

# the blend156 member set, pinned. xgb_lat/xgb_latcat are excluded BY NAME rather than by
# not existing yet: they land in oof/ part-way through this sweep's runtime and would
# otherwise silently change the matrix between one invocation and the next.
DROP156 = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
           "bolt_xgb_d7_alt2", "xgb_lat", "xgb_latcat")


def build(kind, drop, key, refresh=False):
    """Member matrix on the stacker's scale, cached under its own key.

    Deliberately NOT sharing stack_lab.py's cache/meta_<transform>.npz: that file is
    rebuilt with a different drop list by whatever else is running, and a member matrix
    that changes shape underneath a sweep invalidates the whole sweep silently.
    """
    p = os.path.join(CACHE, f"meta_{key}.npz")
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


def c_for(lam, n):
    """Convert a penalty-per-row `lam` into the C that realises it at sample size n.

    C is NOT a transferable regularisation strength: the penalty a row argues against is
    lam = 1/(C*n), so the same C means different things at different n. That matters here
    because one pipeline uses three different n:

        paired instrument   345,684 rows  (half of train)
        cross-fit fold fit  553,095 rows  (4/5 of train)
        full-data fit       691,369 rows  (all of train, this is what predicts test)

    A C tuned on the paired instrument and pasted into the cross-fit is 1.6x more
    regularised than intended, and 2.0x in the full fit that actually produces the
    submission. `blend_lab.build()` uses one C for both the fold fits and the full fit,
    so it has this bug latent in it -- harmless while C=1 leaves everything unpenalised,
    wrong the moment the penalty is switched on. Quote lam, derive C.
    """
    return 1.0 / (lam * n)


def standardize(Z):
    """Divide each member column by its own sd, so the L2 penalty is isotropic.

    An L2 penalty on unstandardised columns is not a neutral prior -- it shrinks a member
    in inverse proportion to its own scale. On the hybrid matrix the column sds run from
    1.81 to 27.58, so at any C the widest member is penalised ~230x less than the
    narrowest, purely because of where its logits happen to sit. That ordering is an
    artefact of the transform, not a statement about which members deserve shrinking.

    The scale is computed on the OOF matrix and applied unchanged to both sides. It uses
    no labels, so it leaks nothing, and it is a per-member linear map, so no member's own
    ranking (or solo AUC) moves.
    """
    s = Z.std(0)
    s[s <= 0] = 1.0
    return (Z / s).astype("float32"), s


def fit_stats(m):
    w = m.coef_[0]
    return dict(l2=float(np.sqrt(w @ w)), l1=float(np.abs(w).sum()),
                wmax=float(w.max()), wmin=float(w.min()), nneg=int((w < 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--key", default="hybrid156")
    ap.add_argument("--drop", default=",".join(DROP156))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cs", default="1.0,1e-2,1e-3,1e-4,3e-5,1e-5,3e-6,1e-6,1e-7")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--crossfit", type=float, default=0.0,
                    help="penalty PER ROW (lam), not C -- see c_for()")
    ap.add_argument("--standardize", action="store_true",
                    help="scale each member to unit sd first, making the L2 isotropic")
    a = ap.parse_args()

    t0 = time.time()
    names, Z, Zt, y = build(a.transform, set(filter(None, a.drop.split(","))),
                            a.key, a.refresh)
    print(f"matrix {Z.shape} ({time.time()-t0:.0f}s)", flush=True)
    print(f"column sd: min {Z.std(0).min():.3f} med {np.median(Z.std(0)):.3f} "
          f"max {Z.std(0).max():.3f}", flush=True)
    if a.standardize:
        Z, s = standardize(Z)
        Zt = (Zt / s).astype("float32")
        print("standardised: every column now sd 1.0 -- note the optimal C moves with "
              f"the scale, roughly by the square of it ({np.median(s):.2f}x here)",
              flush=True)
    Cs = [float(x) for x in a.cs.split(",") if x]

    if a.reps:
        rows, stats = [], {}
        for rep in range(a.reps):
            iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                          .split(np.zeros(len(y)), y))
            print(f"  n_fit={len(iA):,}  penalty per row at each C: "
                  + ", ".join(f"{C:g}->{1/(C*len(iA)):.2g}" for C in Cs), flush=True)
            r = {}
            for C in Cs:
                t1 = time.time()
                m = LogisticRegression(max_iter=5000, C=C).fit(Z[iA], y[iA])
                r[f"C={C:g}"] = roc_auc_score(y[iB], m.decision_function(Z[iB]))
                if rep == 0:
                    stats[C] = fit_stats(m)
                print(f"  [rep {rep}] C={C:<8g} {r[f'C={C:g}']:.6f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
            rows.append(r)
        df = pd.DataFrame(rows)
        ref = f"C={Cs[0]:g}"
        print(f"\npaired 50/50, {a.reps} splits -- vs {ref} on identical rows")
        for c in df.columns:
            d = df[c] - df[ref]
            ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
            print(f"  {c:>10s}  mean {df[c].mean():.6f}  {d.mean():+.6f} "
                  f"+/- {d.std(ddof=1):.6f}  [{ok}]")
        print("\ncoefficient shrinkage (rep 0 fit)")
        print(pd.DataFrame(stats).T.to_string(float_format="%.4g"))

    if a.crossfit:
        # --crossfit takes a penalty-per-row, not a C: each fold fit derives its own C
        # from its own n so every fold is regularised identically.
        folds = get_folds(y)
        mo = np.zeros(len(y))
        for itr, iva in folds:
            C = c_for(a.crossfit, len(itr))
            m = LogisticRegression(max_iter=5000, C=C).fit(Z[itr], y[itr])
            mo[iva] = m.decision_function(Z[iva])
            print(f"  fold n={len(itr):,} -> C={C:.3g}", flush=True)
        print(f"\ncross-fitted lam={a.crossfit:g}: {roc_auc_score(y, mo):.6f} "
              f"({time.time()-t0:.0f}s)")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
