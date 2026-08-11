"""Blend weights, searched on out-of-fold predictions only.

The shipped entry is a single L2 logistic stack over ~150 member logits. This bench asks
the blending questions that have never been measured here, all on the paired 50/50
instrument so split noise cancels:

1. **Transform ensembling.** `logit`, `hybrid`, `rankraw` and `rescale` are four monotone
   views of the SAME members. At 86 members they scored 0.969660 / 0.969678 / 0.969684 /
   0.969686 -- indistinguishable, which the journal read as "pick one, it does not
   matter". But four combiners that score the same need not make the same mistakes, and
   averaging them costs nothing at predict time. Equal weights, so nothing is fitted.

2. **Weighting members by OOF performance** -- the classic blend. Rank-average the members
   with weights (auc - 0.5)**p, p swept. Weights come from the FIT half only.

3. **A searched convex blend** of (1) against (2). Honest three-way split: fit the stacks
   on A, choose alpha on B1, report on B2. If alpha* lands at 0 that is a clean negative
   and the linear stack keeps the crown; if not, it is a free gain.

Nothing here fits a weight on anything it is then scored on, and nothing touches the
public leaderboard.

    blend_lab.py --reps 3                       # the paired comparison
    blend_lab.py --build --submit-name <name>   # cross-fit + write submissions/<name>.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402

# golem_a/golem_f early-stop on their own validation fold (author-disclosed). Our own
# lgbm_tuned_lat* carried the identical defect and are superseded by the fixed-schedule
# retrains, so the honest member set drops all four.
HONEST_DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac")
KINDS = ("logit", "hybrid", "rankraw", "rescale")


def zs(v):
    v = np.asarray(v, np.float64)
    return (v - v.mean()) / (v.std() + 1e-12)


def rk(v):
    """Rank-normalise to (0,1). Ties averaged -- members genuinely emit exact ties."""
    return (rankdata(v) - 0.5) / len(v)


def load_all(kinds, drop, quiet=False):
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = (os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(drop))
    if not quiet:
        print(f"{len(names)} members", flush=True)
    mats = {}
    for k in kinds:
        t0 = time.time()
        Z, Zt = transform(O, T, k)
        mats[k] = (Z.astype("float32"), Zt.astype("float32"))
        if not quiet:
            print(f"  transform {k:8s} {time.time()-t0:5.0f}s", flush=True)
    del O, T
    return names, y, mats, te


def member_rank_blend(Zrank, rows, w):
    """Weighted rank-average of the members themselves, evaluated on `rows`."""
    return Zrank[rows] @ w


def paired(names, y, mats, kinds, reps, C):
    n = len(y)
    # member ranks come from the rankraw view: already a monotone per-member rank map,
    # computed without labels, so using the full-data ranks leaks nothing.
    Zrank = mats["rankraw"][0]
    rows = []
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(n), y))
        # B splits again: B1 chooses any searched weight, B2 reports. Keep the POSITIONS
        # within iB as well -- anything scored against a vector that is indexed by
        # position in iB (the ensembles below) has to be sliced the same way.
        p1, p2 = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=100 + rep)
                      .split(np.zeros(len(iB)), y[iB]))
        p1, p2 = np.sort(p1), np.sort(p2)
        iB1, iB2 = iB[p1], iB[p2]
        r = {}
        dfB = {}
        for k in kinds:
            Z = mats[k][0]
            t0 = time.time()
            m = LogisticRegression(max_iter=3000, C=C).fit(Z[iA], y[iA])
            dfB[k] = m.decision_function(Z[iB])
            r[f"stack_{k}"] = roc_auc_score(y[iB], dfB[k])
            print(f"  [rep {rep}] stack_{k:8s} {r[f'stack_{k}']:.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # --- 1. equal-weight ensembles of the four stacks, no fitted parameter ---
        r["ens_z"] = roc_auc_score(y[iB], np.mean([zs(dfB[k]) for k in kinds], 0))
        ens_rank = np.mean([rk(dfB[k]) for k in kinds], 0)
        r["ens_rank"] = roc_auc_score(y[iB], ens_rank)

        # --- 2. members weighted by OOF AUC measured on the FIT half only ---
        auc_A = np.array([roc_auc_score(y[iA], Zrank[iA, j]) for j in range(len(names))])
        best_p, best_v = None, -1
        for p in (0, 1, 2, 4, 8, 16):
            w = np.maximum(auc_A - 0.5, 1e-9) ** p
            w = w / w.sum()
            v = roc_auc_score(y[iB1], member_rank_blend(Zrank, iB1, w))
            r[f"memrank_p{p}"] = roc_auc_score(y[iB], member_rank_blend(Zrank, iB, w))
            if v > best_v:
                best_p, best_v = p, v
        w = np.maximum(auc_A - 0.5, 1e-9) ** best_p
        w = w / w.sum()
        mr = Zrank @ w
        r["memrank_star_p"] = best_p

        # --- 3. searched convex blend: alpha picked on B1, reported on B2 ---
        base1, base2 = rk(ens_rank[p1]), rk(ens_rank[p2])
        mr1, mr2 = rk(mr[iB1]), rk(mr[iB2])
        grid = np.arange(0, 0.51, 0.05)
        vals = [roc_auc_score(y[iB1], (1 - al) * base1 + al * mr1) for al in grid]
        al = float(grid[int(np.argmax(vals))])
        r["alpha_star"] = al
        r["blend_at_alpha"] = roc_auc_score(y[iB2], (1 - al) * base2 + al * mr2)
        r["ens_rank_B2"] = roc_auc_score(y[iB2], base2)
        rows.append(r)

    df = pd.DataFrame(rows)
    print("\nheld-out AUC per 50/50 split")
    print(df.to_string(float_format="%.6f"))

    print("\nPAIRED DIFFERENCES vs the best single transform stack (same rows)")
    ref = df[[f"stack_{k}" for k in kinds]].mean().idxmax()
    print(f"  reference: {ref}")
    for c in df.columns:
        if c in ("alpha_star", "memrank_star_p", "blend_at_alpha", "ens_rank_B2", ref):
            continue
        d = df[c] - df[ref]
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  {c:>16s}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")

    d = df["blend_at_alpha"] - df["ens_rank_B2"]
    ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
    print(f"\nsearched blend (alpha chosen on B1, scored on B2), alphas "
          f"{list(df['alpha_star'])}, member-weight powers {list(df['memrank_star_p'])}")
    print(f"  vs ens_rank on the same B2 rows: {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")
    return df


def build(names, y, mats, kinds, te, C, submit_name):
    """Cross-fit each transform on the frozen folds, then rank-average the stacks."""
    folds = get_folds(y)
    oof, tst = {}, {}
    for k in kinds:
        Z, Zt = mats[k]
        mo = np.zeros(len(y))
        for itr, iva in folds:
            mo[iva] = (LogisticRegression(max_iter=3000, C=C)
                       .fit(Z[itr], y[itr]).decision_function(Z[iva]))
        oof[k] = mo
        full = LogisticRegression(max_iter=3000, C=C).fit(Z, y)
        tst[k] = full.decision_function(Zt)
        print(f"  cross-fitted stack_{k:8s} {roc_auc_score(y, mo):.6f}", flush=True)
        if submit_name:
            # the single-transform stacks come free with the ensemble's cross-fit, and
            # each is a genuinely different file. Write them so a slot can send one
            # without paying for a second 24-fit pass.
            write(te, tst[k], mo, f"{submit_name}_{k}", roc_auc_score(y, mo))

    o_ens = np.mean([rk(oof[k]) for k in kinds], 0)
    t_ens = np.mean([rk(tst[k]) for k in kinds], 0)
    cv = roc_auc_score(y, o_ens)
    print(f"\ncross-fitted rank-ensemble of {len(kinds)} transform stacks: {cv:.6f}")

    if submit_name:
        write(te, t_ens, o_ens, submit_name, cv)
    return cv, o_ens, t_ens


def write(te, pred_test, oof_pred, name, cv):
    """Submission files here hold ranks or logits, not calibrated probabilities.

    AUC is rank-only so that is equivalent for scoring -- but do NOT read the mean of one
    of these files as a base rate the way the probability stacks were read.
    """
    sub = pd.DataFrame({"id": te["id"].to_numpy(), TARGET: pred_test})
    p = os.path.join(SUB, f"{name}.csv")
    sub.to_csv(p, index=False)
    np.save(os.path.join(SUB, f"oof_{name}.npy"), oof_pred)
    print(f"  wrote {p}  rows={len(sub):,}  cross-fitted CV {cv:.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--drop", default=",".join(HONEST_DROP))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--submit-name", default=None)
    a = ap.parse_args()

    kinds = [k for k in a.kinds.split(",") if k]
    t0 = time.time()
    names, y, mats, te = load_all(tuple(dict.fromkeys(list(kinds) + ["rankraw"])),
                                  set(filter(None, a.drop.split(","))))
    print(f"loaded in {time.time()-t0:.0f}s", flush=True)

    if a.reps:
        paired(names, y, mats, kinds, a.reps, a.C)
    if a.build or a.submit_name:
        build(names, y, mats, kinds, te, a.C, a.submit_name)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
