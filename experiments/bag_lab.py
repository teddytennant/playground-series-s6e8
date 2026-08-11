"""Seed and fold diversity at the COMBINER, which is the one level here that is free.

Member-level seed diversity costs ~2h a model and every core is busy. The stacker is not:
the 156 member OOF vectors are fixed on disk, so refitting the combiner under a different
resampling costs one logistic fit and nothing else.

Three arms, all scored on the same held-out half so split noise cancels:

  single    one L2 logistic fit on the whole fit-half            <- what ships today
  foldbag5  five fits, each on a disjoint 80% of the fit-half, decision functions averaged
  boot5     five bootstrap fits on the fit-half, averaged

`foldbag5` is not an idle variant. The shipped pipeline is INTERNALLY INCONSISTENT about
this: the cross-fitted CV that every decision in the journal rests on is produced by fold
models (n=553,095 each), while the file that is actually submitted comes from a single
full-data fit (n=691,369). Those are two different estimators. If they disagree, the CV
number is validating a procedure the submission does not use. `foldbag5` on test makes the
two match exactly, and this bench prices whether that is worth anything.

    bag_lab.py --kinds rankraw --reps 3            # the paired comparison
    bag_lab.py --build --submit-name blend156fb    # cross-fit + write both test variants
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import TARGET, get_folds  # noqa: E402
from blend_lab import HONEST_DROP, load_all, rk, write  # noqa: E402


def fit_df(Z, y, rows, eval_sets, C, max_iter=5000):
    """Fit on `rows`, return decision_function on each array in `eval_sets`."""
    m = LogisticRegression(max_iter=max_iter, C=C).fit(Z[rows], y[rows])
    return [m.decision_function(E) for E in eval_sets]


def foldbag(Z, y, rows, eval_sets, C, k=5, seed=0):
    """Average the decision functions of k models, each fit on a disjoint (k-1)/k of rows."""
    acc = [np.zeros(len(E)) for E in eval_sets]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for itr, _ in skf.split(np.zeros(len(rows)), y[rows]):
        for a, d in zip(acc, fit_df(Z, y, rows[itr], eval_sets, C)):
            a += d / k
    return acc


def bootbag(Z, y, rows, eval_sets, C, k=5, seed=0):
    """Average the decision functions of k bootstrap fits on rows."""
    acc = [np.zeros(len(E)) for E in eval_sets]
    rng = np.random.default_rng(seed)
    for _ in range(k):
        idx = rows[rng.integers(0, len(rows), len(rows))]
        for a, d in zip(acc, fit_df(Z, y, idx, eval_sets, C)):
            a += d / k
    return acc


def paired(y, mats, kinds, reps, C, k):
    n = len(y)
    rows = []
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(n), y))
        r = {"rep": rep}
        for kd in kinds:
            Z = mats[kd][0]
            ev = [Z[iB]]
            t0 = time.time()
            (d_single,) = fit_df(Z, y, iA, ev, C)
            t1 = time.time()
            (d_fold,) = foldbag(Z, y, iA, ev, C, k, seed=rep)
            t2 = time.time()
            (d_boot,) = bootbag(Z, y, iA, ev, C, k, seed=rep)
            r[f"{kd}_single"] = roc_auc_score(y[iB], d_single)
            r[f"{kd}_foldbag"] = roc_auc_score(y[iB], d_fold)
            r[f"{kd}_boot"] = roc_auc_score(y[iB], d_boot)
            # how much do the two orderings even differ? if spearman is 1.0 to 6dp the
            # arms are the same file and the AUC comparison is measuring nothing.
            r[f"{kd}_sp_fold"] = spearmanr(d_single, d_fold).statistic
            r[f"{kd}_sp_boot"] = spearmanr(d_single, d_boot).statistic
            print(f"  [rep {rep}] {kd:8s} single {r[f'{kd}_single']:.6f} ({t1-t0:.0f}s)  "
                  f"foldbag {r[f'{kd}_foldbag']:.6f} ({t2-t1:.0f}s)  "
                  f"boot {r[f'{kd}_boot']:.6f}  "
                  f"sp {r[f'{kd}_sp_fold']:.6f}/{r[f'{kd}_sp_boot']:.6f}", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    print("\nheld-out AUC per 50/50 split")
    print(df.to_string(float_format="%.6f"))
    print("\nPAIRED vs single (same rows)")
    for kd in kinds:
        for arm in ("foldbag", "boot"):
            d = df[f"{kd}_{arm}"] - df[f"{kd}_single"]
            ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
            print(f"  {kd:8s} {arm:8s} {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")
    return df


def build(y, mats, kinds, te, C, submit_name):
    """Cross-fit as usual, but keep BOTH test-side estimators.

    The OOF side is byte-identical to blend_lab's, so the cross-fitted CV of the two
    variants is the same number by construction -- CV cannot choose between them and this
    bench does not pretend otherwise. What it can report is how far apart the two test
    vectors actually are, which decides whether the fold-bagged file is a real second
    candidate or a duplicate not worth a slot.
    """
    folds = get_folds(y)
    oof, t_full, t_fold = {}, {}, {}
    for kd in kinds:
        Z, Zt = mats[kd]
        mo = np.zeros(len(y))
        tb = np.zeros(len(Zt))
        t0 = time.time()
        for itr, iva in folds:
            d_va, d_te = fit_df(Z, y, itr, [Z[iva], Zt], C)
            mo[iva] = d_va
            tb += d_te / len(folds)
        oof[kd], t_fold[kd] = mo, tb
        (t_full[kd],) = fit_df(Z, y, np.arange(len(y)), [Zt], C)
        sp = spearmanr(t_full[kd], t_fold[kd]).statistic
        print(f"  cross-fitted stack_{kd:8s} {roc_auc_score(y, mo):.6f}   "
              f"spearman(full, foldbag) on test {sp:.6f}   ({time.time()-t0:.0f}s)",
              flush=True)

    o_ens = np.mean([rk(oof[kd]) for kd in kinds], 0)
    cv = roc_auc_score(y, o_ens)
    ens_full = np.mean([rk(t_full[kd]) for kd in kinds], 0)
    ens_fold = np.mean([rk(t_fold[kd]) for kd in kinds], 0)
    sp = spearmanr(ens_full, ens_fold).statistic
    nmove = int((np.abs(rankdata(ens_full) - rankdata(ens_fold)) > 0.5 * len(ens_full) / 1000).sum())
    print(f"\ncross-fitted rank-ensemble over {kinds}: {cv:.6f}")
    print(f"ensemble test vectors: spearman(full, foldbag) {sp:.8f}, "
          f"{nmove:,} of {len(ens_full):,} rows move by >0.1% of the field")

    if submit_name:
        write(te, ens_fold, o_ens, submit_name, cv)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="rankraw")
    ap.add_argument("--drop", default=",".join(HONEST_DROP) + ",bolt_xgb_d7_alt2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--bag", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--submit-name", default=None)
    a = ap.parse_args()

    kinds = [k for k in a.kinds.split(",") if k]
    t0 = time.time()
    names, y, mats, te = load_all(tuple(kinds), set(filter(None, a.drop.split(","))))
    print(f"loaded {len(names)} members in {time.time()-t0:.0f}s", flush=True)

    if a.reps:
        paired(y, mats, kinds, a.reps, a.C, a.bag)
    if a.build or a.submit_name:
        build(y, mats, kinds, te, a.C, a.submit_name)
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
