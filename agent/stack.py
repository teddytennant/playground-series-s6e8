"""Combine the public 74-model OOF library with our own members.

EVERY member here is out-of-fold on the one frozen scheme
(StratifiedKFold(5, shuffle=True, random_state=42), original row order), which is the
only reason these arrays may be stacked together at all. `--verify` re-scores every
library member against `manifest.csv` to prove the row alignment before trusting any of it.

HONEST EVALUATION
-----------------
A combiner fitted on the OOF matrix and scored on that same matrix reads high. So every
combiner is scored by repeated 50/50 stratified splits: fit on half the rows, score on
the other half, and compare variants by **paired** differences on the same splits so the
split noise cancels. That paired number is what selection uses.

The reported headline is a fold-cross-fitted score on the frozen folds, which is the
number comparable to a single model's OOF AUC.

WHY LOGITS
----------
The target saturates hard (the top screen-time decile has an addiction rate of 1.000),
where probabilities have no resolution left and logits still do. Stacking on
clip(log(p/(1-p)), +/-30) was worth +0.00047 on a diverse library. A linear stacker can
also give a weak-but-decorrelated member a NEGATIVE coefficient, using it as a
correction; a hill climber, which only averages, cannot.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, LIB, OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402


# golem_a and golem_f early-stop on the held-out validation fold, which their author
# discloses as mildly optimistic and uncorrected. An optimistic OOF earns undeserved
# stacker weight. experiments/member_value.py measured them: including them takes the
# paired gain from +0.000009 (consistent) to +0.000007 with the SIGN FLIPPING across
# splits. Measured out, not assumed out.
DEFAULT_DROP = ("golem_a", "golem_f")


def to_logit(p, clip=30.0):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def load_members(y, n_test, extra_dirs=(), drop=()):
    """Load every oof_*/test_* pair from the library plus our own oof dir."""
    names, oofs, tests = [], [], []
    for d in [os.path.join(LIB, "oof"), OOF, *extra_dirs]:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.startswith("oof_") or not fn.endswith(".npy"):
                continue
            nm = fn[4:-4]
            if nm in drop or nm in names:
                continue
            tp = os.path.join(d, f"test_{nm}.npy")
            if not os.path.exists(tp):
                continue
            o, t = np.load(os.path.join(d, fn)), np.load(tp)
            if o.shape != (len(y),) or t.shape != (n_test,):
                print(f"[skip] {nm}: shapes {o.shape} {t.shape}")
                continue
            names.append(nm)
            oofs.append(o.astype(np.float64))
            tests.append(t.astype(np.float64))
    return names, np.column_stack(oofs), np.column_stack(tests)


def verify(names, O, y):
    """Re-score every member and compare against the library manifest."""
    man = pd.read_csv(os.path.join(LIB, "manifest.csv")).set_index("model")["oof_auc"]
    rows = []
    for i, nm in enumerate(names):
        a = roc_auc_score(y, O[:, i])
        exp = man.get(nm, np.nan)
        rows.append(dict(model=nm, computed=a, manifest=exp, diff=a - exp))
    df = pd.DataFrame(rows).sort_values("computed", ascending=False)
    bad = df[df["diff"].abs() > 5e-5].dropna(subset=["manifest"])
    print(df.to_string(index=False, float_format="%.5f"))
    if len(bad):
        print("\n!! MISALIGNED MEMBERS -- do not stack these:")
        print(bad.to_string(index=False, float_format="%.5f"))
        raise SystemExit(1)
    print(f"\nall {df['manifest'].notna().sum()} manifest members reproduce "
          f"to <5e-5 -- row alignment confirmed")
    return df


def hill_climb(P, yy, n_iter=60):
    """Greedy forward selection with replacement; weights are counts/total, so >= 0."""
    k = P.shape[1]
    picks = np.zeros(k, int)
    j = int(np.argmax([roc_auc_score(yy, P[:, c]) for c in range(k)]))
    picks[j] = 1
    cur = P[:, j].astype(np.float64).copy()
    best, n = roc_auc_score(yy, cur), 1
    for _ in range(n_iter):
        cb, cj = best, -1
        for t in range(k):
            a = roc_auc_score(yy, (cur * n + P[:, t]) / (n + 1))
            if a > cb + 1e-9:
                cb, cj = a, t
        if cj < 0:
            break
        cur = (cur * n + P[:, cj]) / (n + 1)
        n += 1
        picks[cj] += 1
        best = cb
    return picks / max(picks.sum(), 1)


def combiners(y_tr, Z_tr, Z_te, R_tr, R_te):
    """Return {name: (fit_predict on held-out)} for each candidate combiner."""
    out = {}
    for C in ([0.01, 0.1, 1.0, 10.0]):
        m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
        out[f"logit_stack_C{C}"] = m.predict_proba(Z_te)[:, 1]
    m = LogisticRegression(max_iter=3000).fit(R_tr, y_tr)
    out["prob_stack"] = m.predict_proba(R_te)[:, 1]
    w = hill_climb(R_tr, y_tr)
    out["hill_climb"] = R_te @ w
    out["mean_all"] = R_te.mean(1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--submit-name", default=None)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--drop", default=",".join(DEFAULT_DROP))
    ap.add_argument("--ext", action="store_true",
                    help="also load data/ext_members (FM + golem libraries)")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = (os.path.join(DATA, "ext_members"),) if a.ext else ()
    names, O, T = load_members(y, len(te), extra_dirs=extra,
                               drop=set(filter(None, a.drop.split(","))))
    print(f"{len(names)} members loaded (ext={a.ext}, dropped={a.drop})\n")

    if a.verify:
        verify(names, O, y)

    Z, Zt = to_logit(O), to_logit(T)

    # --- honest comparison of combiners: paired 50/50 splits ---
    rows = []
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        res = combiners(y[iA], Z[iA], Z[iB], O[iA], O[iB])
        res["best_solo"] = None
        r = {k: (roc_auc_score(y[iB], v) if v is not None else
                 max(roc_auc_score(y[iB], O[iB, j]) for j in range(O.shape[1])))
             for k, v in res.items()}
        rows.append(r)
    rob = pd.DataFrame(rows)
    print("\nheld-out AUC per 50/50 split")
    print(rob.to_string(float_format="%.6f"))
    print("\nPAIRED DIFFERENCES vs best_solo (same rows, split noise cancels)")
    for c in rob.columns:
        if c == "best_solo":
            continue
        d = rob[c] - rob["best_solo"]
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  {c:>18s}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")

    # --- headline: cross-fitted on the frozen folds ---
    folds = get_folds(y)
    mo = np.zeros(len(y))
    for itr, iva in folds:
        mo[iva] = (LogisticRegression(max_iter=3000, C=a.C)
                   .fit(Z[itr], y[itr]).predict_proba(Z[iva])[:, 1])
    cv = roc_auc_score(y, mo)
    print(f"\ncross-fitted logit stack (C={a.C}) OOF = {cv:.6f}")
    print(f"best single member          OOF = "
          f"{max(roc_auc_score(y, O[:, j]) for j in range(O.shape[1])):.6f}")

    meta = LogisticRegression(max_iter=3000, C=a.C).fit(Z, y)
    coef = pd.DataFrame({"member": names, "coef": meta.coef_[0],
                         "solo": [roc_auc_score(y, O[:, j]) for j in range(len(names))]})
    print("\ntop/bottom stacker coefficients")
    cs = coef.sort_values("coef", ascending=False)
    print(pd.concat([cs.head(12), cs.tail(6)]).to_string(index=False,
                                                         float_format="%.4f"))

    if a.submit_name:
        pred = meta.predict_proba(Zt)[:, 1]
        sub = pd.DataFrame({"id": te["id"].to_numpy(), TARGET: pred})
        p = os.path.join(SUB, f"{a.submit_name}.csv")
        sub.to_csv(p, index=False)
        # NOTE: stack outputs live in submissions/, never in OOF/ -- a stack's own
        # predictions must never be picked up as a member by the next stacking run.
        np.save(os.path.join(SUB, f"oof_{a.submit_name}.npy"), mo)
        print(f"\nwrote {p}  rows={len(sub):,}  mean {pred.mean():.4f} "
              f"(train rate {y.mean():.4f})  cross-fitted CV {cv:.6f}")


if __name__ == "__main__":
    main()
