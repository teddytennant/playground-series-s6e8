"""CatBoost members, on the frozen folds.

WHY CATBOOST IS NOT "JUST ANOTHER GBDT" HERE
--------------------------------------------
Slot 1 measured LightGBM hyperparameter tuning at +0.00003 (null) and slot 5 measured
that you cannot tune a LightGBM into a decorrelated member -- collapsing depth 7 -> 3 and
leaves 96 -> 8 still landed at maxcorr 0.9961 against its own sibling. The conclusion
recorded in the journal is that where a member lands is set by its *pipeline*, not its
hyperparameters, and that the unit that carries value is the whole upstream (imputation,
encoding, features), not the model.

So a CatBoost trained on `cache/f*_X*.npy` -- our constrained-imputation + lattice-TE
matrices -- is predicted to be worth ~nothing: same pipeline, and the pack already holds
~20 XGB/LGBM/CatBoost members from boltuzamaki plus beicicc's CatBoost. That variant is
built anyway (`--mode lat`) because the angle asks for an identical-folds comparison and
a measured null is worth more than an assumed one.

The variant that is a real experiment is `--mode native`, which replaces all three
upstream decisions at once:

  our pipeline                        native pipeline
  ------------------------------      --------------------------------------------
  constrained imputation from the     NaN becomes its own categorical LEVEL; the
  generator identity, added           generator identity is never used at all
  alongside the NaNs
  fold-safe smoothed mean TE          CatBoost ORDERED target statistics: a running
  (one map per outer fold,            mean over a random permutation, a different
  smoothing 20)                       estimator with a different bias/variance point
  184 dense float columns             12 categorical columns, nothing else
  leaf-wise trees                     oblivious (symmetric) trees

Every numeric column is cast to its lattice level and handed over as a *categorical*.
That is legitimate rather than perverse here: the generator rounded every value, so each
numeric column already IS a lattice of a few thousand repeated levels with ~500 rows
each (RESEARCH.md, "the quantisation lattice"). Ordering is genuinely discarded, which
is the point -- it makes the member wrong in a different direction from every tree model
in the pack, which is the only property that has been shown to pay.

HONESTY
-------
Nothing early-stops on the rows that become this member's OOF. The eval set for early
stopping is carved out of the fold's TRAINING rows only (`--inner`), so the iteration
count never sees a held-out label. This is the defect golem_a/golem_f are dropped for and
that our own lgbm_tuned_lat* members had to be retrained to remove.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, CAT, N_SPLITS, NUM, OOF, SEED, TARGET, get_folds, load_raw, save_preds  # noqa: E402


def native_frames(tr, te):
    """Every column -> an integer categorical code. Missing is its own level.

    Codes are assigned over train+test together. That is transductive preprocessing of
    the feature *alphabet* and uses no labels, so it leaks nothing -- the same argument
    the constrained imputation runs on.
    """
    cols = NUM + CAT
    n = len(tr)
    both = pd.concat([tr[cols], te[cols]], axis=0, ignore_index=True)
    out = pd.DataFrame(index=both.index)
    sizes = {}
    for c in cols:
        codes, uniq = pd.factorize(both[c], sort=True)
        # factorize gives -1 for NaN; lift it to a real level so "missing" is a value
        # the ordered target statistic estimates like any other.
        out[c] = (codes + 1).astype("int32")
        sizes[c] = len(uniq) + 1
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True), sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("native", "lat", "raw"), default="native")
    ap.add_argument("--iterations", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2", type=float, default=6.0)
    ap.add_argument("--ctr-complexity", type=int, default=1)
    ap.add_argument("--border-count", type=int, default=254)
    ap.add_argument("--inner", type=float, default=0.08,
                    help="fraction of the fold's TRAINING rows held out for early stopping")
    ap.add_argument("--stopping", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--folds", default="")
    ap.add_argument("--subsample-rows", type=int, default=0, help="timing probe only")
    a = ap.parse_args()

    params = dict(loss_function="Logloss", eval_metric="AUC", iterations=a.iterations,
                  learning_rate=a.lr, depth=a.depth, l2_leaf_reg=a.l2,
                  random_seed=a.seed, thread_count=a.threads, border_count=a.border_count,
                  bootstrap_type="Bernoulli", subsample=0.85,
                  max_ctr_complexity=a.ctr_complexity, one_hot_max_size=4,
                  allow_writing_files=False, verbose=False)
    print(f"[{a.name}] mode={a.mode} {params}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    Ntr = Nte = None
    cat_idx = None
    if a.mode == "native":
        Ntr, Nte, sizes = native_frames(tr, te)
        cat_idx = list(range(Ntr.shape[1]))
        print(f"  {Ntr.shape[1]} categorical columns, levels: "
              + ", ".join(f"{c}={sizes[c]}" for c in Ntr.columns), flush=True)
    elif a.mode == "raw":
        # The 12 columns exactly as the generator wrote them: numerics stay numeric
        # (CatBoost splits on ordered thresholds and handles NaN with its own default
        # direction), the 3 real categoricals stay categorical. No imputation, no target
        # encoding, no lattice, no ratios. This is the "no TE at all" pipeline the
        # journal names as the honest second lineage -- the encoding channel that carries
        # +0.0023 for everyone else is simply absent, so whatever it gets right it gets
        # right by a route nothing else in the pack uses.
        cols = NUM + CAT
        Ntr, Nte = tr[cols].copy(), te[cols].copy()
        for c in CAT:
            Ntr[c] = Ntr[c].fillna("__NA__").astype(str)
            Nte[c] = Nte[c].fillna("__NA__").astype(str)
        cat_idx = [cols.index(c) for c in CAT]
        print(f"  raw: {len(NUM)} numeric (NaN kept) + {len(CAT)} categorical", flush=True)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    iters = []
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        if a.mode in ("native", "raw"):
            Xa_full, ya_full = Ntr.iloc[itr], y[itr]
            Xb, yb = Ntr.iloc[iva], y[iva]
            Xt = Nte
        else:
            g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
            Xa_full, ya_full = g("Xa"), g("ya")
            Xb, yb = g("Xb"), g("yb")
            Xt = g("Xt")

        if a.subsample_rows:
            keep = np.random.RandomState(0).choice(len(ya_full), a.subsample_rows, False)
            Xa_full = Xa_full.iloc[keep] if Ntr is not None else Xa_full[keep]
            ya_full = ya_full[keep]

        # early stopping on TRAINING rows only -- never on iva, which becomes the OOF
        ia, ie = train_test_split(np.arange(len(ya_full)), test_size=a.inner,
                                  random_state=SEED, stratify=ya_full)
        sl = (lambda X, i: X.iloc[i]) if Ntr is not None else (lambda X, i: X[i])
        ptr = Pool(sl(Xa_full, ia), ya_full[ia], cat_features=cat_idx)
        pev = Pool(sl(Xa_full, ie), ya_full[ie], cat_features=cat_idx)
        m = CatBoostClassifier(**params)
        m.fit(ptr, eval_set=pev, early_stopping_rounds=a.stopping or None,
              use_best_model=bool(a.stopping))
        best = int(m.get_best_iteration() or m.tree_count_)
        iters.append(best)
        del ptr, pev

        oof[iva] = m.predict_proba(Pool(Xb, cat_features=cat_idx))[:, 1]
        tp += m.predict_proba(Pool(Xt, cat_features=cat_idx))[:, 1] / N_SPLITS
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.6f} "
              f"iters {best} ({time.time()-t0:.0f}s)", flush=True)
        del m

    if len(want) < N_SPLITS or a.subsample_rows:
        print(f"[{a.name}] partial run, nothing saved", flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    save_preds(a.name, oof, tp, len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), mode=a.mode, iters=iters,
                   **{k: v for k, v in params.items() if k != "verbose"}),
              open(os.path.join(OOF, f"summary_{a.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
