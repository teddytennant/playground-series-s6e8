"""XGBoost members on the frozen folds, in three different upstream pipelines.

WHY THE PIPELINE, NOT THE HYPERPARAMETERS
-----------------------------------------
The pack already holds ~40 XGBoost members. Two runs have now measured that GBDT
hyperparameter variation inside our own pipeline buys nothing: LightGBM tuned for solo
AUC was +0.00003 (a null), and LightGBM tuned for *decorrelation* -- depth 7 -> 3, leaves
96 -> 8 -- still landed at maxcorr 0.9961 against its own sibling. Where a member lands
is set by its function class and its upstream, not by its knobs.

What did pay, by twenty times, was 63 members from three independent authors, of which
the single largest share came from 35 ordinary XGB/LGBM/CatBoost models. Same families we
already held; different imputation, encoding and features. So the lever is the upstream.

THE THREE MODES
---------------
`lat`  our own pipeline: constrained imputation + full-resolution lattice target
       encoding, read from cache/f*_X*.npy. 184 dense float columns. Predicted to be
       worth ~nothing -- it is the pipeline we already hold, and it is built anyway so
       the null is measured rather than assumed, and so the other two have a same-folds
       same-model control to be read against.

`cat`  TE-free, and the one genuinely new function class here. Every column -- including
       all nine numerics -- is handed over as an UNORDERED pandas categorical at full
       lattice resolution. That is legitimate rather than perverse: the generator rounded
       every value, so each numeric column already is a lattice of a few thousand levels
       with ~500 rows each. XGBoost then splits categoricals by sorting the levels on
       their gradient/hessian ratio at each node and cutting the sorted list, i.e. it
       estimates a target statistic per level *inside the node, refitted at every split*.
       That is a different estimator from our fold-safe smoothed mean TE (one map per
       outer fold, smoothing 20, frozen before the model sees it) and from CatBoost's
       ordered target statistics. Ordering is discarded on purpose: it makes the member
       wrong in a direction no tree in the pack is wrong in.

`raw`  the 12 columns exactly as the generator wrote them. Numerics stay numeric with
       NaN preserved so XGBoost learns a default direction per node; the 3 real
       categoricals stay categorical. No imputation, no TE, no lattice, no ratios. The
       encoding channel worth +0.0023 to everyone else is simply absent, so whatever this
       member gets right, it gets right by a route nothing else in the pack uses.

`latcat`
       the `lat` frame with the 12 unordered lattice categoricals APPENDED as extra
       columns (prefixed `K_`) rather than substituted for it. This is the one-factor
       version of the `cat` experiment. `cat` changed two things at once -- it discarded
       the numeric ordering AND deleted the entire target-encoding channel -- and the
       member that came back was decorrelated (maxcorr 0.9746) but much worse (0.9611
       solo), which is why it bought nothing. Here the TE channel is held fixed and the
       ordering-discarded channel is added on top, so any movement in either statistic is
       attributable to that channel alone. Run it against `lat` on the same folds, same
       hyperparameters and same round count -- `lat` is the control and exists for no
       other reason.

Note `cat` keeps NaN as *missing* (default-direction), where the sibling CatBoost
`native` mode lifts NaN to its own level. Deliberate: two members, not one member twice.

HONESTY
-------
Nothing early-stops on the rows that become a member's OOF. `--probe` carves its holdout
out of fold 0's TRAINING rows only and saves nothing; the round count it suggests is then
frozen into `--rounds` for the real 5-fold run, which uses no eval set at all. This is
the defect golem_a/golem_f are dropped for and that our own lgbm_tuned_lat* members had
to be retrained to remove.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, CAT, N_SPLITS, NUM, OOF, SEED, TARGET, get_folds, load_raw, save_preds  # noqa: E402
from run_lgbm import lattice  # noqa: E402


def frames(tr, te, mode):
    """Build the (train, test) design frames for the TE-free modes.

    Category alphabets are taken over train+test together. That is transductive
    preprocessing of the feature *alphabet* and uses no labels, so it leaks nothing --
    the same argument the constrained imputation already runs on.
    """
    cols = NUM + CAT
    n = len(tr)
    both = pd.concat([tr[cols], te[cols]], axis=0, ignore_index=True)
    out = pd.DataFrame(index=both.index)
    def as_cat(s):
        # XGBoost rejects a category INDEX of floating dtype, and the numeric columns'
        # lattice levels are floats. Factorize to integer level ids instead. `sort=True`
        # makes the id order the value order, which is irrelevant to the model (splits
        # are partition-based) but keeps the ids reproducible run to run. NaN factorizes
        # to code -1, which from_codes turns back into a missing value, so XGBoost routes
        # it by the node's learned default direction rather than treating it as a level.
        codes, uniq = pd.factorize(s, sort=True)
        return pd.Categorical.from_codes(codes, categories=np.arange(len(uniq), dtype="int32"))

    if mode in ("cat", "latcat"):
        for c in cols:
            out[c] = as_cat(both[c])
    else:  # raw
        for c in NUM:
            out[c] = both[c].astype("float32")
        for c in CAT:
            out[c] = as_cat(both[c])
    return out.iloc[:n].reset_index(drop=True), out.iloc[n:].reset_index(drop=True)


def describe(F):
    bits = []
    for c in F.columns:
        bits.append(f"{c}={len(F[c].cat.categories)}" if str(F[c].dtype) == "category"
                    else f"{c}=num")
    return ", ".join(bits)


def gain_share(m):
    """Fraction of total split gain the model spent on the added `K_` categoricals.

    The direct answer to the obvious objection to `latcat`: with 184 target-encoded
    columns already present, does the ordering-discarded channel get used at all, or is
    it simply never selected? A share near 0 means the member is the `lat` control with
    extra columns attached and cannot possibly be decorrelated by them.
    """
    g = m.get_score(importance_type="total_gain")
    tot = sum(g.values())
    return sum(v for k, v in g.items() if k.startswith("K_")) / tot if tot else 0.0


def build_params(a):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
             max_depth=a.depth, eta=a.eta, subsample=a.subsample,
             colsample_bytree=a.colsample, min_child_weight=a.min_child_weight,
             reg_lambda=a.reg_lambda, max_bin=a.max_bin, nthread=a.threads, seed=a.seed)
    if a.mode in ("cat", "raw", "latcat"):
        # max_cat_to_onehot=1 forces the partition-based split for every categorical,
        # including the 2-3 level ones, so the whole frame goes through one mechanism.
        p.update(max_cat_to_onehot=1, max_cat_threshold=a.max_cat_threshold)
    return p


def _latframe(dense, names, K):
    """Dense lattice block + the `K_`-prefixed unordered categoricals, as one frame.

    Assigned column by column rather than concat'd so the 184-column float block keeps
    referencing `dense` instead of being copied (~400 MB per fold).
    """
    F = pd.DataFrame(dense, columns=names, copy=False)
    for c in K.columns:
        F[f"K_{c}"] = K[c].to_numpy()
    return F


def fold_matrices(a, f, itr, iva, y, Ftr, Fte, Ltr, Lte):
    """(Xa, ya, Xb, yb, Xt_getter) for one fold, in whichever mode is active."""
    if a.mode in ("lat", "latcat"):
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
        if a.frac:
            Xa = np.hstack([Xa, Ltr[itr]])
            Xb = np.hstack([Xb, Ltr[iva]])
        get_Xt = (lambda: np.hstack([g("Xt"), Lte])) if a.frac else (lambda: g("Xt"))
        if a.mode == "lat":
            return Xa, ya, Xb, yb, get_Xt
        # cache rows are written as Xtr.iloc[itr] / Xtr.iloc[iva] (build_cache.py), so
        # positional alignment against the same index arrays is exact.
        names = [f"c{i}" for i in range(Xa.shape[1])]
        return (_latframe(Xa, names, Ftr.iloc[itr].reset_index(drop=True)), ya,
                _latframe(Xb, names, Ftr.iloc[iva].reset_index(drop=True)), yb,
                (lambda: _latframe(get_Xt(), names, Fte)))
    return Ftr.iloc[itr], y[itr], Ftr.iloc[iva], y[iva], (lambda: Fte)


def dm(X, y=None, cat=False):
    return xgb.DMatrix(X, label=y, missing=np.nan, enable_categorical=cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=("lat", "cat", "raw", "latcat"), default="lat")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample", type=float, default=0.5)
    ap.add_argument("--min-child-weight", type=float, default=60.0)
    ap.add_argument("--reg-lambda", type=float, default=40.0)
    ap.add_argument("--max-bin", type=int, default=512)
    ap.add_argument("--max-cat-threshold", type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--frac", action="store_true", help="lat mode: append the decimal lattice")
    ap.add_argument("--folds", default="")
    ap.add_argument("--probe", type=float, default=0.0,
                    help="tune-only: hold out this fraction of fold 0's TRAINING rows, "
                         "report the eval curve, save nothing")
    ap.add_argument("--probe-stopping", type=int, default=150)
    a = ap.parse_args()

    params = build_params(a)
    print(f"[{a.name}] mode={a.mode} {params} rounds={a.rounds} frac={a.frac}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    Ftr = Fte = Ltr = Lte = None
    if a.mode in ("cat", "raw", "latcat"):
        Ftr, Fte = frames(tr, te, a.mode)
        print(f"  {Ftr.shape[1]} categorical columns: {describe(Ftr)}", flush=True)
    if a.frac and a.mode in ("lat", "latcat"):
        Ltr, Lte = lattice(tr), lattice(te)

    use_cat = a.mode in ("cat", "raw", "latcat")   # frame carries category dtypes
    fresh = a.mode in ("lat", "latcat")            # fold matrices are per-fold, deletable

    # ---- probe: choose rounds/params without ever seeing a validation fold's labels ----
    if a.probe:
        itr, iva = folds[0]
        Xa, ya, _, _, _ = fold_matrices(a, 0, itr, iva, y, Ftr, Fte, Ltr, Lte)
        ia, ie = train_test_split(np.arange(len(ya)), test_size=a.probe,
                                  random_state=SEED, stratify=ya)
        sl = (lambda X, i: X.iloc[i]) if use_cat else (lambda X, i: X[i])
        d_tr = dm(sl(Xa, ia), ya[ia], use_cat)
        d_ev = dm(sl(Xa, ie), ya[ie], use_cat)
        hist = {}
        t0 = time.time()
        m = xgb.train(params, d_tr, num_boost_round=a.rounds,
                      evals=[(d_ev, "inner")], evals_result=hist,
                      early_stopping_rounds=a.probe_stopping or None, verbose_eval=200)
        curve = hist["inner"]["auc"]
        best = int(np.argmax(curve))
        print(f"[{a.name}] PROBE best inner AUC {curve[best]:.6f} @ round {best+1} "
              f"of {len(curve)} ({time.time()-t0:.0f}s) -- nothing saved", flush=True)
        if a.mode == "latcat":
            print(f"[{a.name}] PROBE {gain_share(m):.4f} of total split gain went to "
                  f"the 12 K_ categoricals", flush=True)
        return

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        Xa, ya, Xb, yb, get_Xt = fold_matrices(a, f, itr, iva, y, Ftr, Fte, Ltr, Lte)
        d = dm(Xa, ya, use_cat)
        if fresh:
            del Xa
        m = xgb.train(params, d, num_boost_round=a.rounds)
        del d
        oof[iva] = m.predict(dm(Xb, cat=use_cat))
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.6f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if fresh:
            del Xb
        if a.mode == "latcat":
            print(f"[{a.name}] fold {f}: {gain_share(m):.4f} of total split gain went to "
                  f"the 12 K_ categoricals", flush=True)
        Xt = get_Xt()
        tp += m.predict(dm(Xt, cat=use_cat)) / N_SPLITS
        del m
        if fresh:
            del Xt

    if len(want) < N_SPLITS:
        print(f"[{a.name}] partial run, nothing saved", flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.6f}  ({time.time()-t0:.0f}s)", flush=True)
    save_preds(a.name, oof, tp, len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), mode=a.mode, rounds=a.rounds, frac=a.frac,
                   **params),
              open(os.path.join(OOF, f"summary_{a.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
