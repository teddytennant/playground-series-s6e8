"""XGBoost on the cached lattice matrices, fixed schedule, no early stopping.

The pack already holds ~40 XGBoost members, but every one of them was built by someone
else's pipeline on someone else's features. The slot-3 attribution is explicit that this
is the thing that matters: the 35 ordinary GBDTs in `rest` contributed the single largest
share of that run's +0.000340 precisely because their author's imputation, encoding and
feature decisions differ from ours even though the model family does not.

Fixed `--rounds`, never early stopping on the validation fold -- see the note in
agent/run_lgbm.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, N_SPLITS, OOF, TARGET, get_folds, load_raw, save_preds  # noqa: E402
from run_lgbm import lattice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--frac", action="store_true")
    ap.add_argument("--folds", default="")
    a = ap.parse_args()

    params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                  max_depth=a.depth, eta=a.eta, subsample=0.9, colsample_bytree=0.5,
                  min_child_weight=60, reg_lambda=40.0, max_bin=512,
                  nthread=16, seed=a.seed)
    print(f"[{a.name}] {params} rounds={a.rounds} frac={a.frac}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    Ftr = Fte = None
    if a.frac:
        Ftr, Fte = lattice(tr), lattice(te)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
        if a.frac:
            Xa = np.hstack([Xa, Ftr[itr]])
            Xb = np.hstack([Xb, Ftr[iva]])
        d = xgb.DMatrix(Xa, label=ya, missing=np.nan)
        del Xa
        m = xgb.train(params, d, num_boost_round=a.rounds)
        del d
        oof[iva] = m.predict(xgb.DMatrix(Xb, missing=np.nan))
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb
        Xt = g("Xt")
        if a.frac:
            Xt = np.hstack([Xt, Fte])
        tp += m.predict(xgb.DMatrix(Xt, missing=np.nan)) / N_SPLITS
        del Xt, m

    if len(want) < N_SPLITS:
        print(f"[{a.name}] partial run, nothing saved", flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    save_preds(a.name, oof, tp, len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), rounds=a.rounds, frac=a.frac, **params),
              open(os.path.join(OOF, f"summary_{a.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
