"""w15f stage 1b: time and size the teacher before committing to 25 nested fits.

Fits the teacher on outer fold 0 only and prints AUC at a ladder of round counts, plus
wall clock. Purpose is to fix ROUNDS for the nested run at a point where the teacher is
a competent model (the author's teacher rank-correlates 0.9908 with a strong blend, so it
has to be a real model, not a stub) without spending an hour per outer fold.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NTHREAD = 3

TEACHER = dict(objective="binary", learning_rate=0.05, num_leaves=96,
               min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.9,
               bagging_freq=1, lambda_l2=5.0, max_depth=-1, max_bin=511,
               verbosity=-1, num_threads=NTHREAD, seed=42,
               deterministic=True, force_row_wise=True)


def main():
    X = np.load(os.path.join(HERE, "w15f_X.npy"))
    cols = json.load(open(os.path.join(HERE, "w15f_cols.json")))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ntr = len(y)
    Xtr = X[:ntr]
    itr, iva = get_folds(y)[0]
    print(f"fold0 train {len(itr):,} valid {len(iva):,}  feats {Xtr.shape[1]}", flush=True)

    ds = lgb.Dataset(Xtr[itr], label=y[itr], feature_name=cols, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(TEACHER, ds, num_boost_round=3000)
    print(f"3000 rounds in {time.time()-t0:.0f}s", flush=True)
    for n in (600, 900, 1200, 1500, 2000, 2500, 3000):
        p = booster.predict(Xtr[iva], num_iteration=n)
        print(f"  rounds {n:>4}: valid AUC {roc_auc_score(y[iva], p):.6f}", flush=True)


if __name__ == "__main__":
    main()
