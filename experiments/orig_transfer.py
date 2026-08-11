"""Does the real source dataset carry signal that transfers to the synthetic frame?

The workspace has been repeating "concatenating the original dataset measures -0.0001"
since day one, but that number is inherited from tomasa2's public ablation notebook and
has never been measured here. This script measures the three things that decide whether
the original is worth anything at all, all of them cheap:

  1. self  -- 5-fold AUC of a model trained on the 7,500 original rows, scored on the
              original. Establishes how much signal the real data holds at all.
  2. fwd   -- that same model, fit on all 7,500 originals, scored against the 691k
              competition train rows. This is the transfer number. A model that never
              saw a competition label is leakage-free, so if it transfers it can be
              used as a feature inside every fold.
  3. rev   -- a competition-trained model scored on the 7,500 originals. Tells whether
              the synthetic structure is a superset of the real one or a different one.

Run: .venv/bin/python experiments/orig_transfer.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CAT, DATA, NUM, SEED, TARGET, get_folds, load_raw  # noqa: E402

import lightgbm as lgb  # noqa: E402

ORIG = os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
COLS = NUM + CAT

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=40, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, seed=SEED, num_threads=os.cpu_count())


def as_frame(df):
    """12 raw predictors, categoricals as pandas category with a shared vocabulary."""
    X = df[COLS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=sorted(VOCAB[c]))
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def main():
    global VOCAB
    orig = pd.read_csv(ORIG)
    tr, te = load_raw()
    VOCAB = {c: sorted(set(orig[c].dropna()) | set(tr[c].dropna())) for c in CAT}

    Xo, yo = as_frame(orig), orig[TARGET].to_numpy()
    Xc, yc = as_frame(tr), tr[TARGET].to_numpy()

    # --- 1. self: how much signal does the real data hold on its own terms? ---
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(orig))
    for a, b in skf.split(Xo, yo):
        m = lgb.train(PARAMS, lgb.Dataset(Xo.iloc[a], yo[a]), num_boost_round=400)
        oof[b] = m.predict(Xo.iloc[b])
    self_auc = roc_auc_score(yo, oof)
    print(f"self  (orig 5-fold, orig)          {self_auc:.6f}", flush=True)

    # --- 2. fwd: orig-trained model scored on the competition frame ---
    full = lgb.train(PARAMS, lgb.Dataset(Xo, yo), num_boost_round=400)
    p_tr = full.predict(Xc)
    p_te = full.predict(as_frame(te))
    fwd_auc = roc_auc_score(yc, p_tr)
    print(f"fwd   (orig-trained, comp train)   {fwd_auc:.6f}", flush=True)
    np.save(os.path.join(DATA, "orig", "origmodel_train.npy"), p_tr)
    np.save(os.path.join(DATA, "orig", "origmodel_test.npy"), p_te)

    # --- 3. rev: competition-trained model scored on the originals ---
    #     one fold's worth of competition data is plenty and keeps this cheap.
    a, _ = get_folds(yc)[0]
    comp_m = lgb.train(PARAMS, lgb.Dataset(Xc.iloc[a], yc[a]), num_boost_round=600)
    rev_auc = roc_auc_score(yo, comp_m.predict(Xo))
    print(f"rev   (comp-trained, orig rows)    {rev_auc:.6f}", flush=True)

    # single-column marginals, both frames, to see which channels survive the generator
    print("\ncolumn            orig AUC   comp AUC")
    for c in NUM:
        vo = orig[c].to_numpy(dtype=float)
        vc = pd.to_numeric(tr[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(vc)
        print(f"{c:26s} {roc_auc_score(yo, vo):.4f}   {roc_auc_score(yc[ok], vc[ok]):.4f}")


if __name__ == "__main__":
    main()
