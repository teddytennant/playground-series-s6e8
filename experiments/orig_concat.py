"""The literal angle: concatenate the 7,500 real rows as extra training data.

The workspace has repeated "concatenating the original dataset measures -0.0001" since
day one on the strength of tomasa2's public ablation notebook. It has never been measured
here. This does that, on the frozen folds, with the extra rows going into the TRAINING
half only -- never into a validation fold, which would make the number meaningless.

Deliberately run on the 12 raw columns with native NaN handling rather than on the full
lattice/target-encoding frame. Two reasons: it is ~15x cheaper, and 7,500 extra rows
inside a target encoding computed over 691,369 is a 1% perturbation of every cell, so the
TE machinery can only dilute whatever effect exists. If concatenation helps anywhere it
helps in the splits, and this frame is where the splits are.

`--weight` repeats the original rows so they carry more than 1% of the training mass;
without it the test is arguably rigged to find nothing.

    .venv/bin/python experiments/orig_concat.py --weight 1,10,50
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CAT, DATA, NUM, SEED, TARGET, get_folds, load_raw  # noqa: E402

import lightgbm as lgb  # noqa: E402

ORIG = os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
COLS = NUM + CAT

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
              min_data_in_leaf=100, feature_fraction=1.0, bagging_fraction=0.9,
              bagging_freq=1, verbosity=-1, seed=SEED, num_threads=os.cpu_count())


def as_frame(df, vocab):
    X = df[COLS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=vocab[c])
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="0,1,10")
    ap.add_argument("--rounds", type=int, default=900)
    a = ap.parse_args()

    orig = pd.read_csv(ORIG)
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    vocab = {c: sorted(set(orig[c].dropna()) | set(tr[c].dropna())) for c in CAT}
    X = as_frame(tr, vocab)
    Xo, yo = as_frame(orig, vocab), orig[TARGET].to_numpy(dtype=float)
    folds = get_folds(y)

    for w in [int(v) for v in a.weight.split(",")]:
        oof = np.zeros(len(tr))
        for a_idx, b_idx in folds:
            Xa, ya = X.iloc[a_idx], y[a_idx]
            if w:
                Xa = pd.concat([Xa] + [Xo] * w, ignore_index=True)
                ya = np.concatenate([ya] + [yo] * w)
            m = lgb.train(PARAMS, lgb.Dataset(Xa, ya), num_boost_round=a.rounds)
            oof[b_idx] = m.predict(X.iloc[b_idx])
        tag = "baseline (no extra rows)" if w == 0 else f"+{w}x the 7,500 originals"
        print(f"{tag:32s} OOF AUC {roc_auc_score(y, oof):.6f}", flush=True)


if __name__ == "__main__":
    main()
