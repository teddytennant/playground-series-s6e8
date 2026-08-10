"""Non-linear, regime-aware meta-models over the member logit matrix.

WHY THIS AND NOT ANOTHER MEMBER
-------------------------------
The 88-member stack is saturated: a new function class (factorization machines) bought
+0.000005 and a new channel inside a strong new GBDT bought +0.000002, against a
0.00039 gap to the top of the public LB. Adding members is provably not the lever.

What the linear stack cannot do is vary its weights by ROW. It fits ONE coefficient
vector for all 691,369 rows, yet member accuracy collapses from 0.9754 on complete rows
to 0.9111 on rows missing >=5 columns. If the members' RELATIVE strengths move across
that range -- and they plausibly do, since the lattice/target-encoding members lose their
signal exactly when the column they encode is absent -- a global stack is leaving that on
the table. A tree meta-model given both the logits and a regime descriptor can express
"trust `lookup` more when four columns are missing" and the linear one cannot.

HONESTY
-------
Every meta-model is scored the same way: cross-fitted on the frozen folds
(StratifiedKFold(5, shuffle=True, random_state=42)). Fit on 4 folds, predict the 5th,
score the assembled 691,369-row vector. That is the same protocol that produced the
0.969660 linear number, so the comparison is like-for-like, and the 5 per-fold deltas
give a sign-consistency check on top of the aggregate.

The `lin` feature (variant `resid`) is itself cross-fitted with an INNER 4-fold inside
each outer training set. Feeding a tree an in-sample linear prediction would let it learn
that the feature is already near-perfect on the rows it can see, and the correction it
then learns would be fitted to the linear model's overfitting rather than to signal.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CAT, NUM, SEED, TARGET  # noqa: E402

PRED = NUM + CAT


def to_logit(p, clip=30.0):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def regime_frame(df):
    """Row descriptors the stacker can condition on. Deliberately NOT predictive
    features in their own right -- no target encoding, no lattice. The members already
    extract the signal; this only has to describe WHICH REGIME a row sits in so the
    meta-model can decide who to trust."""
    out = {}
    na = df[PRED].isna()
    out["n_missing"] = na.sum(1).to_numpy(np.float32)
    for c in PRED:
        out[f"na_{c}"] = na[c].to_numpy(np.float32)
    for c in NUM:
        out[c] = df[c].to_numpy(np.float64)
    for c in CAT:
        out[c] = pd.Categorical(df[c]).codes.astype(np.float32)
    # first decimal digit of the strongest column: an 8.5-point swing in the base rate,
    # and the one place where the generator's rounding is visible to a meta-model.
    d = df["daily_screen_time_hours"].to_numpy(np.float64)
    f = np.where(np.isnan(d), np.nan, np.round((d - np.floor(d)) * 10) % 10)
    out["d1_daily"] = f
    return pd.DataFrame(out)


def meta_frame(Z, R, with_regime=True):
    """logits + cheap order statistics over them + (optionally) the regime block.

    The order statistics matter more than they look: a depth-6 tree cannot form the mean
    of 88 columns by splitting, so handing it the consensus and the disagreement spread
    directly is the difference between it modelling the ensemble and it modelling
    whichever three members it happened to split on first."""
    cols = {f"m{i:02d}": Z[:, i] for i in range(Z.shape[1])}
    cols["z_mean"] = Z.mean(1)
    cols["z_std"] = Z.std(1)
    cols["z_min"] = Z.min(1)
    cols["z_max"] = Z.max(1)
    cols["z_med"] = np.median(Z, 1)
    X = pd.DataFrame(cols)
    if with_regime:
        X = pd.concat([X, R.reset_index(drop=True)], axis=1)
    return X


def inner_linear(Z, y, itr, iva, C=1.0, n_inner=4):
    """Honest linear-stack feature: inner-cross-fitted on the outer training rows,
    full-fit for the outer validation rows."""
    lin_tr = np.zeros(len(itr))
    skf = StratifiedKFold(n_inner, shuffle=True, random_state=SEED)
    for a, b in skf.split(np.zeros(len(itr)), y[itr]):
        m = LogisticRegression(max_iter=3000, C=C).fit(Z[itr[a]], y[itr[a]])
        lin_tr[b] = m.decision_function(Z[itr[b]])
    m = LogisticRegression(max_iter=3000, C=C).fit(Z[itr], y[itr])
    return lin_tr, m.decision_function(Z[iva]), m
