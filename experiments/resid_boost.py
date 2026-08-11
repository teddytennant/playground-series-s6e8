"""The assumption-free version of the angle: boost ON TOP of the stack and see if it moves.

WHY THIS AND NOT THE CHI-SQUARE
-------------------------------
`erroran.py` stratifies on quantile bins of the stack score z and asks whether y still
depends on x inside a bin. Two things poison that at the precision this workspace works
at, and `erroran_na.py` caught the first one red-handed:

1. The chi2/df null is NOT 1.0. It is a function of cell density, and at 2048 z-bins with
   a binary x it sits at ~2.15 -- so the NaN indicators that looked like the top finding
   in the whole scan were exactly at their own matched null.
2. Even with a matched marginal, a permuted control destroys the x-z correlation as well
   as the x-y one. Inside a z-bin, z itself still varies; a real feature correlated with z
   picks that variation up and scores an excess the control cannot produce. The strongest
   predictor in the frame is therefore guaranteed to look like the strongest finding
   whether or not anything is left in it.

Both problems come from replacing z with a bin. So don't. Hand the corrector the
CONTINUOUS stack log-odds as an offset it cannot alter, and let it try to earn AUC on top
with the features:

    logit P(y=1) = logitz  +  f(features)          f fitted, logitz fixed

`f` is free to be any shape, uses interactions automatically, and is regularised by its
own learning rate rather than by a cell count -- so unlike the per-cell empirical
correction it can express a small smooth residual without paying for 26k free parameters.
Out-of-fold on the frozen folds, so the number it reports is directly comparable to every
CV in the journal.

THE CONTROL IS THE POINT. The same run with every feature column independently permuted
gives the null for this exact procedure including the optimism of reading the best round
off the curve. Judge only `real minus control`.
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from features import make_frames  # noqa: E402

CHECK = [25, 50, 100, 200, 350, 500, 750, 1000]


def run(X, y, logitz, folds, params, rounds, tag):
    oof = {r: np.zeros(len(y)) for r in CHECK if r <= rounds}
    for k, (itr, iva) in enumerate(folds):
        dtr = lgb.Dataset(X.iloc[itr], label=y[itr], init_score=logitz[itr],
                          free_raw_data=False)
        m = lgb.train(params, dtr, num_boost_round=rounds)
        for r in oof:
            oof[r][iva] = m.predict(X.iloc[iva], num_iteration=r, raw_score=True)
        print(f"  [{tag}] fold{k} done", flush=True)
    return {r: roc_auc_score(y, logitz + v) for r, v in oof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="blend158_h3")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--leaves", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    z = np.load(os.path.join(SUB, f"oof_{a.stack}.npy"))
    r = pd.Series(z).rank(method="average").to_numpy() / (len(z) + 1.0)
    logitz = np.log(r / (1 - r))
    base = roc_auc_score(y, logitz)

    Xtr, _, _, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    print(f"stack {a.stack}  base OOF AUC {base:.6f}   frame {Xtr.shape}", flush=True)

    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=500, feature_fraction=1.0, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=a.threads,
                  seed=42, deterministic=True, force_row_wise=True)
    folds = get_folds(y)

    real = run(Xtr, y, logitz, folds, params, a.rounds, "real")
    rng = np.random.default_rng(11)
    Xp = Xtr.apply(lambda c: c.iloc[rng.permutation(len(c))].to_numpy())
    ctrl = run(Xp, y, logitz, folds, params, a.rounds, "ctrl")

    print(f"\nbase {base:.6f}")
    print(f"{'round':>7s} {'real':>10s} {'d_real':>10s} {'ctrl':>10s} {'d_ctrl':>10s}"
          f" {'real-ctrl':>10s}")
    for rr in sorted(real):
        print(f"{rr:7d} {real[rr]:10.6f} {real[rr]-base:+10.6f} {ctrl[rr]:10.6f}"
              f" {ctrl[rr]-base:+10.6f} {real[rr]-ctrl[rr]:+10.6f}", flush=True)


if __name__ == "__main__":
    main()
