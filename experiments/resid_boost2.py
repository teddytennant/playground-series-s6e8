"""Residual booster, take two -- with the offset on the right scale.

WHAT WENT WRONG IN TAKE ONE
---------------------------
`resid_boost.py` used `logit(rank percentile of z)` as the LightGBM offset and the real
features LOST 0.0076 AUC against a permuted control that lost 0.00003. Boosting on top of
a fixed offset cannot destroy 76e-4 of AUC by chasing noise -- the control proves that --
so the loss was structural, and the curve says exactly what it was: monotone decline to
round 200, then partial RECOVERY out to 1000. A model chasing noise does not recover.

The rank logit is not a log-odds. It is logistic-distributed by construction with sd ~1.8,
while the true log-odds of this target has a different scale and shape entirely, so the
residual `y - sigmoid(offset)` is dominated by a large, smooth CALIBRATION error that is a
function of z. The corrector cannot see z. So it spends its capacity reconstructing z from
the features -- which it can only do to AUC ~0.967 -- and adds a correction computed
through that lossy proxy. That injects the proxy's error into the score. The late recovery
is the corrector slowly getting its reconstruction of z accurate enough to stop hurting.
Permuted features cannot reconstruct z at all, which is why the control was flat: the
control was never a null for the real run, it was a different experiment.

TWO FIXES, RUN AS TWO VARIANTS
------------------------------
A. `--mode offset` -- calibrate first. An out-of-fold isotonic map z -> P(y=1) makes the
   offset a genuine log-odds, so the residual has no calibration component left and the
   only thing to chase is feature signal. Isotonic is monotone within a fold, so the base
   ranking is essentially untouched.
B. `--mode feature` -- do not use an offset at all. Hand the corrector the stack score AS
   A FEATURE alongside the raw columns. Now it can undo any residual scale error in one
   split instead of approximating it from twelve columns, so a calibration defect can
   never masquerade as a feature finding. This is the strictly safer instrument and its
   baseline is a run with the stack score as the ONLY feature.

Both report against a matched permuted-feature control, and in B the stack score column is
never permuted -- only the features are -- so the control is a true null for the question
"do the features add anything to the stack".
"""
from __future__ import annotations

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from features import make_frames  # noqa: E402

CHECK = [25, 50, 100, 200, 350, 500, 750, 1000, 1500]
OFFCOL = "_stack_logit"


def calibrate(z, y, folds):
    """Out-of-fold isotonic map z -> P(y=1). Monotone, so within-fold ranking is fixed."""
    p = np.zeros(len(y))
    for itr, iva in folds:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(z[itr], y[itr])
        p[iva] = ir.predict(z[iva])
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def run(X, y, folds, params, rounds, tag, init=None):
    oof = {r: np.zeros(len(y)) for r in CHECK if r <= rounds}
    for k, (itr, iva) in enumerate(folds):
        kw = {} if init is None else {"init_score": init[itr]}
        dtr = lgb.Dataset(X.iloc[itr], label=y[itr], free_raw_data=False, **kw)
        m = lgb.train(params, dtr, num_boost_round=rounds)
        for r in oof:
            oof[r][iva] = m.predict(X.iloc[iva], num_iteration=r, raw_score=True)
        print(f"  [{tag}] fold{k}", flush=True)
    add = 0.0 if init is None else init
    return {r: roc_auc_score(y, add + v) for r, v in oof.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="blend158_h3")
    ap.add_argument("--mode", choices=["offset", "feature"], default="feature")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    z = np.load(os.path.join(SUB, f"oof_{a.stack}.npy"))
    folds = get_folds(y)
    raw_auc = roc_auc_score(y, z)
    off = calibrate(z, y, folds)
    cal_auc = roc_auc_score(y, off)
    print(f"stack {a.stack}  raw OOF AUC {raw_auc:.6f}  after OOF isotonic {cal_auc:.6f}"
          f"  (delta {cal_auc - raw_auc:+.6f}; monotone per fold, so ~0 expected)",
          flush=True)

    Xtr, _, _, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    rng = np.random.default_rng(11)
    Xp = Xtr.apply(lambda c: c.iloc[rng.permutation(len(c))].to_numpy())
    print(f"frame {Xtr.shape}  mode {a.mode}", flush=True)

    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=a.threads,
                  seed=42, deterministic=True, force_row_wise=True)

    if a.mode == "offset":
        base = cal_auc
        real = run(Xtr, y, folds, params, a.rounds, "real", init=off)
        ctrl = run(Xp, y, folds, params, a.rounds, "ctrl", init=off)
    else:
        Xr, Xc = Xtr.copy(), Xp.copy()
        Xr[OFFCOL] = off
        Xc[OFFCOL] = off                      # never permuted: it is the thing we add to
        only = pd.DataFrame({OFFCOL: off})
        b = run(only, y, folds, params, a.rounds, "base")
        base = max(b.values())
        print(f"  baseline (stack score as the only feature) best {base:.6f}"
              f" over {sorted(b)} -> {[f'{b[r]:.6f}' for r in sorted(b)]}", flush=True)
        real = run(Xr, y, folds, params, a.rounds, "real")
        ctrl = run(Xc, y, folds, params, a.rounds, "ctrl")

    print(f"\nbase {base:.6f}")
    print(f"{'round':>7s} {'real':>10s} {'d_real':>10s} {'ctrl':>10s} {'d_ctrl':>10s}"
          f" {'real-ctrl':>10s}")
    for rr in sorted(real):
        print(f"{rr:7d} {real[rr]:10.6f} {real[rr]-base:+10.6f} {ctrl[rr]:10.6f}"
              f" {ctrl[rr]-base:+10.6f} {real[rr]-ctrl[rr]:+10.6f}", flush=True)


if __name__ == "__main__":
    main()
