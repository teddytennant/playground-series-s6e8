"""Within a generator cell, is the stack's residual recoverable by a specialist?

`w14d_bandmap.py` shows the remaining loss is WITHIN cells (pooled within-cell AUC
0.9390 against 0.9741 cross-cell) and that per-cell isotonic cannot touch it -- monotone
maps do not reorder rows inside a cell. So the only thing that could is a model that
re-ranks inside the cell.

`resid_boost2.py --mode feature` ran that test GLOBALLY and found real - ctrl negative at
every checkpoint. This runs the strongest form of it: the corrector is trained on ONE
cell's rows only, so 100% of its capacity goes to the region in question and it never has
to spend a split reconstructing which region a row is in. If a targeted correction exists
anywhere, it shows up here.

Instruments, per cell:
  base  within-cell AUC of the stack score itself (invariant to any monotone map)
  real  LightGBM on the 40-column numeric frame + the stack score as a feature,
        fitted and scored on cell rows only, on the frozen folds restricted to the cell
  ctrl  identical, with the 40 feature columns row-permuted WITHIN the cell and the
        stack score left intact -- the matched null the workspace requires. It holds
        constant everything about the fit except whether the features line up with y.

real - ctrl is the whole answer. Anything else is a story about variance.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from features import make_frames  # noqa: E402
from w14d_bandmap import cells  # noqa: E402

ZCOL = "__stack__"


def run_cell(X, y, z, idx, folds, params, rounds, rng):
    """Returns (base, {round: real}, {round: ctrl}) on cell rows `idx`."""
    Xc = X.iloc[idx].reset_index(drop=True)
    yc = y[idx]
    zc = z[idx]
    base = roc_auc_score(yc, zc)

    Xreal = Xc.copy()
    Xreal[ZCOL] = zc
    Xperm = Xc.apply(lambda c: c.iloc[rng.permutation(len(c))].to_numpy())
    Xperm[ZCOL] = zc

    remap = np.full(len(y), -1, dtype=np.int64)     # global row -> position in the cell
    remap[idx] = np.arange(len(idx))
    local = []
    for trn, val in folds:
        it = remap[trn]
        iv = remap[val]
        local.append((it[it >= 0], iv[iv >= 0]))

    out = {}
    for tag, Xu in (("real", Xreal), ("ctrl", Xperm)):
        oof = {r: np.zeros(len(yc)) for r in rounds}
        for it, iv in local:
            d = lgb.Dataset(Xu.iloc[it], label=yc[it], free_raw_data=False)
            m = lgb.train(params, d, num_boost_round=max(rounds))
            for r in rounds:
                oof[r][iv] = m.predict(Xu.iloc[iv], num_iteration=r)
        out[tag] = {r: roc_auc_score(yc, oof[r]) for r in rounds}
    return base, out["real"], out["ctrl"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default="blend159av_h3")
    ap.add_argument("--cells", default="BAND,D,G")
    ap.add_argument("--rounds", default="50,150,400")
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()

    rounds = [int(r) for r in a.rounds.split(",")]
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    z = np.load(os.path.join(SUB, f"oof_{a.blend}.npy"))
    g, order = cells(tr)
    folds = get_folds(y)
    X, _, _, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    X = X.iloc[: len(tr)].reset_index(drop=True)
    print(f"{a.blend}: global OOF {roc_auc_score(y, z):.6f}   frame {X.shape}",
          flush=True)

    params = dict(objective="binary", learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=a.threads,
                  seed=42, deterministic=True, force_row_wise=True)

    want = [c for c in order if c.split()[0] in a.cells.split(",")]
    for c in want:
        idx = np.flatnonzero(g == c)
        rng = np.random.default_rng(20260814)
        base, real, ctrl = run_cell(X, y, z, idx, folds, params, rounds, rng)
        print(f"\n{c}   n={len(idx):,}  base(within-cell AUC of stack) {base:.6f}",
              flush=True)
        print(f"  {'round':>6s} {'real':>10s} {'d_real':>10s} {'ctrl':>10s}"
              f" {'d_ctrl':>10s} {'real-ctrl':>10s}")
        for r in rounds:
            print(f"  {r:6d} {real[r]:10.6f} {real[r]-base:+10.6f} {ctrl[r]:10.6f}"
                  f" {ctrl[r]-base:+10.6f} {real[r]-ctrl[r]:+10.6f}", flush=True)


if __name__ == "__main__":
    main()
