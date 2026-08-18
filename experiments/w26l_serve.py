"""w26l -- serve-time correction of BOTH train/serve skews in the lattice TE block.

w26k established the first one (pre-registered at experiments/w26_prereg.txt SS G): the raw
cell count `CT_k` is built from an inner 4-fold at fit time and from the whole outer train
part at serve time, so it arrives ~4/3 too large. The fix is a pure rescale and needs no
refit, because rescaling a tree's input feature by s is the same function as rescaling that
feature's thresholds by s.

THE SECOND SKEW, same block, same cause, smaller
------------------------------------------------
`TE_k = (S + lam*gm) / (n + lam)` is shrunk toward the global mean by a smoothing weight
`lam = 20` measured against the cell count `n`. At fit time n is a 3/4-size count, so the
train TE is shrunk MORE than the serve TE of the same cell. That is a systematic level
difference, not just noise, and it is concentrated in thin cells exactly as the algebra
says: measured on cache/f0, sd(valid)/sd(train) runs 1.029 over the 12 thinnest keys down
to 0.993 over the 12 densest.

It is correctable at serve time in closed form, because the count is right there in CT_k:

    S      = TE_serve * (n + lam) - lam * gm          (invert the smoothing)
    TE_fit = (f*S + lam*gm) / (f*n + lam),   f = 3/4  (re-shrink at the fit-time count)

n = 0 maps gm -> gm, so unseen cells are untouched. This replicates the systematic part of
the fit-time shrinkage; the extra sampling NOISE a 3/4 subsample carries is not replicable
and is not attempted.

ARMS (one fit per fold, many predictions -- exactly paired, identical trees)
---------------------------------------------------------------------------
    ct1.0  ct1.1  ct1.3333  ct1.5  ct2.0   CT divided by s, TE untouched
    te                                     TE re-shrunk to f=3/4, CT untouched
    both                                   TE re-shrunk AND CT divided by 4/3

Saves the fitted booster per fold, so any further serve-time transform is free from here.
Checkpoints per fold via a temp file + os.replace: background jobs do not survive a session
boundary in this sandbox and `setsid` is not installed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import CACHE, N_SPLITS, TARGET, get_folds, load_raw  # noqa: E402

CKPT = os.path.join(os.path.dirname(HERE), "cache", "ctckpt")
LAM = 20.0            # te_block's `smooth`, and build_cache.py's TE_SMOOTH default
FRAC = 0.75           # inner StratifiedKFold(4) fit part
PARAMS = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
              min_child_samples=250, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=80.0)
SCALES = [1.0, 1.1, 4.0 / 3.0, 1.5, 2.0]
ARMS = [f"ct{s:.4f}" for s in SCALES] + ["te", "both"]


def te_to_fit_scale(te_serve, n, gm, lam=LAM, f=FRAC):
    """Re-shrink a serve-time TE to the smoothing the model was FITTED under."""
    S = te_serve * (n + lam) - lam * gm
    return (f * S + lam * gm) / (f * n + lam)


def blocks():
    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    te = {c[3:]: i for i, c in enumerate(cols) if c.startswith("TE_")}
    ct = {c[3:]: i for i, c in enumerate(cols) if c.startswith("CT_")}
    keys = [k for k in te if k in ct]
    assert len(keys) == len(te) == len(ct), "TE_/CT_ blocks do not pair up"
    return cols, np.array([te[k] for k in keys]), np.array([ct[k] for k in keys])


def apply_arm(X, ti, ci, arm, gm):
    """Return (te_block, ct_block) originals after writing the arm's values into X."""
    t0, c0 = X[:, ti].copy(), X[:, ci].copy()
    if arm.startswith("ct"):
        X[:, ci] = c0 / float(arm[2:])
    elif arm == "te":
        X[:, ti] = te_to_fit_scale(t0.astype("float64"), c0.astype("float64"),
                                   gm).astype("float32")
    elif arm == "both":
        X[:, ti] = te_to_fit_scale(t0.astype("float64"), c0.astype("float64"),
                                   gm).astype("float32")
        X[:, ci] = c0 / (4.0 / 3.0)
    else:
        raise SystemExit(f"unknown arm {arm}")
    return t0, c0


def save_atomic(path, **arrs):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="serve")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols, ti, ci = blocks()
    print(f"[{a.name}] {len(cols)} cols | {len(ti)} TE_/CT_ pairs | arms {ARMS}", flush=True)
    print(f"[{a.name}] rounds={a.rounds} seed={a.seed} jobs={a.jobs} lam={LAM} f={FRAC}",
          flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    oof = np.zeros((len(y), len(ARMS)))
    tp = np.zeros((len(te), len(ARMS)))
    got = []
    t0 = time.time()

    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        p = os.path.join(CKPT, f"{a.name}_f{f}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(iva), len(ARMS)):
                oof[iva] = z["oof"]; tp += z["test"] / N_SPLITS; got.append(f)
                print(f"[{a.name}] fold {f}: resumed", flush=True)
                continue
            print(f"[{a.name}] fold {f}: STALE checkpoint, ignored", flush=True)

        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        gm = float(ya.mean())
        mp = os.path.join(CKPT, f"{a.name}_f{f}.txt")
        if os.path.exists(mp):
            bst = lgb.Booster(model_file=mp)
            print(f"[{a.name}] fold {f}: booster loaded from disk", flush=True)
        else:
            m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                                   verbose=-1, n_jobs=a.jobs, **PARAMS)
            m.fit(Xa, ya)
            bst = m.booster_
            bst.save_model(mp + ".tmp"); os.replace(mp + ".tmp", mp)
        del Xa

        ob = np.zeros((len(yb), len(ARMS)))
        ot = np.zeros((len(Xt), len(ARMS)))
        for j, arm in enumerate(ARMS):
            tb, cb = apply_arm(Xb, ti, ci, arm, gm)
            tt, ct_ = apply_arm(Xt, ti, ci, arm, gm)
            ob[:, j] = bst.predict(Xb)
            ot[:, j] = bst.predict(Xt)
            Xb[:, ti], Xb[:, ci] = tb, cb
            Xt[:, ti], Xt[:, ci] = tt, ct_

        save_atomic(p, oof=ob, test=ot)
        oof[iva] = ob; tp += ot / N_SPLITS; got.append(f)
        line = "  ".join(f"{arm} {roc_auc_score(yb, ob[:, j]):.6f}"
                         for j, arm in enumerate(ARMS))
        print(f"[{a.name}] fold {f}: {line}  ({time.time()-t0:.0f}s)", flush=True)
        del Xb, Xt

    if len(got) < N_SPLITS:
        print(f"[{a.name}] only folds {sorted(got)} -- re-run to resume", flush=True)
        return

    base = roc_auc_score(y, oof[:, 0])
    print(f"\n[{a.name}] POOLED OOF over {len(y):,} rows")
    print(f"  {'arm':>10s}  {'OOF AUC':>14s}  {'vs ct1.0':>12s}")
    for j, arm in enumerate(ARMS):
        cv = roc_auc_score(y, oof[:, j])
        print(f"  {arm:>10s}  {cv:14.10f}  {(cv-base)*1e6:+9.2f}e-6", flush=True)
    np.save(os.path.join(HERE, f"w26l_{a.name}_oof.npy"), oof)
    np.save(os.path.join(HERE, f"w26l_{a.name}_test.npy"), tp)
    print(f"  saved w26l_{a.name}_{{oof,test}}.npy  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
