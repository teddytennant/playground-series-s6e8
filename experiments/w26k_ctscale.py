"""w26k -- the CT_ train/serve scale skew in the lattice target-encoding block.

WHAT IS BEING TESTED (pre-registered in full at experiments/w26_prereg.txt SS G)
-------------------------------------------------------------------------------
`te_block` builds the train part of every lattice target encoding with an inner 4-fold
loop and the valid/test part from the whole outer training part. For the smoothed mean
`TE_k` that is right and scale-free. For the raw cell count `CT_k` it is not: a count over
3/4 of the rows is on a different SCALE from a count over 4/4 of them, so every one of the
72 CT_ columns arrives at serve time ~4/3 larger than it was at fit time. Measured on the
cache with no model at all: median ratio valid/train 1.3325 against a predicted 1.3333.

THE INSTRUMENT, AND WHY IT NEEDS NO REFIT
-----------------------------------------
Rescaling one input feature of a tree model by s is the same function as rescaling that
feature's thresholds by s. So "fit with CT multiplied by s" == "the original model applied
to serve rows whose CT is divided by s". One fit per fold therefore yields the whole
s-curve, and every arm shares the identical trees, rows and folds. Nothing separates the
arms except the serve-time CT scale.

CHECKPOINTS
-----------
Background jobs do not survive a session boundary here and `setsid` is not installed, so
each fold is written to cache/ctckpt/<name>_f<k>.npz via a temp file + os.replace the
moment it finishes. Re-running the identical command resumes.

    .venv/bin/python experiments/w26k_ctscale.py --name ctscale --folds 0 --rounds 100
    .venv/bin/python experiments/w26k_ctscale.py --name ctscale
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

# lgbm_fixed_lat, exactly as it sits in oof/summary_lgbm_fixed_lat.json (CV 0.9677108350)
PARAMS = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
              min_child_samples=250, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=80.0)
SCALES = [1.0, 1.1, 4.0 / 3.0, 1.5, 2.0]


def ct_index():
    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    return cols, np.array([i for i, c in enumerate(cols) if c.startswith("CT_")])


def save_atomic(path, **arrs):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:          # np.savez appends .npz to a *name*, not to a handle
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ctscale")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--folds", default=None, help="comma-separated subset, for probes")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols, ct = ct_index()
    print(f"[{a.name}] {len(cols)} cols, {len(ct)} CT_ cols, scales {SCALES}", flush=True)
    print(f"[{a.name}] rounds={a.rounds} seed={a.seed} jobs={a.jobs}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = ([int(x) for x in a.folds.split(",")] if a.folds
            else list(range(N_SPLITS)))

    oof = np.zeros((len(y), len(SCALES)))
    tp = np.zeros((len(te), len(SCALES)))
    got, imps = [], []
    t0 = time.time()

    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        p = os.path.join(CKPT, f"{a.name}_f{f}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(iva), len(SCALES)):
                oof[iva] = z["oof"]
                tp += z["test"] / N_SPLITS
                imps.append(z["imp"])
                got.append(f)
                print(f"[{a.name}] fold {f}: resumed from checkpoint", flush=True)
                continue
            print(f"[{a.name}] fold {f}: STALE checkpoint, ignored", flush=True)

        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                               verbose=-1, n_jobs=a.jobs, **PARAMS)
        m.fit(Xa, ya)
        del Xa
        # importance share carried by the CT_ block -- diagnostic only
        gain = m.booster_.feature_importance("gain")
        imp = np.array([gain[ct].sum() / max(gain.sum(), 1e-9),
                        gain.sum()])
        imps.append(imp)

        ob = np.zeros((len(yb), len(SCALES)))
        ot = np.zeros((len(Xt), len(SCALES)))
        b0, t0c = Xb[:, ct].copy(), Xt[:, ct].copy()
        for j, s in enumerate(SCALES):
            Xb[:, ct] = b0 / s
            Xt[:, ct] = t0c / s
            ob[:, j] = m.predict_proba(Xb)[:, 1]
            ot[:, j] = m.predict_proba(Xt)[:, 1]
        Xb[:, ct], Xt[:, ct] = b0, t0c

        save_atomic(p, oof=ob, test=ot, imp=imp)
        oof[iva] = ob
        tp += ot / N_SPLITS
        got.append(f)
        aucs = "  ".join(f"s={s:.4f} {roc_auc_score(yb, ob[:, j]):.6f}"
                         for j, s in enumerate(SCALES))
        print(f"[{a.name}] fold {f}: {aucs}   CTgain {imp[0]*100:.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb, Xt, m

    if len(got) < N_SPLITS:
        print(f"[{a.name}] only folds {sorted(got)} done -- nothing pooled, "
              f"re-run the same command to resume", flush=True)
        return

    print(f"\n[{a.name}] POOLED OOF, {len(y):,} rows", flush=True)
    base = roc_auc_score(y, oof[:, 0])
    print(f"  {'scale':>8s}  {'OOF AUC':>14s}  {'vs s=1.0':>12s}")
    for j, s in enumerate(SCALES):
        cv = roc_auc_score(y, oof[:, j])
        print(f"  {s:8.4f}  {cv:14.10f}  {(cv-base)*1e6:+9.2f}e-6", flush=True)
    print(f"\n  stored lgbm_fixed_lat CV = 0.9677108350  "
          f"| s=1.0 here = {base:.10f}  diff {(base-0.9677108350221775)*1e6:+.2f}e-6")
    print(f"  CT_ share of total gain, per fold: "
          f"{[f'{i[0]*100:.1f}%' for i in imps]}")
    np.save(os.path.join(HERE, f"w26k_{a.name}_oof.npy"), oof)
    np.save(os.path.join(HERE, f"w26k_{a.name}_test.npy"), tp)
    print(f"  saved w26k_{a.name}_{{oof,test}}.npy  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
