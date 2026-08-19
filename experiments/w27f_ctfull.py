"""w27f -- the CLEAN fix for the CT_ skew: build the train-side count from the SAME map
the serve side uses, instead of rescaling it afterwards.

WHY THIS IS THE RIGHT FIX AND THE 4/3 RESCALE IS ONLY AN APPROXIMATION TO IT
-----------------------------------------------------------------------------
`agent/features.py:te_block` builds the train part of every encoding with an inner
StratifiedKFold(4) so a row never enters its own encoding, and the valid/test part from the
whole outer training part. For the smoothed mean `TE_k` the inner loop is REQUIRED: a target
mean that saw the row's own label leaks it.

**A cell COUNT carries no label information at all.** `CT_k` is how many training rows share
the row's lattice cell; nothing about y enters it. So the inner loop buys nothing for CT_ and
costs two things:

  1. SCALE. A count over 3/4 of the rows is 4/3 smaller than a count over 4/4 of them, so the
     model learns its CT_ thresholds on one scale and is served another. Measured on the cache
     with no model at all: median valid/train ratio 1.3325 against a predicted 1.3333.
  2. NOISE. A 3/4 subsample of a count carries sqrt(4/3) = 1.155x the sd of the full one. The
     serve-time rescale of w26k/w26l matches the MEAN and cannot touch this.

Building CT_ from the full outer-train map for the train rows too fixes both at once, and it
is the construction the library should have had. `ct1.3333` corrects (1) only; this corrects
(1) exactly and (2) as well.

⚠ THE SELF-COUNT, stated because it is the one thing that is NOT exactly matched. A train row
is in the map it is looking itself up in; a valid or test row is not. So `full` for a train row
counts one more row than `full` for a serve row in the same cell -- an off-by-one that matters
only in cells of size ~1-5. Both arms are run so the question is answered rather than assumed:

    ctfull      CT_train = full_map[cell]        (self included; scale and noise both matched)
    ctfull_m1   CT_train = full_map[cell] - 1    (the exact "other training rows" quantity)

⚠ Unlike every arm in w26k/w26l this CANNOT come off the saved boosters. Changing a feature's
values changes which splits get chosen, so each arm needs its own fit. That is why there are
two arms here and not six.

READ AGAINST (w26l, pooled, 5 folds, 400 rounds, same PARAMS, same seed, same folds):
    ct1.0000  0.9654813306      the status quo
    ct1.3333  0.9657751945     +293.86e-6   the serve-time rescale

Per-fold checkpointed; re-running the identical command resumes.

    .venv/bin/python experiments/w27f_ctfull.py --name ctfull --rounds 400 --jobs 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import CACHE, N_SPLITS, TARGET, get_folds, load_raw  # noqa: E402
from features import make_frames  # noqa: E402
sys.path.insert(0, HERE)
from w26k_ctscale import PARAMS, save_atomic  # noqa: E402

CKPT = os.path.join(os.path.dirname(HERE), "cache", "ctckpt")
ARMS = ["ctfull", "ctfull_m1"]
W26L = {"ct1.0000": 0.9654813306, "ct1.3333": 0.9657751945}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ctfull")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    ct = [(i, c[3:]) for i, c in enumerate(cols) if c.startswith("CT_")]
    ct_i = np.array([i for i, _ in ct])
    ct_key = [k for _, k in ct]
    print(f"[{a.name}] {len(cols)} cols, {len(ct_i)} CT_ cols, arms {ARMS}", flush=True)
    print(f"[{a.name}] rounds={a.rounds} seed={a.seed} jobs={a.jobs}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(pd.Series(y))
    want = ([int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS)))
    todo = [f for f in want
            if not os.path.exists(os.path.join(CKPT, f"{a.name}_f{f}.npz"))]

    Ktr = None
    if todo:                      # the lattice keys cost minutes; skip them on a pure resume
        t0 = time.time()
        _, _, Ktr, _ = make_frames(tr, te)
        missing = [k for k in ct_key if k not in Ktr.columns]
        if missing:
            raise SystemExit(f"{len(missing)} CT_ keys absent from make_frames: {missing[:5]}")
        print(f"[{a.name}] lattice keys rebuilt {Ktr.shape} ({time.time()-t0:.0f}s)", flush=True)

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
                oof[iva] = z["oof"]
                tp += z["test"] / N_SPLITS
                got.append(f)
                print(f"[{a.name}] fold {f}: resumed from checkpoint", flush=True)
                continue
            print(f"[{a.name}] fold {f}: STALE checkpoint, ignored", flush=True)

        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")

        # the full outer-train map, i.e. exactly the map te_block already uses for Xb and Xt
        Ka = Ktr.iloc[itr]
        full = np.empty((len(itr), len(ct_i)), dtype="float32")
        for j, k in enumerate(ct_key):
            col = Ka[k]
            full[:, j] = col.map(col.value_counts()).to_numpy("float32")
        # gate: the status-quo train column must be ~3/4 of this, or the mapping is wrong
        r = (Xa[:, ct_i].mean(0) / np.maximum(full.mean(0), 1e-9))
        print(f"[{a.name}] fold {f}: cached CT / full-map CT  median {np.median(r):.4f} "
              f"(expect ~0.75)  min {r.min():.4f} max {r.max():.4f}", flush=True)
        if not (0.70 < np.median(r) < 0.80):
            raise SystemExit("ratio is not ~3/4 -- the key mapping does not line up, STOP")

        ob = np.zeros((len(yb), len(ARMS)))
        ot = np.zeros((len(Xt), len(ARMS)))
        for j, arm in enumerate(ARMS):
            Xa[:, ct_i] = full - (1.0 if arm.endswith("_m1") else 0.0)
            m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                                   verbose=-1, n_jobs=a.jobs, **PARAMS)
            m.fit(Xa, ya)
            ob[:, j] = m.predict_proba(Xb)[:, 1]
            ot[:, j] = m.predict_proba(Xt)[:, 1]
            print(f"[{a.name}] fold {f} {arm}: {roc_auc_score(yb, ob[:, j]):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del m

        save_atomic(p, oof=ob, test=ot)
        oof[iva] = ob
        tp += ot / N_SPLITS
        got.append(f)
        del Xa, Xb, Xt, full

    if len(got) < N_SPLITS:
        print(f"[{a.name}] only folds {sorted(got)} done -- nothing pooled, "
              f"re-run the same command to resume", flush=True)
        return

    print(f"\n[{a.name}] POOLED OOF over {len(y):,} rows", flush=True)
    for j, arm in enumerate(ARMS):
        cv = roc_auc_score(y, oof[:, j])
        print(f"  {arm:10s} {cv:.10f}   vs w26l ct1.0 {(cv-W26L['ct1.0000'])*1e6:+9.2f}e-6"
              f"   vs w26l ct1.3333 {(cv-W26L['ct1.3333'])*1e6:+9.2f}e-6", flush=True)
    np.save(os.path.join(HERE, f"w27f_{a.name}_oof.npy"), oof)
    np.save(os.path.join(HERE, f"w27f_{a.name}_test.npy"), tp)
    print(f"  saved w27f_{a.name}_{{oof,test}}.npy  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
