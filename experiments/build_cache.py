"""Precompute the per-fold design matrices once so tuning is cheap.

The target encoding is fold-dependent (each outer fold's maps are fit only on that
fold's training part), so it has to be rebuilt per fold -- but it does NOT depend on any
model hyperparameter. Computing it once and caching it turns every subsequent
hyperparameter trial into pure LightGBM time. That is what makes a real sweep affordable
on 16 CPU cores.

Writes, per fold f:  cache/f{f}_{Xa,ya,Xb,yb,Xt}.npy   plus cache/cols.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, TARGET, get_folds, load_raw  # noqa: E402
from features import make_frames, te_block  # noqa: E402

SMOOTH = float(os.environ.get("TE_SMOOTH", 20.0))


def main():
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int)
    print(f"loaded train {tr.shape} test {te.shape} ({time.time()-t0:.0f}s)", flush=True)

    Xtr, Xte, Ktr, Kte = make_frames(tr, te)
    print(f"base {Xtr.shape[1]} feats | {Ktr.shape[1]} lattice keys "
          f"-> +{2*Ktr.shape[1]} TE/CT cols  ({time.time()-t0:.0f}s)", flush=True)

    folds = get_folds(y)
    for f, (itr, iva) in enumerate(folds):
        p = os.path.join(CACHE, f"f{f}_Xa.npy")
        if os.path.exists(p):
            print(f"fold {f}: cached, skip", flush=True)
            continue
        ya = y.iloc[itr]
        t_tr, t_va, t_te = te_block(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte, SMOOTH)
        Xa = pd.concat([Xtr.iloc[itr].reset_index(drop=True),
                        t_tr.reset_index(drop=True)], axis=1)
        Xb = pd.concat([Xtr.iloc[iva].reset_index(drop=True),
                        t_va.reset_index(drop=True)], axis=1)
        Xt = pd.concat([Xte.reset_index(drop=True), t_te.reset_index(drop=True)], axis=1)
        if f == 0:
            json.dump(list(Xa.columns), open(os.path.join(CACHE, "cols.json"), "w"))
        np.save(os.path.join(CACHE, f"f{f}_Xb.npy"), Xb.to_numpy("float32"))
        np.save(os.path.join(CACHE, f"f{f}_yb.npy"), y.iloc[iva].to_numpy("int8"))
        np.save(os.path.join(CACHE, f"f{f}_Xt.npy"), Xt.to_numpy("float32"))
        np.save(os.path.join(CACHE, f"f{f}_ya.npy"), ya.to_numpy("int8"))
        np.save(p, Xa.to_numpy("float32"))
        print(f"fold {f}: Xa{Xa.shape} Xb{Xb.shape} Xt{Xt.shape} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
