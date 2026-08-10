"""Randomised-forest members on our own lattice + target-encoding matrices.

WHY A FOREST, WHEN THE STACK ALREADY HOLDS 149 MEMBERS
-----------------------------------------------------
The member-value attribution (RESEARCH.md, slot 3) measured what a new member is worth
by how decorrelated it is from the pack, not by its solo AUC. The single most
decorrelated array anywhere in the 149 is `bolt_extratrees_support` at max correlation
0.811, where the strong GBDT pack sits at 0.987-0.999. And the highest value per member
ever measured here (17.3e-6) came from a *second independent implementation* of a
function class already present (`lookup2`).

So: a second independent implementation of the most decorrelated function class in the
pack, built on OUR feature representation (constrained imputation + full-resolution
lattice target/count encodings + the decimal lattice) rather than theirs. Both halves of
that sentence are sources of decorrelation.

Extremely randomised trees pick split thresholds at random, which is what makes them
disagree with a boosted ensemble: they cannot chase the same residual sequence, and
averaging deep random trees is a variance-reduction estimator rather than a
bias-reduction one. That is a different function, not a differently-tuned one.

Reads the per-fold caches written by experiments/build_cache.py, so no target encoding is
recomputed. Writes oof/oof_<name>.npy + oof/test_<name>.npy in original row order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, N_SPLITS, OOF, TARGET, get_folds, load_raw, save_preds  # noqa: E402
from run_lgbm import lattice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--kind", default="et", choices=["et", "rf"])
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--min-leaf", type=int, default=40)
    ap.add_argument("--max-features", type=float, default=0.15)
    ap.add_argument("--max-depth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--frac", action="store_true", help="append the decimal lattice")
    ap.add_argument("--folds", default="", help="comma-separated subset, for timing runs")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    Ftr = Fte = None
    if a.frac:
        Ftr, Fte = lattice(tr), lattice(te)

    Model = ExtraTreesClassifier if a.kind == "et" else RandomForestClassifier
    kw = dict(n_estimators=a.trees, min_samples_leaf=a.min_leaf,
              max_features=a.max_features, n_jobs=a.jobs, random_state=a.seed,
              bootstrap=(a.kind == "rf"))
    if a.max_depth:
        kw["max_depth"] = a.max_depth
    print(f"[{a.name}] {a.kind} {kw} frac={a.frac} folds={want}", flush=True)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
        if a.frac:
            Xa = np.hstack([Xa, Ftr[itr]])
            Xb = np.hstack([Xb, Ftr[iva]])
        m = Model(**kw).fit(Xa, ya)
        del Xa
        oof[iva] = m.predict_proba(Xb)[:, 1]
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb
        Xt = g("Xt")
        if a.frac:
            Xt = np.hstack([Xt, Fte])
        # chunk the test scoring: 296k x 300 deep trees at once is a needless peak
        pt = np.concatenate([m.predict_proba(Xt[i:i + 60000])[:, 1]
                             for i in range(0, len(Xt), 60000)])
        tp += pt / N_SPLITS
        del Xt, m
        print(f"[{a.name}] fold {f}: test scored ({time.time()-t0:.0f}s)", flush=True)

    if len(want) < N_SPLITS:
        print(f"[{a.name}] partial run, nothing saved", flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    save_preds(a.name, oof, tp, len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), kind=a.kind, **{k: v for k, v in kw.items()
                                                              if k != "n_jobs"},
                   frac=a.frac),
              open(os.path.join(OOF, f"summary_{a.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
