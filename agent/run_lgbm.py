"""Full 5-fold LightGBM on the cached lattice matrices; saves OOF + test predictions.

This is the finalist runner. `experiments/tune_lgbm.py` ranks configs on one fold
(a paired comparison, cheap); only the numbers this script prints -- full out-of-fold
AUC on the frozen folds -- are ever quoted or compared against the public library.

THE DECIMAL LATTICE (--frac)
----------------------------
Optional extra channel the library's `lat_*` models do NOT have. The generator wrote the
numbers with a fingerprint: the first decimal digit of `daily_screen_time_hours` swings
the addiction rate by 8.5 points across 50-68k rows per digit. Nothing about behaviour
explains that.

Target encoding cannot see it. TE estimates every exact value independently, so it has no
way to express "everything ending in .2 shares something" -- that statement pools across
integer parts, and TE's levels do not. It is a different channel, not a different view of
one already present, which is exactly the distinction that predicts whether an idea adds
anything once the representation is saturated. Measured at +0.0001 elsewhere.

These columns are row-wise functions of the raw data and fold-independent, so they are
appended to the cached matrices rather than forcing a target-encoding rebuild.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE, N_SPLITS, TARGET, get_folds, load_raw, save_preds  # noqa: E402

FRAC_COLS = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
             "work_study_hours", "sleep_hours", "weekend_screen_time"]

PRESETS = {
    # the public library's hand-set vector, our control
    "control": dict(learning_rate=0.035, num_leaves=96, min_child_samples=40,
                    subsample=0.9, subsample_freq=1, colsample_bytree=0.6,
                    reg_lambda=5.0, max_depth=7),
}


def lattice(df):
    """Sub-unit position and first decimal digit for the rounded columns."""
    o = {}
    for c in FRAC_COLS:
        v = df[c].to_numpy("float64")
        o[f"frac_{c}"] = v - np.floor(v)
        o[f"d1_{c}"] = np.floor(v * 10) % 10
    return pd.DataFrame(o).to_numpy("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--preset", default=None, help="a key of PRESETS")
    ap.add_argument("--params", default=None, help="inline JSON param dict")
    ap.add_argument("--frac", action="store_true", help="append the decimal lattice")
    ap.add_argument("--seeds", default="42", help="comma-separated; averaged in-fold")
    ap.add_argument("--n-estimators", type=int, default=8000)
    ap.add_argument("--stopping", type=int, default=200)
    a = ap.parse_args()

    params = dict(PRESETS[a.preset]) if a.preset else {}
    if a.params:
        params.update(json.loads(a.params))
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[{a.name}] params={params}\n  seeds={seeds} frac={a.frac}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)

    Ftr = Fte = None
    if a.frac:
        Ftr, Fte = lattice(tr), lattice(te)
        print(f"  decimal lattice: +{Ftr.shape[1]} cols", flush=True)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    iters = []

    for f, (itr, iva) in enumerate(folds):
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        if a.frac:
            Xa = np.hstack([Xa, Ftr[itr]])
            Xb = np.hstack([Xb, Ftr[iva]])
            Xt = np.hstack([Xt, Fte])

        pb = np.zeros(len(yb))
        pt = np.zeros(len(Xt))
        for sd in seeds:
            reg = str(params.get("objective", "")).startswith(("regression", "l2", "mae"))
            Est = lgb.LGBMRegressor if reg else lgb.LGBMClassifier
            m = Est(n_estimators=a.n_estimators, random_state=sd,
                    verbose=-1, n_jobs=-1, **params)
            if a.stopping:
                # NOTE: this stops on (Xb, yb), the very rows whose predictions become
                # this member's OOF. The iteration count is then chosen with sight of the
                # held-out labels, which is exactly the optimism the journal drops
                # golem_a/golem_f for -- keep it only for tuning, never for a member that
                # will be stacked. --stopping 0 is the honest path.
                m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
                      callbacks=[lgb.early_stopping(a.stopping, verbose=False)])
            else:
                m.fit(Xa, ya)
            p = (m.predict(Xb) if reg else m.predict_proba(Xb)[:, 1])
            pb += p / len(seeds)
            pt += (m.predict(Xt) if reg else m.predict_proba(Xt)[:, 1]) / len(seeds)
            iters.append(m.best_iteration_ or a.n_estimators)
        oof[iva] = pb
        tp += pt / N_SPLITS
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, pb):.5f}  "
              f"iters {iters[-len(seeds):]}  ({time.time()-t0:.0f}s)", flush=True)
        del Xa, Xb, Xt

    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  library's best LightGBM (lattri_lgbm/latmax_lgbm) = 0.96768", flush=True)

    # per-missingness breakdown: where a config wins matters for blending
    nm = tr[[c for c in tr.columns if c not in ("id", TARGET)]].isna().sum(axis=1).values
    for k in range(5):
        s = nm == k
        if s.sum() > 300:
            print(f"    n_missing={k}: AUC {roc_auc_score(y[s], oof[s]):.5f} "
                  f"(n={s.sum():,})", flush=True)
    s = nm >= 5
    if s.sum() > 300:
        print(f"    n_missing>=5: AUC {roc_auc_score(y[s], oof[s]):.5f} (n={s.sum():,})",
              flush=True)

    save_preds(a.name, oof, tp, len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), params=params, seeds=seeds,
                   frac=a.frac, iters=iters),
              open(os.path.join(os.path.dirname(CACHE), "oof", f"summary_{a.name}.json"),
                   "w"), indent=2)


if __name__ == "__main__":
    main()
