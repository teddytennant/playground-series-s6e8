"""LightGBM hyperparameter search on the cached lattice design matrices.

WHY THIS IS WORTH DOING
-----------------------
The public library's LightGBM members on the lattice feature set (`lat_*`, `latwide_*`,
`lattri_*`, `latmax_*`, best OOF 0.96768) all use ONE hand-set parameter vector:

    lr 0.035, num_leaves 96, min_child_samples 40, subsample 0.9,
    colsample_bytree 0.6, reg_lambda 5.0, max_depth 7

The only tuned LightGBM in that library (`lgbm_tuned`) was tuned on the much weaker
raw+iterative-imputation feature set and landed in a totally different regime
(max_depth 4, num_leaves 18). So LightGBM has never actually been tuned against these
features. That is the gap this script attacks.

PROTOCOL
--------
Trials are scored on ONE fold (default fold 0). Comparing configs on the *same* fold is
a paired comparison, so the fold-specific variance largely cancels and the ranking is
usable even though the absolute number is not. Finalists are then re-run on all 5 folds
by `agent/run_lgbm.py`, and only that full-OOF number is ever quoted or compared against
the library.

Results append to experiments/tune_results.jsonl so a killed run loses nothing.
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tune_results.jsonl")

# The library's hand-set vector -- our control. Everything is measured against this.
BASE = dict(learning_rate=0.035, num_leaves=96, min_child_samples=40, subsample=0.9,
            subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, max_depth=7)


def load_fold(f):
    g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
    return g("Xa"), g("ya"), g("Xb"), g("yb")


def run_trial(name, params, Xa, ya, Xb, yb, n_estimators=6000, stopping=200, seed=42):
    t0 = time.time()
    p = dict(n_estimators=n_estimators, random_state=seed, verbose=-1, n_jobs=-1,
             **params)
    m = lgb.LGBMClassifier(**p)
    m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc",
          callbacks=[lgb.early_stopping(stopping, verbose=False)])
    auc = roc_auc_score(yb, m.predict_proba(Xb)[:, 1])
    rec = dict(name=name, auc=float(auc), best_iter=int(m.best_iteration_),
               secs=round(time.time() - t0, 1), params=params, seed=seed)
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"  {name:<28s} AUC {auc:.5f}  iters {m.best_iteration_:>5d}  "
          f"({rec['secs']:.0f}s)", flush=True)
    return auc


def space(stage):
    """Trial list. Stage A explores structure; stage B refines the winner."""
    T = []
    add = lambda n, **kw: T.append((n, {**BASE, **kw}))

    if stage == "a":
        add("control")
        # --- capacity: 184 features, most of them target-derived and correlated ---
        for nl, md in [(31, 6), (63, 7), (128, 8), (192, 9), (255, -1)]:
            add(f"leaves{nl}_depth{md}", num_leaves=nl, max_depth=md)
        # --- leaf-size regularisation: the main defence against TE overfitting ---
        for mcs in [15, 100, 250, 600]:
            add(f"mcs{mcs}", min_child_samples=mcs)
        # --- column sampling: many near-duplicate TE/CT columns ---
        for cs in [0.25, 0.4, 0.8, 1.0]:
            add(f"colsample{cs}", colsample_bytree=cs)
        # --- L1/L2 ---
        for l2 in [0.5, 20.0, 80.0]:
            add(f"l2_{l2}", reg_lambda=l2)
        for l1 in [1.0, 10.0]:
            add(f"l1_{l1}", reg_alpha=l1)
        # --- knobs the library never touched ---
        for ps in [0.5, 5.0, 50.0]:
            add(f"path_smooth{ps}", path_smooth=ps)
        for ffn in [0.5, 0.8]:
            add(f"ff_bynode{ffn}", feature_fraction_bynode=ffn)
        add("extra_trees", extra_trees=True)
        add("max_bin511", max_bin=511)
        add("max_bin127", max_bin=127)
        for ss in [0.6, 1.0]:
            add(f"subsample{ss}", subsample=ss)

    if stage == "b":
        # Stage A moved one knob at a time off the control and every winner pointed the
        # same way: LESS capacity, MORE regularisation. That is what 144 of 184 features
        # being target-derived does -- they are easy to overfit and cheap to split on.
        #
        # Stage A winners vs control 0.96638:
        #   reg_lambda 80         +0.00033      max_bin 511          +0.00031
        #   num_leaves 63/depth 7 +0.00030      feature_frac_bynode  +0.00025
        #   min_child_samples 250 +0.00020      subsample 1.0        +0.00018
        # Stage A losers: extra_trees -0.00037, colsample 1.0 -0.00022, max_bin 127.
        #
        # These are NOT independent -- capacity, leaf size and L2 all regularise the same
        # overfitting, so stacking them can overshoot into underfitting. Measure, do not
        # assume additivity.
        W = dict(num_leaves=63, max_depth=7, max_bin=511)
        add("B_l2_80", **W, min_child_samples=100, reg_lambda=80.0)
        add("B_mcs250_l2_80", **W, min_child_samples=250, reg_lambda=80.0)
        # reg_lambda was still improving at the edge of the stage-A grid (80 > 20 > 5 >
        # 0.5), so push past it rather than stopping where the grid happened to end.
        add("B_l2_200", **W, min_child_samples=100, reg_lambda=200.0)
        add("B_l2_500", **W, min_child_samples=100, reg_lambda=500.0)
        add("B_l2_80_ffn", **W, min_child_samples=100, reg_lambda=80.0,
            feature_fraction_bynode=0.8)
        add("B_l2_80_ffn_nosub", **W, min_child_samples=100, reg_lambda=80.0,
            feature_fraction_bynode=0.8, subsample=1.0)
        add("B_mcs250_l2_200_ffn", **W, min_child_samples=250, reg_lambda=200.0,
            feature_fraction_bynode=0.8)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--stage", default="a")
    ap.add_argument("--lr", type=float, default=None,
                    help="override learning rate for all trials (sweep faster at 0.05)")
    ap.add_argument("--only", default=None, help="comma-separated trial names to run")
    a = ap.parse_args()

    Xa, ya, Xb, yb = load_fold(a.fold)
    print(f"fold {a.fold}: Xa{Xa.shape} Xb{Xb.shape}  pos rate {ya.mean():.4f}",
          flush=True)

    trials = space(a.stage)
    if a.only:
        want = set(a.only.split(","))
        trials = [t for t in trials if t[0] in want]
    if a.lr is not None:
        trials = [(n, {**p, "learning_rate": a.lr}) for n, p in trials]

    print(f"{len(trials)} trials, lr={a.lr or 'per-trial'}\n", flush=True)
    t0 = time.time()
    for n, p in trials:
        run_trial(f"{n}@f{a.fold}", p, Xa, ya, Xb, yb)
    print(f"\nstage {a.stage} done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
