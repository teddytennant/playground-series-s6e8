"""Screen / evaluate meta-models on the frozen folds.

    python experiments/meta_search.py --folds 0 --variants lin,cat_zr,lgb_zr
    python experiments/meta_search.py --folds all --variants lin,cat_zr --save

Per-fold validation predictions for every variant are written to `meta_preds/`, so
blends of variants can be scored afterwards with no refitting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from meta import inner_linear, meta_frame, regime_frame, to_logit  # noqa: E402
from stack import DEFAULT_DROP, load_members  # noqa: E402

PRED_DIR = os.path.join(ROOT, "meta_preds")
os.makedirs(PRED_DIR, exist_ok=True)

# Checkpoints at which the boosted variants are scored, so one fit yields the whole
# iteration curve instead of one point.
STAGES = [50, 100, 200, 300, 600, 1000, 1500, 2000, 3000]


def fit_cat(Xtr, ytr, Xva, depth, lr, iters, seed=0, l2=6.0):
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(
        iterations=iters, depth=depth, learning_rate=lr, l2_leaf_reg=l2,
        loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8,
        border_count=128, thread_count=16, verbose=False, allow_writing_files=False)
    m.fit(Pool(Xtr, ytr))
    out = {}
    pv = Pool(Xva)
    for s in [x for x in STAGES if x <= iters]:
        out[s] = m.predict_proba(pv, ntree_end=s)[:, 1]
    return out


def fit_lgb(Xtr, ytr, Xva, depth, lr, iters, seed=0):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(
        n_estimators=iters, learning_rate=lr, max_depth=depth,
        num_leaves=2 ** depth - 1, min_child_samples=200, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0,
        random_state=seed, n_jobs=16, verbose=-1)
    m.fit(Xtr, ytr)
    return {s: m.predict_proba(Xva, num_iteration=s)[:, 1]
            for s in STAGES if s <= iters}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0")
    ap.add_argument("--variants", default="lin,cat_zr,lgb_zr")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(y, len(te),
                               extra_dirs=(os.path.join(DATA, "ext_members"),),
                               drop=set(DEFAULT_DROP))
    print(f"{len(names)} members", flush=True)
    Z = to_logit(O)
    R = regime_frame(tr)
    Xz = meta_frame(Z, R, with_regime=False)
    Xzr = meta_frame(Z, R, with_regime=True)
    print(f"meta matrices: Z {Xz.shape}  Z+regime {Xzr.shape}", flush=True)

    folds = get_folds(y)
    which = range(len(folds)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    variants = a.variants.split(",")

    results = []
    for k in which:
        itr, iva = folds[k]
        yv = y[iva]
        for v in variants:
            t0 = time.time()
            preds = {}
            if v == "lin":
                m = LogisticRegression(max_iter=3000, C=a.C).fit(Z[itr], y[itr])
                preds[0] = m.predict_proba(Z[iva])[:, 1]
            elif v in ("cat_z", "lgb_z"):
                f = fit_cat if v.startswith("cat") else fit_lgb
                preds = f(Xz.iloc[itr], y[itr], Xz.iloc[iva], a.depth, a.lr, a.iters)
            elif v in ("cat_zr", "lgb_zr"):
                f = fit_cat if v.startswith("cat") else fit_lgb
                preds = f(Xzr.iloc[itr], y[itr], Xzr.iloc[iva], a.depth, a.lr, a.iters)
            elif v in ("lin_rankraw", "lin_rescale"):
                # REPAIRING THE STACKER'S INPUT TRANSFORM.
                # to_logit clips p into [1e-15, 1-1e-15], so anything at or outside the
                # unit interval collapses onto a single value of +/-30. That is not a
                # corner case here: naji03 and naji05 (the library's two BEST members,
                # AUC 0.9688) emit values from -0.013 to 1.022, and 5.79% of their OOF
                # rows land on the plateau against only 0.92% of their test rows. Six
                # TabM nets and the RF/ET pair have the same asymmetry. So the stacker
                # fits its coefficients against an input whose top tail is a flat
                # constant, then applies them to a test input where that tail is
                # resolved -- a train/test mismatch in the meta-features themselves,
                # concentrated in the strongest members.
                # Both repairs below are monotone per member and use the SAME map on OOF
                # and test, so the two sides land on a common scale by construction.
                if v == "lin_rankraw":
                    # No free parameter. Ties averaged, because rf/et genuinely emit
                    # p == 1.0 for 8-15% of rows and argsort would order those ties
                    # arbitrarily -- injecting noise while pretending to remove it.
                    from scipy.special import ndtri
                    from scipy.stats import rankdata
                    B = np.empty_like(O)
                    Bt = np.empty_like(T)
                    for j in range(O.shape[1]):
                        B[:, j] = ndtri((rankdata(O[:, j]) - 0.5) / len(O))
                        Bt[:, j] = ndtri((rankdata(T[:, j]) - 0.5) / len(T))
                else:
                    # Keeps the logit shape, which the 0.9697 stack demonstrably likes,
                    # and only removes the clipping: rescale each member's raw output
                    # onto (d, 1-d) using the range it actually occupies across BOTH
                    # splits, then take the logit. Nothing is pinned, nothing is lost.
                    d = 1e-4
                    lo = np.minimum(O.min(0), T.min(0))
                    hi = np.maximum(O.max(0), T.max(0))
                    rng = (hi - lo) / (1 - 2 * d)
                    B = to_logit((O - lo) / rng + d)
                    Bt = to_logit((T - lo) / rng + d)
                m = LogisticRegression(max_iter=3000, C=a.C).fit(B[itr], y[itr])
                preds[0] = m.predict_proba(B[iva])[:, 1]
                np.save(os.path.join(PRED_DIR, f"TEST_{v}.npy"),
                        m.predict_proba(Bt)[:, 1])
            elif v in ("lin_poly", "lin_rank"):
                # The other way to relax "one linear map of the logits": keep one
                # coefficient vector for all rows, but let each member enter through a
                # flexible monotone-ish transform. A plain logit stack assumes every
                # member is correctly calibrated in logit space up to a scale; a member
                # that is over-confident in the tails violates that, and no amount of
                # reweighting fixes it. Cubic terms (or a rank-gauss remap) buy exactly
                # that per-member recalibration while staying additive.
                if v == "lin_poly":
                    S = (Z - Z.mean(0)) / Z.std(0)
                    B = np.hstack([S, S ** 2, S ** 3])
                else:
                    from scipy.special import ndtri
                    B = np.empty_like(Z)
                    for j in range(Z.shape[1]):
                        r = np.argsort(np.argsort(Z[:, j]))
                        B[:, j] = ndtri((r + 0.5) / len(r))
                m = LogisticRegression(max_iter=3000, C=a.C).fit(B[itr], y[itr])
                preds[0] = m.predict_proba(B[iva])[:, 1]
            elif v == "lin_regime":
                # The sharpest test of the row-dependent-weighting hypothesis, and the
                # one that stays inside the function class that demonstrably suits this
                # problem: still additive in the logits, but with a separate coefficient
                # vector per missingness bucket. The honest global stack rides along as a
                # feature so a thin bucket can fall back on it instead of inventing
                # weights from 20k rows.
                lt, lv, _ = inner_linear(Z, y, itr, iva, C=a.C)
                b = np.minimum(R["n_missing"].to_numpy(), 4).astype(int)
                At = np.column_stack([Z[itr], lt])
                Av = np.column_stack([Z[iva], lv])
                p = np.zeros(len(iva))
                for bb in range(5):
                    mt, mv = b[itr] == bb, b[iva] == bb
                    m = LogisticRegression(max_iter=3000, C=a.C).fit(At[mt], y[itr][mt])
                    p[mv] = m.predict_proba(Av[mv])[:, 1]
                    print(f"    bucket {bb}: {mt.sum():>7d} train / {mv.sum():>6d} val"
                          f"  AUC {roc_auc_score(yv[mv], p[mv]):.6f}", flush=True)
                preds[0] = p
            elif v in ("cat_resid", "lgb_resid"):
                # correction model: the honest linear stack + regime only. If regime-aware
                # weighting is real, this is where it shows up cleanest.
                lt, lv, _ = inner_linear(Z, y, itr, iva, C=a.C)
                Rt = R.iloc[itr].reset_index(drop=True).assign(lin=lt)
                Rv = R.iloc[iva].reset_index(drop=True).assign(lin=lv)
                f = fit_cat if v.startswith("cat") else fit_lgb
                preds = f(Rt, y[itr], Rv, a.depth, a.lr, a.iters)
            else:
                raise SystemExit(f"unknown variant {v}")

            for s, p in preds.items():
                auc = roc_auc_score(yv, p)
                results.append(dict(fold=k, variant=v, stage=s, auc=auc))
                print(f"  fold{k} {v:>10s} stage {s:>5d}  AUC {auc:.6f}", flush=True)
                if a.save:
                    np.save(os.path.join(PRED_DIR, f"f{k}_{v}_s{s}.npy"), p)
            print(f"  [{v} fold{k} took {time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(results)
    print("\n=== per-fold AUC ===")
    print(df.pivot_table(index=["variant", "stage"], columns="fold",
                         values="auc").to_string(float_format="%.6f"))
    with open(os.path.join(ROOT, "experiments", "meta_results.jsonl"), "a") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
