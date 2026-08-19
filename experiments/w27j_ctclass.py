"""w27j -- does the CT train/serve skew correction survive a change of FUNCTION CLASS?

Pre-registered at experiments/w27_prereg_slot3.txt SS M. Read that first; the
predictions there were written before any number in this file existed.

THE QUESTION
------------
Every measurement of the CT_ 4/3 rescale on file -- w26k, w26l, w27g, fourteen
configs' worth -- is LightGBM. The mechanism w26 SS G claims is a property of the
FEATURE MATRIX: `CT_k` is built from an inner 4-fold at fit time and from the whole
outer train part at serve time, so it arrives ~4/3 too large no matter who consumes
it. If that is right, a CatBoost and an XGBoost fitted on the same cache/f0_Xa.npy
must show the same sign. If it is wrong, the +325e-6 is a LightGBM artefact and the
33-member lattice rebuild that SS 9 of the w27 slot-2 entry leaves open is not worth
starting.

WHY THIS IS ONE FIT PER CLASS AND NOT FIVE
------------------------------------------
The correction is a serve-time transform and needs no refit: for a threshold tree,
dividing an input feature by s is the same function as multiplying that feature's
thresholds by s. So every arm is a re-PREDICTION of one fitted model. d_c therefore
carries NO fit noise -- it is exact for that model -- and the only noise left is the
138k-row valid slice, which is COMMON to both arms of the pair and cancels. That is
why fold 0 alone is enough for a sign, and why the LightGBM control is quoted from
w27g rather than refitted.

`apply_arm` and `te_to_fit_scale` are IMPORTED from w26l_serve, not reimplemented,
so the arms here are byte-identical to the ones that produced the LightGBM numbers
this is compared against. Reimplementing them would make the comparison worthless.

HONESTY
-------
No early stopping anywhere -- fixed rounds, so the iteration count never sees a
held-out label and the two classes are not silently tuned to different effective
capacities. Nothing here produces an OOF vector and nothing enters the pack; these
are 400-round fold-0 probes, deliberately matched to w27g's budget.

Checkpointed per class: background jobs do not survive a session boundary here, and
a re-run of the identical command resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import CACHE, TARGET, get_folds, load_raw  # noqa: E402
from w26l_serve import apply_arm, blocks  # noqa: E402  -- the SAME arms as the LGB runs

CKPT = os.path.join(os.path.dirname(HERE), "cache", "ctclassckpt")
ARMS = ["ct1.0000", "ct1.3333", "te", "both"]

# w27g's LightGBM control, fold 0, 400 rounds -- quoted, not refitted.
LGB_CONTROL = dict(auc10=0.964779, auc43=0.965104, d_c=324.99e-6, ctshare=0.010)


def fit_catboost(Xa, ya, rounds, seed, threads):
    """CatBoost on the 184 dense lattice columns. No cat_features: this matrix is
    already the workspace's fold-safe smoothed-mean TE, so handing the columns over
    as numerics is what makes it the SAME pipeline the LightGBM control ran on --
    which is the whole point of the comparison. `cat_native`'s ordered target
    statistic is a different experiment and is already a member."""
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC",
                           iterations=rounds, learning_rate=0.06, depth=6,
                           l2_leaf_reg=6.0, border_count=254,
                           bootstrap_type="Bernoulli", subsample=0.85,
                           random_seed=seed, thread_count=threads,
                           allow_writing_files=False, verbose=False)
    m.fit(Xa, ya)
    imp = np.asarray(m.get_feature_importance(), dtype="float64")
    return (lambda X: m.predict_proba(X)[:, 1]), imp


def fit_xgboost(Xa, ya, rounds, seed, threads):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
             max_depth=7, eta=0.03, subsample=0.9, colsample_bytree=0.5,
             min_child_weight=60.0, reg_lambda=40.0, max_bin=512,
             nthread=threads, seed=seed)
    bst = xgb.train(p, xgb.DMatrix(Xa, label=ya), num_boost_round=rounds)
    sc = bst.get_score(importance_type="total_gain")
    imp = np.zeros(Xa.shape[1])
    for k, v in sc.items():
        imp[int(k[1:])] = v          # xgboost names columns f0..f183 for a numpy input
    return (lambda X: bst.predict(xgb.DMatrix(X))), imp


def fit_lightgbm(Xa, ya, rounds, seed, threads):
    """The control, refitted only if --with-lgb is passed. w27g already has this
    number; refitting it is a harness check, not a result."""
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=rounds, random_state=seed, verbose=-1,
                           n_jobs=threads, learning_rate=0.025, num_leaves=63,
                           max_depth=7, max_bin=511, min_child_samples=250,
                           subsample=0.9, subsample_freq=1, colsample_bytree=0.6,
                           reg_lambda=80.0)
    m.fit(Xa, ya)
    bst = m.booster_
    imp = np.asarray(bst.feature_importance(importance_type="gain"), dtype="float64")
    return bst.predict, imp


FITTERS = {"cat": fit_catboost, "xgb": fit_xgboost, "lgb": fit_lightgbm}
IMPNAME = {"cat": "PredictionValuesChange", "xgb": "total_gain", "lgb": "split gain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ctclass")
    ap.add_argument("--classes", default="cat,xgb")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols, ti, ci = blocks()
    want = [c.strip() for c in a.classes.split(",") if c.strip()]
    for c in want:
        if c not in FITTERS:
            raise SystemExit(f"unknown class {c}; pick from {sorted(FITTERS)}")
    print(f"[{a.name}] {len(cols)} cols | {len(ti)} TE_/CT_ pairs | arms {ARMS}", flush=True)
    print(f"[{a.name}] classes {want} fold={a.fold} rounds={a.rounds} "
          f"seed={a.seed} threads={a.threads}", flush=True)
    print(f"[{a.name}] LGB control (w27g, quoted): ct1.0 {LGB_CONTROL['auc10']:.6f}  "
          f"4/3 {LGB_CONTROL['auc43']:.6f}  d_c {LGB_CONTROL['d_c']*1e6:+.2f}e-6  "
          f"CTshare {LGB_CONTROL['ctshare']*100:.1f}%", flush=True)

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    itr, iva = get_folds(y)[a.fold]
    g = lambda k: np.load(os.path.join(CACHE, f"f{a.fold}_{k}.npy"))
    Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
    gm = float(ya.mean())

    # GATE. The whole experiment presumes the skew is actually in this matrix; if the
    # valid/train CT_ ratio is not ~4/3 the cache is not what SS G says it is and no
    # number below means anything. Same gate w27g runs, same expected value.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = float(np.nanmedian(Xb[:, ci].mean(0) / np.maximum(Xa[:, ci].mean(0), 1e-9)))
    print(f"[{a.name}] GATE  median valid/train CT_ ratio = {ratio:.4f} (expected 1.3333)",
          flush=True)
    if not 1.25 <= ratio <= 1.42:
        raise SystemExit(f"[{a.name}] GATE FAILED at {ratio:.4f} -- stopping")

    rows, t0 = [], time.time()
    for cls in want:
        p = os.path.join(CKPT, f"{a.name}_{cls}_f{a.fold}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
            print(f"[{a.name}] {cls}: resumed from checkpoint", flush=True)
            continue

        tf = time.time()
        predict, imp = FITTERS[cls](Xa, ya, a.rounds, a.seed, a.threads)
        fit_s = time.time() - tf
        print(f"[{a.name}] {cls}: fitted in {fit_s:.0f}s, scoring {len(ARMS)} arms", flush=True)
        ctshare = float(imp[ci].sum() / max(imp.sum(), 1e-12))

        # ⚠ Each arm gets its OWN COPY of the valid matrix rather than mutate-and-restore.
        # The mutate-and-restore version crashed with "assignment destination is read-only":
        # CatBoost clears the writeable flag on an array it has predicted from, so the restore
        # failed AFTER the fit had already cost its full runtime and before any checkpoint
        # existed. A 138,274 x 184 float32 copy is ~100 MB and is the cheapest possible
        # insurance against that whole class of bug -- w26l's in-place version is safe only
        # because LightGBM leaves its input alone, which is a property of LightGBM, not of the
        # arms. Correctness first here: these fits are expensive and unattended.
        aucs = {}
        for arm in ARMS:
            Xw = np.array(Xb, copy=True)
            apply_arm(Xw, ti, ci, arm, gm)
            aucs[arm] = float(roc_auc_score(yb, predict(Xw)))
            del Xw

        rec = dict(cls=cls, fold=a.fold, rounds=a.rounds, seed=a.seed,
                   ctshare=ctshare, imp_kind=IMPNAME[cls], fit_s=fit_s, **aucs)
        rec["d_c"] = aucs["ct1.3333"] - aucs["ct1.0000"]
        rec["d_te"] = aucs["te"] - aucs["ct1.0000"]
        rec["d_both"] = aucs["both"] - aucs["ct1.0000"]
        tmp = p + ".tmp"
        json.dump(rec, open(tmp, "w"))
        os.replace(tmp, p)
        rows.append(rec)
        print(f"[{a.name}] {cls:>4s}  1.0 {aucs['ct1.0000']:.6f}  4/3 {aucs['ct1.3333']:.6f}"
              f"  d_c {rec['d_c']*1e6:+8.2f}e-6   te {rec['d_te']*1e6:+8.2f}e-6"
              f"  both {rec['d_both']*1e6:+8.2f}e-6   CTshare {ctshare*100:5.2f}%"
              f"  ({fit_s:.0f}s fit, {time.time()-t0:.0f}s total)", flush=True)
        del predict

    print(f"\n[{a.name}] === SS M3 SCORECARD, fold {a.fold}, {a.rounds} rounds ===")
    print(f"  {'class':>6s} {'ct1.0':>10s} {'ct4/3':>10s} {'d_c':>12s} "
          f"{'d_both':>12s} {'CTshare':>9s} {'M3c pred':>12s}")
    print(f"  {'lgb*':>6s} {LGB_CONTROL['auc10']:10.6f} {LGB_CONTROL['auc43']:10.6f} "
          f"{LGB_CONTROL['d_c']*1e6:+11.2f}e-6 {'--':>12s} "
          f"{LGB_CONTROL['ctshare']*100:8.2f}% {'--':>12s}")
    for r in rows:
        pred = 131e-6 + 208e-6 * (r["ctshare"] * 100.0)     # SS M3(c), 2-point line
        print(f"  {r['cls']:>6s} {r['ct1.0000']:10.6f} {r['ct1.3333']:10.6f} "
              f"{r['d_c']*1e6:+11.2f}e-6 {r['d_both']*1e6:+11.2f}e-6 "
              f"{r['ctshare']*100:8.2f}% {pred*1e6:+11.2f}e-6")
    print("  * lgb quoted from w27g fold 0, not refitted. CTshare is NOT commensurable "
          "across libraries -- see SS M3(c).")

    sign = all(r["d_c"] > 0 for r in rows)
    print(f"\n[{a.name}] SS M3(a) PRIMARY: d_c > 0 in every tested class -> "
          f"{'HELD' if sign else 'FAILED'}  ({len(rows)} classes)")
    print(f"[{a.name}] SS M3(d): both >= ct4/3 -> "
          + ", ".join(f"{r['cls']} {'yes' if r['d_both'] >= r['d_c'] else 'no'}" for r in rows))
    out = os.path.join(HERE, f"w27j_{a.name}.json")
    json.dump(rows, open(out, "w"), indent=1)
    print(f"[{a.name}] wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
