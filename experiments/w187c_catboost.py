"""w187, the ANGLE arm: CatBoost on the 0.97-notebook's four-representation feature set.

The handed angle is *"CatBoost: it usually handles categoricals better than the others on
survey-style data. Tune and compare on identical folds."* ANGLE INDEX row 3 closes the
TUNING reading of that (20 handings, +4e-7 for tuning any GBDT, +10.04e-6/member for
enrolling a foreign CatBoost). What row 3 has never had is a CatBoost fitted on an
EXACT-VALUE categorical representation -- the thing the sentence actually claims. Our
pipeline turns every raw column into a target-encoded numeric column and hands CatBoost no
categoricals at all, so "handles categoricals better" has never been tested here.

This arm gives CatBoost the 12 raw columns as native high-cardinality categoricals (its
ordered CTR then builds its own target statistic internally) on top of the same continuous
block the RealMLP arm gets. Same frozen folds, same rows, same in-fold TE.

    --nocat  identical run with the 12 categorical columns dropped. The pair is the
             measurement; a single arm would only re-price the feature set.
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
from common import SEED, TARGET, get_folds, load_raw  # noqa: E402
from w187b_realmlp import FEAT, RAW, build_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--folds", default="0")
    ap.add_argument("--nocat", action="store_true")
    ap.add_argument("--nolabelorig", action="store_true",
                    help="drop the columns built from the ORIGINAL'S LABELS, keeping the "
                         "unlabelled ones. Isolates w185/w186's channel from the new one.")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--stopping", type=int, default=200)
    ap.add_argument("--ctr", type=int, default=2, help="max_ctr_complexity")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-test", action="store_true")
    a = ap.parse_args()

    from catboost import CatBoostClassifier, Pool

    cont_cols = json.load(open(os.path.join(FEAT, "cont_cols.json")))
    Xc = np.load(os.path.join(FEAT, "Xc_train.npy"))
    Xct = np.load(os.path.join(FEAT, "Xc_test.npy"))
    cat_tr = np.load(os.path.join(FEAT, "cat_train.npy"))
    cat_te = np.load(os.path.join(FEAT, "cat_test.npy"))

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = list(range(5)) if a.folds == "all" else [int(x) for x in a.folds.split(",")]
    print(f"[{a.name}] nocat={a.nocat} nolabelorig={a.nolabelorig} folds={want} "
          f"lr={a.lr} depth={a.depth} threads={a.threads}", flush=True)

    oof = np.full(len(y), np.nan)
    tp = np.zeros(len(te))
    aucs, iters = {}, []
    for f in want:
        itr, iva = folds[f]
        if a.rows:
            rng = np.random.default_rng(SEED + f)
            itr = np.sort(rng.choice(itr, size=min(a.rows, len(itr)), replace=False))
        Xa, Xb, Xt = build_fold(Xc, Xct, cat_tr, cat_te, cont_cols, itr, iva, y, SEED + f)
        if a.nolabelorig:
            # `__orig_cdf` and `__orig_q50_distance` read only the original's VALUES, which is
            # the w185/w186 channel. Everything below reads its LABEL column.
            drop = [c for c in Xa.columns
                    if any(k in c for k in ("__orig_cdf_gap", "__orig_q50_distance_y0",
                                            "__orig_q50_distance_y1", "__orig_mean",
                                            "__orig_kde_llr"))]
            assert len(drop) == 21, len(drop)   # 5 gap + 5 q50_y0 + 4 q50_y1 + 4 mean + 3 kde
            Xa, Xb, Xt = (d.drop(columns=drop) for d in (Xa, Xb, Xt))
        if a.nocat:
            Xa, Xb, Xt = (d.drop(columns=RAW) for d in (Xa, Xb, Xt))
            cats = []
        else:
            cats = RAW
            for d in (Xa, Xb, Xt):
                for c in RAW:
                    d[c] = d[c].astype(str)
        t0 = time.time()
        m = CatBoostClassifier(iterations=a.iters, learning_rate=a.lr, depth=a.depth,
                               loss_function="Logloss", eval_metric="AUC",
                               random_seed=SEED + f, l2_leaf_reg=3.0,
                               thread_count=a.threads, verbose=200,
                               max_ctr_complexity=a.ctr,
                               early_stopping_rounds=a.stopping)
        # NOTE: the early stop is on the fold's own validation rows, so the iteration
        # count is chosen with sight of the OOF labels. Both arms carry it identically,
        # which is what makes the PAIR readable; neither arm's absolute OOF is clean.
        m.fit(Pool(Xa, y[itr], cat_features=cats),
              eval_set=Pool(Xb, y[iva], cat_features=cats), use_best_model=True)
        p = m.predict_proba(Xb)[:, 1]
        oof[iva] = p
        aucs[f] = roc_auc_score(y[iva], p)
        iters.append(int(m.get_best_iteration()))
        if not a.no_test:
            tp += m.predict_proba(Pool(Xt, cat_features=cats))[:, 1] / len(want)
        print(f"  fold {f}  AUC {aucs[f]:.8f}  best_iter {iters[-1]}  {time.time()-t0:.0f}s",
              flush=True)
        ck = os.path.join(HERE, "..", "oof_w187")
        os.makedirs(ck, exist_ok=True)
        np.save(os.path.join(ck, f"oof_{a.name}.npy"), oof)
        if not a.no_test:
            np.save(os.path.join(ck, f"test_{a.name}_partial.npy"), tp)
        json.dump({"done": sorted(aucs), "aucs": aucs},
                  open(os.path.join(HERE, f"{a.name}_progress.json"), "w"), indent=1)

    print(f"[{a.name}] mean fold AUC {np.mean(list(aucs.values())):.8f}", flush=True)
    if len(want) == 5:
        print(f"[{a.name}] OOF AUC {roc_auc_score(y, oof):.8f}", flush=True)
    out = os.path.join(HERE, "..", "oof_w187")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"oof_{a.name}.npy"), oof)
    if not a.no_test:
        np.save(os.path.join(out, f"test_{a.name}.npy"), tp)
    json.dump({"name": a.name, "nocat": a.nocat, "folds": want, "aucs": aucs,
               "iters": iters, "lr": a.lr, "depth": a.depth, "ctr": a.ctr,
               "nolabelorig": a.nolabelorig},
              open(os.path.join(HERE, f"{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
