"""w15g -- measure the CV->test gap DIRECTLY, on labelled rows, with the missingness
shift removed by construction.

WHY
---
w15c measured that 88% of this workspace's long-standing +1.0e-3 CV->LB gap is the
train/test missingness-allocation shift, leaving a residual of about +98e-6 that nothing
explains.  w15b proposed a mechanism for it -- "our OOF score partly tracks realised cell
counts, which the test predictions cannot do" -- and flagged that nobody had looked.  That
sentence is the second-ranked live lead out of group 1 and it is what this run was sent to
test.

The OOF-only instruments (w15g_coupling.py, w15g_foldscale.py) can bound the coupling term
but they cannot see the whole gap, because the gap is a statement about a comparison
between two DIFFERENT objects: a 5-fold OOF vector, and a 5-fold-model AVERAGE applied to
rows no model ever saw.  agent/run_lgbm.py:117 makes that asymmetry explicit --

    oof[iva] = pb                 # one model, trained on 80%
    tp += pt / N_SPLITS           # the MEAN of five such models

-- so every member's test column is a five-model bag while its OOF column is a single
model.  That is a third CV-only artefact, it is nobody's hypothesis so far, and it is the
one that is obviously the right ORDER of magnitude: the journal's own seed-averaging
result (three xgb_latcat seeds averaged inside the fold) was worth +138e-6 of solo AUC.

WHAT THIS DOES
--------------
Rebuild the whole competition geometry inside the labelled rows, where the truth is known:

    outer split: 691,369 labelled rows -> TRAIN 70% / HOLD 30%, stratified, random
      -> so the missingness distributions of TRAIN and HOLD are identical by
         construction and w15c's +905e-6 confound is switched off.

    CV arm     : StratifiedKFold(5) inside TRAIN, target encoding fitted inside each
                 fold exactly as agent/features.py:te_block does. OOF AUC on TRAIN.
    single arm : each of those five fold-models predicts HOLD on its own. AUC on HOLD.
    bagged arm : the mean of the five predictions on HOLD, i.e. what agent/run_lgbm.py
                 actually ships. AUC on HOLD.
    full arm   : one model refitted on ALL of TRAIN, predicting HOLD.

    gap_total   = bagged - CV        <- the analogue of LB - CV, confound removed
    gap_bagging = bagged - mean(single)
    gap_size    = full  - mean(single)      (100% vs 80% of TRAIN)
    gap_honesty = mean(single) - CV         <- is the OOF an honest estimate of what ONE
                                               fold model does on unseen rows?  This is
                                               where coupling and inter-fold scale live,
                                               and under the lead it should be positive.

Every arm is scored on the SAME 207k HOLD rows, so the three arms are paired and only the
CV arm sits on different rows.  Repeated over independent outer splits for uncertainty.

Usage:  python experiments/w15g_cvgap.py [n_outer_splits] [n_estimators]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, load_raw  # noqa: E402
from features import make_frames, te_block  # noqa: E402

import lightgbm as lgb  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "w15g_cvgap.json")
NTHREAD = 3
HOLD_FRAC = 0.30

# A representative TE-carrying LightGBM. Not tuned -- tuning is on the closed list and the
# quantity being measured is a GAP between two ways of scoring the same model, which does
# not need the model to be the best one.
PARAMS = dict(objective="binary", learning_rate=0.06, num_leaves=63, max_bin=511,
              min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=NTHREAD)


def one_split(X, K, y, seed, n_est):
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y > 0.5)
    idx_neg = np.flatnonzero(y < 0.5)
    hold = []
    for idx in (idx_pos, idx_neg):
        idx = idx.copy()
        rng.shuffle(idx)
        hold.append(idx[:int(round(HOLD_FRAC * len(idx)))])
    hold = np.sort(np.concatenate(hold))
    mask = np.zeros(len(y), bool)
    mask[hold] = True
    train = np.flatnonzero(~mask)
    Xh, Kh, yh = X.iloc[hold], K.iloc[hold], y[hold]
    Xt, Kt, yt = X.iloc[train], K.iloc[train], y[train]
    print(f"  outer seed {seed}: TRAIN {len(train)}  HOLD {len(hold)}", flush=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    hold_preds = []
    for f, (ia, ib) in enumerate(skf.split(np.zeros(len(yt)), yt)):
        t0 = time.time()
        TEa, TEb, TEh = te_block(Kt.iloc[ia], yt[ia], Kt.iloc[ib], Kh)
        Fa = np.hstack([Xt.iloc[ia].to_numpy("float32"), TEa.to_numpy("float32")])
        Fb = np.hstack([Xt.iloc[ib].to_numpy("float32"), TEb.to_numpy("float32")])
        Fh = np.hstack([Xh.to_numpy("float32"), TEh.to_numpy("float32")])
        m = lgb.LGBMClassifier(n_estimators=n_est, random_state=42, **PARAMS)
        m.fit(Fa, yt[ia])
        oof[ib] = m.predict_proba(Fb)[:, 1]
        hp = m.predict_proba(Fh)[:, 1]
        hold_preds.append(hp)
        print(f"    fold {f}: oof {roc_auc_score(yt[ib], oof[ib]):.6f}  "
              f"hold {roc_auc_score(yh, hp):.6f}  ({time.time() - t0:.0f}s)", flush=True)
        del Fa, Fb, Fh

    # the full-data arm: one model on ALL of TRAIN
    t0 = time.time()
    TEa, _, TEh = te_block(Kt, yt, Kt.iloc[:1], Kh)
    Fa = np.hstack([Xt.to_numpy("float32"), TEa.to_numpy("float32")])
    Fh = np.hstack([Xh.to_numpy("float32"), TEh.to_numpy("float32")])
    m = lgb.LGBMClassifier(n_estimators=n_est, random_state=42, **PARAMS)
    m.fit(Fa, yt)
    hold_full = roc_auc_score(yh, m.predict_proba(Fh)[:, 1])
    print(f"    full-TRAIN model: hold {hold_full:.6f}  ({time.time() - t0:.0f}s)", flush=True)
    del Fa, Fh

    cv = roc_auc_score(yt, oof)
    singles = [roc_auc_score(yh, p) for p in hold_preds]
    bagged = roc_auc_score(yh, np.mean(hold_preds, axis=0))
    # rank-average bag as well: our stack consumes ranks, not probabilities
    from scipy.stats import rankdata
    bagged_rank = roc_auc_score(yh, np.mean([rankdata(p) for p in hold_preds], axis=0))
    r = dict(seed=int(seed), n_train=int(len(train)), n_hold=int(len(hold)),
             cv=cv, singles=singles, single_mean=float(np.mean(singles)),
             bagged=bagged, bagged_rank=bagged_rank, full=hold_full,
             gap_total=bagged - cv, gap_bagging=bagged - float(np.mean(singles)),
             gap_size=hold_full - float(np.mean(singles)),
             gap_honesty=float(np.mean(singles)) - cv)
    print(f"  -> CV {cv:.6f} | single mean {np.mean(singles):.6f} | bagged {bagged:.6f} "
          f"| full {hold_full:.6f}")
    print(f"     gap_total {r['gap_total'] * 1e6:+.1f}e-6   bagging {r['gap_bagging'] * 1e6:+.1f}e-6"
          f"   honesty {r['gap_honesty'] * 1e6:+.1f}e-6   size {r['gap_size'] * 1e6:+.1f}e-6",
          flush=True)
    return r


def main() -> None:
    nsplit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    n_est = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    tr, te = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    # keys only over the singles + the four hand-picked pairs: the mechanism under test is
    # a property of how the pipeline is SCORED, not of how wide its key set is, and the
    # 36-pair version costs 8x the encoding time for the same comparison.
    X, _, K, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    print(f"design {X.shape}, {K.shape[1]} lattice keys, {n_est} rounds, "
          f"{nsplit} outer splits", flush=True)

    rows = [one_split(X, K, y, 1000 + 7 * i, n_est) for i in range(nsplit)]
    arr = lambda k: np.array([r[k] for r in rows])
    print("\n" + "=" * 78)
    for k in ("gap_total", "gap_bagging", "gap_honesty", "gap_size"):
        v = arr(k)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"{k:12s} {v.mean() * 1e6:+9.1f}e-6   se {se * 1e6:6.1f}e-6   "
              f"[{', '.join(f'{x * 1e6:+.1f}' for x in v)}]")
    with open(OUT, "w") as fh:
        json.dump(dict(n_est=n_est, splits=rows), fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
