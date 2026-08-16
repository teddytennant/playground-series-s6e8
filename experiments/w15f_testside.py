"""w15f stage 4: the test-side twin of the correction measured in stage 3.

Same teacher, same student, same smoothing, same signed-square renormalisation -- the only
change is that the "unlabeled block the student carries" is now the 296,302 real test rows
instead of a held-out fold. That is the whole point of the object: it is transductive, so
the artefact that goes on a submission has to be built against the actual test rows.

Teacher out-of-fold ranks on the 691,369 train rows are taken from stage 2's `teacher_r`,
which IS a frozen-fold OOF teacher, so no refitting is needed there.

PSEUDO-LABEL MASS. The author carries his outer-validation block at weight 0.5 and states
that his final test artefact uses 0.2450247 "to preserve the same pseudo-label mass ratio".
His ratio is 0.5 * 120,000 / 571,369 = 0.1050, which reproduces his constant exactly
(0.2450247 * 296,302 / 691,369 = 0.1050). Our fold geometry is not his, so copying his
number would NOT reproduce the object we measured. We preserve OUR OWN measured ratio,
0.5 * 138,274 / 553,095 = 0.125, giving 0.125 * 691,369 / 296,302 = 0.29165. Same rule,
our geometry. Nothing here is fitted and nothing is chosen against a leaderboard.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402
from w15f_nested import STUDENT, TEACHER, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEACHER_ROUNDS = 1500
STUDENT_ROUNDS = 400


def main():
    t0 = time.time()
    X = np.load(os.path.join(HERE, "w15f_X.npy"))
    cols = json.load(open(os.path.join(HERE, "w15f_cols.json")))
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ntr, nte = len(y), len(te)
    Xtr, Xte = X[:ntr], X[ntr:]
    del X

    z = np.load(os.path.join(HERE, "w15f_nested.npz"))
    teacher_r = z["teacher_r"]           # frozen-fold OOF teacher, percentile per fold

    folds = get_folds(y)
    ratio = 0.5 * len(folds[0][1]) / len(folds[0][0])
    w_pseudo = ratio * ntr / nte
    print(f"train {ntr:,} test {nte:,}  pseudo mass ratio {ratio:.5f} "
          f"-> test carry weight {w_pseudo:.7f}", flush=True)

    # --- teacher on ALL labelled rows -> the test block --------------------------
    bt = lgb.train(TEACHER, lgb.Dataset(Xtr, label=y, feature_name=cols),
                   num_boost_round=TEACHER_ROUNDS)
    p_test = bt.predict(Xte)
    t_test = pct(p_test)
    print(f"teacher fitted on all train, test preds done ({time.time()-t0:.0f}s)",
          flush=True)

    # --- the transductive student ------------------------------------------------
    Xs = np.vstack([Xtr, Xte])
    ys = np.concatenate([teacher_r, t_test])
    w = np.concatenate([np.ones(ntr), np.full(nte, w_pseudo)])
    bs = lgb.train(STUDENT, lgb.Dataset(Xs, label=ys, weight=w, feature_name=cols),
                   num_boost_round=STUDENT_ROUNDS)
    s_test = pct(bs.predict(Xte))
    print(f"student fitted ({time.time()-t0:.0f}s)", flush=True)

    r = t_test - s_test
    sq = np.sign(r) * np.abs(r) ** 2
    c = (sq - sq.mean()) * (r.std() / sq.std())
    np.save(os.path.join(HERE, "w15f_c_test.npy"), c)
    np.save(os.path.join(HERE, "w15f_teacher_test.npy"), p_test)
    print(f"residual sd {r.std():.5f}  correction sd {c.std():.5f} "
          f"[{c.min():+.4f}, {c.max():+.4f}]")
    print(f"wrote w15f_c_test.npy  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
