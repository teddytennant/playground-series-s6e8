"""w15f stage 7: is the answer robust to the ONE hyperparameter I had to guess?

`w15f_fidelity.py` reports the honest weakness of stage 2. The author published no training
code, only prose, so the teacher and the student are reconstructions. The teacher landed
almost exactly on his -- rank corr **+0.99529** against his published `test_teacher`, which is
about as close as two independently-configured GBDTs get. The STUDENT did not: my residual
has sd 0.0464 against his 0.0217, i.e. my student is roughly twice as smooth as his, and the
two corrections share only rank corr +0.517.

"Smoother configuration" is all he says. So the correct response is not to guess again, and
certainly not to tune my student until it agrees with him -- it is to sweep the student's
smoothness across a range that BRACKETS his residual sd and report whether the conclusion
moves. If three very different students give the same verdict, the verdict does not depend on
the thing I could not read.

Also saves `inner_oof`, which stage 2 computed and threw away. It is the expensive part
(20 teacher fits) and the reason this stage costs an hour instead of five minutes. Any
future run wanting a different student can now have one for the price of the student alone.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402
from w15f_nested import N_INNER, STUDENT, TEACHER, TEACHER_ROUNDS, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "w15f_nested2.npz")
PSEUDO_W = 0.5

# A smoothness ladder. `smooth` is stage 2's student, repeated here as a determinism check;
# `rough` sits close to the teacher's own capacity, so its residual should collapse. His
# sd 0.0217 should fall inside the bracket.
STUDENTS = {
    "smooth": (dict(STUDENT), 400),
    "mid": ({**STUDENT, "num_leaves": 63, "min_child_samples": 500, "max_depth": 8,
             "max_bin": 127, "lambda_l2": 5.0}, 700),
    "rough": ({**STUDENT, "num_leaves": 255, "min_child_samples": 100, "max_depth": -1,
               "max_bin": 511, "lambda_l2": 1.0}, 1200),
}


def main():
    t0 = time.time()
    X = np.load(os.path.join(HERE, "w15f_X.npy"))
    cols = json.load(open(os.path.join(HERE, "w15f_cols.json")))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ntr = len(y)
    Xtr = X[:ntr]
    del X
    folds = get_folds(y)

    # The teachers are deterministic (fixed seed, `deterministic=True`, same rows), and the
    # fold-0 rerun above reproduced stage 2's held-out AUC to every printed digit, so the
    # OUTER teacher never needs refitting: stage 2's `teacher_r` already is pct(p_out) per
    # fold, which is all the student target and the residual use. Only the INNER teachers
    # have to be recomputed, and those are cached to disk on first pass.
    prev = np.load(os.path.join(HERE, "w15f_nested.npz"))
    teacher_r, fold_id = prev["teacher_r"], prev["fold_id"]
    students = {k: np.zeros(ntr) for k in STUDENTS}

    for k, (itr, iva) in enumerate(folds):
        Xa, ya, Xb = Xtr[itr], y[itr], Xtr[iva]

        cache = os.path.join(HERE, f"w15f_inner_oof_f{k}.npy")
        if os.path.exists(cache):
            inner_oof = np.load(cache)
            assert inner_oof.shape == (len(itr),)
            print(f"fold {k}: inner OOF cached ({time.time()-t0:.0f}s)", flush=True)
        else:
            inner_oof = np.zeros(len(itr))
            skf = StratifiedKFold(N_INNER, shuffle=True, random_state=1000 + k)
            for j, (jtr, jva) in enumerate(skf.split(np.zeros(len(itr)), ya)):
                b = lgb.train(TEACHER, lgb.Dataset(Xa[jtr], label=ya[jtr],
                                                   feature_name=cols),
                              num_boost_round=TEACHER_ROUNDS)
                inner_oof[jva] = b.predict(Xa[jva])
            np.save(cache, inner_oof)
            print(f"fold {k}: inner OOF AUC {roc_auc_score(ya, inner_oof):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        t_train = pct(inner_oof)
        t_held = teacher_r[iva]
        Xs = np.vstack([Xa, Xb])
        ys = np.concatenate([t_train, t_held])
        w = np.concatenate([np.ones(len(itr)), np.full(len(iva), PSEUDO_W)])
        for nm, (params, rounds) in STUDENTS.items():
            # a fresh Dataset per student: max_bin is baked into the binned handle, so
            # sharing one Dataset across the smoothness ladder is not allowed
            bs = lgb.train(params, lgb.Dataset(Xs, label=ys, weight=w,
                                               feature_name=cols),
                           num_boost_round=rounds)
            students[nm][iva] = pct(bs.predict(Xb))
            r = t_held - students[nm][iva]
            print(f"  fold {k} student {nm:>6}: sd(resid) {r.std():.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(OUT, teacher_r=teacher_r, fold_id=fold_id,
             **{f"student_{k}": v for k, v in students.items()})
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
