"""w15f stage 11: the test-side correction under the students that actually match his scale.

Stage 10 is the run's turning point. My pre-registered student (`smooth`, residual sd 0.0479)
was the WORST of the three, and it was worst for a mechanical reason rather than a mysterious
one: an over-smooth student inflates the residual, the additive move at a fixed weight 0.10
therefore lands twice as large as the author's, and the toll on a strict ranking goes with the
square of that size (-5.5e-5 here against w15e's -1.1e-5 for a perturbation 2.14x smaller).
So `smooth`'s null at w=0.10 was mostly toll, not absence of signal.

The selection criterion for which student to carry forward is NOT the AUC. It is the author's
own published, target-free fidelity statistic: his residual sd is 0.0217, and stage 7 was
written to bracket it before any of these AUCs existed. `rough` (0.0199) and `mid` (0.0311)
bracket it; `smooth` (0.0479) misses it by more than a factor of two.

Stage 12 then checks that criterion against something completely independent of any AUC --
whether the resulting correction agrees better with HIS published test-space correction. If
it does, the choice is confirmed by an instrument that cannot see the label at all.

Reuses `w15f_teacher_test.npy` (the teacher fitted on all 691,369 labelled rows) so this
costs one student fit rather than another teacher.
"""
from __future__ import annotations

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402
from w15f_nested import pct  # noqa: E402
from w15f_nested2 import STUDENTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    t0 = time.time()
    X = np.load(os.path.join(HERE, "w15f_X.npy"))
    cols = json.load(open(os.path.join(HERE, "w15f_cols.json")))
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ntr, nte = len(y), len(te)
    Xtr, Xte = X[:ntr], X[ntr:]
    del X

    teacher_r = np.load(os.path.join(HERE, "w15f_nested.npz"))["teacher_r"]
    p_test = np.load(os.path.join(HERE, "w15f_teacher_test.npy"))
    t_test = pct(p_test)

    folds = get_folds(y)
    ratio = 0.5 * len(folds[0][1]) / len(folds[0][0])
    w_pseudo = ratio * ntr / nte
    print(f"test carry weight {w_pseudo:.7f}", flush=True)

    Xs = np.vstack([Xtr, Xte])
    ys = np.concatenate([teacher_r, t_test])
    w = np.concatenate([np.ones(ntr), np.full(nte, w_pseudo)])

    for nm in ("mid", "rough"):
        params, rounds = STUDENTS[nm]
        bs = lgb.train(params, lgb.Dataset(Xs, label=ys, weight=w, feature_name=cols),
                       num_boost_round=rounds)
        s_test = pct(bs.predict(Xte))
        r = t_test - s_test
        sq = np.sign(r) * np.abs(r) ** 2
        c = (sq - sq.mean()) * (r.std() / sq.std())
        np.save(os.path.join(HERE, f"w15f_c_test_{nm}.npy"), c)
        print(f"{nm:>6}: residual sd {r.std():.5f}  (his 0.02171)  "
              f"correction sd {c.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
