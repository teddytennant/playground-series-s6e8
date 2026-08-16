"""w15f stage 2: rebuild raykkretzschmar's transductive teacher/student residual
INDUCTIVELY, on our own frozen SKF5 seed42 folds, so it can be given a CV number.

WHY THIS RUN EXISTS
-------------------
Group 1 converged on this object from two independent directions on 2026-08-15:

  * w15e measured it against all 168 of our members and found max |rho| 0.0772 -- the only
    sub-0.98-correlation direction anyone here has found all week. Every one of our members
    is INDUCTIVE (fitted on train rows, applied to test rows blind); a signal defined by
    what a smoother model fails to reconstruct ON THE TEST DISTRIBUTION is orthogonal to
    that class by construction. w15e could not score it: the author's artefact exists only
    on the 296,302 test rows, so it has no OOF. Its item 3 asks for exactly this rebuild.
  * w15b's power study showed the closed-list nulls are genuine bounds (78-102% recovery of
    a leader-sized injected signal), so the missing signal is NOT an inductive function of
    the 12 columns -- which points at the transductive class as the only room left. Its
    item 3 names the same rebuild.

THE CONSTRUCTION (author's prose, notebook cell 28; no training code was published)
----------------------------------------------------------------------------------
"In each outer fold, four inner models produce OOF teacher targets for the outer training
rows. The student sees those OOF targets and the unlabeled outer validation rows with their
teacher predictions at weight 0.5. No outer-fold labels enter either fit." The correction is
the teacher-minus-student rank residual, signed-squared, renormalised.

Per outer fold k of the frozen five:
  1. four INNER folds over the outer-training rows -> out-of-fold teacher ranks there;
  2. one teacher on all outer-training rows -> teacher predictions on the held-out rows;
  3. the STUDENT, a deliberately smoother LightGBM regressing percentile-ranked teacher
     output, trained on outer-training rows at weight 1.0 PLUS the held-out rows carried
     with their own teacher predictions at weight 0.5. No label of any kind enters here.
  4. residual = pct(teacher) - pct(student) on the held-out rows.

THREE ARMS, because "is it transductive?" is a separate question from "does it work"
------------------------------------------------------------------------------------
  trans -- the author's student (held-out rows carried at weight 0.5)
  induc -- the identical student WITHOUT the held-out rows. If `trans` and `induc` measure
           the same, the mechanism is just teacher-minus-smooth-student and the word
           "transductive" is decoration. Nobody has separated these.
  (the permuted control lives in stage 3, where it can be run cheaply many times)

NOTHING IS FITTED AGAINST ANY LEADERBOARD. The 0.10 weight is the author's published
constant and is not searched here; see w15e-research.md on why the LB cannot adjudicate a
correction of this size in the first place.

Writes experiments/w15f_nested.npz: per-row teacher/student percentile ranks and fold ids.
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
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "w15f_nested.npz")
NTHREAD = 3
N_INNER = 4
TEACHER_ROUNDS = int(os.environ.get("W15F_TEACHER_ROUNDS", 1500))
STUDENT_ROUNDS = int(os.environ.get("W15F_STUDENT_ROUNDS", 400))
PSEUDO_W = 0.5           # the author's outer-validation carry weight

TEACHER = dict(objective="binary", learning_rate=0.05, num_leaves=96,
               min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.9,
               bagging_freq=1, lambda_l2=5.0, max_bin=511, verbosity=-1,
               num_threads=NTHREAD, seed=42, deterministic=True, force_row_wise=True)

# "a deliberately smoother configuration": few leaves, very large leaves, shallow, and
# coarse binning so it cannot resolve individual lattice values. That coarseness is the
# point -- the residual is defined as what this model CANNOT reconstruct.
STUDENT = dict(objective="regression", learning_rate=0.05, num_leaves=31,
               min_child_samples=2000, feature_fraction=0.8, bagging_fraction=0.9,
               bagging_freq=1, lambda_l2=10.0, max_depth=6, max_bin=63, verbosity=-1,
               num_threads=NTHREAD, seed=42, deterministic=True, force_row_wise=True)


def pct(v):
    """Percentile rank in [0,1], ties averaged -- the author's `pct`."""
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


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
    print(f"{ntr:,} rows, {Xtr.shape[1]} target-free features, "
          f"teacher {TEACHER_ROUNDS} rounds, student {STUDENT_ROUNDS} rounds", flush=True)

    teacher_r = np.zeros(ntr)        # pct(teacher) within each held-out block
    student_t = np.zeros(ntr)        # pct(transductive student) within each block
    student_i = np.zeros(ntr)        # pct(inductive student) within each block
    fold_id = np.zeros(ntr, np.int8)
    teacher_auc = np.zeros(len(folds))

    for k, (itr, iva) in enumerate(folds):
        fold_id[iva] = k
        Xa, ya, Xb = Xtr[itr], y[itr], Xtr[iva]

        # --- 1. inner OOF teacher ranks on the outer-training rows -------------------
        inner_oof = np.zeros(len(itr))
        skf = StratifiedKFold(N_INNER, shuffle=True, random_state=1000 + k)
        for j, (jtr, jva) in enumerate(skf.split(np.zeros(len(itr)), ya)):
            b = lgb.train(TEACHER, lgb.Dataset(Xa[jtr], label=ya[jtr],
                                               feature_name=cols),
                          num_boost_round=TEACHER_ROUNDS)
            inner_oof[jva] = b.predict(Xa[jva])
            print(f"  fold {k} inner {j}: AUC {roc_auc_score(ya[jva], inner_oof[jva]):.6f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

        # --- 2. outer teacher -> predictions on the held-out block -------------------
        bt = lgb.train(TEACHER, lgb.Dataset(Xa, label=ya, feature_name=cols),
                       num_boost_round=TEACHER_ROUNDS)
        p_out = bt.predict(Xb)
        teacher_auc[k] = roc_auc_score(y[iva], p_out)
        teacher_r[iva] = pct(p_out)
        print(f"fold {k}: TEACHER held-out AUC {teacher_auc[k]:.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        # --- 3. the student, on teacher RANKS, no label anywhere ---------------------
        # Percentile ranks are taken within each block separately: the inner-OOF teacher
        # output and the outer teacher output are different estimators on different row
        # sets, so only their ranks are on a common scale. That is what makes the two
        # blocks mixable in one regression target, and it is what the author's `pct` does.
        t_train = pct(inner_oof)
        t_held = teacher_r[iva]

        Xs = np.vstack([Xa, Xb])
        ys = np.concatenate([t_train, t_held])
        w = np.concatenate([np.ones(len(itr)), np.full(len(iva), PSEUDO_W)])
        bs_t = lgb.train(STUDENT, lgb.Dataset(Xs, label=ys, weight=w,
                                              feature_name=cols),
                         num_boost_round=STUDENT_ROUNDS)
        student_t[iva] = pct(bs_t.predict(Xb))

        bs_i = lgb.train(STUDENT, lgb.Dataset(Xa, label=t_train, feature_name=cols),
                         num_boost_round=STUDENT_ROUNDS)
        student_i[iva] = pct(bs_i.predict(Xb))

        rt = teacher_r[iva] - student_t[iva]
        ri = teacher_r[iva] - student_i[iva]
        print(f"fold {k}: student sd(resid) trans {rt.std():.5f} induc {ri.std():.5f}  "
              f"corr(trans,induc) {np.corrcoef(rt, ri)[0,1]:.5f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(OUT, teacher_r=teacher_r, student_t=student_t, student_i=student_i,
             fold_id=fold_id, teacher_auc=teacher_auc,
             teacher_rounds=TEACHER_ROUNDS, student_rounds=STUDENT_ROUNDS,
             pseudo_w=PSEUDO_W)
    print(f"\nteacher held-out AUC per fold: "
          f"{' '.join(f'{a:.6f}' for a in teacher_auc)}  mean {teacher_auc.mean():.6f}")
    print(f"wrote {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
