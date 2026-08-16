"""w15g -- does K-fold OOF coupling distort our CV criterion?

THE LEAD (w15b next-run item 5b, the second-ranked live lead out of group 1)
---------------------------------------------------------------------------
w15b_lackfit.py / w15b_tecause.py measured, on the frozen folds and against
size-matched permuted controls landing at exactly 1.000, that the pack's residuals
inside real lattice cells are UNDER-dispersed (T/df 0.54) while raw-feature members
are OVER-dispersed (4.8-7.1).  w15b attributed that to full-resolution target
encoding: a row's OOF score is built on the other four folds, which contain the
other rows of its own lattice cell, so E_c tracks the realised O_c.  Then it wrote:

    "The same table implies our OOF score partly tracks realised cell counts, which
     the test predictions cannot do.  That is a candidate mechanism for part of the
     CV->LB offset and nobody has looked at it from this direction."

The T/df ladder IS measured on the frozen folds and survives an evidence check.
The sentence above is NOT measured -- it is an inference, and it is the part with a
decision attached, because every deadline pick in this workspace is made on pooled
OOF AUC.  This script measures it.

THE INSTRUMENT
--------------
Pooled OOF AUC decomposes EXACTLY over the fold-pair grid.  With P_k / N_l the
positives of fold k and negatives of fold l,

    AUC_pooled = sum_{k,l} U(P_k, N_l) / (Npos * Nneg)

Split that sum in two:

    A_within = sum_k    U(P_k,N_k) / sum_k    |P_k||N_l|      (k == l)
    A_cross  = sum_{k!=l} U(P_k,N_l) / sum_{k!=l} |P_k||N_l|

Why this is exactly the right cut for the lead:

  * WITHIN-fold pair (i,j both in fold k): both rows are scored by the SAME model,
    and that model saw neither label.  That is precisely the train->test
    configuration -- one encoder, evaluated rows outside it.
  * CROSS-fold pair (i in k, j in l): s_i's model was trained on folds != k, which
    CONTAINS y_j; s_j's model was trained on folds != l, which contains y_i.  The
    two scores are coupled through each other's labels.  Nothing like this happens
    at test time.

So `A_cross - A_within` is the coupling term, measured on the frozen folds, with no
model refits and no assumption about the mechanism.  Sign prediction from the lead:
for a discordant cross-fold pair (y_i=1, y_j=0), seeing y_i=1 pushes s_j UP and
seeing y_j=0 pushes s_i DOWN, so coupling should make cross-fold pairs HARDER:
A_cross < A_within, i.e. pooled OOF AUC is PESSIMISTIC relative to test.  That is
the same sign as the observed CV->LB gap, which is why it is worth measuring rather
than dismissing.

CONTROL
-------
The contrast is not zero-mean by construction, so a null is mandatory.  The control
is a stratified re-partition of the SAME rows into five same-sized pseudo-folds:
identical score vector, identical label vector, identical fold sizes, only the
row->fold correspondence destroyed.  Under the control no coupling can exist, so the
spread of `A_cross - A_within` over control draws IS the resolution of the
instrument.

Dose-response: w15b's own member ladder gives a free graded exposure.  TE members
concentrate the cross-fold dependence into small lattice cells; raw members spread
it over a GBDT.  If coupling is real, TE members must show a larger |contrast| than
raw ones.  `orig_binm` is a bonus zero-dose arm: it is trained on the 7,500-row
ORIGINAL, so its OOF has no dependence on competition labels at all and its contrast
must sit exactly on the control.

Usage:
    python experiments/w15g_coupling.py [n_control_draws]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402

NFOLD = 5
SEED = 20260815
OUT = os.path.join(ROOT, "experiments", "w15g_coupling.json")

LIB = os.path.join(ROOT, "data", "oof", "oof")

# w15b_tecause.py's ladder plus a CatBoost dose ladder that controls for the learning
# algorithm, plus zero-dose anchors whose OOF cannot depend on the partition at all.
#
#   cat_raw     12 raw numeric columns, no categoricals -> CatBoost builds NO target
#               statistics.  Zero label-dependent feature construction.
#   cat_native  all 12 columns as unordered lattice categoricals -> ordered target
#               statistics on every column at full lattice resolution.
#   cat_lat     our own full-resolution TE pipeline, then CatBoost on top.
#
# Same family, same implementation, three doses.  `orig_*` are trained on the 7,500-row
# ORIGINAL only, so their OOF is one model applied to all rows: the partition never
# entered them and their contrast is a pure partition draw.
MEMBERS = [
    ("PACK blend159av_h3", os.path.join(SUB, "oof_blend159av_h3.npy"), "stack", 3),
    ("blend158_h3", os.path.join(SUB, "oof_blend158_h3.npy"), "stack", 3),
    ("blend158_logit", os.path.join(SUB, "oof_blend158_logit.npy"), "stack", 3),
    ("cat_lat", os.path.join(OOF, "oof_cat_lat.npy"), "cat + our TE", 3),
    ("cat_native", os.path.join(OOF, "oof_cat_native.npy"), "cat, native lattice cats", 2),
    ("cat_raw", os.path.join(OOF, "oof_cat_raw.npy"), "cat, raw numerics only", 0),
    ("xgb_latcat", os.path.join(OOF, "oof_xgb_latcat.npy"), "TE", 3),
    ("xgb_lat", os.path.join(OOF, "oof_xgb_lat.npy"), "TE", 3),
    ("xgb_cat_lattice", os.path.join(OOF, "oof_xgb_cat_lattice.npy"), "unordered cats, no TE", 0),
    ("xgb_raw_nan", os.path.join(OOF, "oof_xgb_raw_nan.npy"), "raw", 0),
    ("lgbm_tuned_lat", os.path.join(OOF, "oof_lgbm_tuned_lat.npy"), "TE", 3),
    ("lgbm_fixed_lat", os.path.join(OOF, "oof_lgbm_fixed_lat.npy"), "TE", 3),
    ("lgbm_stump_lat_frac", os.path.join(OOF, "oof_lgbm_stump_lat_frac.npy"), "TE, stumps", 3),
    ("et_lat_frac", os.path.join(OOF, "oof_et_lat_frac.npy"), "TE", 3),
    ("linlat", os.path.join(OOF, "oof_linlat.npy"), "TE, linear", 3),
    ("lib lat_cat", os.path.join(LIB, "oof_lat_cat.npy"), "lib TE", 3),
    ("lib lat_lgbm", os.path.join(LIB, "oof_lat_lgbm.npy"), "lib TE", 3),
    ("lib lat_xgb", os.path.join(LIB, "oof_lat_xgb.npy"), "lib TE", 3),
    ("lib lookup", os.path.join(LIB, "oof_lookup.npy"), "lib exact lookup", 3),
    ("lib cat", os.path.join(LIB, "oof_cat.npy"), "lib cat", 1),
    ("lib lgbm", os.path.join(LIB, "oof_lgbm.npy"), "lib plain", 0),
    ("lib knn", os.path.join(LIB, "oof_knn.npy"), "lib knn", 0),
    ("lib logreg", os.path.join(LIB, "oof_logreg.npy"), "lib linear", 0),
    ("orig_binm", os.path.join(OOF, "oof_orig_binm.npy"), "ZERO DOSE (orig-trained)", -1),
    ("orig_bin", os.path.join(OOF, "oof_orig_bin.npy"), "ZERO DOSE (orig-trained)", -1),
    ("w15d_origrep_r", os.path.join(OOF, "oof_w15d_origrep_r.npy"), "ZERO DOSE (orig-trained)", -1),
]


class FoldPairAUC:
    """Exact U(P_k, N_l) for every fold pair, from one sort of the score vector.

    Walking the distinct score values in ascending order, maintain the number of
    negatives of each fold strictly below the current value.  A positive of fold k
    at value v beats those, and ties with the negatives of fold l sitting at v:

        U[k,l] += npos_k(v) * ( cum_neg_l(<v) + 0.5 * nneg_l(v) )

    which is the tie-aware Mann-Whitney statistic, summed over the value grid.
    Vectorised with reduceat over the runs of equal scores, so re-doing it for a
    different fold assignment costs O(n) and not another sort.
    """

    def __init__(self, score: np.ndarray, y: np.ndarray) -> None:
        order = np.argsort(score, kind="mergesort")
        s = score[order]
        self.pos = y[order] > 0.5
        self.order = order
        # start index of each run of equal score values
        new = np.empty(len(s), dtype=bool)
        new[0] = True
        np.not_equal(s[1:], s[:-1], out=new[1:])
        self.starts = np.flatnonzero(new)
        self.nval = len(self.starts)

    def u_matrix(self, fold_of_row: np.ndarray, nfold: int = NFOLD) -> np.ndarray:
        f = fold_of_row[self.order]
        # one-hot (n, nfold) split by class, summed within each run of equal scores
        onehot_p = np.zeros((len(f), nfold))
        onehot_n = np.zeros((len(f), nfold))
        idx = np.arange(len(f))
        onehot_p[idx[self.pos], f[self.pos]] = 1.0
        onehot_n[idx[~self.pos], f[~self.pos]] = 1.0
        posgrp = np.add.reduceat(onehot_p, self.starts, axis=0)   # (nval, nfold)
        neggrp = np.add.reduceat(onehot_n, self.starts, axis=0)
        cum = np.cumsum(neggrp, axis=0) - neggrp                  # strictly below
        return posgrp.T @ (cum + 0.5 * neggrp)                    # (nfold, nfold)


def decompose(u: np.ndarray, npos: np.ndarray, nneg: np.ndarray):
    """(pooled, within, cross) AUC from the fold-pair U matrix."""
    denom = np.outer(npos, nneg)
    eye = np.eye(len(npos), dtype=bool)
    pooled = u.sum() / denom.sum()
    within = u[eye].sum() / denom[eye].sum()
    cross = u[~eye].sum() / denom[~eye].sum()
    return pooled, within, cross


def stratified_permute(y: np.ndarray, sizes_pos, sizes_neg, rng) -> np.ndarray:
    """Random fold assignment with the same per-fold positive/negative counts."""
    f = np.empty(len(y), dtype=np.int64)
    for cls, sizes in ((1.0, sizes_pos), (0.0, sizes_neg)):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        cuts = np.cumsum(sizes)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            f[part] = k
    return f


def main() -> None:
    ndraw = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    n = len(y)
    folds = get_folds(y)
    fold_of_row = np.empty(n, dtype=np.int64)
    for k, (_, va) in enumerate(folds):
        fold_of_row[va] = k
    npos = np.array([(y[fold_of_row == k] > 0.5).sum() for k in range(NFOLD)], float)
    nneg = np.array([(y[fold_of_row == k] < 0.5).sum() for k in range(NFOLD)], float)
    print(f"n={n}  fold pos {npos.astype(int)}  fold neg {nneg.astype(int)}")

    rng = np.random.default_rng(SEED)
    sizes_pos = npos.astype(int)
    sizes_neg = nneg.astype(int)
    ctrl_folds = [stratified_permute(y, sizes_pos, sizes_neg, rng) for _ in range(ndraw)]

    results = []
    print(f"\n{'member':22s} {'recipe':25s} {'pooled':>9} {'within':>9} "
          f"{'cross-within':>13} {'ctrl mu':>9} {'ctrl sd':>9} {'z':>7} {'bias(e-6)':>10}")
    for name, path, recipe, dose in MEMBERS:
        if not os.path.exists(path):
            print(f"  (missing {path})")
            continue
        s = np.load(path).astype(np.float64)
        assert s.shape == (n,), (name, s.shape)
        t0 = time.time()
        fp = FoldPairAUC(s, y)
        u = fp.u_matrix(fold_of_row)
        pooled, within, cross = decompose(u, npos, nneg)
        # gate the decomposition against sklearn on the first member
        if not results:
            ref = roc_auc_score(y, s)
            print(f"  [gate] pooled {pooled:.12f} vs roc_auc_score {ref:.12f} "
                  f"diff {pooled - ref:+.2e}")
            assert abs(pooled - ref) < 1e-10, "decomposition is not exact"
        deltas = []
        for cf in ctrl_folds:
            uc = fp.u_matrix(cf)
            _, w_c, x_c = decompose(uc, npos, nneg)
            deltas.append(x_c - w_c)
        deltas = np.asarray(deltas)
        obs = cross - within
        z = (obs - deltas.mean()) / deltas.std(ddof=1)
        # what the criterion actually pays: pooled AUC minus the coupling-free one
        bias = pooled - within
        print(f"{name:22s} {recipe:25s} {pooled:9.6f} {within:9.6f} "
              f"{obs * 1e6:+13.2f} {deltas.mean() * 1e6:+9.2f} {deltas.std(ddof=1) * 1e6:9.2f} "
              f"{z:+7.2f} {bias * 1e6:+10.2f}   ({time.time() - t0:.1f}s)")
        results.append(dict(member=name, recipe=recipe, dose=dose, pooled=pooled,
                            within=within, cross=cross, delta=obs, bias=bias,
                            ctrl_mean=float(deltas.mean()),
                            ctrl_sd=float(deltas.std(ddof=1)), z=float(z)))

    with open(OUT, "w") as fh:
        json.dump(dict(ndraw=ndraw, seed=SEED, members=results), fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
