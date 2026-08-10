"""Does the linear stack mis-rank ACROSS missingness regimes?

Global AUC compares every row against every other row, including rows from completely
different information regimes: a complete row is ranked at AUC 0.9764 while a row missing
>=4 columns is ranked at 0.9280. Within-regime ranking is the members' job and they do it
well. The part nobody checks is whether the two regimes are on a COMMON SCALE -- whether
a 0.8 predicted on a complete row means the same thing as a 0.8 predicted on a row with
four holes in it. If it does not, global AUC pays for it even with perfect within-regime
ranking.

`lin_regime` tested a different thing (refitting the weights per bucket) and lost 0.000026,
but it paid for that with 5x less data per coefficient vector, which could easily mask a
calibration gain. This isolates calibration instead: a per-bucket ISOTONIC map on the
global stack's output. Isotonic is monotone, so within-bucket ranking is unchanged by
construction and within-bucket AUC cannot move at all. Every AUC change here is purely
the cross-bucket regrading. That makes it a clean one-sided test of the hypothesis.

The map is fitted out-of-fold (inner 4-fold inside each outer training set), because an
isotonic curve fitted on its own training predictions is a memorisation device.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from meta import PRED, inner_linear, to_logit  # noqa: E402
from stack import DEFAULT_DROP, load_members  # noqa: E402

N_BUCKET = 5


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(y, len(te),
                               extra_dirs=(os.path.join(DATA, "ext_members"),),
                               drop=set(DEFAULT_DROP))
    Z = to_logit(O)
    b = np.minimum(tr[PRED].isna().sum(1).to_numpy(), N_BUCKET - 1)
    print(f"{len(names)} members; bucket sizes {np.bincount(b)}", flush=True)

    base = np.zeros(len(y))
    iso = np.zeros(len(y))
    for k, (itr, iva) in enumerate(get_folds(y)):
        lin_tr, lin_va, _ = inner_linear(Z, y, itr, iva)
        base[iva] = lin_va
        for bb in range(N_BUCKET):
            mt, mv = b[itr] == bb, b[iva] == bb
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(lin_tr[mt], y[itr][mt])
            iso[iva[mv]] = ir.predict(lin_va[mv])
        print(f"fold{k}  base {roc_auc_score(y[iva], base[iva]):.6f}"
              f"  iso {roc_auc_score(y[iva], iso[iva]):.6f}"
              f"  delta {roc_auc_score(y[iva], iso[iva]) - roc_auc_score(y[iva], base[iva]):+.6f}",
              flush=True)

    ab, ai = roc_auc_score(y, base), roc_auc_score(y, iso)
    print(f"\nfull OOF  base {ab:.6f}  per-bucket isotonic {ai:.6f}  delta {ai - ab:+.6f}")
    print("(noise floor on a full-OOF difference is ~0.00005)")


if __name__ == "__main__":
    main()
