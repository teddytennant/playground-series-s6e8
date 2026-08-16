"""w15g -- a unit test for the claim the rest of this run leans on.

CLAIM.  Write the pooled OOF AUC as a sum over fold pairs (w15g_coupling.py).  Then

    A_cross - A_within  is a function of the five folds' SCORE MARGINALS and of
    nothing else.  Label coupling between rows in different folds -- s_i depending on
    y_j through a shared target-encoded lattice cell -- cannot move it.

PROOF SKETCH (all steps are empirical identities, no independence is assumed):
    A_kl  = int Hhat_l dGhat_k                                  (definition of the U-stat)
    Hhat_l = (Fhat_l - pi Ghat_l) / (1 - pi)                     (definition of Fhat_l)
    int Ghat_l dGhat_k + int Ghat_k dGhat_l = 1                  (tie-aware symmetry)
    int Ghat_k dGhat_k = 1/2                                     (tie-aware self-pairing)
  => with Fhat_l = Fhat common to all folds and equal fold sizes,
     mean_{k!=l} A_kl == mean_k A_kk,  EXACTLY.

Coupling therefore cannot enter, because the O(n) coupled pairs are a vanishing share of
the O(n^2) pairs the statistic averages over.  The consequence for the run is that
w15b's proposed mechanism -- "our OOF tracks realised cell counts, which test predictions
cannot do" -- has no channel into pooled OOF AUC even in principle.

A claim like that has to be tested, not asserted, so this builds the most extreme coupling
the mechanism admits and checks it:

    cells of exactly 5 rows, one per fold, and the score IS the leave-fold-out cell mean.
    Every row's score is 100% other rows' labels.  Nothing carries more coupling than this.

Expected under the claim: pooled AUC = 0.5 (no true signal exists), and cross == within
after fold-normalisation.  If the pooled AUC comes out above 0.5, w15b is right and this
whole run's conclusion is wrong.

Arm 2 breaks the premise on purpose: fold 0's scores are shifted, so the marginals no
longer match.  cross - within must then become large, which is what shows the statistic
responds to marginal mismatch and to nothing else.

Usage:  python experiments/w15g_identity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15g_coupling import FoldPairAUC, decompose  # noqa: E402
from w15g_foldscale import fold_normalise  # noqa: E402

NCELL = 40000
NFOLD = 5
PI = 0.70


def main() -> None:
    rng = np.random.default_rng(7)
    n = NCELL * NFOLD
    cell = np.repeat(np.arange(NCELL), NFOLD)
    fold = np.tile(np.arange(NFOLD), NCELL)
    y = (rng.random(n) < PI).astype(np.float64)

    # score = leave-own-fold-out mean of the cell's OTHER four labels. 100% coupling.
    tot = np.bincount(cell, weights=y, minlength=NCELL)
    s = (tot[cell] - y) / (NFOLD - 1)
    s = s + 1e-9 * rng.standard_normal(n)      # break exact ties reproducibly

    npos = np.array([(y[fold == k] > 0.5).sum() for k in range(NFOLD)], float)
    nneg = np.array([(y[fold == k] < 0.5).sum() for k in range(NFOLD)], float)

    for tag, sc in (("pure coupling", s),
                    ("+ fold-0 marginal shift", s + 0.02 * (fold == 0))):
        pooled, within, cross = decompose(FoldPairAUC(sc, y).u_matrix(fold), npos, nneg)
        pn, wn, xn = decompose(
            FoldPairAUC(fold_normalise(sc, fold), y).u_matrix(fold), npos, nneg)
        print(f"\n{tag}")
        print(f"  pooled {pooled:.6f}  (sklearn {roc_auc_score(y, sc):.6f})")
        print(f"  within {within:.6f}   cross {cross:.6f}   "
              f"cross-within {(cross - within) * 1e6:+.2f}e-6")
        print(f"  after fold-normalisation: pooled {pn:.6f}  within {wn:.6f}  "
              f"cross-within {(xn - wn) * 1e6:+.2f}e-6")

    # and the same coupling channel with a REAL signal underneath, to show that the
    # coupled component still contributes nothing on top of it
    base = rng.standard_normal(n) + 1.2 * y
    for lam in (0.0, 0.5, 2.0):
        sc = base + lam * s
        pooled, within, cross = decompose(FoldPairAUC(sc, y).u_matrix(fold), npos, nneg)
        print(f"\nbase + {lam:.1f} x (pure coupling): pooled {pooled:.6f}  "
              f"cross-within {(cross - within) * 1e6:+.2f}e-6")


if __name__ == "__main__":
    main()
