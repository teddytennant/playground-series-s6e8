"""Is the stack's residual concentrated in the generator's coin-flip band?

RESEARCH.md reads a two-threshold rule off the ORIGINAL 7,500-row dataset:

    social > 4                 -> rate 1.0000   (2,748 rows)
    social <= 4 & daily > 8    -> rate 1.0000   (2,093)
    social <= 4 & daily <= 6   -> rate 0.0000   (1,634)
    social <= 4 & 6 < daily<=8 -> rate 0.4556   (1,025)   <- "irreducible coin flip"

and states that the generator smeared that rule into a ramp. `errormap.py` (2026-08-11)
segmented our OOF on `daily_band` ALONE and found the hard population is 4-8h. Nobody has
ever cut our own OOF on the JOINT rule cells, which is the segmentation the claim is
actually about, and nobody has priced the headroom in AUC.

Three questions, in the order that decides whether a correction is worth building:

1. WHERE IS THE PAIR MASS. AUC is an average over positive/negative pairs, so a cell can
   be very hard and still be worth nothing. Exact decomposition: every pair falls in
   exactly one (cell of the positive, cell of the negative) bucket, so

       AUC = sum_{i,j} U(pos_i, neg_j) / (Npos * Nneg)

   with U the Mann-Whitney statistic (ties at 0.5). Summing the diagonal gives the
   within-cell mass; the rest is cross-cell. Verified against roc_auc_score.

2. HOW MUCH IS ON THE TABLE. Set a bucket's term to its maximum (perfect ordering) and
   re-add: that is the exact global AUC an oracle for that bucket would produce. Done for
   the diagonal, the off-diagonal, and the band alone. An upper bound no method can pass.

3. IS THE MODEL SYSTEMATICALLY WRONG BETWEEN CELLS. Within-cell ranking cannot be fixed
   by any monotone per-cell correction (monotone maps do not reorder within a cell), so
   the only thing a targeted correction can touch is the cross-cell scale. Cross-fitted
   per-cell isotonic tests exactly that, with a SIZE-MATCHED permuted-cell control --
   the workspace's own rule after `erroran.py`'s chi2 near-miss: a segmentation with the
   same cell sizes and random membership is the only admissible null.

Everything reads saved vectors. No model is refitted.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DAILY, SUB, TARGET, get_folds, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
BLEND = os.environ.get("W14D_BLEND", "blend159av_h3")
SEED = 20260814


def mw(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U: #(a > b) + 0.5 * #(a == b), over the full cross product."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a = np.sort(a)
    lo = np.searchsorted(a, b, side="left")     # #(a < b)
    hi = np.searchsorted(a, b, side="right")    # #(a <= b)
    gt = a.size - hi                            # #(a > b)
    eq = hi - lo
    return float(gt.sum() + 0.5 * eq.sum())


def cells(tr: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The generator's rule cells, with the missing-driver rows kept separate."""
    s = tr[SOCIAL].to_numpy("float64")
    d = tr[DAILY].to_numpy("float64")
    sm, dm = np.isnan(s), np.isnan(d)
    g = np.full(len(tr), "?", dtype=object)
    g[~sm & (s > 4.0)] = "A social>4"
    lo = ~sm & (s <= 4.0)
    g[lo & ~dm & (d > 8.0)] = "B soc<=4 daily>8"
    g[lo & ~dm & (d > 6.0) & (d <= 8.0)] = "BAND soc<=4 6<daily<=8"
    g[lo & ~dm & (d <= 6.0)] = "D soc<=4 daily<=6"
    g[lo & dm] = "E soc<=4 daily NA"
    g[sm & ~dm] = "F social NA"
    g[sm & dm] = "G both NA"
    order = ["A social>4", "B soc<=4 daily>8", "BAND soc<=4 6<daily<=8",
             "D soc<=4 daily<=6", "E soc<=4 daily NA", "F social NA", "G both NA"]
    return g, order


def decompose(y, s, g, order):
    """U[i, j] = Mann-Whitney of positives in cell i against negatives in cell j."""
    pos = {c: s[(g == c) & (y == 1)] for c in order}
    neg = {c: s[(g == c) & (y == 0)] for c in order}
    k = len(order)
    U = np.zeros((k, k))
    M = np.zeros((k, k))       # max possible = npos_i * nneg_j
    for i, ci in enumerate(order):
        for j, cj in enumerate(order):
            U[i, j] = mw(pos[ci], neg[cj])
            M[i, j] = pos[ci].size * neg[cj].size
    return U, M, pos, neg


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    s = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    assert s.shape == y.shape
    g, order = cells(tr)
    assert (g != "?").all(), "unassigned rows"

    gauc = roc_auc_score(y, s)
    print(f"{BLEND}: global OOF AUC {gauc:.6f}   n={len(y):,}\n")

    U, M, pos, neg = decompose(y, s, g, order)
    tot = M.sum()
    check = U.sum() / tot
    print(f"decomposition check: sum(U)/Npos*Nneg = {check:.6f} "
          f"(roc_auc_score {gauc:.6f}, diff {check - gauc:+.2e})\n")

    # ---- 1. per-cell description -----------------------------------------
    print(f"{'cell':24s} {'n':>9s} {'share':>7s} {'base':>7s} "
          f"{'within AUC':>11s} {'within pair share':>18s}")
    for i, c in enumerate(order):
        m = g == c
        n = int(m.sum())
        wshare = M[i, i] / tot
        wauc = U[i, i] / M[i, i] if M[i, i] > 0 else float("nan")
        print(f"{c:24s} {n:9,d} {n/len(y):7.3f} {y[m].mean():7.4f} "
              f"{wauc:11.6f} {wshare:18.4f}")

    diag = np.diag(M).sum() / tot
    print(f"\nwithin-cell pair mass {diag:.4f}   cross-cell {1 - diag:.4f}")
    within_auc = np.diag(U).sum() / np.diag(M).sum()
    off = (U.sum() - np.diag(U).sum()) / (tot - np.diag(M).sum())
    print(f"pooled within-cell AUC {within_auc:.6f}   pooled cross-cell AUC {off:.6f}")

    # ---- 2. oracle headroom ----------------------------------------------
    print("\noracle headroom (exact upper bounds; global AUC if that bucket were "
          "ordered perfectly)")
    rows = []
    Uw = U.copy()
    np.fill_diagonal(Uw, np.diag(M))
    rows.append(("perfect WITHIN every cell", Uw.sum() / tot))
    Uo = M.copy()
    np.fill_diagonal(Uo, np.diag(U))
    rows.append(("perfect ACROSS all cell pairs", Uo.sum() / tot))
    bi = order.index("BAND soc<=4 6<daily<=8")
    Ub = U.copy()
    Ub[bi, bi] = M[bi, bi]
    rows.append(("perfect within the BAND only", Ub.sum() / tot))
    Ub2 = U.copy()
    Ub2[bi, :] = M[bi, :]
    Ub2[:, bi] = M[:, bi]
    rows.append(("BAND ordered perfectly vs everything", Ub2.sum() / tot))
    hard = [order.index("BAND soc<=4 6<daily<=8"), order.index("D soc<=4 daily<=6")]
    Uh = U.copy()
    for i in hard:
        Uh[i, i] = M[i, i]
    rows.append(("perfect within BAND + D", Uh.sum() / tot))
    for name, v in rows:
        print(f"  {name:40s} {v:.6f}  ({v - gauc:+.6f})")

    # ---- 2b. where the AUC deficit actually lives ------------------------
    # Every pair is in exactly one (cell of pos, cell of neg) bucket, so the deficit
    # 1 - AUC splits exactly over the 49 buckets. This is the "where is the residual"
    # question in the only units that matter.
    loss = (M - U) / tot
    print(f"\nAUC deficit {1 - gauc:.6f} split over buckets (pos cell x neg cell), "
          f"top 12 by share of the deficit")
    flat = sorted(((loss[i, j], i, j) for i in range(len(order))
                   for j in range(len(order))), reverse=True)
    print(f"  {'pos cell':24s} {'neg cell':24s} {'deficit':>9s} {'share':>7s}")
    for v, i, j in flat[:12]:
        mark = " <- WITHIN" if i == j else ""
        print(f"  {order[i]:24s} {order[j]:24s} {v:9.6f} "
              f"{v/(1-gauc):7.3f}{mark}")
    dl = sum(loss[i, i] for i in range(len(order)))
    print(f"  within-cell deficit {dl:.6f} ({dl/(1-gauc):.3f} of total); "
          f"cross-cell {1-gauc-dl:.6f} ({1-(dl/(1-gauc)):.3f})")

    # ---- 3. targeted correction: cross-fitted per-cell isotonic ----------
    # Monotone within a cell => within-cell AUC is unchanged by construction, so every
    # movement here is cross-cell regrading, i.e. exactly the "systematic wrongness"
    # a targeted correction could fix. Matched control: permuted cell labels, same sizes.
    print("\ncross-fitted per-cell isotonic (5 outer folds, map fitted on the outer "
          "training half only)")
    rng = np.random.default_rng(SEED)
    gperm = g.copy()
    rng.shuffle(gperm)
    for label, gg in (("real cells", g), ("size-matched permuted control", gperm)):
        out = np.empty_like(s)
        for trn, val in get_folds(y):
            for c in order:
                mt = gg[trn] == c
                mv = gg[val] == c
                if mv.sum() == 0:
                    continue
                if mt.sum() < 50 or y[trn][mt].std() == 0:
                    out[val[mv]] = s[val[mv]]
                    continue
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(s[trn][mt], y[trn][mt])
                out[val[mv]] = iso.predict(s[val[mv]])
        a = roc_auc_score(y, out)
        print(f"  {label:32s} {a:.6f}  ({a - gauc:+.6f})")


if __name__ == "__main__":
    main()
