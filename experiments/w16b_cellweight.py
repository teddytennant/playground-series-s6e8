"""w16b: act on w16a's finding — the one live residual is NOT spatially uniform.

WHAT w16a MEASURED (experiments/w16a_where.json)
------------------------------------------------
`c_avg` (w15f's averaged teacher-minus-student correction) is the only object this
workspace has ever built that beats its own matched control on the labelled rows. w16a
asked where it lives, cutting the OOF on the generator's own rule cells with a within-bin
permutation control re-binned INSIDE each segment (so the base-score conditioning is
matched segment by segment, and segments of different sizes compare).

    segment                cond AUC   ctrl      z      uniform-effect z would be
    A social>4             0.589702   0.500498  +5.16   0.11
    B soc<=4 daily>8       0.528770   0.500393  +4.42   0.52
    BAND 6<daily<=8        0.510507   0.500912  +3.50   0.74
    D soc<=4 daily<=6      0.500297   0.499584  +0.32   1.17
    E soc<=4 dailyNA       0.509562   0.499189  +2.09   0.43
    F socialNA             0.505312   0.500882  +1.16   0.58
    G both NA              0.500411   0.499953  +0.10   0.45
    chi2 vs a uniform effect = 52.21 on 7 df

The effect is inverted relative to the deficit map. w14d showed the AUC *loss* lives in
cell D (13.5% within-cell plus the four largest cross-cell terms) and that the band is
only 4.9%. But the one correction that works is worth essentially NOTHING in D (z +0.32)
and everything in A (z +5.16), the cell where the pack is already strongest (within-cell
AUC 0.9845). So "fix the model where it is worst" is the wrong instruction here: the
recoverable signal sits where the pack is already good, not where it is bad.

THE DECISION TEST, AND WHY IT IS WORTH A FILE
---------------------------------------------
w16a's cross-fitted weight search on the frozen SKF5 seed42 folds, coordinate ascent, two
passes, weights searched on four folds and scored on the fifth:

    GLOBAL   1 weight    +3.338e-06   4/5 folds
    PER-CELL 7 weights   +6.195e-06   4/5 folds
    CONTROL  7 weights on a size-matched PERMUTED segmentation   +2.370e-06   4/5

per-cell minus global +2.857e-06; per-cell minus control +3.825e-06. The control sits
BELOW the global arm, which is the right shape: splitting one weight into seven at random
costs selection noise, so the real segmentation has to pay that cost back before it shows
a gain, and it does. And the fitted weights are stable across folds in a way noise is not:

    fold      A       B     BAND      D        E       F      G
    0      0.0075  0.0025  0.0015  0.0000  0.0005  0.0005  0.0000
    1      0.0090  0.0020  0.0015  0.0000  0.0010  0.0005  0.0000
    2      0.0080  0.0025  0.0015  0.0000  0.0015  0.0005  0.0000
    3      0.0060  0.0025  0.0020  0.0005  0.0010  0.0010  0.0000
    4      0.0080  0.0020  0.0010  0.0000  0.0020  0.0015  0.0000

D is 0.0000 in 4/5 folds and G is 0.0000 in 5/5, while A lands 6-9x the global weight
(0.0011) every single time. Seven independent coordinate ascents agreeing on that ordering
is the finding; the AUC delta is just its price.

WHAT THIS RUN ADDS
------------------
1. A THIRD ARM WITH TWO PARAMETERS. Seven free weights on a +6e-6 effect is exactly the
   shape this workspace has been burned by. If the whole gain is cell A, then "A gets its
   own weight, everything else shares one" captures it with 2 parameters instead of 7 and
   is a strictly better file. Measured cross-fitted, same protocol, with its own
   size-matched permuted control (a random 11.2% of rows called 'A').
2. The full-data fit and the test file, cell labels recomputed on test.csv by the same
   function, so nothing is transferred by row index.

Everything reads saved vectors. No model is refitted. Nothing here is chosen with
reference to the public leaderboard, and this is NOT a deadline pick: it fits 2-7
parameters above the stack where blend159av_h3 / blend160origm_h3 fit zero.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DAILY, DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
SOCIAL = "social_media_hours"
N_TEST = 296_302
GRID = np.linspace(0.0, 0.02, 41)          # w16a's grid, unchanged
PASSES = 2
CELL_A = "A social>4"


def fast_auc(y, s):
    r = rankdata(s)
    npos = int(y.sum())
    nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def pct(x):
    return rankdata(x) / len(x)


def rule_cells(df):
    """w14d/w16a's cell definitions, computed from raw columns only."""
    soc = df[SOCIAL].to_numpy(np.float64)
    dly = df[DAILY].to_numpy(np.float64)
    n = len(df)
    cell = np.full(n, "?", dtype=object)
    sna, dna = np.isnan(soc), np.isnan(dly)
    cell[(~sna) & (soc > 4)] = CELL_A
    cell[(~sna) & (soc <= 4) & (~dna) & (dly > 8)] = "B soc<=4 daily>8"
    cell[(~sna) & (soc <= 4) & (~dna) & (dly > 6) & (dly <= 8)] = "BAND 6<daily<=8"
    cell[(~sna) & (soc <= 4) & (~dna) & (dly <= 6)] = "D soc<=4 daily<=6"
    cell[(~sna) & (soc <= 4) & dna] = "E soc<=4 dailyNA"
    cell[sna & (~dna)] = "F socialNA"
    cell[sna & dna] = "G both NA"
    assert (cell != "?").all()
    return cell


def ascend(y, br, c, assign, idx):
    """Coordinate ascent for one weight per level of `assign`, on rows `idx`."""
    lv = sorted(set(assign[idx]))
    w = {v: 0.0 for v in lv}
    cur = br[idx].copy()
    for _ in range(PASSES):
        for v in lv:
            mv = assign[idx] == v
            if mv.sum() < 500:
                continue
            z = cur.copy()
            best, bw = -1.0, w[v]
            for g in GRID:
                z[mv] = br[idx][mv] + g * c[idx][mv]
                a = fast_auc(y[idx], z)
                if a > best:
                    best, bw = a, g
            w[v] = bw
            cur[mv] = br[idx][mv] + bw * c[idx][mv]
    return w


def apply_w(br, c, assign, w, idx):
    z = br[idx].copy()
    for v, wv in w.items():
        mv = assign[idx] == v
        z[mv] = br[idx][mv] + wv * c[idx][mv]
    return z


def xfit(y, br, c, assign, folds, tag):
    deltas, ws = [], []
    for itr, iva in folds:
        w = ascend(y, br, c, assign, itr)
        zv = apply_w(br, c, assign, w, iva)
        deltas.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
        ws.append({str(k): float(v) for k, v in w.items()})
    deltas = np.array(deltas)
    print(f"{tag:<28} xfit dAUC {deltas.mean():+.3e}  "
          f"({' '.join(f'{d:+.1e}' for d in deltas)})  "
          f"{int((deltas > 0).sum())}/5 folds", flush=True)
    return deltas, ws


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    c_avg = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    base_auc = fast_auc(y, base_p)
    br = pct(base_p)

    cell = rule_cells(tr)
    a_share = float((cell == CELL_A).mean())
    print(f"base {BASE} pooled OOF AUC {base_auc:.7f}   n {n:,}   "
          f"cell A share {a_share:.4f}\n")

    out = {"base_auc": float(base_auc), "a_share": a_share}

    # ---------------- the three real arms ----------------
    one = np.zeros(n, dtype=object)
    d_glob, w_glob = xfit(y, br, c_avg, one, folds, "GLOBAL   1 weight")

    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    d_a, w_a = xfit(y, br, c_avg, a_only, folds, "A-ONLY   2 weights")

    d_cell, w_cell = xfit(y, br, c_avg, cell, folds, "PER-CELL 7 weights")

    # ---------------- size-matched permuted controls ----------------
    print()
    rng = np.random.default_rng(4242)
    perm7 = cell[rng.permutation(n)]
    d_c7, _ = xfit(y, br, c_avg, perm7, folds, "CTRL permuted 7")
    rng2 = np.random.default_rng(909)
    perm2 = a_only[rng2.permutation(n)]
    d_c2, _ = xfit(y, br, c_avg, perm2, folds, "CTRL permuted 2")

    print()
    for tag, d in (("global", d_glob), ("A-only", d_a), ("per-cell", d_cell)):
        print(f"CV({tag:<9}) = {base_auc + d.mean():.7f}")
    print(f"\nA-only   MINUS global {d_a.mean()-d_glob.mean():+.3e}   "
          f"MINUS its ctrl {d_a.mean()-d_c2.mean():+.3e}")
    print(f"per-cell MINUS global {d_cell.mean()-d_glob.mean():+.3e}   "
          f"MINUS its ctrl {d_cell.mean()-d_c7.mean():+.3e}")
    print(f"per-cell MINUS A-only {d_cell.mean()-d_a.mean():+.3e}")

    out["arms"] = {
        k: dict(xfit=float(v.mean()), folds_pos=int((v > 0).sum()),
                per_fold=[float(x) for x in v],
                cv=float(base_auc + v.mean()))
        for k, v in (("global", d_glob), ("a_only", d_a), ("per_cell", d_cell),
                     ("ctrl7", d_c7), ("ctrl2", d_c2))
    }
    out["fold_weights"] = dict(a_only=w_a, per_cell=w_cell, glob=w_glob)

    # ---------------- pick the arm, full-data fit, test file ----------------
    # Selection rule fixed before the numbers: take the highest cross-fitted arm, and
    # break a tie inside 1e-6 in favour of FEWER parameters.
    cands = [("a_only", d_a.mean(), a_only, 2), ("per_cell", d_cell.mean(), cell, 7)]
    cands.sort(key=lambda q: (-q[1],))
    top = cands[0]
    if abs(cands[0][1] - cands[1][1]) < 1e-6:
        top = min(cands, key=lambda q: q[3])
    name, dmean, assign_tr, npar = top
    print(f"\nchosen arm: {name}  ({npar} parameters, cross-fitted {dmean:+.3e})")

    w_full = ascend(y, br, c_avg, assign_tr, np.arange(n))
    print(f"full-data weights: { {k: round(v, 4) for k, v in w_full.items()} }")
    out["chosen"] = dict(arm=name, n_params=npar, xfit=float(dmean),
                         cv=float(base_auc + dmean),
                         full_weights={str(k): float(v) for k, v in w_full.items()})

    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test_avg.npy")).astype(np.float64)
    assert ct.shape == (N_TEST,)

    cell_te = rule_cells(te)
    if name == "a_only":
        assign_te = np.where(cell_te == CELL_A, "A", "rest").astype(object)
    else:
        assign_te = cell_te
    # every level fitted on train must exist on test, and vice versa
    assert set(assign_te) == set(w_full), (sorted(set(assign_te)), sorted(w_full))
    shares = {v: float((assign_te == v).mean()) for v in sorted(set(assign_te))}
    print("test-side segment shares:", {k: round(v, 4) for k, v in shares.items()})
    out["test_shares"] = shares

    pred = apply_w(btr, ct, assign_te, w_full, np.arange(N_TEST))
    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})

    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16b_cellweight.csv")
    sub.to_csv(path, index=False)

    rho_base = float(np.corrcoef(rankdata(strict), rankdata(bp))[0, 1])
    prev = os.path.join(SUB, "w15f_antistudent_avg.csv")
    rho_prev = np.nan
    if os.path.exists(prev):
        pv = pd.read_csv(prev).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        rho_prev = float(np.corrcoef(rankdata(strict), rankdata(pv))[0, 1])
        ident = bool(np.array_equal(rankdata(strict), rankdata(pv)))
        print(f"identical to w15f_antistudent_avg? {ident}")
        out["identical_to_w15f"] = ident
    print(f"\nwrote {path}  rows {len(sub):,}  "
          f"spearman vs base {rho_base:.7f}  vs w15f_avg {rho_prev:.7f}")
    out["file"] = dict(path=path, spearman_vs_base=rho_base,
                       spearman_vs_w15f=rho_prev)

    json.dump(out, open(os.path.join(HERE, "w16b_cellweight.json"), "w"), indent=1)
    print("wrote experiments/w16b_cellweight.json")


if __name__ == "__main__":
    main()
