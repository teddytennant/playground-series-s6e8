"""Where does the pack's ONE known residual signal actually live?

The assigned angle is "segment the OOF errors and look for structure a feature could
capture". That angle is closed as a *search over the 12 columns* and the closure is a
bound, not a shrug:

  * w14d cut the OOF on the generator's own rule cells: per-cell isotonic real-minus-
    permuted-control +6e-6 with BOTH arms negative, and a cell-local booster negative at
    9 of 9 checkpoints in the three cells carrying most of the deficit.
  * w15b then power-calibrated exactly that family of nulls: inject a leader-sized signal
    (tau=0.134, +175e-6) into labels it generates itself and re-run resid_boost2.py's own
    instrument -> control-corrected recovery 78-102% at every checkpoint on both frames.
    So the nulls are genuine bounds. Nothing an error segmentation proposes as a NEW
    feature of the 12 columns can be there.
  * w15c closed row identity and in-fold lookups; w15b closed the exact-lattice surface.

So re-running "segment and hunt for a feature" is a measured waste. But there is exactly
one place where the pack is KNOWN to be wrong in a way something captures, and nobody has
ever asked where it is: w15f's averaged teacher-minus-student correction `c_avg`. It is
the only object all week that beats its own matched control on the labelled rows
(cond AUC 0.505819 vs 0.500227 +- 0.00116, z +4.84; cross-fitted dAUC +3.58e-6, 4/5 folds).
w15f also showed it is NOT transductive -- it is an ordinary inductive function of the 12
columns, i.e. it sits INSIDE w15b's power bound and is a genuine, if tiny, miss.

That makes one question live and cheap: **is that miss spatially structured?**

  - If it concentrates in one region, the region is nameable, an interaction of `c_avg`
    with a region indicator is a real feature, and a later slot can build and send it.
  - If it is uniform across every segmentation the generator and the mask define, then the
    last thing the pack gets wrong is diffuse, and the error-analysis family closes with a
    positive statement instead of another null.

Instrument (w15f_eval.cond_auc, unchanged): pooled within-bin Mann-Whitney of `c_avg`
against the label, bins taken on the base score, RESTRICTED to a segment and re-binned
inside it so the base-score conditioning is matched within the segment. Control: the same
values permuted inside (segment x bin), 24 seeds -- marginal and bin structure survive,
only the row correspondence dies. Scale-free, so segments of different sizes compare.

Second question, the one with a decision attached: does letting the weight vary by segment
beat one global weight OUT OF FOLD? Coordinate ascent on the frozen SKF5 seed42 folds,
searched on four folds and scored on the fifth, against the global-weight arm and against
a permuted-membership control segmentation of the same sizes.

Everything reads saved vectors. No model is refitted.
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DAILY, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15f_eval import cond_auc, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_BINS = 200
N_SEEDS = 24
SOCIAL = "social_media_hours"


def fast_auc(y, s):
    r = rankdata(s)
    npos = int(y.sum())
    nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def cond_auc_seg(score, c, y, mask, seeds=N_SEEDS, seed0=7000):
    """cond AUC of c|score restricted to `mask`, re-binned inside the segment.

    Returns (real, ctrl_mean, ctrl_sd, n_pairs). Control permutes c inside each
    within-segment bin, so it is matched on segment membership AND on base score.
    """
    s, cc, yy = score[mask], c[mask], y[mask]
    if yy.sum() < 50 or (1 - yy).sum() < 50:
        return np.nan, np.nan, np.nan, 0
    nb = min(N_BINS, max(2, int(mask.sum() // 400)))
    edges = np.quantile(s, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(edges, s, "right") - 1, 0, nb - 1)
    idx = [np.flatnonzero(b == k) for k in range(nb)]

    def _run(vals):
        num = den = 0.0
        for ix in idx:
            yb = yy[ix]
            npos, nneg = int(yb.sum()), int(len(yb) - yb.sum())
            if npos == 0 or nneg == 0:
                continue
            r = rankdata(vals[ix])
            num += r[yb == 1].sum() - npos * (npos + 1) / 2.0
            den += npos * nneg
        return num / den, den

    real, den = _run(cc)
    ctrl = []
    for t in range(seeds):
        rng = np.random.default_rng(seed0 + t)
        v = cc.copy()
        for ix in idx:
            v[ix] = cc[rng.permutation(ix)]
        ctrl.append(_run(v)[0])
    ctrl = np.array(ctrl)
    return real, ctrl.mean(), ctrl.std(), den


def build_segments(tr):
    """The segmentations the data itself defines. No free choices."""
    soc = tr[SOCIAL].to_numpy()
    dly = tr[DAILY].to_numpy()
    n = len(tr)
    segs = {}

    # (a) the generator's rule cells, exactly w14d's definitions
    cell = np.full(n, "?", dtype=object)
    sna, dna = np.isnan(soc), np.isnan(dly)
    cell[(~sna) & (soc > 4)] = "A social>4"
    cell[(~sna) & (soc <= 4) & (~dna) & (dly > 8)] = "B soc<=4 daily>8"
    cell[(~sna) & (soc <= 4) & (~dna) & (dly > 6) & (dly <= 8)] = "BAND 6<daily<=8"
    cell[(~sna) & (soc <= 4) & (~dna) & (dly <= 6)] = "D soc<=4 daily<=6"
    cell[(~sna) & (soc <= 4) & dna] = "E soc<=4 dailyNA"
    cell[sna & (~dna)] = "F socialNA"
    cell[sna & dna] = "G both NA"
    segs["rule_cell"] = cell

    # (b) missing-count band -- w15c showed the mask is the whole train/test difference
    cols = [c for c in tr.columns if c not in ("id", TARGET)]
    nm = tr[cols].isna().sum(axis=1).to_numpy()
    segs["n_missing"] = np.array([f"{min(v, 3)}{'+' if v >= 3 else ''} missing" for v in nm],
                                 dtype=object)

    # (c) base-score decile -- where in the ranking, rather than where in the frame
    return segs


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    c_avg = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    z2 = np.load(os.path.join(HERE, "w15f_nested2.npz"))
    fold_id = z2["fold_id"]
    folds = get_folds(y)
    base_auc = fast_auc(y, base_p)
    br = pct(base_p)

    print(f"base {BASE} pooled OOF AUC {base_auc:.7f}   n {n}")
    r, m, sd, den = cond_auc_seg(base_p, c_avg, y, np.ones(n, bool))
    print(f"GLOBAL cond AUC(c_avg|base) {r:.6f}  ctrl {m:.6f} sd {sd:.2e}  "
          f"z {(r-m)/sd:+.2f}   pairs {den:.3e}\n")
    out = {"base_auc": float(base_auc),
           "global": dict(real=float(r), ctrl=float(m), sd=float(sd),
                          z=float((r - m) / sd), pairs=float(den))}

    segs = build_segments(tr)

    # ---- score decile is derived from base_p, so build it here ----
    dec = np.clip(np.searchsorted(np.quantile(base_p, np.linspace(0, 1, 11)),
                                  base_p, "right") - 1, 0, 9)
    segs["score_decile"] = np.array([f"decile {d}" for d in dec], dtype=object)

    out["segmentations"] = {}
    for name, lab in segs.items():
        print(f"=== {name} " + "=" * (60 - len(name)))
        print(f"{'segment':<20} {'n':>8} {'share':>7} {'packAUC':>9} "
              f"{'condAUC':>9} {'ctrl':>9} {'sd':>8} {'z':>7} {'pairshare':>10}")
        rows = []
        total_pairs = out["global"]["pairs"]
        for v in sorted(set(lab)):
            mask = lab == v
            k = int(mask.sum())
            pa = fast_auc(y[mask], base_p[mask]) if 0 < y[mask].sum() < k else np.nan
            r, m, sd, den = cond_auc_seg(base_p, c_avg, y, mask)
            zz = (r - m) / sd if sd and np.isfinite(sd) else np.nan
            print(f"{str(v):<20} {k:>8} {k/n:>7.3f} {pa:>9.6f} "
                  f"{r:>9.6f} {m:>9.6f} {sd:>8.2e} {zz:>+7.2f} {den/total_pairs:>10.3f}")
            rows.append(dict(segment=str(v), n=k, share=float(k / n),
                             pack_auc=float(pa), cond=float(r), ctrl=float(m),
                             sd=float(sd), z=float(zz),
                             pair_share=float(den / total_pairs)))
        out["segmentations"][name] = rows
        # heterogeneity: is the spread of z larger than iid sampling permits?
        zs = np.array([q["z"] for q in rows if np.isfinite(q["z"])])
        ws = np.array([q["pair_share"] for q in rows if np.isfinite(q["z"])])
        exp = out["global"]["z"] * np.sqrt(ws)          # z if the effect were uniform
        chi2 = float(((zs - exp) ** 2).sum())
        print(f"  uniform-effect prediction z_s = z_global*sqrt(pairshare) -> "
              f"{np.round(exp, 2)}")
        print(f"  observed                                                  "
              f"{np.round(zs, 2)}")
        print(f"  chi2 vs uniform = {chi2:.2f} on {len(zs)} df\n")
        out["segmentations"][name + "_chi2"] = dict(chi2=chi2, df=int(len(zs)))

    # ---- the decision test: does a per-segment weight beat one global weight? ----
    print("=== per-segment weight, cross-fitted on the frozen folds " + "=" * 8)
    lab = segs["rule_cell"]
    levels = sorted(set(lab))
    GRID = np.linspace(0.0, 0.02, 41)

    def xfit(assign):
        """assign: array of segment labels (or all-one for the global arm)."""
        lv = sorted(set(assign))
        deltas, wlist = [], []
        for itr, iva in folds:
            w = {v: 0.0 for v in lv}
            cur = br[itr].copy()
            for _ in range(2):                       # two coordinate-ascent passes
                for v in lv:
                    mv = assign[itr] == v
                    if mv.sum() < 500:
                        continue
                    z = cur.copy()
                    best, bw = -1.0, w[v]
                    for g in GRID:
                        z[mv] = br[itr][mv] + g * c_avg[itr][mv]
                        a = fast_auc(y[itr], z)
                        if a > best:
                            best, bw = a, g
                    w[v] = bw
                    cur[mv] = br[itr][mv] + bw * c_avg[itr][mv]
            zv = br[iva].copy()
            for v in lv:
                mv = assign[iva] == v
                zv[mv] = br[iva][mv] + w[v] * c_avg[iva][mv]
            deltas.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
            wlist.append({str(k): float(x) for k, x in w.items()})
        return np.array(deltas), wlist

    d_glob, w_glob = xfit(np.zeros(n, dtype=object))
    print(f"GLOBAL   one weight     xfit dAUC {d_glob.mean():+.3e}  "
          f"{int((d_glob > 0).sum())}/5 folds   w {[list(q.values())[0] for q in w_glob]}")
    d_cell, w_cell = xfit(lab)
    print(f"PER-CELL {len(levels)} weights    xfit dAUC {d_cell.mean():+.3e}  "
          f"{int((d_cell > 0).sum())}/5 folds")
    rng = np.random.default_rng(4242)
    perm = lab[rng.permutation(n)]                   # size-matched random segmentation
    d_ctrl, _ = xfit(perm)
    print(f"CONTROL  {len(levels)} weights on a size-matched PERMUTED segmentation  "
          f"xfit dAUC {d_ctrl.mean():+.3e}  {int((d_ctrl > 0).sum())}/5 folds")
    print(f"\nper-cell MINUS global   {d_cell.mean()-d_glob.mean():+.3e}")
    print(f"control  MINUS global   {d_ctrl.mean()-d_glob.mean():+.3e}")
    print(f"per-cell MINUS control  {d_cell.mean()-d_ctrl.mean():+.3e}")

    out["weight"] = dict(
        glob=float(d_glob.mean()), glob_folds=int((d_glob > 0).sum()),
        cell=float(d_cell.mean()), cell_folds=int((d_cell > 0).sum()),
        ctrl=float(d_ctrl.mean()), ctrl_folds=int((d_ctrl > 0).sum()),
        cell_weights=w_cell, glob_weights=w_glob,
        cell_minus_glob=float(d_cell.mean() - d_glob.mean()),
        cell_minus_ctrl=float(d_cell.mean() - d_ctrl.mean()),
    )

    with open(os.path.join(HERE, "w16a_where.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote experiments/w16a_where.json")


if __name__ == "__main__":
    main()
