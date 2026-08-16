"""w16r: re-read two POOLED nulls segmented by the strong categorical.

Slot 7's assignment, second half: a null measured POOLED over a strong categorical can hide a
real effect inside one level, because a +X in one level cancels a -X in another. w16a proved
this shape is live in this dataset -- the one surviving residual `c_avg` reads pooled cond-AUC
z +4.11 which decomposes into cell A **+5.16** and cell D **+0.32**, a 16x spread, chi2
52.21/7df against uniformity.

The strong categorical is the 7-level generator rule cell. The nominal categoricals
(`gender`, `stress_level`, `academic_work_impact`) are weak -- `errormap.py` put
`stress_level` at +0.000219 pooled-within -- and are not used here.

TWO NULLS, both re-readable with NO model refits because the vectors are already on disk.

ITEM A -- w15f 6(a), "the transductive component is decoration"
---------------------------------------------------------------
w15f measured the pure transductive component `c_trans - c_induc` at cond AUC 0.500579
vs control 0.499935 +- 0.00125, **z +0.52**, and called it a clean null. That closed the
specific teacher-minus-student object and is quoted as the reason the transductive CLASS is
"narrowed rather than settled" in w16a 6, w16c 7, w16h 6 and w16l 4.

It was measured POOLED. `experiments/w15f_extract.py` contains no segment code at all; the
segmented instrument `cond_auc_seg` did not exist until `w16a_where.py` wrote it a day later.
A pooled z of +0.52 is exactly what a genuine +2.5 to +3 confined to cell A would look like
after averaging against nothing elsewhere -- w16a's own "z if the effect were uniform" column
is that arithmetic.

Re-read here with `w16a_where.py`'s `cond_auc_seg`, unchanged, 24-seed within-(segment x bin)
permutation control, on the same base `blend159av_h3`. `c_avg` is run alongside as the POSITIVE
CONTROL: the instrument must reproduce w16a's published per-cell z column, or the harness has
drifted and nothing else in this file may be believed.

ITEM B -- w16l 2, "the mask as a training weight has no directional component"
------------------------------------------------------------------------------
w16l measured `imp - unw` = -3.730e-6 and `imp - anti` = +0.052e-6 and concluded the entire
effect is the effective-sample-size toll with no directional part. That CLOSED the transductive
class outright.

It was measured POOLED over all 691,369 rows. Its only two segmentation axes are the transform
family and the arm -- neither is a data categorical, and the entry advertises the per-transform
cut as if it settled the question. The mechanism makes the pooled reading suspect in a specific
way: importance weighting only pays under misspecification, misspecification is local, and the
weights move mass directly between the driver-missing cells (E/F/G) and the driver-observed
cells (A/B/BAND/D). A +X in one group against a -X in the other is the textbook cancellation.

Only the `imp` arm's OOF was saved (`submissions/oof_w16l_maskw_h3.npy`, written at
`w16l_maskweight.py:277`), so `anti` is not available without a refit. What IS free is
`imp - unw` per cell, where `unw` is `oof_blend159av_h3.npy` -- w16l reproduced that at rank
correlation 1.00000000 and this script re-asserts the pooled delta against w16l's published
-3.432e-6 before reading any cell.

WHAT THIS CAN AND CANNOT CONCLUDE, fixed before the run
--------------------------------------------------------
- A cell whose delta is large and positive while the pooled number is negative is a LEAD, not
  a result: within-cell AUC on 40k-175k rows is noisy and seven cells is seven looks. The
  per-fold spread is printed so the reader can see the error, and for item B a size-matched
  PERMUTED-MEMBERSHIP control partition is run so "seven looks at noise" has a floor.
- Item B's `imp - unw` confounds the directional component with the ESS toll. w16l separated
  those with the `anti` arm and `anti` is not on disk. So a positive cell here does NOT
  establish a directional effect; it establishes that the pooled null is not uniform, which is
  the only thing this script claims and the only thing needed to justify a refit.
- Neither item can produce a submission file. This script writes no CSV.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from w16a_where import cond_auc_seg  # noqa: E402
from w16b_cellweight import fast_auc, pct, rule_cells  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
CTRL_SEED = 20260816


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    cell = rule_cells(tr)
    cells = sorted(set(cell))
    base = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base)
    out = {}

    # ================================================================= ITEM A
    print("=" * 84)
    print("ITEM A  w15f 6(a): the transductive component, segmented by rule cell")
    print("=" * 84)
    c_tr = np.load(os.path.join(HERE, "w15f_c_trans.npy")).astype(np.float64)
    c_in = np.load(os.path.join(HERE, "w15f_c_induc.npy")).astype(np.float64)
    c_avg = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    d_trans = c_tr - c_in
    print(f"  c_trans - c_induc   sd {d_trans.std():.6f}   "
          f"rank corr(c_trans, c_induc) "
          f"{np.corrcoef(rankdata(c_tr), rankdata(c_in))[0,1]:+.6f}")

    for tag, vec in (("c_avg  [POSITIVE CONTROL, must match w16a]", c_avg),
                     ("c_trans - c_induc  [the null under test]", d_trans)):
        print(f"\n  --- {tag}")
        allm = np.ones(n, bool)
        r, cm, cs, _ = cond_auc_seg(br, vec, y, allm)
        z_glob = (r - cm) / cs
        print(f"  {'GLOBAL':<22} n {n:>7,}  cond {r:.6f}  ctrl {cm:.6f} +- {cs:.5f}  "
              f"z {z_glob:+6.2f}")
        rows = {}
        for cl in cells:
            m = cell == cl
            r, cm, cs, _ = cond_auc_seg(br, vec, y, m)
            z = (r - cm) / cs
            rows[cl] = dict(n=int(m.sum()), cond=float(r), ctrl=float(cm),
                            ctrl_sd=float(cs), z=float(z))
            print(f"  {cl:<22} n {int(m.sum()):>7,}  cond {r:.6f}  ctrl {cm:.6f} +- {cs:.5f}"
                  f"  z {z:+6.2f}")
        zs = np.array([rows[c]["z"] for c in cells])
        print(f"  chi2 against a uniform-zero effect (sum z^2) = {float((zs**2).sum()):.2f} "
              f"on {len(cells)} df   max |z| {np.abs(zs).max():.2f}")
        out["A_" + ("c_avg" if "c_avg" in tag else "trans")] = dict(
            z_global=float(z_glob), cells=rows, chi2=float((zs ** 2).sum()))

    # ================================================================= ITEM B
    print("\n" + "=" * 84)
    print("ITEM B  w16l 2: the mask training weight, segmented by rule cell")
    print("=" * 84)
    imp = np.load(os.path.join(SUB, "oof_w16l_maskw_h3.npy")).astype(np.float64)
    unw = base
    pooled = (fast_auc(y, imp) - fast_auc(y, unw)) * 1e6
    print(f"  pooled imp - unw = {pooled:+.3f}e-6   "
          f"(w16l published -3.432e-6 for the h3 object)")
    assert abs(pooled - (-3.432)) < 0.35, ("harness drift", pooled)
    print("  harness gate PASSED: the pooled delta reproduces w16l's published number.\n")

    def per_cell(assign, levels, label):
        res = {}
        for lv in levels:
            m = assign == lv
            d = []
            for _, iva in folds:
                mm = m[iva]
                if mm.sum() < 500 or y[iva][mm].sum() < 25 or (1 - y[iva][mm]).sum() < 25:
                    continue
                d.append(fast_auc(y[iva][mm], imp[iva][mm])
                         - fast_auc(y[iva][mm], unw[iva][mm]))
            d = np.array(d)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            res[str(lv)] = dict(n=int(m.sum()), mean=float(d.mean() * 1e6),
                                se=float(se * 1e6), pos=int((d > 0).sum()), k=len(d))
            print(f"  {label} {str(lv):<22} n {int(m.sum()):>7,}  "
                  f"imp-unw {d.mean()*1e6:+8.2f}e-6  se {se*1e6:6.2f}  "
                  f"{int((d > 0).sum())}/{len(d)}")
        return res

    print("  real partition (generator rule cells):")
    real = per_cell(cell, cells, "  ")
    rng = np.random.default_rng(CTRL_SEED)
    perm = cell[rng.permutation(n)]
    print("\n  CONTROL, size-matched permuted membership:")
    ctrl = per_cell(perm, cells, "  ")

    rm = np.array([real[c]["mean"] for c in cells])
    cm = np.array([ctrl[c]["mean"] for c in cells])
    print(f"\n  spread across levels   real sd {rm.std(ddof=1):.2f}e-6   "
          f"control sd {cm.std(ddof=1):.2f}e-6")
    print(f"  max cell               real {rm.max():+.2f}e-6 ({cells[int(rm.argmax())]})   "
          f"control {cm.max():+.2f}e-6")
    print(f"  min cell               real {rm.min():+.2f}e-6 ({cells[int(rm.argmin())]})   "
          f"control {cm.min():+.2f}e-6")
    out["B"] = dict(pooled=float(pooled), real=real, ctrl=ctrl,
                    real_sd=float(rm.std(ddof=1)), ctrl_sd=float(cm.std(ddof=1)))

    json.dump(out, open(os.path.join(HERE, "w16r_pooledsweep.json"), "w"), indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
