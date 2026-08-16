"""w15f stage 6: stage 3 left a tension. Resolve it.

Stage 3 measured two things that do not obviously agree:

  * the author's construction -- base rank + 0.10 * sign(r)|r|^2 -- is a NULL on the frozen
    folds. Real minus the matched permuted control is -3.6e-6 +- 7.1e-6, and the per-fold
    signs are 0/5 against his 60/60.
  * but c is not noise. Conditional on the base score (200 quantile bins, 24-seed
    within-bin permutation control) it orders the label at 0.503745 against a control at
    0.500065 +- 0.00122, z = +3.02.

Both can be true: the additive signed-square is a very badly conditioned way to spend a weak
signal. Its range is [-1.45, +0.80] in percentile-rank units, so a handful of rows are thrown
the entire length of the ranking while the rest barely move -- which is exactly why the toll
is -5.5e-5 here against w15e's -1.1e-5 for a perturbation 2.2x smaller (and 2.2^2 x 1.1e-5 =
5.4e-5, so the toll model replicates).

So the question stage 3 cannot answer is: IS THERE EXTRACTABLE SIGNAL IN THE RESIDUAL AT ALL,
in the best-conditioned form available? Three arms:

 A. weight curve of the author's additive construction, real minus control at each weight.
    Reported whole. The curve is a diagnostic, not a menu -- the headline stays at his 0.10.
 B. the same for the RAW residual r (no signed square) at matched perturbation size. Note
    that sign(r)|r|^2 is monotone in r, so the two have IDENTICAL rank orderings; they differ
    only in how the additive move is distributed. This isolates the signed square itself.
 C. rank(r) as an ordinary MEMBER in a one-parameter rank blend against the base, weight
    searched on a 181-point grid, in-sample AND cross-fitted on the frozen folds. This is
    w15d's perfectly-conditioned instrument: one parameter, not 159, so a zero here cannot be
    blamed on the stack's 1e18 condition number. In-sample is the load-bearing arm because it
    is free to overfit TOWARD a useful member and therefore cannot penalise one.

 D. the purely transductive component, c_trans - c_induc. If carrying the unlabeled rows adds
    anything, it lives here and nowhere else. Nobody has separated this before.

Controls: permuted (same values, rows shuffled within fold) and uniform noise, as in w15d.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15f_eval import cond_auc, cond_auc_ctrl, make_correction, pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_CTRL = 120
RNG = np.random.default_rng(815)


def perm_within_fold(v, fold_id, rng):
    out = np.empty_like(v)
    for k in np.unique(fold_id):
        m = np.flatnonzero(fold_id == k)
        out[m] = v[rng.permutation(m)]
    return out


def main():
    z = np.load(os.path.join(HERE, "w15f_nested.npz"))
    teacher_r, fold_id = z["teacher_r"], z["fold_id"]
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    base_rank = pct(base_p)
    base_auc = roc_auc_score(y, base_p)
    folds = get_folds(y)
    out = {"base_auc": float(base_auc)}

    c = np.load(os.path.join(HERE, "w15f_c_trans.npy"))
    ci = np.load(os.path.join(HERE, "w15f_c_induc.npy"))
    # raw residual, per fold, on the same footing as make_correction's output
    r = np.zeros(n)
    for k in np.unique(fold_id):
        m = fold_id == k
        r[m] = teacher_r[m] - z["student_t"][m]

    # ---------- A / B: weight curves, each against its own permuted control ----------
    print("=== A/B: additive weight curves, real minus matched permuted control ===")
    print(f"{'vector':>8} {'weight':>7} {'sd(move)':>9} {'real-base':>11} "
          f"{'toll':>11} {'REAL-CTRL':>11} {'z':>6}")
    curves = {}
    for tag, v in [("c(sq)", c), ("r(raw)", r)]:
        rows = []
        for w in (0.01, 0.02, 0.05, 0.10, 0.20):
            real = roc_auc_score(y, base_rank + w * v)
            ctrl = np.array([roc_auc_score(y, base_rank + w * perm_within_fold(v, fold_id, RNG))
                             for _ in range(N_CTRL)])
            net, sd = real - ctrl.mean(), ctrl.std()
            rows.append(dict(w=w, real=float(real), ctrl=float(ctrl.mean()),
                             net=float(net), z=float(net / sd)))
            print(f"{tag:>8} {w:>7.2f} {w*v.std():>9.5f} {real-base_auc:>+11.2e} "
                  f"{ctrl.mean()-base_auc:>+11.2e} {net:>+11.3e} {net/sd:>+6.2f}")
        curves[tag] = rows
    out["curves"] = curves

    # ---------- C: rank(r) as a member, one-parameter blend, w15d's instrument -------
    print("\n=== C: one-parameter rank blend, base vs member (w15d instrument) ===")
    print(f"{'member':>14} {'in-sample w*':>13} {'gain':>11} {'xfit w':>7} "
          f"{'xfit dAUC':>11} {'folds+':>7}")
    grid = np.linspace(0.0, 0.18, 181)
    br = rankdata(base_p) / n
    members = {
        "resid_trans": r,
        "resid_induc": teacher_r - z["student_i"],
        "trans_minus_induc": c - ci,
        "PERM ctrl": perm_within_fold(r, fold_id, np.random.default_rng(7)),
        "NOISE ctrl": np.random.default_rng(11).uniform(size=n),
    }
    memres = {}
    for nm, v in members.items():
        mr = rankdata(v) / n
        aucs = np.array([roc_auc_score(y, (1 - w) * br + w * mr) for w in grid])
        i = int(aucs.argmax())
        # cross-fitted: weight chosen on the other four folds, scored on the held-out one
        xf, deltas = [], []
        for k, (itr, iva) in enumerate(folds):
            a = np.array([roc_auc_score(y[itr], (1 - w) * br[itr] + w * mr[itr])
                          for w in grid])
            wk = grid[int(a.argmax())]
            xf.append(wk)
            deltas.append(roc_auc_score(y[iva], (1 - wk) * br[iva] + wk * mr[iva])
                          - roc_auc_score(y[iva], br[iva]))
        deltas = np.array(deltas)
        print(f"{nm:>14} {grid[i]:>13.3f} {aucs[i]-aucs[0]:>+11.2e} "
              f"{np.mean(xf):>7.3f} {deltas.mean():>+11.2e} "
              f"{int((deltas>0).sum())}/5")
        memres[nm] = dict(w_star=float(grid[i]), gain=float(aucs[i] - aucs[0]),
                          xfit_w=float(np.mean(xf)), xfit=float(deltas.mean()),
                          folds_pos=int((deltas > 0).sum()))
    out["members"] = memres

    # ---------- D: is the transductive part worth anything on its own? --------------
    print("\n=== D: the purely transductive component (trans minus induc) ===")
    d = c - ci
    ca = cond_auc(base_p, d, y)
    cc = cond_auc_ctrl(base_p, d, y)
    print(f"  corr(c_trans, c_induc) = {np.corrcoef(c, ci)[0,1]:+.5f}")
    print(f"  cond AUC(trans-induc | base) = {ca:.6f}  control {cc.mean():.6f} "
          f"sd {cc.std():.2e}  z = {(ca-cc.mean())/cc.std():+.2f}")
    out["transductive_only"] = dict(cond_auc=float(ca), ctrl=float(cc.mean()),
                                    z=float((ca - cc.mean()) / cc.std()))

    json.dump(out, open(os.path.join(HERE, "w15f_extract.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'w15f_extract.json')}")


if __name__ == "__main__":
    main()
