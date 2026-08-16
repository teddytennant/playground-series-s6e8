"""w15f stage 10: does the verdict survive the student hyperparameter I had to guess?

The author's prose gives the student one adjective, "smoother", and no numbers. Stage 5 shows
my teacher landed on his (+0.99529) but my student did not: my residual sd is 0.0464 against
his 0.0217, so my student is about twice as smooth as his and our corrections share only rank
corr +0.517. That is the single biggest threat to this run's conclusion.

Stage 7 fits three students spanning sd 0.048 / 0.031 / 0.020, which BRACKETS his 0.0217.
This scores all three the same way stage 6 scored the first, so the question "would a student
configured like his have changed the answer" gets a measurement rather than a caveat.

If the verdict is flat across the bracket, the conclusion does not depend on the guess.
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
from w15f_eval import cond_auc, cond_auc_ctrl, pct  # noqa: E402
from w15f_extract import perm_within_fold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_CTRL = 120
RNG = np.random.default_rng(99)
GRID = np.linspace(0.0, 0.02, 101)


def main():
    z = np.load(os.path.join(HERE, "w15f_nested2.npz"))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    fold_id, teacher_r = z["fold_id"], z["teacher_r"]
    folds = get_folds(y)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    base_auc = roc_auc_score(y, base_p)
    br = pct(base_p)
    brr = rankdata(base_p) / n

    print(f"base {BASE} CV {base_auc:.7f}   (author's residual sd 0.0217)")
    print(f"{'student':>8} {'sd(r)':>7} {'add@0.10 net':>13} {'z':>6} "
          f"{'condAUC':>9} {'z':>6} {'xfit rankblend':>15} {'f+':>4}")
    out = {}
    for nm in ("smooth", "mid", "rough"):
        r = teacher_r - z[f"student_{nm}"]
        c = np.zeros(n)
        for k in np.unique(fold_id):
            m = fold_id == k
            rr = r[m]
            sq = np.sign(rr) * np.abs(rr) ** 2
            c[m] = (sq - sq.mean()) * (rr.std() / sq.std())

        real = roc_auc_score(y, br + 0.10 * c)
        ctrl = np.array([roc_auc_score(y, br + 0.10 * perm_within_fold(c, fold_id, RNG))
                         for _ in range(N_CTRL)])
        net, sd = real - ctrl.mean(), ctrl.std()

        ca = cond_auc(base_p, c, y)
        cc = cond_auc_ctrl(base_p, c, y)

        mr = rankdata(r) / n
        deltas = []
        for itr, iva in folds:
            a = np.array([roc_auc_score(y[itr], (1 - w) * brr[itr] + w * mr[itr])
                          for w in GRID])
            wk = GRID[int(a.argmax())]
            deltas.append(roc_auc_score(y[iva], (1 - wk) * brr[iva] + wk * mr[iva])
                          - roc_auc_score(y[iva], brr[iva]))
        deltas = np.array(deltas)

        print(f"{nm:>8} {r.std():>7.4f} {net:>+13.3e} {net/sd:>+6.2f} "
              f"{ca:>9.6f} {(ca-cc.mean())/cc.std():>+6.2f} "
              f"{deltas.mean():>+15.3e} {int((deltas>0).sum())}/5")
        out[nm] = dict(resid_sd=float(r.std()), add_net=float(net),
                       add_z=float(net / sd), cond_auc=float(ca),
                       cond_z=float((ca - cc.mean()) / cc.std()),
                       xfit=float(deltas.mean()),
                       folds_pos=int((deltas > 0).sum()))

    json.dump(out, open(os.path.join(HERE, "w15f_sens.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'w15f_sens.json')}")


if __name__ == "__main__":
    main()
