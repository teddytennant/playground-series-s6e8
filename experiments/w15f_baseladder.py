"""w15f stage 8: reconcile the author's 60/60 with our 0/5 instead of just contradicting it.

Stage 3 found his construction is a null on our folds. He reports it positive in 60 of 60
anchor-by-fold comparisons and he is plainly careful -- he states the leaderboard chose
nothing and reports OOF numbers with per-fold signs. Two careful measurements disagreeing is
a fact that wants an explanation, not a winner.

The obvious candidate is BASE STRENGTH. His four anchors sit at OOF AUC 0.969667-0.969721.
Ours is 0.970049. A correction carrying a fixed small amount of information adds more to a
weaker ranking, because the stronger ranking has already captured more of what the correction
knows. If the gain falls monotonically as the base rises and lands near his value at his
anchor's strength, both measurements are right and the disagreement dissolves.

This is cheap and completely determined: every vector already exists on disk. The base ladder
spans 0.969641 to 0.970049, and `stack_pub74_logit` is very nearly his own anchor object -- a
plain logistic stack over the same public 74-model library on the same frozen folds.

Both instruments at every rung: his additive construction at his published 0.10 against a
matched permuted control, and the perfectly-conditioned one-parameter rank blend.
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
from w15f_eval import pct  # noqa: E402
from w15f_extract import perm_within_fold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_CTRL = 120
RNG = np.random.default_rng(1508)

# spans his anchor level (~0.9697) up to ours (0.970049)
BASES = ["stack_pub74_logit", "stack_pub86_hybrid", "blend158_logit",
         "blend158_hybrid", "blend159av_h3"]


def main():
    z = np.load(os.path.join(HERE, "w15f_nested.npz"))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    fold_id = z["fold_id"]
    folds = get_folds(y)

    c = np.load(os.path.join(HERE, "w15f_c_trans.npy"))
    r = z["teacher_r"] - z["student_t"]
    mr = rankdata(r) / n
    grid = np.linspace(0.0, 0.18, 181)

    print(f"{'base':>20} {'base AUC':>10} | {'add@0.10':>10} {'toll':>10} "
          f"{'REAL-CTRL':>11} {'z':>6} | {'w*':>6} {'in-samp':>10} {'xfit':>10} {'f+':>4}")
    out = {}
    for nm in BASES:
        bp = np.load(os.path.join(SUB, f"oof_{nm}.npy")).astype(np.float64)
        ba = roc_auc_score(y, bp)
        br = pct(bp)

        real = roc_auc_score(y, br + 0.10 * c)
        ctrl = np.array([roc_auc_score(y, br + 0.10 * perm_within_fold(c, fold_id, RNG))
                         for _ in range(N_CTRL)])
        net, sd = real - ctrl.mean(), ctrl.std()

        brr = rankdata(bp) / n
        aucs = np.array([roc_auc_score(y, (1 - w) * brr + w * mr) for w in grid])
        i = int(aucs.argmax())
        deltas = []
        for itr, iva in folds:
            a = np.array([roc_auc_score(y[itr], (1 - w) * brr[itr] + w * mr[itr])
                          for w in grid])
            wk = grid[int(a.argmax())]
            deltas.append(roc_auc_score(y[iva], (1 - wk) * brr[iva] + wk * mr[iva])
                          - roc_auc_score(y[iva], brr[iva]))
        deltas = np.array(deltas)

        print(f"{nm:>20} {ba:>10.6f} | {real-ba:>+10.2e} {ctrl.mean()-ba:>+10.2e} "
              f"{net:>+11.3e} {net/sd:>+6.2f} | {grid[i]:>6.3f} "
              f"{aucs[i]-aucs[0]:>+10.2e} {deltas.mean():>+10.2e} "
              f"{int((deltas>0).sum())}/5")
        out[nm] = dict(base_auc=float(ba), add_net=float(net), add_z=float(net / sd),
                       w_star=float(grid[i]), insample=float(aucs[i] - aucs[0]),
                       xfit=float(deltas.mean()), folds_pos=int((deltas > 0).sum()))

    json.dump(out, open(os.path.join(HERE, "w15f_baseladder.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(HERE, 'w15f_baseladder.json')}")


if __name__ == "__main__":
    main()
