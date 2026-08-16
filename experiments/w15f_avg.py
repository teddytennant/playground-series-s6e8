"""w15f stage 12: the final candidate, with the guessed hyperparameter averaged out.

WHY AN AVERAGE AND NOT THE BEST STUDENT
---------------------------------------
Stage 10 measures three students. The temptation is to carry forward the one that scores
best. That is the move this workspace exists to not make, and here it would be especially
cheap because the two fidelity criteria available DISAGREE about which student is the
faithful reconstruction:

    student   resid sd   (his 0.01961)   rank corr vs HIS correction
    smooth      0.04641                            +0.51669
    mid         0.03043                            +0.53824
    rough       0.01986                            +0.33825

`rough` matches his residual scale to 1.3% and is the worst match in shape; `mid` is the best
match in shape and is 55% too large in scale. Neither criterion sees the label, both are
legitimate, and they point at different students. There is no honest way to call one of them
"the" reconstruction.

So the hyperparameter is averaged out instead of chosen. Each student's correction is
standardised to unit sd within its fold and the three are equally averaged -- zero fitted
parameters, exactly the reasoning behind this workspace's own `h3` rule. The additive weight
is then chosen CROSS-FITTED (searched on four folds, scored on the fifth), so the one number
that is fitted is fitted honestly.

The base is `blend159av_h3`, our joint-best CV file and a deadline pick, chosen on CV and
never on the leaderboard.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15f_eval import cond_auc, cond_auc_ctrl, pct  # noqa: E402
from w15f_extract import perm_within_fold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
FALLBACK = "blend159"
N_TEST = 296_302
N_CTRL = 200
ARMS = ("smooth", "mid", "rough")
# c_avg is standardised to unit sd, so the weight IS the perturbation size in percentile
# units. The author's published setting is weight 0.10 on a correction of sd 0.02171, i.e.
# a perturbation of 0.00217 -- so his construction lives at w ~ 0.0022 on this scale and the
# grid has to resolve that. The first version of this file swept 0..0.30 in steps of 0.002
# and could only choose between "nothing" and "ten times too much"; the per-fold weights came
# out quantised to {0, 0.002} and the selection noise swamped the effect.
GRID = np.linspace(0.0, 0.02, 201)
RNG = np.random.default_rng(1234)


def signed_sq(r):
    sq = np.sign(r) * np.abs(r) ** 2
    return (sq - sq.mean()) * (r.std() / sq.std())


def main():
    z2 = np.load(os.path.join(HERE, "w15f_nested2.npz"))
    teacher_r, fold_id = z2["teacher_r"], z2["fold_id"]
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    base_auc = roc_auc_score(y, base_p)
    fb_auc = roc_auc_score(y, np.load(os.path.join(SUB, f"oof_{FALLBACK}.npy")))
    br = pct(base_p)

    # ---- the equal average, standardised within fold, zero fitted parameters ----
    c_avg = np.zeros(n)
    for k in np.unique(fold_id):
        m = fold_id == k
        acc = np.zeros(int(m.sum()))
        for nm in ARMS:
            c = signed_sq(teacher_r[m] - z2[f"student_{nm}"][m])
            acc += c / c.std()
        acc /= len(ARMS)
        c_avg[m] = acc / acc.std()          # unit sd within fold; weight carries the scale
    np.save(os.path.join(HERE, "w15f_c_avg.npy"), c_avg)

    print(f"base     {BASE:>16} cross-fitted CV {base_auc:.7f}")
    print(f"fallback {FALLBACK:>16} cross-fitted CV {fb_auc:.7f}")
    print(f"c_avg: sd {c_avg.std():.4f}  solo AUC {roc_auc_score(y, c_avg):.6f}\n")

    ca = cond_auc(base_p, c_avg, y)
    cc = cond_auc_ctrl(base_p, c_avg, y)
    print(f"cond AUC(c_avg | base) = {ca:.6f}  control {cc.mean():.6f} "
          f"sd {cc.std():.2e}  z = {(ca-cc.mean())/cc.std():+.2f}")

    # ---- cross-fitted additive weight, then the matched permuted control at it ----
    ws, deltas = [], []
    for itr, iva in folds:
        a = np.array([roc_auc_score(y[itr], br[itr] + w * c_avg[itr]) for w in GRID])
        wk = GRID[int(a.argmax())]
        ws.append(wk)
        deltas.append(roc_auc_score(y[iva], br[iva] + wk * c_avg[iva])
                      - roc_auc_score(y[iva], br[iva]))
    deltas = np.array(deltas)
    full = np.array([roc_auc_score(y, br + w * c_avg) for w in GRID])
    w_full = GRID[int(full.argmax())]
    cand_cv = base_auc + deltas.mean()

    print(f"\ncross-fitted weight per fold {np.round(ws, 4)}   full-data w* {w_full:.4f}")
    print(f"cross-fitted dAUC {deltas.mean():+.3e}  "
          f"({' '.join(f'{d:+.1e}' for d in deltas)})  {int((deltas>0).sum())}/5 folds")
    print(f"candidate CV {cand_cv:.7f}   vs fallback {cand_cv-fb_auc:+.3e}   "
          f"vs base {deltas.mean():+.3e}")

    real = roc_auc_score(y, br + w_full * c_avg)
    ctrl = np.array([roc_auc_score(y, br + w_full * perm_within_fold(c_avg, fold_id, RNG))
                     for _ in range(N_CTRL)])
    net, sd = real - ctrl.mean(), ctrl.std()
    print(f"at w*={w_full:.4f}: real {real-base_auc:+.2e}  toll {ctrl.mean()-base_auc:+.2e}"
          f"  REAL-CTRL {net:+.3e}  z {net/sd:+.2f}")

    out = dict(base_auc=float(base_auc), fallback_auc=float(fb_auc),
               cond_auc=float(ca), cond_z=float((ca - cc.mean()) / cc.std()),
               xfit_w=[float(x) for x in ws], w_full=float(w_full),
               xfit=float(deltas.mean()), folds_pos=int((deltas > 0).sum()),
               candidate_cv=float(cand_cv), net=float(net), net_z=float(net / sd))

    # ---------------------------- the test file ---------------------------------
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    assert (test["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)

    acc = np.zeros(N_TEST)
    for nm, f in [("smooth", "w15f_c_test.npy"), ("mid", "w15f_c_test_mid.npy"),
                  ("rough", "w15f_c_test_rough.npy")]:
        c = np.load(os.path.join(HERE, f))
        assert c.shape == (N_TEST,)
        acc += c / c.std()
    acc /= len(ARMS)
    ct = acc / acc.std()
    np.save(os.path.join(HERE, "w15f_c_test_avg.npy"), ct)

    pred = btr + w_full * ct
    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})

    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w15f_antistudent_avg.csv")
    sub.to_csv(path, index=False)
    rho = float(np.corrcoef(rankdata(strict), rankdata(bp))[0, 1])
    print(f"\nwrote {path}  rows {len(sub):,}  spearman vs base {rho:.7f}")
    out["file"] = dict(path=path, spearman_vs_base=rho)
    json.dump(out, open(os.path.join(HERE, "w15f_avg.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
