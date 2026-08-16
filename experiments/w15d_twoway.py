"""Does `orig_binm` add anything to the pack OUTSIDE the singular stacker?

Background. `orig_binm` is a LightGBM fitted on the 7,500-row real source dataset only,
with the competition's own per-column missingness MCAR-masked in (RESEARCH.md, route 1).
Solo OOF AUC 0.8864, maxcorr to the pack 0.879 -- by a wide margin the most decorrelated
object this workspace holds. Inside the 160-member logistic stack it is worth -1e-6..-2e-6
and exactly 0 under h3.

w14a then measured that stack's design matrix at condition number ~1e18 -- numerically
singular, reproducibility floor ~2e-6. So the recorded "zero" is confounded: a solution
that is not determined below 2e-6 cannot report a contribution of 2e-6 either way.

This script asks the same question with a perfectly-conditioned instrument: ONE parameter.

    z(w) = (1 - w) * rank(pack) + w * rank(orig_binm)

evaluated on the frozen folds, with the weight searched (a) in-sample and (b) cross-fitted
(searched on 4 folds, scored on the held-out one). Controls: the identical search run
against a row-permuted copy of the member and against uniform noise, which price the
optimism of searching one parameter on 691k rows.

    .venv/bin/python experiments/w15d_twoway.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, SUB, OOF, TARGET, get_folds  # noqa: E402

RNG = np.random.default_rng(20260815)
PACKS = ["blend159av_h3", "blendtop3", "blend158_h3", "blend159av_rankraw"]
MEMBERS = ["orig_binm", "orig_bin"]


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def auc(y, s):
    return roc_auc_score(y, s)


def search_w(y, a, b, grid):
    """Best w for (1-w)*a + w*b on the rows given, plus the whole curve."""
    scores = np.array([auc(y, (1.0 - w) * a + w * b) for w in grid])
    i = int(np.argmax(scores))
    return grid[i], scores[i], scores


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    grid = np.round(np.concatenate([np.arange(0.0, 0.201, 0.002),
                                    np.arange(0.22, 1.001, 0.02)]), 4)

    packs = {p: rk(np.load(os.path.join(SUB, f"oof_{p}.npy"))) for p in PACKS}
    mems = {m: rk(np.load(os.path.join(OOF, f"oof_{m}.npy"))) for m in MEMBERS}
    # controls: same marginal distribution, no signal / no structure at all
    mems["PERM_binm"] = mems["orig_binm"][RNG.permutation(n)]
    mems["NOISE"] = rk(RNG.random(n))

    print(f"n={n:,}  base rate {y.mean():.6f}\n")
    for m, v in mems.items():
        print(f"  member {m:12s} solo AUC {auc(y, v):.6f}   "
              f"corr(spearman) to blend159av_h3 {np.corrcoef(v, packs['blend159av_h3'])[0,1]:+.4f}")
    print()

    rows = []
    for p, pv in packs.items():
        base = auc(y, pv)
        print(f"=== pack {p}   OOF AUC {base:.6f} ===")
        for m, mv in mems.items():
            w_in, s_in, curve = search_w(y, pv, mv, grid)

            # cross-fitted: weight chosen without seeing the fold it is scored on
            fold_d, fold_w = [], []
            for tr_i, va_i in folds:
                w_cf, _, _ = search_w(y[tr_i], pv[tr_i], mv[tr_i], grid)
                blend_va = (1.0 - w_cf) * pv[va_i] + w_cf * mv[va_i]
                fold_d.append(auc(y[va_i], blend_va) - auc(y[va_i], pv[va_i]))
                fold_w.append(w_cf)
            fold_d = np.array(fold_d)

            rows.append(dict(pack=p, member=m, base=base, w_in=w_in,
                             gain_in=s_in - base, cf_mean=fold_d.mean(),
                             cf_sd=fold_d.std(ddof=1) / np.sqrt(len(fold_d)),
                             cf_pos=int((fold_d > 0).sum()), w_cf=float(np.mean(fold_w))))
            print(f"  {m:12s} in-sample w*={w_in:.3f} gain {(s_in-base)*1e6:+8.2f}e-6   |   "
                  f"cross-fitted w={np.mean(fold_w):.3f} dAUC {fold_d.mean()*1e6:+8.2f}e-6 "
                  f"+-{fold_d.std(ddof=1)/np.sqrt(len(fold_d))*1e6:.2f}  ({int((fold_d>0).sum())}/5 positive)")
        print()

    df = pd.DataFrame(rows)
    out = os.path.join("experiments", "w15d_twoway.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    # --- two-parameter logistic on the same pair, cross-fitted: does a fitted scale help?
    print("\n=== 2-parameter logistic  y ~ logit(pack) + rank(member), cross-fitted ===")
    eps = 1e-9
    for p in ("blend159av_h3",):
        pv = packs[p]
        lg = np.log(np.clip(pv, eps, 1 - eps) / (1 - np.clip(pv, eps, 1 - eps)))
        for m in ("orig_binm", "PERM_binm"):
            mv = mems[m]
            X = np.column_stack([lg, mv])
            oof = np.zeros(len(y))
            for tr_i, va_i in folds:
                clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=3)
                clf.fit(X[tr_i], y[tr_i])
                oof[va_i] = clf.decision_function(X[va_i])
            print(f"  {p} + {m:12s}: {auc(y, oof):.6f}  vs pack {auc(y, pv):.6f}  "
                  f"delta {(auc(y, oof)-auc(y, pv))*1e6:+.2f}e-6")


if __name__ == "__main__":
    main()
