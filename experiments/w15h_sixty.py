"""w15h: what is "positive in 60 of 60 anchor-by-fold comparisons" actually worth?

THE CLAIM UNDER ATTACK
----------------------
raykkretzschmar's anti-student correction is accepted in w15e §3 partly on the strength of
his stated evidence: adding the same 10% signed-square residual to four independent OOF
anchors moved them +0.000018 / +0.000019 / +0.000036 / +0.000025, "positive in 60 of 60
anchor-by-fold comparisons". Read naively that is a sign test at p = 2^-60.

It is not, and the reason is structural rather than statistical sloppiness on his part:

  * the four anchors are stacks over overlapping member pools, mutually correlated ~0.999;
  * the 15 folds per anchor are 5 folds x 3 meta-CV partitions of the SAME 691,369 rows,
    so each partition is a re-tiling of one dataset rather than a fresh sample;
  * all 60 comparisons reuse ONE correction vector.

So the 60 numbers are 60 views of one measurement. This script prices exactly how many
independent bits they carry, using our own anchors (which have the same correlation
structure) and matched-noise corrections, and it does it two ways:

  1. the empirical correlation matrix of the 60 statistics across draws, reduced to an
     effective number of independent tests;
  2. the distribution of "how many of the 60 are positive" under matched noise.

NOTHING HERE IS TRAINED. Pure numpy over stored OOF vectors.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
It is an audit of the STRENGTH OF EVIDENCE, not of the effect. If the null for this move
is displaced negative -- w15e §6 measured a signal-free correction of this size at
-1.1e-5 -- then an observed POSITIVE point estimate is still informative even when the 60
counts collapse to ~1 bit. The honest reading has to combine both, and this script
supplies the pieces for it.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET  # noqa: E402

NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
OUT = os.path.join(ROOT, "experiments", "w15h_sixty.json")

# Four anchors of the same kind and mutual correlation as rayk's four: different stacks
# over an overlapping member pool, all within 1e-4 of each other.
ANCHORS = ["blend158_logit", "blend153", "blend159av_rankraw", "blend159av_h3"]
PARTITION_SEEDS = [42, 2026, 314159]     # rayk's own meta-CV seeds
N_DRAWS = 300
W_ANTI = 0.10


def pct(v):
    v = np.asarray(v, np.float64)
    return rankdata(v) / len(v)


def build_anti(residual, reference):
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    reference_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    return (test_sq - reference_sq.mean()) * reference.std() / reference_sq.std()


def main():
    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"))[TARGET].astype(int).to_numpy()
    n = len(y)

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)
    resid = pct(z["test_teacher"].astype(np.float64)) - pct(z["test_student"].astype(np.float64))
    anti_pub = build_anti(resid, reference)

    # A matched-marginal correction on 691,369 rows: resample the published correction's
    # own values. Same sd, same heavy tails, zero information about y by construction.
    rng = np.random.default_rng(20260815)

    bases, base_pct = {}, {}
    for a in ANCHORS:
        v = np.load(os.path.join(SUB, f"oof_{a}.npy"))
        bases[a] = v
        base_pct[a] = pct(v)
        print(f"anchor {a:22s} OOF {roc_auc_score(y, v):.6f}", flush=True)

    # mutual correlation of the anchors, to show they match rayk's ~0.999 setting
    zs = {a: (lambda r: (r - r.mean()) / r.std())(rankdata(bases[a])) for a in ANCHORS}
    cors = [float(zs[ANCHORS[i]] @ zs[ANCHORS[j]] / n)
            for i in range(len(ANCHORS)) for j in range(i + 1, len(ANCHORS))]
    print(f"anchor mutual spearman: min {min(cors):.5f} max {max(cors):.5f}", flush=True)

    # the 60 (anchor, partition, fold) cells
    cells = []
    for a in ANCHORS:
        for s in PARTITION_SEEDS:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            for _, va in skf.split(np.zeros(n), y):
                cells.append((a, s, va))
    print(f"{len(cells)} anchor-by-fold cells", flush=True)

    base_auc = {}
    for i, (a, s, va) in enumerate(cells):
        base_auc[i] = roc_auc_score(y[va], base_pct[a][va])

    D = np.empty((N_DRAWS, len(cells)))
    for d in range(N_DRAWS):
        corr_vec = rng.choice(anti_pub, size=n, replace=True)
        for i, (a, s, va) in enumerate(cells):
            p = base_pct[a][va] + W_ANTI * corr_vec[va]
            D[d, i] = roc_auc_score(y[va], p) - base_auc[i]
        if (d + 1) % 25 == 0:
            print(f"  draw {d+1}/{N_DRAWS}  {time.time()-t0:.0f}s", flush=True)

    pos = (D > 0).sum(1)
    C = np.corrcoef(D.T)
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    n_eff = float(ev.sum() ** 2 / (ev ** 2).sum())          # participation ratio

    res = dict(
        n_draws=N_DRAWS, n_cells=len(cells),
        anchor_corr_min=min(cors), anchor_corr_max=max(cors),
        mean_pairwise_cell_corr=float((C.sum() - len(cells)) / (len(cells) ** 2 - len(cells))),
        n_eff_participation=n_eff,
        top_eigenvalue_share=float(ev.max() / ev.sum()),
        delta_mean=float(D.mean()), delta_sd_over_draws=float(D.mean(1).std(ddof=1)),
        pos_count_mean=float(pos.mean()), pos_count_sd=float(pos.std(ddof=1)),
        p_all60_positive=float((pos == len(cells)).mean()),
        p_all60_negative=float((pos == 0).mean()),
        p_extreme=float(((pos == 0) | (pos == len(cells))).mean()),
        pos_count_hist={str(int(k)): int(v) for k, v in
                        zip(*np.unique(pos, return_counts=True))},
    )
    print("\n=== result ===")
    print(f"  matched-noise delta, mean over all cells : {res['delta_mean']*1e6:+.3f}e-6")
    print(f"  sd of the FULL-DATA delta across draws   : {res['delta_sd_over_draws']*1e6:.3f}e-6")
    print(f"  mean pairwise correlation of the 60 cells: {res['mean_pairwise_cell_corr']:.4f}")
    print(f"  effective independent tests (participation ratio of 60): {n_eff:.2f}")
    print(f"  first eigenvalue carries {res['top_eigenvalue_share']*100:.1f}% of the variance")
    print(f"  P(all 60 same sign) under matched noise  : {res['p_extreme']:.3f}"
          f"   (all+ {res['p_all60_positive']:.3f}, all- {res['p_all60_negative']:.3f})")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
