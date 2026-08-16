"""w15g -- is the coupling-free criterion a BETTER criterion?

w15g_coupling.py measures, exactly, how much of a member's pooled OOF AUC comes from
cross-fold pairs whose two scores are coupled through each other's labels.  That
coupling has no analogue at test time, so

    A_within  = the pooled AUC restricted to pairs inside one fold

is the coupling-free criterion: every pair in it is scored by a single model that saw
neither of the two rows, which is exactly the train->test configuration.

Whether that matters is a question about DECISIONS, not about mechanisms, and this
workspace's only decision is which file to select at the deadline.  So:

  1. Does `A_within` re-rank the top-CV cluster?  The deadline picks are separated by
     ~1e-6 of pooled CV, so a member-dependent shift of a few e-6 could move them.
  2. Is `A_within` a BETTER predictor of the leaderboard than pooled CV?  We hold 30
     files with both a CV and an LB score (experiments/w15c_shift_results.csv), so this
     is answerable rather than arguable.
  3. What does it COST?  A_within uses ~1/5 of the pairs, so it is noisier.  A criterion
     that removes a 4e-6 bias at the price of 20e-6 of extra variance is a worse
     criterion, and the honest way to find that out is a paired bootstrap over rows.

Usage:  python experiments/w15g_criterion.py [n_boot]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15g_coupling import NFOLD, FoldPairAUC, decompose  # noqa: E402

SHIFT = os.path.join(ROOT, "experiments", "w15c_shift_results.csv")
OUT = os.path.join(ROOT, "experiments", "w15g_criterion.json")
CSV = os.path.join(ROOT, "experiments", "w15g_criterion.csv")
SEED = 20260815


def main() -> None:
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    n = len(y)
    folds = get_folds(y)
    fold_of_row = np.empty(n, dtype=np.int64)
    for k, (_, va) in enumerate(folds):
        fold_of_row[va] = k
    npos = np.array([(y[fold_of_row == k] > 0.5).sum() for k in range(NFOLD)], float)
    nneg = np.array([(y[fold_of_row == k] < 0.5).sum() for k in range(NFOLD)], float)

    # ---- every blend we hold an OOF for -----------------------------------------
    names = sorted(f[4:-4] for f in os.listdir(SUB)
                   if f.startswith("oof_") and f.endswith(".npy"))
    rows = []
    for name in names:
        s = np.load(os.path.join(SUB, f"oof_{name}.npy")).astype(np.float64)
        if s.shape != (n,):
            continue
        fp = FoldPairAUC(s, y)
        pooled, within, cross = decompose(fp.u_matrix(fold_of_row), npos, nneg)
        rows.append(dict(file=name, pooled=pooled, within=within, cross=cross,
                         bias=pooled - within))
    df = pd.DataFrame(rows).sort_values("pooled", ascending=False).reset_index(drop=True)
    df["rank_pooled"] = df["pooled"].rank(ascending=False).astype(int)
    df["rank_within"] = df["within"].rank(ascending=False).astype(int)
    df.to_csv(CSV, index=False)

    print(f"{len(df)} blend OOFs.  Top 14 by POOLED CV, with the coupling-free criterion:\n")
    print(f"{'file':26s} {'pooled':>11} {'within':>11} {'bias e-6':>9} "
          f"{'rk_pool':>8} {'rk_within':>10}")
    for _, r in df.head(14).iterrows():
        print(f"{r['file']:26s} {r['pooled']:11.7f} {r['within']:11.7f} "
              f"{r['bias'] * 1e6:+9.2f} {r['rank_pooled']:8d} {r['rank_within']:10d}")

    top_pooled = df.iloc[0]["file"]
    top_within = df.sort_values("within", ascending=False).iloc[0]["file"]
    print(f"\nargmax pooled = {top_pooled}   argmax within = {top_within}"
          f"   {'SAME' if top_pooled == top_within else '*** DIFFERENT ***'}")
    print("spearman(pooled, within) over all blends: "
          f"{spearmanr(df['pooled'], df['within']).statistic:.4f}")

    # ---- 3. what does the coupling-free criterion cost in variance? --------------
    # Paired bootstrap over rows for the two deadline picks, which pooled CV separates
    # by ~0.5e-6.  If A_within's paired sd is much larger than pooled's, it is not a
    # usable tie-breaker no matter how unbiased it is.
    pair = ("blend159av_h3", "blend160origm_h3")
    sa = np.load(os.path.join(SUB, f"oof_{pair[0]}.npy")).astype(np.float64)
    sb = np.load(os.path.join(SUB, f"oof_{pair[1]}.npy")).astype(np.float64)
    rng = np.random.default_rng(SEED)
    dp, dw = [], []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        yy, ff = y[idx], fold_of_row[idx]
        npb = np.array([(yy[ff == k] > 0.5).sum() for k in range(NFOLD)], float)
        nnb = np.array([(yy[ff == k] < 0.5).sum() for k in range(NFOLD)], float)
        pa, wa, _ = decompose(FoldPairAUC(sa[idx], yy).u_matrix(ff), npb, nnb)
        pb, wb, _ = decompose(FoldPairAUC(sb[idx], yy).u_matrix(ff), npb, nnb)
        dp.append(pa - pb)
        dw.append(wa - wb)
    dp, dw = np.asarray(dp), np.asarray(dw)
    print(f"\npaired bootstrap ({nboot} reps) on {pair[0]} - {pair[1]}:")
    print(f"  pooled  diff {dp.mean() * 1e6:+7.3f}e-6  sd {dp.std(ddof=1) * 1e6:6.3f}e-6")
    print(f"  within  diff {dw.mean() * 1e6:+7.3f}e-6  sd {dw.std(ddof=1) * 1e6:6.3f}e-6")
    print(f"  variance price of the coupling-free criterion: "
          f"x{dw.std(ddof=1) / dp.std(ddof=1):.2f}")

    # ---- 2. does it predict the leaderboard better? -----------------------------
    lb = pd.read_csv(SHIFT)
    m = lb.merge(df, on="file", how="inner")
    print(f"\n{len(m)} of {len(lb)} LB-scored files have a blend OOF here.")
    out = dict(argmax_pooled=top_pooled, argmax_within=top_within,
               boot_pair=list(pair), boot_pooled_sd=float(dp.std(ddof=1)),
               boot_within_sd=float(dw.std(ddof=1)), n_lb=int(len(m)))
    if len(m) >= 8:
        print(f"{'criterion':14s} {'spearman vs LB':>15} {'resid sd (lb~crit)':>20}")
        for crit in ("pooled", "within", "cv_w"):
            if crit not in m.columns:
                continue
            x, yv = m[crit].to_numpy(), m["lb"].to_numpy()
            rho = spearmanr(x, yv).statistic
            b, a = np.polyfit(x, yv, 1)
            resid = yv - (a + b * x)
            print(f"{crit:14s} {rho:15.4f} {resid.std(ddof=2) * 1e6:17.1f}e-6")
            out[f"rho_{crit}"] = float(rho)
            out[f"residsd_{crit}"] = float(resid.std(ddof=2))
        # restricted to the tight cluster, where the decision actually lives
        tight = m[m["lb"] >= 0.97099]
        if len(tight) >= 8:
            print(f"\nrestricted to LB >= 0.97099 ({len(tight)} files):")
            for crit in ("pooled", "within", "cv_w"):
                rho = spearmanr(tight[crit], tight["lb"]).statistic
                print(f"  {crit:12s} spearman {rho:+.4f}")
                out[f"rho_tight_{crit}"] = float(rho)

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {CSV}\nwrote {OUT}")


if __name__ == "__main__":
    main()
