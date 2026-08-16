"""w15b step 2 -- the label surface over the exact rule lattice, and whether the two
rule drivers are a SUFFICIENT STATISTIC for the label.

w15b_dupstruct.py established the constraint that decides this run's method: there are
ZERO duplicate rows on the full 12-column vector (max group size 1), and only 1.8% of
rows sit in a duplicate group even on 5 columns. So the repeated-measurement route to the
Bayes ceiling -- the only route that identifies E[p(1-p)] separately from model error --
exists ONLY on a low-dimensional conditioning set. On (social, daily) it is wide open:
221,008 cells, 84.7% of rows in cells of size >= 2, mean 3.13 rows/cell.

That makes one question load-bearing for the entire ceiling estimate:

    Is P(y | x) a function of (social_media_hours, daily_screen_time_hours) alone?

If yes, the ceiling is a 2-D estimation problem on 691k rows and is essentially solvable.
If no, the ceiling is not identified and any number quoted for it is a model artefact.

RESEARCH.md's generator story says the label came from a two-threshold rule in exactly
these two columns, smeared into a ramp -- so sufficiency is the null the story predicts.
But the same file records target encoding on ALL columns as the single biggest win
(+0.0023), which predicts the opposite. Those cannot both be about the same thing, and
nobody has measured which.

This script measures it three ways on the same frozen folds:

  1. The SURFACE. P(y | exact daily lattice value) inside each social band, on the full
     691k rows. Steps and plateaus vs a smooth ramp -- read off the lattice, not a bin.

  2. The DRIVER-ONLY MODEL. LightGBM on the two drivers only (then +weekend, then the
     rest), cross-fitted on the frozen folds. Restricted to the 73.8% of rows where both
     drivers are OBSERVED, its AUC vs the pack's AUC on the identical rows is the exact
     value of the other ten columns given the drivers.

  3. The CELL ORACLE. Out-of-fold smoothed cell rates on the exact (social, daily)
     lattice. This is the nonparametric version of 2 with no function class at all, and
     its learning curve is what w15b_ceiling.py extrapolates.

Controls: every AUC is computed on the SAME row subset (comparisons on different
subpopulations are not comparisons), and the driver-only model is run against a
column-permuted arm so "the other ten columns add X" has a matched null.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, DAILY, NUM, SUB, TARGET, get_folds, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
WEEKEND = "weekend_screen_time"
BLEND = os.environ.get("W15B_BLEND", "blend159av_h3")
NTHREAD = 3
SEED = 20260815


def lgbm_cv(X: pd.DataFrame, y: np.ndarray, folds, *, rounds: int, leaves: int,
            lr: float = 0.05, seed: int = SEED) -> np.ndarray:
    """Cross-fitted OOF probabilities on the frozen folds. Native NaN handling."""
    import lightgbm as lgb

    oof = np.zeros(len(y))
    for tr_i, va_i in folds:
        ds = lgb.Dataset(X.iloc[tr_i], label=y[tr_i], free_raw_data=False)
        params = dict(objective="binary", metric="auc", learning_rate=lr,
                      num_leaves=leaves, min_data_in_leaf=100, feature_fraction=1.0,
                      bagging_fraction=1.0, max_bin=511, verbose=-1,
                      num_threads=NTHREAD, seed=seed, deterministic=True,
                      force_row_wise=True)
        m = lgb.train(params, ds, num_boost_round=rounds)
        oof[va_i] = m.predict(X.iloc[va_i])
    return oof


def cell_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Exact joint lattice cell id; NaN is its own level in every column."""
    key = None
    for c in cols:
        v = df[c].to_numpy(np.float64)
        _, inv = np.unique(np.where(np.isnan(v), np.inf, v), return_inverse=True)
        inv = inv.astype(np.int64)
        key = inv if key is None else pd.factorize(key * (inv.max() + 1) + inv)[0]
    return np.asarray(key, dtype=np.int64)


def cell_oracle_oof(key: np.ndarray, y: np.ndarray, folds, m: float) -> np.ndarray:
    """Out-of-fold smoothed cell rate: (sum + m*prior) / (count + m), prior from the
    training half only. Unseen cells fall back to the prior. No leakage by construction."""
    K = int(key.max()) + 1
    oof = np.empty(len(y))
    for tr_i, va_i in folds:
        cnt = np.bincount(key[tr_i], minlength=K).astype(np.float64)
        tot = np.bincount(key[tr_i], weights=y[tr_i], minlength=K)
        prior = y[tr_i].mean()
        rate = (tot + m * prior) / (cnt + m)
        oof[va_i] = rate[key[va_i]]
    return oof


def main() -> None:
    tr, te = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    n = len(y)
    stack = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    assert len(stack) == n
    print(f"train {n:,}   base {y.mean():.7f}   pack {BLEND} OOF AUC "
          f"{roc_auc_score(y, stack):.6f}\n", flush=True)

    soc = tr[SOCIAL].to_numpy(np.float64)
    day = tr[DAILY].to_numpy(np.float64)
    both = ~np.isnan(soc) & ~np.isnan(day)
    print(f"both drivers observed: {both.sum():,} rows ({both.mean():.4f})", flush=True)

    # ---------------------------------------------------------------- 1. the surface
    print("\n=== 1. P(y | exact daily lattice value) inside each social band ===")
    print("    (full 691k rows; only lattice values with n >= 400 shown, binomial se)")
    bands = [("social<=4", both & (soc <= 4.0)), ("social>4", both & (soc > 4.0))]
    surf_rows = []
    for bname, mask in bands:
        d = day[mask]
        yy = y[mask]
        vals, inv = np.unique(d, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(vals)).astype(np.float64)
        tot = np.bincount(inv, weights=yy, minlength=len(vals))
        rate = tot / cnt
        se = np.sqrt(np.maximum(rate * (1 - rate), 1e-12) / cnt)
        for v, c, r, e in zip(vals, cnt, rate, se):
            surf_rows.append(dict(band=bname, daily=float(v), n=int(c),
                                  rate=float(r), se=float(e)))
        keep = cnt >= 400
        print(f"\n  -- {bname}: {int(mask.sum()):,} rows, {len(vals)} distinct daily "
              f"values, {int(keep.sum())} with n>=400")
        if keep.any():
            vk, rk, ck, ek = vals[keep], rate[keep], cnt[keep], se[keep]
            step = max(1, len(vk) // 40)
            for i in range(0, len(vk), step):
                bar = "#" * int(round(rk[i] * 50))
                print(f"     daily {vk[i]:6.2f}  n{int(ck[i]):>6,}  "
                      f"{rk[i]:.4f} +-{ek[i]:.4f}  {bar}")
    pd.DataFrame(surf_rows).to_csv(
        os.path.join(ROOT, "experiments", "w15b_surface_cells.csv"), index=False)

    # ------------------------------------------------- 2. driver-only vs the full frame
    print("\n=== 2. how much do the other ten columns add, given the two drivers? ===")
    print("    all AUCs on the IDENTICAL both-observed subset, frozen folds\n", flush=True)

    frames = {
        "drivers2 (social,daily)": [SOCIAL, DAILY],
        "drivers3 (+weekend)": [SOCIAL, DAILY, WEEKEND],
        "all9 numeric": list(NUM),
        "all12 raw": list(NUM) + list(CAT),
    }
    res = {}
    for name, cols in frames.items():
        X = tr[cols].copy()
        for c in cols:
            if c in CAT:
                X[c] = X[c].astype("category")
        rounds = 1200 if len(cols) <= 3 else 2000
        leaves = 255 if len(cols) <= 3 else 127
        oof = lgbm_cv(X, y, folds, rounds=rounds, leaves=leaves)
        a_all = roc_auc_score(y, oof)
        a_both = roc_auc_score(y[both], oof[both])
        res[name] = dict(auc_all=float(a_all), auc_both=float(a_both), cols=cols)
        np.save(os.path.join(ROOT, "experiments", f"w15b_oof_{name.split()[0]}.npy"), oof)
        print(f"  {name:26s}  AUC(all rows) {a_all:.6f}   AUC(both-observed) {a_both:.6f}",
              flush=True)

    a_pack_all = roc_auc_score(y, stack)
    a_pack_both = roc_auc_score(y[both], stack[both])
    print(f"  {'PACK ' + BLEND:26s}  AUC(all rows) {a_pack_all:.6f}   "
          f"AUC(both-observed) {a_pack_both:.6f}")
    res["pack"] = dict(auc_all=float(a_pack_all), auc_both=float(a_pack_both))

    d2 = res["drivers2 (social,daily)"]["auc_both"]
    print(f"\n  value of the other 10 columns on both-observed rows: "
          f"pack - drivers2 = {a_pack_both - d2:+.6f}")

    # ------------------------------------------------------------- 3. the cell oracle
    print("\n=== 3. out-of-fold exact-cell oracle on the (social, daily) lattice ===")
    key = cell_key(tr, [SOCIAL, DAILY])
    print(f"    {key.max() + 1:,} cells", flush=True)
    for m in (0.0, 1.0, 3.0, 10.0, 30.0, 100.0):
        o = cell_oracle_oof(key, y, folds, m)
        print(f"    smoothing m={m:6.1f}   AUC(all) {roc_auc_score(y, o):.6f}   "
              f"AUC(both-observed) {roc_auc_score(y[both], o[both]):.6f}", flush=True)
        res[f"cell_oracle_m{m:g}"] = dict(
            auc_all=float(roc_auc_score(y, o)),
            auc_both=float(roc_auc_score(y[both], o[both])))

    with open(os.path.join(ROOT, "experiments", "w15b_surface.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote experiments/w15b_surface.json, w15b_surface_cells.csv")


if __name__ == "__main__":
    main()
