"""w15b step 3 -- does the pack's score already contain the exact-lattice structure?

w15b_surface.py shows P(y | exact daily lattice value) inside social<=4 is NOT a smooth
ramp: adjacent lattice values move by 0.2-0.35 against binomial se ~0.02, e.g.

    daily 6.87 -> 0.683    7.18 -> 0.473    7.38 -> 0.839      (n ~ 500 each)

That is 10-15 sigma of non-smoothness between neighbouring lattice values. Two readings:

  (i) the generator's p really is a rough function of the exact lattice value, and a GBDT
      (max_bin 511 over 1,389 distinct daily values -> ~2.7 values merged per bin) plus
      smooth per-column target encoding cannot represent it. Then there is unexploited
      signal sitting in plain sight and it is worth a great deal.
 (ii) the pack already has it -- via full-resolution per-column TE, which RESEARCH.md
      records as the single biggest win (+0.0023) -- and the roughness is reproduced in
      the score, not missed by it.

Only a residual test separates those, and it must be run against a MATCHED null. The
workspace has been caught three times by an unmatched one (chi2/df, the permuted-feature
booster, the unpaired LB bootstrap), so:

  * the statistic is  z_c = (O_c - E_c) / sqrt(V_c)  with O_c the cell's positive count,
    E_c = sum of the CALIBRATED pack score in the cell and V_c = sum of s(1-s). Under
    "the pack is right inside this cell" the z_c are ~N(0,1) whatever the cell rate is.
  * the null is a SIZE-MATCHED CELL PERMUTATION: the same number of cells with the same
    sizes, membership shuffled among the same rows. That absorbs global miscalibration,
    the cell-size distribution, and any residual overdispersion of the score, and leaves
    only "is the real lattice partition special".
  * calibration is CROSS-FITTED isotonic on the frozen folds, so E_c is honest.

Then the only number that matters for a decision: blend the out-of-fold cell rate into
the pack and measure the AUC change, against the identical blend of the PERMUTED-cell
oracle. A gain the permuted arm also produces is not a gain.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DAILY, SUB, TARGET, get_folds, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
BLEND = os.environ.get("W15B_BLEND", "blend159av_h3")
NPERM = int(os.environ.get("W15B_NPERM", "40"))
SEED = 20260815


def cell_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    key = None
    for c in cols:
        v = df[c].to_numpy(np.float64)
        _, inv = np.unique(np.where(np.isnan(v), np.inf, v), return_inverse=True)
        inv = inv.astype(np.int64)
        key = inv if key is None else pd.factorize(key * (inv.max() + 1) + inv)[0]
    return np.asarray(key, dtype=np.int64)


def calibrate_oof(score: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Cross-fitted isotonic. Fit on the 4 training folds, apply to the held-out fold, so
    the calibrated value of a row never saw that row's label."""
    out = np.empty(len(y))
    for tr_i, va_i in folds:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(score[tr_i], y[tr_i])
        out[va_i] = ir.predict(score[va_i])
    return np.clip(out, 1e-6, 1 - 1e-6)


def zstats(key: np.ndarray, y: np.ndarray, s: np.ndarray, min_n: int) -> tuple:
    K = int(key.max()) + 1
    cnt = np.bincount(key, minlength=K).astype(np.float64)
    O = np.bincount(key, weights=y, minlength=K)
    E = np.bincount(key, weights=s, minlength=K)
    V = np.bincount(key, weights=s * (1 - s), minlength=K)
    ok = (cnt >= min_n) & (V > 1e-9)
    z = (O[ok] - E[ok]) / np.sqrt(V[ok])
    return float(np.sum(z ** 2)), int(ok.sum()), z


def cell_oracle_oof(key: np.ndarray, y: np.ndarray, folds, m: float) -> np.ndarray:
    K = int(key.max()) + 1
    oof = np.empty(len(y))
    for tr_i, va_i in folds:
        cnt = np.bincount(key[tr_i], minlength=K).astype(np.float64)
        tot = np.bincount(key[tr_i], weights=y[tr_i], minlength=K)
        prior = y[tr_i].mean()
        oof[va_i] = ((tot + m * prior) / (cnt + m))[key[va_i]]
    return oof


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    n = len(y)
    raw = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    print(f"pack {BLEND}: OOF AUC {roc_auc_score(y, raw):.6f}", flush=True)

    # w15b_calib.py showed plain cross-fitted isotonic is over-dispersed out of fold
    # (A* - observed = +117e-6 where a calibrated field must give 0). Prefer the
    # binned-PAVA field it validated; fall back only if that run has not happened.
    phat_path = os.path.join(ROOT, "experiments", "w15b_phat.npy")
    if os.path.exists(phat_path):
        s = np.load(phat_path)
        print("using w15b_phat.npy (identity-validated binned-PAVA calibration)")
    else:
        s = calibrate_oof(raw, y, folds)
    print(f"calibrated pack: AUC {roc_auc_score(y, s):.6f}  "
          f"Brier {np.mean((y - s) ** 2):.6f}  mean {s.mean():.6f} vs base {y.mean():.6f}",
          flush=True)

    rng = np.random.default_rng(SEED)
    res = {}

    for setname, cols in (("social+daily", [SOCIAL, DAILY]),
                          ("daily only", [DAILY]),
                          ("social only", [SOCIAL])):
        key = cell_key(tr, cols)
        K = int(key.max()) + 1
        print(f"\n=== lattice {setname}: {K:,} cells ===", flush=True)
        for min_n in (2, 20):
            T, df, z = zstats(key, y, s, min_n)
            # size-matched permuted-cell null: identical cell sizes, shuffled membership
            perm_T = np.empty(NPERM)
            for b in range(NPERM):
                pk = key[rng.permutation(n)]
                perm_T[b] = zstats(pk, y, s, min_n)[0]
            mu, sd = perm_T.mean(), perm_T.std(ddof=1)
            zz = (T - mu) / sd if sd > 0 else np.nan
            print(f"  cells n>={min_n:3d}: df {df:>7,}  T {T:>12,.1f}   "
                  f"T/df {T / df:.4f}   permuted T {mu:>12,.1f} +- {sd:,.1f}  "
                  f"=> z {zz:+.1f}", flush=True)
            res[f"{setname}|min_n{min_n}"] = dict(
                df=df, T=T, T_over_df=T / df, perm_mu=float(mu), perm_sd=float(sd),
                z=float(zz))

    # ------------------------------------------------- the decision-relevant version
    print("\n=== does the out-of-fold cell oracle add AUC to the pack? ===")
    print("    real (social,daily) cells vs a size-matched permuted-cell arm\n", flush=True)
    key = cell_key(tr, [SOCIAL, DAILY])
    pkey = key[rng.permutation(n)]
    lp = to_logit(s)
    base = roc_auc_score(y, raw)
    rows = []
    for m in (10.0, 30.0, 100.0):
        o_real = to_logit(cell_oracle_oof(key, y, folds, m))
        o_perm = to_logit(cell_oracle_oof(pkey, y, folds, m))
        for w in (0.02, 0.05, 0.10, 0.20, 0.35):
            a_r = roc_auc_score(y, lp + w * o_real)
            a_p = roc_auc_score(y, lp + w * o_perm)
            rows.append(dict(m=m, w=w, auc_real=a_r, auc_perm=a_p,
                             d_real=a_r - base, d_perm=a_p - base,
                             real_minus_perm=a_r - a_p))
            print(f"  m={m:6.1f} w={w:4.2f}   real {a_r:.6f} ({a_r - base:+.6f})   "
                  f"perm {a_p:.6f} ({a_p - base:+.6f})   real-perm {a_r - a_p:+.6f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT, "experiments", "w15b_lackfit_blend.csv"), index=False)

    with open(os.path.join(ROOT, "experiments", "w15b_lackfit.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote experiments/w15b_lackfit.json, w15b_lackfit_blend.csv")


if __name__ == "__main__":
    main()
