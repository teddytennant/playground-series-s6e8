"""w15b step 5 -- POWER-CALIBRATE the residual search. The control the closed list never had.

The closed list rests on a pile of nulls that all have the same shape: "we searched the 12
columns for something the pack's score has missed, against a matched control, and found
nothing." resid_boost.py, resid_boost2.py, w14d_cellboost.py (9/9 negative), iso_regime.py,
w14d's per-cell isotonic (real - ctrl = +6e-6). Every one of them is a NULL.

A null is only as strong as the power behind it, and NOT ONE of those runs ever asked the
question that turns it into a bound:

    if a signal the size of the leaders' gap really were sitting in these 12 columns,
    would this search have found it?

w15b_price.py priced that gap: reaching MILANFX's 0.97124 needs an independent signal worth
tau ~ 0.36 in logit units, standalone AUC ~ 0.53. So inject exactly that, into labels we
generate ourselves, and re-run the search. Now the truth is known and the recovery fraction
is measurable.

  recovery = (AUC gain the search actually recovers) / (AUC gain that was injected)

  recovery ~ 1  -> the search sees signal of this size. The real nulls are then genuine
                   bounds and the leaders' edge is NOT a function of these 12 columns.
  recovery ~ 0  -> the searches were underpowered, every null on the closed list is
                   uninformative at this effect size, and the closed list is wrong.

Two injected signal SHAPES, because "learnable" is not one thing:

  smooth3   -- a smooth random-Fourier function of (age, sleep_hours, notifications_per_day).
               Low-frequency, three columns, exactly what a GBDT is built to find. This is
               the GENEROUS arm: if the search misses this, it misses everything.
  latticefp -- a random function of the EXACT `daily_screen_time_hours` lattice level (1,389
               levels, ~500 rows each). This is the generator-fingerprint shape that
               w15b_surface.py found in the raw rates, and it is the adversarial arm: a GBDT
               binning 1,389 values into max_bin=511 merges ~2.7 levels per bin and cannot
               represent it, while full-resolution target encoding can. Running the search
               both ways prices the pack's feature recipe against its own blind spot.

Arms at tau = 0 are the FALSE-POSITIVE control: same pipeline, same everything, no injected
signal. Whatever gain that arm reports is the search's own optimism and is subtracted.

Everything is evaluated against the SYNTHETIC labels it was generated with, on the frozen
folds, so the recovery fraction is an honest cross-fitted quantity.
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15b_price import bayes_auc, expit, logit  # noqa: E402

NTHREAD = 3
SEED = 20260815
# Match resid_boost2.py EXACTLY -- this run calibrates the instrument that produced the
# closed list's nulls, so its hyperparameters and its round-checkpoint ladder are not
# free choices. resid_boost2.py: min_data_in_leaf=500, feature_fraction=0.9,
# bagging 0.8/freq1, lambda_l2=5.0, lr 0.05, leaves 127, CHECK at every round below.
CHECK = [25, 50, 100, 200, 350, 500, 750]
ROUNDS = max(CHECK)
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=500, feature_fraction=0.9, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=5.0, max_bin=511, verbose=-1,
              num_threads=NTHREAD, seed=SEED, deterministic=True, force_row_wise=True)


def auc_under(p_true: np.ndarray, s: np.ndarray) -> float:
    """Expected AUC of ranking by `s` when y ~ Bernoulli(p_true). Same Bernoulli identity
    as bayes_auc but with the ranking decoupled from the probability field.

    This exists because the obvious empirical version -- roc_auc_score(one label draw, s)
    -- carries the MARGINAL single-draw noise, sd ~1.4e-4 at n=691k, which is as large as
    the effects being injected. Measured on the tau=0 control arm: an "injected gain" of
    +144e-6 where the true value is exactly 0. The denominator of a recovery fraction has
    to be analytic or the fraction is meaningless.
    """
    o = np.argsort(s, kind="stable")
    ps, ss = p_true[o], s[o]
    qs = 1.0 - ps
    bnd = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])   # tie-groups in s
    gp = np.add.reduceat(ps, bnd[:-1])
    gq = np.add.reduceat(qs, bnd[:-1])
    gpq = np.add.reduceat(ps * qs, bnd[:-1])
    below = np.r_[0.0, np.cumsum(gq)[:-1]]
    num = float(np.sum(gp * below) + 0.5 * (np.sum(gp * gq) - np.sum(gpq)))
    den = float(ps.sum() * qs.sum() - np.sum(ps * qs))
    return num / den


def make_signal(kind: str, tr: pd.DataFrame, rng) -> np.ndarray:
    """A standardised, genuinely-learnable function of the observed columns."""
    n = len(tr)
    if kind == "smooth3":
        cols = ["age", "sleep_hours", "notifications_per_day"]
        Z = tr[cols].to_numpy(np.float64)
        # median-impute only for building the signal; the search still sees real NaNs
        for j in range(Z.shape[1]):
            m = np.isnan(Z[:, j])
            Z[m, j] = np.nanmedian(Z[:, j])
        Z = (Z - Z.mean(0)) / Z.std(0)
        D = 24
        W = rng.standard_normal((Z.shape[1], D)) * 0.7
        b = rng.uniform(0, 2 * np.pi, D)
        a = rng.standard_normal(D)
        z = np.cos(Z @ W + b) @ a
    elif kind == "latticefp":
        v = tr[DAILY].to_numpy(np.float64)
        v = np.where(np.isnan(v), np.inf, v)
        _, inv = np.unique(v, return_inverse=True)
        z = rng.standard_normal(inv.max() + 1)[inv]
    else:
        raise ValueError(kind)
    return (z - z.mean()) / z.std()


def orthogonalise(z: np.ndarray, lp: np.ndarray) -> np.ndarray:
    """Remove the component along logit(p-hat) so the injected signal is genuinely NEW."""
    b = np.dot(z, lp) / np.dot(lp, lp)
    r = z - b * lp
    return (r - r.mean()) / r.std()


def base_frame(tr: pd.DataFrame) -> pd.DataFrame:
    X = tr[NUM + CAT].copy()
    for c in CAT:
        X[c] = X[c].astype("category")
    return X


def lattice_levels(tr: pd.DataFrame) -> dict:
    """Full-resolution lattice level id per numeric column, NaN as its own level."""
    out = {}
    for c in NUM:
        v = tr[c].to_numpy(np.float64)
        _, inv = np.unique(np.where(np.isnan(v), np.inf, v), return_inverse=True)
        out[c] = inv.astype(np.int64)
    return out


def add_te(X: pd.DataFrame, lv: dict, yy: np.ndarray, tr_i, va_i, smooth=20.0):
    """Out-of-fold full-resolution target encoding -- the pack's actual recipe
    (RESEARCH.md: "the single biggest win", +0.0023). Fitted on the training rows of THIS
    fold only, so it is leakage-free.

    The first version of this script used the raw lattice level as a LightGBM categorical
    instead. That frame overfits catastrophically -- its tau=0 false-positive control read
    -435e-6 at only 25 rounds against the raw frame's -29e-6 -- which makes it useless as a
    detector. Target encoding is both the faithful reproduction of the pack and a far
    better-conditioned instrument.
    """
    Xa, Xb = X.iloc[tr_i].copy(), X.iloc[va_i].copy()
    prior = yy[tr_i].mean()
    for c, inv in lv.items():
        K = int(inv.max()) + 1
        a = inv[tr_i]
        cnt = np.bincount(a, minlength=K).astype(np.float64)
        tot = np.bincount(a, weights=yy[tr_i], minlength=K)
        rate = (tot + smooth * prior) / (cnt + smooth)
        Xa[f"te_{c}"] = rate[a]
        Xb[f"te_{c}"] = rate[inv[va_i]]
    return Xa, Xb


def resid_search(X: pd.DataFrame, lv, yy: np.ndarray, lp: np.ndarray, folds,
                 use_te: bool) -> dict:
    """The closed list's own instrument: boost on top of the pack score as init_score.
    Returns {n_rounds: oof score vector} so every checkpoint on resid_boost2's ladder is
    read, exactly as that script reads them."""
    import lightgbm as lgb

    out = {r: np.empty(len(yy)) for r in CHECK}
    for tr_i, va_i in folds:
        if use_te:
            Xa, Xb = add_te(X, lv, yy, tr_i, va_i)
        else:
            Xa, Xb = X.iloc[tr_i], X.iloc[va_i]
        ds = lgb.Dataset(Xa, label=yy[tr_i], init_score=lp[tr_i], free_raw_data=False)
        m = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
        for r in CHECK:
            out[r][va_i] = lp[va_i] + m.predict(Xb, num_iteration=r, raw_score=True)
    return out


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    p = np.load(os.path.join(ROOT, "experiments", "w15b_phat.npy"))
    lp = logit(p)
    a_base = bayes_auc(p)
    print(f"calibrated pack p-hat: A* {a_base:.6f}   observed {roc_auc_score(y, p):.6f}",
          flush=True)

    rng = np.random.default_rng(SEED)
    X0 = base_frame(tr)
    lv = lattice_levels(tr)
    frames = {"raw": False, "te": True}
    print(f"  base frame: {X0.shape[1]} columns; 'te' adds {len(lv)} out-of-fold "
          f"full-resolution target encodings")

    results = []
    # tau = 0.134 is the LEADER-SIZED signal: w15b_price2.py's corrected ladder puts the
    # +180e-6 gap to MILANFX at tau 0.134 (solo AUC 0.512). tau = 0.36 is a strong positive
    # control at ~7x that. The first version of this script injected only 0.36/0.80 because
    # it used w15b_price.py's uncorrected inversion, which overstated tau by ~2.7x.
    plan = [("none", 0.0), ("smooth3", 0.134), ("latticefp", 0.134),
            ("smooth3", 0.36), ("latticefp", 0.36)]

    for kind, tau in plan:
        if kind == "none":
            z = np.zeros(len(y))
            pt = p.copy()
        else:
            z = orthogonalise(make_signal(kind, tr, np.random.default_rng(SEED + 7)), lp)
            pt = expit(lp + tau * z)
        yy = (np.random.default_rng(SEED + 99).random(len(y)) < pt).astype(np.float64)

        a_star = bayes_auc(pt)               # ceiling under the synthetic truth (analytic)
        a_pack_ana = auc_under(pt, p)        # what the pack alone gets, ANALYTIC
        a_pack = roc_auc_score(yy, p)        # same thing on this one draw, for reference
        injected = a_star - a_pack_ana
        print(f"\n=== inject {kind:10s} tau={tau:4.2f} ===")
        print(f"  ceiling A*(p_tau) {a_star:.6f}   pack alone (analytic) {a_pack_ana:.6f}"
              f"   INJECTED gain {injected:+.6f}")
        print(f"  [one-draw empirical pack AUC {a_pack:.6f}, draw noise "
              f"{a_pack - a_pack_ana:+.6f}]", flush=True)

        for mode, use_te in frames.items():
            oofs = resid_search(X0, lv, yy, lp, folds, use_te)
            for r in CHECK:
                a_found = roc_auc_score(yy, oofs[r])
                # `recovered` is PAIRED against the same label draw, so it carries only
                # the paired noise (~1e-5), not the marginal single-draw noise (1.4e-4).
                recovered = a_found - a_pack
                rec = recovered / injected if abs(injected) > 1e-9 else np.nan
                print(f"    search[{mode:3s}] r={r:4d}  AUC {a_found:.6f}   "
                      f"recovered {recovered:+.6f}   recovery "
                      f"{rec * 100 if np.isfinite(rec) else float('nan'):7.1f}%",
                      flush=True)
                results.append(dict(kind=kind, tau=tau, frame=mode, rounds=r,
                                    a_star=float(a_star),
                                    a_pack_analytic=float(a_pack_ana),
                                    a_pack_draw=float(a_pack), injected=float(injected),
                                    a_found=float(a_found), recovered=float(recovered),
                                    recovery=float(rec) if np.isfinite(rec) else None))
            pd.DataFrame(results).to_csv(
                os.path.join(ROOT, "experiments", "w15b_power.csv"), index=False)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(ROOT, "experiments", "w15b_power.csv"), index=False)
    with open(os.path.join(ROOT, "experiments", "w15b_power.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/w15b_power.csv, w15b_power.json")


if __name__ == "__main__":
    main()
