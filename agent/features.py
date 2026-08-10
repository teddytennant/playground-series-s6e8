"""Feature construction: constrained imputation + the quantisation lattice.

WHERE THIS COMES FROM
---------------------
The recipe is adapted from the `src/` code shipped with the public OOF library
`szymonkapiski/s6e8-oof-library-47-models` (train_impute.py, train_lattice.py), which in
turn credits the all-columns target-encoding idea to
https://www.kaggle.com/code/omidbaghchehsaraei/tabm-for-predicting-smartphone-addiction
and the digit/quantisation families to
https://www.kaggle.com/code/donmarch14/s6e8-catboost

TWO MECHANISMS DO THE WORK
--------------------------
1. Constrained imputation. Every column is 4-20% missing. `daily >= social + gaming +
   work_study` holds in 100% of competition rows (and is violated in 60.7% of the real
   source dataset -- the generator manufactured it). So the observed component sum is a
   hard LOWER bound on a missing `daily`, and the slack is a hard UPPER bound on a
   missing component. That beats any statistical guess. Imputed values are added
   ALONGSIDE the NaN-bearing originals, never as a substitute: a GBM's native NaN
   handling learns a default split direction per node, which is strictly more expressive
   than one imputed point estimate.

2. The quantisation lattice. The generator drew values and rounded them, so each numeric
   column is a lattice of a few thousand repeated values (`daily_screen_time_hours` has
   1,389 distinct values over 691k rows, ~500 rows per level). Casting every column to a
   string level and target-encoding it at full resolution therefore estimates a
   well-determined quantity at each lattice point. This was worth more than feature
   engineering, tuning and model selection combined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import CAT, COMP, DAILY, NUM, SEED, TARGET


def constrained_impute(df):
    """Reconstruct missing screen-block values using the generator inequality.

    Target-free and row-wise, so it is leakage-safe and is computed once on train+test
    together (transductive preprocessing, not leakage).
    """
    o = pd.DataFrame(index=df.index)
    d = df[DAILY]
    C = df[COMP]
    comp_sum_obs = C.sum(axis=1, skipna=True)
    n_comp_known = C.notna().sum(axis=1)

    # --- statistical backbone: iterative imputer over the correlated screen block ---
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    block = [DAILY, "weekend_screen_time", "social_media_hours", "gaming_hours",
             "work_study_hours", "sleep_hours"]
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=12, random_state=SEED)
    est = pd.DataFrame(imp.fit_transform(df[block]), columns=block, index=df.index)

    # --- CASE 1: daily missing -> hard LOWER bound from the observed components ---
    lo_daily = comp_sum_obs.where(n_comp_known > 0, 0.0)
    daily_imp = d.copy()
    need = d.isna()
    daily_imp[need] = np.maximum(est[DAILY][need], lo_daily[need])
    o["daily_imp"] = daily_imp
    o["daily_lower_bound"] = lo_daily
    # how much the constraint actually moved the estimate (0 => constraint was slack)
    o["daily_bound_bind"] = np.where(need, np.maximum(0.0, lo_daily - est[DAILY]), 0.0)
    o["daily_was_missing"] = need.astype("int8")

    # --- CASE 2: one component missing, daily known -> TWO-SIDED bound [0, slack] ---
    for c in COMP:
        others = [x for x in COMP if x != c]
        others_sum = df[others].sum(axis=1, skipna=True)
        others_known = df[others].notna().all(axis=1)
        slack = (daily_imp - others_sum).where(others_known, np.nan).clip(lower=0.0)
        v = df[c].copy()
        m = v.isna()
        cand = est[c].clip(lower=0.0)
        upper = slack.where(slack.notna(), np.inf)
        v[m] = np.minimum(np.maximum(cand[m], 0.0), upper[m])
        o[f"{c}_imp"] = v
        o[f"{c}_upper_bound"] = slack
        # width of the feasible interval = how certain this imputation is
        o[f"{c}_bound_width"] = np.where(m & slack.notna(), slack, np.nan)
        o[f"{c}_was_missing"] = m.astype("int8")

    for c in ["weekend_screen_time", "sleep_hours"]:
        o[f"{c}_imp"] = df[c].fillna(est[c])
        o[f"{c}_was_missing"] = df[c].isna().astype("int8")

    # --- consistency features computed on the RECONSTRUCTED values ---
    ci = o[[f"{c}_imp" for c in COMP]].sum(axis=1)
    o["other_screen_imp"] = o["daily_imp"] - ci
    o["comp_share_imp"] = ci / (o["daily_imp"] + 1e-6)
    o["identity_violation"] = np.clip(ci - o["daily_imp"], 0, None)  # should be ~0
    o["n_screen_missing"] = df[[DAILY] + COMP].isna().sum(axis=1).astype("int8")
    o["n_missing_all"] = df[NUM + CAT].isna().sum(axis=1).astype("int8")
    return o


def _part(full, c, res="floor"):
    """One lattice coordinate for column `c` at the requested resolution."""
    if c not in NUM:
        return full[c].astype(str)
    v = pd.to_numeric(full[c], errors="coerce")
    if res == "r1":
        return v.round(1).astype(str)
    if res == "full":
        return v.astype(str)
    return np.floor(v).astype("Int32").astype(str)


def make_frames(tr, te, wide_pairs=True, pair_res="floor", triples=True):
    """Numeric design matrix + the string lattice keys that get target-encoded.

    Defaults reproduce the `lattri_*` feature set (all-36-pair lattice + 3-way screen
    cells), the strongest LightGBM configuration in the public library at OOF 0.96768.
    """
    ntr = len(tr)
    full = pd.concat([tr.drop(columns=[TARGET]), te], axis=0, ignore_index=True)

    imp = constrained_impute(full)

    X = pd.concat([full[NUM].astype("float32"), imp.astype("float32")], axis=1)
    for c in CAT:
        X[c] = pd.Categorical(full[c].astype(str)).codes.astype("float32")

    # domain relations that measured positive in the library's own ablations
    X["leisure_hours"] = full["social_media_hours"] + full["gaming_hours"]
    X["weekend_minus_daily"] = full["weekend_screen_time"] - full[DAILY]
    X["notif_per_open"] = full["notifications_per_day"] / (full["app_opens_per_day"] + 1.0)

    # --- the lattice keys: every predictor at FULL resolution, plus coarsenings ---
    keys = pd.DataFrame(index=full.index)
    for c in NUM + CAT:
        keys[c] = full[c].astype(str)
    # coarsenings give denser cells where the full-resolution one is too thin to trust
    for c in NUM:
        v = pd.to_numeric(full[c], errors="coerce")
        keys[f"{c}_r1"] = v.round(1).astype(str)
        keys[f"{c}_fl"] = np.floor(v).astype("Int32").astype(str)

    pairs = [(DAILY, "weekend_screen_time"), ("social_media_hours", "gaming_hours"),
             (DAILY, "social_media_hours"), ("stress_level", "academic_work_impact")]
    if wide_pairs:
        pairs = sorted(set(pairs) | {(a, b) for i, a in enumerate(NUM)
                                     for b in NUM[i + 1:]})
    for l, r in pairs:
        keys[f"P_{l}__{r}"] = _part(full, l) + "__" + _part(full, r)

    if pair_res == "r1":
        for l, r in pairs:
            keys[f"R_{l}__{r}"] = _part(full, l, "r1") + "__" + _part(full, r, "r1")

    if triples:
        # 3-way cells over the screen block only: the generator identity binds daily to
        # its three components jointly, so a 3-way cell is the lowest order at which that
        # relation appears as one lattice key rather than being rebuilt from marginals.
        trip = [(DAILY, "social_media_hours", "gaming_hours"),
                (DAILY, "social_media_hours", "work_study_hours"),
                (DAILY, "gaming_hours", "work_study_hours"),
                ("social_media_hours", "gaming_hours", "work_study_hours"),
                (DAILY, "weekend_screen_time", "sleep_hours")]
        for a3, b3, c3 in trip:
            keys[f"T_{a3}__{b3}__{c3}"] = (_part(full, a3) + "__" + _part(full, b3)
                                           + "__" + _part(full, c3))

    X = X.replace([np.inf, -np.inf], np.nan)
    return (X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True),
            keys.iloc[:ntr].reset_index(drop=True), keys.iloc[ntr:].reset_index(drop=True))


def te_block(Ktr, y_tr, Kva, Kte, smooth=20.0):
    """Fold-safe smoothed target + frequency encoding for every lattice key.

    Leakage is the entire difficulty. The train part uses an INNER 4-fold out-of-fold
    loop so no row ever enters its own encoding; valid and test use the map fit on the
    whole outer training part. The cell COUNT is emitted alongside the mean so the model
    can discount thin, noisy lattice cells instead of being forced to trust them equally.
    """
    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED + 101)
    gm = float(y_tr.mean())
    otr, ova, ote = {}, {}, {}
    splits = list(inner.split(Ktr, y_tr))
    yv = np.asarray(y_tr)
    for c in Ktr.columns:
        col = Ktr[c].to_numpy()
        oof_te = np.full(len(Ktr), gm, dtype="float32")
        oof_ct = np.zeros(len(Ktr), dtype="float32")
        for fi, hi in splits:
            g = pd.DataFrame({"k": col[fi], "y": yv[fi]}).groupby("k").y.agg(["sum", "count"])
            mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
            s = pd.Series(col[hi])
            oof_te[hi] = s.map(mp).fillna(gm).to_numpy()
            oof_ct[hi] = s.map(g["count"]).fillna(0.0).to_numpy()
        g = pd.DataFrame({"k": col, "y": yv}).groupby("k").y.agg(["sum", "count"])
        mp = (g["sum"] + smooth * gm) / (g["count"] + smooth)
        otr[f"TE_{c}"] = oof_te
        otr[f"CT_{c}"] = oof_ct
        ova[f"TE_{c}"] = Kva[c].map(mp).fillna(gm).to_numpy("float32")
        ova[f"CT_{c}"] = Kva[c].map(g["count"]).fillna(0.0).to_numpy("float32")
        ote[f"TE_{c}"] = Kte[c].map(mp).fillna(gm).to_numpy("float32")
        ote[f"CT_{c}"] = Kte[c].map(g["count"]).fillna(0.0).to_numpy("float32")
    return (pd.DataFrame(otr, index=Ktr.index), pd.DataFrame(ova, index=Kva.index),
            pd.DataFrame(ote, index=Kte.index))
