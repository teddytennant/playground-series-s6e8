"""Does REPAIRING the original's joint distribution rescue the transfer?

`w15d_screen_source.py` establishes that the generator did not merely smear the 7,500-row
source: it rebuilt the joint distribution of the four budget columns.

    ratio r = (social + gaming + work_study) / daily
      original:    median 1.140, q95 2.713, max 5.026        -- unconstrained
      competition: median 0.885, q95 0.949, max 1.0000       -- piled up against a hard 1.0

    weekend - daily
      original:    [0.50, 3.00] in 100.0% of rows            -- weekend = daily + U(.5,3)
      competition: [-7.91, 11.49], only 52.0% inside that band

`orig_binm` is fitted on the unrepaired original and asked to score the repaired frame. So
the standing "the original adds nothing" verdict may be a verdict on the mismatch rather
than on the 7,500 rows. This repairs the original THROUGH the same two transformations --
quantile-map each original row's `r` onto the competition's `r`, rescale its three
components to hit it; same for `weekend - daily` -- refits the identical `orig_member.py`
recipe (10 MCAR-masked copies at the competition's own per-column rates, bagged), and
reports the transfer AUC against `orig_binm`'s 0.8864.

Controls: `none` must reproduce 0.8864, and `shuffle` assigns each original row a RANDOM
competition `r` instead of its quantile-matched one -- same marginal, no rank information --
which separates "matching the joint" from "moving the marginal".

    .venv/bin/python experiments/w15d_origrepair.py --bags 8 --miss-reps 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CAT, DATA, NUM, OOF, SEED, SUB, TARGET, load_raw, save_preds  # noqa: E402

import lightgbm as lgb  # noqa: E402

ORIG = os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
COLS = NUM + CAT
COMPONENTS = ["social_media_hours", "gaming_hours", "work_study_hours"]
BASE = dict(metric="None", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, verbosity=-1)


def as_frame(df, vocab):
    X = df[COLS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=vocab[c])
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def qmap(src, target_sorted, rng=None, shuffle=False):
    """Map `src` onto `target_sorted`'s distribution, preserving rank (or not, if shuffle)."""
    n = len(src)
    if shuffle:
        return target_sorted[rng.integers(0, len(target_sorted), n)]
    # rank of each src value -> same quantile of the target
    order = np.argsort(np.argsort(src))
    q = (order + 0.5) / n
    idx = np.clip((q * len(target_sorted)).astype(int), 0, len(target_sorted) - 1)
    return target_sorted[idx]


def repair(orig, tr, mode, rng):
    """Return a copy of `orig` with its budget geometry mapped onto the competition's."""
    o = orig.copy()
    if mode == "none":
        return o

    s_c = tr[COMPONENTS].sum(axis=1)
    ok = s_c.notna() & tr["daily_screen_time_hours"].notna()
    r_target = np.sort((s_c[ok] / tr.loc[ok, "daily_screen_time_hours"]).to_numpy())

    if mode in ("r", "r+wd", "shuffle"):
        s_o = o[COMPONENTS].sum(axis=1).to_numpy()
        r_o = s_o / o["daily_screen_time_hours"].to_numpy()
        r_new = qmap(r_o, r_target, rng, shuffle=(mode == "shuffle"))
        scale = np.where(s_o > 0, r_new / np.maximum(r_o, 1e-12), 1.0)
        for c in COMPONENTS:
            o[c] = np.round(o[c].to_numpy() * scale, 2)

    if mode == "r+wd":
        wd_c = (tr["weekend_screen_time"] - tr["daily_screen_time_hours"])
        wd_target = np.sort(wd_c.dropna().to_numpy())
        wd_o = (o["weekend_screen_time"] - o["daily_screen_time_hours"]).to_numpy()
        o["weekend_screen_time"] = np.round(
            o["daily_screen_time_hours"].to_numpy() + qmap(wd_o, wd_target), 2)
    return o


def mask_like_comp(X, rates, rng, reps):
    out = []
    for _ in range(reps):
        Z = X.copy()
        for c in X.columns:
            Z.loc[rng.random(len(X)) < rates[c], c] = np.nan
        out.append(Z)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--miss-reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--save", default="", help="mode whose predictions to persist")
    a = ap.parse_args()
    BASE["num_threads"] = a.threads

    orig = pd.read_csv(ORIG)
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    vocab = {c: sorted(set(orig[c].dropna()) | set(tr[c].dropna())) for c in CAT}
    Xtr = as_frame(tr, vocab)
    Xte = as_frame(te, vocab) if a.save else None
    y_bin = orig[TARGET].to_numpy(dtype=float)
    rates = {c: float(tr[c].isna().mean()) for c in COLS}
    pack = np.load(os.path.join(SUB, "oof_blend159av_h3.npy"))

    print(f"competition r: median {np.median((tr[COMPONENTS].sum(1) / tr.daily_screen_time_hours).dropna()):.4f}")
    res = {}
    for mode in ("none", "r", "r+wd", "shuffle"):
        rng0 = np.random.default_rng(SEED)
        o = repair(orig, tr, mode, rng0)
        s = o[COMPONENTS].sum(axis=1)
        viol = float((o["daily_screen_time_hours"] < s - 1e-9).mean())
        Xo = as_frame(o, vocab)
        ptr = np.zeros(len(tr))
        pte = np.zeros(len(te)) if a.save == mode else None
        for b in range(a.bags):
            p = dict(BASE, objective="binary", seed=SEED + 101 * b,
                     bagging_seed=SEED + 7 * b, feature_fraction_seed=SEED + 13 * b)
            rng = np.random.default_rng(SEED + 1009 * b)
            Xb = mask_like_comp(Xo, rates, rng, a.miss_reps)
            yb = np.tile(y_bin, a.miss_reps)
            m = lgb.train(p, lgb.Dataset(Xb, yb), num_boost_round=a.rounds)
            ptr += m.predict(Xtr)
            if pte is not None:
                pte += m.predict(Xte)
        ptr /= a.bags
        auc = roc_auc_score(y, ptr)
        rho = float(np.corrcoef(pd.Series(ptr).rank(), pd.Series(pack).rank())[0, 1])
        res[mode] = ptr
        print(f"  mode {mode:8s} viol {viol:6.3f}  transfer AUC {auc:.6f}  "
              f"spearman to pack h3 {rho:+.4f}", flush=True)
        if pte is not None:
            save_preds(f"w15d_origrep_{mode}", ptr, pte / a.bags, len(tr), len(te), out=OOF)

    print("\nbaseline: orig_binm 0.886379 (RESEARCH route 1)")
    print("pairwise spearman between modes:")
    ks = list(res)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            print(f"  {ks[i]:8s} vs {ks[j]:8s} {np.corrcoef(res[ks[i]], res[ks[j]])[0,1]:+.4f}")


if __name__ == "__main__":
    main()
