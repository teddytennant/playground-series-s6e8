"""Build a stack member out of the REAL source dataset only.

The angle is "concatenate the original dataset". Concatenation is the wrong shape for it
here: 7,500 real rows against 691,369 synthetic ones is a 1% dilution of a frame whose
structure the generator invented, which is why the public ablation record measures it at
-0.0001 and why nobody has been able to make it pay.

The right shape is a *separate estimator*. A model fitted on the 7,500 originals never
sees a competition label, so its prediction on the competition frame is honest on every
row of both splits -- it can go straight into the stack as a member with no fold
structure at all. And because it is fitted on a different distribution, it is a genuinely
different function of the same 12 columns, which is the one thing the 159-member pack is
short of (median maxcorr 0.9949).

Three targets are tried, all on the same 7,500 rows:

  bin   -- the binary `addicted_label`.
  ord   -- the 4-level ordinal recovered from `addiction_level`
           (NaN=None < Mild < Moderate < Severe; label = 1 iff >= Moderate). A finer
           target should pin the latent better from only 7,500 rows.
  rule  -- the explicit decision rule read off the original:
           social > 4 -> 1, social <= 4 & daily > 8 -> 1, social <= 4 & daily <= 6 -> 0,
           otherwise a ~0.456 coin flip. 86.3% of the original is deterministic under it.

Each is bagged over `--bags` seeds because 7,500 rows is small. Reported by transfer AUC
against the 691k competition labels; that number is honest, not out-of-fold-shaped.

    .venv/bin/python experiments/orig_member.py --bags 12
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CAT, DATA, NUM, OOF, SEED, TARGET, load_raw, save_preds  # noqa: E402

import lightgbm as lgb  # noqa: E402

ORIG = os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
COLS = NUM + CAT
LEVELS = ["None", "Mild", "Moderate", "Severe"]

BASE = dict(metric="None", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            verbosity=-1, num_threads=os.cpu_count())


def as_frame(df, vocab):
    X = df[COLS].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=vocab[c])
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def rule_score(df):
    """The explicit rule read off the original, as a score in [0,1].

    Uses the original's own within-band positive rate (0.4556) for the ambiguous cell, so
    the three groups rank in the order the real data puts them in. NaN in either driver
    falls back to the original base rate.
    """
    d = pd.to_numeric(df["daily_screen_time_hours"], errors="coerce")
    s = pd.to_numeric(df["social_media_hours"], errors="coerce")
    out = np.full(len(df), 0.7077)
    band = (s <= 4.0) & (d > 6.0) & (d <= 8.0)
    out[band.to_numpy(na_value=False)] = 0.4556
    out[((s <= 4.0) & (d <= 6.0)).to_numpy(na_value=False)] = 0.0
    out[((s <= 4.0) & (d > 8.0)).to_numpy(na_value=False)] = 1.0
    out[(s > 4.0).to_numpy(na_value=False)] = 1.0
    return out


def mask_like_comp(X, rates, rng, reps):
    """Stack `reps` MCAR-masked copies of X at the competition's per-column rates.

    The original has no missing values; 61% of competition rows carry at least one NaN,
    and the two columns the rule runs on are missing 13.9% (daily) and 19.4% (social).
    A model fitted on complete rows has never been asked which way a NaN should go, so
    LightGBM's default direction for those splits is whatever the library falls back to
    rather than anything learned. Masking the training copies at the competition's own
    rates makes that a fitted decision, which is the difference between scoring 29% of
    the competition rows arbitrarily and scoring them.
    """
    out = []
    for _ in range(reps):
        Z = X.copy()
        for c in X.columns:
            m = rng.random(len(X)) < rates[c]
            Z.loc[m, c] = np.nan
        out.append(Z)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--save", default="")
    ap.add_argument("--miss-reps", type=int, default=0,
                    help="if >0, train on this many MCAR-masked copies of the original")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default="", help="member name to save under")
    a = ap.parse_args()
    if a.threads:
        BASE["num_threads"] = a.threads

    orig = pd.read_csv(ORIG)
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    vocab = {c: sorted(set(orig[c].dropna()) | set(tr[c].dropna())) for c in CAT}
    Xo = as_frame(orig, vocab)
    Xtr, Xte = as_frame(tr, vocab), as_frame(te, vocab)

    y_bin = orig[TARGET].to_numpy(dtype=float)
    y_ord = orig["addiction_level"].fillna("None").map({k: i for i, k in enumerate(LEVELS)}).to_numpy(dtype=float)

    rates = {c: float(tr[c].isna().mean()) for c in COLS}

    results = {}
    for tag, target, obj in (("bin", y_bin, "binary"), ("ord", y_ord, "regression")):
        ptr = np.zeros(len(tr))
        pte = np.zeros(len(te))
        for b in range(a.bags):
            p = dict(BASE, objective=obj, seed=SEED + 101 * b,
                     bagging_seed=SEED + 7 * b, feature_fraction_seed=SEED + 13 * b)
            if a.miss_reps:
                rng = np.random.default_rng(SEED + 1009 * b)
                Xb = mask_like_comp(Xo, rates, rng, a.miss_reps)
                yb = np.tile(target, a.miss_reps)
            else:
                Xb, yb = Xo, target
            m = lgb.train(p, lgb.Dataset(Xb, yb), num_boost_round=a.rounds)
            ptr += m.predict(Xtr)
            pte += m.predict(Xte)
        ptr /= a.bags
        pte /= a.bags
        results[tag] = (ptr, pte)
        print(f"{tag:5s} transfer AUC on comp train  {roc_auc_score(y, ptr):.6f}", flush=True)

    rtr, rte = rule_score(tr), rule_score(te)
    results["rule"] = (rtr, rte)
    print(f"rule  transfer AUC on comp train  {roc_auc_score(y, rtr):.6f}", flush=True)

    # do the two learned views agree, or are they separate channels?
    for x, z in (("bin", "ord"), ("bin", "rule"), ("ord", "rule")):
        print(f"corr({x},{z}) train {np.corrcoef(results[x][0], results[z][0])[0,1]:.4f}")

    if a.save:
        ptr, pte = results[a.save]
        nm = a.name or f"orig_{a.save}"
        save_preds(nm, ptr, pte, len(tr), len(te), out=OOF)
        import json
        with open(os.path.join(OOF, f"summary_{nm}.json"), "w") as f:
            json.dump({"name": nm,
                       "auc": float(roc_auc_score(y, ptr)),
                       "note": "fitted on the 7,500-row real source dataset only; "
                               "never sees a competition label, so every row is honest"},
                      f, indent=2)


if __name__ == "__main__":
    main()
