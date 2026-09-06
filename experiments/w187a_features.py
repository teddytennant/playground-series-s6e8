"""w187: port the feature set of `kodaifukuda0311/s6e8-how-to-achieve-0-97-with-realmlp-only`
(public LB 0.97016, a SINGLE model) onto this workspace's frozen folds.

WHY THIS FILE EXISTS
--------------------
w184 read the winners' writeups and found the ceiling was the base model, not the blend.
The named public notebook that reaches 0.97 with one model appears nowhere in JOURNAL.md,
RESEARCH.md or LEADERBOARD.md. Its recipe is two things stacked:

  1. every raw column in FOUR representations at once -- raw value, exact-value CATEGORY,
     exact-value target encoding, exact-value frequency;
  2. features read off the 7,500-row original WITH ITS LABELS -- class-conditional CDF gap,
     distance to the y0/y1 medians, a binned original target mean, and a Gaussian-KDE
     log-likelihood ratio.

Channel (2) is what this workspace has never built. w185/w186 measured the UNLABELLED
original CDF as columns (`cdfd_*`, +87.7e-6 solo, null in the stack). Every feature here
that carries `_y0`, `_y1`, `_gap`, `_mean` or `_kde_llr` in its name uses the original's
LABEL column, which is a strictly larger channel than the CDF, and is leakage-free with
respect to the competition target because the original rows that also appear in train are
dropped by row hash before any reference is fitted.

The builder is fold-independent by construction: nothing here touches the competition
labels. The exact-value target encoding is the one piece that does, and it is built
INSIDE each outer fold by the model script, never here.

Outputs (experiments/w187_feat/):
    Xc_train.npy / Xc_test.npy      float32 continuous block
    cont_cols.json                  its column names
    cat_train.npy / cat_test.npy    int32 exact-value category codes, 12 columns
    cat_cards.json                  cardinality per categorical column
    keys_train.npy / keys_test.npy  the same exact values as int32 codes for the TE
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, TARGET  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w187_feat")
os.makedirs(OUT, exist_ok=True)

BASE_NUM = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours",
            "work_study_hours", "sleep_hours", "notifications_per_day",
            "app_opens_per_day", "weekend_screen_time"]
BASE_CAT = ["gender", "stress_level", "academic_work_impact"]
RAW = BASE_NUM + BASE_CAT

ORIG_N_BINS = 20
KDE_BW_MIN, KDE_BW_MAX, KDE_CLIP = 0.10, 1.00, 20.0

CDF_SRC = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours"]
Q50_SRC = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
           "notifications_per_day", "app_opens_per_day"]
Q50_Y1_SRC = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
              "app_opens_per_day"]
MEAN_SRC = ["daily_screen_time_hours", "weekend_screen_time", "notifications_per_day",
            "app_opens_per_day"]
GAP_SRC = ["daily_screen_time_hours", "weekend_screen_time", "social_media_hours",
           "notifications_per_day", "app_opens_per_day"]
KDE_SRC = ["weekend_screen_time", "notifications_per_day", "app_opens_per_day"]


def static_features(df):
    """The notebook's ratio / difference / balance / domain block, in pandas."""
    o = pd.DataFrame(index=df.index)
    def ratio(n, d, name):
        num, den = df[n], df[d]
        v = num / den
        o[name] = v.where(np.isfinite(v) & (den != 0), np.nan)
    ratio("social_media_hours", "daily_screen_time_hours", "social_share_of_screen")
    ratio("gaming_hours", "daily_screen_time_hours", "gaming_share_of_screen")
    ratio("daily_screen_time_hours", "sleep_hours", "screen_to_sleep_ratio")
    ratio("social_media_hours", "sleep_hours", "social_to_sleep_ratio")
    ratio("social_media_hours", "work_study_hours", "social_to_work_ratio")
    ratio("daily_screen_time_hours", "work_study_hours", "screen_to_work_ratio")
    ratio("weekend_screen_time", "sleep_hours", "weekend_screen_to_sleep_ratio")
    ratio("work_study_hours", "daily_screen_time_hours", "work_study_share_of_screen")
    o["screen_minus_work"] = df["daily_screen_time_hours"] - df["work_study_hours"]
    awake = 24.0 - df["sleep_hours"]
    v = df["daily_screen_time_hours"] / awake
    o["screen_share_of_awake"] = v.where(np.isfinite(v) & (awake != 0), np.nan)
    o["total_breakdown_hours"] = (df["social_media_hours"].fillna(0)
                                  + df["gaming_hours"].fillna(0)
                                  + df["work_study_hours"].fillna(0))
    o["unaccounted_screen_time"] = df["daily_screen_time_hours"] - o["total_breakdown_hours"]
    return o


def empirical_cdf(values, ref_sorted):
    out = np.full(len(values), np.nan)
    ok = np.isfinite(values)
    if len(ref_sorted):
        out[ok] = np.searchsorted(ref_sorted, values[ok], side="right") / len(ref_sorted)
    return out


def quantile_edges(v, n_bins):
    v = v[np.isfinite(v)]
    if not len(v):
        return np.array([-np.inf, np.inf])
    e = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1))).astype(float)
    if len(e) < 2:
        return np.array([-np.inf, np.inf])
    e[0], e[-1] = -np.inf, np.inf
    return e


def assign_bins(v, edges):
    b = np.full(len(v), -1, dtype=np.int32)
    ok = np.isfinite(v)
    b[ok] = np.clip(np.searchsorted(edges, v[ok], side="right") - 1, 0, len(edges) - 2)
    return b


def silverman(v):
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return 0.30
    sd = float(np.std(v, ddof=1))
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float((q75 - q25) / 1.34)
    scales = [s for s in (sd, iqr) if np.isfinite(s) and s > 1e-12]
    scale = min(scales) if scales else 1.0
    return float(np.clip(0.9 * scale * len(v) ** (-0.2), KDE_BW_MIN, KDE_BW_MAX))


def fit_orig_refs(orig):
    y = orig[TARGET].to_numpy().astype(np.int8)
    gr = float(y.mean())
    refs = {"cdf": {}, "class_cdf": {}, "q50": {}, "mean": {}, "kde": {}}
    for c in CDF_SRC:
        v = orig[c].to_numpy(dtype=float)
        refs["cdf"][c] = np.sort(v[np.isfinite(v)])
    for c in GAP_SRC:
        v = orig[c].to_numpy(dtype=float)
        refs["class_cdf"][c] = {"y0": np.sort(v[(y == 0) & np.isfinite(v)]),
                                "y1": np.sort(v[(y == 1) & np.isfinite(v)])}
    for c in Q50_SRC:
        v = orig[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        r = {"all": float(np.median(v[ok])), "y0": float(np.median(v[ok & (y == 0)]))}
        if c in Q50_Y1_SRC:
            r["y1"] = float(np.median(v[ok & (y == 1)]))
        refs["q50"][c] = r
    for c in MEAN_SRC:
        v = orig[c].to_numpy(dtype=float)
        edges = quantile_edges(v, ORIG_N_BINS)
        b = assign_bins(v, edges)
        means = np.full(len(edges) - 1, gr)
        for k in range(len(means)):
            m = b == k
            if m.any():
                means[k] = float(y[m].mean())
        refs["mean"][c] = {"edges": edges, "means": means, "global_rate": gr}
    for c in KDE_SRC:
        v = orig[c].to_numpy(dtype=float)
        ok = np.isfinite(v)
        mu, sd = float(v[ok].mean()), float(v[ok].std())
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        z0 = ((v[(y == 0) & ok] - mu) / sd).reshape(-1, 1)
        z1 = ((v[(y == 1) & ok] - mu) / sd).reshape(-1, 1)
        refs["kde"][c] = {
            "mean": mu, "std": sd,
            "kde0": KernelDensity(kernel="gaussian", bandwidth=silverman(z0)).fit(z0),
            "kde1": KernelDensity(kernel="gaussian", bandwidth=silverman(z1)).fit(z1),
        }
    return refs


def orig_features(df, refs):
    o = pd.DataFrame(index=df.index)
    for c in CDF_SRC:
        v = df[c].to_numpy(dtype=float)
        o[f"{c}__orig_cdf"] = empirical_cdf(v, refs["cdf"][c])
    for c in GAP_SRC:
        v = df[c].to_numpy(dtype=float)
        o[f"{c}__orig_cdf_gap"] = (empirical_cdf(v, refs["class_cdf"][c]["y0"])
                                   - empirical_cdf(v, refs["class_cdf"][c]["y1"]))
    for c in Q50_SRC:
        v = df[c].to_numpy(dtype=float)
        r = refs["q50"][c]
        o[f"{c}__orig_q50_distance"] = v - r["all"]
        o[f"{c}__orig_q50_distance_y0"] = v - r["y0"]
        if "y1" in r:
            o[f"{c}__orig_q50_distance_y1"] = v - r["y1"]
    for c in MEAN_SRC:
        v = df[c].to_numpy(dtype=float)
        r = refs["mean"][c]
        b = assign_bins(v, r["edges"])
        res = np.full(len(v), r["global_rate"])
        ok = b >= 0
        res[ok] = r["means"][b[ok]]
        o[f"{c}__orig_mean"] = res
    for c in KDE_SRC:
        v = df[c].to_numpy(dtype=float)
        r = refs["kde"][c]
        res = np.full(len(v), np.nan)
        ok = np.isfinite(v)
        if ok.any():
            # The columns are a quantisation lattice, so scoring the ~10^3 DISTINCT values
            # and mapping back is the same number as scoring all 553k rows, ~200x cheaper.
            uv, inv = np.unique(v[ok], return_inverse=True)
            z = ((uv - r["mean"]) / r["std"]).reshape(-1, 1)
            llr = np.clip(r["kde1"].score_samples(z) - r["kde0"].score_samples(z),
                          -KDE_CLIP, KDE_CLIP)
            res[ok] = llr[inv]
        o[f"{c}__orig_kde_llr"] = res
    return o


def exact_strings(df):
    """The exact printed value of every raw column, missing folded to one level."""
    out = {}
    for c in RAW:
        s = df[c].astype("string")
        out[c] = s.fillna("__MISSING__").astype(str)
    return pd.DataFrame(out, index=df.index)


def main():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    orig = pd.read_csv(os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv"))
    print(f"train {tr.shape}  test {te.shape}  orig {orig.shape}", flush=True)

    orig = orig[orig[TARGET].notna()].copy()
    orig[TARGET] = orig[TARGET].astype(np.int8)

    # drop original rows that are byte-identical to a competition training row, so no
    # reference below is fitted on a row whose label we also train on
    def rowkey(df):
        parts = []
        for c in RAW:
            if c in BASE_NUM:
                parts.append(pd.to_numeric(df[c], errors="coerce").round(8)
                             .fillna(-999999.0).astype(str))
            else:
                parts.append(df[c].astype("string").fillna("__MISSING__").astype(str))
        return parts[0].str.cat(parts[1:], sep="|")
    tk, ok_ = set(rowkey(tr)), rowkey(orig)
    before = len(orig)
    orig = orig[~ok_.isin(tk)].drop_duplicates(subset=RAW, keep="first")
    print(f"orig after overlap drop + dedupe: {len(orig)}  (was {before})", flush=True)

    refs = fit_orig_refs(orig)

    blocks = {}
    for name, df in (("train", tr), ("test", te)):
        cont = pd.concat([static_features(df), orig_features(df, refs)], axis=1)
        blocks[name] = cont
    cont_cols = list(blocks["train"].columns)
    assert cont_cols == list(blocks["test"].columns)

    # exact-value frequency over train+test together (transductive, target-free)
    Str, Ste = exact_strings(tr), exact_strings(te)
    for c in RAW:
        counts = pd.concat([Str[c], Ste[c]], ignore_index=True).value_counts(dropna=False)
        for name, S in (("train", Str), ("test", Ste)):
            f = S[c].map(counts).fillna(0.0).to_numpy(dtype=np.float32)
            blocks[name][f"{c}_freq"] = f
            blocks[name][f"{c}_logfreq"] = np.log1p(f)
    cont_cols = list(blocks["train"].columns)

    # median fill on TRAIN medians, applied to both sides
    Xtr = blocks["train"].replace([np.inf, -np.inf], np.nan)
    Xte = blocks["test"].replace([np.inf, -np.inf], np.nan)
    med = Xtr.median()
    med[~np.isfinite(med)] = 0.0
    Xtr = Xtr.fillna(med).astype(np.float32)
    Xte = Xte.fillna(med).astype(np.float32)

    # exact-value categories, one vocabulary shared by train and test
    cat_tr = np.zeros((len(tr), len(RAW)), dtype=np.int32)
    cat_te = np.zeros((len(te), len(RAW)), dtype=np.int32)
    cards = {}
    for i, c in enumerate(RAW):
        cats = pd.Index(pd.concat([Str[c], Ste[c]], ignore_index=True).unique())
        dt = pd.CategoricalDtype(categories=cats)
        cat_tr[:, i] = Str[c].astype(dt).cat.codes.to_numpy()
        cat_te[:, i] = Ste[c].astype(dt).cat.codes.to_numpy()
        cards[c] = int(len(cats))
    assert cat_tr.min() >= 0 and cat_te.min() >= 0

    np.save(os.path.join(OUT, "Xc_train.npy"), Xtr.to_numpy())
    np.save(os.path.join(OUT, "Xc_test.npy"), Xte.to_numpy())
    np.save(os.path.join(OUT, "cat_train.npy"), cat_tr)
    np.save(os.path.join(OUT, "cat_test.npy"), cat_te)
    json.dump(cont_cols, open(os.path.join(OUT, "cont_cols.json"), "w"), indent=1)
    json.dump(cards, open(os.path.join(OUT, "cat_cards.json"), "w"), indent=1)
    print(f"continuous {Xtr.shape[1]} cols, categorical {len(RAW)} cols", flush=True)
    print("cardinalities:", cards, flush=True)
    print(f"[saved] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
