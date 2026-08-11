"""Where is the best stack wrong, and can any feature see it?

THE QUESTION
------------
`blend158_h3` is 158 members deep and every feature-engineering idea in the journal has
already been folded into the members. The honest way to ask whether anything is LEFT is a
conditional-independence test: given the stack's own score z, does the target still depend
on a raw feature x?

    y  _||_  x  |  z      <- if this holds for every x, the stack has extracted
                             everything those columns carry and no feature can help.

Stratifying on quantile bins of z is what makes it a fair question. Marginally, every
feature is wildly associated with y; that is not news and is not error. What matters is
whether two rows the stack scores IDENTICALLY have different addiction rates because they
differ in x. That residual is the only thing a new feature could ever recover.

THREE STATISTICS PER FEATURE, cheapest first
--------------------------------------------
1. `chi2/df`  -- omnibus conditional-independence test on the (z-bin x x-bin) table,
   allowing the x-effect to differ by z-bin. 1.0 is the null. Detects anything.
2. `pooled z` -- the largest single-x-bin standardised deviation under a common effect.
   Says WHERE in x the miss is, which is what a feature would have to encode.
3. `dAUC`     -- the decision-relevant one. Fit the empirical per-cell log-odds
   correction on 4 folds, apply it to the 5th, and measure the AUC change. This is an
   honest out-of-fold estimate of what a perfect exploitation of that residual is worth,
   in the same units the journal records everything else in.

Permuted control columns are scanned alongside the real ones so the null is measured on
this exact pipeline rather than assumed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, COMP, DAILY, NUM, SUB, TARGET, get_folds, load_raw  # noqa: E402

NAN_BIN = 0          # NaN always gets its own bin, so missingness is tested too
SMOOTH = 200.0       # cell prior weight for the log-odds correction


def qbin(v, k):
    """Equal-count bins of a numeric column; bin 0 is reserved for NaN."""
    v = np.asarray(v, dtype="float64")
    out = np.zeros(len(v), dtype="int32")
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return out, 1
    r = pd.Series(v[ok]).rank(method="average").to_numpy()
    b = np.minimum((r / (len(r) + 1e-9) * k).astype("int32"), k - 1)
    # collapse to the bins actually populated (ties in a lattice column can empty some)
    uniq, inv = np.unique(b, return_inverse=True)
    out[ok] = inv + 1
    return out, len(uniq) + 1


def catbin(v):
    s = pd.Series(v).astype("object")
    m = s.isna()
    codes = pd.Categorical(s.fillna("__NA__")).codes.astype("int32")
    # remap so NaN is bin 0
    lv = pd.Categorical(s.fillna("__NA__")).categories
    na_code = int(np.where(lv == "__NA__")[0][0]) if "__NA__" in list(lv) else -1
    out = codes + 1
    if na_code >= 0:
        out[codes == na_code] = NAN_BIN
        out[codes > na_code] -= 1
    del m
    return out.astype("int32"), int(out.max()) + 1


def cells(b, k, B, K):
    return b.astype("int64") * K + k.astype("int64")


def scan_one(c, y, B, K, folds, logitz):
    """Return (chi2/df, top pooled |z| and its bin, honest dAUC) for one binned feature."""
    n = np.bincount(c, minlength=B * K).astype("float64")
    o = np.bincount(c, weights=y.astype("float64"), minlength=B * K)
    n2, o2 = n.reshape(B, K), o.reshape(B, K)
    rn, ro = n2.sum(1), o2.sum(1)
    p = np.divide(ro, rn, out=np.full(B, y.mean()), where=rn > 0)
    e = n2 * p[:, None]
    v = n2 * (p * (1 - p))[:, None]
    live = v > 1e-9
    resid = o2 - e
    chi2 = float((resid[live] ** 2 / v[live]).sum())
    df = max(int(live.sum() - B), 1)

    sk = resid.sum(0)
    vk = v.sum(0)
    zk = np.divide(sk, np.sqrt(vk), out=np.zeros(K), where=vk > 1e-9)
    top = int(np.argmax(np.abs(zk)))

    # honest out-of-fold empirical correction
    adj = np.zeros(len(y))
    for itr, iva in folds:
        nt = np.bincount(c[itr], minlength=B * K).astype("float64")
        ot = np.bincount(c[itr], weights=y[itr].astype("float64"), minlength=B * K)
        nt2, ot2 = nt.reshape(B, K), ot.reshape(B, K)
        pt = np.divide(ot2.sum(1), nt2.sum(1), out=np.full(B, y[itr].mean()),
                       where=nt2.sum(1) > 0)
        pc = (ot2 + SMOOTH * pt[:, None]) / (nt2 + SMOOTH)
        pc = np.clip(pc, 1e-6, 1 - 1e-6)
        pb = np.clip(pt, 1e-6, 1 - 1e-6)
        d = (np.log(pc / (1 - pc)) - np.log(pb / (1 - pb))[:, None]).ravel()
        adj[iva] = d[c[iva]]
    dauc = roc_auc_score(y, logitz + adj) - roc_auc_score(y, logitz)
    return chi2 / df, float(zk[top]), top, float(dauc), int(live.sum())


def build_candidates(tr, rng):
    """Every column, digit channel, ratio and identity quantity worth asking about."""
    f = {}
    for c in NUM:
        f[c] = tr[c].to_numpy("float64")
    for c in CAT:
        f[c] = tr[c].to_numpy()

    d = tr[DAILY].to_numpy("float64")
    comp = tr[COMP].to_numpy("float64")
    comp_sum = np.nansum(comp, axis=1)
    n_known = np.isfinite(comp).sum(1)
    slack = np.where(np.isfinite(d) & (n_known == 3), d - comp_sum, np.nan)
    f["slack_daily_minus_comp"] = slack
    f["comp_share"] = np.where(np.isfinite(d) & (n_known == 3) & (d > 0),
                               comp_sum / np.maximum(d, 1e-6), np.nan)
    f["n_missing_all"] = tr[NUM + CAT].isna().sum(1).to_numpy("float64")
    f["n_screen_missing"] = tr[[DAILY] + COMP].isna().sum(1).to_numpy("float64")
    f["weekend_minus_daily"] = (tr["weekend_screen_time"] - tr[DAILY]).to_numpy("float64")
    f["notif_per_open"] = (tr["notifications_per_day"]
                           / (tr["app_opens_per_day"] + 1.0)).to_numpy("float64")
    f["sleep_plus_daily"] = (tr["sleep_hours"] + tr[DAILY]).to_numpy("float64")
    f["free_hours"] = (24.0 - tr["sleep_hours"] - tr[DAILY]).to_numpy("float64")
    for c in ["social_media_hours", "gaming_hours", "work_study_hours",
              "weekend_screen_time"]:
        f[f"ratio_{c}"] = (tr[c] / np.maximum(d, 1e-6)).to_numpy("float64")

    # the quantisation lattice: first and second decimal digit of every numeric
    for c in NUM:
        v = tr[c].to_numpy("float64")
        f[f"d1_{c}"] = np.where(np.isfinite(v), np.round(v * 10) % 10, np.nan)
        f[f"d2_{c}"] = np.where(np.isfinite(v), np.round(v * 100) % 10, np.nan)

    # permuted controls: identical marginal distribution, zero conditional information
    for i, c in enumerate(["daily_screen_time_hours", "age", "notifications_per_day"]):
        f[f"CTRL_perm_{c}"] = rng.permutation(tr[c].to_numpy("float64"))
    f["CTRL_uniform"] = rng.random(len(tr))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", default="blend158_h3")
    ap.add_argument("--zbins", type=int, default=64)
    ap.add_argument("--xbins", type=int, default=12)
    ap.add_argument("--pairs", action="store_true", help="also scan all NUM x NUM pairs")
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    z = np.load(os.path.join(SUB, f"oof_{a.stack}.npy"))
    assert len(z) == len(y)

    # rank -> logit: monotone, so base AUC is untouched, but the scale is additive-friendly
    r = pd.Series(z).rank(method="average").to_numpy() / (len(z) + 1.0)
    logitz = np.log(r / (1 - r))
    zb, B = qbin(z, a.zbins)
    zb -= 1                      # z is never NaN, so drop the reserved slot
    B -= 1
    base = roc_auc_score(y, z)
    print(f"stack {a.stack}  OOF AUC {base:.6f}   z-bins {B} (~{len(y)//B} rows each)",
          flush=True)

    folds = get_folds(y)
    rng = np.random.default_rng(0)
    cand = build_candidates(tr, rng)

    rows = []
    for name, v in cand.items():
        if v.dtype == object or isinstance(v[0], str):
            k, K = catbin(v)
        else:
            k, K = qbin(v, a.xbins)
        if K < 2:
            continue
        c = cells(zb, k, B, K)
        chi, zt, tb, da, nc = scan_one(c, y, B, K, folds, logitz)
        rows.append(dict(feature=name, K=K, chi2_df=chi, top_z=zt, top_bin=tb,
                         dAUC=da, cells=nc))
        print(f"  {name:34s} K={K:3d}  chi2/df {chi:7.3f}  top|z| {zt:+7.2f}"
              f" @bin{tb:3d}  dAUC {da:+.6f}", flush=True)

    if a.pairs:
        print("\n--- 2-way cells (8x8 coarse), interaction structure ---", flush=True)
        bins = {c: qbin(tr[c].to_numpy("float64"), 8)[0] for c in NUM}
        for i, l in enumerate(NUM):
            for rr in NUM[i + 1:]:
                kl, kr = bins[l], bins[rr]
                Kl = int(kl.max()) + 1
                k = kl * (int(kr.max()) + 1) + kr
                K = int(k.max()) + 1
                del Kl
                c = cells(zb, k, B, K)
                chi, zt, tb, da, nc = scan_one(c, y, B, K, folds, logitz)
                rows.append(dict(feature=f"PAIR_{l}__{rr}", K=K, chi2_df=chi, top_z=zt,
                                 top_bin=tb, dAUC=da, cells=nc))
                print(f"  {l[:16]:16s} x {rr[:16]:16s} chi2/df {chi:6.3f}"
                      f"  top|z| {zt:+6.2f}  dAUC {da:+.6f}", flush=True)

    out = pd.DataFrame(rows).sort_values("dAUC", ascending=False)
    p = os.path.join(ROOT, "experiments", f"erroran_{a.stack}.csv")
    out.to_csv(p, index=False)
    print(f"\nwrote {p}")
    ctrl = out[out.feature.str.startswith("CTRL")]
    print("\ncontrols (the null on this exact pipeline):")
    print(ctrl[["feature", "chi2_df", "top_z", "dAUC"]].to_string(index=False))
    print("\ntop 12 by dAUC:")
    print(out.head(12)[["feature", "K", "chi2_df", "top_z", "dAUC"]].to_string(index=False))


if __name__ == "__main__":
    main()
