"""How well can the public slice resolve a gap BETWEEN TEAMS?

WHY THIS EXISTS
---------------
Today's brief states the 18e-5 gap to MILANFX as established fact: "the measurement noise
floor established in this workspace is ~5e-5, so the gap is roughly 3.6x the noise floor —
it is real."

That 5e-5 was measured on OUR OWN files, which correlate ~0.9999 with each other (w14a:
spearman 0.99997 between two rebuilds of the same pick; w14b: paired-slice sd 22-32e-6 for
within-pack transform contrasts).  The sd of a paired AUC difference is

    sd(gap) = sd(single) * sqrt(2 * (1 - rho))

so the resolvable difference is a function of how ALIKE the two candidates are.  Two
independently built solutions from different teams are not 0.9999 alike.  Nobody in this
workspace has ever measured the cross-team correlation, so nobody has ever priced the
cross-team noise floor, so the "3.6x the noise floor" claim has never been checked against
the right floor.

WHAT THIS MEASURES
------------------
1. rho between our shipped test predictions and other teams' published test predictions
   (the actual scoring domain, 296,302 rows).
2. sd of the public-slice AUC gap, by resampling the leaderboard's geometry out of the
   691,369 LABELLED training rows: draw a pseudo-test of 296,302, cut a 20% public slice
   (59,260 rows), score both vectors on the slice, take the difference.  Same construction
   as w14b_slicenoise2, which is the workspace's calibrated instrument for this.
3. The two together: how many sigma is 18e-5 at the measured cross-team rho, and at a
   range of rho for the teams whose predictions we cannot see.

Internal within-pack pairs are included as controls: they should reproduce w14b's 22-32e-6
and they should sit at the high-rho end of the same curve.

    w15a_crossteam.py --reps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
EXT = os.path.join(ROOT, "data", "ext")
N_TEST = 296_302
F_PUBLIC = 0.20

REF = "blend159av_h3"          # the CV pick, and the file whose LB is 0.97105/0.97106


# --------------------------------------------------------------------- fast AUC
def prep(v):
    o = np.argsort(v, kind="stable")
    return o, np.ascontiguousarray(v[o])


def _auc_sorted(s, yy):
    n = s.size
    n1 = float(yy.sum())
    n0 = float(n) - n1
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    b = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    ends = np.concatenate((b[1:], [n]))
    avg = (b + ends + 1) * 0.5
    pos = np.add.reduceat(yy, b)
    return (float((avg * pos).sum()) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def subset_auc(o, vs, ys, mask):
    sel = mask[o]
    return _auc_sorted(vs[sel], ys[sel])


# --------------------------------------------------------------- competitor loading
def rank01(v):
    return rankdata(v) / (len(v) + 1.0)


def load_competitors(n_pool):
    """Return {name: (oof_vector, test_vector_or_None, provenance)}."""
    out = {}

    # -- our own, as controls spanning a range of similarity to REF
    for nm in ["blend158_logit", "blend150fx_hybrid", "blend156_h3"]:
        p = os.path.join(SUB, f"oof_{nm}.npy")
        if os.path.exists(p):
            t = os.path.join(SUB, f"{nm}.csv")
            tv = pd.read_csv(t)["addicted_label"].to_numpy() if os.path.exists(t) else None
            out[f"OURS:{nm}"] = (np.load(p), tv, "same 150-160 member stack, different transform/set")

    # -- najiama: a real other team's published blends (LB 0.97092-0.97101 family)
    nd = os.path.join(EXT, "najiama_predicting-smartphone-addiction-oof-submission-csv")
    for tag in ["18", "12"]:
        fo = os.path.join(nd, f"{tag}_blend_oof_predictions.csv")
        ft = os.path.join(nd, f"{tag}_blend_submission.csv")
        if os.path.exists(fo):
            do = pd.read_csv(fo)
            col = [c for c in do.columns if c != "id"][0]
            v = do.sort_values("id")[col].to_numpy() if "id" in do.columns else do[col].to_numpy()
            if len(v) != n_pool:
                print(f"  skip najiama {tag}: {len(v)} rows != {n_pool}")
                continue
            tv = None
            if os.path.exists(ft):
                dt = pd.read_csv(ft).sort_values("id")
                tv = dt["addicted_label"].to_numpy()
            out[f"NAJI:{tag}_blend"] = (v, tv, "other team, published blend (weights fit in-sample; see RESEARCH)")

    # -- boltuzamaki: 47 honest streams -> an independent rank-average
    bd = os.path.join(EXT, "boltuzamaki_s6e8-oof-prediction-library")
    fo, ft = os.path.join(bd, "oof_predictions.parquet"), os.path.join(bd, "test_predictions.parquet")
    if os.path.exists(fo):
        do = pd.read_parquet(fo)
        cols = [c for c in do.columns if c not in ("id", "row_id", "target")]
        y_ = None
        lp = os.path.join(bd, "train_labels.parquet")
        if os.path.exists(lp):
            dl = pd.read_parquet(lp)
            y_ = dl[[c for c in dl.columns if c not in ("id", "row_id")][0]].to_numpy()
        aucs = {}
        for c in cols:
            v = do[c].to_numpy()
            if len(v) != n_pool or not np.isfinite(v).all():
                continue
            o, vs = prep(v)
            aucs[c] = _auc_sorted(vs, y_[o])
        top = sorted(aucs, key=aucs.get, reverse=True)[:12]
        v = np.mean([rank01(do[c].to_numpy()) for c in top], axis=0)
        tv = None
        if os.path.exists(ft):
            dt = pd.read_parquet(ft)
            if all(c in dt.columns for c in top):
                tv = np.mean([rank01(dt[c].to_numpy()) for c in top], axis=0)
        out["BOLT:rankavg_top12"] = (v, tv, "other team's 47-stream library, honest OOF, top-12 rank-average")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "w15a_crossteam.json"))
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)

    ref = np.load(os.path.join(SUB, f"oof_{REF}.npy"))
    ref_test = pd.read_csv(os.path.join(SUB, f"{REF}.csv")).sort_values("id")["addicted_label"].to_numpy()
    assert len(ref) == n_pool and len(ref_test) == N_TEST

    comp = load_competitors(n_pool)
    print(f"pool {n_pool:,} labelled rows | pseudo-test {N_TEST:,} | public slice "
          f"{int(N_TEST * F_PUBLIC):,} | {len(comp)} competitors\n")

    # correctness check of the fast AUC against scipy, same gate w14b uses
    o0, vs0 = prep(ref)
    ys0 = y[o0]
    m = np.zeros(n_pool, bool)
    m[np.random.default_rng(0).permutation(n_pool)[:50_000]] = True
    n1 = int(y[m].sum())
    n0 = int(m.sum()) - n1
    scipy_ref = (rankdata(ref[m])[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    fast = subset_auc(o0, vs0, ys0, m)
    assert abs(scipy_ref - fast) < 1e-12, (scipy_ref, fast)
    print(f"fast AUC checked against scipy: {fast:.12f} == {scipy_ref:.12f}\n")

    PRE = {"REF": (o0, vs0, ys0)}
    for nm, (v, _, _) in comp.items():
        o, vs = prep(v)
        PRE[nm] = (o, vs, y[o])

    # --- rank correlations
    rk_ref_oof = rankdata(ref)
    rk_ref_test = rankdata(ref_test)
    rows = {}
    for nm, (v, tv, prov) in comp.items():
        rho_oof = float(np.corrcoef(rk_ref_oof, rankdata(v))[0, 1])
        rho_test = float(np.corrcoef(rk_ref_test, rankdata(tv))[0, 1]) if tv is not None else float("nan")
        rows[nm] = dict(prov=prov, rho_oof=rho_oof, rho_test=rho_test)

    # --- slice simulation with the leaderboard's partition geometry
    rng = np.random.default_rng(a.seed)
    n_pub = int(round(N_TEST * F_PUBLIC))
    gaps = {nm: [] for nm in comp}
    singles = []
    for _ in range(a.reps):
        perm = rng.permutation(n_pool)
        pub = np.zeros(n_pool, bool)
        pub[perm[:n_pub]] = True
        au_ref = subset_auc(*PRE["REF"], pub)
        singles.append(au_ref)
        for nm in comp:
            gaps[nm].append(au_ref - subset_auc(*PRE[nm], pub))

    sd_single = float(np.std(singles, ddof=1))
    full_ref = _auc_sorted(vs0, ys0)
    print(f"{REF} pooled AUC {full_ref:.6f} | sd of its own {n_pub:,}-row slice AUC "
          f"{sd_single*1e6:.1f}e-6\n")

    print(f"{'competitor':28} {'rho(test)':>10} {'rho(oof)':>10} {'pooled AUC':>11} "
          f"{'mean gap':>10} {'sd(gap)':>9} {'18e-5 in sd':>12}")
    print("-" * 96)
    res = {}
    for nm in comp:
        g = np.array(gaps[nm])
        sd = float(g.std(ddof=1))
        o, vs, ys = PRE[nm]
        pooled = _auc_sorted(vs, ys)
        sigma = 18e-5 / sd
        rows[nm].update(pooled_auc=pooled, mean_gap=float(g.mean()), sd_gap=sd,
                        sigma_of_18e5=sigma)
        res[nm] = rows[nm]
        print(f"{nm:28} {rows[nm]['rho_test']:>10.5f} {rows[nm]['rho_oof']:>10.5f} "
              f"{pooled:>11.6f} {g.mean()*1e6:>9.1f}e-6 {sd*1e6:>8.1f}e-6 {sigma:>11.2f}")

    # --- the algebraic curve, so the answer can be read off at any assumed rho
    print(f"\nsd(gap) = sd(single) * sqrt(2(1-rho)) with sd(single) = {sd_single*1e6:.1f}e-6")
    print(f"{'assumed rho':>12} {'predicted sd(gap)':>19} {'18e-5 in sd':>12}")
    curve = {}
    for rho in [0.99999, 0.9999, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.90]:
        s = sd_single * np.sqrt(2 * (1 - rho))
        curve[rho] = dict(sd_gap=float(s), sigma=float(18e-5 / s))
        print(f"{rho:>12.5f} {s*1e6:>18.1f}e-6 {18e-5/s:>11.2f}")

    json.dump(dict(ref=REF, reps=a.reps, n_pub=n_pub, sd_single=sd_single,
                   ref_pooled_auc=full_ref, competitors=res, curve={str(k): v for k, v in curve.items()}),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
