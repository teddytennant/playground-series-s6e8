"""Three things the priced recommendation needs and `w15i_cvlb.py` does not cover.

1.  THE CV->LB MAP, READ WITH TRANSFORM-FAMILY FIXED EFFECTS.
    w14a's headline is "CV has stopped predicting LB" -- pooled slope +0.15, R^2 0.035.
    But the same run measured a per-family residual offset spanning 51e-6, which is 7x the
    CV spread of the top pack.  Pooling a strong family offset into a regression whose
    regressor spans 1e-5 is exactly the setup where an omitted categorical destroys a
    within-group slope.  So refit with family fixed effects (demean CV and LB inside each
    transform family) and ask whether CV predicts LB *among comparable files*.  Also count
    within-family concordant pairs, which is quantisation-robust and assumption-free.

2.  THE OPTION VALUE OF THE PAIR.  Kaggle scores a selection as the MAX over the selected
    entries.  Two entries with identical CV are therefore NOT equivalent to one: the pair's
    expected private score rises with how much the two disagree.  Nothing in this workspace
    has ever priced that, and the current recommendation (`blend159av_h3` +
    `blend160origm_h3`) is two files at spearman ~0.99999.  Simulate the private slice and
    rank every pair from the CV-top cluster by E[max].

3.  WHAT THE COST IS WORTH IN PLACES.  An AUC number is not a decision until it is
    converted into leaderboard positions at the observed local density.

    w15i_pick.py --reps 600
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

from w15i_cvlb import classify, full_auc, prep, subset_auc  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
HERE = os.path.join(ROOT, "experiments")
N_TEST = 296_302

# every file at the top of the CV table -- all of them scored exactly 0.97105 on the
# public slice, so the public reading carries no information that separates them and an
# unconditional simulation is the right instrument.
CLUSTER_MIN_CV = 0.970046
PICK = ("blend159av_h3", "blend160origm_h3")
TIE = ("blend156", "blend158", "blend158_logit", "blend159av")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=151)
    ap.add_argument("--f", type=float, default=0.20)
    ap.add_argument("--lb", default=sorted(glob.glob("/tmp/w15i_lb/*publicleaderboard*.csv"))[-1])
    ap.add_argument("--out", default=os.path.join(HERE, "w15i_pick.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = {}

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)

    s = pd.read_csv(os.path.join(HERE, "w15i_subs_raw.csv"))
    s = s[s["status"].astype(str).str.endswith("COMPLETE")].copy()
    s["stem"] = s["fileName"].str.replace(".csv", "", regex=False)
    rows = []
    for _, r in s.iterrows():
        p = os.path.join(SUB, f"oof_{r['stem']}.npy")
        if os.path.exists(p):
            rows.append(dict(name=r["stem"], lb=float(r["publicScore"]),
                             cv=full_auc(np.load(p), y)))
    t = pd.DataFrame(rows).drop_duplicates("name")
    t[["mset", "kind"]] = t["name"].apply(lambda n: pd.Series(classify(n)))
    t["big"] = ~t["name"].str.startswith("stack_pub7") & ~t["name"].str.startswith("stack_pub8")
    big = t[t["big"]].copy()

    # ------------------------------------------- 1. the map, with family fixed effects
    print("=" * 78)
    print("1. CV -> LB WITH TRANSFORM-FAMILY FIXED EFFECTS")
    print("=" * 78)
    pooled_s, pooled_i = np.polyfit(big["cv"], big["lb"], 1)
    print(f"\n  pooled (what w14a fitted):  slope {pooled_s:+.3f}   "
          f"R^2 {1 - (big['lb'] - (pooled_s*big['cv']+pooled_i)).var()/big['lb'].var():.3f}")

    g = big.groupby("kind")
    big["cv_d"] = big["cv"] - g["cv"].transform("mean")
    big["lb_d"] = big["lb"] - g["lb"].transform("mean")
    keep = big[g["cv"].transform("count") >= 2]
    num = float((keep["cv_d"] * keep["lb_d"]).sum())
    den = float((keep["cv_d"] ** 2).sum())
    beta = num / den
    resid = keep["lb_d"] - beta * keep["cv_d"]
    dfree = len(keep) - keep["kind"].nunique() - 1
    se = float(np.sqrt((resid ** 2).sum() / dfree / den))
    print(f"  within-family (fixed effects), n={len(keep)}, "
          f"{keep['kind'].nunique()} families, df={dfree}")
    print(f"    slope  {beta:+.3f}  +/- {se:.3f} (se)   t = {beta/se:+.2f}")
    print(f"    within R^2 {1 - resid.var()/keep['lb_d'].var():.3f}   "
          f"residual sd {resid.std(ddof=1):.2e} = {resid.std(ddof=1)/1e-5:.2f} LB grid steps")
    print(f"    (a slope of 1 is the null 'LB is an unbiased noisy read of the same "
          f"quantity CV measures')")

    print("\n  per-family slope, and the CV span each family can actually resolve:")
    print(f"    {'kind':8s} {'n':>3s} {'CV span':>9s} {'LB span':>9s} {'slope':>8s}  concordant pairs")
    tot_c = tot_d = 0
    for k, gg in big.groupby("kind"):
        span_cv = gg["cv"].max() - gg["cv"].min()
        span_lb = gg["lb"].max() - gg["lb"].min()
        sl = np.polyfit(gg["cv"], gg["lb"], 1)[0] if len(gg) > 1 and span_cv > 0 else float("nan")
        c = d = 0
        for i, j in itertools.combinations(range(len(gg)), 2):
            dc = gg["cv"].iloc[i] - gg["cv"].iloc[j]
            dl = gg["lb"].iloc[i] - gg["lb"].iloc[j]
            if dl == 0 or dc == 0:
                continue
            if np.sign(dc) == np.sign(dl):
                c += 1
            else:
                d += 1
        tot_c += c
        tot_d += d
        print(f"    {k:8s} {len(gg):3d} {span_cv:9.2e} {span_lb:9.2e} {sl:8.2f}  {c} agree / {d} disagree")
    n_res = tot_c + tot_d
    print(f"\n  within-family pairs the LB grid can resolve at all: {n_res} "
          f"({tot_c} agree, {tot_d} disagree) -> {tot_c/n_res:.1%} concordant")
    # exact binomial two-sided p
    from math import comb
    p_bin = sum(comb(n_res, i) for i in range(tot_c, n_res + 1)) / 2 ** n_res
    print(f"  one-sided binomial p under 'LB ignores CV': {p_bin:.2e}")
    out["fe"] = dict(pooled_slope=float(pooled_s), within_slope=float(beta), within_se=float(se),
                     n=int(len(keep)), concord=int(tot_c), discord=int(tot_d), p=float(p_bin))

    # ------------------------------------------------- 2. option value of the pair
    print("\n" + "=" * 78)
    print("2. OPTION VALUE -- private score of a selection is the MAX over its entries")
    print("=" * 78)
    cl = big[(big["cv"] >= CLUSTER_MIN_CV)].sort_values("cv", ascending=False)
    names = cl["name"].tolist()
    print(f"\n  CV-top cluster, {len(names)} files, all at public {sorted(set(cl['lb']))}:")
    for _, r in cl.iterrows():
        print(f"    {r['name']:20s} CV {r['cv']:.8f}  kind {r['kind']}")

    pk = {n: prep(np.load(os.path.join(SUB, f"oof_{n}.npy")), y) for n in names}
    n_priv = N_TEST - int(round(N_TEST * a.f))
    print(f"\n  simulating the private slice: {N_TEST}-row pseudo-test drawn from the "
          f"{n_pool}-row labelled pool,\n  f={a.f} public cut away, private = the remaining "
          f"{n_priv} rows, {a.reps} reps")
    A = np.zeros((a.reps, len(names)))
    for i in range(a.reps):
        idx = rng.choice(n_pool, size=N_TEST, replace=False)
        m = np.zeros(n_pool, dtype=bool)
        m[idx[int(round(N_TEST * a.f)):]] = True
        for j, n in enumerate(names):
            A[i, j] = subset_auc(pk[n], m)
    mean = A.mean(axis=0)
    print(f"\n  mean private AUC and pairwise sd of the difference (e-6, vs the top file):")
    base = A[:, 0]
    for j, n in enumerate(names):
        d = A[:, j] - base
        print(f"    {n:20s} mean {mean[j]:.8f}   d vs top {d.mean()*1e6:+7.2f}e-6   "
              f"sd(d) {d.std(ddof=1)*1e6:6.2f}e-6")

    print(f"\n  every pair by E[max] (the quantity Kaggle actually scores), best first:")
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        mx = np.maximum(A[:, i], A[:, j])
        pairs.append((mx.mean(), names[i], names[j], mx.std(ddof=1),
                      (A[:, i] - A[:, j]).std(ddof=1)))
    pairs.sort(reverse=True)
    solo = {n: mean[j] for j, n in enumerate(names)}
    best_solo = max(solo.values())
    for e, n1, n2, sd, sdd in pairs[:10]:
        tag = "  <-- current recommendation" if {n1, n2} == set(PICK) else ""
        print(f"    E[max] {e:.8f}  (+{ (e-best_solo)*1e6:5.2f}e-6 over the best single)  "
              f"sd(d) {sdd*1e6:5.2f}e-6   {n1} + {n2}{tag}")
    cur = [p for p in pairs if {p[1], p[2]} == set(PICK)][0]
    rank_cur = 1 + [i for i, p in enumerate(pairs) if {p[1], p[2]} == set(PICK)][0]
    print(f"\n    current recommendation ranks {rank_cur} of {len(pairs)} pairs; "
          f"E[max] {cur[0]:.8f}")
    print(f"    best pair beats it by {(pairs[0][0]-cur[0])*1e6:+.2f}e-6 "
          f"-- against the {2e-6:.0e} stack reproducibility floor (w14a)")
    out["pairs"] = [dict(e_max=float(e), a=n1, b=n2, sd_diff=float(sdd))
                    for e, n1, n2, sd, sdd in pairs[:12]]
    out["current_pair_rank"] = int(rank_cur)
    out["n_pairs"] = int(len(pairs))
    out["best_solo"] = float(best_solo)

    # ---------------------------------------------------- 3. AUC -> leaderboard places
    print("\n" + "=" * 78)
    print("3. WHAT AN AUC COST IS WORTH IN PLACES (public board, as a density proxy)")
    print("=" * 78)
    lb = pd.read_csv(a.lb)
    ours = lb[lb["TeamMemberUserNames"].astype(str).str.contains("thtennant", na=False)]
    print(f"\n  board: {len(lb)} teams, leader {lb['Score'].max():.5f}, "
          f"ours {float(ours['Score'].iloc[0]):.5f} at rank {int(ours['Rank'].iloc[0])}")
    sc = float(ours["Score"].iloc[0])
    for w in (1e-5, 2e-5, 5e-5):
        n_in = int(((lb["Score"] >= sc - w) & (lb["Score"] <= sc + w)).sum())
        print(f"    teams within +/-{w:.0e} of us: {n_in}  "
              f"=> {n_in/(2*w)*1e-5:.1f} teams per 1e-5 of AUC")
    dens = int(((lb["Score"] >= sc - 2e-5) & (lb["Score"] <= sc + 2e-5)).sum()) / (4e-5)
    for cost, lab in ((9.2e-6, "limit 2, E[cost]"), (3.65e-5, "limit 1, E[cost]"),
                      (1.12e-4, "limit 1, worst case"), (2e-6, "option value of the pair")):
        print(f"    {lab:26s} {cost:.2e} AUC  ~=  {cost*dens:5.1f} places at the "
              f"local density")
    out["density_per_1e5"] = float(dens * 1e-5)
    out["rank"] = int(ours["Rank"].iloc[0])
    out["team_count"] = int(len(lb))

    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
