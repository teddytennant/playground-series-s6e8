"""Price the public slice's own draw noise, and ask whether it alone explains the
transform-dependent CV->LB displacement.

THE OPEN QUESTION
-----------------
On the blend158 member set every one of the six stack variants was scored on the public
leaderboard.  The two orderings disagree completely:

    CV     h3 > ens4 > rankraw > hybrid > rescale > logit      (spread 87e-6)
    LB     logit = ens4 > h3 = rescale > rankraw > hybrid      (spread 30e-6, 3 ulp)

`logit` has the worst CV of the six by 66e-6 and is tied for the best public score.  The
journal has read that as a transform property nine times across member sets.  `oofsim.py`
tested the proposed mechanism on a labelled pseudo-test and found it absent (flat across
dose, sign-inconsistent, three splits).  That kills the MECHANISM.  It does not by itself
establish the remaining explanation, which is that the fixed public slice is simply a draw
that happens to flatter `logit`.  Nobody has ever priced that draw.

THE MEASUREMENT
---------------
Every one of the six variants has its cross-fitted OOF vector on 691,369 labelled train
rows.  So build the leaderboard's own geometry out of labelled data:

    draw a pseudo-test of 296,302 rows (the real test size) from the OOF pool
    split it f / (1-f) into a pseudo-public slice and its complement
    score all six variants on each

This is the null "there is no transform effect beyond what the OOF already says" made
concrete: the pooled ordering of these vectors IS the CV ordering, by construction, so any
public/private disagreement the simulation produces is pure slice draw.  Three questions
fall out, and all three are decision-relevant:

1.  How big is sd(public AUC difference) between two of our variants at slice size?  If it
    is comparable to 1e-4 then the +97e-6 `logit - hybrid` displacement is one ordinary
    draw, and the nine "replications" are one draw re-read (the files correlate 0.9999).

2.  How often does the null reproduce the observed LB pattern -- `logit` at or above `h3`
    on the slice, despite being 87e-6 below it in the pool?  That is a p-value for the
    transform hypothesis with the mechanism already excluded.

3.  The public slice and the private slice are drawn from ONE test set, so they are
    negatively coupled: a slice that flatters a file is taken out of the complement that
    scores it.  Regress the private deviation on the public deviation and the observed
    public displacement converts into a PREDICTED PRIVATE PENALTY for `blend158_logit` --
    which is the file Kaggle's default selection will otherwise hand the private score to.

CAVEAT, stated up front.  OOF cells are single-fold predictions and real test cells are
5-fold averages, so this simulation cannot reproduce the clip asymmetry that the logit
mechanism was built on.  It is not meant to: `oofsim.py` covers that half on data with the
asymmetry present, and found nothing.  This covers the other half.  Together they are a
complete account only if those are the only two candidate explanations.

    w14b_slicenoise.py --reps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
N_TEST = 296_302

# The blend158 set: the only member set on which all six variants were scored.
VARIANTS = ["logit", "hybrid", "rankraw", "rescale", "h3", "ens4"]
OOF_FILE = {
    "logit": "oof_blend158_logit.npy",
    "hybrid": "oof_blend158_hybrid.npy",
    "rankraw": "oof_blend158_rankraw.npy",
    "rescale": "oof_blend158_rescale.npy",
    "h3": "oof_blend158_h3.npy",
    "ens4": "oof_blend158.npy",
}
LB = {  # public leaderboard, 2026-08-13 batch, quantised at 1e-5
    "logit": 0.97106, "hybrid": 0.97103, "rankraw": 0.97104,
    "rescale": 0.97105, "h3": 0.97105, "ens4": 0.97106,
}


def auc(y, s):
    """Mann-Whitney AUC with average ranks, so ties are handled exactly."""
    r = rankdata(s)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return (r[y == 1].sum() - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--frac", type=float, default=0.20,
                    help="public slice fraction of the test set")
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments",
                                                  "w14b_slicenoise.json"))
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)
    P = {k: np.load(os.path.join(SUB, v)) for k, v in OOF_FILE.items()}
    for k, v in P.items():
        assert len(v) == n_pool, (k, len(v), n_pool)

    print(f"pool {n_pool:,} labelled rows   pseudo-test {N_TEST:,}   "
          f"public frac {a.frac:.2f} -> {int(N_TEST * a.frac):,} rows")

    print("\n=== pooled AUC (this IS the cross-fitted CV, and it is the null's truth) ===")
    pooled = {k: auc(y, P[k]) for k in VARIANTS}
    for k in sorted(VARIANTS, key=lambda z: -pooled[z]):
        print(f"  {k:8s} {pooled[k]:.6f}   LB {LB[k]:.5f}")

    rng = np.random.default_rng(a.seed)
    n_pub = int(round(N_TEST * a.frac))
    pub = np.zeros((a.reps, len(VARIANTS)))
    prv = np.zeros((a.reps, len(VARIANTS)))
    ful = np.zeros((a.reps, len(VARIANTS)))

    for r in range(a.reps):
        idx = rng.permutation(n_pool)[:N_TEST]
        ip, iv = idx[:n_pub], idx[n_pub:]
        for j, k in enumerate(VARIANTS):
            v = P[k]
            pub[r, j] = auc(y[ip], v[ip])
            prv[r, j] = auc(y[iv], v[iv])
            ful[r, j] = auc(y[idx], v[idx])
        if (r + 1) % 50 == 0:
            print(f"  rep {r+1}/{a.reps}", flush=True)

    res = {"frac": a.frac, "reps": a.reps, "n_pub": n_pub,
           "pooled": {k: pooled[k] for k in VARIANTS}, "lb": LB}

    # ---- 1. how big is the slice draw on a PAIRED difference? ----------------
    print("\n=== 1. paired difference between variants: slice draw vs the CV gap ===")
    print("    d = AUC(a) - AUC(b).  'full' is the whole 296,302-row pseudo-test,")
    print("    'public' is the f-slice of it.  All e-6.\n")
    hdr = f"    {'pair':18s} {'CV gap':>9s} {'sd(full)':>9s} {'sd(public)':>11s} {'observed LB gap':>16s}"
    print(hdr)
    pairs = [("logit", "hybrid"), ("logit", "h3"), ("logit", "ens4"),
             ("h3", "ens4"), ("rankraw", "hybrid"), ("h3", "rankraw")]
    res["pairs"] = {}
    for aa, bb in pairs:
        ja, jb = VARIANTS.index(aa), VARIANTS.index(bb)
        dpub = pub[:, ja] - pub[:, jb]
        dful = ful[:, ja] - ful[:, jb]
        cvgap = pooled[aa] - pooled[bb]
        lbgap = LB[aa] - LB[bb]
        print(f"    {aa+'-'+bb:18s} {cvgap*1e6:+9.1f} {dful.std()*1e6:9.1f} "
              f"{dpub.std()*1e6:11.1f} {lbgap*1e6:+16.1f}")
        res["pairs"][f"{aa}-{bb}"] = dict(
            cv_gap=cvgap, lb_gap=lbgap, sd_full=float(dful.std()),
            sd_pub=float(dpub.std()),
            z=float((lbgap - cvgap) / dpub.std()) if dpub.std() > 0 else float("nan"))

    # ---- 2. p-value for the observed LB pattern under the null ---------------
    print("\n=== 2. does slice noise alone reproduce the observed LB pattern? ===")
    jl, jh, jhy = (VARIANTS.index(k) for k in ("logit", "h3", "hybrid"))
    q = np.round(pub, 5)                      # the LB's own 1e-5 quantisation
    ev = {
        "logit >= h3 on the slice": (q[:, jl] >= q[:, jh]),
        "logit strictly > h3": (q[:, jl] > q[:, jh]),
        "logit tied for best of the six": (q.max(1) == q[:, jl]),
        "logit > hybrid (the 9x-read contrast)": (q[:, jl] > q[:, jhy]),
        "hybrid strictly last of the six": (q.min(1) == q[:, jhy]) & (
            (q < q[:, jhy][:, None]).sum(1) == 0),
    }
    res["events"] = {}
    for lbl, m in ev.items():
        p = float(m.mean())
        print(f"    P[{lbl:40s}] = {p:.3f}")
        res["events"][lbl] = p

    # rank correlation between the pooled (CV) ordering and the simulated slice ordering
    cvrank = rankdata([pooled[k] for k in VARIANTS])
    rhos = np.array([np.corrcoef(cvrank, rankdata(q[r]))[0, 1] for r in range(a.reps)])
    obs_rho = np.corrcoef(cvrank, rankdata([LB[k] for k in VARIANTS]))[0, 1]
    print(f"\n    spearman(CV order, slice order) under the null: "
          f"{rhos.mean():+.3f} +/- {rhos.std():.3f}")
    print(f"    observed spearman(CV order, LB order)            : {obs_rho:+.3f}"
          f"   percentile {100*(rhos < obs_rho).mean():.0f}")
    res["rho_null_mean"] = float(rhos.mean())
    res["rho_null_sd"] = float(rhos.std())
    res["rho_observed"] = float(obs_rho)
    res["rho_percentile"] = float((rhos < obs_rho).mean())

    # ---- 3. public and private are negatively coupled ------------------------
    print("\n=== 3. the public slice is TAKEN OUT OF the private one ===")
    print("    regress (private deviation) on (public deviation), both vs the same")
    print("    pseudo-test's full value, on the paired difference.\n")
    res["coupling"] = {}
    for aa, bb in [("logit", "h3"), ("logit", "hybrid")]:
        ja, jb = VARIANTS.index(aa), VARIANTS.index(bb)
        dp = (pub[:, ja] - pub[:, jb]) - (ful[:, ja] - ful[:, jb])
        dv = (prv[:, ja] - prv[:, jb]) - (ful[:, ja] - ful[:, jb])
        beta = float(np.polyfit(dp, dv, 1)[0])
        rho = float(np.corrcoef(dp, dv)[0, 1])
        print(f"    {aa}-{bb}: beta {beta:+.3f}  corr {rho:+.3f}   "
              f"(exact-average prediction {-a.frac/(1-a.frac):+.3f})")
        res["coupling"][f"{aa}-{bb}"] = dict(beta=beta, corr=rho,
                                             naive=-a.frac / (1 - a.frac))

    # the consequence: what the observed public reading implies for the private slice
    ja, jb = VARIANTS.index("logit"), VARIANTS.index("h3")
    beta = res["coupling"]["logit-h3"]["beta"]
    cvgap = pooled["logit"] - pooled["h3"]
    lbgap = LB["logit"] - LB["h3"]
    dev = lbgap - cvgap                       # how much the slice flattered logit
    pred = cvgap + beta * dev
    print(f"\n    logit - h3:  CV gap {cvgap*1e6:+.1f}e-6, public gap {lbgap*1e6:+.1f}e-6")
    print(f"    the slice flattered logit by {dev*1e6:+.1f}e-6 "
          f"({dev/res['pairs']['logit-h3']['sd_pub']:+.2f} sd of the slice draw)")
    print(f"    => predicted PRIVATE gap  {pred*1e6:+.1f}e-6  "
          f"(worse than the CV gap, not better)")
    res["private_prediction"] = dict(cv_gap=cvgap, lb_gap=lbgap, slice_dev=dev,
                                     beta=beta, predicted_private_gap=pred)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
