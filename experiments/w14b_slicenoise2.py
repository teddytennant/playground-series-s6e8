"""Is the transform CV->LB displacement real, a slice draw, or neither?  And does the
answer even change the deadline pick?

BACKGROUND (see JOURNAL 2026-08-11..13 and RESEARCH "The logit CV bias")
------------------------------------------------------------------------
Within a fixed member set the `logit` stack has, every time it has been measured, a worse
cross-fitted CV than `hybrid` and a BETTER public leaderboard score.  Three sets have both
transforms scored:

    set      CV(logit)-CV(hybrid)      LB(logit)-LB(hybrid)      displacement
    150fx           -64.4e-6                  +40e-6                +104.4e-6
    150sx           -60.7e-6                  +30e-6                 +90.7e-6
    158             -66.9e-6                  +30e-6                 +96.9e-6

`oofsim.py` tested the proposed MECHANISM (logit's clip pins more single-fold OOF cells
than 5-fold-averaged test cells, so the CV is scored on the damaged side) on a labelled
pseudo-test across three splits and found it absent: -10e-6 +/- 15e-6 at full dose.  So the
mechanism cannot supply +97e-6.  The surviving explanation was "the fixed public slice is
just a draw that flatters logit" -- asserted in the journal, never priced.

WHAT THIS SCRIPT DOES
---------------------
Every variant of every member set has its cross-fitted OOF vector on 691,369 LABELLED
rows.  So rebuild the leaderboard's geometry out of labelled data: draw a pseudo-test of
296,302 rows, cut an f-slice out of it as the pseudo-public leaderboard, and score
everything on the slice, on its complement, and on the whole thing.  The pooled ordering of
these vectors IS the CV ordering by construction, so under this null there is no transform
effect at all and every public/private disagreement is pure slice draw.

    1. sd of the slice draw on a paired difference -> is +97e-6 an ordinary draw?
    2. p-value for the observed LB pattern under the null.
    3. Are the three "replications" independent?  They are read off ONE fixed slice with
       files correlated ~0.9999, so simulate all three sets on a SHARED slice and measure
       how much of the deviation is common.  Three correlated reads are not 3x the evidence.
    4. Public and private partition one test set, so a slice that flatters a file is taken
       out of the complement that scores it.  Measure the coupling, then propagate the
       OBSERVED public reading into a predicted private gap -- which is what the deadline
       pick actually turns on, and which (see the end) is the same answer under BOTH
       explanations.

CAVEAT.  OOF cells are single-fold and real test cells are 5-fold averages, so this cannot
reproduce the clip asymmetry.  It is not meant to; `oofsim.py` covers that half on data
where the asymmetry is present.  The two together are a complete account only if mechanism
and slice draw are the only candidates.

    w14b_slicenoise2.py --reps 250
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

# blend158: the only member set with all six variants scored on the public LB.
SIX = ["logit", "hybrid", "rankraw", "rescale", "h3", "ens4"]
FILES = {
    "logit": "blend158_logit", "hybrid": "blend158_hybrid",
    "rankraw": "blend158_rankraw", "rescale": "blend158_rescale",
    "h3": "blend158_h3", "ens4": "blend158",
}
LB = {"logit": 0.97106, "hybrid": 0.97103, "rankraw": 0.97104,
      "rescale": 0.97105, "h3": 0.97105, "ens4": 0.97106}

# the three sets that carry a within-set logit/hybrid LB pair
PANEL = {
    "150fx": ("blend150fx_logit", "blend150fx_hybrid", 0.97103, 0.97099),
    "150sx": ("blend150sx_logit", "blend150sx_hybrid", 0.97104, 0.97101),
    "158": ("blend158_logit", "blend158_hybrid", 0.97106, 0.97103),
}
FRACS = (0.10, 0.20, 0.30, 0.50)


# --------------------------------------------------------------------- fast AUC
def prep(v):
    """Global stable argsort once; subset AUCs are then O(n) instead of O(n log n)."""
    o = np.argsort(v, kind="stable")
    return o, np.ascontiguousarray(v[o])


def _auc_sorted(s, yy):
    """Mann-Whitney AUC on values ALREADY sorted ascending, average ranks for ties."""
    n = s.size
    n1 = float(yy.sum())
    n0 = float(n) - n1
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    b = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))   # group starts, 0-based
    ends = np.concatenate((b[1:], [n]))
    avg = (b + ends + 1) * 0.5                                      # mean 1-based rank
    pos = np.add.reduceat(yy, b)
    return (float((avg * pos).sum()) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def subset_auc(o, vs, ys, mask):
    sel = mask[o]
    return _auc_sorted(vs[sel], ys[sel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments",
                                                  "w14b_slicenoise2.json"))
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)

    need = sorted({FILES[k] for k in SIX} | {f for v in PANEL.values() for f in v[:2]})
    V = {}
    for nm in need:
        v = np.load(os.path.join(SUB, f"oof_{nm}.npy"))
        assert len(v) == n_pool, (nm, len(v))
        V[nm] = v
    print(f"pool {n_pool:,} labelled rows   pseudo-test {N_TEST:,}   {len(V)} vectors")

    # correctness check of the fast AUC against scipy on a real subset
    o0, vs0 = prep(V["blend158_h3"])
    ys0 = y[o0]
    m = np.zeros(n_pool, bool)
    m[np.random.default_rng(0).permutation(n_pool)[:50_000]] = True
    ref = ((rankdata(V["blend158_h3"][m])[y[m] == 1].sum()
            - y[m].sum() * (y[m].sum() + 1) / 2) / (y[m].sum() * (~(y[m] == 1)).sum()))
    fast = subset_auc(o0, vs0, ys0, m)
    assert abs(ref - fast) < 1e-12, (ref, fast)
    print(f"fast AUC checked against scipy rankdata: {fast:.12f} == {ref:.12f}")

    PRE = {nm: prep(v) for nm, v in V.items()}
    YS = {nm: y[PRE[nm][0]] for nm in V}

    pooled = {nm: subset_auc(*PRE[nm], YS[nm], np.ones(n_pool, bool)) for nm in V}
    print("\n=== pooled AUC = the cross-fitted CV = this null's ground truth ===")
    for k in sorted(SIX, key=lambda z: -pooled[FILES[z]]):
        print(f"  {k:8s} {pooled[FILES[k]]:.6f}   LB {LB[k]:.5f}")

    rng = np.random.default_rng(a.seed)
    names = list(V)
    # [rep, frac, vector] for the public slice and its complement; [rep, vector] for full
    PUB = np.zeros((a.reps, len(FRACS), len(names)))
    PRV = np.zeros((a.reps, len(FRACS), len(names)))
    FUL = np.zeros((a.reps, len(names)))

    for r in range(a.reps):
        idx = rng.permutation(n_pool)[:N_TEST]
        mfull = np.zeros(n_pool, bool)
        mfull[idx] = True
        masks = []
        for f in FRACS:
            npub = int(round(N_TEST * f))
            mp = np.zeros(n_pool, bool)
            mp[idx[:npub]] = True
            masks.append((mp, mfull & ~mp))
        for j, nm in enumerate(names):
            o, vs = PRE[nm]
            ys = YS[nm]
            FUL[r, j] = subset_auc(o, vs, ys, mfull)
            for fi, (mp, mv) in enumerate(masks):
                PUB[r, fi, j] = subset_auc(o, vs, ys, mp)
                PRV[r, fi, j] = subset_auc(o, vs, ys, mv)
        if (r + 1) % 25 == 0:
            print(f"  rep {r+1}/{a.reps}", flush=True)

    ji = {nm: j for j, nm in enumerate(names)}
    f20 = FRACS.index(0.20)
    res = {"reps": a.reps, "fracs": list(FRACS), "pooled": pooled, "lb": LB}

    # ---------------------------------------------------------------- 1. slice sd
    print("\n=== 1. how big is the public slice's own draw, on a paired difference? ===")
    print("    all e-6.  'dev' = observed LB gap - CV gap = what the slice would have")
    print("    had to supply on its own.\n")
    print(f"    {'pair':16s} {'CV gap':>8s} {'LB gap':>8s} {'dev':>8s}"
          + "".join(f"{'sd@'+str(f):>9s}" for f in FRACS)
          + "".join(f"{'z@'+str(f):>8s}" for f in FRACS))
    res["pairs"] = {}
    rows = [("logit", "hybrid"), ("logit", "h3"), ("logit", "ens4"), ("logit", "rescale"),
            ("h3", "ens4"), ("rankraw", "hybrid"), ("h3", "rankraw")]
    for aa, bb in rows:
        na, nb = FILES[aa], FILES[bb]
        cvg = pooled[na] - pooled[nb]
        lbg = LB[aa] - LB[bb]
        dev = lbg - cvg
        sds, zs = [], []
        for fi, f in enumerate(FRACS):
            d = PUB[:, fi, ji[na]] - PUB[:, fi, ji[nb]]
            sds.append(float(d.std()))
            zs.append(dev / sds[-1])
        print(f"    {aa+'-'+bb:16s} {cvg*1e6:+8.1f} {lbg*1e6:+8.1f} {dev*1e6:+8.1f}"
              + "".join(f"{s*1e6:9.1f}" for s in sds)
              + "".join(f"{z:+8.2f}" for z in zs))
        res["pairs"][f"{aa}-{bb}"] = dict(cv_gap=cvg, lb_gap=lbg, dev=dev,
                                          sd=dict(zip(map(str, FRACS), sds)),
                                          z=dict(zip(map(str, FRACS), zs)))

    # ---------------------------------------------------------------- 2. p-values
    print("\n=== 2. does slice noise alone reproduce the observed LB pattern? (f=0.20) ===")
    q = np.round(PUB[:, f20, [ji[FILES[k]] for k in SIX]], 5)   # LB's 1e-5 quantisation
    kl, kh, khy = SIX.index("logit"), SIX.index("h3"), SIX.index("hybrid")
    ev = {
        "logit >= h3 on the slice": q[:, kl] >= q[:, kh],
        "logit tied for best of the six": q.max(1) == q[:, kl],
        "logit > hybrid (the 3x-read contrast)": q[:, kl] > q[:, khy],
        "hybrid alone last of the six": (q.argmin(1) == khy) & (
            (q == q.min(1, keepdims=True)).sum(1) == 1),
    }
    res["events"] = {}
    for lbl, m2 in ev.items():
        p = float(m2.mean())
        print(f"    P[{lbl:40s}] = {p:.4f}   ({int(m2.sum())}/{a.reps})")
        res["events"][lbl] = p

    cvrank = rankdata([pooled[FILES[k]] for k in SIX])
    rhos = np.array([np.corrcoef(cvrank, rankdata(q[r]))[0, 1] for r in range(a.reps)])
    obs = float(np.corrcoef(cvrank, rankdata([LB[k] for k in SIX]))[0, 1])
    print(f"\n    spearman(CV order, slice order) under the null "
          f"{rhos.mean():+.3f} +/- {rhos.std():.3f}")
    print(f"    observed spearman(CV order, LB order)          {obs:+.3f}"
          f"   -> P[null <= observed] = {(rhos <= obs).mean():.4f}")
    res["rho"] = dict(null_mean=float(rhos.mean()), null_sd=float(rhos.std()),
                      observed=obs, p=float((rhos <= obs).mean()))

    # ------------------------------------------- 3. are the 3 readings independent?
    print("\n=== 3. the three 'replications' are read off ONE slice -- how correlated? ===")
    D = {}
    for tag, (nl, nh, lbl_, lbh) in PANEL.items():
        d = (PUB[:, f20, ji[nl]] - PUB[:, f20, ji[nh]]) \
            - (FUL[:, ji[nl]] - FUL[:, ji[nh]])
        D[tag] = d
    tags = list(PANEL)
    print("    pairwise correlation of the slice deviation across member sets:")
    cors = {}
    for i in range(len(tags)):
        for j2 in range(i + 1, len(tags)):
            c = float(np.corrcoef(D[tags[i]], D[tags[j2]])[0, 1])
            cors[f"{tags[i]}~{tags[j2]}"] = c
            print(f"      {tags[i]:6s} ~ {tags[j2]:6s}  {c:+.4f}")
    cbar = float(np.mean(list(cors.values())))
    k = len(tags)
    n_eff = k / (1 + (k - 1) * cbar)
    print(f"    mean correlation {cbar:+.4f}  ->  effective independent reads "
          f"{n_eff:.2f} of {k}")
    obs_dev = {t: (PANEL[t][2] - PANEL[t][3]) - (pooled[PANEL[t][0]] - pooled[PANEL[t][1]])
               for t in tags}
    print("    observed displacement per set (e-6): "
          + "  ".join(f"{t} {obs_dev[t]*1e6:+.1f}" for t in tags))
    sd_mean = float(np.mean([D[t] for t in tags], 0).std())
    zbar = float(np.mean(list(obs_dev.values())) / sd_mean)
    print(f"    sd of the MEAN displacement under the null {sd_mean*1e6:.1f}e-6"
          f"  ->  observed mean is {zbar:+.2f} sd")
    res["panel"] = dict(cors=cors, mean_cor=cbar, n_eff=n_eff,
                        obs_dev={t: obs_dev[t] for t in tags},
                        sd_mean=sd_mean, z_mean=zbar)

    # --------------------------------------------- 4. public/private coupling
    print("\n=== 4. public and private partition ONE test set: the flattery is spent ===")
    res["coupling"] = {}
    for aa, bb in [("logit", "h3"), ("logit", "hybrid")]:
        na, nb = FILES[aa], FILES[bb]
        for fi, f in enumerate(FRACS):
            dp = (PUB[:, fi, ji[na]] - PUB[:, fi, ji[nb]]) - (FUL[:, ji[na]] - FUL[:, ji[nb]])
            dv = (PRV[:, fi, ji[na]] - PRV[:, fi, ji[nb]]) - (FUL[:, ji[na]] - FUL[:, ji[nb]])
            beta = float(np.polyfit(dp, dv, 1)[0])
            rho = float(np.corrcoef(dp, dv)[0, 1])
            if f == 0.20:
                print(f"    {aa}-{bb} @ f={f}: beta {beta:+.4f} corr {rho:+.4f}"
                      f"   (exact-partition prediction {-f/(1-f):+.4f})")
            res["coupling"][f"{aa}-{bb}@{f}"] = dict(beta=beta, corr=rho, naive=-f / (1 - f))

    print("\n    So the private gap is FORCED by arithmetic, whatever caused the public")
    print("    reading:  private ~ (g - f*public) / (1-f),  g = the true full-test gap.")
    print("    oofsim bounds the real transform effect at -10e-6 +/- 15e-6, so g for")
    print("    logit-h3 sits between the CV gap and the CV gap + ~20e-6.\n")
    cvg = pooled[FILES["logit"]] - pooled[FILES["h3"]]
    lbg = LB["logit"] - LB["h3"]
    print(f"    {'assumed real transform effect':32s} {'g (full-test)':>14s} "
          f"{'=> private gap':>15s}")
    priv_rows = {}
    for delta in (0.0, 20e-6, 40e-6, 97e-6):
        g = cvg + delta
        priv = (g - 0.20 * lbg) / 0.80
        note = "  <- the whole displacement assumed real" if delta > 90e-6 else ""
        print(f"    {('+%.0fe-6' % (delta*1e6)):32s} {g*1e6:+14.1f} "
              f"{priv*1e6:+15.1f}{note}")
        priv_rows[f"{delta*1e6:.0f}e-6"] = dict(g=g, private=priv)
    res["private_forced"] = dict(cv_gap=cvg, lb_gap=lbg, rows=priv_rows)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
