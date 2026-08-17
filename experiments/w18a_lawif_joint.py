"""w18a — w17d's conditioning done in CLOSED FORM, with zero draws and therefore zero ESS.

Slot 3's next-run item 2, verbatim: "Re-cut every published paired sd in the workspace with
LAW-IF ... w17d's ESS-14.3 GLOBAL branch is the prize."

WHAT WAS WRONG. w17d asks: given that we OBSERVED these six public scores, what is the private
contrast between Kaggle's auto-pick and the CV pick? It answers by drawing 6,000 pseudo-tests and
reweighting each draw by the length of the interval of common gaps G under which every one of the
six files rounds to the score we saw. Six simultaneous +/-5e-6 constraints are improbable under
the prior, so the weights collapse: ESS 14.3 of 6,000. w17g then had to build a per-contrast
parametric substitute and its own docstring flagged the loss -- "w17d's non-parametric form
conditions on all six files' public scores jointly, which is strictly more information".

WHAT THIS FILE DOES. The deviations being conditioned on are AUC deviations of fixed score
vectors on random row subsets, and w17h's LAW-IF gives their ENTIRE covariance in closed form
from two K x K matrices of influence values. Under that Gaussian the conditioning is a GLS solve:

    A_k(T) = cv_k + d_k          d ~ N(0, S_t)   S_t = (1 - m/n) [C1/m1 + C0/m0]
    A_k(U) = A_k(T) + e_k        e ~ N(0, S_p)   S_p = (1 - f)   [C1/p1 + C0/p0]
    A_k(V) = A_k(T) + b e_k      b = w17d's measured -0.2477 (exact partition -f/(1-f) = -0.25)
    LB_k   = round_1e-5( cv_k + G + d_k + e_k )
    C1 = cov over pool POSITIVES of F0_k(s_i);  C0 = cov over pool NEGATIVES of 1 - F1_k(s_j)

    x := d + e is what the board shows (up to G and rounding), and
    E[d + b e | x] = b x + (1 - b) M x,   M = S_t (S_t + S_p)^-1
    Cov[d + b e | x] = (1 - b)^2 (S_t - S_t (S_t+S_p)^-1 S_t)

Two branches, and the difference between them is the reporting grid, not the physics:
  CLOSED  drop the grid, condition on equality, G by GLS. One 6x6 solve. No draws at all.
  GRID    keep the +/-5e-6 boxes: a Gibbs sampler on (x, G) over the truncated Gaussian. This is
          the same target w17d importance-sampled, but a chain that MIXES instead of weights
          that collapse -- the ESS is the chain's, not 14.3.
GLOBAL (one free G) and PERFAM (one free G per transform family) are just two design matrices D.

Pre-registration with P1-P7 and what falsifies each: experiments/w18_prereg.txt, written first.

    .venv/bin/python experiments/w18a_lawif_joint.py
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
F = 0.20
GRID = 1e-5
HALF = GRID / 2.0
U = 1e-6                                   # work in units of 1e-6 so the 6x6 solves are conditioned

# w17d's file set and family map, unchanged, so every comparison is like-for-like.
NAMES = ["blend159av_h3", "blend159av", "w16i_schemeavg", "w16q_ens4avg",
         "w16t_cellens4", "w16e_aonly"]
FAM = {"blend159av_h3": "h3", "blend159av": "ens4", "w16i_schemeavg": "h3",
       "w16q_ens4avg": "ens4", "w16t_cellens4": "ens4", "w16e_aonly": "h3"}
WANTED = ("w16i_schemeavg", "blend159av_h3")
AUTO1 = ("w16e_aonly", "w16q_ens4avg", "w16t_cellens4")


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def emax(mu, sd, rho):
    """E[max(X,Y)] for a bivariate normal (Clark). Exact, no simulation."""
    th = np.sqrt(max(sd[0] ** 2 + sd[1] ** 2 - 2 * rho * sd[0] * sd[1], 1e-300))
    a = (mu[0] - mu[1]) / th
    return mu[0] * norm.cdf(a) + mu[1] * norm.cdf(-a) + th * norm.pdf(a)


def gibbs_box(z, S, D, seed, sweeps, burn):
    """E[x] under x ~ N(0,S) truncated to { exists g : |x + D g - z| <= HALF componentwise }.

    Gibbs on (x, g): g_f | x uniform on the surviving interval for family f, x_k | rest a
    truncated normal. Feasible by construction from x = z - D g, so the chain never stalls.
    """
    K = len(z)
    L = np.linalg.inv(S)
    dg = np.diag(L).copy()
    rng = np.random.default_rng(seed)
    g = np.linalg.lstsq(D, z, rcond=None)[0]
    x = z - D @ g
    acc = np.zeros(K)
    acc2 = np.zeros((K, K))
    n = 0
    for it in range(sweeps):
        # g | x, one coordinate per family: uniform on the intersection of the boxes
        for j in range(D.shape[1]):
            c = np.flatnonzero(D[:, j] != 0)
            lo = (z[c] - HALF - x[c]).max()
            hi = (z[c] + HALF - x[c]).min()
            g[j] = rng.uniform(lo, hi) if hi > lo else 0.5 * (lo + hi)
        off = D @ g
        # x_k | x_-k, truncated to its own box
        for k in range(K):
            m = -(L[k] @ x - L[k, k] * x[k]) / dg[k]
            s = 1.0 / np.sqrt(dg[k])
            a, bnd = z[k] - HALF - off[k], z[k] + HALF - off[k]
            ca, cb = norm.cdf((a - m) / s), norm.cdf((bnd - m) / s)
            if cb - ca < 1e-12:                       # numerically degenerate slice: take the midpoint
                x[k] = 0.5 * (a + bnd)
            else:
                x[k] = m + s * norm.ppf(np.clip(ca + (cb - ca) * rng.random(), 1e-12, 1 - 1e-12))
        if it >= burn:
            acc += x
            acc2 += np.outer(x, x)
            n += 1
    mu = acc / n
    return mu, acc2 / n - np.outer(mu, mu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=60_000)
    ap.add_argument("--burn", type=int, default=5_000)
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree — scores are not deterministic"
    LB = agg["max"].to_dict()
    print(f"live: {len(sub)} submissions, {len(agg)} distinct files "
          f"(page size 200 — the w17-slot-2 pagination fix)")

    j17d = json.load(open(os.path.join(HERE, "w17d_coupling.json")))
    for k, v in j17d["lb"].items():
        assert abs(LB[k] - v) < 1e-12, f"{k}: live LB moved since w17d ran"
    beta = float(j17d["coupling_beta_median"])
    print(f"w17d: {j17d['reps']} draws, GLOBAL ESS {j17d['ess']['GLOBAL']:.1f}, "
          f"PERFAM ESS {j17d['ess']['PERFAM']:.1f}, beta {beta:+.4f}")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}
    for k in NAMES:
        assert abs(cv[k] - j17d["cv"][k]) < 1e-12, f"{k}: CV moved since w17d ran"
    print(f"      CV and LB of all {len(NAMES)} files re-checked against w17d: unchanged")

    # ---------------------------------------------------------------- LAW-IF, once
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.empty((len(NAMES), n1))
    B = np.empty((len(NAMES), n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])           # F0_k over positives
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])     # 1 - F1_k over negatives
    C1 = np.cov(A)
    C0 = np.cov(B)
    m = N_TEST
    n_pub = int(round(N_TEST * F))
    pi1 = n1 / n
    m1, m0 = m * pi1, m * (1 - pi1)
    p1, p0 = n_pub * pi1, n_pub * (1 - pi1)
    S_t = (1.0 - m / n) * (C1 / m1 + C0 / m0)
    S_p = (1.0 - n_pub / m) * (C1 / p1 + C0 / p0)
    print(f"\n=== LAW-IF covariances from {n1:,} positives / {n0:,} negatives, zero draws ===")
    print(f"  pool {n:,} -> test {m:,} (fpc {1-m/n:.4f}) -> public {n_pub:,} (fpc {1-n_pub/m:.4f})")
    sd_pool_pub = np.sqrt((1.0 - n_pub / n) * np.diag(C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1))))
    print(f"  own-slice sd of ONE file: {np.sqrt(np.diag(S_p)).mean()*1e6:.1f}e-6 drawing the "
          f"public slice out of the TEST, {sd_pool_pub.mean()*1e6:.1f}e-6 drawing it out of the "
          f"POOL")
    print(f"  -> RESEARCH.md's load-bearing 567e-6 is the pool convention; the two differ only by "
          f"the fpc sqrt({1-n_pub/n:.4f}/{1-n_pub/m:.4f}) = {np.sqrt((1-n_pub/n)/(1-n_pub/m)):.4f}")

    col = {k: i for i, k in enumerate(NAMES)}

    def uvec(x, z):
        u = np.zeros(len(NAMES))
        u[col[x]], u[col[z]] = 1.0, -1.0
        return u

    # ---------------------------------------------------------------- P1 GATE
    print("\n=== P1 GATE: LAW-IF's sigma_t / sigma_p vs w17d's 6,000 simulated draws ===")
    print(f"{'pair':38s} {'sig_t sim':>9s} {'law':>8s} {'ratio':>6s} "
          f"{'sig_p sim':>10s} {'law':>8s} {'ratio':>6s}")
    rt, rp, wlaw, wsim = [], [], [], []
    for x, z in itertools.combinations(NAMES, 2):
        key = f"{x}|{z}" if f"{x}|{z}" in j17d["varsplit"] else f"{z}|{x}"
        v = j17d["varsplit"][key]
        u = uvec(x, z)
        st = float(np.sqrt(u @ S_t @ u))
        sp = float(np.sqrt(u @ S_p @ u))
        rt.append(st / v["sigma_t"])
        rp.append(sp / v["sigma_p"])
        wlaw.append(st ** 2 / (st ** 2 + sp ** 2))
        wsim.append(v["w"])
        print(f"  {x[:17]:17s}-{z[:17]:17s} {v['sigma_t']*1e6:8.3f} {st*1e6:7.3f} "
              f"{st/v['sigma_t']:6.3f} {v['sigma_p']*1e6:9.3f} {sp*1e6:7.3f} {sp/v['sigma_p']:6.3f}")
    rt, rp = np.array(rt), np.array(rp)
    err = np.abs(np.concatenate([rt, rp]) - 1.0)
    p1 = bool(np.median(err) < 0.05 and err.max() < 0.20)
    print(f"  |ratio-1|: median {np.median(err)*100:.2f}%  max {err.max()*100:.2f}%   "
          f"(registered median<5%, max<20%)")
    print(f"  P1 {'PASSED' if p1 else '*** FAILED — everything below is void ***'}")

    # ---------------------------------------------------------------- P2
    print("\n=== P2: the variance split w is a pure row-count constant under LAW-IF ===")
    wlaw, wsim = np.array(wlaw), np.array(wsim)
    at, ap = (1 - m / n) / m, (1 - n_pub / m) / n_pub       # the only pair-dependence-free factors
    closed = at / (at + ap)
    print(f"  LAW-IF w over the 15 pairs: {wlaw.min():.9f} .. {wlaw.max():.9f}  "
          f"(spread {(wlaw.max()-wlaw.min()):.2e})")
    print(f"  closed form from row counts alone: {closed:.9f}      w17d simulated median "
          f"{np.median(wsim):.5f}")
    p2 = bool(wlaw.max() - wlaw.min() < 1e-9 and np.all(np.abs(wsim - 0.125) <= 0.010))
    print(f"  w17d's 15 simulated w: {wsim.min():.4f} .. {wsim.max():.4f}  "
          f"(registered all inside 0.125 +/- 0.010)")
    print(f"  P2 {'CONFIRMED' if p2 else '*** FALSIFIED ***'}")

    # ---------------------------------------------------------------- the conditioning
    z = np.array([LB[k] - cv[k] for k in NAMES]) / U
    St, Sp = S_t / U ** 2, S_p / U ** 2
    Sx = St + Sp
    M = St @ np.linalg.inv(Sx)
    Kmap = beta * np.eye(len(NAMES)) + (1 - beta) * M
    Cpri = (1 - beta) ** 2 * (St - M @ Sx @ M.T)
    Cpri = 0.5 * (Cpri + Cpri.T)

    # ------------------------------------------------- WHY the joint form collapses (see P6)
    # The pseudo-test draw and the public-slice draw are THE SAME sampling operation at two
    # different sizes, so under LAW-IF  S_t = a*C  and  S_p = c*C  with the SAME C, and the
    # regression matrix M = S_t (S_t+S_p)^-1 is a SCALAR times the identity. Conditioning on all
    # six files jointly then cannot differ from conditioning each contrast on itself. Checked,
    # not assumed:
    prop = St / Sp
    print(f"\n=== is S_t proportional to S_p? (the reason the joint form collapses) ===")
    print(f"  elementwise S_t/S_p over the 36 entries: {prop.min():.9f} .. {prop.max():.9f}"
          f"   predicted a/c = {at/ap:.9f}")
    offI = np.abs(M - np.eye(len(NAMES)) * np.diag(M).mean()).max()
    print(f"  max |M - w*I| = {offI:.3e}   -> M is w*I, so E[.|x] is a SCALAR map, "
          f"gamma = {np.diag(Kmap).mean():+.6f}")

    # w17d measured |corr| 0.992-0.997 between the private and public deviations at fixed test:
    # AUC is not additive over a partition. Carry the shortfall as an independent per-pair term.
    print("\n=== the non-additivity residual, from w17d's measured corr ===")
    resid = {}
    for x, zz in itertools.combinations(NAMES, 2):
        key = f"{x}|{zz}" if f"{x}|{zz}" in j17d["coupling"] else f"{zz}|{x}"
        c = j17d["coupling"][key]
        v = j17d["varsplit"][key]
        s_pri = abs(c["beta"]) * v["sigma_p"] / U / abs(c["corr"])
        resid[frozenset((x, zz))] = float(s_pri * np.sqrt(max(1 - c["corr"] ** 2, 0.0)))
    print(f"  residual sd per contrast: {min(resid.values()):.3f} .. {max(resid.values()):.3f}e-6"
          f"  (vs conditional sd below)")

    D_glob = np.ones((len(NAMES), 1))
    D_fam = np.zeros((len(NAMES), 2))
    for i, k in enumerate(NAMES):
        D_fam[i, 0 if FAM[k] == "h3" else 1] = 1.0

    branches = {}
    for tag, D in (("GLOBAL", D_glob), ("PERFAM", D_fam)):
        iS = np.linalg.inv(Sx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        xh = z - D @ gh
        mu = Kmap @ xh
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T          # + uncertainty in the free gap(s)
        branches[tag] = dict(mode="CLOSED", mu=mu, cov=cov, g=gh, x=xh)
        print(f"\n  {tag} CLOSED: free gap(s) {np.round(gh, 2)}e-6, "
              f"x (public deviation implied) {np.round(xh, 2)}e-6")

    print(f"\n=== GRID branch: Gibbs over the +/-5e-6 boxes, {a.sweeps:,} sweeps ===")
    for tag, D in (("GLOBAL", D_glob), ("PERFAM", D_fam)):
        mx, cx = gibbs_box(z, Sx, D, seed=18_2026, sweeps=a.sweeps, burn=a.burn)
        mu = Kmap @ mx
        cov = Cpri + Kmap @ cx @ Kmap.T
        branches[tag + "_GRID"] = dict(mode="GRID", mu=mu, cov=cov, g=None, x=mx)
        d = np.abs(mu - branches[tag]["mu"]).max()
        print(f"  {tag:7s} E[x] {np.round(mx, 2)}   max |mu_GRID - mu_CLOSED| {d:.3f}e-6")

    # ---------------------------------------------------------------- the contrasts
    def contrast(br, x, zz):
        u = uvec(x, zz)
        mu = float(cv[x] - cv[zz]) / U + float(u @ br["mu"])
        sd = float(np.sqrt(max(u @ br["cov"] @ u, 0.0) + resid[frozenset((x, zz))] ** 2))
        return mu, sd

    print("\n=== the private contrasts, conditioned on ALL SIX public scores jointly ===")
    print("  positive = the AUTO file beats the CV pick on the private slice.  units 1e-6")
    print(f"{'auto':15s} {'vs CV pick':16s} {'dCV':>6s} {'dLB':>6s} "
          f"{'CLOSED':>8s} {'sd':>5s} {'P':>6s} | {'GRID':>8s} {'w17g par':>9s} {'w17d G':>8s}")
    j17g = json.load(open(os.path.join(HERE, "w17g_parametric.json")))
    par = {(r["auto"], r["wanted"]): r for r in j17g["contrasts"]}
    rows, dpar = [], []
    for x in AUTO1:
        for zz in WANTED:
            mu, sd = contrast(branches["GLOBAL"], x, zz)
            mug, _ = contrast(branches["GLOBAL_GRID"], x, zz)
            muf, sdf = contrast(branches["PERFAM"], x, zz)
            pr = float(1.0 - norm.cdf(0.0, loc=mu, scale=sd))
            r = par[(x, zz)]
            dpar.append(abs(mu - r["e_cond"] / U))
            rows.append(dict(auto=x, wanted=zz, dcv=(cv[x] - cv[zz]) / U, dlb=(LB[x] - LB[zz]) / U,
                             e_closed=mu, sd_closed=sd, p_closed=pr, e_grid=mug,
                             e_perfam=muf, sd_perfam=sdf, e_w17g=r["e_cond"] / U))
            print(f"  {x:13s} {zz:16s} {(cv[x]-cv[zz])/U:+6.2f} {(LB[x]-LB[zz])/U:+6.1f} "
                  f"{mu:+8.3f} {sd:5.2f} {pr:6.3f} | {mug:+8.3f} {r['e_cond']/U:+9.3f} "
                  f"{'':>8s}")
    R = pd.DataFrame(rows)

    # ---------------------------------------------------------------- the click
    print("\n=== the click, repriced with no ESS anywhere in it ===")
    print("  cost of NOT clicking = E[max over WANTED] - E[auto], on the private slice")
    out = {"p1": p1, "p2": p2, "beta": beta, "limit1": {}, "limit2": {},
           "closed_w": float(closed), "law_w": float(wlaw.mean()),
           "gate_ratio_median": float(np.median(err)), "gate_ratio_max": float(err.max())}
    iw = [col[k] for k in WANTED]
    print(f"  {'auto file':16s} {'w16w none':>10s} {'w17d ESS-14':>12s} {'w17d PERFAM':>12s} "
          f"{'w17g par':>9s} {'w18a CLOSED':>12s} {'w18a GRID':>10s} {'P(auto)':>8s}")
    for x in AUTO1:
        e = {}
        for tag in ("GLOBAL", "GLOBAL_GRID", "PERFAM", "PERFAM_GRID"):
            br = branches[tag]
            mw = np.array([br["mu"][i] + cv[NAMES[i]] / U for i in iw])
            sw = np.array([np.sqrt(br["cov"][i, i]) for i in iw])
            rho = br["cov"][iw[0], iw[1]] / (sw[0] * sw[1])
            ma = br["mu"][col[x]] + cv[x] / U
            e[tag] = float(emax(mw, sw, rho) - ma)
        # P(auto beats BOTH wanted files) — the joint event, which the parametric form cannot ask
        br = branches["GLOBAL"]
        mus = np.array([contrast(br, x, k)[0] for k in WANTED])
        u0, u1 = uvec(x, WANTED[0]), uvec(x, WANTED[1])
        cc = np.array([[u0 @ br["cov"] @ u0, u0 @ br["cov"] @ u1],
                       [u1 @ br["cov"] @ u0, u1 @ br["cov"] @ u1]])
        cc[0, 0] += resid[frozenset((x, WANTED[0]))] ** 2
        cc[1, 1] += resid[frozenset((x, WANTED[1]))] ** 2
        pj = float(multivariate_normal(mean=-mus, cov=cc, allow_singular=True).cdf([0.0, 0.0]))
        d17d, d17g = j17d["limit1"][x], j17g["limit1"][x]
        out["limit1"][x] = dict(none=d17d["NONE"] / U, w17d_glob=d17d["GLOBAL"] / U,
                                w17d_perfam=d17d["PERFAM"] / U, w17g=d17g["parametric"] / U,
                                closed=e["GLOBAL"], grid=e["GLOBAL_GRID"],
                                closed_perfam=e["PERFAM"], grid_perfam=e["PERFAM_GRID"],
                                p_joint=pj)
        print(f"  {x:16s} {d17d['NONE']/U:+10.3f} {d17d['GLOBAL']/U:+12.3f} "
              f"{d17d['PERFAM']/U:+12.3f} {d17g['parametric']/U:+9.3f} {e['GLOBAL']:+12.3f} "
              f"{e['GLOBAL_GRID']:+10.3f} {pj:8.3f}")

    for x, zz in itertools.combinations(AUTO1, 2):
        br = branches["GLOBAL"]
        idx = [col[x], col[zz]]
        ma = np.array([br["mu"][i] + cv[NAMES[i]] / U for i in idx])
        sa = np.array([np.sqrt(br["cov"][i, i]) for i in idx])
        ra = br["cov"][idx[0], idx[1]] / (sa[0] * sa[1])
        mw = np.array([br["mu"][i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(br["cov"][i, i]) for i in iw])
        rw = br["cov"][iw[0], iw[1]] / (sw[0] * sw[1])
        c2 = float(emax(mw, sw, rw) - emax(ma, sa, ra))
        out["limit2"][f"{x}+{zz}"] = dict(closed=c2, w17d_glob=j17d["limit2"][f"{x}+{zz}"]["GLOBAL"] / U
                                          if f"{x}+{zz}" in j17d["limit2"] else None)
        print(f"  limit2 {x:14s} + {zz:14s} [{FAM[x]}+{FAM[zz]}] CLOSED {c2:+8.3f}")

    # ---------------------------------------------------------------- P3-P6
    print("\n=== the registered predictions ===")
    cl = np.array([out["limit1"][x]["closed"] for x in AUTO1])
    g14 = np.array([out["limit1"][x]["w17d_glob"] for x in AUTO1])
    pg = np.array([out["limit1"][x]["w17g"] for x in AUTO1])
    d_par, d_ess = float(np.abs(cl - pg).mean()), float(np.abs(cl - g14).mean())
    p3 = bool(cl.min() >= 1.0 and cl.max() <= 6.0 and d_par < d_ess)
    print(f"  P3 CLOSED limit-1 {np.round(cl,3)}  band [1.0,6.0]; mean|CLOSED-w17g| {d_par:.3f} "
          f"vs mean|CLOSED-ESS14| {d_ess:.3f}  ->  {'CONFIRMED' if p3 else '*** FALSIFIED ***'}")
    ps = np.array([out["limit1"][x]["p_joint"] for x in AUTO1])
    p4 = bool(ps.max() <= 0.25 and ps.min() >= 0.001)
    print(f"  P4 P(auto beats BOTH wanted) {np.round(ps,4)}  band [0.001,0.25]  ->  "
          f"{'CONFIRMED' if p4 else '*** FALSIFIED ***'}")
    dg = float(np.abs(R["e_closed"] - R["e_grid"]).max())
    p5 = bool(dg < 1.0)
    print(f"  P5 max |CLOSED - GRID| over the six contrasts {dg:.4f}e-6  band <1.0  ->  "
          f"{'CONFIRMED' if p5 else '*** FALSIFIED ***'}")
    p6 = bool(max(dpar) > 0.5)
    print(f"  P6 max |CLOSED - w17g parametric| over the six contrasts {max(dpar):.4f}e-6  "
          f"band >0.5  ->  {'CONFIRMED' if p6 else '*** FALSIFIED ***'}")
    print(f"  P7 WANTED = {WANTED} — untouched by construction; nothing here is a CV quantity.")

    out.update(p3=p3, p4=p4, p5=p5, p6=p6, sweeps=a.sweeps,
               contrasts=R.to_dict("records"),
               d_closed_to_w17g=d_par, d_closed_to_ess14=d_ess)
    json.dump(out, open(os.path.join(HERE, "w18a_lawif_joint.json"), "w"), indent=1, sort_keys=True)
    R.to_csv(os.path.join(HERE, "w18a_contrasts.csv"), index=False)
    print("\nwrote experiments/w18a_lawif_joint.json, w18a_contrasts.csv")


if __name__ == "__main__":
    main()
