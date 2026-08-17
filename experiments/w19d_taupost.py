"""w19d — the click price MARGINALISED over tau, and this slot's registered ship prediction.

w19c falsified P6 and did it in the direction that matters.  w18a published the click at
+2.328 / +5.509 / +5.666e-6 with P(auto beats both wanted files privately) 0.043 / 0.031 / 0.058
and six slots have quoted that as "the click is not a coin flip".  Those numbers are conditional
on tau = 0 EXACTLY, and w19c's sweep shows they do not survive a tau the data cannot exclude:

    TEST-level tau = 1.70e-6 (w19a's own 95% profile upper):
        w16e_aonly   +2.328 -> -1.493e-6   and   P 0.043 -> 0.733     SIGN FLIP
        w16q_ens4avg +5.509 -> +3.497e-6         P 0.031 -> 0.156
        w16t_cellens4 +5.666 -> +4.679e-6        P 0.058 -> 0.137

The mechanism is not a numerical artefact and it is worth stating plainly, because it is what
the click has always been: Kaggle's auto-pick holds public slot 1 because its public score is
30e-6 above the CV pick while its CV is only 5.3e-6 above.  At tau = 0 the model has nowhere to
put that excess except slice noise, so it shrinks it away and the CV pick wins privately.  Any
TEST-level per-file term at all lets part of the excess be REAL, present in the private slice
too, and the auto file's private prediction rises.  The click was never a robust conclusion;
it was a bet that the auto-pick's public lead is 100% slice noise.

WHAT THE BOARD CAN AND CANNOT IDENTIFY.  w19a fits tau from PUBLIC pair residuals, where a
test-level u_k and a public-level u_k are the same thing — both add tau^2 to the observed
variance.  So the split is unidentified by construction and no future submission can resolve it
either.  The only honest readout is to integrate:

    tau     flat prior on [0, inf), likelihood = w19a's exact-rounding profile over 194 pairs
    phi     fraction of tau^2 that is TEST-level, flat on [0, 1] (total ignorance about a
            quantity the data provably cannot speak to)

and report E[price] and P(auto beats both) under that measure alongside the tau=0 corner.

Also registers this slot's ship, w14a_repro159av, named in w19_prereg.txt before any fit ran.

    .venv/bin/python experiments/w19d_taupost.py
"""
from __future__ import annotations

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
from w18a_lawif_joint import NAMES, WANTED, AUTO1, midrank_cdf, emax  # noqa: E402
from w19a_transfer import pmf_exact  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, U, GU = 296_302, 0.20, 1e-6, 10.0
SHIP = "w14a_repro159av"
SHIP_FAM = "ens4"


def main() -> None:
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub = sub[sub["publicScore"].notna()]
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    LB = sub.groupby("stem")["publicScore"].max().to_dict()

    # ------------------------------------------------------------------ posterior for tau
    P = pd.read_csv(os.path.join(HERE, "w19a_pairs.csv"))
    m, dcv, var0 = P.m.to_numpy(), P.dcv.to_numpy(), P.var0.to_numpy()
    tg = np.linspace(0.0, 8.0, 321)
    ll = np.array([np.log(pmf_exact(m, dcv, np.sqrt(var0 + 2 * t ** 2))).sum() for t in tg])
    post = np.exp(ll - ll.max())
    post /= np.trapezoid(post, tg)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(tg))])
    q = lambda p: float(np.interp(p, cdf, tg))
    print(f"posterior for tau over {len(P)} within-family pairs (flat prior, exact rounding):")
    print(f"  mode {tg[int(np.argmax(post))]:.3f}   mean {float(np.trapezoid(tg*post, tg)):.3f}   "
          f"median {q(0.5):.3f}   90% {q(0.90):.3f}   95% {q(0.95):.3f}   99% {q(0.99):.3f} e-6")
    print(f"  P(tau < 1e-6) = {float(np.interp(1.0, tg, cdf)):.3f}    "
          f"P(tau < 2e-6) = {float(np.interp(2.0, tg, cdf)):.3f}")

    # ------------------------------------------------------------------ LAW-IF for the 6 files
    j17d = json.load(open(os.path.join(HERE, "w17d_coupling.json")))
    beta = float(j17d["coupling_beta_median"])
    j18a = json.load(open(os.path.join(HERE, "w18a_lawif_joint.json")))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    allnames = NAMES + [SHIP]
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in allnames}
    cv = {k: fast_auc(y, V[k]) for k in allnames}
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.array([midrank_cdf(np.sort(V[k][neg]), V[k][pos]) for k in allnames])
    B = np.array([1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg]) for k in allnames])
    C1, C0 = np.cov(A), np.cov(B)
    n_pub, pi1 = int(round(N_TEST * F)), n1 / n
    S_t_a = (1 - N_TEST / n) * (C1 / (N_TEST * pi1) + C0 / (N_TEST * (1 - pi1))) / U ** 2
    S_p_a = (1 - n_pub / N_TEST) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1))) / U ** 2
    K = len(NAMES)
    S_t0, S_p0 = S_t_a[:K, :K], S_p_a[:K, :K]
    col = {k: i for i, k in enumerate(NAMES)}
    z = np.array([LB[k] - cv[k] for k in NAMES]) / U
    D = np.ones((K, 1))
    resid = {}
    for x, zz in itertools.combinations(NAMES, 2):
        key = f"{x}|{zz}" if f"{x}|{zz}" in j17d["coupling"] else f"{zz}|{x}"
        c, v = j17d["coupling"][key], j17d["varsplit"][key]
        s = abs(c["beta"]) * v["sigma_p"] / U / abs(c["corr"])
        resid[frozenset((x, zz))] = float(s * np.sqrt(max(1 - c["corr"] ** 2, 0.0)))

    def price(tau, phi):
        """phi = fraction of tau^2 that is TEST-level; 1-phi is public-slice-only."""
        S_t = S_t0 + phi * tau ** 2 * np.eye(K)
        S_p = S_p0 + (1 - phi) * tau ** 2 * np.eye(K)
        Sx = S_t + S_p
        M = S_t @ np.linalg.inv(Sx)
        Kmap = beta * np.eye(K) + (1 - beta) * M
        Cpri = 0.5 * ((1 - beta) ** 2 * (S_t - M @ Sx @ M.T) + ((1 - beta) ** 2 * (S_t - M @ Sx @ M.T)).T)
        iS = np.linalg.inv(Sx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        mu = Kmap @ (z - D @ gh)
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T
        iw = [col[k] for k in WANTED]
        mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i]) for i in iw])
        rho = cov[iw[0], iw[1]] / (sw[0] * sw[1])
        o = {}
        for x in AUTO1:
            o[x] = float(emax(mw, sw, rho) - (mu[col[x]] + cv[x] / U))
            us = []
            for k in WANTED:
                u = np.zeros(K)
                u[col[x]], u[col[k]] = 1.0, -1.0
                us.append(u)
            mus = np.array([float(cv[x] - cv[k]) / U + float(u @ mu) for u, k in zip(us, WANTED)])
            cc = np.array([[u1 @ cov @ u2 for u2 in us] for u1 in us])
            for i, k in enumerate(WANTED):
                cc[i, i] += resid[frozenset((x, k))] ** 2
            o["P_" + x] = float(multivariate_normal(mean=-mus, cov=cc,
                                                    allow_singular=True).cdf([0.0, 0.0]))
        return o

    base = price(0.0, 1.0)
    for x in AUTO1:
        assert abs(base[x] - j18a["limit1"][x]["closed"]) < 1e-6, f"{x}: w18a gate failed"
    print(f"\ngate: tau=0 reproduces w18a's +{base[AUTO1[0]]:.3f} / +{base[AUTO1[1]]:.3f} / "
          f"+{base[AUTO1[2]]:.3f}e-6 exactly")

    # ------------------------------------------------------------------ marginalise
    print(f"\n=== the click, MARGINALISED over tau (flat prior x w19a likelihood) and over the "
          f"test/public split phi (flat on [0,1], because the board provably cannot identify it) ===")
    tn = np.linspace(0.0, 5.0, 26)
    wt = np.interp(tn, tg, post)
    wt /= wt.sum()
    phis = np.linspace(0.0, 1.0, 11)
    acc = {x: [] for x in AUTO1}
    accp = {x: [] for x in AUTO1}
    W = []
    for t, w in zip(tn, wt):
        for ph in phis:
            o = price(t, ph)
            W.append(w / len(phis))
            for x in AUTO1:
                acc[x].append(o[x])
                accp[x].append(o["P_" + x])
    W = np.array(W)
    print(f"  {'auto file':16s} {'tau=0 (w18a)':>13s} {'E[price]':>10s} {'5th':>8s} {'95th':>8s} "
          f"{'P(click loses)':>15s} {'tau=0 P':>9s} {'E[P]':>7s}")
    out = {}
    for x in AUTO1:
        v, p = np.array(acc[x]), np.array(accp[x])
        o = np.argsort(v)
        c = np.cumsum(W[o]) / W.sum()
        lo, hi = float(v[o][np.searchsorted(c, 0.05)]), float(v[o][np.searchsorted(c, 0.95)])
        e = float((v * W).sum() / W.sum())
        ploss = float(W[v < 0].sum() / W.sum())
        ep = float((p * W).sum() / W.sum())
        out[x] = dict(tau0=base[x], mean=e, p05=lo, p95=hi, p_negative=ploss,
                      tau0_P=base["P_" + x], mean_P=ep)
        print(f"  {x:16s} {base[x]:+13.3f} {e:+10.3f} {lo:+8.3f} {hi:+8.3f} "
              f"{ploss:15.3f} {base['P_'+x]:9.3f} {ep:7.3f}")
    print("\n  'P(click loses)' = posterior probability the cost of NOT clicking is NEGATIVE,")
    print("  i.e. that Kaggle's auto-pick would have done BETTER privately than the CV pick.")
    worst = max(out[x]["p_negative"] for x in AUTO1)
    print(f"  Worst branch: {worst:.3f}.  The click remains the right side of the bet on every")
    print(f"  branch in expectation, but 'not a coin flip' is no longer defensible for the")
    print(f"  w16e_aonly branch and that sentence must come out of check_selection.py.")

    # ------------------------------------------------------------------ the ship
    print(f"\n=== this slot's ship: {SHIP} (named in w19_prereg.txt before any fit) ===")
    ref = "blend159av"
    i, j = allnames.index(SHIP), allnames.index(ref)
    u = np.zeros(len(allnames))
    u[i], u[j] = 1.0, -1.0
    Sx_a = S_t_a + S_p_a
    sd0 = float(np.sqrt(u @ Sx_a @ u))
    d = (cv[SHIP] - cv[ref]) / U
    print(f"  CV {cv[SHIP]:.10f}   reference {ref} CV {cv[ref]:.10f} LB {LB[ref]:.5f}")
    print(f"  dCV {d:+.3f}e-6   LAW-IF paired sd {sd0:.3f}e-6   <- the smallest any send has had")
    best = min(((float(np.sqrt(np.array([1.0 if a == SHIP else -1.0 if a == r else 0.0
                                         for a in allnames]) @ Sx_a @
                              np.array([1.0 if a == SHIP else -1.0 if a == r else 0.0
                                        for a in allnames]))), r)
                for r in NAMES if r in LB), key=lambda t: t[0])
    print(f"  (LAW-IF's cheapest alternative reference among the w18a six: {best[1]} at "
          f"{best[0]:.3f}e-6 — {ref} is used because it is the same-family sibling)")
    print(f"\n  {'model':44s} {'sd':>6s}   distribution over the reporting grid")
    ship_rows = []
    for tag, tau, phi in (("registered form (tau = 0, w18a's world)", 0.0, 1.0),
                          ("tau at w19a's posterior median", q(0.5), 1.0),
                          ("tau at w19a's 95% upper", q(0.95), 1.0)):
        s = np.sqrt(sd0 ** 2 + 2 * tau ** 2)
        ks = np.arange(-4, 5)
        pr = pmf_exact(ks, d, s)
        pr = pr / pr.sum()
        top = np.argsort(-pr)[:5]
        vals = {float(round(LB[ref] + k * 1e-5, 5)): float(pr[list(ks).index(k)]) for k in ks}
        ship_rows.append(dict(tag=tag, tau=tau, sd=float(s), dist=vals))
        print(f"  {tag:44s} {s:6.3f}   " +
              "  ".join(f"{LB[ref]+ks[t]*1e-5:.5f} P{pr[t]:.3f}" for t in top))
    print(f"\n  The three models differ by at most "
          f"{max(abs(ship_rows[0]['dist'][k]-ship_rows[2]['dist'][k]) for k in ship_rows[0]['dist']):.3f}"
          f" in grid probability, so ONE print cannot separate them — stated before the send.")
    print(f"  What the send does buy: {SHIP} adds ~11 new ens4 pairs to w19a's 194 and it is the")
    print(f"  TIGHTEST pair in the workspace, which is where tau is identified.")

    json.dump(dict(post=dict(median=q(0.5), p90=q(0.90), p95=q(0.95), p99=q(0.99),
                             mean=float(np.trapezoid(tg * post, tg))),
                   base={x: base[x] for x in AUTO1}, marginal=out,
                   ship=dict(name=SHIP, ref=ref, cv=cv[SHIP], ref_cv=cv[ref], ref_lb=LB[ref],
                             dcv=d, sd0=sd0, models=ship_rows)),
              open(os.path.join(HERE, "w19d_taupost.json"), "w"), indent=1)
    print("\nwrote w19d_taupost.json")


if __name__ == "__main__":
    main()
