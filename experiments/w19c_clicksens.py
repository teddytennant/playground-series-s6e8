"""w19c — P6: how much does the click price move if the transfer variance tau is NOT zero?

Slot 4's next-run item 2 asked for a per-file transfer variance to be added to w18a's
conditioning and the click re-priced, on the strength of blend159av_rankraw printing z -3.58
against its family.  w19a fitted that variance over 194 within-family pairs with exact rounding
and got tau_hat = 0.000e-6, 95% profile upper 1.7e-6, and w19b's null control showed even that
upper bound is what a PERFECTLY CALIBRATED board produces (null mean tau_hat 1.01, sd 1.45).
So the honest form of item 2 is not "insert tau_hat" — it is a SENSITIVITY: sweep tau across
everything the data still permit and show the click price does not care.

The split the board cannot resolve, and why both ends are priced.  The public scores identify
the TOTAL variance of (observed LB - cv).  They cannot say whether an extra per-file term lives

  TEST-level    u_k is a property of the file on the whole test set (e.g. the train/test mask
                shift hitting this file differently).  It is in BOTH slices:  S_t -> S_t + tau^2 I.
                Note this BREAKS w18a's collapse -- with tau > 0, S_t is no longer proportional
                to S_p, M = S_t (S_t+S_p)^-1 stops being a scalar times I, and conditioning on
                all six files jointly stops being equivalent to conditioning each on itself.
  PUBLIC-level  u_k is specific to the slice we happened to see.  S_p -> S_p + tau^2 I.  The
                public score then carries less information about the private one, so the
                conditioning weakens and the price should slide back toward w16w's
                unconditioned +1.253..+4.346e-6.

Everything else is w18a unchanged: same six files, same LAW-IF covariance, same beta, same
non-additivity residual, same E[max] and joint-probability readouts.

    .venv/bin/python experiments/w19c_clicksens.py
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
from w18a_lawif_joint import NAMES, FAM, WANTED, AUTO1, midrank_cdf, emax  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
F = 0.20
U = 1e-6


def main() -> None:
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub = sub[sub["publicScore"].notna()]
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    LB = sub.groupby("stem")["publicScore"].max().to_dict()

    j17d = json.load(open(os.path.join(HERE, "w17d_coupling.json")))
    beta = float(j17d["coupling_beta_median"])
    j18a = json.load(open(os.path.join(HERE, "w18a_lawif_joint.json")))
    j19a = json.load(open(os.path.join(HERE, "w19a_transfer.json")))
    tau_hat, tau_up = j19a["tau"], j19a["prof_hi"]
    print(f"w19a: tau_hat {tau_hat:.3f}e-6, 95% profile upper {tau_up:.3f}e-6 "
          f"over {j19a['n_pairs']} within-family pairs")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}
    for k in NAMES:
        assert abs(cv[k] - j17d["cv"][k]) < 1e-12, f"{k}: CV moved"
        assert abs(LB[k] - j17d["lb"][k]) < 1e-12, f"{k}: LB moved"
    print(f"gate: CV and LB of all {len(NAMES)} files unchanged since w17d/w18a")

    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.array([midrank_cdf(np.sort(V[k][neg]), V[k][pos]) for k in NAMES])
    B = np.array([1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg]) for k in NAMES])
    C1, C0 = np.cov(A), np.cov(B)
    m_t, n_pub, pi1 = N_TEST, int(round(N_TEST * F)), n1 / n
    S_t0 = (1 - m_t / n) * (C1 / (m_t * pi1) + C0 / (m_t * (1 - pi1))) / U ** 2
    S_p0 = (1 - n_pub / m_t) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1))) / U ** 2
    K = len(NAMES)
    col = {k: i for i, k in enumerate(NAMES)}
    z = np.array([LB[k] - cv[k] for k in NAMES]) / U
    D = np.ones((K, 1))

    resid = {}
    for x, zz in itertools.combinations(NAMES, 2):
        key = f"{x}|{zz}" if f"{x}|{zz}" in j17d["coupling"] else f"{zz}|{x}"
        c, v = j17d["coupling"][key], j17d["varsplit"][key]
        s_pri = abs(c["beta"]) * v["sigma_p"] / U / abs(c["corr"])
        resid[frozenset((x, zz))] = float(s_pri * np.sqrt(max(1 - c["corr"] ** 2, 0.0)))

    def price(S_t, S_p):
        """w18a's GLOBAL CLOSED branch, verbatim, for any (S_t, S_p)."""
        Sx = S_t + S_p
        M = S_t @ np.linalg.inv(Sx)
        Kmap = beta * np.eye(K) + (1 - beta) * M
        Cpri = (1 - beta) ** 2 * (S_t - M @ Sx @ M.T)
        Cpri = 0.5 * (Cpri + Cpri.T)
        iS = np.linalg.inv(Sx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        mu = Kmap @ (z - D @ gh)
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T
        scal = float(np.abs(M - np.eye(K) * np.diag(M).mean()).max())
        iw = [col[k] for k in WANTED]
        mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i]) for i in iw])
        rho = cov[iw[0], iw[1]] / (sw[0] * sw[1])
        out = {"collapse": scal, "g": float(gh[0])}
        for x in AUTO1:
            ma = mu[col[x]] + cv[x] / U
            out[x] = float(emax(mw, sw, rho) - ma)
            us = []
            for k in WANTED:
                u = np.zeros(K)
                u[col[x]], u[col[k]] = 1.0, -1.0
                us.append(u)
            mus = np.array([float(cv[x] - cv[k]) / U + float(u @ mu)
                            for u, k in zip(us, WANTED)])
            cc = np.array([[u1 @ cov @ u2 for u2 in us] for u1 in us])
            for i, k in enumerate(WANTED):
                cc[i, i] += resid[frozenset((x, k))] ** 2
            out["P_" + x] = float(multivariate_normal(mean=-mus, cov=cc,
                                                      allow_singular=True).cdf([0.0, 0.0]))
        return out

    base = price(S_t0, S_p0)
    print(f"\ngate: tau=0 reproduces w18a's published limit-1 prices")
    for x in AUTO1:
        d = abs(base[x] - j18a["limit1"][x]["closed"])
        print(f"  {x:16s} w19c {base[x]:+8.4f}   w18a {j18a['limit1'][x]['closed']:+8.4f}   "
              f"|diff| {d:.2e}  {'OK' if d < 1e-6 else '*** MISMATCH ***'}")

    print(f"\n=== sensitivity sweep: tau from 0 to 4e-6, well past w19a's 95% upper "
          f"{tau_up:.2f}e-6 ===")
    taus = [0.0, 0.5, 1.0, tau_up, 2.0, 3.0, 4.0]
    rows = []
    for branch in ("TEST", "PUBLIC"):
        print(f"\n  {branch}-level tau   "
              + ("(u in both slices; breaks w18a's S_t ~ S_p collapse)" if branch == "TEST"
                 else "(u only in the slice we saw; conditioning weakens)"))
        print(f"  {'tau':>5s} {'collapse':>9s} " +
              " ".join(f"{x[:13]:>13s}" for x in AUTO1) + " | " +
              " ".join(f"{'P '+x[:8]:>11s}" for x in AUTO1))
        for t in taus:
            S_t = S_t0 + (t ** 2) * np.eye(K) if branch == "TEST" else S_t0
            S_p = S_p0 + (t ** 2) * np.eye(K) if branch == "PUBLIC" else S_p0
            r = price(S_t, S_p)
            rows.append(dict(branch=branch, tau=t, collapse=r["collapse"],
                             **{x: r[x] for x in AUTO1},
                             **{"P_" + x: r["P_" + x] for x in AUTO1}))
            print(f"  {t:5.2f} {r['collapse']:9.2e} " +
                  " ".join(f"{r[x]:+13.3f}" for x in AUTO1) + " | " +
                  " ".join(f"{r['P_'+x]:11.3f}" for x in AUTO1))

    R = pd.DataFrame(rows)
    mv = {}
    for branch in ("TEST", "PUBLIC"):
        s = R[(R.branch == branch) & (R.tau <= tau_up)]
        mv[branch] = {x: float(np.abs(s[x] - base[x]).max()) for x in AUTO1}
    worst = max(max(v.values()) for v in mv.values())
    pmax = float(R[R.tau <= tau_up][["P_" + x for x in AUTO1]].to_numpy().max())
    p6 = bool(worst < 2.0 and pmax < 0.15)
    print(f"\n=== P6 (registered in w19_prereg.txt) ===")
    print(f"  max |price move| over tau in [0, {tau_up:.2f}]e-6, both branches: "
          f"{worst:.4f}e-6   (registered < 2.0)")
    print(f"  max P(auto beats BOTH wanted) over the same range: {pmax:.4f}   "
          f"(registered < 0.15)")
    print(f"  P6 {'CONFIRMED' if p6 else '*** FALSIFIED ***'}")
    w4 = {b: {x: float(np.abs(R[(R.branch == b) & (R.tau == 4.0)][x].iloc[0] - base[x]))
              for x in AUTO1} for b in ("TEST", "PUBLIC")}
    print(f"  even at tau = 4.0e-6, far outside anything w19a permits, the prices move by at "
          f"most {max(max(v.values()) for v in w4.values()):.3f}e-6")

    json.dump(dict(tau_hat=tau_hat, tau_upper=tau_up, base={x: base[x] for x in AUTO1},
                   sweep=R.to_dict("records"), move_within_upper=mv, move_at_4=w4,
                   P6=p6, worst_move=worst, p_max=pmax),
              open(os.path.join(HERE, "w19c_clicksens.json"), "w"), indent=1)
    R.to_csv(os.path.join(HERE, "w19c_sweep.csv"), index=False)
    print("\nwrote w19c_clicksens.json, w19c_sweep.csv")


if __name__ == "__main__":
    main()
