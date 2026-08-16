"""How much of the leaderboard's top-30 spread is true skill, and what does that imply
for the private split?

Inputs, both measured elsewhere and NOT fitted here:
  * sd(gap) between two independently built strong solutions on a 59,260-row public slice
    = 53-84e-6  (w15a_crossteam.py, against najiama's and boltuzamaki's published vectors)
  * public and private partition ONE 296,302-row test set, f = 0.20, so
        private_gap = (g - f * public_gap) / (1 - f)
    where g is the true full-test gap.  The coupling coefficient was measured at
    beta = -0.2517 against the exact-partition -0.2500 (w14b, corr -0.9988).

Model of the top-K board: public_i = t_i + eps_i, t_i ~ N(0, tau), eps_i ~ N(0, s),
s = sd(gap)/sqrt(2).  tau is the only free parameter and it is fitted to ONE statistic:
the observed sd of the top-K public scores.  Then, for the leader's observed public gap p,
the posterior mean of their true full-test edge is the usual shrinkage g_hat =
p * tau^2/(tau^2 + 2 s^2)  (2 s^2 because a GAP carries two teams' noise), and the
predicted private gap follows from the partition identity.

    w15a_private.py
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

F = 0.20
REPS = 6000
TOPK = 30


def sim_sd(tau, s, K, nrep, rng, npool=155):
    """sd of the TOP-K of a plateau of `npool` teams.

    The real top 30 is the top 30 of a plateau, not a sample of 30, and upper order
    statistics are compressed relative to a full sample.  Ignoring that would understate
    the noise level needed to reproduce the observed spread, i.e. it would bias the
    answer AGAINST this script's conclusion, so it is modelled explicitly.
    """
    out = np.empty(nrep)
    for r in range(nrep):
        v = np.round(rng.normal(0.0, np.hypot(tau, s), npool), 5)
        out[r] = np.sort(v)[::-1][:K].std(ddof=1)
    return out.mean()


def main():
    path = sorted(glob.glob("/tmp/w15a_lb/*publicleaderboard*.csv"))[-1]
    d = pd.read_csv(path).sort_values("Score", ascending=False).reset_index(drop=True)
    top = d.head(TOPK)
    obs_sd = float(top["Score"].std(ddof=1))
    ours = float(d.loc[d["TeamName"] == "Teddy Tennant", "Score"].iloc[0])
    rng = np.random.default_rng(15)

    print(f"observed sd of the top-{TOPK} public scores: {obs_sd*1e6:.1f}e-6   "
          f"(us {ours:.5f})\n")
    TEAMS = ["MILANFX", "Maher el Ouahabi", "Don Mani", "Optimistix", "Utkarsh",
             "cstdy", "Szymon Kłapiński"]
    print(f"{'plateau':>8} {'sd(gap)':>8} {'tau fitted':>11} {'shrink':>7} "
          f"{'leader p':>9} {'g_hat':>8} {'private gap':>12}")
    print("-" * 72)
    rows = []
    for npool in [30, 155, 267]:
        for sd_gap in [53e-6, 65e-6, 84e-6]:
            s = sd_gap / np.sqrt(2.0)
            # fit tau so the simulated sd(top-K) matches the observed one
            taus = np.linspace(0.0, 160e-6, 41)
            vals = np.array([sim_sd(t, s, TOPK, 400, rng, npool) for t in taus])
            tau = float(np.interp(obs_sd, vals, taus)) if vals[0] < obs_sd < vals[-1] else (
                0.0 if obs_sd <= vals[0] else float(taus[-1]))
            shrink = tau ** 2 / (tau ** 2 + 2 * s ** 2)
            for name in TEAMS:
                sc = float(d.loc[d["TeamName"] == name, "Score"].iloc[0])
                p = sc - ours
                g = p * shrink
                priv = (g - F * p) / (1 - F)
                rows.append(dict(npool=npool, sd_gap=sd_gap, tau=tau, shrink=shrink,
                                 team=name, p=p, sigma=p / sd_gap, g_hat=g,
                                 private_gap=priv))
                if name == "MILANFX":
                    print(f"{npool:>8} {sd_gap*1e6:>7.0f}e-6 {tau*1e6:>10.1f}e-6 "
                          f"{shrink:>7.3f} {p*1e6:>8.0f}e-6 {g*1e6:>7.0f}e-6 "
                          f"{priv*1e6:>11.0f}e-6")
    print()
    for npool in [30, 155]:
        print(f"--- plateau {npool} teams, sd(gap) 65e-6")
        print(f"{'team':22} {'public gap':>11} {'sigma':>7} {'g_hat':>9} "
              f"{'pred private gap':>17}")
        for r in [x for x in rows if x["npool"] == npool and abs(x["sd_gap"] - 65e-6) < 1e-9]:
            print(f"{r['team']:22} {r['p']*1e6:>10.0f}e-6 {r['sigma']:>7.2f} "
                  f"{r['g_hat']*1e6:>8.0f}e-6 {r['private_gap']*1e6:>16.0f}e-6")
        print()

    json.dump(rows, open("experiments/w15a_private.json", "w"), indent=1)
    print("\npositive private gap = that team beats us privately under this model.")
    print("wrote experiments/w15a_private.json")


if __name__ == "__main__":
    main()
