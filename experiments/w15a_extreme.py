"""Is the 18e-5 gap to the leader a method, or the maximum of a near-tied field?

The brief asserts the gap is real because it is 3.6x the workspace's 5e-5 noise floor.
w15a_crossteam.py shows that floor is the WITHIN-PACK one (rho ~ 0.9999) and that the
measured cross-team floor is 53-84e-6 (rho 0.994-0.997 against the only two other teams
whose predictions we can read).  This script asks the next question: given that floor, is
the observed shape of the top of the leaderboard distinguishable from a field of equally
good teams each taking one draw of public-slice noise?

MODEL (H0: every team in the top plateau has the SAME true full-test AUC)
    public_i = mu + eps_i          eps_i ~ N(0, s)      s = sd(gap)/sqrt(2)
plus, optionally, the best-of-n_i selection every team gets for free because the public LB
shows their best submission and submissions do not evict each other here:
    public_i = mu + eps_i + max over n_i draws of N(0, w)
with w the within-team spread of one team's own candidate files (measured: our own
same-family files sit at sd(gap) 6.0e-6, so w = 6.0/sqrt(2) e-6).

Scores are quantised to 1e-5 on the real board, so the simulation quantises too.

    w15a_extreme.py --lb /tmp/w15a_lb/<file>.csv --reps 20000
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", default=None)
    ap.add_argument("--reps", type=int, default=20_000)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()

    path = a.lb or sorted(glob.glob("/tmp/w15a_lb/*publicleaderboard*.csv"))[-1]
    d = pd.read_csv(path).sort_values("Score", ascending=False).reset_index(drop=True)
    top = d.head(a.topk)
    obs = top["Score"].to_numpy()
    nsub = top["SubmissionCount"].to_numpy()
    our_rank = int(d.index[d["TeamName"] == "Teddy Tennant"][0])          # 0-based
    print(f"{len(d)} teams | top-{a.topk} spread {obs[0]-obs[-1]:.5f} | "
          f"we are rank {our_rank+1} at {d.loc[our_rank,'Score']:.5f} "
          f"with {int(d.loc[our_rank,'SubmissionCount'])} submissions")
    print(f"leader {top.loc[0,'TeamName']} {obs[0]:.5f} with {int(nsub[0])} submissions, "
          f"last {top.loc[0,'LastSubmissionDate']}")

    obs_gap_leader = obs[0] - d.loc[our_rank, "Score"]
    obs_sd = float(obs.std(ddof=1))
    obs_spread = float(obs[0] - obs[-1])
    print(f"\nOBSERVED  leader-minus-us {obs_gap_leader*1e6:.0f}e-6 | "
          f"sd(top{a.topk}) {obs_sd*1e6:.1f}e-6 | range {obs_spread*1e6:.0f}e-6")

    rng = np.random.default_rng(a.seed)
    K = a.topk
    rows = []
    # cross-team sd(gap) from w15a_crossteam.py: 53.0 (naji18), 74.8 (naji12), 83.9 (bolt);
    # within-pack 6.0 / 19.3 / 27.7.  Sweep the plausible range.
    for sd_gap in [20e-6, 30e-6, 40e-6, 53e-6, 65e-6, 84e-6]:
        s = sd_gap / np.sqrt(2.0)
        for w_gap, wlab in [(0.0, "off"), (6.0e-6, "on")]:
            w = w_gap / np.sqrt(2.0)
            sim_gap = np.empty(a.reps)
            sim_sd = np.empty(a.reps)
            sim_range = np.empty(a.reps)
            for r in range(a.reps):
                eps = rng.normal(0.0, s, K)
                if w > 0:
                    eps = eps + np.array([rng.normal(0.0, w, n).max() for n in nsub])
                v = np.round(eps, 5)                       # LB quantisation, 1e-5
                v = np.sort(v)[::-1]
                sim_gap[r] = v[0] - v[min(our_rank, K - 1)]
                sim_sd[r] = v.std(ddof=1)
                sim_range[r] = v[0] - v[-1]
            rows.append(dict(sd_gap=sd_gap, bestof=wlab,
                             mean_gap=float(sim_gap.mean()),
                             p_gap_ge_obs=float((sim_gap >= obs_gap_leader - 5e-7).mean()),
                             mean_sd=float(sim_sd.mean()),
                             mean_range=float(sim_range.mean())))
            print(f"  sd(gap)={sd_gap*1e6:>4.0f}e-6 bestof={wlab:>3} | "
                  f"E[leader-us]={sim_gap.mean()*1e6:>5.0f}e-6 "
                  f"P(>= {obs_gap_leader*1e6:.0f}e-6)={rows[-1]['p_gap_ge_obs']:.3f} | "
                  f"E[sd(top{K})]={sim_sd.mean()*1e6:>5.1f}e-6 (obs {obs_sd*1e6:.1f}) | "
                  f"E[range]={sim_range.mean()*1e6:>5.0f}e-6 (obs {obs_spread*1e6:.0f})")

    json.dump(dict(lb=os.path.basename(path), topk=K, our_rank=our_rank + 1,
                   obs_gap_leader=obs_gap_leader, obs_sd=obs_sd, obs_range=obs_spread,
                   sims=rows), open("experiments/w15a_extreme.json", "w"), indent=1)
    print("\nwrote experiments/w15a_extreme.json")


if __name__ == "__main__":
    main()
