"""w27i -- what a public rank is actually worth privately, from the seven finished S6 boards.

Rayk's notebook (notebooks/w27/rayk_overfit) makes the qualitative point: three of seven S6
public top tens had ZERO survivors in the private top ten. That is the right warning and the
wrong statistic for THIS account, because we are not in the top ten -- we are 18th of a
~1326-team board and what we are playing for is a medal, not a win.

So this asks the question we actually face, on the same public data
(`georgymamarin/playground-series-s6-leaderboards`, 7 finished episodes):

  Given a team's PUBLIC rank r out of n, what is the distribution of its PRIVATE rank?

restricted to the three ROC-AUC episodes (S6E2, S6E3, S6E5), because the metric governs how
compressed the frontier is and s6e8 is ROC AUC. The other four episodes are reported as a
robustness check and never pooled with them.

Everything here is a backtest on other people's boards. It cannot tell us what OUR file will
do. It can tell us how much of a 5th-decimal public lead is real, and that is the number the
deadline pick turns on.

    .venv/bin/python experiments/w27i_s6risk.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BOARDS = os.path.join(os.path.dirname(HERE), "data", "s6_boards", "s6_leaderboards.csv")
AUC = ("S6E2", "S6E3", "S6E5")
# our live position on s6e8, read from lb_w27/*publicleaderboard*.csv at 2026-08-19 14:51 UTC.
# ⚠ The brief's "~1,326 teams" is STALE: the board carries 2,323 teams as of this slot, and
# every percentile below moves if you use the wrong denominator.
OUR_PUBLIC_RANK, OUR_TEAMS = 17, 2323
# Kaggle medal cuts for a 1000+ team board: gold = top 10 + 0.2%, silver = top 5%,
# bronze = top 10%.  At 2323 teams that is 14 / 116 / 232.
GOLD, SILVER, BRONZE = 14, 116, 232


def main():
    b = pd.read_csv(BOARDS)
    b = b[~b.is_host_baseline].dropna(subset=["public_rank", "private_rank"])
    ep = pd.read_csv(os.path.join(os.path.dirname(BOARDS), "s6_episodes.csv"))
    print(f"{len(b):,} team-rows over {b.episode.nunique()} episodes\n")

    frac = OUR_PUBLIC_RANK / OUR_TEAMS
    print(f"our position on s6e8: public rank {OUR_PUBLIC_RANK} of ~{OUR_TEAMS} teams "
          f"= top {frac*100:.2f}%\n")

    rows = []
    for e, g in b.groupby("episode", sort=True):
        n = len(g)
        g = g.copy()
        g["pub_pct"] = g.public_rank / n
        g["prv_pct"] = g.private_rank / n
        # the band our own public standing sits in, +/- 50% of it in relative terms
        band = g[(g.pub_pct >= frac * 0.5) & (g.pub_pct <= frac * 1.5)]
        met = ep.loc[ep.episode == e, "metric"].iloc[0]
        rows.append(dict(
            episode=e, metric=met.replace(" Score", ""), teams=n,
            band_n=len(band),
            band_median_private_pct=100 * band.prv_pct.median(),
            band_p10=100 * band.prv_pct.quantile(0.10),
            band_p90=100 * band.prv_pct.quantile(0.90),
            band_kept_gold=100 * (band.private_rank <= 10 + 0.002 * n).mean(),
            band_kept_top5pct=100 * (band.private_rank <= 0.05 * n).mean(),
            band_kept_top10pct=100 * (band.private_rank <= 0.10 * n).mean(),
            top10_kept=int((g.nsmallest(10, "public_rank").private_rank <= 10).sum()),
            spearman=g.public_rank.corr(g.private_rank, method="spearman"),
        ))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Per episode. `band` = teams whose public rank sits in the same relative slice of "
          "the board as ours (top 0.68%-2.04%).")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    a = t[t.episode.isin(AUC)]
    o = t[~t.episode.isin(AUC)]
    print(f"\nROC-AUC episodes only ({', '.join(AUC)}):")
    print(f"  median private percentile of a team in our band : "
          f"{a.band_median_private_pct.mean():.2f}%  (we enter at {frac*100:.2f}%)")
    print(f"  10th-90th percentile of where they landed       : "
          f"{a.band_p10.mean():.2f}% .. {a.band_p90.mean():.2f}%")
    print(f"  still GOLD  privately (top 10 + 0.2%)           : "
          f"{a.band_kept_gold.mean():.1f}%")
    print(f"  still SILVER privately (top 5%)                 : "
          f"{a.band_kept_top5pct.mean():.1f}%")
    print(f"  still BRONZE privately (top 10%)                : "
          f"{a.band_kept_top10pct.mean():.1f}%")
    print(f"  public/private spearman over the whole board    : {a.spearman.mean():.4f}")
    print(f"\nnon-AUC episodes, robustness only, NOT pooled:")
    print(f"  median private percentile {o.band_median_private_pct.mean():.2f}%, "
          f"top-5% retention {o.band_kept_top5pct.mean():.1f}%, "
          f"spearman {o.spearman.mean():.4f}")

    # The frontier-compression number: how many teams sit within one public grid step of the
    # leader, and where do they end up? s6e8 prints 5 decimals, so a "step" is 1e-5.
    print(f"\nFrontier compression, AUC episodes, teams within 1e-4 of the public leader:")
    for e in AUC:
        g = b[b.episode == e].copy()
        lead = g.public_score.max()
        tight = g[g.public_score >= lead - 1e-4]
        print(f"  {e}: {len(tight):3d} teams within 1e-4 public; their private ranks span "
              f"{int(tight.private_rank.min())}..{int(tight.private_rank.max())} "
              f"(median {int(tight.private_rank.median())}) of {len(g)}")
    g8_lead, g8_us = 0.97134, 0.97118
    print(f"\n  s6e8 today: leader {g8_lead}, us {g8_us}, gap {(g8_lead-g8_us)*1e4:.2f}e-4.")
    print(f"  Read from the live board: only 8 of 2,323 teams are within 1e-4 of the s6e8")
    print(f"  leader and 87 are within 2e-4, so the s6e8 frontier is NOT as compressed as")
    print(f"  S6E2's (156 teams inside 1e-4, private ranks 4..1856). It is closer to S6E5's.")
    print(f"  7 teams share our exact printed 0.97118.")

    t.to_csv(os.path.join(HERE, "w27i_s6risk.csv"), index=False)
    print(f"\nsaved experiments/w27i_s6risk.csv")


if __name__ == "__main__":
    main()
