"""What is a public-leaderboard rank actually worth, empirically, in this series?

Every strategic claim in this workspace about the public LB has been argued from first
principles on *our own* files: paired bootstraps, transform residuals, resolvability
estimates. None of it has ever been checked against the thing it is trying to predict —
what happens when a Playground episode closes and the private board is revealed.

`georgymamarin/playground-series-s6-leaderboards` closes that gap. It carries public rank,
private rank and both scores for every team in the seven completed S6 episodes. Three of
them (E2, E3, E5) are ROC AUC on a synthetic tabular binary target, i.e. the same task
shape as S6E8, so they are the reference class for this competition and not a loose analogy.

Four questions, all of them decision-relevant right now:

1.  **Rank churn at the top.** We sit at public rank 13. Conditional on finishing public
    rank ~13 in an AUC episode, what is the private rank distribution? This is the number
    that says whether chasing public rank is worth anything.
2.  **Is the top-of-board compression real or resolved?** Here ranks 5-14 span 4e-5 and the
    workspace calls that pile-up "genuinely unordered". If that reading is right, past
    episodes should show near-total reshuffling inside equally-tight public clusters.
3.  **The shift's scale vs the gaps it has to overturn.** Public->private score movement,
    measured against the adjacent-team gaps at the top of the same board. A shift that
    dwarfs the gaps means the final ordering is close to a lottery among the leaders.
4.  **Does over-submitting predict getting hurt?** The Rogii failure was tuning to public
    feedback. If that is a general hazard, teams with many submissions should drop more
    at private than teams with few, at matched public rank.

    lbhist.py
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LBH = os.path.join(os.path.dirname(HERE), "data", "lbhist")
AUC_EPISODES = ("S6E2", "S6E3", "S6E5")
TOPN = 30


def load():
    ep = pd.read_csv(os.path.join(LBH, "s6_episodes.csv"))
    lb = pd.read_csv(os.path.join(LBH, "s6_leaderboards.csv"))
    lb = lb[~lb.is_host_baseline].copy()
    lb = lb.merge(ep[["episode", "metric", "higher_is_better", "teams"]], on="episode")
    return ep, lb


def q1_rank_churn(lb):
    """Conditional on public rank r, where does a team finish privately?"""
    print("\n=== 1. rank churn at the top, AUC episodes (E2/E3/E5) ===")
    a = lb[lb.episode.isin(AUC_EPISODES)]
    for lo, hi in ((1, 3), (4, 10), (11, 20), (21, 50), (51, 100)):
        s = a[(a.public_rank >= lo) & (a.public_rank <= hi)]
        if not len(s):
            continue
        pr = s.private_rank.to_numpy()
        print(f"public {lo:3d}-{hi:<3d} n={len(s):3d}  private rank "
              f"median {np.median(pr):6.1f}  p10 {np.percentile(pr, 10):6.1f}  "
              f"p90 {np.percentile(pr, 90):7.1f}  "
              f"P(stay top10) {np.mean(pr <= 10):5.1%}  P(medal-ish top10%) "
              f"{np.mean(pr <= 0.10 * s.teams.to_numpy()):5.1%}")

    print("\n  per-episode: who actually won, and where were they publicly?")
    for e in AUC_EPISODES:
        s = a[a.episode == e]
        w = s[s.private_rank == 1].iloc[0]
        p1 = s[s.public_rank == 1].iloc[0]
        print(f"  {e}: private #1 was public #{int(w.public_rank):<4d} | "
              f"public #1 finished private #{int(p1.private_rank):<4d} | {int(s.teams.iloc[0])} teams")


def q1b_per_episode(lb):
    """Split q1 by episode: a single shakeup episode must not be read as the norm.

    Also reports the top-30 public-vs-private rank correlation, which is the direct
    analogue of our own paired-rank question, and the fraction of the public top 30 that
    is still in the private top 30 at all.
    """
    print("\n=== 1b. the same, per episode — is the churn general or one bad draw? ===")
    print(f"{'ep':6s} {'teams':>6s} {'pub11-20 -> private median':>27s} "
          f"{'top30 rho':>10s} {'top30 kept':>11s} {'pub#1 -> priv':>14s}")
    for e in sorted(lb.episode.unique()):
        s = lb[lb.episode == e]
        band = s[(s.public_rank >= 11) & (s.public_rank <= 20)]
        top = s[s.public_rank <= TOPN]
        rho = top.public_rank.corr(top.private_rank, method="spearman")
        kept = np.mean(top.private_rank.to_numpy() <= TOPN)
        p1 = int(s[s.public_rank == 1].private_rank.iloc[0])
        tag = "AUC" if e in AUC_EPISODES else "   "
        print(f"{e:3s}{tag:3s} {int(s.teams.iloc[0]):6d} "
              f"{np.median(band.private_rank):27.1f} {rho:+10.2f} {kept:11.0%} {p1:14d}")


def q2_cluster_resolution(lb):
    """Inside a tight public cluster, is the private ordering related to the public one?

    Matched to our own situation: take the teams whose public score is within `w` of the
    public leader, and ask how well public order predicts private order among just those.
    """
    print("\n=== 2. does a tight public cluster carry any ordering into private? ===")
    print("  (Spearman of public vs private rank among teams within w of the public lead)")
    for e in AUC_EPISODES:
        s = lb[lb.episode == e].sort_values("public_rank")
        lead = s.public_score.max()
        row = [f"  {e}"]
        for w in (2e-5, 5e-5, 1e-4, 4e-4):
            c = s[s.public_score >= lead - w]
            if len(c) < 4:
                row.append(f"w={w:.0e} n={len(c):<3d}   --   ")
                continue
            rho = c.public_rank.corr(c.private_rank, method="spearman")
            row.append(f"w={w:.0e} n={len(c):<3d} rho {rho:+.2f}")
        print("  ".join(row))


def q3_shift_vs_gaps(lb):
    """How large is the public->private movement compared with the gaps it must overturn?"""
    print("\n=== 3. public->private shift vs adjacent-team gaps in the top 30 ===")
    print(f"{'ep':6s} {'metric':10s} {'mean shift':>11s} {'sd of shift':>12s} "
          f"{'median top30 gap':>17s} {'sd/gap':>8s}")
    for e in sorted(lb.episode.unique()):
        s = lb[lb.episode == e].sort_values("public_rank")
        hib = bool(s.higher_is_better.iloc[0])
        top = s.head(TOPN)
        d = top.private_score.to_numpy() - top.public_score.to_numpy()
        gaps = np.abs(np.diff(top.public_score.to_numpy()))
        gaps = gaps[gaps > 0]
        mg = np.median(gaps) if len(gaps) else np.nan
        # Only the *spread* of the shift can change an ordering; a common offset cannot.
        print(f"{e:6s} {'AUC' if hib else 'lower-better':10s} {d.mean():+11.6f} "
              f"{d.std(ddof=1):12.6f} {mg:17.6f} {d.std(ddof=1) / mg:8.1f}")
    print("\n  sd/gap is the headline: how many adjacent-team gaps the private noise spans.")


def q4_submission_count(lb):
    """Do heavy submitters get punished at private, at matched public rank?"""
    print("\n=== 4. does submitting a lot predict a private drop? (AUC episodes, public top 100) ===")
    a = lb[lb.episode.isin(AUC_EPISODES) & (lb.public_rank <= 100)].copy()
    a = a.dropna(subset=["submissions", "rank_delta"])
    # rank_delta as published = private_rank - public_rank? verify direction, then bucket.
    chk = (a.private_rank - a.public_rank - a.rank_delta).abs().mean()
    sign = "private-public" if chk < 1e-9 else "UNKNOWN — recomputing"
    a["pdrop"] = a.private_rank - a.public_rank
    print(f"  rank_delta convention: {sign} (mean |resid| {chk:.3g}); using recomputed drop")
    qs = a.submissions.quantile([0.25, 0.5, 0.75]).to_numpy()
    a["bucket"] = np.digitize(a.submissions, qs)
    for b in sorted(a.bucket.unique()):
        s = a[a.bucket == b]
        print(f"  submissions bucket {b} (n={len(s):3d}, {s.submissions.min():3.0f}-"
              f"{s.submissions.max():3.0f} subs): mean private drop {s['pdrop'].mean():+6.1f} "
              f"ranks, median {s['pdrop'].median():+5.1f}")
    r = a.submissions.corr(a["pdrop"], method="spearman")
    print(f"  spearman(submissions, private drop) = {r:+.3f}  "
          f"(positive = more submissions -> worse private rank)")


def q5_noise_null(lb, s6e8_board=None):
    """How much of the observed churn is just a noisy re-draw, and how much is overfitting?

    The density story tried in the first pass failed (spearman -0.25 over 7 episodes): it
    anchored on the leader's score, and in S6E2 the leader was 5e-5 clear of the field and
    still finished 570th, which density cannot explain. Anchoring was the wrong idea.

    The matched null instead: private IS a different random slice, so *some* reshuffling is
    guaranteed. Take each episode's public top 30, add i.i.d. Gaussian noise with that
    episode's own observed shift sd, re-rank against the whole board, and read off the
    simulated top-30 retention. The scale is calibrated from the episode's own data, so the
    simulation and reality differ only in SHAPE:

      observed ~= simulated  -> churn is an exchangeable re-draw. No team overfit more than
                                any other; public rank was simply a noisy measurement.
      observed <<  simulated  -> shifts are heterogeneous. Some teams moved far more than the
                                common sd, which is the fingerprint of public-LB overfitting.

    The residual is the quantity this workspace actually needs: how much of a private drop
    is escapable by CV discipline, versus how much is a coin toss nobody can dodge.
    """
    print("\n=== 5. matched null: how much churn is explained by a noisy re-draw alone? ===")
    rng = np.random.default_rng(42)
    print(f"{'ep':7s} {'sd(shift)':>10s} {'obs kept':>9s} {'sim kept':>9s} {'obs-sim':>8s} "
          f"{'med|shift|':>11s} {'p90|shift|/med':>15s}")
    for e in sorted(lb.episode.unique()):
        s = lb[lb.episode == e]
        hib = bool(s.higher_is_better.iloc[0])
        sgn = 1.0 if hib else -1.0
        top = s[s.public_rank <= TOPN]
        d = (top.private_score - top.public_score).to_numpy()
        sd = d.std(ddof=1)
        pub = sgn * s.public_score.to_numpy()
        idx_top = np.argsort(-pub)[:TOPN]
        keeps = []
        for _ in range(400):
            sim = pub + rng.normal(0.0, sd, size=len(pub))
            new_top = set(np.argsort(-sim)[:TOPN].tolist())
            keeps.append(len(new_top & set(idx_top.tolist())) / TOPN)
        sim_kept = float(np.mean(keeps))
        obs_kept = float(np.mean(top.private_rank.to_numpy() <= TOPN))
        ad = np.abs(d - d.mean())
        het = np.percentile(ad, 90) / np.median(ad)
        tag = "AUC" if e in AUC_EPISODES else "   "
        print(f"{e:4s}{tag:3s} {sd:10.6f} {obs_kept:9.0%} {sim_kept:9.0%} "
              f"{obs_kept - sim_kept:+8.0%} {np.median(ad):11.6f} {het:15.2f}")
    print("\n  p90/median of the centred shift is a shape check: 1.9 is Gaussian, higher means")
    print("  a few teams moved far more than the rest, i.e. heterogeneous overfitting.")

    if s6e8_board is None:
        return
    b = s6e8_board.sort_values("Score", ascending=False).reset_index(drop=True)
    pub = b.Score.to_numpy()
    lead = pub[0]
    ours = b.index[b.TeamName.str.contains("Teddy Tennant", case=False, na=False)]
    our_i = int(ours[0]) if len(ours) else None
    print("\n  --- S6E8 live board under the same null ---")
    print(f"  {len(b)} teams, leader {lead:.5f}"
          + (f", us rank {our_i + 1} at {pub[our_i]:.5f} ({lead - pub[our_i]:+.5f})" if our_i is not None else ""))
    for sd, src in ((0.000043, "S6E2"), (0.000067, "S6E3"), (0.000124, "S6E5")):
        ranks = []
        for _ in range(400):
            sim = pub + rng.normal(0.0, sd, size=len(pub))
            ranks.append(int(np.argsort(np.argsort(-sim))[our_i]) + 1)
        r = np.array(ranks)
        print(f"  sd {sd:.6f} ({src}-like): our private rank median {np.median(r):5.1f}, "
              f"p10 {np.percentile(r, 10):5.1f}, p90 {np.percentile(r, 90):6.1f}, "
              f"P(top10) {np.mean(r <= 10):5.1%}, P(top 10%) {np.mean(r <= 0.10 * len(b)):5.1%}")


def q6_local_density(lb, s6e8_board=None):
    """Density measured where we actually stand, which is what the first pass got wrong.

    q5's first operationalisation anchored on the leader and failed. The board check below
    shows why S6E2 detonated: its public #1 scored 0.95419 and finished 570th, but the
    private scores of ranks 1 and 570 differ by 1e-4. Six hundred teams were tied to within
    the private slice's own resolution. That is compression, not overfitting — S6E2's
    whole-board spearman(public, private) is 0.99 and its score correlation is 1.00.

    So measure the density in the neighbourhood that matters: how many teams sit within a
    slice-noise-sized window of the score at public rank 13, our own position.
    """
    print("\n=== 6. local density at public rank 13 — the compression check ===")
    print(f"{'board':11s} {'teams':>6s} {'rank13':>9s} {'+-5e-5':>8s} {'+-1e-4':>8s} {'top30 kept':>11s}")
    for e in AUC_EPISODES:
        s = lb[lb.episode == e]
        ref = float(s[s.public_rank == 13].public_score.iloc[0])
        n5 = int(((s.public_score - ref).abs() <= 5e-5).sum())
        n1 = int(((s.public_score - ref).abs() <= 1e-4).sum())
        kept = float(np.mean(s[s.public_rank <= TOPN].private_rank <= TOPN))
        print(f"{e:11s} {len(s):6d} {ref:9.5f} {n5:8d} {n1:8d} {kept:11.0%}")
    if s6e8_board is None:
        return
    b = s6e8_board.sort_values("Score", ascending=False)
    ref = float(b.Score.iloc[12])
    n5 = int(((b.Score - ref).abs() <= 5e-5).sum())
    n1 = int(((b.Score - ref).abs() <= 1e-4).sum())
    print(f"{'S6E8 live':11s} {len(b):6d} {ref:9.5f} {n5:8d} {n1:8d} {'--':>11s}")
    print("\n  Density is necessary but not sufficient: S6E3 packs 140 teams into +-5e-5 and")
    print("  still kept 53%, because its shift is mostly a COMMON offset and a common offset")
    print("  cannot reorder anyone. q5's simulation prices density and scale jointly and is")
    print("  the number to quote; treat it as +-25% given its obs-sim spread over 7 episodes.")


def main():
    ep, lb = load()
    print(f"{len(ep)} completed S6 episodes, {len(lb)} team rows (host baselines removed)")
    print(f"AUC episodes used as the reference class for S6E8: {', '.join(AUC_EPISODES)}")
    live = None
    g = sorted(glob.glob(os.path.join(os.path.dirname(HERE), "lb_s6e8_now", "*publicleaderboard*.csv")))
    if g:
        live = pd.read_csv(g[-1], encoding="utf-8-sig")
        print(f"live S6E8 board: {os.path.basename(g[-1])}")
    q1_rank_churn(lb)
    q1b_per_episode(lb)
    q2_cluster_resolution(lb)
    q3_shift_vs_gaps(lb)
    q4_submission_count(lb)
    q5_noise_null(lb, live)
    q6_local_density(lb, live)


if __name__ == "__main__":
    main()
