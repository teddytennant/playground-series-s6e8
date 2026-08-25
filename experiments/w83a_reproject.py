"""Where does public rank 139 of 2,874 actually finish? The 08-11 projection, refreshed.

RESEARCH carries a private-side projection for this account under the heading "the
private-side projection for our own position (rank 13, 0.97106, 1,415 teams)". Every one of
those three numbers is now wrong: the field has doubled to 2,874 and we sit at rank 139. The
table said P(top 10%) = 100.0% and P(top 10) = 24-35%, and a run that quotes it today is
quoting a fortnight-old board. This file recomputes it where we actually stand.

Two instruments, deliberately of different kinds, because the first one needs an assumption
the second does not:

  A. the matched null (lbhist.py q5, unchanged in method): add i.i.d. Gaussian noise at a
     past AUC episode's own observed public->private shift sd to every team's public score,
     re-rank, read our rank off the redraw. Parametric, and it assumes every team moves with
     the same sd -- which is exactly what makes it a null rather than a forecast.

  B. the empirical band, which assumes nothing: on each finished S6 board, take the teams who
     stood where we stand as a SHARE of the field, and read where they finished as a share of
     the field. A move measured as a share is dimensionless, so boards of different sizes
     pool. Reported per-episode as well as pooled, because georgymamarin's section 12 shows
     this column is bimodal -- boards that erased their top ten and boards that held are two
     populations, and a pooled quantile describes neither.

Neither instrument changes a decision. The final selection is on CV and the CV ordering is
settled (SELECT_THESE.md); nothing here touches it. What it changes is the number this
workspace is allowed to quote about its own finish.

Controls, all of which must pass or the file exits 1:

  C1  the published top-30 retention series (77/3/53/20/57/0/0 percent) recomputes from the
      dataset. Deterministic, no RNG -- catches a swapped or re-uploaded source file.
  C2  the matched null at sd = 0 returns our current rank exactly, on every rep.
  C3  the live board is fresh (<24h) and at least as large as the last one recorded (2,791).
      A truncated or stale download fails here instead of being reported as a rank.
  C4  our rank on the board equals 1 + the number of teams strictly above us, and our score
      is the account best.
  C5  instrument B run public->public returns our own band back.

    w83a_reproject.py
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LBH = os.path.join(ROOT, "data", "lbhist")
AUC_EPISODES = ("S6E2", "S6E3", "S6E5")
TEAM = "Teddy Tennant"
ACCOUNT_BEST = 0.97119
MIN_TEAMS = 2791          # w82's board read; the field only grows
PUBLISHED_KEPT30 = (0.77, 0.03, 0.53, 0.20, 0.57, 0.00, 0.00)   # RESEARCH, measured 08-11
REPS = 4000
FAILURES: list[str] = []


def ok(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def newest_board() -> tuple[pd.DataFrame, str, dt.datetime]:
    """Newest public board on disk, chosen by the timestamp IN THE FILENAME.

    Kaggle stamps the download, so the filename is the authority and mtime is not: a file
    copied or re-extracted gets a fresh mtime while still holding a fortnight-old board.
    That is the exact failure this whole file exists to correct, so it is not repeated here.
    """
    pat = re.compile(r"publicleaderboard-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    best = None
    for p in glob.glob(os.path.join(ROOT, "lb_*", "**", "*publicleaderboard*.csv"), recursive=True):
        m = pat.search(os.path.basename(p))
        if not m:
            continue
        stamp = dt.datetime.fromisoformat(m.group(1))
        if best is None or stamp > best[1]:
            best = (p, stamp)
    if best is None:
        raise SystemExit("no public leaderboard csv found under lb_*/")
    path, stamp = best
    b = pd.read_csv(path, encoding="utf-8-sig").sort_values("Score", ascending=False)
    return b.reset_index(drop=True), path, stamp


def load_history() -> pd.DataFrame:
    ep = pd.read_csv(os.path.join(LBH, "s6_episodes.csv"))
    lb = pd.read_csv(os.path.join(LBH, "s6_leaderboards.csv"))
    lb = lb[~lb.is_host_baseline].copy()
    return lb.merge(ep[["episode", "metric", "higher_is_better", "teams"]], on="episode")


def matched_null(pub: np.ndarray, our_i: int, sd: float, rng) -> np.ndarray:
    out = np.empty(REPS, dtype=int)
    for r in range(REPS):
        sim = pub + rng.normal(0.0, sd, size=pub.size)
        out[r] = int(np.argsort(np.argsort(-sim))[our_i]) + 1
    return out


def band_finish(lb: pd.DataFrame, lo: float, hi: float, use_private: bool = True) -> dict:
    """Where teams standing in public share-band [lo, hi] finished, per episode."""
    res = {}
    for e in sorted(lb.episode.unique()):
        s = lb[lb.episode == e]
        n = float(s.teams.iloc[0])
        share = s.public_rank.to_numpy() / n
        sel = (share >= lo) & (share <= hi)
        if sel.sum() < 5:
            continue
        col = s.private_rank if use_private else s.public_rank
        res[e] = col.to_numpy()[sel] / n
    return res


def main() -> int:
    lb = load_history()
    board, path, stamp = newest_board()
    now = dt.datetime.utcnow()
    n_teams = len(board)
    rows = board.index[board.TeamName.str.strip().str.lower() == TEAM.lower()]
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one row for {TEAM!r}, found {len(rows)}")
    our_i = int(rows[0])
    our_rank = our_i + 1
    our_score = float(board.Score.iloc[our_i])
    our_share = our_rank / n_teams

    print(f"board   {os.path.basename(path)}")
    print(f"stamped {stamp} UTC, {(now - stamp).total_seconds() / 3600:.1f}h old, "
          f"{n_teams} teams, leader {board.Score.iloc[0]:.5f}")
    print(f"us      public rank {our_rank} at {our_score:.5f} "
          f"= top {our_share:.2%} of the field, {board.Score.iloc[0] - our_score:+.5f} vs leader")

    print("\n=== A. matched null at OUR position (parametric; assumes a common shift sd) ===")
    pub = board.Score.to_numpy(dtype=float)
    print(f"{'shift sd':>12s} {'source':8s} {'median':>7s} {'p10':>6s} {'p90':>7s} "
          f"{'P(top 0.5%)':>12s} {'P(top 5%)':>10s} {'P(top 10%)':>11s}")
    for sd, src in ((0.000043, "S6E2"), (0.000067, "S6E3"), (0.000124, "S6E5")):
        r = matched_null(pub, our_i, sd, np.random.default_rng(42))
        print(f"{sd:12.6f} {src:8s} {np.median(r):7.0f} {np.percentile(r, 10):6.0f} "
              f"{np.percentile(r, 90):7.0f} {np.mean(r <= 0.005 * n_teams):12.1%} "
              f"{np.mean(r <= 0.05 * n_teams):10.1%} {np.mean(r <= 0.10 * n_teams):11.1%}")
    print("  Read the sd column, not the medians: the three past AUC boards disagree about it")
    print("  by 3x, and that spread is wider than anything this account can influence.")

    lo, hi = max(0.0, our_share - 0.02), our_share + 0.02
    print(f"\n=== B. empirical band: teams who stood in the top {lo:.2%}-{hi:.2%} of their own "
          f"board ===")
    per_ep = band_finish(lb, lo, hi)
    print(f"{'episode':9s} {'metric':4s} {'n':>4s} {'median finish':>14s} {'p10':>7s} "
          f"{'p90':>7s} {'P(<=5%)':>9s} {'P(<=10%)':>9s}")
    pooled_auc: list[np.ndarray] = []
    for e, v in per_ep.items():
        tag = "AUC" if e in AUC_EPISODES else "   "
        if e in AUC_EPISODES:
            pooled_auc.append(v)
        print(f"{e:9s} {tag:4s} {len(v):4d} {np.median(v):13.2%} {np.percentile(v, 10):6.2%} "
              f"{np.percentile(v, 90):6.2%} {np.mean(v <= 0.05):9.1%} {np.mean(v <= 0.10):9.1%}")
    p = np.concatenate(pooled_auc)
    print(f"{'AUC pool':9s} {'AUC':4s} {len(p):4d} {np.median(p):13.2%} "
          f"{np.percentile(p, 10):6.2%} {np.percentile(p, 90):6.2%} "
          f"{np.mean(p <= 0.05):9.1%} {np.mean(p <= 0.10):9.1%}")
    print("  The pooled row is printed last and should be trusted least: the per-episode rows")
    print("  are the two populations, and pooling them describes neither.")

    print("\n=== controls ===")
    kept = []
    for e in sorted(lb.episode.unique()):
        s = lb[lb.episode == e]
        kept.append(float(np.mean(s[s.public_rank <= 30].private_rank <= 30)))
    ok("C1 published top-30 retention recomputes",
       all(abs(a - b) < 0.005 for a, b in zip(kept, PUBLISHED_KEPT30)),
       " ".join(f"{k:.0%}" for k in kept))
    z = matched_null(pub, our_i, 0.0, np.random.default_rng(7))
    ok("C2 zero-noise null returns our rank", bool(np.all(z == our_rank)),
       f"all {REPS} reps = {int(z[0])}")
    ok("C3 board fresh and not truncated",
       (now - stamp).total_seconds() < 24 * 3600 and n_teams >= MIN_TEAMS,
       f"{(now - stamp).total_seconds() / 3600:.1f}h, {n_teams} teams >= {MIN_TEAMS}")
    above = int((board.Score.to_numpy() > our_score).sum())
    ok("C4 rank self-consistent and score is the account best",
       above + 1 == our_rank and abs(our_score - ACCOUNT_BEST) < 5e-6,
       f"{above} strictly above, score {our_score:.5f}")
    ident = band_finish(lb, lo, hi, use_private=False)
    ok("C5 band instrument is the identity on public->public",
       all(v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9 for v in ident.values()),
       f"{len(ident)} episodes")

    print(f"\nFAILURES {len(FAILURES)}" + (": " + ", ".join(FAILURES) if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
