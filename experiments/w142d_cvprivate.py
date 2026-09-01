"""w142d — the only question left: did CV predict PRIVATE better than the public LB did?

This workspace was handed a warning ("the Rogii failure": entries tuned against public LB
feedback that moved sharply the wrong way in private) and answered it with a rule -- select on
CV, never on public. The rule was never tested, because the test needs private scores and they
did not exist until today. They exist now, on all 201 sends.

So this compares the two selection rules that were both computable BEFORE the close:

    argmax-CV      pick the sent file with the highest published CV
    argmax-public  pick the sent file with the highest public LB score

against each other and against the oracle. Nothing here is hindsight-tuned: both rules are
fixed, and the only new information is the private column.

CVs come from the submission DESCRIPTIONS, which are immutable once sent -- the same source
w84a_pickargmax has always used, same regex.

⚠ RESOLUTION. The API publishes scores at 5 d.p. Differences below 1e-5 are invisible here,
which is exactly why P1 came back UNGRADED. Ties are handled as ties, not broken arbitrarily.
"""
import json, re, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "experiments")
import kaggle_board

CV_RE = re.compile(r"CV (0\.\d{6,})")
TEAM_ID = 16701216

rows = json.load(open("experiments/w142b_allsubs.json"))
df = pd.DataFrame(rows)
df["stem"] = df["fileName"].str.replace(r"\.csv$", "", regex=True)
df["pub"] = pd.to_numeric(df["publicScore"], errors="coerce")
df["priv"] = pd.to_numeric(df["privateScore"], errors="coerce")
df["cv"] = df["description"].astype(str).str.extract(CV_RE)[0].astype(float)

print("=" * 92)
print("w142d — CV vs public leaderboard as a predictor of the PRIVATE score")
print("=" * 92)
print(f"  sends {len(df)}   with public {df['pub'].notna().sum()}   "
      f"with private {df['priv'].notna().sum()}   with a parseable CV {df['cv'].notna().sum()}")

# One row per distinct file: best published CV, best public, the private score.
best = (df.dropna(subset=["priv"]).groupby("stem")
          .agg(cv=("cv", "max"), pub=("pub", "max"), priv=("priv", "max"))
          .reset_index())
have = best.dropna(subset=["cv"]).copy()
print(f"  distinct files with a private score {len(best)}, of which carry a CV {len(have)}")
print(f"  private spread over the {len(best)} files: min {best['priv'].min():.5f} "
      f"max {best['priv'].max():.5f} ({(best['priv'].max()-best['priv'].min())*1e6:.0f}e-6)")
print(f"  public  spread: min {best['pub'].min():.5f} max {best['pub'].max():.5f} "
      f"({(best['pub'].max()-best['pub'].min())*1e6:.0f}e-6)")

# ---------------------------------------------------------------- 1. rank correlation
print("\n" + "-" * 92)
print("1. WHICH SIGNAL TRACKS PRIVATE?  Spearman over the files that carry both.")
print("-" * 92)
r_cv = stats.spearmanr(have["cv"], have["priv"])
r_pub = stats.spearmanr(have["pub"], have["priv"])
r_cvpub = stats.spearmanr(have["cv"], have["pub"])
print(f"  CV     vs private   rho {r_cv.statistic:+.3f}   p {r_cv.pvalue:.2e}   n {len(have)}")
print(f"  public vs private   rho {r_pub.statistic:+.3f}   p {r_pub.pvalue:.2e}   n {len(have)}")
print(f"  CV     vs public    rho {r_cvpub.statistic:+.3f}   p {r_cvpub.pvalue:.2e}")

# Paired bootstrap on the DIFFERENCE of the two rhos -- the two are measured on the same
# files, so an unpaired comparison would overstate the uncertainty.
rng = np.random.default_rng(0)
cv_a, pub_a, priv_a = have["cv"].values, have["pub"].values, have["priv"].values
d = []
for _ in range(20000):
    i = rng.integers(0, len(have), len(have))
    if len(np.unique(priv_a[i])) < 3:
        continue
    d.append(stats.spearmanr(cv_a[i], priv_a[i]).statistic
             - stats.spearmanr(pub_a[i], priv_a[i]).statistic)
d = np.array(d)
lo, hi = np.percentile(d, [5, 95])
print(f"  rho(CV) - rho(public): {r_cv.statistic - r_pub.statistic:+.3f}, "
      f"paired-bootstrap 90% CI [{lo:+.3f}, {hi:+.3f}], P(CV better) {100*(d>0).mean():.1f}%")

# ---------------------------------------------------------------- 2. the selection rules
print("\n" + "-" * 92)
print("2. WHAT EACH RULE WOULD HAVE SELECTED.  Kaggle scores a pick as max over 2 files.")
print("-" * 92)
priv_board = kaggle_board.board(private=True)
scores = np.array([float(r["score"]) for r in priv_board])

def rank_for(s):
    """1-based rank a score would take on the private board, ties resolved to the WORST
    position in the tie block -- the conservative read, never flattering."""
    return int((scores > s + 1e-12).sum() + (np.abs(scores - s) <= 1e-12).sum())

def arm(name, picks):
    sel = have[have["stem"].isin(picks)] if picks else have.iloc[:0]
    if not len(sel):
        print(f"  {name:22s} (files not found)"); return None
    m = sel["priv"].max()
    print(f"  {name:22s} priv {m:.5f}  rank {rank_for(m):4d}   {list(sel['stem'])}")
    return m

top2_cv = list(have.nlargest(2, "cv")["stem"])
top2_pub = list(have.nlargest(2, "pub")["stem"])
top2_or = list(have.nlargest(2, "priv")["stem"])
m_cv = arm("argmax-CV pair", top2_cv)
m_pub = arm("argmax-public pair", top2_pub)
m_wanted = arm("WANTED (the pick)", ["w36_ad199stdcorr", "w23_ad187stdcorr"])
m_auto = arm("AUTO (what happened)", ["w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"])
m_or = arm("ORACLE pair (hindsight)", top2_or)

print(f"\n  realised (board): 0.97093 rank 319 of {len(scores)}")
print(f"  oracle cost: the best pair beats what was auto-selected by "
      f"{(m_or - m_auto)*1e6:+.0f}e-6, worth {rank_for(m_auto) - rank_for(m_or):+d} ranks")
print(f"  argmax-public would have scored {(m_pub - m_auto)*1e6:+.0f}e-6 vs AUTO; "
      f"argmax-CV {(m_cv - m_auto)*1e6:+.0f}e-6 vs AUTO")

# ---------------------------------------------------------------- 3. did public mislead?
print("\n" + "-" * 92)
print("3. THE ROGII CHECK.  Public->private movement of this account vs the field.")
print("-" * 92)
pub_board = kaggle_board.board(private=False)
pub_rank = {r["teamId"]: i + 1 for i, r in enumerate(pub_board)}
priv_rank = {r["teamId"]: i + 1 for i, r in enumerate(priv_board)}
common = [t for t in pub_rank if t in priv_rank]
mv = np.array([priv_rank[t] - pub_rank[t] for t in common])
ours = priv_rank[TEAM_ID] - pub_rank[TEAM_ID]
print(f"  {len(common)} teams on both boards. rank movement (private - public):")
print(f"    median {np.median(mv):+.0f}   mean {mv.mean():+.1f}   sd {mv.std():.1f}")
print(f"    |move| percentiles  50% {np.percentile(np.abs(mv),50):.0f}   "
      f"90% {np.percentile(np.abs(mv),90):.0f}   99% {np.percentile(np.abs(mv),99):.0f}")
print(f"  ours: {pub_rank[TEAM_ID]} -> {priv_rank[TEAM_ID]} ({ours:+d}), "
      f"which is quieter than {100*(np.abs(mv) > abs(ours)).mean():.1f}% of the field")

# The top of the public board is where overfitting shows. Did it?
top100 = [t for t in common if pub_rank[t] <= 100]
mv100 = np.array([priv_rank[t] - pub_rank[t] for t in top100])
print(f"  public top-100: median move {np.median(mv100):+.0f}, "
      f"{(mv100 > 0).sum()} of {len(mv100)} fell, worst {mv100.max():+d}")
