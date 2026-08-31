"""w135b — grade w135's pre-registered forecast against the private leaderboard.

Run this AFTER the 2026-08-31 23:59 UTC close, and run NOTHING ELSE first. Every threshold
below is frozen in experiments/w135_prereg.txt (commit 761f341), written before the private
board existed. The whole point is that the post-close run cannot choose what it predicted.

    .venv/bin/python experiments/w135b_grade.py

Exits 0 whether the predictions held or not -- being wrong is a result, not a failure. It
exits 1 only if it could not read the board, i.e. if it graded nothing.
"""
import datetime as dt
import json, os, subprocess, sys, glob, tempfile, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "playground-series-s6e8"
TEAM = "Teddy Tennant"

# ---- FROZEN AT COMMIT 761f341. Do not edit these after reading the board. -----------------
WANTED = ["w36_ad199stdcorr", "w23_ad187stdcorr"]
AUTO   = ["w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"]
P3_INTERVAL       = (122, 426)     # 90% central, common-shift null at sd 67e-6
P3_UNION_INTERVAL = (62, 485)      # union across sd 43 / 67 / 124e-6
PUBLIC_AT_CLOSE   = dict(rank=309, n=3463, score=0.97119, cut=346, margin=37)
# -------------------------------------------------------------------------------------------

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, timeout=600).stdout

def private_board():
    d = tempfile.mkdtemp()
    sh("kaggle", "competitions", "leaderboard", "-c", COMP, "-d", "-p", d)
    for z in glob.glob(os.path.join(d, "*.zip")):
        zipfile.ZipFile(z).extractall(d)
    csvs = glob.glob(os.path.join(d, "*.csv"))
    return pd.read_csv(csvs[0]) if csvs else None

def verdict(tag, claim, held, detail):
    mark = "HELD" if held else "FALSIFIED" if held is False else "UNGRADED"
    print(f"  {mark:10s} {tag}  {claim}")
    print(f"             {detail}")

print("=" * 92)
print(f"w135b — grading the frozen forecast.  now {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}Z")
print("=" * 92)

lb = private_board()
if lb is None:
    print("  could not download a leaderboard. Nothing graded."); sys.exit(1)
lb = lb.sort_values("Score", ascending=False).reset_index(drop=True)
n = len(lb)
row = lb.index[lb["TeamName"] == TEAM]
if not len(row):
    print(f"  {TEAM} not on the board. Nothing graded."); sys.exit(1)
r = int(row[0]) + 1
cut = int(np.floor(0.10 * n))
score = float(lb.iloc[row[0]]["Score"])
subs = pd.read_csv(pd.io.common.StringIO(
    sh("kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "300")))
subs["stem"] = subs["fileName"].str.replace(r"\.csv$", "", regex=True)
# Two independent signals, because a private score that rounds to the same 5 d.p. as the
# public one would make the score test alone refuse a board that is genuinely private.
n_priv = int(subs["privateScore"].notna().sum())
is_private = (score != PUBLIC_AT_CLOSE["score"]) or n_priv > 0
print(f"\n  board: rank {r} of {n}, score {score}, top-decile line {cut}, margin {cut - r:+d}")
print(f"  public at close was rank {PUBLIC_AT_CLOSE['rank']} of {PUBLIC_AT_CLOSE['n']}, "
      f"score {PUBLIC_AT_CLOSE['score']}, line {PUBLIC_AT_CLOSE['cut']}")
if not is_private:
    print("\n  ⚠ STILL THE PUBLIC BOARD. The score is unchanged from public AND no submission")
    print("    carries a privateScore yet, so the private board is not out. STOP: grading it")
    print("    now grades nothing.")
    sys.exit(1)

pv = {s: subs.loc[subs["stem"] == s, "privateScore"].max() for s in WANTED + AUTO}
print("\n  private score of each candidate file:")
for k, v in pv.items():
    print(f"    {k:26s} {'(none)' if pd.isna(v) else f'{v:.5f}'}"
          f"   {'WANTED' if k in WANTED else 'AUTO'}")

print("\n" + "=" * 92); print("THE FROZEN PREDICTIONS"); print("=" * 92)

if any(pd.isna(v) for v in pv.values()):
    verdict("P1", "nobody clicks; Kaggle auto-selects on public score", None,
            "private scores are not all published yet")
else:
    w_max, a_max = max(pv[k] for k in WANTED), max(pv[k] for k in AUTO)
    if w_max == a_max:
        verdict("P1", "nobody clicks; Kaggle auto-selects on public score", None,
                f"both pairs land on {w_max:.5f} at 5 d.p., so the board cannot tell them apart")
    else:
        clicked = abs(score - w_max) < abs(score - a_max)
        verdict("P1", "nobody clicks; Kaggle auto-selects on public score", not clicked,
                f"final {score:.5f}; WANTED max {w_max:.5f}, AUTO max {a_max:.5f} -> "
                f"{'THE CLICK HAPPENED' if clicked else 'auto-selected, nobody clicked'}")
        print(f"             realised click delta (WANTED max - AUTO max): "
              f"{(w_max - a_max)*1e6:+.1f}e-6 at the board's 5 d.p.")

verdict("P2", "the account finishes inside the top decile", r <= cut,
        f"rank {r} against the top-10% line {cut}, margin {cut - r:+d}")
verdict("P3", f"final private rank inside {P3_INTERVAL}", P3_INTERVAL[0] <= r <= P3_INTERVAL[1],
        f"rank {r}; union interval {P3_UNION_INTERVAL} "
        f"{'also holds' if P3_UNION_INTERVAL[0] <= r <= P3_UNION_INTERVAL[1] else 'ALSO FAILS'}")

pw = pd.read_csv(os.path.join(ROOT, "experiments", "w135a_clickpower.csv"))
ppos = float(pw.loc[pw["scale"] == "private 80%", "ppos"].iloc[0])
sd = float(pw.loc[pw["scale"] == "private 80%", "sd"].iloc[0])
verdict("P4", "one realisation cannot grade the CV-selection thesis", ppos < 0.95,
        f"w135a measured P(delta>0) = {100*ppos:.1f}% and sd {sd*1e6:.2f}e-6 on one private-"
        f"scale draw. Below 95%, so the power claim stands and the realised sign above is "
        f"ONE DRAW. Do not rewrite the pick's rationale to match it.")

print("\n" + "=" * 92)
print("  Movement public -> private: rank "
      f"{PUBLIC_AT_CLOSE['rank']} -> {r} ({r - PUBLIC_AT_CLOSE['rank']:+d}), "
      f"score {PUBLIC_AT_CLOSE['score']} -> {score:.5f} "
      f"({(score - PUBLIC_AT_CLOSE['score'])*1e6:+.0f}e-6)")
print("  Write the four verdicts above into JOURNAL.md verbatim, including the ones that")
print("  failed, and do not adjust an interval to fit. That is the entire point of w135.")
