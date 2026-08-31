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

# ---- w140 additions (2026-08-31 16:0xZ, still PRE-CLOSE). Plumbing, not predictions. -------
# w140a audited this script before its single execution and found two defects; the two
# constants below exist only to close the second one. Neither is a threshold and neither
# grades a P -- the frozen block above is untouched.
#
#   A  `--page-size 300` was the whole submission read. The server caps a page at 200
#      (measured: 200/300/1000 all return 200 against a true total of 201), so the read
#      silently dropped the oldest row -- the third site of the bug w138 fixed in
#      check_selection.py. All four graded stems survived it, so no verdict moved, but a
#      page size is not a pagination fix. Now paginated, with the count asserted.
#
#   D  `is_private` was `(score != PUBLIC_AT_CLOSE) or n_priv > 0`, and `n_priv` counts rows
#      in the SUBMISSION LIST. After the close that disjunct is true whichever board `lb`
#      holds, so a public leaderboard passes the check that exists to reject one. Now the
#      board has to testify about itself, and the submission list only corroborates.
PUBLIC_TOP_AT_AUDIT = 0.97207     # leader's public score, read 2026-08-31 15:59Z, pre-close
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"   # the one holding kagglesdk
# -------------------------------------------------------------------------------------------

def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, timeout=600).stdout


_SUBS_SNIPPET = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup

tok = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["access_token"]
rows, page_token, pages = [], None, 0
while True:
    with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
        r = ApiListSubmissionsRequest()
        r.competition_name = %r
        r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL
        r.page_size = 200
        if page_token:
            r.page_token = page_token
        resp = c.competitions.competition_api_client.list_submissions(r)
    rows += [(str(s.ref), s.file_name, s.public_score, s.private_score)
             for s in resp.submissions]
    page_token = resp.next_page_token
    pages += 1
    if not page_token or pages >= 25:
        break
print(json.dumps({"rows": rows, "pages": pages, "truncated": bool(page_token)}))
"""


def submissions():
    """Every successful submission, paginated to exhaustion. Never a single page."""
    p = subprocess.run([KAGGLE_PY, "-c", _SUBS_SNIPPET % COMP],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print(p.stderr[-1500:])
        return None, "kagglesdk read failed"
    d = json.loads(p.stdout)
    df = pd.DataFrame(d["rows"], columns=["ref", "fileName", "publicScore", "privateScore"])
    for c in ("publicScore", "privateScore"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if d["truncated"]:
        return df, f"hit the 25-page cap with a token still live at {len(df)} rows"
    if df["ref"].duplicated().any():
        return df, "the paginated read returned duplicate refs"
    print(f"  submissions: {len(df)} rows over {d['pages']} page(s), "
          f"{df['ref'].nunique()} unique refs — paginated to exhaustion")
    return df, None

def private_board():
    """Download the board and say WHICH FILE it came from. w140: `csvs[0]` on an unsorted
    glob was picking a leaderboard nobody named out loud."""
    d = tempfile.mkdtemp()
    sh("kaggle", "competitions", "leaderboard", "-c", COMP, "-d", "-p", d)
    for z in glob.glob(os.path.join(d, "*.zip")):
        zipfile.ZipFile(z).extractall(d)
    csvs = sorted(glob.glob(os.path.join(d, "*.csv")))
    if not csvs:
        return None, None
    # If the zip ever carries both boards, take the one that does not announce itself as
    # public. Deterministic, and it prefers the board we came for.
    pick = next((c for c in csvs if "publicleaderboard" not in os.path.basename(c)), csvs[0])
    if len(csvs) > 1:
        print(f"  ⚠ {len(csvs)} leaderboard CSVs in the download: "
              f"{[os.path.basename(c) for c in csvs]}")
    return pd.read_csv(pick), os.path.basename(pick)

def verdict(tag, claim, held, detail):
    mark = "HELD" if held else "FALSIFIED" if held is False else "UNGRADED"
    print(f"  {mark:10s} {tag}  {claim}")
    print(f"             {detail}")

print("=" * 92)
print(f"w135b — grading the frozen forecast.  now {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M}Z")
print("=" * 92)

lb, lb_file = private_board()
if lb is None:
    print("  could not download a leaderboard. Nothing graded."); sys.exit(1)
print(f"  board file: {lb_file}")
lb = lb.sort_values("Score", ascending=False).reset_index(drop=True)
n = len(lb)
row = lb.index[lb["TeamName"] == TEAM]
if not len(row):
    print(f"  {TEAM} not on the board. Nothing graded."); sys.exit(1)
r = int(row[0]) + 1
cut = int(np.floor(0.10 * n))
score = float(lb.iloc[row[0]]["Score"])
subs, subs_err = submissions()
if subs is None:
    print(f"  could not read the submission list ({subs_err}). Nothing graded."); sys.exit(1)
if subs_err:
    print(f"\n  ⚠ INCOMPLETE SUBMISSION READ: {subs_err}. STOP."); sys.exit(1)
subs["stem"] = subs["fileName"].str.replace(r"\.csv$", "", regex=True)

# w140 gate D. The board must testify about ITSELF. `n_priv` counts rows in the submission
# list, so after the close it is true whichever board `lb` holds -- it corroborates, it does
# not decide. The filename only ever argues FOR private, so a post-close rename cannot cause
# a false refusal on its own.
n_priv = int(subs["privateScore"].notna().sum())
top = float(lb.iloc[0]["Score"])
ev_board = {
    "our score moved off public":  score != PUBLIC_AT_CLOSE["score"],
    "leader's score moved off public": top != PUBLIC_TOP_AT_AUDIT,
    "filename does not say publicleaderboard": "publicleaderboard" not in (lb_file or ""),
}
ev_subs = n_priv > 0
is_private = any(ev_board.values()) and ev_subs
print(f"\n  board: rank {r} of {n}, score {score}, top-decile line {cut}, margin {cut - r:+d}")
print(f"  public at close was rank {PUBLIC_AT_CLOSE['rank']} of {PUBLIC_AT_CLOSE['n']}, "
      f"score {PUBLIC_AT_CLOSE['score']}, line {PUBLIC_AT_CLOSE['cut']}")
print(f"\n  is this the private board? ({len(lb)} teams, leader {top}, "
      f"{n_priv} submissions carry a privateScore)")
for k, v in ev_board.items():
    print(f"    {'yes' if v else 'no ':3s}  [board] {k}")
print(f"    {'yes' if ev_subs else 'no ':3s}  [submission list] at least one privateScore "
      f"(corroborating only)")
print("  top 3 on the board it loaded: "
      + ", ".join(f"{t} {s}" for t, s in zip(lb["TeamName"].head(3), lb["Score"].head(3))))
if not is_private and "--force-private" not in sys.argv:
    print("\n  ⚠ THIS DOES NOT LOOK LIKE THE PRIVATE BOARD. Either the close has not happened,")
    print("    or the download handed back the public leaderboard. Grading it now would put")
    print("    PUBLIC ranks under a PRIVATE label in P2 and P3, which is the one outcome this")
    print("    check exists to prevent. STOP.")
    print("    If the evidence above is wrong -- you have read it and the board IS private --")
    print("    re-run with `--force-private` and say in the journal that you overrode it.")
    sys.exit(1)
if "--force-private" in sys.argv and not is_private:
    print("\n  ⚠ OVERRIDDEN with --force-private. Record this in JOURNAL.md.")

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
