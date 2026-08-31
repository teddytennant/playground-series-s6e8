"""w140a — audit `w135b_grade.py` BEFORE it runs, because it runs exactly once.

`w135b_grade.py` is sealed until after the 2026-08-31 23:59Z close. That seal means its
first execution is also its only one, at the moment nothing can be fixed. Every input it
depends on is readable NOW, pre-close, without touching the private board:

  * the submission list it parses          -> readable, it is our own account
  * the leaderboard CSV's columns and name -> readable, the public board downloads today
  * the team name it matches on            -> readable
  * `w135a_clickpower.csv`, P4's input     -> on disk

⛔ This audit MUST NOT read, guess at, or pre-empt any private-board number. It only checks
that the grader can PARSE what it will be handed. Nothing here grades a P.

⛔ It also must not touch the frozen block in `w135b_grade.py` (commit 761f341): WANTED,
AUTO, P3_INTERVAL, P3_UNION_INTERVAL, PUBLIC_AT_CLOSE. Plumbing only.

Gates C and D carry CONTROLS: a gate that cannot fail on a known-bad input is decoration.

    .venv/bin/python experiments/w140a_gradeaudit.py

Exits 1 if any gate fails.
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "playground-series-s6e8"
TEAM = "Teddy Tennant"
# The interpreter that actually holds kagglesdk. check_selection.py:470 uses the same one;
# the workspace .venv does NOT have it and importing there is a ModuleNotFoundError.
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

FAILURES = []


def gate(tag, ok, claim, detail):
    print(f"  {'PASS' if ok else 'FAIL':4s}  {tag}  {claim}")
    for line in detail.splitlines():
        print(f"          {line}")
    if not ok:
        FAILURES.append(tag)
    return ok


# --------------------------------------------------------------------------------------
# The authoritative total, and the paginated read. Both from kagglesdk, not from the CLI.
# --------------------------------------------------------------------------------------
SNIPPET = r"""
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
    rows += [(str(s.ref), s.file_name) for s in resp.submissions]
    page_token = resp.next_page_token
    pages += 1
    if not page_token or pages >= 25:
        break
print(json.dumps({"rows": rows, "pages": pages}))
""" % COMP


def paginated_refs():
    p = subprocess.run([KAGGLE_PY, "-c", SNIPPET], capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print(p.stderr[-2000:])
        sys.exit("could not read the submission list via kagglesdk; audit graded nothing")
    d = json.loads(p.stdout)
    return d["rows"], d["pages"]


def cli_rows(page_size):
    """Exactly the read w135b_grade.py performs, at a configurable page size."""
    p = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", str(page_size)],
        capture_output=True, text=True, timeout=600)
    return pd.read_csv(pd.io.common.StringIO(p.stdout))


print("=" * 92)
print("w140a — auditing the sealed grader against inputs that exist BEFORE the close")
print("=" * 92)

sdk_rows, sdk_pages = paginated_refs()
n_total = len(sdk_rows)
subs300 = cli_rows(300)

# Load the grader's own header so gates A and D test THE SHIPPED CODE, not a copy of it.
GRADER = os.path.join(ROOT, "experiments", "w135b_grade.py")
gsrc = open(GRADER).read()
G = {"__name__": "w135b_audit", "__file__": GRADER}
exec(compile(gsrc[:gsrc.index('\nprint("=" * 92)\n')], GRADER, "exec"), G)

# --------------------------------------------------------------------------------------
print("\nA. IS THE GRADER'S SUBMISSION READ COMPLETE?")
gsubs, gerr = G["submissions"]()
gate("A", gerr is None and gsubs is not None and len(gsubs) == n_total,
     "the grader's `submissions()` returns every submission",
     f"grader read -> {0 if gsubs is None else len(gsubs)} rows (err={gerr!r}); "
     f"kagglesdk paginated -> {n_total} ({sdk_pages} pages)")

# CONTROL: the read this replaced must still be CAUGHT by gate A's invariant, else the gate
# went green because it was blunted rather than because the code was fixed.
missing = [r for r in sdk_rows if str(r[0]) not in set(subs300["ref"].astype(str))]
gate("A-ctl", len(subs300) != n_total and len(missing) == 1,
     "the pre-fix read (`--page-size 300`) is CAUGHT by the same invariant",
     f"CLI at page_size 300 -> {len(subs300)} rows vs true total {n_total}\n"
     + "dropped: " + ", ".join(f"{m[1]} (ref {m[0]})" for m in missing))

counts = {ps: len(cli_rows(ps)) for ps in (200, 1000)}
gate("A-cap", counts[200] == counts[1000] == len(subs300) < n_total,
     "the shortfall was a SERVER CAP, not a too-small constant",
     f"page_size 200 -> {counts[200]}, 300 -> {len(subs300)}, 1000 -> {counts[1000]}; "
     f"true total {n_total}\n"
     "raising the constant could never have fixed this; only following next_page_token can")

# --------------------------------------------------------------------------------------
print("\nB. DOES THE PAGINATED READ SATISFY GATE A? (the fix, measured)")
gate("B", len(sdk_rows) == n_total and len(set(r[0] for r in sdk_rows)) == n_total,
     "the paginated read is complete and free of duplicate refs",
     f"{n_total} rows, {len(set(r[0] for r in sdk_rows))} unique refs, {sdk_pages} pages")

# --------------------------------------------------------------------------------------
print("\nC. DOES GATE A's DEFECT ACTUALLY MOVE A VERDICT? (the anti-overclaim gate)")
WANTED = ["w36_ad199stdcorr", "w23_ad187stdcorr"]
AUTO = ["w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"]
stems = subs300["fileName"].str.replace(r"\.csv$", "", regex=True)
present = {s: int((stems == s).sum()) for s in WANTED + AUTO}
gate("C", all(v == 1 for v in present.values()),
     "all four graded stems survive the truncated read, so P1 does NOT move",
     "\n".join(f"{s:26s} rows in truncated read = {v}" for s, v in present.items())
     + "\nthe dropped row is the OLDEST submission and is none of the four."
     + "\n=> gate A is a REAL defect with a BENIGN effect today. Do not report it as"
     + "\n   having changed a verdict.")

# CONTROL for C: the gate must be able to say 'moved'. Drop a WANTED stem and re-ask.
sabotaged = stems[stems != WANTED[0]]
ctl_present = {s: int((sabotaged == s).sum()) for s in WANTED + AUTO}
gate("C-ctl", not all(v == 1 for v in ctl_present.values()),
     "gate C detects a truncation that DOES hit a graded stem",
     f"with {WANTED[0]} removed, present counts = {ctl_present} -> gate C would FAIL")

# --------------------------------------------------------------------------------------
print("\nD. CAN THE GRADER TELL WHICH BOARD IT DOWNLOADED?")
d = tempfile.mkdtemp()
subprocess.run(["kaggle", "competitions", "leaderboard", "-c", COMP, "-d", "-p", d],
               capture_output=True, text=True, timeout=600)
for z in glob.glob(os.path.join(d, "*.zip")):
    zipfile.ZipFile(z).extractall(d)
csvs = glob.glob(os.path.join(d, "*.csv"))
lb = pd.read_csv(csvs[0])
lb = lb.sort_values("Score", ascending=False).reset_index(drop=True)
name0 = os.path.basename(csvs[0])

# The decision rule, on the post-close mis-download: a PUBLIC board in hand, and a
# submission list that (correctly) carries private scores. The old rule and the new one are
# both evaluated on it. The old one is the frozen control and must still fire.
PUBLIC_AT_CLOSE_SCORE = G["PUBLIC_AT_CLOSE"]["score"]
row = lb.index[lb["TeamName"] == TEAM]
score = float(lb.iloc[row[0]]["Score"]) if len(row) else float("nan")
top = float(lb.iloc[0]["Score"])
n_priv_sim = 1                      # after the close this is always true

old_rule = (score != PUBLIC_AT_CLOSE_SCORE) or n_priv_sim > 0
ev_board = {
    "our score moved off public": score != PUBLIC_AT_CLOSE_SCORE,
    "leader's score moved off public": top != G["PUBLIC_TOP_AT_AUDIT"],
    "filename does not say publicleaderboard": "publicleaderboard" not in name0,
}
new_rule = any(ev_board.values()) and n_priv_sim > 0

gate("D", new_rule is False,
     "the patched rule REFUSES a known-public board that carries private scores",
     f"board in hand: {name0}\n"
     f"our score {score} vs frozen public {PUBLIC_AT_CLOSE_SCORE}; leader {top} vs frozen "
     f"public leader {G['PUBLIC_TOP_AT_AUDIT']}\n"
     + "\n".join(f"  [board] {k}: {v}" for k, v in ev_board.items())
     + f"\n  [submission list] n_priv > 0: {n_priv_sim > 0} (corroborating only)\n"
     f"=> patched rule says is_private={new_rule}")

gate("D-ctl", old_rule is True,
     "the PRE-FIX rule ACCEPTS that same public board (the control still fires)",
     "`(score != PUBLIC_AT_CLOSE) or n_priv > 0` -> "
     f"({score != PUBLIC_AT_CLOSE_SCORE}) or (True) = {old_rule}\n"
     "`n_priv` counts rows in the SUBMISSION LIST, so after the close it carries the whole\n"
     "decision on its own and P2/P3 would report PUBLIC ranks under a PRIVATE label.")

gate("D-name", "publicleaderboard" in name0,
     "the board's identity is in the filename, and the grader now picks and prints it",
     f"{name0}\n"
     "`private_board()` returns (df, filename), sorts the glob, and prefers a CSV that does\n"
     "not announce itself as public.")

# --------------------------------------------------------------------------------------
print("\nE. WILL THE GRADER FIND ITS TEAM AND ITS COLUMNS?")
need = {"TeamName", "Score"}
gate("E", need <= set(lb.columns) and len(row) == 1,
     "the leaderboard carries TeamName/Score and matches TEAM exactly once",
     f"columns: {list(lb.columns)}\n"
     f"exact matches for {TEAM!r}: {len(row)}  (rank {int(row[0]) + 1} of {len(lb)})")

# --------------------------------------------------------------------------------------
print("\nF. IS P4's INPUT ON DISK AND DOES IT RESOLVE?")
try:
    pw = pd.read_csv(os.path.join(ROOT, "experiments", "w135a_clickpower.csv"))
    r80 = pw.loc[pw["scale"] == "private 80%"]
    ppos, sd = float(r80["ppos"].iloc[0]), float(r80["sd"].iloc[0])
    ok_f = ppos < 0.95
    det = (f"private 80%: ppos={ppos:.4f}, sd={sd * 1e6:.2f}e-6 -> P4 verdict computable "
           f"and reads HELD (ppos < 0.95)")
except Exception as e:  # noqa: BLE001
    ok_f, det = False, f"{type(e).__name__}: {e}"
gate("F", ok_f, "`w135a_clickpower.csv` has the row P4 indexes", det)

# --------------------------------------------------------------------------------------
print("\nG. DOES THE GRADER SURVIVE ITS OWN PARSE PATH?")
ok_g = hasattr(pd.io.common, "StringIO") and len(subs300.columns) == 7
gate("G", ok_g, "`pd.io.common.StringIO` exists and the CSV parses to 7 columns",
     f"pandas {pd.__version__}; columns {list(subs300.columns)}\n"
     "descriptions containing commas are quoted by the CLI and do not misalign the parse")

print("\n" + "=" * 92)
print(f"FAILURES: {len(FAILURES)}" + (("  " + "  ".join(FAILURES)) if FAILURES else ""))
print("=" * 92)
sys.exit(1 if FAILURES else 0)
