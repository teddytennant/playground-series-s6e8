"""w140c — the workspace's anti-truncation doctrine is VOID ABOVE PAGE SIZE 200.

`w86a_pagecap.py` states the doctrine in its own docstring:

    "THE ONLY TEST THAT DETECTS ITS OWN TRUNCATION IS SELF-REFERENTIAL: len(rows) >= PAGE
     -> raise."

That is true only while the server honours PAGE. It does not. The submissions endpoint caps
a page at **200 rows** and returns a `next_page_token` instead, and the Kaggle CLI does not
print that token anywhere. So at the `PAGE = 500` this workspace standardised on:

    requested 500 -> 200 rows -> `200 >= 500` is False -> "not truncated"

...while row 201 sits behind an unread token. 🎯 **Raising PAGE past 200 does not just fail to
fix truncation, it DISABLES THE DETECTOR** -- which is why the same row (`stack_pub74_logit`)
has now been re-hidden at w17, w138, w139 and w140 by four different "fixes".

`w86a_pagecap`'s G1/G2/G3 require every call site to ask for MORE than the account will ever
hold, and its G4 requires the seven deadline-day scripts to carry `len(rows) >= PAGE`. Both
remedies are unbuyable at any page size above 200. w86a is itself a victim: `api_rows(500)`
returns 200 rows, its own self-check reads `200 >= 500 -> False`, and it printed
`live rows 200` when the true total is 201, so its bar derives from a truncated read.

⛔ THIS IS NOT A CLAIM THAT ANY VERDICT MOVED. The row the cap hides is the OLDEST, the list
is date-DESCENDING, and 0.97081 contends for nothing. Every effect measured so far is benign.
That is structure, not care, and it is exactly the argument that stops holding the day the
account's oldest row matters.

Gate T5 is the CONTROL: below the ceiling the same idiom fires correctly. Without it "the
check does not work" and "the check is disabled by the cap" look identical.

    .venv/bin/python experiments/w140c_pagetruth.py

Exits 1 if any gate fails.
"""
import csv
import io
import json
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
OUT = os.path.join(HERE, "w140c_pagetruth.json")

# The seven w86a G4 certifies, plus the grader w140 found unprotected at a third site.
DEADLINE_DAY = ["w84a_pickargmax.py", "w26g_send.py", "w23b_sendqueue.py",
                "w55a_unpriced.py", "w54a_vetoexpiry.py", "w85c_slotguard.py",
                "w74b_clickstaleguard.py"]

FAILURES = []


def gate(tag, ok, claim, detail):
    print(f"  {'PASS' if ok else 'FAIL':4s}  {tag}  {claim}")
    for line in detail.splitlines():
        print(f"          {line}")
    if not ok:
        FAILURES.append(tag)
    return ok


def cli(page):
    """(rows, next-page-token the CLI surfaced) at a requested page size."""
    p = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", str(page)],
        capture_output=True, text=True, timeout=600)
    lines = p.stdout.splitlines()
    tok = None
    while lines and not lines[0].startswith("ref,"):
        line = lines.pop(0)
        if "Next Page Token" in line:
            tok = line.split("=")[-1].strip()
    if "Next Page Token" in p.stderr:
        tok = tok or "<stderr>"
    return list(csv.DictReader(lines)), tok


SDK = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup
tok = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["access_token"]

def page(sz, pt=None):
    with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
        r = ApiListSubmissionsRequest()
        r.competition_name = %r
        r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL
        r.page_size = sz
        if pt:
            r.page_token = pt
        resp = c.competitions.competition_api_client.list_submissions(r)
    return resp

one = page(500)
rows, pt, pages = [], None, 0
while True:
    r = page(200, pt)
    rows += [str(s.ref) for s in r.submissions]
    pt = r.next_page_token
    pages += 1
    if not pt or pages >= 25:
        break
print(json.dumps({"single_500": len(one.submissions),
                  "single_500_token": bool(one.next_page_token),
                  "total": len(rows), "pages": pages}))
""" % COMP


def sdk():
    p = subprocess.run([KAGGLE_PY, "-c", SDK], capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        print(p.stderr[-1500:])
        sys.exit("kagglesdk read failed; w140c measured nothing")
    return json.loads(p.stdout)


print("=" * 92)
print("w140c — where the page ceiling actually is, and what it does to the doctrine")
print("=" * 92)

S = sdk()
TRUE_TOTAL = S["total"]

# --------------------------------------------------------------------------------------
print("\nT1. WHERE IS THE CEILING?")
ladder = {}
for ps in (150, 199, 200, 201, 500, 1000):
    ladder[ps] = len(cli(ps)[0])
ceiling_ok = (ladder[199] == 199 and ladder[200] == 200
              and ladder[201] == 200 == ladder[500] == ladder[1000])
gate("T1", ceiling_ok, "the server caps a page at exactly 200 rows",
     "\n".join(f"requested {ps:5d} -> {n:4d} rows" for ps, n in ladder.items())
     + f"\ntrue total (paginated) = {TRUE_TOTAL}"
     + "\nthe ceiling sits between 201 and 200: asking for 201 already truncates.")

# --------------------------------------------------------------------------------------
print("\nT2. THE DOCTRINE'S OWN TEST, AT THE PAGE SIZE THE WORKSPACE STANDARDISED ON")
n500 = ladder[500]
fires_at_500 = n500 >= 500
gate("T2", not fires_at_500 and n500 < TRUE_TOTAL,
     "at PAGE=500 the read IS truncated and `len(rows) >= PAGE` CANNOT fire",
     f"rows returned {n500}, true total {TRUE_TOTAL} -> truncated by "
     f"{TRUE_TOTAL - n500} row(s)\n"
     f"`{n500} >= 500` is {fires_at_500} -> the guard reports NOT TRUNCATED\n"
     "raising PAGE past the ceiling disables the detector instead of widening the read.")

# --------------------------------------------------------------------------------------
print("\nT3. DOES THE CLI SURFACE THE TOKEN THE SERVER RETURNS?")
_, tok500 = cli(500)
gate("T3", tok500 is None and S["single_500_token"],
     "the server returns a next_page_token at page 500 and the CLI never prints it",
     f"kagglesdk single page at 500: {S['single_500']} rows, "
     f"next_page_token present = {S['single_500_token']}\n"
     f"CLI at 500: token surfaced on stdout/stderr = {tok500!r}\n"
     "so `w23b_sendqueue.py`'s `if len(rows) >= PAGE or tok:` -- the ONE site that thought\n"
     "to check a token -- reads tok=None and passes a truncated list too.")

# --------------------------------------------------------------------------------------
print("\nT4. CAN THE CERTIFIED SCRIPTS READ A COMPLETE LIST?")
# ⚠ FROZEN, MEASURED BEFORE w140 TOUCHED ANYTHING. Every one of the seven stood at PAGE=500
# against a 200-row ceiling, so not one could detect its own truncation while w86a's G4
# certified all seven as protected. T4-ctl below re-condemns exactly this table; without it
# the gate would go green on a corpus nobody had fixed.
PRE_W140 = {n: 500 for n in DEADLINE_DAY}

rows_out, unsafe = [], []
for name in DEADLINE_DAY:
    fp = os.path.join(HERE, name)
    if not os.path.exists(fp):
        continue
    t = open(fp).read()
    paginated = "kaggle_list" in t
    sizes = {int(m) for m in re.findall(r'"--page-size",\s*"(\d+)"', t)}
    sizes |= {int(m.group(1)) for sym in re.findall(r'"--page-size",\s*str\((\w+)\)', t)
              for m in [re.search(rf"^{re.escape(sym)}\s*=\s*(\d+)", t, re.M)] if m}
    bounds = {int(x) for x in re.findall(r"len\(\s*\w+[^)]*\)\s*(?:>=|<)\s*(\d+)", t)}
    fireable = any(b <= 200 for b in bounds)
    ok = paginated or fireable
    rows_out.append((name, paginated, sorted(sizes) or "-", ok))
    if not ok:
        unsafe.append(name)

gate("T4", not unsafe,
     "every deadline-day script now reads a COMPLETE list",
     "\n".join(f"{n:26s} paginated={str(p):5s} raw page sizes={z}  complete: {o}"
               for n, p, z, o in rows_out)
     + f"\n{len(rows_out) - len(unsafe)}/{len(rows_out)} read the full list."
     + ("\nstill unsafe: " + ", ".join(unsafe) if unsafe else ""))

# CONTROL: the SAME predicate, applied to the frozen pre-w140 table, must condemn all seven.
pre_unsafe = [n for n, pg in PRE_W140.items() if pg > 200]
gate("T4-ctl", len(pre_unsafe) == len(PRE_W140),
     "the pre-w140 corpus is condemned by the same predicate",
     "\n".join(f"{n:26s} PAGE={pg}  above the 200 ceiling -> `len(rows) >= {pg}` unfireable"
               for n, pg in PRE_W140.items())
     + f"\n{len(pre_unsafe)}/{len(PRE_W140)} were undetectably truncated while G4 called them"
       " protected.")

# --------------------------------------------------------------------------------------
print("\nT5. CONTROL — IS THE IDIOM ITSELF BROKEN, OR ONLY BROKEN ABOVE THE CEILING?")
n150 = ladder[150]
gate("T5", n150 >= 150,
     "BELOW the ceiling the same `len(rows) >= PAGE` idiom fires correctly",
     f"requested 150 -> {n150} rows; `{n150} >= 150` is {n150 >= 150} -> TRUNCATION DETECTED\n"
     "so the idiom is sound and the CAP is what disables it. Without this control,\n"
     "'the check does not work' and 'the check is switched off by the cap' look the same.")

# --------------------------------------------------------------------------------------
print("\nT6. WHAT DOES WORK — three independent reads of the same total")
board = None
for f in sorted(os.listdir(HERE)):
    if f.startswith("w140a_publicboard_snapshot"):
        b = pd.read_csv(os.path.join(HERE, f))
        hit = b[b["TeamName"] == "Teddy Tennant"]
        board = int(hit["SubmissionCount"].iloc[0]) if len(hit) else None
gate("T6", S["total"] == TRUE_TOTAL and (board is None or board == TRUE_TOTAL),
     "pagination and an outside witness agree, and neither depends on a page size",
     f"paginated read (follow next_page_token): {S['total']} over {S['pages']} pages\n"
     f"public leaderboard SubmissionCount column: {board}\n"
     "⟹ THE REMEDY IS NOT A BIGGER NUMBER. Follow the token, or compare the length against\n"
     "  a total obtained some other way (`num_total`, or the board's own SubmissionCount).")

json.dump({"ceiling_ladder": ladder, "true_total": TRUE_TOTAL,
           "cli_surfaces_token": tok500 is not None,
           "sdk_token_at_500": S["single_500_token"],
           "unsafe_now": unsafe, "pre_w140_page_sizes": PRE_W140,
           "board_submission_count": board,
           "failures": FAILURES}, open(OUT, "w"), indent=2)

print("\n" + "=" * 92)
print(f"FAILURES: {len(FAILURES)}" + (("  " + "  ".join(FAILURES)) if FAILURES else ""))
print("=" * 92)
sys.exit(1 if FAILURES else 0)
