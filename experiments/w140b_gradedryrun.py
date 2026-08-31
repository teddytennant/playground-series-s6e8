"""w140b — run `w135b_grade.py`'s GRADING SECTION on synthetic private boards.

w140a checked that the grader can parse its inputs. That is not the same as checking it can
grade. The grading section -- the four verdicts, P1's three branches, the movement footer --
has never executed at all, and after the close it executes once.

⛔ EVERY NUMBER BELOW IS FABRICATED. This script invents private scores and private ranks so
the code paths run. It grades NOTHING, it writes NOTHING, and no verdict it prints is a
result. Its only output is "the branch ran" or "the branch crashed".

How it works, stated plainly because it matters: the grader's source is split at its first
top-level banner. The header (imports, frozen block, `private_board`, `submissions`,
`verdict`) is exec'd, the two I/O functions are then REPLACED with fixtures, and the body is
exec'd against them. Nothing in `w135b_grade.py` is edited, and the frozen constants are the
real ones -- P3_INTERVAL and PUBLIC_AT_CLOSE come from the file, not from here.

    .venv/bin/python experiments/w140b_gradedryrun.py

Exits 1 if any scenario crashes or if a branch that should have been reached was not.
"""
import io
import os
import sys
import contextlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "experiments", "w135b_grade.py")
ANCHOR = '\nprint("=" * 92)\n'

src = open(SRC).read()
cut = src.index(ANCHOR)
HEADER, BODY = src[:cut], src[cut:]

# The real submission list, so stems and shapes are genuine; only privateScore is invented.
REAL = pd.read_csv(os.path.join(ROOT, "experiments", "w140a_subs_snapshot.csv"))
WANTED = ["w36_ad199stdcorr", "w23_ad187stdcorr"]
AUTO = ["w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4"]

FAILURES = []


def fixtures(pv, team_score, team_rank, n_teams=3487, fname="…-privateleaderboard-….csv"):
    """A synthetic private board placing TEAM at `team_rank`, and a submission list whose
    four graded stems carry the private scores in `pv` (None -> unpublished)."""
    # Build the board so TEAM lands at exactly `team_rank` AFTER the grader re-sorts by
    # Score descending: k-1 scores strictly above it, the rest strictly below.
    above = team_score + np.linspace(0.002, 1e-6, team_rank - 1)
    below = team_score - np.linspace(1e-6, 0.05, n_teams - team_rank)
    scores = np.concatenate([above, [team_score], below])
    assert scores[team_rank - 1] == team_score and (np.diff(scores) < 0).all()
    lb = pd.DataFrame({
        "Rank": np.arange(1, n_teams + 1),
        "TeamId": np.arange(n_teams),
        "TeamName": [f"team{i}" for i in range(n_teams)],
        "Score": scores,
    })
    lb.loc[team_rank - 1, "TeamName"] = "Teddy Tennant"

    subs = REAL.copy()
    subs["privateScore"] = np.nan
    for stem, v in pv.items():
        if v is not None:
            subs.loc[subs["fileName"] == stem + ".csv", "privateScore"] = v
    return lb, fname, subs


def run(tag, pv, team_score, team_rank, expect, board=None, argv=None, want_exit=0):
    ns = {"__name__": "w135b_dryrun", "__file__": SRC}
    exec(compile(HEADER, SRC, "exec"), ns)
    lb, fname, subs = fixtures(pv, team_score, team_rank)
    if board is not None:
        lb, fname = board
    old_argv = sys.argv
    sys.argv = ["w135b_grade.py"] + (argv or [])
    ns["private_board"] = lambda: (lb, fname)
    ns["submissions"] = lambda: (subs, None)
    buf = io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(BODY, SRC, "exec"), ns)
    except SystemExit as e:
        code = e.code or 0
    except Exception as e:  # noqa: BLE001
        sys.argv = old_argv
        out = buf.getvalue()
        print(f"  CRASH  {tag}: {type(e).__name__}: {e}")
        print("\n".join("         " + l for l in out.splitlines()[-12:]))
        FAILURES.append(tag)
        return
    sys.argv = old_argv
    out = buf.getvalue()
    hit = expect in out
    ok = hit and code == want_exit
    print(f"  {'PASS' if ok else 'FAIL':4s}  {tag}  (exit {code}, wanted {want_exit})")
    for line in out.splitlines():
        if line.strip().startswith(("HELD", "FALSIFIED", "UNGRADED")):
            print("          " + line.strip())
    if not hit:
        print(f"          expected to see {expect!r} and did not")
    if not ok:
        FAILURES.append(tag)


print("=" * 92)
print("w140b — DRY RUN. Every private number below is FABRICATED and grades nothing.")
print("=" * 92)
print("\nP1's three branches, plus P2/P3 on both sides of the frozen interval:\n")

# Nobody clicked: the account's score equals the AUTO pair's max. P1 predicted this -> HELD.
run("S1 auto-selected, rank 300 (inside P3)",
    {WANTED[0]: 0.97100, WANTED[1]: 0.97098, AUTO[0]: 0.97105, AUTO[1]: 0.97103},
    team_score=0.97105, team_rank=300, expect="auto-selected, nobody clicked")

# The click happened: the account's score equals the WANTED pair's max. P1 -> FALSIFIED.
run("S2 the click happened, rank 300",
    {WANTED[0]: 0.97110, WANTED[1]: 0.97098, AUTO[0]: 0.97105, AUTO[1]: 0.97103},
    team_score=0.97110, team_rank=300, expect="THE CLICK HAPPENED")

# Both pairs land on the same 5 d.p. -- the board cannot tell them apart. P1 -> UNGRADED.
run("S3 pairs tie at 5 d.p., rank 300",
    {WANTED[0]: 0.97105, WANTED[1]: 0.97098, AUTO[0]: 0.97105, AUTO[1]: 0.97103},
    team_score=0.97105, team_rank=300, expect="cannot tell them apart")

# Private scores not published yet. P1 -> UNGRADED, and P2/P3 must still grade.
run("S4 private scores unpublished, rank 300",
    {WANTED[0]: 0.97105, WANTED[1]: None, AUTO[0]: None, AUTO[1]: 0.97103},
    team_score=0.97105, team_rank=300, expect="not all published yet")

# Outside the frozen P3 interval [122, 426] and outside the top decile -> both FALSIFIED.
run("S5 rank 900: outside P3 and outside the decile",
    {WANTED[0]: 0.97100, WANTED[1]: 0.97098, AUTO[0]: 0.97105, AUTO[1]: 0.97103},
    team_score=0.97105, team_rank=900, expect="ALSO FAILS")

# The post-close mis-download: a genuinely PUBLIC board in hand while the submission list
# already carries private scores. This is the case the pre-fix `or n_priv > 0` waved through.
PUB = pd.read_csv(os.path.join(ROOT, "experiments", "w140a_publicboard_snapshot.csv"))
PUB_BOARD = (PUB, "playground-series-s6e8-publicleaderboard-2026-08-31T16:04:10.csv")
FULL_PV = {WANTED[0]: 0.97100, WANTED[1]: 0.97098, AUTO[0]: 0.97105, AUTO[1]: 0.97103}

print("\nThe board-identity gate, end to end through the real code path:\n")
run("S6 PUBLIC board + private scores -> must REFUSE",
    FULL_PV, team_score=0.97105, team_rank=300,
    expect="THIS DOES NOT LOOK LIKE THE PRIVATE BOARD", board=PUB_BOARD, want_exit=1)

run("S7 same, with --force-private -> must grade and say so",
    FULL_PV, team_score=0.97105, team_rank=300,
    expect="OVERRIDDEN with --force-private", board=PUB_BOARD,
    argv=["--force-private"], want_exit=0)

print("\n" + "=" * 92)
print(f"FAILURES: {len(FAILURES)}" + (("  " + "  ".join(FAILURES)) if FAILURES else ""))
print("⛔ Nothing above is a result. The real grading happens once, after the close.")
print("=" * 92)
sys.exit(1 if FAILURES else 0)
