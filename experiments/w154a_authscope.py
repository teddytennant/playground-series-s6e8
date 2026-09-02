"""w154a -- a bare KaggleClient() answers the opening read with the user fields silently zeroed.

w153 fixed HOW the opening request is constructed (bare, then assign -- a kwarg raises
TypeError). It did not fix WHICH CLIENT the request is sent through, and that is a second,
independent axis with its own failure mode.

    KaggleApi(); authenticate(); build_kaggle_client()  -> user_rank 319  user_has_entered True
    KaggleClient()                                      -> user_rank 0    user_has_entered False

Both calls succeed. Both return the SAME correct values for every competition-scoped field --
deadline, team_count, evaluation_metric, max_daily_submissions, submissions_disabled. The two
fields that describe THIS ACCOUNT are the only ones that differ, and they differ toward
"you were never here": a rank of 0 and has_entered False on a board where this account has
201 scored submissions and finished 319 / 3531.

That direction is what makes it worth a guard rather than a note. A future run that reads
`user_has_entered False` off a bare client has been handed a true-looking, complete-looking
read whose only wrong fields are the ones that would make it conclude it never competed --
and, unlike w151's empty list or w152's TypeError, nothing raises and nothing looks empty.

This is the week's genus once more, one layer down from w153:
  w150 a true measurement over an incomplete population
  w151 the same over an empty one
  w152 a passing falsifier with no power
  w153 a fixed procedure published in a form that never ran
  w154 the fixed procedure, run through a client that drops the answer's user half

  C1  the two clients agree on every competition-scoped field -- so the defect cannot be
      found by checking whether the read "worked". Measured field by field, not asserted.
  C2  the two clients DISAGREE on the user-scoped fields, and the authenticated one carries
      the values the workspace's own records corroborate (rank 319, entered True).
  C3  the bare client's answer is wrong in the DIRECTION of absence: user_rank == 0 and
      user_has_entered is False, i.e. indistinguishable from a never-entered competition.
  C4  RESEARCH.md's published opening read goes through an authenticated client. A bare
      `KaggleClient()` feeding get_competition in the docs is the defect and fires.
  C5  --control over a frozen bare-client snippet: C4 fires on it, is silent on what shipped.
  C6  scope, measured: how many bare-client reads exist in experiments/ and agent/, and what
      this guard is blind to (it cannot tell a cached credential from a live one).

    .venv/bin/python experiments/w154a_authscope.py            # rc 0 = clean
    .venv/bin/python experiments/w154a_authscope.py --control  # exits 0, having fired

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
# The suite runs every stem under .venv/bin/python, which has numpy but NOT the kaggle SDK.
# Hand off to the uv tool interpreter rather than dying with ModuleNotFoundError, which inside
# the runner is indistinguishable from a real failure. The marker is an env var, NOT a realpath
# comparison -- both paths resolve to the same interpreter binary (w153).
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
try:
    import kaggle  # noqa: F401
except ModuleNotFoundError:
    import os
    import sys as _sys
    if os.path.exists(KAGGLE_PY) and os.environ.get("KAGGLE_PY_REEXEC") != "1":
        os.environ["KAGGLE_PY_REEXEC"] = "1"
        os.execve(KAGGLE_PY, [KAGGLE_PY, os.path.abspath(__file__), *_sys.argv[1:]], os.environ)
    raise

import pathlib
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiGetCompetitionRequest

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
COMP = "playground-series-s6e8"

# What the workspace's own records say, independent of this endpoint: w142b_privatecheck
# paginates 201 rows, and the leaderboard read puts this account at 319 of 3531.
KNOWN_RANK = 319

# Competition-scoped fields: these describe the contest, not the caller, and must match.
COMP_FIELDS = ("deadline", "team_count", "evaluation_metric", "max_daily_submissions",
               "submissions_disabled", "id", "category")
# User-scoped fields: these describe the caller, and are the ones a bare client drops.
USER_FIELDS = ("user_rank", "user_has_entered")

# Frozen specimen for --control: a bare-client opening read, the form this guard forbids.
CONTROL_TEXT = """
## THE OPENING READ

    r = ApiGetCompetitionRequest()
    r.competition_name = COMP
    with KaggleClient() as kc:
        comp = kc.competitions.competition_api_client.get_competition(r)
"""


def read_through(make_client, label):
    """Run the opening read through a client factory, return {field: value}."""
    r = ApiGetCompetitionRequest()
    r.competition_name = COMP
    with make_client() as kc:
        c = kc.competitions.competition_api_client.get_competition(r)
    out = {f: getattr(c, f) for f in COMP_FIELDS + USER_FIELDS}
    out["_label"] = label
    return out


def strip_code_comments(lines):
    """Blank out `#` comments on INDENTED lines, i.e. inside fenced/indented code blocks.

    A guard that forbids a token cannot read a comment ABOUT that token as an instance of it
    (w153's lesson, one level down). The head section's fixed snippet carries the trailing
    comment `# <- NOT a bare KaggleClient(); see w154 head`, which names the defect in order
    to warn about it. That is prose and must not fire.

    Only indented lines are treated as code: a markdown heading starts `#` at column 0, so
    stripping unconditionally would eat every heading in the document.
    """
    out = []
    for ln in lines:
        if ln[:1] in (" ", "\t") and "#" in ln:
            ln = ln[:ln.index("#")]
        out.append(ln)
    return out


def bare_client_reads(text):
    """Lines where a get_competition read is fed by an unauthenticated KaggleClient().

    Deliberately narrow: a bare `with KaggleClient() as ...` within a few lines of a
    get_competition call. `KaggleClient` named in prose, in a comment, or imported, is
    not a defect.
    """
    hits = []
    raw = text.splitlines()
    lines = strip_code_comments(raw)
    for i, ln in enumerate(lines):
        if not re.search(r"\bKaggleClient\(\s*\)", ln):
            continue
        window = "\n".join(lines[i:i + 6])
        if "get_competition" in window:
            hits.append((i + 1, raw[i].strip()))
    return hits


def main():
    control = "--control" in sys.argv
    fails = 0

    api = KaggleApi()
    api.authenticate()
    auth = read_through(api.build_kaggle_client, "AUTHENTICATED")
    bare = read_through(KaggleClient, "bare KaggleClient()")

    print("C1 the two clients agree on every competition-scoped field")
    for f in COMP_FIELDS:
        same = auth[f] == bare[f]
        fails += not same
        print(f"  {f:24s} auth={auth[f]!r:<28.28} bare={bare[f]!r:<28.28} "
              f"{'agree' if same else 'DIFFER -- FAIL'}")
    print("  -> the read cannot be validated by asking whether it 'worked'. OK")

    print("C2 the two clients disagree on the user-scoped fields")
    for f in USER_FIELDS:
        differ = auth[f] != bare[f]
        fails += not differ
        print(f"  {f:24s} auth={auth[f]!r:<10} bare={bare[f]!r:<10} "
              f"{'DIFFER, as documented' if differ else 'agree -- FAIL, defect gone?'}")
    rank_ok = auth["user_rank"] == KNOWN_RANK and auth["user_has_entered"] is True
    fails += not rank_ok
    print(f"  authenticated rank {auth['user_rank']} vs the workspace's own {KNOWN_RANK}, "
          f"entered {auth['user_has_entered']}  {'OK' if rank_ok else 'FAIL'}")

    print("C3 the bare client is wrong in the direction of ABSENCE")
    absent = bare["user_rank"] == 0 and bare["user_has_entered"] is False
    fails += not absent
    print(f"  bare reads rank 0 and entered False on a board with 201 scored submissions: "
          f"{'confirmed' if absent else 'NOT reproduced -- FAIL'}")
    print("  -> nothing raises and nothing looks empty, which is why w151/w152's shapes miss it.")

    print("C4 RESEARCH.md's opening read goes through an authenticated client")
    text = CONTROL_TEXT if control else RESEARCH.read_text()
    where = "the frozen bare-client specimen" if control else "RESEARCH.md"
    hits = bare_client_reads(text)
    if hits:
        for ln, snippet in hits[:10]:
            print(f"  L{ln:6d}  {snippet}")
        print(f"  {len(hits)} bare-client opening read(s) in {where}. "
              + ("FIRES, as it must on --control. OK" if control else "FAIL"))
        fails += not control
    else:
        print(f"  none in {where}. "
              + ("FAIL -- control did not fire" if control else "OK"))

    # The comment exemption must be an exemption for COMMENTS, not a blanket hole. Same line,
    # same token, differing only in whether it is commented out: silent then firing.
    commented = ("    api = KaggleApi()  # <- NOT a bare KaggleClient(); see w154\n"
                 "    c = kc.competitions.competition_api_client.get_competition(r)\n")
    uncommented = ("    with KaggleClient() as kc:\n"
                   "    c = kc.competitions.competition_api_client.get_competition(r)\n")
    n_com, n_unc = len(bare_client_reads(commented)), len(bare_client_reads(uncommented))
    pair_ok = n_com == 0 and n_unc == 1
    fails += not pair_ok
    print(f"  exemption is for comments only: commented -> {n_com} hit(s), "
          f"uncommented -> {n_unc} hit(s)  {'OK' if pair_ok else 'FAIL'}")

    print("C5 scope, measured rather than assumed")
    scanned = scoped = 0
    for d in ("experiments", "agent"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.name == pathlib.Path(__file__).name:
                continue
            try:
                t = p.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            scanned += 1
            n = len(bare_client_reads(t))
            if n:
                scoped += n
                print(f"  {p.relative_to(ROOT)}  {n} bare-client opening read(s)")
    print(f"  {scanned} file(s) scanned, {scoped} bare-client opening read(s) in code.")

    print("C6 what this guard is blind to")
    print("  it compares two live clients, so it cannot distinguish a stale cached credential")
    print("  from a live one -- both would read as AUTHENTICATED here.")
    print("  it checks only get_competition; other endpoints may drop user fields too.")

    print()
    print(f"FAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
