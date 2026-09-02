"""w153a -- the opening read RESEARCH.md publishes cannot be run as published.

w152 replaced w151's broken list call with `get_competition` and wrote the replacement into
the head of RESEARCH.md as

    ApiGetCompetitionRequest(competition_name='playground-series-s6e8')

Every kagglesdk request type has a ZERO-ARGUMENT __init__. That line raises TypeError before
a socket is opened. It is therefore not the call w151/w152 ran -- it cannot be, because it
produces no output at all, and the deadline/rank block printed underneath it is real. Every
executable artefact in this workspace (w134a, w136a) builds the request the only way that
works: construct bare, then assign the attribute.

The head section that publishes the uncallable form is the same section that warns about two
OTHER ways this read can look like a dead token (`len()` on a response object raises TypeError;
fuzzy search makes row[0] a different competition). This is the third, and it was introduced by
the warning itself -- w152's own lesson, that a procedure living in prose is unguarded by
construction, applied to w152.

  C1  the kwarg form raises TypeError for every request type this workspace uses, and the
      attribute form constructs -- measured off inspect.signature, not asserted.
  C2  the attribute form completes the opening read live: deadline, rank, metric.
  C3  RESEARCH.md publishes no uncallable `ApiXRequest(kwarg=...)` construction anywhere.
  C4  --control over the frozen pre-fix text: C3 fires on it, and is silent on what shipped.
  C5  scope -- the executable artefacts were never wrong, so this defect's genus is
      DOC-ONLY. Every `ApiXRequest(` in experiments/ and agent/ is the bare form.

    .venv/bin/python experiments/w153a_openreadguard.py            # rc 0 = clean
    .venv/bin/python experiments/w153a_openreadguard.py --control  # exits 0, having fired

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
# The suite runs every stem under .venv/bin/python, which has numpy but NOT the kaggle SDK;
# API work in this workspace runs under the uv tool interpreter. A guard that needs the SDK
# must therefore hand itself over rather than die with ModuleNotFoundError, which inside the
# runner is indistinguishable from a real failure. Re-exec once, never in a loop.
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
try:
    import kaggle  # noqa: F401
except ModuleNotFoundError:
    import os
    import sys as _sys
    # NOT a realpath comparison: .venv/bin/python and the uv tool python resolve to the SAME
    # interpreter binary and differ only in site-packages, so realpath equality would suppress
    # the hand-off. An env marker is what makes the loop impossible.
    if os.path.exists(KAGGLE_PY) and os.environ.get("KAGGLE_PY_REEXEC") != "1":
        os.environ["KAGGLE_PY_REEXEC"] = "1"
        os.execve(KAGGLE_PY, [KAGGLE_PY, os.path.abspath(__file__), *_sys.argv[1:]], os.environ)
    raise

import inspect
import pathlib
import re
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetCompetitionRequest, ApiListCompetitionsRequest, ApiListSubmissionsRequest)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
COMP = "playground-series-s6e8"

# The published form, frozen verbatim as w152 shipped it. --control checks the guard against
# this rather than against a hand-written stand-in.
PREFIX_TEXT = """    KP=/home/nixos/.local/share/uv/tools/kaggle/bin/python
    $KP -c "...ApiGetCompetitionRequest(competition_name='playground-series-s6e8')..."
    -> deadline 2026-08-31 23:59:00  teams 3531  rank 319  metric Roc Auc Score
"""

# (type, the type's OWN field) -- so C1 tests the correct field name being rejected as a
# kwarg, not a misspelling. Assigning a field the type does not have raises AttributeError,
# which is a fourth way this read can look like a dead token.
REQ_TYPES = [
    (ApiGetCompetitionRequest, "competition_name"),
    (ApiListCompetitionsRequest, "search"),
    (ApiListSubmissionsRequest, "competition_name"),
]

# `ApiSomethingRequest(name=...)` -- a construction that passes an argument. The bare form
# `ApiSomethingRequest()` is what works and must not be flagged.
KWARG_CALL = re.compile(r"Api\w*Request\(\s*[^)\s][^)]*\)")

# A guard that forbids a string cannot describe the defect it forbids without tripping itself.
# The w153 section quotes the broken snippet once, as a specimen. It is exempt by EXACT text
# AND by position -- it has to sit inside that section, which ends where w152's header starts.
# Anywhere else in the document the same string is a live instruction and is a failure.
SPECIMENS = {"ApiGetCompetitionRequest(competition_name='playground-series-s6e8')"}
SECTION_END = "# (w152, 2026-09-02)"


def bad_constructions(text, allow_specimens=True):
    lines = text.splitlines()
    end = next((i for i, l in enumerate(lines, 1) if l.startswith(SECTION_END)), 0)
    out, exempt = [], []
    for i, line in enumerate(lines, 1):
        for m in KWARG_CALL.finditer(line):
            hit = m.group(0)
            if allow_specimens and hit in SPECIMENS and i < end:
                exempt.append((i, hit))
            else:
                out.append((i, hit))
    return out, exempt


def main():
    control = "--control" in sys.argv
    fails = 0

    print("C1 kagglesdk request types take no constructor arguments")
    for T, field in REQ_TYPES:
        sig = str(inspect.signature(T.__init__))
        try:
            T(**{field: COMP})
            raised = None
        except TypeError as e:
            raised = str(e)
        # the form every artefact uses constructs without complaint
        r = T()
        setattr(r, field, COMP)
        attr_ok = getattr(r, field) == COMP
        ok = sig == "(self)" and raised is not None and attr_ok
        fails += not ok
        print(f"  {T.__name__:28s} __init__{sig}  {field}= as kwarg -> "
              f"{'TypeError' if raised else 'ACCEPTED'}  as attribute -> "
              f"{'set' if attr_ok else 'FAILED'}  {'OK' if ok else 'FAIL'}")

    print("C2 the attribute form completes the opening read live")
    api = KaggleApi()
    api.authenticate()
    req = ApiGetCompetitionRequest()
    req.competition_name = COMP
    with api.build_kaggle_client() as kc:
        c = kc.competitions.competition_api_client.get_competition(req)
    live_ok = c.deadline is not None and c.evaluation_metric
    fails += not live_ok
    print(f"  deadline {c.deadline}  teams {c.team_count}  rank {c.user_rank}  "
          f"metric {c.evaluation_metric}  {'OK' if live_ok else 'FAIL'}")
    print(f"  submissions_disabled {c.submissions_disabled}  max_daily {c.max_daily_submissions}"
          "   (submissions_disabled reads False on a closed board -- w145)")

    print("C3 RESEARCH.md publishes no uncallable request construction")
    text = PREFIX_TEXT if control else RESEARCH.read_text()
    bad, exempt = bad_constructions(text)
    where = "the frozen pre-fix text" if control else "RESEARCH.md"
    for ln, hit in exempt:
        print(f"  L{ln:6d}  {hit}   allowlisted specimen, inside the w153 section")
    if bad:
        for ln, snippet in bad[:10]:
            print(f"  L{ln:6d}  {snippet}")
        print(f"  {len(bad)} uncallable construction(s) in {where}. "
              + ("FIRES, as it must on --control. OK" if control else "FAIL"))
        fails += not control
    else:
        print(f"  none in {where}. " + ("FAIL -- control did not fire" if control else "OK"))
        fails += control

    print("C4 the control differs from the shipped text on the defect and nothing else")
    ship_bad, ship_exempt = bad_constructions(RESEARCH.read_text())
    ctrl_bad, _ = bad_constructions(PREFIX_TEXT)
    # the specimen must NOT be exempt outside its section: same text, moved, is a failure
    moved = RESEARCH.read_text().replace(SECTION_END, "# (w152 MOVED)", 1)
    moved_bad, _ = bad_constructions(moved)
    ok = len(ctrl_bad) == 1 and len(ship_bad) == 0 and len(ship_exempt) == 1 and len(moved_bad) == 1
    fails += not ok
    print(f"  shipped {len(ship_bad)} bad / {len(ship_exempt)} allowlisted   frozen pre-fix "
          f"{len(ctrl_bad)} bad   specimen outside its section -> {len(moved_bad)} bad  "
          f"{'OK' if ok else 'FAIL'}")

    print("C5 scope: the executable artefacts were never wrong")
    hits = []
    for p in sorted(list((ROOT / "experiments").glob("*.py")) + list((ROOT / "agent").glob("*.py"))):
        if p.name == pathlib.Path(__file__).name:
            continue
        for ln, snippet in bad_constructions(p.read_text(), allow_specimens=False)[0]:
            hits.append((p.name, ln, snippet))
    fails += bool(hits)
    for name, ln, snippet in hits[:10]:
        print(f"  {name}:{ln}  {snippet}")
    print(f"  {len(hits)} uncallable construction(s) in code. "
          f"{'OK -- the defect is DOC-ONLY' if not hits else 'FAIL'}")

    print("C6 what this guard is blind to, measured rather than assumed")
    # RESEARCH.md is the document a run is told to copy commands out of, so it is the only
    # one C3 enforces. JOURNAL.md is append-only and LEADERBOARD.md is a dated log: both
    # carry the old form in entries that are history. Counted here so the blind spot is a
    # number in the output, not an omission.
    for name in ("JOURNAL.md", "LEADERBOARD.md"):
        f = ROOT / name
        n = len(bad_constructions(f.read_text(), allow_specimens=False)[0]) if f.exists() else 0
        print(f"  {name:16s} {n:3d} uncallable construction(s) — NOT enforced (history, append-only)")

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
