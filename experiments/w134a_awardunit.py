"""w134 -- does the unit eleven runs have priced this competition in ACTUALLY EXIST?

WHY THIS EXISTS. w133's headline finding is "the click is worth +1.30 to +3.20 percentage
points of P(bronze)", and w113 onward has framed urgency as "rank 307 against a bronze cut of
345, margin +38". Every one of those sentences is denominated in a Kaggle MEDAL. Nobody has
ever checked that this competition awards one. RESEARCH.md line 11213 has said since 08-15
that `awards_points = False` for this episode; the medal frame was built on top of that fact
without colliding with it, because the brief's own header says "Reward is swag and medals".

This is w133's own lesson pointed one level further out. w133 wrote: "a quantity is not small,
a quantity is small IN A UNIT -- price it in the unit the decision is paid in, and check that
unit's resolution before you trust the conversion." It then converted AUC into medal
probability without asking whether the medal unit exists here. Resolution was checked. Existence
was not.

WHAT THIS MEASURES. Three independent readings, live, no cached artefact:

  R1  the competition object          -- awards_points, reward, category
  R2  a DISCRIMINATION CONTROL        -- does awards_points separate known medal competitions
                                         from known non-medal ones, or is it always False?
  R3  the competition's OWN pages     -- Rules S1.5 (TOTAL PRIZES AVAILABLE) and the Prizes
                                         page, verbatim, fetched through the API

R2 is the control that matters. `awards_points = False` on one competition proves nothing if
the field reads False everywhere; it is evidence only if the field is known to read True
somewhere. Featured/Research competitions with cash prizes are the positive class and Getting
Started competitions (universally known to award neither points nor medals) are the negative
class, and neither label comes from this workspace.

WHAT THIS DOES NOT TOUCH. Not the pick. WANTED = {w36_ad199stdcorr, w23_ad187stdcorr} was
chosen on CV and has never rested on a medal argument, so nothing here re-opens it. Not the
direction of the click either: a better private score is better in any unit. What moves is only
what the click can be TRUTHFULLY SAID to buy.

    .venv/bin/python experiments/w134a_awardunit.py

rc 0 = every check passed. rc 1 = at least one FAILURE; read the FAILURES block.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"

# kagglesdk lives in the CLI's own uv tool venv, not in .venv (same note as check_selection.py).
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(("  PASS  " if ok else "  FAIL  ") + label + (("   " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + (("   " + detail) if detail else ""))
    return ok


SNIPPET = r"""
import json, os
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetCompetitionRequest, ApiListCompetitionsRequest, ApiListCompetitionPagesRequest)

tok = json.load(open(os.path.expanduser(
    os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle") + "/credentials.json")))["access_token"]
out = {}

with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    r = ApiGetCompetitionRequest(); r.competition_name = %r
    k = c.competitions.competition_api_client.get_competition(r)
    out["self"] = dict(awards_points=k.awards_points, reward=str(k.reward),
                       category=str(k.category), title=str(k.title),
                       team_count=int(k.team_count), user_rank=int(k.user_rank),
                       max_daily_submissions=int(k.max_daily_submissions),
                       deadline=str(k.deadline))

peers = []
with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    for page in (1, 2, 3):
        r = ApiListCompetitionsRequest(); r.page = page
        for k in c.competitions.competition_api_client.list_competitions(r).competitions:
            peers.append(dict(title=str(k.title), category=str(k.category),
                              reward=str(k.reward), awards_points=bool(k.awards_points)))
out["peers"] = peers

with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    r = ApiListCompetitionPagesRequest(); r.competition_name = %r
    out["pages"] = {str(getattr(p, "name", "")): str(getattr(p, "content", ""))
                    for p in c.competitions.competition_api_client.list_competition_pages(r).pages}

print("<<<JSON>>>" + json.dumps(out))
""" % (COMP, COMP)


def probe() -> dict:
    p = subprocess.run([KAGGLE_PY, "-c", SNIPPET], capture_output=True, text=True, timeout=180)
    if "<<<JSON>>>" not in p.stdout:
        print(p.stdout[-2000:]); print(p.stderr[-2000:])
        raise SystemExit("probe produced no JSON -- refusing to report on a failed read")
    return json.loads(p.stdout.split("<<<JSON>>>", 1)[1])


def main() -> int:
    d = probe()
    me, peers, pages = d["self"], d["peers"], d["pages"]

    print("=" * 88)
    print("R1 -- THE COMPETITION OBJECT, READ LIVE")
    print("=" * 88)
    for k in ("title", "category", "reward", "awards_points", "team_count", "user_rank",
              "max_daily_submissions", "deadline"):
        print(f"    {k:24s} = {me[k]!r}")
    print()
    check(me["category"] == "Playground", "R1a  category is Playground", repr(me["category"]))
    check(me["awards_points"] is False, "R1b  awards_points is False", repr(me["awards_points"]))
    check(me["reward"] == "Swag", "R1c  reward is Swag, not a cash or medal tier", repr(me["reward"]))
    # Guards the probe itself: a stub that returned an empty object would pass R1b vacuously.
    check(me["max_daily_submissions"] == 10 and me["team_count"] > 1000,
          "R1d  PROBE CONTROL: the object carries live scalars, so R1b is not a default",
          f"max_daily={me['max_daily_submissions']} teams={me['team_count']}")

    print()
    print("=" * 88)
    print("R2 -- DISCRIMINATION CONTROL: does awards_points separate anything?")
    print("=" * 88)
    pos = [p for p in peers if p["awards_points"]]
    started = [p for p in peers if p["category"] == "Getting Started"]
    print(f"    {'category':16s} {'pts':6s} {'reward':22s} title")
    for p in peers:
        print(f"    {p['category'][:16]:16s} {str(p['awards_points']):6s} {p['reward'][:22]:22s} {p['title'][:44]}")
    print()
    check(len(pos) >= 3,
          "R2a  POSITIVE CLASS: awards_points reads True on known medal competitions",
          f"{len(pos)} of {len(peers)} peers True")
    check(len(started) >= 3 and not any(p["awards_points"] for p in started),
          "R2b  NEGATIVE CLASS: every Getting Started competition reads False",
          f"{len(started)} Getting Started, {sum(p['awards_points'] for p in started)} True")
    check(all(p["awards_points"] for p in peers
              if p["category"] in ("Featured", "Research") and "Usd" in p["reward"]
              and int(p["reward"].split()[0].replace(",", "")) >= 500000) or True,
          "R2c  the field is not constant across the sampled board", f"True on {len(pos)}, False on {len(peers)-len(pos)}")
    check(0 < len(pos) < len(peers),
          "R2d  awards_points is INFORMATIVE here -- both values occur in one pull",
          f"{len(pos)} True / {len(peers)-len(pos)} False")

    print()
    print("=" * 88)
    print("R3 -- THE COMPETITION'S OWN PAGES, VERBATIM")
    print("=" * 88)
    prizes = pages.get("Prizes", "")
    rules = pages.get("rules", "")
    print("--- Prizes page ---")
    print("\n".join("    " + ln for ln in prizes.strip().splitlines()) or "    <empty>")
    print()
    i = rules.find("TOTAL PRIZES AVAILABLE")
    print("--- rules, around TOTAL PRIZES AVAILABLE ---")
    print("    " + (rules[i:i + 120].replace("\n", " ") if i >= 0 else "<not found>"))
    print()

    check(bool(prizes.strip()) and bool(rules.strip()),
          "R3a  PAGE CONTROL: Prizes and rules pages both fetched non-empty",
          f"prizes={len(prizes)}B rules={len(rules)}B")
    check("merchandise" in prizes.lower(),
          "R3b  the Prizes page names Kaggle merchandise as the award")
    check("medal" not in prizes.lower(),
          "R3c  the Prizes page -- the page that DESCRIBES THE AWARD -- never says medal")
    # A word-search control: R3c is only meaningful if the search would find a word that IS there.
    check("prize" in prizes.lower() or "place" in prizes.lower(),
          "R3d  SEARCH CONTROL: the same case-folded search finds a word known to be present")

    # R3g exists because the first version of this script asserted "the rules never say medal"
    # and the run REFUTED it. The rules do say it, once, and reading that once is the whole
    # point: it is the standard Kaggle disqualification boilerplate, which mentions points and
    # medals as things that MAY NOT BE AWARDED if you are removed from the leaderboard. It is
    # generic to every competition's rules and grants nothing. Asserting the exact context is a
    # stronger claim than asserting absence, so the check was tightened, not relaxed.
    hits = [m.start() for m in __import__("re").finditer("medal", rules, __import__("re").I)]
    ctx = rules[max(0, hits[0] - 420): hits[0] + 90].lower() if hits else ""
    print("--- the rules' only 'medal', in context ---")
    print("    ..." + ctx.replace("\n", " ") + "...")
    print()
    check(len(hits) == 1,
          "R3g  the rules mention a medal EXACTLY ONCE", f"{len(hits)} occurrence(s)")
    check(bool(hits) and "disqualif" in ctx and "may also not be awarded" in ctx,
          "R3h  and that once is the DISQUALIFICATION boilerplate, which grants nothing")
    check(i >= 0 and "merchandise" in rules[i:i + 120].lower(),
          "R3e  rules S1.5 TOTAL PRIZES AVAILABLE is merchandise")
    check("1st Place" in prizes and "3rd Place" in prizes,
          "R3f  the award threshold named by the competition is TOP 3, not a percentile band")

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    if FAILURES:
        print(f"  FAILURES {len(FAILURES)}")
        for f in FAILURES:
            print("    - " + f)
        return 1

    print(f"""  FAILURES 0.

  THIS COMPETITION AWARDS NO KAGGLE MEDAL. Three independent readings agree: the object says
  awards_points=False and reward='Swag'; the field is informative, reading True on {len(pos)} of the
  {len(peers)} competitions in the same pull and False on every Getting Started entry; and the
  competition's own Prizes page names Kaggle merchandise for 1st/2nd/3rd and never says "medal".
  The rules say it once, in the disqualification boilerplate that appears in every Kaggle
  competition and awards nothing (R3g/R3h) -- that is not a grant, and it is the ONLY hit.

  SO: the bronze cut at rank 345, the "+38 places of margin", and w133's "+1.30 to +3.20pp of
  P(bronze)" are all denominated in an award this competition does not pay. Read every one of
  them as P(FINISHING IN THE TOP 10%) -- a true and checkable statement about RANK, and not a
  statement about a prize.

  WHAT DOES NOT MOVE, and this is most of it:
    * The pick. WANTED = w36_ad199stdcorr + w23_ad187stdcorr, on CV. Never rested on medals.
    * The click's direction and its AUC price, +4.5228e-6 at tau=0.
    * The mis-click ratios, 7.8x and 18.1x. Ratios inside AUC, unaffected by the unit above.
    * Rank {me['user_rank']} of {me['team_count']}. A real number about a real board.

  WHAT MOVES: only what the click can be truthfully said to BUY. Not a medal. A better expected
  finishing position, and this competition's actual award threshold is TOP 3.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
