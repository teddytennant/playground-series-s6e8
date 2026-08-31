"""w136 -- is the BASELINE ARM of every price in this workspace real?

WHY THIS EXISTS. Every number this workspace has published about the final-selection click is
a difference against one counterfactual: "if nobody clicks, Kaggle auto-selects the best TWO
submissions by PUBLIC score." The click's +4.5228e-6, the mis-click's +35.17e-6 and +81.92e-6,
the auto-selection TIER that makes a send free or a liability in w26g_send.py, w50b's whole
enumeration -- all of them subtract that arm.

The sentence is used as a given at 30+ sites. Its earliest statement here is w15i_cvlb.py's
docstring (2026-08-15), which asserts it with no source, and I can find nowhere that it was
ever READ off Kaggle rather than assumed from convention.

This is w134's defect one door down. w134 asked whether the UNIT the click is priced in is paid
at all. This asks whether the COUNTERFACTUAL it is priced against actually happens.

WHAT THIS MEASURES. Two layers, because the documentary layer turns out not to settle it:

  R1  the competition object     -- live fields: daily cap, deadline, metric
  R2  the competition's OWN RULES -- the SUBMISSION LIMITS clause and the "Final Submission"
                                     definition, verbatim, fetched through the API
  R3  DISCRIMINATION CONTROL      -- is the final-submission limit competition-specific, or
                                     boilerplate that reads the same everywhere? A constant
                                     field is not evidence about THIS competition.
  R4  THE EMPIRICAL TEST          -- rogii-wellbore-geology-prediction, closed 2026-08-05, this
                                     same account, 8 scored submissions, a real public/private
                                     split, and nothing manually selected. Which selection rule
                                     reproduces the team's realised private score?

R4 is the check that carries the run. R2 establishes only that automatic selection HAPPENS; it
never states the criterion. R4 recovers the criterion from an outcome Kaggle actually produced.

WHY ROGII IS THE RIGHT PROBE, AND NOT A LUCKY ONE:
  * Its metric is MSE -- LOWER is better -- so "best" points the opposite way from S6E8's AUC.
    A rule that only looked like argmax by coincidence would break here.
  * The three candidate rules predict three DIFFERENT realised scores (9.606 / 9.573 / 9.529),
    so the test can fail. It is not a rule with only one possible outcome.
  * Exactly one of the 28 possible pairs of its 8 submissions reproduces the observed score,
    so the chance of a match under random pairing is 1/28 = 3.6%.

    .venv/bin/python experiments/w136a_autorule.py

rc 0 = every check passed. rc 1 = at least one FAILURE; read the FAILURES block.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import subprocess

COMP = "playground-series-s6e8"
PROBE = "rogii-wellbore-geology-prediction"
PROBE_TEAM = 16559999

# kagglesdk lives in the CLI's own uv tool venv, not in .venv (same note as check_selection.py).
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"

FAILURES: list[str] = []
N_CHECKS = 0


def check(ok: bool, label: str, detail: str = "") -> bool:
    global N_CHECKS
    N_CHECKS += 1
    print(("  PASS  " if ok else "  FAIL  ") + label + (("   " + detail) if detail else ""))
    if not ok:
        FAILURES.append(label + (("   " + detail) if detail else ""))
    return ok


SNIPPET = r'''
import json, os, re
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetCompetitionRequest, ApiListCompetitionPagesRequest, ApiListSubmissionsRequest,
    ApiGetLeaderboardRequest, SubmissionGroup)

tok = json.load(open(os.path.expanduser(
    os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle") + "/credentials.json")))["access_token"]
COMP, PROBE, TEAM = %r, %r, %d
out = {}

def rules_text(client, name):
    r = ApiListCompetitionPagesRequest(); r.competition_name = name
    pages = {str(p.name): str(p.content)
             for p in client.competitions.competition_api_client.list_competition_pages(r).pages}
    t = pages.get("rules", "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t)

with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    r = ApiGetCompetitionRequest(); r.competition_name = COMP
    k = c.competitions.competition_api_client.get_competition(r)
    out["self"] = dict(max_daily_submissions=int(k.max_daily_submissions),
                       deadline=str(k.deadline), evaluation_metric=str(k.evaluation_metric),
                       category=str(k.category), team_count=int(k.team_count))
    out["rules"] = rules_text(c, COMP)
    # R3 discrimination control: the same clause in OTHER live competitions.
    peer_rules = {}
    for p in ("pokemon-tcg-ai-battle", "kaggriculture", "rsna-knee-abnormality-detection"):
        try:
            peer_rules[p] = rules_text(c, p)
        except Exception as e:
            peer_rules[p] = "ERROR: %%s" %% e
    out["peer_rules"] = peer_rules

# R4: the probe competition's submissions, both score columns.
subs = []
with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    for page in (1, 2, 3, 4, 5):
        r = ApiListSubmissionsRequest(); r.competition_name = PROBE
        r.page = page; r.page_size = 200
        r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL
        res = c.competitions.competition_api_client.list_submissions(r).submissions
        if not res: break
        for s in res:
            subs.append(dict(ref=int(s.ref), pub=str(s.public_score), prv=str(s.private_score),
                             date=str(s.date), desc=str(s.description)[:80]))
        if len(res) < 200: break
out["probe_subs"] = subs

sel = []
with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    r = ApiListSubmissionsRequest(); r.competition_name = PROBE; r.page_size = 200
    r.group = SubmissionGroup.SUBMISSION_GROUP_SELECTED
    sel = [int(s.ref) for s in
           c.competitions.competition_api_client.list_submissions(r).submissions]
out["probe_selected"] = sel

# R4: the PRIVATE leaderboard. override_public is left unset, which returns the final board
# for a closed competition; R4h checks that it is not the public one.
def board(client, name, public):
    tokn, rank, rows, top = None, 0, {}, None
    for _ in range(400):
        r = ApiGetLeaderboardRequest(); r.competition_name = name; r.page_size = 200
        if public: r.override_public = True
        if tokn: r.page_token = tokn
        resp = client.competitions.competition_api_client.get_leaderboard(r)
        ss = resp.submissions
        if not ss: break
        for s in ss:
            rank += 1
            if top is None: top = str(s.score)
            if int(s.team_id) == TEAM:
                rows = dict(rank=rank, score=str(s.score), name=str(s.team_name))
        tokn = getattr(resp, "next_page_token", "") or None
        if rows or not tokn: break
    return dict(me=rows, top=top, scanned=rank)

with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
    out["probe_private"] = board(c, PROBE, public=False)
    out["probe_public"] = board(c, PROBE, public=True)

print("<<<JSON>>>" + json.dumps(out))
''' % (COMP, PROBE, PROBE_TEAM)


def probe() -> dict:
    p = subprocess.run([KAGGLE_PY, "-c", SNIPPET], capture_output=True, text=True, timeout=900)
    if "<<<JSON>>>" not in p.stdout:
        print(p.stdout[-3000:]); print(p.stderr[-3000:])
        raise SystemExit("probe produced no JSON -- refusing to report on a failed read")
    return json.loads(p.stdout.split("<<<JSON>>>", 1)[1])


LIMIT_RE = re.compile(r"You may select up to (\w+) \((\d+)\) Final Submissions", re.I)
DAILY_RE = re.compile(r"You may submit a maximum of \w+ \((\d+)\) Submissions per day", re.I)
DEFN_RE = re.compile(r"A .Final Submission. is[^.]*\.", re.I)


def main() -> int:
    d = probe()
    me, rules, peers = d["self"], d["rules"], d["peer_rules"]

    print("=" * 90)
    print("R1 -- THE COMPETITION OBJECT, READ LIVE")
    print("=" * 90)
    for k, v in me.items():
        print(f"    {k:24s} = {v!r}")
    print()
    check(me["max_daily_submissions"] == 10, "R1a  daily cap is 10", repr(me["max_daily_submissions"]))
    check(me["evaluation_metric"] == "Roc Auc Score", "R1b  metric is ROC AUC",
          repr(me["evaluation_metric"]))
    # The object exposes NO final-submission-limit field. That absence is why R2 is needed.
    check("final" not in " ".join(me.keys()).lower(),
          "R1c  the competition object exposes NO final-submission-limit field",
          "so the limit of 2 cannot come from here")

    print()
    print("=" * 90)
    print("R2 -- THE COMPETITION'S OWN RULES, VERBATIM")
    print("=" * 90)
    mlim, mday, mdef = LIMIT_RE.search(rules), DAILY_RE.search(rules), DEFN_RE.search(rules)
    print(f"    SUBMISSION LIMITS  : {mday.group(0) if mday else '<ABSENT>'}")
    print(f"                       : {mlim.group(0) if mlim else '<ABSENT>'}")
    print(f"    'Final Submission' : {mdef.group(0).strip() if mdef else '<ABSENT>'}")
    print()
    check(mday is not None and int(mday.group(1)) == 10,
          "R2a  the rules state the daily cap is ten (10)",
          mday.group(1) if mday else "absent")
    check(mlim is not None and int(mlim.group(2)) == 2,
          "R2b  the rules state the FINAL-SUBMISSION LIMIT is two (2)  [Q3]",
          mlim.group(2) if mlim else "absent")
    check(mdef is not None and "automatically selected by kaggle" in mdef.group(0).lower(),
          "R2c  the rules confirm automatic selection HAPPENS when the user does not select")
    # Q1 as registered: the rules name the PUBLIC leaderboard as the criterion. They do not.
    stated = mdef is not None and "public" in mdef.group(0).lower()
    check(not stated,
          "R2d  [Q1 FALSIFIED] the rules NEVER state the CRITERION -- 'public' does not appear "
          "in the definition clause",
          "the criterion is undocumented, so R4 is required")
    check("most recent" not in rules.lower(),
          "R2e  the rules never say 'most recent' either", "no documentary support for recency")

    print()
    print("=" * 90)
    print("R3 -- DISCRIMINATION CONTROL: is the limit competition-specific or a constant?")
    print("=" * 90)
    lims = {}
    for name, txt in peers.items():
        m = LIMIT_RE.search(txt or "")
        dm = DAILY_RE.search(txt or "")
        lims[name] = (m.group(2) if m else None, dm.group(1) if dm else None)
        print(f"    {name:38s} final={lims[name][0]}  daily={lims[name][1]}")
    seen_daily = {v[1] for v in lims.values() if v[1]} | {"10"}
    check(len(seen_daily) > 1,
          "R3a  the DAILY cap varies across competitions, so the clause is parsed per-competition "
          "and is not a constant string", f"values seen: {sorted(seen_daily)}")
    # Honest reading of the other half: the FINAL limit reads 2 everywhere sampled. That does not
    # weaken R2b -- the clause is per-competition (R3a proves the parse is live) -- but 2 is a
    # Kaggle-wide default, not something S6E8 chose, and the run should say so.
    check(all(v[0] == "2" for v in lims.values() if v[0]),
          "R3b  the FINAL limit reads 2 on every peer sampled: it is a Kaggle-wide default, "
          "not an S6E8 choice", str({k: v[0] for k, v in lims.items()}))
    # Q4 as registered: is the no-selection fallback CLAUSE boilerplate, or specific to S6E8?
    peer_defn = {n: bool(DEFN_RE.search(t or "")) and
                 "automatically selected by kaggle" in (DEFN_RE.search(t or "").group(0).lower()
                                                        if DEFN_RE.search(t or "") else "")
                 for n, t in peers.items()}
    print(f"    fallback clause present in peers: {peer_defn}")
    check(all(peer_defn.values()),
          "R3c  [Q4 HELD] the no-selection fallback clause is Kaggle-wide boilerplate, present "
          "in every peer sampled", str(peer_defn))

    print()
    print("=" * 90)
    print("R4 -- THE EMPIRICAL TEST: which rule reproduces a REALISED private placement?")
    print("=" * 90)
    subs = [s for s in d["probe_subs"] if s["pub"] and s["prv"]]
    for s in subs:
        s["pubf"], s["prvf"] = float(s["pub"]), float(s["prv"])
    subs.sort(key=lambda s: s["date"])
    print(f"    {PROBE}, closed 2026-08-05, metric MSE (LOWER IS BETTER)")
    for s in subs:
        print(f"      {s['ref']}  {s['date'][:19]}  pub={s['pubf']:.3f}  prv={s['prvf']:.3f}")
    print(f"    manually SELECTED submissions reported by the API: {d['probe_selected']}")
    print()
    check(len(subs) == 8, "R4a  8 scored submissions with BOTH score columns", str(len(subs)))
    check(d["probe_selected"] == [],
          "R4b  nothing was manually selected there either", str(d["probe_selected"]))

    priv, pub = d["probe_private"], d["probe_public"]
    print(f"    PRIVATE board: top={priv['top']}  me={priv['me']}")
    print(f"    PUBLIC  board: top={pub['top']}   me={pub['me']}")
    print()
    check(priv["top"] != pub["top"],
          "R4c  the private board is NOT the public board", f"tops {priv['top']} vs {pub['top']}")
    realised = float(priv["me"]["score"])
    check(abs(realised - 9.606) < 1e-9,
          "R4d  realised FINAL private score read live", f"{realised:.3f} at rank {priv['me']['rank']}")

    # The three candidate rules. MSE: "best" = LOWEST.
    by_pub = sorted(subs, key=lambda s: s["pubf"])[:2]
    by_recent = sorted(subs, key=lambda s: s["date"], reverse=True)[:2]
    best_priv = min(s["prvf"] for s in subs)
    pred_pub = min(s["prvf"] for s in by_pub)
    pred_rec = min(s["prvf"] for s in by_recent)
    print(f"    rule 'best 2 by PUBLIC'  -> {[s['ref'] for s in by_pub]}    predicts {pred_pub:.3f}")
    print(f"    rule 'most recent 2'     -> {[s['ref'] for s in by_recent]}    predicts {pred_rec:.3f}")
    print(f"    rule 'best 2 by PRIVATE' -> (an oracle)                     predicts {best_priv:.3f}")
    print()
    check(abs(pred_pub - realised) < 1e-9,
          "R4e  *** 'best 2 by PUBLIC score' REPRODUCES the realised score ***",
          f"{pred_pub:.3f} == {realised:.3f}")
    check(abs(pred_rec - realised) > 1e-9,
          "R4f  'most recent 2' is REFUTED", f"predicts {pred_rec:.3f} != {realised:.3f}")
    check(abs(best_priv - realised) > 1e-9,
          "R4g  'best by PRIVATE' is REFUTED", f"predicts {best_priv:.3f} != {realised:.3f}")

    # Uniqueness: how many of the C(8,2) pairs reproduce the realised score at all?
    pairs = list(itertools.combinations(subs, 2))
    hits = [p for p in pairs if abs(min(p[0]["prvf"], p[1]["prvf"]) - realised) < 1e-9]
    print(f"    of the {len(pairs)} possible pairs, {len(hits)} reproduce {realised:.3f} "
          f"-> P(match by chance) = {len(hits)}/{len(pairs)} = {len(hits)/len(pairs):.3f}")
    check(len(hits) == 1,
          "R4h  the realised score identifies a UNIQUE pair, so this is a real test",
          f"{len(hits)}/{len(pairs)} pairs, chance = {len(hits)/len(pairs):.1%}")
    check({s["ref"] for s in by_pub} == {hits[0][0]["ref"], hits[0][1]["ref"]} if hits else False,
          "R4i  and the unique pair IS the best-2-by-public pair",
          str(sorted(s["ref"] for s in by_pub)))

    print()
    print("=" * 90)
    print("R5 -- WHAT THAT RULE COST THE ACCOUNT THERE (the Rogii failure, from the artefact)")
    print("=" * 90)
    worst = max(s["prvf"] for s in subs)
    print(f"    best private available   {best_priv:.3f}   ({[s['ref'] for s in subs if s['prvf']==best_priv]})")
    print(f"    realised (auto-selected) {realised:.3f}")
    print(f"    worst private available  {worst:.3f}")
    print(f"    cost of auto-selection over the oracle: +{realised - best_priv:.3f} MSE")
    check(abs(realised - worst) < 1e-9,
          "R5a  auto-selection landed on the WORST private score of all 8",
          f"{realised:.3f} == max private {worst:.3f}")

    print()
    print("=" * 90)
    print("R6 -- DOES A CLOSED COMPETITION EXPOSE WHAT w135b_grade.py WILL NEED TONIGHT?")
    print("=" * 90)
    print("    w135b decides the private board is out from two signals: the board score moving")
    print("    off its frozen public value, and ANY submission carrying a privateScore. Both are")
    print("    assumptions about post-close Kaggle. rogii is a closed competition, so it can")
    print("    say whether they hold.")
    n_priv = sum(1 for s in subs if s["prv"])
    print(f"    submissions carrying a privateScore: {n_priv} of {len(subs)}")
    print(f"    board score public {pub['me']['score']} -> private {priv['me']['score']}; "
          f"rank {pub['me']['rank']} -> {priv['me']['rank']}")
    check(n_priv == len(subs),
          "R6a  after a close, EVERY submission carries a privateScore "
          "(w135b's n_priv signal fires)", f"{n_priv}/{len(subs)}")
    check(pub["me"]["score"] != priv["me"]["score"],
          "R6b  after a close, the board score moves off its public value "
          "(w135b's score signal fires)", f"{pub['me']['score']} -> {priv['me']['score']}")

    print()
    print("=" * 90)
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for f in FAILURES:
            print("   " + f)
        return 1
    print(f"FAILURES 0 -- all {N_CHECKS} checks passed")
    print()
    print("VERDICT: the baseline arm is REAL and now has a source. The rules do NOT state the")
    print("criterion (Q1 falsified); the criterion is recovered empirically from a realised")
    print("placement on this same account, where 'best 2 by PUBLIC' is the unique rule of the")
    print("three that reproduces it, at a 1-in-28 chance of matching by accident.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
