"""Paginated reads of the Kaggle LEADERBOARD, public or private. One place, with a cache.

WHY THIS EXISTS (w142, 2026-09-01). `w135b_grade.py` fetched the board by shelling out to
`kaggle competitions leaderboard -d`, which is `ApiDownloadLeaderboardRequest` -- a request
type whose ONLY field is `competition_name`. That endpoint serves the PUBLIC board and has no
way to ask for anything else, so after the close the grader kept loading public scores, saw its
own board-identity gate reject them, and refused to grade. The gate was right; the fetch was
wrong.

The board switch lives on the OTHER endpoint. `ApiGetLeaderboardRequest` has `override_public`:

    override_public = False (or unset)  -> the FINAL board (private, once the comp has closed)
    override_public = True              -> the public board, still, after the close

Measured on 2026-09-01: the two disagree. Private leader 0.97176, public leader 0.97207, and
2nd/3rd place swap between them. That disagreement is what proves the flag does something.

⛔ DO NOT fetch a board with `kaggle competitions leaderboard -d` and then test whether it is
   private. It cannot be. Use this module.

Rate limits are real: four full 18-page reads in a row earned a 429. Every page retries with
backoff, and boards are cached to `experiments/board_cache/` because a closed competition's
board does not change.
"""
from __future__ import annotations

import json
import os
import subprocess

COMP = "playground-series-s6e8"
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
PAGE = 200
MAX_PAGES = 60
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board_cache")

_SNIPPET = r"""
import json, os, time, sys
from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest

tok = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["access_token"]
rows, page_token, pages = [], None, 0
while True:
    last = None
    for attempt in range(6):
        try:
            with KaggleClient(env=KaggleEnv.PROD, api_token=tok) as c:
                r = ApiGetLeaderboardRequest()
                r.competition_name = COMP_NAME
                r.page_size = PAGE_SIZE
                r.override_public = OVERRIDE
                if page_token:
                    r.page_token = page_token
                resp = c.competitions.competition_api_client.get_leaderboard(r)
            break
        except Exception as e:              # 429 and transient 5xx
            last = e
            time.sleep(4 * (attempt + 1))
    else:
        print(json.dumps({"error": "%s: %s" % (type(last).__name__, last)}))
        sys.exit(3)
    rows += [{"teamId": int(s.team_id), "teamName": s.team_name, "score": s.score,
              "date": str(s.submission_date)} for s in resp.submissions]
    page_token = resp.next_page_token
    pages += 1
    time.sleep(0.35)
    if not page_token or pages >= MAX_P:
        break
print(json.dumps({"rows": rows, "pages": pages, "truncated": bool(page_token)}))
"""


def board(private: bool, comp: str = COMP, use_cache: bool = True) -> list[dict]:
    """Every row of the board, in rank order, paginated to exhaustion.

    Raises SystemExit if the read could not be completed -- never returns a partial board.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{comp}-{'private' if private else 'public'}.json")
    if use_cache and os.path.exists(path):
        return json.load(open(path))["rows"]

    src = (_SNIPPET.replace("COMP_NAME", repr(comp))
           .replace("PAGE_SIZE", str(PAGE))
           .replace("OVERRIDE", "False" if private else "True")
           .replace("MAX_P", str(MAX_PAGES)))
    p = subprocess.run([KAGGLE_PY, "-c", src], capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise SystemExit(f"leaderboard read failed (rc={p.returncode}): "
                         f"{p.stdout.strip()[:300]} {p.stderr.strip()[-300:]}")
    out = json.loads(p.stdout)
    if out.get("error"):
        raise SystemExit(f"leaderboard read failed: {out['error']}")
    if out["truncated"]:
        raise SystemExit(f"leaderboard read TRUNCATED after {out['pages']} pages "
                         f"({len(out['rows'])} rows) -- a page token was still outstanding.")
    json.dump(out, open(path, "w"))
    return out["rows"]


def rank_of(rows: list[dict], team_id: int) -> int | None:
    """1-based position in the board's own ordering. None if the team is absent."""
    for i, r in enumerate(rows):
        if r["teamId"] == team_id:
            return i + 1
    return None
