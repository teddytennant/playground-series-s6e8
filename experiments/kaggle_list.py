"""Paginated reads of the Kaggle submission list. One place, because five files got it wrong.

WHY THIS EXISTS (w140, 2026-08-31). The submissions endpoint caps a page at **200 rows** and
returns a `next_page_token`; the Kaggle CLI never prints that token. So the workspace's
standard defence --

    "--page-size", str(PAGE)   with   if len(rows) >= PAGE: raise      # PAGE = 500

-- is unbuyable: the read comes back at 200, `200 >= 500` is False, and the guard reports NOT
TRUNCATED while row 201 sits behind an unread token. Raising PAGE past 200 does not widen the
read, it DISABLES THE DETECTOR. Measured in `w140c_pagetruth.py` (T1/T2/T5); the same row
(`stack_pub74_logit`) was re-hidden by that idiom at w17, w138, w139 and w140.

⛔ DO NOT "fix" a truncation failure here by raising a page size. There is no page size that
   works. Follow the token, which is what this module does.

The rows come back with the CLI's own column names (`ref`, `fileName`, `date`, `description`,
`status`, `publicScore`, `privateScore`) so call sites that used to parse the CLI's CSV can
swap to this with no other change. Scores are strings, empty when absent, exactly as the CSV
had them.
"""
from __future__ import annotations

import json
import os
import subprocess

COMP = "playground-series-s6e8"
# The interpreter that holds kagglesdk. The workspace .venv does NOT -- importing it there is
# a ModuleNotFoundError. check_selection.py resolves it the same way.
KAGGLE_PY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
PAGE_CAP = 200          # measured, w140c T1: 201 -> 200, 500 -> 200, 1000 -> 200
MAX_PAGES = 25

_SNIPPET = r"""
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
        r.competition_name = COMP_NAME
        r.group = GROUP
        r.page_size = PAGE_CAP
        if page_token:
            r.page_token = page_token
        resp = c.competitions.competition_api_client.list_submissions(r)
    rows += [{"ref": str(s.ref), "fileName": s.file_name,
              # the CLI renders this column as a string; match it exactly so call
              # sites that used to parse the CSV can swap over untouched.
              "date": s.date.isoformat(sep=" ") if hasattr(s.date, "isoformat")
                      else str(s.date),
              "description": s.description, "status": str(s.status),
              "publicScore": s.public_score, "privateScore": s.private_score}
             for s in resp.submissions]
    page_token = resp.next_page_token
    pages += 1
    if not page_token or pages >= MAX_PAGES:
        break
print(json.dumps({"rows": rows, "pages": pages, "truncated": bool(page_token)}))
"""


def _script(comp: str, group: str) -> str:
    g = {"successful": "SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL",
         "selected": "SubmissionGroup.SUBMISSION_GROUP_SELECTED",
         "all": "SubmissionGroup.SUBMISSION_GROUP_ALL"}[group]
    return (_SNIPPET.replace("COMP_NAME", repr(comp)).replace("GROUP", g)
            .replace("PAGE_CAP", str(PAGE_CAP)).replace("MAX_PAGES", str(MAX_PAGES)))


def submissions(comp: str = COMP, group: str = "successful") -> list[dict]:
    """Every submission in `group`, paginated to exhaustion.

    Raises SystemExit if the read could not be completed -- never returns a partial list
    quietly, which is the entire failure mode this module exists for.
    """
    p = subprocess.run([KAGGLE_PY, "-c", _script(comp, group)],
                       capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        raise SystemExit(f"paginated submission read failed:\n{p.stderr.strip()[-800:]}")
    d = json.loads(p.stdout)
    if d["truncated"]:
        raise SystemExit(
            f"submission list still had a next_page_token after {MAX_PAGES} pages "
            f"({len(d['rows'])} rows). Raise MAX_PAGES in kaggle_list.py -- NOT a page size.")
    refs = [r["ref"] for r in d["rows"]]
    if len(set(refs)) != len(refs):
        raise SystemExit("paginated read returned duplicate refs; the pages overlap")
    return d["rows"]


def total(comp: str = COMP) -> int:
    """Row count of the full paginated read. An outside witness for any capped read."""
    return len(submissions(comp))
