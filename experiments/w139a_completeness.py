"""w139 (2026-08-31) -- an EXTERNAL completeness oracle for the submission list.

Twice in this workspace the submission list has been read short and nobody noticed:

    w17  (08-17)  asked for 50, the account held 50, "fixed" by asking for 200
    w138 (08-31)  asked for 200, the account held 201, fixed by following next_page_token

Both times the bug was invisible from inside the read: a full page and a truncated page
look identical when you only have the page. The tell has to come from OUTSIDE the list.

`get_submission_limits` carries `num_total`, maintained server-side, free, and never
paginated. Comparing it to the length of the list you just assembled is a complete
check on the read -- it would have fired on 08-17 and again on 08-31, on any day, at a
cost of one extra RPC.

    GATE A   paginated list length == num_total          (the invariant)
    GATE B   the single-page read that w138 replaced is DETECTED by gate A
             (a control: an instrument that cannot fail is not an instrument)
    GATE C   num_today / num_allowed_now agree with the day's rows in the list
"""
import json, os, sys, datetime

from kagglesdk import KaggleClient
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.competitions.types.competition_api_service import (
    ApiGetSubmissionLimitsRequest, ApiListSubmissionsRequest)
from kagglesdk.competitions.types.competition_enums import SubmissionGroup

COMP = "playground-series-s6e8"
TOK = json.load(open(os.path.expanduser("~/.kaggle/credentials.json")))["access_token"]


def limits():
    with KaggleClient(env=KaggleEnv.PROD, api_token=TOK) as c:
        r = ApiGetSubmissionLimitsRequest()
        r.competition_name = COMP
        return c.competitions.competition_api_client.get_submission_limits(r)


def listing(page_size=200, follow=True, max_pages=25):
    """Return (rows, n_pages, saw_token). follow=False reproduces the pre-w138 read."""
    rows, token, pages, saw = [], None, 0, False
    while True:
        with KaggleClient(env=KaggleEnv.PROD, api_token=TOK) as c:
            r = ApiListSubmissionsRequest()
            r.competition_name = COMP
            r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL
            r.page_size = page_size
            if token:
                r.page_token = token
            resp = c.competitions.competition_api_client.list_submissions(r)
        rows += [(s.ref, s.file_name, s.public_score, s.date) for s in resp.submissions]
        token = resp.next_page_token
        saw = saw or bool(token)
        pages += 1
        if not follow or not token or pages >= max_pages:
            break
    return rows, pages, saw


def main():
    fails = []
    lim = limits()
    print(f"UTC now            {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    print(f"num_total          {lim.num_total}")
    print(f"num_today          {lim.num_today}")
    print(f"num_allowed_now    {lim.num_allowed_now}")
    print(f"limited_by_total   {lim.limited_by_total}")

    # ---- GATE A: the invariant -------------------------------------------------
    rows, pages, saw = listing(follow=True)
    print(f"\nA  paginated read: {len(rows)} rows over {pages} page(s), "
          f"unique refs {len({r[0] for r in rows})}")
    if len(rows) != lim.num_total:
        fails.append(f"A: list has {len(rows)} rows, server says num_total={lim.num_total}")
    else:
        print(f"A  OK  list length == num_total == {lim.num_total}")

    # ---- GATE B: the control ---------------------------------------------------
    # An oracle that never disagrees with anything proves nothing. Reproduce the exact
    # read w138 replaced and require gate A's comparison to CATCH it.
    short, spages, ssaw = listing(follow=False)
    caught = len(short) != lim.num_total
    print(f"\nB  single-page read (the pre-w138 code): {len(short)} rows, "
          f"next_page_token present={ssaw}")
    if not caught:
        fails.append("B: the truncated read was NOT caught -- gate A is inert today")
    else:
        print(f"B  OK  CAUGHT: {len(short)} != {lim.num_total}. The invariant fires on the "
              f"real historical defect, it is not decoration.")
        missed = {r[0] for r in rows} - {r[0] for r in short}
        for ref in missed:
            row = next(r for r in rows if r[0] == ref)
            print(f"       hidden by truncation: ref={row[0]} file={row[1]} public={row[2]}")

    # ---- GATE C: the day's rows ------------------------------------------------
    today = datetime.datetime.now(datetime.timezone.utc).date()
    n_today = sum(1 for r in rows if getattr(r[3], "date", lambda: None)() == today)
    print(f"\nC  rows dated {today} in the list: {n_today}; server num_today={lim.num_today}")
    if n_today != lim.num_today:
        fails.append(f"C: {n_today} rows dated today, server says num_today={lim.num_today}")
    else:
        print(f"C  OK  {n_today} == {lim.num_today}")

    print("\n" + "=" * 74)
    if fails:
        for f in fails:
            print("FAIL " + f)
        print(f"FAILURES: {len(fails)}")
        return 1
    print("FAILURES: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
