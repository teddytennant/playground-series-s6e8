"""w152a -- w151 read a DELISTING off a call that filters closed competitions out.

w151 opened with a hand-built ApiListCompetitionsRequest carrying only `search`, got []
for playground-series-s6e8 and a row for playground-series-s6e9, and concluded S6E8 had
dropped out of Kaggle's listing. Its falsifier was "search a competition known to be live
in the same process; a row back means the API is fine" -- which has no power here, because
LIVE vs CLOSED is the exact dimension the broken call discriminates on.

Measured: the bare request behaves as an open-competitions-only filter. The wrapper
KaggleApi.competitions_list (which sets group/category/sort_by/paging, per w145) returns
closed competitions too, S6E8 among them.

Exits 0 when every assertion holds, 1 otherwise, and prints FAILURES: n as its last line.
"""
import datetime
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiListCompetitionsRequest

SELF = "playground-series-s6e8"

# Slugs must be EXACT: a wrapper search for "arc-prize-2026" returns arc-prize-2026-arc-agi-3
# first, so taking row[0] classifies a different competition than the one asked for.
PROBES = [
    "playground-series-s5e3",
    "playground-series-s6e2",
    SELF,
    "playground-series-s6e9",
    "arc-prize-2026-arc-agi-2",
]


def refs(resp):
    return [c.ref.rsplit("/", 1)[-1] for c in (getattr(resp, "competitions", None) or [])]


def bare(api, slug):
    """The w151 call: every field but `search` left at its proto zero-value."""
    req = ApiListCompetitionsRequest()
    req.search = slug
    with api.build_kaggle_client() as kc:
        return refs(kc.competitions.competition_api_client.list_competitions(req))


def wrapper(api, slug):
    return api.competitions_list(search=slug)


def main():
    api = KaggleApi()
    api.authenticate()
    now = datetime.datetime.now()
    fails = 0

    print("C1 the bare request drops CLOSED competitions and keeps OPEN ones")
    rows = []
    for slug in PROBES:
        resp = wrapper(api, slug)
        comps = getattr(resp, "competitions", None) or []
        hit = [c for c in comps if c.ref.rsplit("/", 1)[-1] == slug]
        if not hit:
            print(f"  {slug:24s} NOT IN WRAPPER RESULT -- cannot classify")
            fails += 1
            continue
        deadline = hit[0].deadline
        is_open = deadline > now
        in_bare = slug in bare(api, slug)
        ok = in_bare == is_open
        fails += not ok
        rows.append((slug, deadline, is_open, in_bare, ok))
        print(f"  {slug:24s} deadline={str(deadline):20s} open={str(is_open):5s} "
              f"bare={str(in_bare):5s} {'OK' if ok else 'MISMATCH'}")
    if rows and all(r[4] for r in rows):
        print(f"  open/closed predicts bare membership on {len(rows)}/{len(rows)} probes. OK")

    print("C2 the wrapper still returns S6E8, so it is LISTED, not delisted")
    got = refs(wrapper(api, SELF))
    ok = SELF in got
    fails += not ok
    print(f"  wrapper(search={SELF!r}) -> {got}  {'OK' if ok else 'FAIL'}")

    print("C3 the bare request returns nothing for S6E8 -- w151's reading reproduces")
    got_bare = bare(api, SELF)
    ok = got_bare == []
    fails += not ok
    print(f"  bare(search={SELF!r}) -> {got_bare}  {'OK (w151 reproduces)' if ok else 'CHANGED'}")

    print("C4 w151's falsifier passes anyway, so it never had power")
    live = "playground-series-s6e9"
    ok = live in bare(api, live)
    fails += not ok
    print(f"  bare(search={live!r}) -> row present: {ok}  "
          f"-> 'the API is fine' is TRUE and the call is STILL broken for S6E8")

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
