"""w141a — can the post-close grader authenticate at the moment it runs?

w140 proved w135b_grade.py can parse a board and grade four predictions. It never asked
whether the script can AUTHENTICATE. Every workspace script, the grader included, reads
credentials.json["access_token"] as a raw string and hands it to KaggleClient(api_token=...).
That path has no refresh. The grader fires once, after the close, and a dead token makes it
exit 1 having graded nothing -- the one failure no later run can catch.

Measures, does not reason. Every refresh test runs against a COPY of the credential in a
temporary HOME; the live ~/.kaggle/credentials.json is never written by this script.

    /home/nixos/.local/share/uv/tools/kaggle/bin/python experiments/w141a_authaudit.py
"""
import datetime as dt
import inspect, json, os, shutil, sys, tempfile

from kagglesdk import KaggleClient, KaggleCredentials
from kagglesdk.kaggle_env import KaggleEnv
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
from kagglesdk.competitions.types.competition_enums import SubmissionGroup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.expanduser("~/.kaggle/credentials.json")
COMP = "playground-series-s6e8"
CLOSE = dt.datetime(2026, 8, 31, 23, 59, tzinfo=dt.timezone.utc)   # sourced live from the
                                                                   # competition object, w141
FAIL = []


def check(tag, ok, msg):
    print(f"{tag:<12} {'OK  ' if ok else 'FAIL'}  {msg}")
    if not ok:
        FAIL.append(tag)


def raw_token():
    return json.load(open(LIVE))["access_token"]


def one_call(token):
    """The grader's exact construction. Returns (ok, detail)."""
    try:
        with KaggleClient(env=KaggleEnv.PROD, api_token=token) as c:
            r = ApiListSubmissionsRequest()
            r.competition_name = COMP
            r.group = SubmissionGroup.SUBMISSION_GROUP_SUCCESSFUL
            r.page_size = 1
            resp = c.competitions.competition_api_client.list_submissions(r)
        return True, f"{len(resp.submissions)} row(s)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


print("=" * 78)
print("w141a — grader authentication audit")
print("UTC now :", dt.datetime.now(dt.timezone.utc).isoformat())
print("=" * 78)

# ---- T1  the raw-string path cannot refresh --------------------------------------------
src = inspect.getsource(KaggleHttpClient)
# `get_access_token_from_env` also matches a bare "get_access_token" grep and is NOT a refresh --
# it reads an env var. Strip it first, or the gate fails on its own looseness (it did, first run).
stripped = src.replace("get_access_token_from_env", "")
hits = [w for w in ("KaggleCredentials", "refresh_access_token", "get_access_token", ".save(") if w in stripped]
check("T1", not hits, f"KaggleHttpClient (the api_token= path) references no refresh symbol; found {hits}")

from kagglesdk import get_access_token_from_env as _env_tok
env_src = inspect.getsource(_env_tok)
check("T1b", "environ" in env_src and "refresh" not in env_src,
      "the one token helper the api_token= path CAN reach is env-var-only, not a refresh")

# ---- T1-ctl  the CLI path does, so T1's absence is a fact and not a bad grep ------------
from kaggle.api import kaggle_api_extended as kae
cli = inspect.getsource(kae.KaggleApi._authenticate_with_oauth_creds)
ok = "KaggleCredentials.load" in cli and "get_access_token" in cli
check("T1-ctl", ok, "the CLI path DOES call KaggleCredentials.load().get_access_token() -> the grep discriminates")

# ---- T2  a dead token through the grader's construction, and what it looks like ---------
bad_ok, bad_detail = one_call("CfDJ8" + "x" * 140)
check("T2", not bad_ok, f"invalid token through the grader's construction FAILS -> {bad_detail}")

# ---- T2-ctl  the same construction with the live token succeeds -------------------------
good_ok, good_detail = one_call(raw_token())
check("T2-ctl", good_ok, f"live token, same construction, succeeds -> {good_detail}  (T2 blames the token, not the code)")

# ---- T3  access_token_has_expired() probed ACROSS the boundary, both sides --------------
now = dt.datetime.now(dt.timezone.utc)
probes = [(-40, True), (-10, False), (+10, False), (+600, False)]
rows = []
for mins, want in probes:
    c = KaggleCredentials(client=None, refresh_token="x", access_token="y",
                          access_token_expiration=now + dt.timedelta(minutes=mins))
    got = c.access_token_has_expired()
    rows.append((mins, got, want, got == want))
for mins, got, want, ok in rows:
    print(f"             expiry now{mins:+5d}m -> has_expired()={got!s:<5} expected {want}")
check("T3", all(r[3] for r in rows),
      "expiry is declared 30 MINUTES LATE (`expiration < now - 30min`): a token 10 min dead reports healthy")

# ---- T4  the remedy refreshes, on a COPY in a temp HOME --------------------------------
old_home = os.environ.get("HOME")
tmp = tempfile.mkdtemp(prefix="w141a_")
try:
    os.makedirs(os.path.join(tmp, ".kaggle"))
    copy = os.path.join(tmp, ".kaggle", "credentials.json")
    shutil.copy(LIVE, copy)
    d = json.load(open(copy))
    d["access_token_expiration"] = (now - dt.timedelta(hours=2)).isoformat()   # stale on purpose
    json.dump(d, open(copy, "w"))
    before = json.load(open(copy))["access_token"]
    os.environ["HOME"] = tmp
    with KaggleClient(env=KaggleEnv.PROD) as k:
        creds = KaggleCredentials.load(client=k)
        tok = creds.get_access_token()
    after = json.load(open(copy))
    os.environ["HOME"] = old_home
    new_exp = dt.datetime.fromisoformat(after["access_token_expiration"])
    moved = after["access_token"] != before
    check("T4", moved and new_exp > now,
          f"stale copy refreshed: token changed={moved}, new expiry {new_exp.isoformat()} "
          f"(+{(new_exp - now).total_seconds()/3600:.1f}h)")
    fresh_ok, fresh_detail = one_call(tok)
    check("T4-new", fresh_ok, f"the refreshed token works -> {fresh_detail}")
    # the live token must still work after a refresh was issued -- issuing must not revoke
    still_ok, still_detail = one_call(raw_token())
    check("T4-live", still_ok, f"the LIVE token still works after a refresh was issued -> {still_detail}")
    print(f"             temp HOME written: {copy}")
    print(f"             live file untouched, mtime {dt.datetime.utcfromtimestamp(os.path.getmtime(LIVE)).isoformat()}Z")
finally:
    if old_home:
        os.environ["HOME"] = old_home
    shutil.rmtree(tmp, ignore_errors=True)

# ---- T5  the clock the grader actually runs against -------------------------------------
exp = dt.datetime.fromisoformat(json.load(open(LIVE))["access_token_expiration"])
margin_h = (exp - CLOSE).total_seconds() / 3600
print()
print(f"             close            {CLOSE.isoformat()}")
print(f"             live token dies  {exp.isoformat()}")
print(f"             margin after close: {margin_h:+.2f}h   (dead zone {margin_h:+.2f}h .. {margin_h + 0.5:+.2f}h "
      f"where even the CLI path hands back a dead token, per T3)")
check("T5", exp > CLOSE,
      f"the live token outlives the close by {margin_h:.1f}h -- a grading run LATER than that gets T2's failure")

# ---- T6  the grader inherits the raw-token path ------------------------------------------
g = open(os.path.join(ROOT, "experiments", "w135b_grade.py")).read()
raw_sites = g.count('["access_token"]')
check("T6", raw_sites > 0,
      f"w135b_grade.py reads the raw access_token at {raw_sites} site(s) -> it inherits T1/T2")

print()
print("=" * 78)
print(f"FAILURES: {len(FAIL)}" + (f"  {FAIL}" if FAIL else ""))
print("=" * 78)
sys.exit(1 if FAIL else 0)
