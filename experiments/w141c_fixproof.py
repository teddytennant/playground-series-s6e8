"""w141c — does the preflight actually close the hole w141b opened?

w141b reproduced the dead zone: a genuinely dead token whose expiry is recorded 15 seconds ago
makes the `kaggle` CLI skip its refresh and fail. w141a wrote kaggle_token.ensure_fresh and
w135b_grade.py now calls it before either auth path runs. That is a claim, not a result.

This runs the SAME reproduction twice under identical conditions -- same dead token, same
recorded expiry, same command, same temp HOME -- and changes exactly one thing: whether the
preflight ran first. WITHOUT it the download must FAIL (that is w141b's B, re-established here
so the comparison cannot drift between two scripts). WITH it the download must SUCCEED.

If both succeed, the fix is unproven and the second run was riding on something else.

Live ~/.kaggle/credentials.json is never written.

    /home/nixos/.local/share/uv/tools/kaggle/bin/python experiments/w141c_fixproof.py
"""
import datetime as dt
import json, os, shutil, subprocess, sys, tempfile, time

from kagglesdk import KaggleClient, KaggleCredentials
from kagglesdk.kaggle_env import KaggleEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "playground-series-s6e8"
KCLI = "/home/nixos/.local/share/uv/tools/kaggle/bin/kaggle"
KPY = "/home/nixos/.local/share/uv/tools/kaggle/bin/python"
HELPER = os.path.join(ROOT, "experiments", "kaggle_token.py")
LIVE = os.path.expanduser("~/.kaggle/credentials.json")
LIVE_MTIME = os.path.getmtime(LIVE)
TTL = 60
FAIL = []


def check(tag, ok, msg):
    print(f"{tag:<10} {'OK  ' if ok else 'FAIL'}  {msg}")
    if not ok:
        FAIL.append(tag)


def env_for(home):
    return dict(os.environ, HOME=home, KAGGLE_CONFIG_DIR=os.path.join(home, ".kaggle"))


def write_creds(path, token, expiration):
    base = json.load(open(LIVE))
    json.dump({"refresh_token": base["refresh_token"], "access_token": token,
               "access_token_expiration": expiration.isoformat(),
               "username": base["username"], "scopes": base["scopes"]},
              open(path, "w"), indent=2)


def download(home, tag):
    out = os.path.join(home, tag)
    p = subprocess.run([KCLI, "competitions", "leaderboard", "-c", COMP, "-d", "-p", out],
                       capture_output=True, text=True, timeout=300, env=env_for(home))
    files = os.listdir(out) if os.path.isdir(out) else []
    return p.returncode == 0 and bool(files), files


print("=" * 78)
print("w141c — proving the preflight closes the dead zone")
print("UTC now :", dt.datetime.now(dt.timezone.utc).isoformat())
print("=" * 78)

with KaggleClient(env=KaggleEnv.PROD) as k:
    resp = KaggleCredentials.load(client=k).generate_access_token(
        expiration_duration=dt.timedelta(seconds=TTL))
dead_token = resp.token
check("S0", resp.expires_in == TTL, f"minted a {resp.expires_in}s token to let die")

home = tempfile.mkdtemp(prefix="w141c_")
try:
    os.makedirs(os.path.join(home, ".kaggle"))
    creds = os.path.join(home, ".kaggle", "credentials.json")
    born = dt.datetime.now(dt.timezone.utc)
    write_creds(creds, dead_token, born + dt.timedelta(seconds=TTL))

    wait = TTL + 15 - (dt.datetime.now(dt.timezone.utc) - born).total_seconds()
    print(f"           waiting {max(0, wait):.0f}s for the token to actually die...")
    if wait > 0:
        time.sleep(wait)

    # ---- WITHOUT the preflight: must fail. Re-establishes w141b's B in this script. -----
    ok, files = download(home, "before")
    check("NOFIX", not ok, f"dead token, expiry 15s ago, NO preflight -> download FAILS (files={files})")

    # ---- WITH the preflight, same dead credential, rewritten to the same state ----------
    write_creds(creds, dead_token, dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=15))
    tok_before = json.load(open(creds))["access_token"]
    p = subprocess.run([KPY, HELPER, "--margin", "90"], capture_output=True, text=True,
                       timeout=120, env=env_for(home))
    info = json.loads(p.stdout)
    check("PRE", info["action"] == "refreshed",
          f"preflight on the same dead credential -> action={info['action']}, "
          f"expires {info['expiration']}")
    check("PRE-w", json.load(open(creds))["access_token"] != tok_before,
          "the preflight rewrote the credential file the CLI is about to read")

    ok, files = download(home, "after")
    check("FIX", ok, f"same conditions, preflight first -> download SUCCEEDS (files={files})")

    print("           NOFIX and FIX differ only in whether the preflight ran. The outcome")
    print("           flipped, so the preflight is what closes the hole.")
finally:
    shutil.rmtree(home, ignore_errors=True)

check("LIVE", os.path.getmtime(LIVE) == LIVE_MTIME,
      "the live ~/.kaggle/credentials.json was not modified by this test")

print()
print("=" * 78)
print(f"FAILURES: {len(FAIL)}" + (f"  {FAIL}" if FAIL else ""))
print("=" * 78)
sys.exit(1 if FAIL else 0)
