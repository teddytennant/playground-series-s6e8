"""w141b — how wide is the window in which the grader cannot authenticate?

w141a established that expiry is declared 30 minutes LATE
(`access_token_has_expired` is `expiration < now - 30min`). That predicts a window: for the
first 30 minutes after a token really dies, the CLI still believes it is good, does NOT
refresh, and hands a corpse to the server. Before that window the token works; after it, the
CLI refreshes and everything works again. So the grader's failure is a 30-minute hole, not a
cliff -- and that is a different instruction to leave behind.

This does not reason about the hole, it reproduces it with a GENUINELY DEAD token: the server
honours `expiration_duration`, so a 60-second token can be minted and allowed to die.

THE CONTROL IS THE POINT. Both runs use the SAME dead token. The only thing that differs is
the expiration this script wrote next to it. If the outcome flips, the 30-minute rule is what
causes the hole -- not the network, not the competition, not the token.

Runs entirely in a temp HOME + KAGGLE_CONFIG_DIR. The live credential is never written.

    /home/nixos/.local/share/uv/tools/kaggle/bin/python experiments/w141b_deadzone.py
"""
import datetime as dt
import json, os, shutil, subprocess, sys, tempfile, time

from kagglesdk import KaggleClient, KaggleCredentials
from kagglesdk.kaggle_env import KaggleEnv

COMP = "playground-series-s6e8"
KCLI = "/home/nixos/.local/share/uv/tools/kaggle/bin/kaggle"
LIVE = os.path.expanduser("~/.kaggle/credentials.json")
LIVE_MTIME = os.path.getmtime(LIVE)
TTL = 60
FAIL = []


def check(tag, ok, msg):
    print(f"{tag:<10} {'OK  ' if ok else 'FAIL'}  {msg}")
    if not ok:
        FAIL.append(tag)


def write_creds(path, token, expiration):
    base = json.load(open(LIVE))
    json.dump({"refresh_token": base["refresh_token"], "access_token": token,
               "access_token_expiration": expiration.isoformat(),
               "username": base["username"], "scopes": base["scopes"]},
              open(path, "w"), indent=2)


def run_cli(home, outdir):
    env = dict(os.environ, HOME=home, KAGGLE_CONFIG_DIR=os.path.join(home, ".kaggle"))
    p = subprocess.run([KCLI, "competitions", "leaderboard", "-c", COMP, "-d", "-p", outdir],
                       capture_output=True, text=True, timeout=300, env=env)
    got = [f for f in os.listdir(outdir)] if os.path.isdir(outdir) else []
    return p.returncode, (p.stdout + p.stderr).strip()[-200:], got


print("=" * 78)
print("w141b — the authentication dead zone, reproduced with a real dead token")
print("UTC now :", dt.datetime.now(dt.timezone.utc).isoformat())
print("=" * 78)

with KaggleClient(env=KaggleEnv.PROD) as k:
    resp = KaggleCredentials.load(client=k).generate_access_token(
        expiration_duration=dt.timedelta(seconds=TTL))
dead_token, granted = resp.token, resp.expires_in
check("S0", granted == TTL, f"minted a {granted}s access token (server honoured the duration)")

home = tempfile.mkdtemp(prefix="w141b_")
try:
    os.makedirs(os.path.join(home, ".kaggle"))
    creds = os.path.join(home, ".kaggle", "credentials.json")
    born = dt.datetime.now(dt.timezone.utc)

    # ---- A: token alive. Establishes the token and the command work at all. -------------
    write_creds(creds, dead_token, born + dt.timedelta(seconds=TTL))
    rc, out, got = run_cli(home, os.path.join(home, "a"))
    check("A", rc == 0 and got, f"while ALIVE the same token downloads the board -> rc={rc} files={got}")

    wait = TTL + 15 - (dt.datetime.now(dt.timezone.utc) - born).total_seconds()
    print(f"           waiting {max(0, wait):.0f}s for the token to actually die...")
    if wait > 0:
        time.sleep(wait)

    # ---- B: dead token, expiry recorded ~15s ago -> INSIDE the 30-minute hole -----------
    tok_before = json.load(open(creds))["access_token"]
    rc, out, got = run_cli(home, os.path.join(home, "b"))
    tok_after = json.load(open(creds))["access_token"]
    check("B", rc != 0 or not got,
          f"DEAD token, expiry 15s ago -> the CLI does NOT refresh and the download FAILS: rc={rc} files={got}")
    check("B-nr", tok_after == tok_before, "and the credential file was NOT rewritten (no refresh was attempted)")
    print(f"           CLI said: {out[:160]!r}")

    # ---- C: the CONTROL. Same dead token, expiry backdated past 30 min. -----------------
    write_creds(creds, dead_token, dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=31))
    tok_before = json.load(open(creds))["access_token"]
    rc, out, got = run_cli(home, os.path.join(home, "c"))
    after = json.load(open(creds))
    check("C", rc == 0 and got,
          f"SAME dead token, expiry backdated 31min -> the CLI DOES refresh and the download SUCCEEDS: rc={rc} files={got}")
    check("C-r", after["access_token"] != tok_before,
          f"and the credential file WAS rewritten, new expiry {after['access_token_expiration']}")
    print("           B and C differ only in the expiration string. The outcome flipped, so")
    print("           the 30-minute rule is the cause of the hole.")
finally:
    shutil.rmtree(home, ignore_errors=True)

check("LIVE", os.path.getmtime(LIVE) == LIVE_MTIME,
      "the live ~/.kaggle/credentials.json was not modified by this test")

print()
print("=" * 78)
print(f"FAILURES: {len(FAIL)}" + (f"  {FAIL}" if FAIL else ""))
print("=" * 78)
sys.exit(1 if FAIL else 0)
