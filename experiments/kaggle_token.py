"""Keep ~/.kaggle/credentials.json holding a token that will still be alive in a moment.

WHY THIS EXISTS (w141). Two auth paths run in this workspace and both read the same file:
the `kaggle` CLI loads it through KaggleCredentials, and ten scripts read
credentials.json["access_token"] as a raw string and pass it to KaggleClient(api_token=...).
The raw-string path has NO refresh at all (w141a T1). The CLI path has one, but it fires
30 minutes late:

    access_token_has_expired() -> `expiration < now - 30min`

so for the first half hour after a token really dies the CLI believes it is healthy, skips the
refresh and hands a corpse to the server. w141b reproduced that with a genuinely dead 60s
token: identical token, expiry recorded 15s ago -> download FAILS and the file is not
rewritten; expiry recorded 31min ago -> refresh happens and the download SUCCEEDS.

So do not ask the SDK whether the token has expired. Compare the expiry yourself, against a
margin that is ahead of now rather than behind it, and refresh through
`refresh_access_token()` (which regenerates and saves) rather than `get_access_token()`.

Refreshing does not revoke tokens already issued -- measured, w141a T4-live.

    python kaggle_token.py [--margin MINUTES]     # must be an interpreter holding kagglesdk

Prints one JSON line: {"action": "kept"|"refreshed", "expiration": ..., "seconds_left": ...}.
It never prints the token.
"""
import argparse
import datetime as dt
import json
import os

DEFAULT_MARGIN_MIN = 90
CREDS = os.path.expanduser("~/.kaggle/credentials.json")


def ensure_fresh(margin_minutes: int = DEFAULT_MARGIN_MIN) -> dict:
    """Refresh the on-disk access token if it dies within `margin_minutes`.

    Returns a dict describing what happened. Raises only if the file is unusable; a failed
    refresh is reported in the return value so a caller holding a still-valid token can carry
    on rather than being stopped by a precaution.
    """
    now = dt.datetime.now(dt.timezone.utc)
    with open(CREDS) as f:
        data = json.load(f)

    raw = data.get("access_token_expiration")
    try:
        exp = dt.datetime.fromisoformat(raw) if raw else None
    except ValueError:
        exp = None
    if exp is not None and exp.tzinfo is None:
        exp = exp.replace(tzinfo=dt.timezone.utc)

    out = {"before": exp.isoformat() if exp else None,
           "margin_minutes": margin_minutes,
           "error": None}

    if exp is not None and exp > now + dt.timedelta(minutes=margin_minutes):
        out.update(action="kept", expiration=exp.isoformat(),
                   seconds_left=int((exp - now).total_seconds()))
        return out

    try:
        from kagglesdk import KaggleClient, KaggleCredentials
        from kagglesdk.kaggle_env import KaggleEnv
        with KaggleClient(env=KaggleEnv.PROD) as client:
            creds = KaggleCredentials.load(client=client)
            if creds is None:
                raise RuntimeError(f"no usable credentials in {CREDS} (missing refresh_token?)")
            creds.refresh_access_token()      # regenerates AND saves; not get_access_token()
    except Exception as e:
        out.update(action="failed", expiration=out["before"], seconds_left=None,
                   error=f"{type(e).__name__}: {e}")
        return out

    with open(CREDS) as f:
        new_exp = json.load(f).get("access_token_expiration")
    new = dt.datetime.fromisoformat(new_exp)
    out.update(action="refreshed", expiration=new.isoformat(),
               seconds_left=int((new - now).total_seconds()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN_MIN,
                    help="refresh if the token dies within this many minutes")
    print(json.dumps(ensure_fresh(ap.parse_args().margin)))
