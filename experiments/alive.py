r"""alive.py — POSITIVE liveness check for a long build, immune to the two ways
`pgrep -af <token>` lies in this harness.

WHY THIS EXISTS. RESEARCH's "check the build POSITIVELY" step has prescribed
`pgrep -af w96c_build` since w103. Typed at the agent's shell that command is wrong in
BOTH directions, and w105 measured both:

  FALSE PRESENCE   The agent runs every command as `bash -c '<the whole script text>'`, so the
                   invoking wrapper's argv CONTAINS the token being searched for. `pgrep -af`
                   matches that wrapper and exits 0. With the build dead, the prescribed
                   positive liveness check still prints a line and still says "alive".
  FALSE ABSENCE    `pgrep` matches with ERE, not BRE. `pgrep -af 'a\|b'` is a literal and
                   matches nothing real -- indistinguishable from the process being gone.
                   w105 typed exactly that and read a live build as dead.

THE FIX, and it generalises past pgrep: a liveness probe must never count its own ancestors,
and must assert the matched line LOOKS LIKE THE JOB rather than that some line matched.

    .venv/bin/python experiments/alive.py --unit w102a-build.service \
        --cmd-contains w96c_build_teprior_member.py

rc 0 = alive, 1 = dead, 2 = could not decide (say so; do not read it as either).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

SYSBIN = "/run/current-system/sw/bin"          # w103a: system tools are not on the agent PATH
WRAPPERS = "/run/wrappers/bin"                 # w104: setuid binaries live here


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{WRAPPERS}:{SYSBIN}:" + e.get("PATH", "")
    e.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return e


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env(), timeout=30)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def ancestors(pid: int) -> set[int]:
    """Every pid from `pid` up to init. The harness wrapper whose argv carries our search
    string is always one of these, which is exactly why the naive probe self-matches."""
    seen, cur = set(), pid
    while cur and cur not in seen:
        seen.add(cur)
        try:
            with open(f"/proc/{cur}/stat", "rb") as fh:
                # comm may contain spaces and parens; ppid is the field after the last ')'
                fields = fh.read().decode("latin-1").rsplit(")", 1)[1].split()
            cur = int(fields[1])
        except (OSError, IndexError, ValueError):
            break
    return seen


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().decode("latin-1").replace("\0", " ").strip()
    except OSError:
        return ""


def matches(needle: str, exclude: set[int]) -> list[tuple[int, str]]:
    """Every pid whose cmdline contains `needle`, minus us and our ancestors. Plain substring:
    no regex, so there is no BRE/ERE dialect to get wrong (the false-absence half)."""
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid in exclude:
            continue
        cl = cmdline(pid)
        if needle in cl:
            out.append((pid, cl))
    return sorted(out)


def etime(pid: int) -> str:
    rc, out = run([f"{SYSBIN}/ps", "-o", "etime=", "-p", str(pid)])
    return out.strip() if rc == 0 else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", default=None, help="transient systemd USER unit to consult first")
    ap.add_argument("--cmd-contains", required=True,
                    help="substring the job's command line must contain (not a regex)")
    a = ap.parse_args()

    exclude = ancestors(os.getpid())
    hits = matches(a.cmd_contains, exclude)

    unit_state = None
    if a.unit:
        rc, out = run([f"{SYSBIN}/systemctl", "--user", "is-active", a.unit])
        unit_state = out.strip() or f"rc={rc}"
        print(f"unit   {a.unit}: {unit_state}")

    for pid, cl in hits:
        print(f"proc   {pid}  etime {etime(pid)}  {cl[:120]}")
    print(f"excluded {len(exclude)} ancestor pid(s) of this probe: {sorted(exclude)}")

    if hits:
        # A unit that has gone inactive while the process lingers is not a clean "alive".
        if unit_state is not None and unit_state != "active":
            print(f"AMBIGUOUS: process present but unit is {unit_state!r}")
            return 2
        print("ALIVE")
        return 0
    if unit_state == "active":
        print("AMBIGUOUS: unit active but no process matches --cmd-contains "
              "(the job may have moved on to a later stage; check the log)")
        return 2
    print("DEAD")
    return 1


if __name__ == "__main__":
    sys.exit(main())
