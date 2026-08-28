"""w104a — STANDING GUARD: `setsid` DOES NOT ESCAPE A CGROUP, AND SYSTEMD KILLS BY CGROUP.

WHAT CLAIM THIS COVERS. The daily agent runner is a systemd SYSTEM unit,
`kaggle-playground.service`, with the default `KillMode=control-group`. When it deactivates at
the end of slot 10, systemd sends `KillSignal` to EVERY process in its cgroup. Cgroup membership
is inherited by `fork()` and is NOT changed by `setsid()`, by a double fork, or by orphaning to
init. So a job "detached" with any of those is still in the runner's cgroup and is still killed.

⚠⚠ THIS COST THIS WORKSPACE A SEVEN-HOUR BUILD AND THEN THREE MORE RUNS OF MISDIAGNOSIS.
`experiments/detach.py` was written on 08-20 to do exactly the double-fork + `os.setsid()` dance,
and RESEARCH carried "use detach.py for every long build from now on" for eight days. w96d's
member build WAS launched with it (pid 32247, ppid=1, own session — all three checks green) and
died anyway. The kernel log shows no OOM at that time; the journal shows
`kaggle-playground.service: Deactivated successfully` at 11:04:33 EDT on 08-27, three minutes
after the build's log stops mid-fold. It was SIGTERMed with the cgroup.

🎯 THE SHAPE WORTH KEEPING: **`ppid=1` and "own session" are the two things a detach tool can
show you, and neither one is the thing that decides.** The tool reports success in the vocabulary
it controls. The killer works in a vocabulary — cgroups — that the tool never mentions, so the
tool cannot report failure even in principle. w41 verified detach.py by checking ppid and session
id, which is exactly what it would have printed on the day it lost the build.

⚠ AND THE PROBE BUILT TO SETTLE THIS COULD NOT HAVE. w103 left a `setsid` heartbeat running to
see whether it outlived the session. It did — and that proves nothing, because the runner unit
stays ACTIVE across slot boundaries. The discriminating event is unit deactivation after slot 10,
which no single slot can observe. Same defect w103 itself named one paragraph earlier: a probe
whose two hypotheses predict the same observation reads exactly like a pass.

✅ THE ONE RECIPE THAT WORKS is a transient systemd USER unit (`systemd-run --user`), because the
user manager owns a DIFFERENT cgroup tree (`/user.slice/...`) that the system unit's teardown does
not reach. RESEARCH's "HOW TO RUN A JOB THAT OUTLIVES THE SESSION" section has the full command.
That recipe was right; the reason recorded for it was not, which is why nothing stopped a later
run from "simplifying" it back to setsid.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +-  THE HAZARD AND ITS FIX, IN CODE. Run against the real shipped `detach.py`, not a
         stand-in. Without `--force` it must REFUSE (rc=2) while we are inside the runner's
         cgroup. With `--force` its child must land in THE SAME cgroup as this process. If the
         child ever lands elsewhere the premise of this guard is wrong and RESEARCH's w104
         section must be retracted, not patched.
  C2 +   THE REMEDY, IN CODE. A child launched through the `systemd-run --user` recipe must land
         in a DIFFERENT cgroup, and one that is not under the runner unit. If this fails the
         documented recipe does not work and every long build is unprotected.
  C3 +-  THE READER, BOTH WAYS. A plain subprocess must share our cgroup, and pid 1 must not.
         If the cgroup reader returns a constant, C1 passes for the wrong reason.
  C4 +-  THE KILL MECHANISM. The runner unit must still be `KillMode=control-group`. If it is
         ever `process` or `none` the hazard is GONE — good news — and this guard plus its
         RESEARCH section must be retired DELIBERATELY. Good news goes RED (w103a C4).
  C5 +   LINGER. The remedy depends on the user manager surviving with no session. If
         `Linger=no` the transient user unit is no safer than detach.py and C2's green is a lie.
  C6 +   NOT VACUOUS. Every probe must have actually produced a cgroup string.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "w104a_cgroupguard.json")

SYSBIN = "/run/current-system/sw/bin"          # w103a: system tools live here, not on our PATH
RUNNER = "kaggle-playground.service"           # the system unit whose teardown does the killing
PROBE_TIMEOUT = 30.0

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL {msg}")


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{SYSBIN}{os.pathsep}{e.get('PATH', '')}"
    e.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return e


def cgroup_of(pid) -> str:
    """The cgroup-v2 path for a pid. '0::' is the unified-hierarchy prefix."""
    try:
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("0::"):
                    return line.strip()[3:]
    except OSError:
        return ""
    return ""


def wait_for(path: str, timeout: float = PROBE_TIMEOUT) -> str:
    """Poll for a probe's output. An empty return means the probe never wrote — which is a
    failure of the probe, reported as such, never silently read as 'same cgroup'."""
    end = time.time() + timeout
    while time.time() < end:
        if os.path.exists(path) and os.path.getsize(path):
            return open(path, encoding="utf-8").read().strip()
        time.sleep(0.25)
    return ""


def main() -> int:
    ours = cgroup_of("self")
    tag = f"w104a_{os.getpid()}"
    tmp = f"/tmp/{tag}"
    os.makedirs(tmp, exist_ok=True)
    print(f"this process cgroup: {ours or '(unreadable)'}")
    if not ours:
        fail("C6: cannot read our own cgroup — every comparison below is meaningless")
        print(f"\nFAILURES {FAILS}")
        return 1
    in_runner = RUNNER in ours
    print(f"running inside {RUNNER}: {in_runner}\n")

    probes = {}

    # ---- C1 +: detach.py, the real shipped tool, must NOT escape the cgroup.
    d_out = os.path.join(tmp, "detach.cg")
    d_log = os.path.join(tmp, "detach.log")
    detach = os.path.join(HERE, "detach.py")
    if not os.path.exists(detach):
        print("C1 detach.py is gone — the hazard it embodies cannot be measured here.")
        fail("C1: detach.py missing. If it was deleted on purpose, delete this control with it "
             "and say so; do not leave a probe pointed at nothing")
    else:
        # Two directions. Without --force detach.py must REFUSE while we are inside the
        # runner's cgroup (that refusal is the fix w104 shipped, and a guard that never
        # exercises it lets the fix rot). With --force it must still detach, and the child
        # must land in OUR cgroup -- which is the hazard this whole guard is about.
        refusal = subprocess.run([sys.executable, detach, d_log, "/bin/true"],
                                 env=env(), capture_output=True, text=True, timeout=60)
        print(f"C1 detach.py without --force inside the runner: rc={refusal.returncode} "
              f"(want 2)")
        if in_runner and refusal.returncode != 2:
            fail(f"C1: detach.py returned rc={refusal.returncode} from inside {RUNNER}. It is "
                 "supposed to refuse here; it is handing back a pid that looks detached and "
                 "is not, which is precisely how the 08-27 build was lost")
        subprocess.run([sys.executable, detach, "--force", d_log, "/bin/sh", "-c",
                        f"cat /proc/self/cgroup > {d_out}"],
                       env=env(), capture_output=True, timeout=60)
        raw = wait_for(d_out)
        got = raw[3:] if raw.startswith("0::") else raw
        probes["detach"] = got
        print(f"C1 detach.py child cgroup: {got or '(no output)'}")
        if not got:
            fail("C1: the detach.py probe wrote nothing within the timeout — it did not run, so "
                 "this control measured nothing (not 'nothing is wrong')")
        elif got != ours:
            fail(f"C1: detach.py landed in a DIFFERENT cgroup ({got}) from ours ({ours}). "
                 "That contradicts this guard's premise — setsid/double-fork would then be a "
                 "real escape, and RESEARCH's w104 section must be retracted, not patched")
        else:
            print("   == ours. setsid + double fork does NOT leave the cgroup, as claimed.")

    # ---- C2 +: the systemd-run --user recipe must escape it.
    s_out = os.path.join(tmp, "srun.cg")
    unit = f"{tag}-probe"
    # `--setenv=PATH` is not decoration: a transient unit inherits an almost-empty PATH, so the
    # obvious `cat /proc/self/cgroup` exits 127 with `cat: command not found` and the probe reads
    # as "the remedy is broken". That is RESEARCH's documented trap #1 for this recipe, and this
    # guard tripped over it on its first run. Launch the way the recipe says to launch.
    r = subprocess.run([f"{SYSBIN}/systemd-run", "--user", "--quiet", "--collect",
                        f"--setenv=PATH={SYSBIN}:/usr/bin:/bin",
                        f"--unit={unit}", "/bin/sh", "-c",
                        f"cat /proc/self/cgroup > {s_out}"],
                       env=env(), capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        fail(f"C2: `systemd-run --user` failed rc={r.returncode}: "
             f"{(r.stderr or r.stdout).strip()[:200]} — the documented launch recipe is "
             "UNUSABLE right now, so every long build is running unprotected")
        probes["systemd_run"] = ""
    else:
        raw = wait_for(s_out)
        got = raw[3:] if raw.startswith("0::") else raw
        probes["systemd_run"] = got
        print(f"C2 systemd-run --user child cgroup: {got or '(no output)'}")
        if not got:
            fail("C2: the systemd-run probe wrote nothing within the timeout — the unit started "
                 "but produced no cgroup, so the remedy is unverified this run")
        else:
            if got == ours:
                fail(f"C2: the transient user unit landed in OUR cgroup ({ours}). The recipe "
                     "does not isolate anything and buys nothing over detach.py")
            if RUNNER in got:
                fail(f"C2: the transient user unit is under {RUNNER} ({got}) — it dies with the "
                     "runner exactly like detach.py does")
            if got != ours and RUNNER not in got:
                print("   != ours and outside the runner unit. The remedy works.")

    # ---- C3 +-: the reader itself, both directions.
    plain = subprocess.run(["/bin/sh", "-c", "cat /proc/self/cgroup"],
                           capture_output=True, text=True, timeout=30).stdout.strip()
    plain = plain[3:] if plain.startswith("0::") else plain
    init = cgroup_of(1)
    probes["plain_child"] = plain
    probes["pid1"] = init
    print(f"\nC3 plain child cgroup: {plain or '-'}   (want == ours)")
    print(f"C3 pid 1 cgroup:       {init or '-'}   (want != ours)")
    if plain != ours:
        fail(f"C3: an ordinary child does not share our cgroup ({plain} vs {ours}) — cgroup "
             "membership is not being inherited the way this guard assumes")
    if not init or init == ours:
        fail("C3: pid 1 reads the same cgroup as us, so the reader is returning a constant and "
             "C1's equality green means nothing")

    # ---- C4 +-: the kill mechanism. Good news goes RED.
    km = subprocess.run([f"{SYSBIN}/systemctl", "show", RUNNER, "-p", "KillMode",
                         "--value"], env=env(), capture_output=True, text=True,
                        timeout=30).stdout.strip()
    ks = subprocess.run([f"{SYSBIN}/systemctl", "show", RUNNER, "-p", "KillSignal",
                         "--value"], env=env(), capture_output=True, text=True,
                        timeout=30).stdout.strip()
    print(f"\nC4 {RUNNER} KillMode={km or '-'} KillSignal={ks or '-'}")
    if km != "control-group":
        fail(f"C4: KillMode is {km!r}, not 'control-group'. If the unit was changed on purpose "
             "the cgroup hazard is GONE and this guard plus RESEARCH's w104 section must be "
             "retired deliberately. If it was not changed, read why systemctl disagrees before "
             "trusting any other line here.")

    # ---- C5 +: the remedy's own dependency.
    linger = subprocess.run([f"{SYSBIN}/loginctl", "show-user", str(os.getuid()),
                             "-p", "Linger", "--value"],
                            env=env(), capture_output=True, text=True, timeout=30).stdout.strip()
    print(f"C5 Linger={linger or '-'}  (want yes)")
    if linger != "yes":
        fail(f"C5: Linger={linger!r}. The user manager may stop when no session remains, which "
             "takes every transient user unit with it — C2's isolation would then be worthless "
             "at exactly the moment it is needed")

    # ---- C6 +: not vacuous.
    got_any = [k for k, v in probes.items() if v]
    print(f"\nC6 probes that produced a cgroup: {len(got_any)}/{len(probes)} {got_any}")
    if len(got_any) < 3:
        fail(f"C6: only {len(got_any)} probes produced output — a guard whose probes all failed "
             "to run prints almost exactly what a passing guard prints")

    json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                   ours=ours, in_runner=in_runner, runner=RUNNER,
                   kill_mode=km, kill_signal=ks, linger=linger,
                   probes=probes, failures=FAILS), open(OUT, "w"), indent=1)

    for p in (d_out, d_log, s_out):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass

    print(f"\nFAILURES {FAILS}")
    if FAILS == 0:
        print("OK: detach.py stays in the runner's cgroup and dies with it; the systemd-run "
              "--user recipe does not. Launch long jobs with the recipe.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
