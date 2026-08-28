"""detach.py -- double-fork + os.setsid(). ⛔ THIS DOES NOT SURVIVE THE DAILY RUNNER. READ ON.

WHAT THIS ACTUALLY DOES: it detaches a child from the controlling terminal, gives it a new
session and process group, and orphans it to init. Those are real things and it does them.

⛔ WHAT IT DOES NOT DO, DESPITE EIGHT DAYS OF RESEARCH.md SAYING OTHERWISE: survive the end of a
daily run. The runner is a systemd SYSTEM unit, `kaggle-playground.service`, with the default
`KillMode=control-group`. When it deactivates after slot 10 systemd signals EVERY process in its
cgroup. **Cgroup membership is inherited across fork() and is not changed by setsid(), by a
second fork, or by being reparented to init** -- so the "detached" child is still in the runner's
cgroup and is still killed. `experiments/w104a_cgroupguard.py` C1 measures exactly this.

THIS IS NOT THEORETICAL. w96d's member build was launched through this script on 08-27 (pid
32247, ppid=1, own session -- every check this script can offer came back green) and was
SIGTERMed at 11:04:33 EDT with the unit. No OOM in the kernel log; the log simply stops
mid-fold. Roughly seven hours of build time across two attempts, plus three later runs spent
diagnosing a "truncated log" that was never the evidence.

🎯 The reason nobody caught it: **ppid=1 and "own session" are the only two facts a detach tool
can report, and neither is the one that decides.** The tool reports success in the vocabulary it
controls; the killer works in a vocabulary the tool never mentions.

✅ USE THIS INSTEAD, for anything that must outlive the run (RESEARCH, "HOW TO RUN A JOB THAT
OUTLIVES THE SESSION THAT STARTS IT"). A transient systemd USER unit lives in the user manager's
cgroup tree, which the system unit's teardown does not reach:

    export PATH=/run/current-system/sw/bin:$PATH
    export XDG_RUNTIME_DIR=/run/user/$(id -u)
    systemd-run --user --unit=<name> --collect -p WorkingDirectory="$PWD" \
        /bin/bash "$PWD/experiments/<script>.sh"

    systemctl --user is-active <name>.service      # verify POSITIVELY, always

⚠ The unit's PATH is nearly empty -- have the script export a real one and `cd` to an absolute
path before it does anything else, or it fails 30 lines later with a message about the venv.

WHAT THIS SCRIPT DOES NOW: it checks its own cgroup and REFUSES rather than handing back a pid
that looks detached and is not. A loud failure at second zero beats a silent one seven hours in.

Usage:  python experiments/detach.py [--force] <logfile> <cmd> [args...]
        --force  detach anyway. Legitimate for a job meant to die with the run.
"""
import os, sys

RUNNER_MARKERS = ("kaggle-playground.service", "/system.slice/")


def cgroup_of_self():
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("0::"):
                    return line.strip()[3:]
    except OSError:
        pass
    return ""


argv = sys.argv[1:]
force = False
if argv and argv[0] == "--force":
    force, argv = True, argv[1:]
if len(argv) < 2:
    sys.exit("usage: detach.py [--force] <logfile> <cmd> [args...]")
log, cmd = argv[0], argv[1:]

cg = cgroup_of_self()
if not force and any(m in cg for m in RUNNER_MARKERS):
    sys.stderr.write(
        f"detach.py REFUSING: this process is in cgroup\n    {cg}\n"
        "which systemd kills as a unit (KillMode=control-group). setsid and the double fork do\n"
        "NOT leave a cgroup, so the child would be killed with the run -- exactly how the w96\n"
        "build was lost on 08-27. Launch it as a transient systemd user unit instead:\n\n"
        "    export XDG_RUNTIME_DIR=/run/user/$(id -u)\n"
        "    systemd-run --user --unit=<name> --collect -p WorkingDirectory=\"$PWD\" \\\n"
        "        /bin/bash \"$PWD/experiments/<script>.sh\"\n\n"
        "Pass --force only if the job is genuinely meant to die when the run ends.\n")
    sys.exit(2)

if os.fork():                      # parent returns to the shell immediately
    os.wait()
    sys.exit(0)
os.setsid()                        # new session: no controlling terminal, new process group
pid = os.fork()
if pid:                            # intermediate exits so the daemon is orphaned to init
    print(f"detached pid {pid} -> {log}  (cgroup {cg or 'unknown'})", flush=True)
    os._exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.close(os.open(os.devnull, os.O_RDONLY))
os.execvp(cmd[0], cmd)
