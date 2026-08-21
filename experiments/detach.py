"""detach.py -- run a command as a real daemon, fully detached from this session.

WHY THIS EXISTS: the w36g -> w38d -> w40f chain died at 18:44 UTC on 08-20 when the session
that launched it ended, losing ~7 hours of build time silently. `nohup ... &` was not enough.
`setsid` DOES NOT EXIST on this box (like `ps` and `pgrep` -- see the w40 journal §0), and it
fails with "command not found" INSIDE the backgrounded shell, so the launch looks like it
worked and nothing runs. This does the double-fork + os.setsid() in Python, which is always
available here.

Usage:  python experiments/detach.py <logfile> <cmd> [args...]
Prints the daemon PID. Verify with the /proc scan, never with ps/pgrep.
"""
import os, sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork():                      # parent returns to the shell immediately
    os.wait()
    sys.exit(0)
os.setsid()                        # new session: no controlling terminal, new process group
pid = os.fork()
if pid:                            # intermediate exits so the daemon is orphaned to init
    print(f"detached pid {pid} -> {log}", flush=True)
    os._exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.close(os.open(os.devnull, os.O_RDONLY))
os.execvp(cmd[0], cmd)
