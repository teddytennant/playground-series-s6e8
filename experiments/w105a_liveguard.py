r"""w105a — STANDING GUARD: `pgrep -af <token>` LIES IN BOTH DIRECTIONS HERE.

WHAT CLAIM THIS COVERS. RESEARCH's handoff step 2 has said "check the build POSITIVELY" with
`pgrep -af w96c_build` since w103, and w103 was right that a liveness check must be positive
rather than inferred from a silent log. The command it prescribed cannot deliver that, for two
independent reasons that fail in OPPOSITE directions. w105 hit both inside ten minutes.

  FALSE ABSENCE.  `pgrep` matches with ERE. `\|` is BRE. So `pgrep -af 'w96c_build\|w26i_value'`
                  is a literal that matches nothing real, and its empty output is
                  INDISTINGUISHABLE from the process being dead. w105 typed that, read a
                  running build as a third lost build, and only caught it because
                  `systemctl --user is-active` disagreed. Cost of believing it: relaunching a
                  build that is already running, i.e. 16 cores double-booked and `oof_w96/`
                  written by two processes at once.
  FALSE PRESENCE. The agent runs every command as `bash -c '<the entire script text>'`, so the
                  invoking wrapper's argv CONTAINS the token being searched for. `pgrep -af`
                  matches that wrapper and exits 0. ⟹ with the build DEAD, the prescribed
                  positive liveness check still prints a line and still reads as alive. This is
                  the more expensive half: it is how a fourth lost build would go unnoticed for
                  a whole run.

🎯 THE SHAPE, and it is w103's and w104's lesson arriving together in ONE command. w103: a false
absence closes routes. w104: a false presence keeps broken routes open. Here the SAME prescribed
command produces both, depending only on how the pattern was typed -- and neither failure is
visible in its output, because the output of a lying pgrep looks exactly like the output of an
honest one. ⟹ **A LIVENESS CHECK MUST ASSERT THAT THE MATCHED LINE LOOKS LIKE THE JOB, NOT THAT
SOME LINE MATCHED**, and it must never count its own ancestors. `experiments/alive.py` does both.

⚠ THE BRACKET TRICK IS SOUND BUT FRAGILE HERE, AND THE DIFFERENCE MATTERS. RESEARCH has
prescribed `pgrep -f 'w96c_[b]uild'` since w14. Measured this run: ALONE in its own command it
works (rc=1 against a token nothing carries). But the wrapper's argv holds the ENTIRE script, so
any OTHER arm -- another probe, an echo, even a comment -- that mentions the plain token defeats
the bracketed arm silently. w105 was fooled by exactly that twice in one run and nearly wrote
down "the bracket trick does not work", which would have been a false claim manufactured by a
confounded probe: the very failure this guard is about. It also does nothing about the ERE half.
⟹ Bracket if you like, but the property that actually holds is ancestor-exclusion + shape check.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +-  THE FALSE PRESENCE, IN CODE. With NO process carrying the token, run both readers from
         a wrapper whose argv DOES carry it. Naive `pgrep -af` must report a match (the hazard
         is live); `alive.py` must report DEAD (the fix works). 🔴 If the naive reader comes back
         empty the harness no longer puts the script in argv -- the hazard is GONE, which is good
         news, and this guard plus its RESEARCH section must then be retired DELIBERATELY rather
         than left to pass on nothing. Good news goes RED (w103a C4, w104a C4).
  C2 +-  THE FALSE ABSENCE, IN CODE. Against a REAL sentinel process, from a wrapper that does
         NOT carry the token: bare `pgrep -af TOK` must FIND it, `pgrep -af 'TOK\|zz'` must NOT,
         and `alive.py` must find it. 🔴 If the BRE form ever finds it, procps has changed
         dialect, the hazard is gone, and again the retirement is a decision.
  C3 +-  THE READER, BOTH WAYS. `alive.py` must say ALIVE for a genuinely running sentinel and
         DEAD for a token nothing carries. Without this, C1 and C2 pass for a reader that always
         returns the answer they wanted.
  C4 +   THE DOCUMENT. RESEARCH.md must name `alive.py` as the liveness recipe, and no `pgrep`
         line anywhere in RESEARCH.md may carry a `\|` alternation. The defect this guard exists
         for entered through a prescription in that document, so the document is in scope.
  C5 +   NOT VACUOUS. Every sentinel must actually have started, and every arm must actually
         have run. A probe that failed to spawn must not be scored as evidence of anything.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESEARCH = os.path.join(ROOT, "RESEARCH.md")
ALIVE = os.path.join(HERE, "alive.py")
PY = os.path.join(ROOT, ".venv", "bin", "python")
OUT = os.path.join(HERE, "w105a_liveguard.json")

SYSBIN = "/run/current-system/sw/bin"      # w103a: pgrep/ps/systemctl are not on the agent PATH
WRAPPERS = "/run/wrappers/bin"             # w104: setuid binaries
TIMEOUT = 30.0

FAILS = 0
NOTES: dict = {}


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL {msg}")


def env() -> dict:
    e = dict(os.environ)
    e["PATH"] = f"{WRAPPERS}:{SYSBIN}:" + e.get("PATH", "")
    return e


def sh(script: str, extra_env: dict | None = None) -> tuple[int, str]:
    """Run `script` through `bash -c`, so the wrapper's argv holds the script text verbatim --
    which is precisely the harness shape C1 needs to reproduce."""
    e = env()
    if extra_env:
        e.update(extra_env)
    p = subprocess.run([f"{SYSBIN}/bash", "-c", script], capture_output=True, text=True,
                       env=e, timeout=TIMEOUT)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def spawn_sentinel(token: str) -> subprocess.Popen:
    """A process whose command line contains `token` and which does nothing for 25s."""
    # NO `exec`: bash tail-call-optimises `bash -c '<one command>'` into an execve, which
    # REPLACES the argv and takes the token with it. The trailing `:` defeats that, so the
    # shell survives as a process whose command line still carries the token. w105 wrote the
    # exec form first and C2 correctly reported a sentinel it could not find.
    proc = subprocess.Popen([f"{SYSBIN}/bash", "-c", f"# {token}\n{SYSBIN}/sleep 25; :"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env())
    deadline = time.time() + 3.0
    while time.time() < deadline:             # visible in /proc AND still carrying the token
        try:
            with open(f"/proc/{proc.pid}/cmdline", "rb") as fh:
                if token.encode() in fh.read():
                    time.sleep(0.2)           # outlast any late exec that would wipe the argv
                    with open(f"/proc/{proc.pid}/cmdline", "rb") as fh2:
                        if token.encode() in fh2.read():
                            return proc
        except OSError:
            pass
        time.sleep(0.05)
    return proc


def main() -> int:
    if not os.path.exists(ALIVE):
        print(f"ERROR: {ALIVE} missing -- the fix this guard certifies is not on disk")
        return 2

    # ---- C1: the false presence -------------------------------------------------------------
    # Token carried ONLY by the bash -c wrapper. Nothing is running under it.
    tok1 = "w105tok" + uuid.uuid4().hex[:12]
    rc_naive, out_naive = sh(f'{SYSBIN}/pgrep -a -f -- "{tok1}"; echo "RC=$?"')
    naive_rc = int(re.search(r"RC=(\d+)", out_naive).group(1))
    rc_fix, out_fix = sh(f'{PY} {ALIVE} --cmd-contains "{tok1}"; echo "RC=$?"')
    fix_rc = int(re.search(r"RC=(\d+)", out_fix).group(1))
    NOTES["c1"] = {"token": tok1, "naive_rc": naive_rc, "fix_rc": fix_rc,
                   "naive_lines": len([l for l in out_naive.splitlines()
                                       if l and not l.startswith("RC=")])}
    if naive_rc != 0:
        fail("C1 naive `pgrep -af` did NOT self-match the wrapper (rc=%d). The harness no "
             "longer puts the script in argv: THE HAZARD IS GONE. Good news -- retire this "
             "guard and RESEARCH's w105 section deliberately, do not delete this line."
             % naive_rc)
    if fix_rc != 1:
        fail(f"C1 alive.py should report DEAD (rc=1) with nothing running; got rc={fix_rc}")

    # ---- C2: the false absence (pgrep is ERE, not BRE) ---------------------------------------
    tok2 = "w105tok" + uuid.uuid4().hex[:12]
    sent = spawn_sentinel(tok2)
    try:
        started = False
        try:
            with open(f"/proc/{sent.pid}/cmdline", "rb") as fh:
                started = tok2.encode() in fh.read()
        except OSError:
            pass
        NOTES["c2"] = {"token": tok2, "pid": sent.pid, "sentinel_started": started}
        if not started:                                                             # C5
            fail("C2/C5 sentinel never appeared in /proc -- no arm below is evidence")
        else:
            # Both pgrep arms run from a script whose argv does NOT hold the token: the token
            # arrives through the environment. Otherwise every arm self-matches (w105 §2).
            rc_bare, out_bare = sh(f'{SYSBIN}/pgrep -a -f -- "$W105TOK"; echo "RC=$?"',
                                   {"W105TOK": tok2})
            rc_bre, out_bre = sh(f'{SYSBIN}/pgrep -a -f -- "$W105TOK\\|zzq_nothing"; echo "RC=$?"',
                                 {"W105TOK": tok2})
            bare_rc = int(re.search(r"RC=(\d+)", out_bare).group(1))
            bre_rc = int(re.search(r"RC=(\d+)", out_bre).group(1))
            rc_fix2, _ = sh(f'{PY} {ALIVE} --cmd-contains "$W105TOK" >/dev/null; echo "RC=$?"',
                            {"W105TOK": tok2})
            fix2_rc = int(re.search(r"RC=(\d+)", _).group(1))
            NOTES["c2"].update({"bare_rc": bare_rc, "bre_rc": bre_rc, "alive_rc": fix2_rc})
            if bare_rc != 0:
                fail(f"C2 bare `pgrep -af` failed to find a live sentinel (rc={bare_rc}) -- "
                     "the instrument is broken, not the dialect claim")
            if bre_rc == 0:
                fail("C2 `pgrep -af 'TOK\\|zz'` FOUND the sentinel. procps now accepts BRE "
                     "alternation: the false-absence hazard is GONE. Good news -- retire this "
                     "control deliberately.")
            if fix2_rc != 0:
                fail(f"C2 alive.py failed to find a live sentinel (rc={fix2_rc})")

            # ---- C3: the reader, both ways ---------------------------------------------------
            rc_alive, _o = sh(f'{PY} {ALIVE} --cmd-contains "$W105TOK" >/dev/null; echo "RC=$?"',
                              {"W105TOK": tok2})
            rc_dead, _o2 = sh(f'{PY} {ALIVE} --cmd-contains "$W105TOK" >/dev/null; echo "RC=$?"',
                              {"W105TOK": "w105tok" + uuid.uuid4().hex[:12]})
            a_rc = int(re.search(r"RC=(\d+)", _o).group(1))
            d_rc = int(re.search(r"RC=(\d+)", _o2).group(1))
            NOTES["c3"] = {"alive_rc": a_rc, "dead_rc": d_rc}
            if a_rc != 0 or d_rc != 1:
                fail(f"C3 reader is not firing both ways: live={a_rc} (want 0), "
                     f"absent={d_rc} (want 1)")
    finally:
        sent.kill()
        sent.wait(timeout=5)

    # ---- C4: the document ---------------------------------------------------------------------
    txt = open(RESEARCH, encoding="utf-8").read()
    names_fix = "alive.py" in txt
    # The w105 section quotes the anti-pattern ON PURPOSE -- it is the section documenting it.
    # Exclude exactly that span (its header to the next top-level header) and scan the rest, so
    # the exemption dies with the section rather than outliving it.
    lines = txt.splitlines()
    lo = next((i for i, l in enumerate(lines) if l.startswith("# ") and "w105" in l.lower()), None)
    if lo is None:
        lo = next((i for i, l in enumerate(lines) if "(w105," in l), None)
    hi = len(lines)
    if lo is not None:
        hi = next((j for j in range(lo + 1, len(lines))
                   if lines[j].startswith("# ") and "w105" not in lines[j].lower()), len(lines))
    exempt = range(lo, hi) if lo is not None else range(0)
    bad = [l.strip()[:100] for i, l in enumerate(lines)
           if "pgrep" in l and r"\|" in l and i not in exempt]
    NOTES.setdefault("c4_span", {"lo": lo, "hi": hi})
    NOTES["c4"] = {"research_names_alive_py": names_fix, "pgrep_with_bre_alternation": bad}
    if not names_fix:
        fail("C4 RESEARCH.md does not name `alive.py`; the prescription that caused this "
             "defect is still the only one on offer")
    if bad:
        fail(f"C4 RESEARCH.md still shows a pgrep pattern with a BRE alternation: {bad}")

    # ---- C5: not vacuous ----------------------------------------------------------------------
    ran = [k for k in ("c1", "c2", "c3", "c4") if k in NOTES]
    NOTES["c5"] = {"arms_run": ran}
    if len(ran) < 4:
        fail(f"C5 only {len(ran)} of 4 arms produced a reading: {ran}")

    NOTES["utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    NOTES["failures"] = FAILS
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(NOTES, fh, indent=2, sort_keys=True)
    print(f"FAILURES {FAILS}   -> {os.path.relpath(OUT, ROOT)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
