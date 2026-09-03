"""w160a (#70) — a run's record must be PUSHED, not merely committed, and the remote must be
read LIVE rather than remembered.

WHY THIS EXISTS. w160 opened on `origin/main` two commits behind `HEAD`:

    975bde7  09-03 08:53  Update the closed-column guard's entry counts to the post-fix text
    f3a837e  09-03 09:22  Correct the price column's t to delta/(sd/sqrt(REPS)) and guard the scale

The second one is guard #68 in its entirety. w157 pushed at 08:48:57 and then made a fourth
commit at 08:53:46; w159 committed at 09:22:14 and pushed nothing. Neither run's push was
refused -- neither run ran one.

⛔ AND THE WORKSPACE HAD ALREADY WRITTEN DOWN BOTH THE FAILURE AND ITS FIX. RESEARCH.md, w136,
2026-08-31, five runs before the first miss:

    🔴 AND IT EXITS 0 WHEN PIPED, so `git push … | tail; echo RC=$?` prints `RC=0` on a push
    that did not happen. Confirm with `git status -sb` (look for `ahead N`) or
    `git log --oneline origin/main -1`, never with the exit code of a pipeline.

Reproduced live at w160, with git on PATH and `gh` deliberately off it:

    piped    gh auth git-credential get: line 1: gh: command not found     RC_OF_PIPELINE=0
    unpiped  same two lines                                                RC_DIRECT=128
    after    git status -sb  ->  ## main...origin/main [ahead 2]

🎯 SO THE PRESCRIPTION WAS RIGHT AND IT WAS PROSE. It is read by whoever greps for it; a check
runs every time. Measured over the shipped STEMS list: of the 69 standing checks, ZERO read the
remote -- no `origin/`, no `ls-remote`, no `@{u}`, no `rev-list` anywhere in `experiments/*.py`.
#69 declared this hole one run ago in its own C7 ("it reads HEAD, so a run that commits its
entry and never PUSHES is invisible") and the hole already had two commits sitting in it.

WHAT THIS ADDS, IN ONE LINE. #66 asks "was the entry written". #69 asks "is it in HEAD".
#70 asks "is HEAD on the remote", and answers it by talking to the remote.

⚠ THERE IS NO IN-FLIGHT EXEMPTION HERE, ON PURPOSE. #66's open-ended exemption is what let
three runs vanish and #69 had to supply its expiry. A push has no preparatory state -- unlike
an entry, which must be written before it can be committed, you have either pushed or you have
not. So a red between this run's commit and this run's push is not a false positive: it is the
guard saying, accurately, that the record is not published yet.

  C1   HEAD, the branch, its upstream, and the remote tip read LIVE over the network.
  C1b  the read tracks the remote rather than returning a constant.
  C2   the live read is ANSWERABLE. A read that fails is UNREADABLE and FAILS -- a standing
       check that cannot see its subject must never report clean. The message names PATH
       first, because in this sandbox that is what is broken (RESEARCH, w136).
  C2b  the cached `origin/main` ref agrees with the live tip. Drift there is exactly what
       makes w136's own prescribed confirmation (`git log --oneline origin/main -1`) answer
       out of memory.
  C3   the assertion: HEAD is reachable from the remote tip. Every unpublished commit is
       listed by sha, date and subject.
  C4   the exit-code hazard reproduced, as the reason C3 re-reads the remote instead of
       trusting a return code.
  C5   the anchors in w159a_committedguard.py and RESEARCH.md this guard reasons about.
  C6   --control, every input frozen (w115a:80-86, and w156's rebuilt control: never borrow
       live state, or the control stops working the moment the fix lands).
  C7   blindness.

    .venv/bin/python experiments/w160a_pushguard.py             # suite form, rc 0/1
    .venv/bin/python experiments/w160a_pushguard.py --control   # exits 0, having fired
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESEARCH = ROOT / "RESEARCH.md"
COMMITTEDGUARD = ROOT / "experiments" / "w159a_committedguard.py"

# `gh` is the credential helper and lives ONLY here; the agent shell's PATH omits it, so every
# network git call dies with a message that names credentials rather than PATH (RESEARCH, w136
# and w160 §"gh IS NOT ON THE SANDBOX PATH"). The guard supplies it rather than inheriting it,
# because a check that is inert in its own sandbox is worse than no check.
GH_BIN = "/run/current-system/sw/bin"

# The two commits w160 opened on, and the tip the remote actually had. Frozen for C6: these are
# real objects that never change, so the control keeps working after the fix lands.
FROZEN_HEAD = "f3a837e7811783de3d934900bb711066a1883fe4"
FROZEN_REMOTE_PREFIX = "2981d1cdedb0b43f132c474f5c9819d70ea51bd9"

ANCHORS = {
    COMMITTEDGUARD: {
        "#69's C7 line, the blindness this guard closes":
            "it reads HEAD, so a run that commits its entry and never PUSHES is invisible",
        "#69's HEAD-side read, which C3 mirrors onto the remote":
            "git show HEAD:JOURNAL.md",
    },
    RESEARCH: {
        "w136's PATH recipe, which GH_BIN copies":
            'PATH=/run/current-system/sw/bin:/usr/bin:/bin:$PATH git push origin main',
        "w136's prescription, which C3 encodes":
            "never with the exit code of a pipeline",
    },
}

fails: list[str] = []


def fail(msg: str) -> None:
    fails.append(msg)
    print(f"  FAIL: {msg}")


def git(*args: str, net: bool = False) -> tuple[int, str]:
    """Run git and return (rc, stdout+stderr). rc is taken DIRECTLY, never through a pipe."""
    env = None
    if net:
        env = dict(os.environ)
        env["PATH"] = GH_BIN + ":" + env.get("PATH", "")
    p = subprocess.run(("git",) + args, cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=180)
    return p.returncode, (p.stdout + p.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true",
                    help="run against frozen pre-fix and post-fix inputs, then exit 0")
    a = ap.parse_args()

    if a.control:
        return control()

    print("C1 HEAD, the branch, its upstream, and the remote tip read live")
    rc, head = git("rev-parse", "HEAD")
    if rc:
        fail(f"cannot read HEAD: {head}")
        print(f"\nFAILURES: {len(fails)}")
        return 1
    rc, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    rc_up, upstream = git("rev-parse", "--abbrev-ref", "HEAD@{u}")
    upstream = upstream if rc_up == 0 else None
    print(f"  HEAD      {head[:7]}  on branch {branch}")
    print(f"  upstream  {upstream if upstream else '(none configured)'}")
    if upstream is None:
        fail("the branch has no upstream, so nothing can say whether it is published")
        print(f"\nFAILURES: {len(fails)}")
        return 1
    remote, _, remote_branch = upstream.partition("/")

    print("C2 the live read is answerable")
    rc, out = git("ls-remote", remote, f"refs/heads/{remote_branch}", net=True)
    if rc != 0 or not out:
        fail(f"UNREADABLE: `git ls-remote {remote}` rc={rc}. This is NOT a pass. First thing "
             f"to check is PATH -- `gh` is the credential helper and lives only at {GH_BIN} "
             f"(RESEARCH, w136). Output: {out[:200]!r}")
        print(f"\nFAILURES: {len(fails)}")
        return 1
    live_tip = out.split()[0]
    print(f"  ls-remote {remote} refs/heads/{remote_branch}  ->  {live_tip[:7]}   OK")

    print("C1b the read tracks the remote rather than returning a constant")
    rc, out_none = git("ls-remote", remote, "refs/heads/no-such-branch-w160a", net=True)
    if out_none.strip():
        fail(f"a branch that cannot exist returned {out_none!r} -- the parse is not selective")
    else:
        print("  a ref that cannot exist returns nothing at all: ''  OK")

    print("C2b the cached remote-tracking ref agrees with the live tip")
    rc, cached = git("rev-parse", f"{upstream}")
    if rc:
        fail(f"no cached ref for {upstream}")
    elif cached != live_tip:
        fail(f"{upstream} is cached at {cached[:7]} but the remote has {live_tip[:7]} -- "
             f"`git log --oneline {upstream} -1` is answering out of memory")
    else:
        print(f"  {upstream} cached at {cached[:7]} == live {live_tip[:7]}  OK")

    print("C3 every commit in HEAD is reachable from the remote tip")
    rc, _ = git("merge-base", "--is-ancestor", head, live_tip)
    if rc == 0:
        print(f"  HEAD {head[:7]} is published on {upstream}   OK")
    else:
        rc2, listing = git("log", "--format=%h %ad %s", "--date=short",
                           f"{live_tip}..{head}")
        rows = [r for r in listing.splitlines() if r.strip()]
        fail(f"{len(rows)} commit(s) in HEAD are NOT on {remote}. "
             f"Push them: PATH={GH_BIN}:$PATH git push {remote} {remote_branch}")
        for r in rows:
            print(f"        {r}")

    print("C4 the hazard C3 exists to route around, reproduced rather than quoted")
    piped = subprocess.run("exit 128 | tail -1", shell=True, capture_output=True, text=True)
    direct = subprocess.run(["sh", "-c", "exit 128"], capture_output=True, text=True)
    print(f"  a command that exits 128, piped:   rc={piped.returncode}   "
          f"unpiped: rc={direct.returncode}")
    if piped.returncode == 0 and direct.returncode == 128:
        print("  so a push's exit code proves nothing; C3 asks the remote instead   OK")
    else:
        fail(f"the pipeline hazard did not reproduce (piped {piped.returncode}, "
             f"direct {direct.returncode}) -- C3's rationale needs re-reading")

    print("C5 the anchors this guard reasons about are still where it read them")
    for path, anchors in ANCHORS.items():
        src = path.read_text()
        for what, line in anchors.items():
            if line in src:
                print(f"  OK   {path.name}: {what}")
            else:
                fail(f"{path.name}: {what} -- the line this guard assumes is gone")

    print("C7 scope and blindness, measured")
    print("  it reads ONE branch's tip. A record committed on another branch, or a tag, or a")
    print("    stash, is invisible -- as is anything never committed at all (#69 covers that)")
    print("  it proves the commits are ON the remote, not that the remote is INTACT: a")
    print("    force-push that dropped history leaves HEAD an ancestor and this stays green")
    print("  it says nothing about UNTRACKED files, which is where most of this workspace's")
    print("    artefacts live -- `git status` remains the only thing that sees those")
    print("  it needs the network. Offline it FAILS as UNREADABLE, which is noisy by design;")
    print("    the alternative is a check that reports clean when it cannot see")

    print(f"\nFAILURES: {len(fails)}")
    return 1 if fails else 0


def control() -> int:
    """Both arms frozen: real, immutable objects, no live HEAD and no live remote."""
    print("C6 --control, every input frozen")
    ok = True
    for label, head, tip, want in (
        ("pre-fix  (w160's opening state)", FROZEN_HEAD, FROZEN_REMOTE_PREFIX, "FIRES"),
        ("post-fix (after the push)      ", FROZEN_HEAD, FROZEN_HEAD, "SILENT"),
    ):
        rc, _ = git("merge-base", "--is-ancestor", head, tip)
        rc2, listing = git("log", "--format=%h %s", f"{tip}..{head}")
        rows = [r for r in listing.splitlines() if r.strip()]
        got = "SILENT" if rc == 0 else "FIRES"
        mark = "OK" if got == want else "BROKEN"
        ok = ok and mark == "OK"
        print(f"  {label}  head {head[:7]} tip {tip[:7]}  "
              f"unpublished {len(rows)}  -> {got:6s} (want {want})  {mark}")
        for r in rows:
            print(f"        {r}")
    print("  the control WORKS" if ok else "  ⛔ THE CONTROL IS BROKEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
