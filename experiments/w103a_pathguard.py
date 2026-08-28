"""w103a — STANDING GUARD: A `command not found` HERE MEANS "NOT ON MY PATH", NEVER "NOT ON THE BOX".

WHAT CLAIM THIS COVERS, AND WHY NOTHING ELSE COVERS IT. The agent shell in this workspace runs
with a curated nix-store PATH that OMITS `/run/current-system/sw/bin` — the directory where every
NixOS system package on this machine actually lives. Nothing announces that. A run types
`setsid ...`, gets `setsid: command not found`, and writes down the only conclusion the evidence
seems to support: the tool is not installed. It is installed. It is 30 characters away.

⚠⚠ THIS IS NOT HYPOTHETICAL AND IT IS NOT CHEAP. Three separate durable claims in RESEARCH.md
were wrong for exactly this reason, and each one cost real work:

  * `setsid is NOT INSTALLED in this sandbox`      → the whole systemd-user launch recipe was
                                                     built to route around a tool that exists.
  * `pgrep is not installed here, like ps and at`  → w102 §1 found a build dead and concluded
                                                     "the log cannot tell you whether it crashed
                                                     or was killed". With `ps` the answer is one
                                                     command. A lost build was diagnosed as
                                                     undiagnosable.
  * `gh ... is not installed`                      → w102's push failed with `could not read
                                                     Username`, read as an auth/token problem.

🎯 THE SHAPE, WHICH IS THE PART WORTH KEEPING: **an absence is the one observation a broken
lookup path cannot distinguish from a presence it is blind to.** Every other kind of wrong answer
looks wrong. `command not found` looks exactly like the truth, arrives with the authority of the
shell, and is then written into a durable document as a fact about the machine rather than a fact
about the environment the probe ran in. It also propagates: two of the three claims above went on
to justify architecture (the systemd recipe) or to close an investigation (w102 §1).

⚠ THE EXPENSIVE DIRECTION IS THE SAME ONE THE ANGLE INDEX HAS: a false absence retires a route.
w97/w99 concluded "no browser binary of any kind on disk" and closed the final-selection click as
human-only. That conclusion was taken under this exact defect. w103 re-took it with the corrected
PATH — still no browser, so the closure stands — but it stood on evidence that had already been
shown unreliable everywhere else it was used, and nobody had noticed.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +   every tool in TOOLS resolves under the CORRECTED path. A tool that does not is genuinely
         absent, and RESEARCH's positive claim about it is wrong — the opposite error, equally
         durable, so it is fatal too.
  C2 +   NOT VACUOUS. At least MIN_TOOLS tools are probed and every one is actually looked up.
         A guard that iterates an empty list prints what a passing guard prints (w101 §3).
  C3 +-  FIRED BOTH WAYS. A name that cannot exist must resolve NOWHERE, and a name that does
         exist must resolve somewhere, using the same resolver. If a nonsense name comes back
         found, the resolver is broken and C1's greens mean nothing.
  C4 +-  THE HAZARD ITSELF, IN CODE, NOT AS PROSE. At least one tool must resolve under the
         corrected path and NOT under the naive one. That difference IS the defect. If it ever
         drops to zero the hazard is gone (or the two paths are being built wrong) — either way
         a human must look, so it goes RED rather than quietly green. Retiring this guard is a
         decision, not a drift.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "w103a_pathguard.json")

# The directory the agent shell's PATH omits. Everything below is about this one string.
SYSBIN = "/run/current-system/sw/bin"

# Tools this workspace has recorded as ABSENT, plus the ones its launch/push recipes depend on.
# Each of these has a false "not installed" claim behind it somewhere in RESEARCH.md or JOURNAL.md.
TOOLS = ["setsid", "pgrep", "ps", "free", "gh", "nohup"]
MIN_TOOLS = 5

# Genuinely absent on this box even with SYSBIN on the path — kept so the list does not quietly
# become "things that happen to be installed". `at` was named in the same breath as ps/pgrep and
# is the one that was right.
KNOWN_ABSENT = ["at"]

FAILS = 0


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL {msg}")


def corrected_path() -> str:
    p = os.environ.get("PATH", "")
    return SYSBIN if not p else f"{SYSBIN}{os.pathsep}{p}"


def naive_path() -> str:
    """What a fresh agent shell sees. SYSBIN is stripped in case a caller already exported it,
    so the comparison measures the defect and not this process's own environment."""
    parts = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and d != SYSBIN]
    return os.pathsep.join(parts)


def main() -> int:
    corrected, naive = corrected_path(), naive_path()
    print(f"SYSBIN {SYSBIN}  exists={os.path.isdir(SYSBIN)}")
    print(f"naive PATH contains SYSBIN: {SYSBIN in naive.split(os.pathsep)} (want False)\n")

    # ---- C1 +: the positive claim. Each of these IS on the box.
    found_corrected, found_naive, probed = {}, {}, 0
    for t in TOOLS:
        c = shutil.which(t, path=corrected)
        n = shutil.which(t, path=naive)
        found_corrected[t], found_naive[t] = c, n
        probed += 1
        flag = "" if c else "   <-- genuinely absent"
        print(f"C1 {t:<8} corrected={c or '-'}{flag}")
        print(f"   {'':<8} naive    ={n or '-'}")
        if not c:
            fail(f"C1: {t} does not resolve even with {SYSBIN} on PATH — it is genuinely "
                 f"absent, and any RESEARCH claim that it exists is wrong")

    # ---- C2 +: not vacuous.
    print(f"\nC2 probed {probed} tools (floor {MIN_TOOLS})")
    if probed < MIN_TOOLS or probed != len(TOOLS):
        fail(f"C2: probed {probed} of {len(TOOLS)}, floor {MIN_TOOLS} — a guard that checks "
             f"nothing prints what a guard that checks everything prints")

    # ---- C3 +-: the resolver itself, both directions.
    nonsense = "w103a_no_such_binary_zz_do_not_create"
    bad_c = shutil.which(nonsense, path=corrected)
    bad_n = shutil.which(nonsense, path=naive)
    # TOOLS can be empty only when C2 has already failed; do not crash on top of that report.
    real = TOOLS[0] if TOOLS else "sh"
    good = shutil.which(real, path=corrected)
    print(f"C3 nonsense name  corrected={bad_c or '-'}  naive={bad_n or '-'}  (want both '-')")
    print(f"C3 real name      {real}={good or '-'}  (want a path)")
    if bad_c or bad_n:
        fail("C3: a name that cannot exist resolved — the resolver reports success "
             "unconditionally, so every C1 green above is meaningless")
    if not good:
        fail("C3: the resolver found nothing for a tool C1 just located, so the two halves "
             "of this guard disagree and neither can be trusted")

    for t in KNOWN_ABSENT:
        if shutil.which(t, path=corrected):
            print(f"C3 note: {t} is now present; it was recorded absent. Not a failure — "
                  f"update the note in RESEARCH so the list stays honest.")

    # ---- C4 +-: THE DEFECT, MEASURED. Tools reachable only via SYSBIN.
    only_via_sysbin = [t for t in TOOLS if found_corrected[t] and not found_naive[t]]
    print(f"\nC4 reachable ONLY via {SYSBIN}: {len(only_via_sysbin)} {only_via_sysbin}")
    if not only_via_sysbin:
        fail("C4: no tool is hidden by the naive PATH. Either the agent PATH now includes "
             f"{SYSBIN} — in which case this hazard is GONE and this guard plus its RESEARCH "
             "section must be retired DELIBERATELY, not left to rot — or the two paths are "
             "being constructed wrongly and C4 has stopped testing anything.")

    json.dump(dict(day=dt.datetime.now(dt.timezone.utc).date().isoformat(),
                   sysbin=SYSBIN, sysbin_exists=os.path.isdir(SYSBIN),
                   probed=probed, only_via_sysbin=only_via_sysbin,
                   corrected={k: v for k, v in found_corrected.items()},
                   naive={k: v for k, v in found_naive.items()},
                   failures=FAILS), open(OUT, "w"), indent=1)

    print(f"\nFAILURES {FAILS}")
    if FAILS == 0:
        print(f"OK: {len(only_via_sysbin)} of {len(TOOLS)} tools are invisible to a bare "
              f"`command -v`. A 'command not found' here is a statement about PATH.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
