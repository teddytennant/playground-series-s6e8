"""w93a — run the standing check suite from ONE list, instead of retyping it every run.

WHY THIS EXISTS. RESEARCH.md holds the 35 stems under "THE 35 STANDING CHECKS, FULL STEMS —
COPY THESE, DO NOT RECONSTRUCT THEM", and that warning is there because w88 expanded five of
them from memory and got all five wrong: `.venv/bin/python experiments/<wrong>.py` exits 2
with `No such file`, which reads exactly like a guard failure. A hand-typed loop reproduces
that hazard every run. This module holds the list once and refuses to run a stem that is not
on disk, so a typo is an ERROR line, never a silent FAIL.

⚠ THIS IS A RUNNER, NOT A GUARD. It asserts nothing about the competition. It executes the
guards and reports their exit codes, and its own exit code is 1 iff one of them failed.

  C1  every stem in STEMS resolves to a file on disk. A missing file is reported as MISSING
      and is fatal, distinct from a check that ran and failed.
  C2  the list here matches the list published in RESEARCH.md, parsed live out of the
      document. Neither copy is allowed to drift from the other silently.

  🔴 w54a_vetoexpiry and w85c_slotguard FAIL BY DESIGN when the send queue on disk is for a
     day already sent. That is the freshness guard working. Rebuild and re-run those two:
         .venv/bin/python experiments/w23b_sendqueue.py
         .venv/bin/python experiments/w48e_order.py --day <next UTC day> --write

    .venv/bin/python experiments/w93a_suite.py            # 0 = all green, 1 = a check failed
    .venv/bin/python experiments/w93a_suite.py --only w84a_pickargmax,w57c_muguard
"""
from __future__ import annotations

import os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, ".venv", "bin", "python")
RESEARCH = os.path.join(ROOT, "RESEARCH.md")

# The 36, verbatim from RESEARCH.md. C2 re-parses that document and compares.
STEMS = [
    "w54a_vetoexpiry", "w55a_unpriced", "w56b_wantedguard", "w57c_muguard", "w59b_barguard",
    "w60b_ineligguard", "w60d_memberguard", "w62b_barstaleguard", "w63b_setguard",
    "w64b_hedgeguard", "w65b_pinguard", "w65c_subsetcheck", "w66d_rangeguard",
    "w67b_slopeguard", "w68b_floorguard", "w70b_basisguard", "w70d_chainguard",
    "w71b_dupguard", "w72b_dayguard", "w74b_clickstaleguard", "w75b_muguard", "w76b_addguard",
    "w77b_bracketguard", "w78b_treatguard", "w79b_fillguard", "w80f_packguard",
    "w82a_pricecal", "w84a_pickargmax", "w85c_slotguard", "w86a_pagecap",
    "w87a_registrarguard", "w88a_calexposure", "w89a_foldid", "w91b_dateguard",
    "w92a_smokerun", "w93c_pickverify",
]

# Fails by design after the day's send until the queue is rebuilt (RESEARCH, w85/w92 §7).
POST_SEND_EXPECTED = {"w54a_vetoexpiry", "w85c_slotguard"}

TIMEOUT = 900          # w65c and w66d need ~4 min; w92a executes five real instruments.


def research_stems():
    """The published list, parsed out of RESEARCH.md so the two copies cannot drift (C2)."""
    txt = open(RESEARCH, encoding="utf-8").read()
    i = txt.find("STANDING CHECKS, FULL STEMS")
    if i < 0:
        return None
    block = txt[i:txt.find("\n⚠", i)]
    # ONLY the 4-space-indented table rows. The section also carries prose that names
    # modules (w93 added a paragraph naming this very runner) and a prose mention is not a
    # standing check -- w91b's own bug, in a different file: a MENTION is not a USE.
    rows = [l for l in block.splitlines() if l.startswith("    ") and l.strip()
            and not l.strip().startswith(("#", "`", ".venv", "*", "-"))]
    return set(re.findall(r"\bw\d+[a-z]_[a-z0-9]+\b", "\n".join(rows)))


def main() -> int:
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--only"):
            only = set(a.split("=", 1)[1].split(",")) if "=" in a else None
    if only is None and "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    stems = [s for s in STEMS if only is None or s in only]

    print(f"w93a suite — {len(stems)} checks, {TIMEOUT}s each, python {PY}")

    # C2 -- the two copies of the list agree.
    pub = research_stems()
    if pub is None:
        print("⛔ C2 could not locate the stem block in RESEARCH.md")
        return 1
    if pub != set(STEMS):
        print(f"⛔ C2 DRIFT  only-here {sorted(set(STEMS) - pub)}  only-RESEARCH {sorted(pub - set(STEMS))}")
        return 1
    print(f"C2 list matches RESEARCH.md ({len(pub)} stems)  PASS")

    # C1 -- resolve every stem before running anything, so a typo is never a FAIL line.
    missing = [s for s in stems if not os.path.exists(os.path.join(HERE, s + ".py"))]
    if missing:
        print(f"⛔ C1 MISSING ON DISK: {missing}  — a typo reads like a guard failure. Fix the list.")
        return 1
    print(f"C1 all {len(stems)} stems resolve on disk  PASS")

    # w65b_pinguard reads the thread environment and goes red under an exported shell.
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

    results, t_all = [], time.time()
    for n, stem in enumerate(stems, 1):
        t0 = time.time()
        try:
            p = subprocess.run([PY, os.path.join(HERE, stem + ".py")], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            rc, tail = p.returncode, (p.stdout or "").strip().splitlines()
        except subprocess.TimeoutExpired:
            rc, tail = 124, ["TIMEOUT"]
        dt = time.time() - t0
        note = "  (expected post-send; rebuild the queue)" if rc and stem in POST_SEND_EXPECTED else ""
        print(f"[{n:2d}/{len(stems)}] {stem:24s} rc={rc}  {dt:6.1f}s"
              f"  {tail[-1][:90] if tail else ''}{note}", flush=True)
        results.append((stem, rc, dt))

    bad = [(s, rc) for s, rc, _ in results if rc != 0]
    print(f"\nTOTAL {time.time() - t_all:.0f}s   {len(results) - len(bad)}/{len(results)} green")
    if bad:
        print("FAILURES: " + "  ".join(f"{s}(rc={rc})" for s, rc in bad))
        if all(s in POST_SEND_EXPECTED for s, _ in bad):
            print("⚠ every failure is the post-send queue-freshness guard doing its job. "
                  "Rebuild the queue for the next unsent day and re-run those two.")
        return 1
    print("FAILURES: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
