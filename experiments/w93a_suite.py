"""w93a — run the standing check suite from ONE list, instead of retyping it every run.

WHY THIS EXISTS. RESEARCH.md holds the stems under "THE N STANDING CHECKS, FULL STEMS —
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

# Verbatim from RESEARCH.md; C2 re-parses that document and compares. ⚠ NO COUNT IN THIS
# COMMENT ON PURPOSE — it said "the 45" while the list held 46, which is exactly the rot
# the header in RESEARCH.md is matched by SHAPE (`^## THE \d+ ...`) to avoid.
STEMS = [
    "w54a_vetoexpiry", "w55a_unpriced", "w56b_wantedguard", "w57c_muguard", "w59b_barguard",
    "w60b_ineligguard", "w60d_memberguard", "w62b_barstaleguard", "w63b_setguard",
    "w64b_hedgeguard", "w65b_pinguard", "w65c_subsetcheck", "w66d_rangeguard",
    "w67b_slopeguard", "w68b_floorguard", "w70b_basisguard", "w70d_chainguard",
    "w71b_dupguard", "w72b_dayguard", "w74b_clickstaleguard", "w75b_muguard", "w76b_addguard",
    "w77b_bracketguard", "w78b_treatguard", "w79b_fillguard", "w80f_packguard",
    "w82a_pricecal", "w84a_pickargmax", "w85c_slotguard", "w86a_pagecap",
    "w87a_registrarguard", "w88a_calexposure", "w89a_foldid", "w91b_dateguard",
    "w92a_smokerun", "w93c_pickverify", "w100a_complement", "w101a_angleguard",
    "w103a_pathguard", "w104a_cgroupguard", "w105a_liveguard", "w106a_claimguard",
    "w107a_lineref", "w109b_colguard", "w110b_covguard", "w111b_baseguard",
    "w112a_templateguard", "w114b_selectguard",
    "w115a_docselectguard",
    "w117a_handcount",
    "w122a_slotguard",
    "w123b_groupguard",
    "w124b_priceunitguard",
    "w125b_layerguard",
    "w126c_scopeguard",
    "w127b_rateguard",
    "w128b_pricedguard",
    "w129b_optoutguard",
    "w130b_zerobaselineguard",
    "w131b_selfbaselineguard",
    "w132b_significanceguard",
    "w152a_listshape",
    "w153a_openreadguard",
    "w154a_authscope",
    "w155a_poolguard",
    "w156a_recordguard",
    "w157a_closedguard",
]

# Fails by design after the day's send until the queue is rebuilt (RESEARCH, w85/w92 §7).
POST_SEND_EXPECTED = {"w54a_vetoexpiry", "w85c_slotguard"}

TIMEOUT = 900          # w65c and w66d need ~4 min; w92a executes five real instruments.


def research_stems():
    """The published list, parsed out of RESEARCH.md so the two copies cannot drift (C2).

    ⚠ THE LOCATOR MUST SKIP THE ANGLE INDEX, AND w110 FOUND OUT WHY THE HARD WAY. This used
    `txt.find("STANDING CHECKS, FULL STEMS")`, i.e. FIRST occurrence. On 2026-08-28 the ANGLE
    INDEX gained a row whose pointer is the backticked string `STANDING CHECKS, FULL STEMS` --
    and the index sits ABOVE the real block, so the locator landed on the POINTER, read the
    prose between it and the next '⚠', found no indented rows, and reported C2 DRIFT on all 45
    stems with `only-RESEARCH []`. 🎯 A LOCATOR THAT TAKES THE FIRST OCCURRENCE OF A STRING
    FINDS THE POINTER, NOT THE TARGET, THE MOMENT ANYONE WRITES A POINTER -- and writing
    pointers to this block is exactly what the index is for. Anchoring on the real header text
    (`## THE`) is not enough on its own either, because the count in it moves 44 -> 45 -> 46; so
    excise the index span first and then match the header shape.
    """
    txt = open(RESEARCH, encoding="utf-8").read()
    k = txt.find("# 📇 THE ANGLE INDEX")
    if k >= 0:                                  # drop the index block from the search corpus
        e = txt.find("\n# ", txt.find("\n|", k))
        txt = txt[:k] + txt[(len(txt) if e < 0 else e):]
    m = re.search(r"^## THE (\d+) STANDING CHECKS, FULL STEMS", txt, re.M)
    if m is None:
        return None
    claimed = int(m.group(1))
    i = m.start()
    block = txt[i:txt.find("\n⚠", i)]
    # ONLY the 4-space-indented table rows. The section also carries prose that names
    # modules (w93 added a paragraph naming this very runner) and a prose mention is not a
    # standing check -- w91b's own bug, in a different file: a MENTION is not a USE.
    rows = [l for l in block.splitlines() if l.startswith("    ") and l.strip()
            and not l.strip().startswith(("#", "`", ".venv", "*", "-"))]
    return set(re.findall(r"\bw\d+[a-z]_[a-z0-9]+\b", "\n".join(rows))), claimed


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
    got = research_stems()
    if got is None:
        print("⛔ C2 could not locate the stem block in RESEARCH.md")
        return 1
    pub, claimed = got
    if pub != set(STEMS):
        print(f"⛔ C2 DRIFT  only-here {sorted(set(STEMS) - pub)}  only-RESEARCH {sorted(pub - set(STEMS))}")
        return 1
    # C2b -- the HEADING's count too. It is hand-maintained prose and nothing checked it: on
    # 2026-08-30 the list held 54 stems under a heading reading 53, because w125 added its
    # stem to the table and not to the number above it. A published count that disagrees with
    # the published list is the same defect class as w117's handing counts, one document up.
    if claimed != len(STEMS):
        print(f"⛔ C2b COUNT DRIFT  the heading claims {claimed} standing checks; the list and "
              f"STEMS both hold {len(STEMS)}. Fix the number in the heading.")
        return 1
    print(f"C2 list matches RESEARCH.md ({len(pub)} stems, heading says {claimed})  PASS")

    # C1 -- resolve every stem before running anything, so a typo is never a FAIL line.
    missing = [s for s in stems if not os.path.exists(os.path.join(HERE, s + ".py"))]
    if missing:
        print(f"⛔ C1 MISSING ON DISK: {missing}  — a typo reads like a guard failure. Fix the list.")
        return 1
    print(f"C1 all {len(stems)} stems resolve on disk  PASS")

    # w65b_pinguard reads the thread environment and goes red under an exported shell.
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

    results, fails_logged, t_all = [], [], time.time()
    for n, stem in enumerate(stems, 1):
        t0 = time.time()
        try:
            p = subprocess.run([PY, os.path.join(HERE, stem + ".py")], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            rc, tail = p.returncode, (p.stdout or "").strip().splitlines()
        except subprocess.TimeoutExpired:
            rc, tail = 124, ["TIMEOUT"]
        dt = time.time() - t0
        # KEEP THE EVIDENCE. Until w104 this runner captured stdout and stderr and then printed
        # ONE 90-character line of stdout, discarding the rest -- so a red check's reason was
        # gone by the time anyone read the summary. w104 hit a w85c_slotguard red that did not
        # reproduce on three later runs and could not be diagnosed, because the output that
        # would have explained it had been thrown away. stderr was never shown at all, so a
        # check dying on a traceback displayed its last ordinary stdout line instead.
        if rc != 0:
            log = os.path.join(HERE, f"w93a_fail_{stem}.log")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write(f"# {stem} rc={rc} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
                fh.write("# ---- stdout ----\n" + (p.stdout or "") if hasattr(p, "stdout") else "")
                fh.write("\n# ---- stderr ----\n" + (getattr(p, "stderr", "") or ""))
            fails_logged.append(log)
        # Not "expected" yet -- whether this is the post-send pattern depends on w54a, which may
        # not have run. The verdict is in the summary; this line only says the stem is a
        # candidate. The old wording asserted the excuse before the evidence existed.
        note = "  (in the post-send set — see the summary)" if rc and stem in POST_SEND_EXPECTED else ""
        print(f"[{n:2d}/{len(stems)}] {stem:24s} rc={rc}  {dt:6.1f}s"
              f"  {tail[-1][:90] if tail else ''}{note}", flush=True)
        results.append((stem, rc, dt))

    bad = [(s, rc) for s, rc, _ in results if rc != 0]
    print(f"\nTOTAL {time.time() - t_all:.0f}s   {len(results) - len(bad)}/{len(results)} green")
    if bad:
        print("FAILURES: " + "  ".join(f"{s}(rc={rc})" for s, rc in bad))
        for log in fails_logged:
            print(f"  full output: {log}")
        # ⚠ Only claim "expected post-send" when the freshness guard the story rests on ACTUALLY
        # failed. w85c inherits its G4 from w54a, so w85c red WITHOUT w54a red is not the
        # post-send pattern -- it is something else, and w104 was sent looking in the wrong place
        # by this note firing on stem membership alone.
        names = {s for s, _ in bad}
        if names and names <= POST_SEND_EXPECTED and "w54a_vetoexpiry" in names:
            print("⚠ every failure is the post-send queue-freshness guard doing its job. "
                  "Rebuild the queue for the next unsent day and re-run those two.")
        elif names & POST_SEND_EXPECTED and "w54a_vetoexpiry" not in names:
            print("⛔ w85c is red but w54a is GREEN, so this is NOT the post-send freshness "
                  "pattern (w85c's G4 runs w54a). Read the saved output above before assuming "
                  "it is expected. If it does not reproduce, say so rather than closing it.")
        return 1
    print("FAILURES: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
