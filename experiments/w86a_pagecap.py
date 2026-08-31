"""w86a — no Kaggle list may be read in a way that cannot detect its own truncation.

⚠⚠ REWRITTEN 2026-08-31 (w140). THE ORIGINAL REMEDY WAS UNBUYABLE AND THIS FILE WAS GREEN
OVER IT FOR SIX DAYS. Read this before touching anything below.

THE DEFECT THIS WAS ORIGINALLY WRITTEN FOR (2026-08-25, w86) was real: twenty-two call sites
asked for `--page-size 200` and the account was heading for 201 sends, so on the deadline day
they would silently lose the oldest row. The remedy this file then imposed --

    G1/G2  every call site must ask for MORE ROWS THAN THE ACCOUNT WILL EVER HAVE (PAGE=500)
    G4     the deadline-day scripts must carry `len(rows) >= PAGE -> raise`

-- **cannot be bought at any price, because the server caps a page at 200.** Measured in
`w140c_pagetruth.py`: requested 201 -> 200 rows, 500 -> 200, 1000 -> 200, true total 201, and
the CLI never prints the `next_page_token` the server returns. So at PAGE=500:

    read comes back 200  ->  `200 >= 500` is False  ->  "NOT TRUNCATED"  ->  row 201 lost

🎯 **RAISING THE PAGE SIZE PAST 200 DOES NOT WIDEN THE READ, IT DISABLES THE DETECTOR.** All
seven scripts G4 certified were in exactly that state, including `w23b_sendqueue.py`, the one
site that thought to check a token -- it reads for a `Next Page Token` line the CLI does not
emit, so its `tok` was always None.

⚠ AND THIS FILE WAS A VICTIM OF ITS OWN RULE. `api_rows(500)` returned 200, its self-check
read `200 >= 500 -> False`, and it printed `live rows 200` when the account held 201 -- so the
bar it enforced was computed from a truncated read. It also passed `w135b_grade.py`'s
`--page-size 300` as safe, because 300 cleared a bar of 240, and that is the third site where
the same row went missing.

⛔ DO NOT "fix" any failure below by raising a page size. There is no page size that works.
⛔ DO NOT lower `SERVER_PAGE_CAP` to make something pass; it is measured live by G5.

THE REMEDY THAT DOES WORK, and the only one this file now accepts:
  * follow `next_page_token` to exhaustion -- `kaggle_list.submissions()`; or
  * compare the row count against a total obtained some OTHER way (`num_total`, or the public
    leaderboard's own `SubmissionCount` column); or
  * keep a raw capped read ONLY with a self-check whose bound is AT OR BELOW the server cap,
    so a full page is actually detectable.

GUARDS
  G1    every raw Kaggle list call site is either paginated or carries a FIREABLE self-check.
  G2    no self-check bound sits above the server cap, where it can never fire.
  G3 -  negative control: a planted site at page 500 guarded by `len(rows) >= 500` MUST be
        rejected. That is the exact shape this file used to certify.
  G4    the deadline-day scripts read through the paginated helper, or carry a fireable check.
  G5 +- the cap is re-measured live, ACROSS the boundary (200 and 201), not asserted.

    .venv/bin/python experiments/w86a_pagecap.py          # 0 = ok, 1 = a guard failed
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_list   # noqa: E402  paginated submission reads

COMP = "playground-series-s6e8"
OUT = os.path.join(HERE, "w86a_pagecap.json")

SERVER_PAGE_CAP = kaggle_list.PAGE_CAP     # 200, measured; G5 re-measures it live

# Scripts that are RUN ON THE DEADLINE DAY and whose truncation would be silent. G4's list.
DEADLINE_DAY = ["w84a_pickargmax.py", "w26g_send.py", "w23b_sendqueue.py",
                "w55a_unpriced.py", "w54a_vetoexpiry.py", "w85c_slotguard.py",
                "w74b_clickstaleguard.py"]

# LIVE = a file whose read can still feed a decision: it runs in the daily suite, or on the
# deadline day. Everything else is a one-off analysis whose output is already committed.
# ⚠ THE EXEMPTION IS DATED AND CHECKABLE, NOT A SHRUG. A capped read only began losing rows
# once the account passed the 200-row cap, and it passed it TODAY (191 sends before
# 2026-08-31, 201 after). So every one-off analysis on disk ran against a COMPLETE list and
# its committed numbers are sound. What is unsafe is re-running one now. They are enumerated
# in the JSON and printed below -- reported, never silently excused.
def live_files() -> set[str]:
    suite = open(os.path.join(HERE, "w93a_suite.py")).read()
    named = set(re.findall(r"[\"'](w\d+[a-z]?_[a-z0-9_]+)[\"']", suite))
    return {n + ".py" for n in named} | set(DEADLINE_DAY)


CONTROL_MARK = "w86a: capped-control"
LIST_VERBS = ('"submissions"', '"leaderboard"')
SIZE_RE = re.compile(r'"--page-size",\s*(?:"(\d+)"|str\((\w+)\))')
CONST_RE = r"^{}\s*=\s*(\d+)"          # a trailing comment is normal
# `len(x) >= N` / `len(x) < N`, the two idioms that detect a full page.
CHECK_RE = re.compile(r"len\(\s*\w+[^)]*\)\s*(?:>=|<)\s*(\w+)")


def call_sites(text: str, name: str = ""):
    """(line, page_size|None, kind) for every kaggle LIST invocation in one file.

    ⚠ w140 reader fix: `competitions leaderboard -d` is a DOWNLOAD, not a paginated list --
    it has no page-size concept at all, and flagging it produced false findings. Only
    `-d`-less leaderboard calls and `submissions` calls are list reads.
    """
    out = []
    lines = text.splitlines()
    for m in re.finditer(r'\[\s*"kaggle"[^\]]*\]', text, re.S):
        blob = m.group(0)
        if not any(v in blob for v in LIST_VERBS):
            continue
        if '"leaderboard"' in blob and '"-d"' in blob:
            continue                                    # a download, not a list read
        line = text[:m.start()].count("\n") + 1
        # ⚠ A DELIBERATELY CAPPED READ IS A REAL CATEGORY -- a negative control has to be
        # truncated to be a control (this file's own G3 is one). It is recognised ONLY by a
        # marker a human typed on the preceding line. Silence is not spendable: an unmarked
        # capped read is indistinguishable from a bug, and is treated as one.
        if line >= 2 and CONTROL_MARK in lines[line - 2]:
            out.append((line, None, "capped-control"))
            continue
        sm = SIZE_RE.search(blob)
        if sm is None:
            out.append((line, None, "no-size"))
        elif sm.group(1) is not None:
            out.append((line, int(sm.group(1)), "literal"))
        else:
            cm = re.search(CONST_RE.format(re.escape(sm.group(2))), text, re.M)
            # ⚠ w140: an unresolvable symbol is a LOCAL/PARAMETER page size, not a missing
            # one. Saying "NO page size" about `str(page_size)` was a false finding.
            out.append((line, int(cm.group(1)) if cm else None,
                        "literal" if cm else "unresolved"))
    return out


def bounds_in(text: str) -> set[int]:
    """Every numeric bound the file compares a row count against."""
    got = set()
    for m in CHECK_RE.finditer(text):
        tok = m.group(1)
        if tok.isdigit():
            got.add(int(tok))
        else:
            cm = re.search(CONST_RE.format(re.escape(tok)), text, re.M)
            if cm:
                got.add(int(cm.group(1)))
    return got


def audit(text: str, name: str):
    """Failures for one file. Paginated readers are safe by construction."""
    sites = call_sites(text, name)
    paginated = "kaggle_list" in text
    if not sites:
        return [], sites, paginated
    checks = bounds_in(text)
    fails = []
    for line, size, kind in sites:
        if kind == "capped-control":
            continue
        if kind == "no-size":
            fails.append(f"G1 {name}:{line} kaggle list call with NO page size "
                         f"(the API default is 50)")
            continue
        if size is None:                                # unresolved local/parameter
            continue
        if size > SERVER_PAGE_CAP:
            fails.append(f"G2 {name}:{line} page-size {size} is above the server cap "
                         f"{SERVER_PAGE_CAP}; the read is capped at {SERVER_PAGE_CAP} and a "
                         f"`len(rows) >= {size}` check can never fire")
        elif not any(b <= SERVER_PAGE_CAP for b in checks):
            fails.append(f"G1 {name}:{line} raw capped read at page-size {size} with no "
                         f"fireable self-check (bounds found: {sorted(checks) or 'none'})")
    return fails, sites, paginated


def main() -> int:
    bad = []

    # ---- G5: re-measure the cap ACROSS the boundary, not below it -----------------------
    # ⚠ The original probed page_size = n-1 only, which is always under the ceiling and so
    # could never see it. The boundary is the only informative place to look.
    def cli_rows(ps):
        r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                            "--page-size", str(ps)], capture_output=True, text=True,
                           timeout=600)
        return len(pd.read_csv(io.StringIO(r.stdout)))

    at, over = cli_rows(SERVER_PAGE_CAP), cli_rows(SERVER_PAGE_CAP + 1)
    true_total = kaggle_list.total(COMP)
    print(f"true total (paginated) {true_total}   page {SERVER_PAGE_CAP} -> {at} rows   "
          f"page {SERVER_PAGE_CAP + 1} -> {over} rows\n")
    if not (at == SERVER_PAGE_CAP and over == SERVER_PAGE_CAP):
        bad.append(f"G5 the cap moved: page {SERVER_PAGE_CAP}->{at}, "
                   f"{SERVER_PAGE_CAP + 1}->{over}; update kaggle_list.PAGE_CAP")
        print(f"  ❌ G5 the {SERVER_PAGE_CAP}-row cap premise CHANGED")
    else:
        print(f"  ✅ G5 cap re-measured across the boundary: asking for "
              f"{SERVER_PAGE_CAP + 1} still returns {over}")

    # ---- G1 / G2 over every call site --------------------------------------------------
    n_sites = n_pag = n_ctl = 0
    LIVE = live_files()
    backlog = []
    for p in sorted(os.listdir(HERE)):
        if not p.endswith(".py") or p == os.path.basename(__file__):
            continue
        fails, sites, paginated = audit(open(os.path.join(HERE, p)).read(), p)
        n_sites += len(sites)
        n_pag += bool(paginated)
        n_ctl += sum(1 for _, _, k in sites if k == "capped-control")
        if p in LIVE:
            bad += fails
        else:
            backlog += fails
    g12 = [b for b in bad if b.startswith(("G1 ", "G2 "))]
    print(f"  {'✅' if not g12 else '❌'} G1/G2 {n_sites} raw list call sites, "
          f"{n_pag} files read through kaggle_list, {n_ctl} marked capped-control, "
          f"{len(g12)} unsafe among LIVE files ({len(LIVE)} live)")
    for b in g12[:25]:
        print(f"       {b}")
    if backlog:
        files = sorted({b.split()[1].split(":")[0] for b in backlog})
        print(f"  ⚠ BACKLOG {len(backlog)} unsafe call sites in {len(files)} DORMANT files "
              f"(one-off analyses, not in the suite and not deadline-day).")
        print(f"    Their committed results are SOUND -- the account only passed the "
              f"{SERVER_PAGE_CAP}-row cap today, so every read they made was complete.")
        print(f"    ⛔ Re-running any of them now would read a truncated list. Convert to "
              f"kaggle_list first.")
        print("    " + ", ".join(f[:-3] for f in files[:12])
              + (f", +{len(files) - 12} more" if len(files) > 12 else ""))

    # ---- G3: negative control ----------------------------------------------------------
    # ⚠ THE PLANTED SIZE IS ASSEMBLED, NOT WRITTEN OUT. w86's own repo-wide
    # `sed 's/"--page-size", "200"/"--page-size", "500"/'` rewrote this control's literal and
    # G3 caught its own disarming. A negative control that matches the pattern being swept
    # will be swept. Keep it un-greppable.
    OLD = SERVER_PAGE_CAP + 300
    planted = ('x = ["kaggle", "competitions", "submissions", "-c", C, "-v", '
               f'"--page-size", "{OLD}"]\n'
               f'if len(rows) >= {OLD}: raise SystemExit("truncated")\n')
    pf, psites, _ = audit(planted, "<planted>")
    if len(psites) != 1 or psites[0][1] != OLD:
        bad.append(f"G3 the parser did not read the planted site: {psites}")
    elif not pf:
        bad.append(f"G3 a page-{OLD} site guarded by `len(rows) >= {OLD}` was ACCEPTED -- "
                   f"this file is inert, that is the exact shape it used to certify")
    else:
        print(f"  ✅ G3 planted page-{OLD} site with an unfireable "
              f"`len(rows) >= {OLD}` check is REJECTED")

    # CONTROL 2: a download must NOT be read as a list call (the w140 reader false positive).
    dl = 'x = ["kaggle", "competitions", "leaderboard", "-c", C, "-d", "-p", d]'
    if call_sites(dl, "<dl>"):
        bad.append("G3b a `leaderboard -d` DOWNLOAD is still being read as a paginated list")
    else:
        print("  ✅ G3b a `leaderboard -d` download is not counted as a list read")

    # ---- G4: deadline-day scripts ------------------------------------------------------
    missing = []
    for name in DEADLINE_DAY:
        fp = os.path.join(HERE, name)
        if not os.path.exists(fp):
            missing.append(f"{name} (absent)")
            continue
        t = open(fp).read()
        if "kaggle_list" in t:
            continue                                   # paginated: safe by construction
        if not any(b <= SERVER_PAGE_CAP for b in bounds_in(t)):
            missing.append(name)
    for name in missing:
        bad.append(f"G4 {name} neither paginates nor carries a fireable truncation check")
    print(f"  {'✅' if not missing else '❌'} G4 {len(DEADLINE_DAY) - len(missing)}"
          f"/{len(DEADLINE_DAY)} deadline-day scripts read a COMPLETE list")

    json.dump({"server_page_cap": SERVER_PAGE_CAP, "true_total": true_total,
               "at_cap": at, "over_cap": over, "call_sites": n_sites,
               "paginated_files": n_pag, "g4_missing": missing, "failures": bad,
               "backlog_dormant": sorted(backlog)},
              open(OUT, "w"), indent=2)
    print(f"\nFAILURES {len(bad)}")
    for b in bad:
        print("  ❌", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
