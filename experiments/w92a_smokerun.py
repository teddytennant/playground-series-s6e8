"""w92a — EXECUTE the instruments the suite only ever type-checked.

WHY THIS EXISTS (w92, 2026-08-26).
w91 found seven modules broken by the submissions-timestamp format, patched all seven, and
shipped `w91b_dateguard` — which proves the *parse* is right in every one of them. It ran
none of them. Today the first one executed, `w28a_cvlb_refresh`, died two lines below the
patched line on a SECOND, independent break: it re-derived a frozen model's centring from
`w25a_cvlb_full.csv`, a registry w30 legitimately refreshed on 2026-08-22, so the module had
been dead since 10:12 that day and the date repair could never have revived it.

That is w91's own lesson arriving one level down. A guard is evidence about the property it
checks. The only evidence that a module WORKS is running it.

WHAT IT DOES.
  G1  every runnable module in the repair surface exits 0.
  G2  every module excluded from G1 is excluded BY A STATED RULE and still exists on disk,
      so a rename cannot silently shrink the set into vacuous success.
  G3  COVERAGE. The surface is not a hardcoded list — it is derived live, as every module
      that imports `w91a_subdate.parse_sub_dates`, and G3 asserts that set equals
      RUN u EXCLUDED. w89b's lesson: the hole is in the QUERY, not in the checker. A module
      that starts reading the submissions frame tomorrow fails G3 until it is classified.
  G4  CONTROL-. A planted module that exits 1 is reported as a failure by the same runner.

⛔ `w25a_cvlb_full.py` IS EXCLUDED AND MUST STAY EXCLUDED. It rewrites `w25a_cvlb_full.csv`,
which is the source `w57c_muguard` and `w75b_muguard` pin the LIVE pricer's MU against.
Running it here would re-centre the pricer as a side effect of a health check.

    .venv/bin/python experiments/w92a_smokerun.py        # 0 = ok, 1 = a gate failed
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 900

# The modules that read the submissions frame and are safe to execute as a health check.
RUN = [
    "w28a_cvlb_refresh.py",
    "w50b_autoselect.py",
    "w75a_erarefresh.py",
    "w15i_cvlb.py",
    "w15j_cvlb.py",
    "w93b_cvlbaudit.py",     # added w93: the live CV->LB gap audit. RUN, not excluded — it is
                             # read-only by construction (writes only its own JSON) and its
                             # own G1-G4 are what make it worth executing rather than trusting.
]

# Excluded, each with the rule that excludes it. Not a convenience list.
EXCLUDED = {
    "w25a_cvlb_full.py":
        "rewrites w25a_cvlb_full.csv, the source w57c/w75b pin the LIVE pricer's MU against; "
        "running it would re-centre the pricer as a side effect of a health check",
    "w15j_tiebreak.py":
        "w15-era one-off, superseded in full by w50b_autoselect, and it reads /tmp/w15j_subs.csv "
        "— a scratch path, not a repo artefact. Repairing it would be a second owner of the "
        "auto-selection enumeration",
    # the helper and its guard are the repair itself, not modules under repair
    "w91a_subdate.py": "the helper that owns the format",
    "w91b_dateguard.py": "the guard that proves the helper is called; itself a standing check",
}


def _run(path):
    r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=TIMEOUT)
    return r.returncode, (r.stderr.strip().splitlines() or [""])[-1]


def surface():
    """Every module that imports parse_sub_dates. Derived live, never hardcoded."""
    out = set()
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(".py"):
            continue
        try:
            tree = ast.parse(open(os.path.join(HERE, fn)).read())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == "w91a_subdate":
                out.add(fn)
            elif isinstance(n, ast.Import) and any(a.name == "w91a_subdate" for a in n.names):
                out.add(fn)
        if fn in ("w91a_subdate.py", "w91b_dateguard.py"):
            out.add(fn)
    return out


def main():
    fails = []
    print("w92a — the repair surface, executed\n")

    # ---- G3 first: if the set is wrong, G1 is measuring the wrong thing ------------------
    live, claimed = surface(), set(RUN) | set(EXCLUDED)
    missing, stale = live - claimed, claimed - live
    print(f"G3  surface {len(live)} module(s) import w91a_subdate; classified {len(claimed)}")
    if missing:
        fails.append(f"G3: NOT CLASSIFIED — {sorted(missing)} read the submissions frame and "
                     f"are neither run nor excluded here")
    if stale:
        fails.append(f"G3: classified but gone — {sorted(stale)}")

    # ---- G2: the exclusions are real files ----------------------------------------------
    for fn, why in EXCLUDED.items():
        ok = os.path.exists(os.path.join(HERE, fn))
        print(f"G2  excluded {fn:<24s} {'exists' if ok else 'MISSING'}  — {why[:58]}")
        if not ok:
            fails.append(f"G2: {fn} is excluded but no longer exists; the exclusion is vacuous")

    # ---- G1: run them --------------------------------------------------------------------
    print()
    for fn in RUN:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            fails.append(f"G1: {fn} does not exist")
            print(f"G1  {fn:<24s} MISSING")
            continue
        rc, last = _run(p)
        print(f"G1  {fn:<24s} rc={rc}" + ("" if rc == 0 else f"   {last[:96]}"))
        if rc != 0:
            fails.append(f"G1: {fn} exits {rc} — {last[:160]}")

    # ---- G4: CONTROL- --------------------------------------------------------------------
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("import sys\nsys.stderr.write('planted\\n')\nsys.exit(1)\n")
        planted = fh.name
    try:
        rc, _ = _run(planted)
    finally:
        os.remove(planted)
    print(f"\nG4  CONTROL-: planted failing module reported rc={rc} (want 1)")
    if rc != 1:
        fails.append("G4: the runner does not detect a failing module")

    for f in fails:
        print(f"  FAIL  {f}")
    print(f"\nFAILURES {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
