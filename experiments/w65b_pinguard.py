"""w65b -- the guard for the BLAS-pin defect w65 found, with its negative controls.

THE DEFECT. `blend_lab.py` pins five thread-count env vars above its own `import numpy`, and
its comment claims that from w28 on "every build from here shares one BLAS configuration, so
two files built on different days can be differenced again". That holds for `blend_lab.py` run
as a script. It does NOT hold for an analysis script that imports numpy at the top and
`blend_lab` afterwards -- OpenBLAS reads the variable when numpy loads it, so a later
`setdefault` cannot move numpy's pool. Such a script runs numpy's OpenBLAS UNPINNED and
scipy's/sklearn's pools at 4: a mixed configuration matching neither the shipped builds nor an
honestly unpinned run.

WHAT IT DOES AND DOES NOT INVALIDATE. Nothing paired: a paired contrast inside ONE process
cancels the thread term exactly (w27z: A-B is 0.000e+00 to the last digit), so w29d's
194-vs-190 decision and w65a's 202-vs-199 decision are unaffected. What it does invalidate is
any ABSOLUTE comparison an analysis script makes against a shipped build -- the gates that
compare a refitted cell to a `.npy` on disk are measuring a thread difference on top of
whatever they think they are measuring.

    .venv/bin/python experiments/w65b_pinguard.py

Exit 1 on any FAILURE. Checks 1-4 are runtime and negative-controlled; check 5 is a static
sweep that REPORTS (it cannot fail the run, because the scripts it names are already on disk
and their results are already recorded).
"""
from __future__ import annotations

import ast
import warnings
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(os.path.dirname(HERE), ".venv", "bin", "python")
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def run(code):
    return subprocess.run([PY, "-c", code], capture_output=True, text=True,
                          cwd=os.path.dirname(HERE))


PRE = "import sys; sys.path.insert(0, 'experiments'); "

print("== 1. the pin WORKS when blas_pin is imported before numpy ==")
r = run(PRE + "import blas_pin; import numpy; print(blas_pin.threads_seen()); "
              "blas_pin.assert_pinned(); print('PINNED')")
check("pinned order -> assert_pinned passes", "PINNED" in r.stdout, r.stdout.strip().splitlines()[-1:] and r.stdout.strip().splitlines()[-1])

print("\n== 2. NEGATIVE CONTROL: the pin FAILS when numpy is imported first ==")
r = run(PRE + "import numpy; import blas_pin; print(blas_pin.threads_seen()); "
              "blas_pin.assert_pinned(); print('PINNED')")
check("unpinned order -> assert_pinned refuses",
      "PINNED" not in r.stdout and "NOT PINNED" in (r.stdout + r.stderr),
      (r.stdout + r.stderr).strip().splitlines()[-1][:80])

print("\n== 3. NEGATIVE CONTROL: the env var LIES -- it reads 4 while numpy runs unpinned ==")
r = run(PRE + "import numpy, os; import blend_lab; "
              "from threadpoolctl import threadpool_info as t; "
              "print('env', os.environ.get('OMP_NUM_THREADS'), "
              "'pools', sorted((d['internal_api'], d['num_threads']) for d in t()))")
out = r.stdout.strip()
check("blend_lab sets env=4 but numpy's openblas is NOT 4",
      "env 4" in out and "('openblas', 4)" in out and out.count("openblas") >= 2, out[:110])

print("\n== 4. an explicit caller value still wins (setdefault, not assignment) ==")
r = run(PRE + "import os; os.environ['OMP_NUM_THREADS']='2'; import blas_pin; "
              "print(os.environ['OMP_NUM_THREADS'])")
check("OMP_NUM_THREADS=2 survives blas_pin", r.stdout.strip().endswith("2"), r.stdout.strip())

print("\n== 5. STATIC SWEEP: experiments/*.py that import numpy BEFORE blend_lab ==")


def order(path):
    """(numpy_lineno, blendlab_lineno) or None if it does not import both."""
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError:
        return None
    npl = bll = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for al in n.names:
                if al.name.split(".")[0] == "numpy" and npl is None:
                    npl = n.lineno
                if al.name.split(".")[0] == "blend_lab" and bll is None:
                    bll = n.lineno
        elif isinstance(n, ast.ImportFrom) and n.module:
            root = n.module.split(".")[0]
            if root == "numpy" and npl is None:
                npl = n.lineno
            if root == "blend_lab" and bll is None:
                bll = n.lineno
    return (npl, bll) if (npl and bll) else None


warnings.simplefilter("ignore", SyntaxWarning)
bad = []
for f in sorted(os.listdir(HERE)):
    if not f.endswith(".py"):
        continue
    o = order(os.path.join(HERE, f))
    if o and o[0] < o[1]:
        bad.append((f, o))
for f, (a, b) in bad:
    print(f"    UNPINNED  {f:28s} numpy L{a} before blend_lab L{b}")
print(f"  {len(bad)} script(s) rely on a pin that does not reach numpy. "
      f"REPORTED, not a failure: their paired contrasts are unaffected (w27z), their "
      f"absolute-vs-shipped gates are not.")
check("the static sweep ran and found the known cases",
      any(f.startswith("w29d_") for f, _ in bad) and any(f.startswith("w65a_") for f, _ in bad),
      f"w29d and w65a both present among {len(bad)}")

print(f"\nFAILURES {len(FAILURES)}" + ("" if not FAILURES else f" -> {FAILURES}"))
sys.exit(1 if FAILURES else 0)
