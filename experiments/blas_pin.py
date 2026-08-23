"""Pin the BLAS/OpenMP thread count. IMPORT THIS BEFORE numpy, OR IT DOES NOTHING.

w28 added the same five `os.environ.setdefault` calls to the TOP of `blend_lab.py`, above its
`import numpy`, with the note: "every build from here shares one BLAS configuration, so two
files built on different days can be differenced again." That is true for `blend_lab.py` run as
a SCRIPT, because there the pin executes before numpy loads OpenBLAS.

⚠ IT IS FALSE FOR EVERY ANALYSIS SCRIPT THAT DOES `import numpy as np` AT THE TOP AND
`from blend_lab import load_all` AFTERWARDS -- which is the house style and includes
`w29d_partition.py`, the instrument that decided the 194-vs-190 promotion. OpenBLAS reads
OMP_NUM_THREADS when numpy loads it; setting the variable later cannot move it. Measured
2026-08-23 (w65) in that exact order:

    before blend_lab import:  [('openblas', 16)]
    env after  blend_lab import:  OMP_NUM_THREADS=4  OPENBLAS_NUM_THREADS=4
    after  blend_lab import:  [('openblas', 16), ('openblas', 4), ('openmp', 4)]

numpy's OpenBLAS stays at 16. The 4-thread entries are scipy's and sklearn's own copies, loaded
after the variable was set -- so the process runs a MIXED configuration that matches neither the
shipped builds (all 4) nor an unpinned run (all 16).

Usage, as the FIRST import of the module, above numpy:

    from blas_pin import pin; pin()          # or just `import blas_pin` -- pin() runs on import

`assert_pinned()` verifies it actually took, and is what a guard should call rather than
trusting the env var, because the env var is exactly the thing that lies here.
"""
from __future__ import annotations

import os
import sys

VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DEFAULT = "4"          # blend_lab's value. Do not change it without rebuilding the pack.


def pin(n: str = DEFAULT) -> bool:
    """setdefault (never assignment -- an explicit env var from the caller wins).

    Returns False if numpy is ALREADY imported, in which case the call is a no-op on
    numpy's own OpenBLAS and the caller is in the failure mode this module documents.
    """
    already = "numpy" in sys.modules
    for v in VARS:
        os.environ.setdefault(v, n)
    return not already


def threads_seen():
    """The thread counts the loaded native libraries ACTUALLY report, or None."""
    try:
        from threadpoolctl import threadpool_info
    except Exception:
        return None
    return sorted((d["internal_api"], d["num_threads"]) for d in threadpool_info())


def assert_pinned(n: int = int(DEFAULT)):
    """Raise unless every loaded native pool is at `n`. Returns what it saw."""
    seen = threads_seen()
    if seen is None:
        raise SystemExit("blas_pin: threadpoolctl is not installed -- cannot verify the pin")
    bad = [s for s in seen if s[1] != n]
    if bad:
        raise SystemExit(f"blas_pin: NOT PINNED -- pools at {seen}, expected every one at {n}. "
                         f"Import blas_pin BEFORE numpy.")
    return seen


pin()
