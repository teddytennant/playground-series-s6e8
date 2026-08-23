"""w68b -- pin R-M11f: every transform in agent/stack.py is STRICTLY PER-COLUMN.

w68a proved by direct comparison that the subsetted 202 matrix is BITWISE equal to a natively
loaded 199 pack, which is what licenses w29d_partition.py and w65a_armpair.py to serve every
arm from one load. That proof cost two full loads (~3 min) and 691,369 rows. The PROPERTY it
rests on is checkable in milliseconds on a synthetic matrix, so it is checked on every run
instead of once: if anyone ever adds a cross-member step to a transform -- a global
quantile, a whole-matrix normaliser, an imputation that borrows from neighbouring members --
the arm instruments silently start comparing arms that are not the arms they name, and the
190-over-188, 194-over-190 and 202-over-199 decisions all inherit it.

⚠ This guards the DATA half of R-M11f only. The FIT half (that lbfgs returns the same answer
on the same bytes) is w68a's P6 and is a within-process claim; ACROSS processes it is false
by 5.095e-6 and that is the whole point of w68 -- see JOURNAL w68 section 3.

    .venv/bin/python experiments/w68b_floorguard.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from stack import transform  # noqa: E402

KINDS = ("logit", "rankraw", "hybrid", "rescale")
FAILS = []


def ck(name, cond):
    print(f"  {'ok  ' if cond else '*** '}{name}")
    if not cond:
        FAILS.append(name)


def main():
    rng = np.random.default_rng(68)
    n, m, nt = 400, 12, 150
    # Probabilities, with a plateau so `hybrid`'s repair branch is actually exercised, and
    # with out-of-unit values so the `bad` mask is non-trivial.
    O = rng.random((n, m))
    T = rng.random((nt, m))
    O[:40, 3] = 1.0          # clipping plateau
    T[:10, 3] = 1.0
    O[:25, 7] = 0.0
    O[:5, 9] = 1.02          # outside the unit interval
    T[:3, 9] = -0.013
    O[:18, 5] = 1.0          # a clipping column that `keep` RETAINS, so the hybrid test
    T[:6, 5] = 1.0           # exercises a repaired column on BOTH sides of the subset
    keep = np.array([i for i in range(m) if i not in (3, 7, 9)])   # drop clipping columns too

    print("== R-M11f: transform(full)[:, keep] == transform(full[:, keep]) ==")
    for k in KINDS:
        A, At = transform(O, T, k)
        B, Bt = transform(O[:, keep], T[:, keep], k)
        ck(f"{k:8s} OOF  bitwise", np.array_equal(A[:, keep], B))
        ck(f"{k:8s} test bitwise", np.array_equal(At[:, keep], Bt))

    print("\n== the mask that makes the test non-vacuous ==")
    bad = (((O <= 0) | (O >= 1)).any(0) | ((T <= 0) | (T >= 1)).any(0))
    ck(f"hybrid repairs some columns (repaired {int(bad.sum())} of {m})", 0 < bad.sum() < m)
    ck("at least one REPAIRED column is dropped by keep",
       bool(set(np.flatnonzero(bad)) - set(keep.tolist())))
    ck("at least one REPAIRED column is kept by keep",
       bool(set(np.flatnonzero(bad)) & set(keep.tolist())))

    print("\n== the per-column std w65a/w68a apply after transform ==")
    print("  R-M11f's per-column argument does NOT extend to np.std(axis=0): that reduction")
    print("  is not bitwise invariant to the COLUMN COUNT (measured: 8.9e-16 float64,")
    print("  3.0e-7 float32 on a 12-vs-9 synthetic). w68a measured EXACT equality on the")
    print("  real 202/199 matrices -- an empirical fact about those matrices, not a theorem.")
    print("  This block pins the size of what is NOT guaranteed.")
    ULP32 = 1e-6
    for k in KINDS:
        A, _ = transform(O, T, k)
        A = A.astype("float32")
        s = A.std(0)
        s[s <= 0] = 1.0
        sub = A[:, keep]
        s2 = sub.std(0)
        s2[s2 <= 0] = 1.0
        fin = np.isfinite(s[keep]) & np.isfinite(s2)
        rel = np.abs(s[keep][fin] - s2[fin]) / np.maximum(np.abs(s2[fin]), 1e-30)
        worst = float(rel.max()) if fin.any() else 0.0
        ck(f"{k:8s} std(0) subset-vs-full within 1 ulp (rel {worst:.2e})",
           bool(fin.any()) and worst < ULP32)

    print("\n== NEGATIVE CONTROL: a cross-member transform MUST be caught ==")
    def cross(O, T):
        """Deliberately not per-column: divides by a whole-matrix scalar."""
        return O / O.std(), T / T.std()
    A, At = cross(O, T)
    B, Bt = cross(O[:, keep], T[:, keep])
    ck("cross-member transform is DETECTED as differing",
       not np.array_equal(A[:, keep], B))

    print("\n== NEGATIVE CONTROL: the comparison itself can fail ==")
    A, _ = transform(O, T, "rankraw")
    ck("a perturbed column IS detected",
       not np.array_equal(A[:, keep], A[:, keep] + np.eye(len(A), len(keep)) * 1e-9))

    print(f"\nFAILURES {len(FAILS)}")
    if FAILS:
        for f in FAILS:
            print(f"  !! {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
