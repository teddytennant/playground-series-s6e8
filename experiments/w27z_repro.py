"""M13 -- is the cross-fitted CV estimator reproducible to 1e-6 at all?

Pre-registered at experiments/w27_prereg_slot6.txt §M13 BEFORE this file was written.

R-M11e failed on 4 of 5 cells in w27w. Two hypotheses with opposite consequences:
H1 the column-subset equivalence R-M11f is false, or H2 the estimator itself is not
reproducible to 1e-6 because float32 lbfgs early-stops on gradient noise. The decisive
test is to fit the SAME cell two ways in ONE process, so threading and BLAS are held
fixed, and see whether the two agree with each other and whether either lands on the
shipped number.

    w27z_repro.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import DATA, get_folds  # noqa: E402
from blend_lab import load_all  # noqa: E402

DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400")
NEW = ("lat_ctdrop_r400", "lat_encdrop_r400")
D346 = tuple(os.path.join(DATA, d) for d in
             ("ext_members3", "ext_members4", "ext_members6"))
D3467 = D346 + (os.path.join(DATA, "ext_members7pin"),)
SHIPPED = 0.9700993121          # w27_ad188std_hybrid, from w27h
KIND = "hybrid"


def stdz(M):
    s = M.std(0)
    s[s <= 0] = 1.0
    return (M / s).astype(M.dtype)


def crossfit(Z, y, folds, tag, C=1.0):
    mo = np.zeros(len(y))
    iters = []
    for itr, iva in folds:
        m = LogisticRegression(max_iter=5000, C=C).fit(Z[itr], y[itr])
        iters.append(int(np.ravel(m.n_iter_)[0]))
        mo[iva] = m.decision_function(Z[iva])
    a = roc_auc_score(y, mo)
    print(f"  {tag:34s} {a:.10f}   n_iter {iters}", flush=True)
    return a


def main():
    t0 = time.time()
    for p in D3467:
        assert os.path.isdir(p), p
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}", flush=True)

    # --- B: the genuine 188-member load (what w27h shipped) --------------------------
    namesB, y, matsB, _ = load_all((KIND,), DROP, extra_dirs=D346, dtype="float32")
    assert len(namesB) == 188, len(namesB)
    ZB = stdz(matsB[KIND][0]); del matsB
    folds = get_folds(y)

    # --- A: the 190 load, standardised, then subset to 188 columns (R-M11f) ----------
    namesA, yA, matsA, _ = load_all((KIND,), DROP, extra_dirs=D3467, dtype="float32")
    assert len(namesA) == 190 and np.array_equal(y, yA)
    ZA_full = stdz(matsA[KIND][0]); del matsA
    keep = np.array([i for i, n in enumerate(namesA) if n not in NEW])
    assert len(keep) == 188
    assert [namesA[i] for i in keep] == list(namesB), "member ORDER differs between loads"
    ZA = ZA_full[:, keep]
    print(f"loaded both in {time.time()-t0:.0f}s", flush=True)

    # matrices identical? this separates H1 from H2 before a single fit runs.
    same = np.array_equal(ZA, ZB)
    mx = float(np.abs(ZA.astype(np.float64) - ZB.astype(np.float64)).max())
    print(f"\nR-M11f, checked on the MATRIX not the score: "
          f"ZA == ZB -> {same}   max|ZA-ZB| = {mx:.3e}")

    print(f"\n== the seed-42 188-arm {KIND} cell, one process, threads held fixed ==")
    print(f"  {'shipped (w27h, on disk)':34s} {SHIPPED:.10f}")
    aB = crossfit(ZB, y, folds, "B: genuine 188 load, float32")
    aA = crossfit(ZA, y, folds, "A: 190 load subset to 188, f32")
    aA2 = crossfit(ZA, y, folds, "A again (in-process determinism)")
    a64 = crossfit(ZA.astype(np.float64), y, folds, "A in float64")
    a190 = crossfit(ZA_full, y, folds, "190 arm, float32")

    print("\n== M13 readout ==")
    print(f"  M13(a) A - B                    {(aA-aB)*1e6:+8.3f}e-6   "
          f"(R-M11f {'HOLDS' if abs(aA-aB) < 2e-7 else 'FAILS'})")
    print(f"  A - A (same process, twice)     {(aA2-aA)*1e6:+8.3f}e-6")
    print(f"  M13(b) B - shipped              {(aB-SHIPPED)*1e6:+8.3f}e-6")
    print(f"         A - shipped              {(aA-SHIPPED)*1e6:+8.3f}e-6")
    print(f"  M13(c) float64 - float32 (A)    {(a64-aA)*1e6:+8.3f}e-6")
    print(f"  paired 190 - 188 (this process) {(a190-aA)*1e6:+8.3f}e-6  "
          f"[w27w seed42 hybrid said +0.31e-6]")
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
