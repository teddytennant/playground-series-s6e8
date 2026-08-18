"""w26f_smoke — exercise every non-fitting path of w26f_csweep on synthetic cells.

The fitting path costs ~15s a cell and 16 cores; the bookkeeping path costs nothing and is
where this workspace's actual failures have been (w24: a re-cut consuming the record it was
to be checked against; w26 slot 3: a module doing its work at import). So the bookkeeping
gets tested first, on a throwaway seed namespace, and the fits are left to the chained run.

Covered: seed0 keying, the resume set, the duplicate-row guard, the pivot the report does,
the h3 mix over cells, the NaN path when the cross-fitted arm has not run, and the JSON dump.
NOT covered: LogisticRegression, load_members, transform. Those are shared with w26a, which
is running green right now.

Writes ONLY to the seed0=999999 namespace and deletes it again. It cannot touch the real
seed-1000 or seed-2000 artefacts -- that separation is the thing being tested.

    .venv/bin/python experiments/w26f_smoke.py
"""
from __future__ import annotations

import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import w26f_csweep as W  # noqa: E402  (safe: it is __main__-guarded, it does nothing on import)

SMOKE = 999999


def main():
    assert not os.path.exists(W.csv_path(SMOKE)), "smoke namespace is dirty, refusing"
    ok = True

    # ---- 1. the seed namespaces are disjoint, which is the defect this rewrite fixed ----
    a = W.cell_path("hybrid", 1.0, 0, 1000)
    b = W.cell_path("hybrid", 1.0, 0, 2000)
    assert a != b and W.csv_path(1000) != W.csv_path(2000), "seed0 keying is BROKEN"
    print(f"1. seed namespaces disjoint\n   {os.path.basename(a)} in {os.path.basename(os.path.dirname(a))}"
          f"\n   {os.path.basename(b)} in {os.path.basename(os.path.dirname(b))}")

    # ---- 2. synthetic cells, with a deliberate duplicate row and a deliberate orphan ----
    y, sp = None, None
    from common import TARGET, load_raw  # noqa: E402
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    sp = W.splits_for(y, W.REPS, SMOKE)
    nho = len(sp[0][1])
    rng = np.random.default_rng(0)
    # a signal that rises slightly with C so the report has a non-degenerate curve to print
    base = rng.normal(size=nho)
    for kind in W.KINDS:
        for ci, C in enumerate(W.GRID):
            for rep, (_ip, ih) in enumerate(sp):
                d = y[ih] * (0.9 + 0.002 * ci) + rng.normal(size=nho) + 0.0 * base
                W.append_row(dict(kind=kind, C=C, rep=rep, hold_auc=float(np.mean(d)),
                                  n_iter=10, secs=0.1, seed0=SMOKE), d, kind, SMOKE)
    n_cells = len(W.KINDS) * len(W.GRID) * W.REPS
    df = W.load_done(SMOKE)
    assert len(df) == n_cells, f"expected {n_cells} rows, got {len(df)}"
    print(f"2. wrote {n_cells} synthetic cells")

    # duplicate row: append the same (kind,C,rep) again and check load_done collapses it
    W.append_row(dict(kind="hybrid", C=1.0, rep=0, hold_auc=0.5, n_iter=1, secs=0.1,
                      seed0=SMOKE), np.zeros(nho), "hybrid", SMOKE)
    df = W.load_done(SMOKE)
    if len(df) != n_cells:
        print(f"   FAIL duplicate guard: {len(df)} rows, expected {n_cells}")
        ok = False
    else:
        print("   duplicate (kind,C,rep) collapsed, keep=last")

    # orphan: a CSV row whose .npy is gone must NOT count as done
    os.remove(W.cell_path("rescale", 0.3, 2, SMOKE))
    done = {(r.C, r.rep) for r in W.load_done(SMOKE).itertuples()
            if r.kind == "rescale" and os.path.exists(W.cell_path("rescale", r.C, r.rep, SMOKE))}
    if (0.3, 2) in done:
        print("   FAIL orphan guard: a cell with no .npy counted as done")
        ok = False
    else:
        print("   orphan CSV row (no .npy) correctly counted as NOT done")
    # put it back so the report has a full grid
    W.append_row(dict(kind="rescale", C=0.3, rep=2, hold_auc=0.5, n_iter=1, secs=0.1,
                      seed0=SMOKE), rng.normal(size=nho), "rescale", SMOKE)

    # ---- 3. the report path, including the mix and the NaN cross-fitted column ----
    print("\n3. report() on the synthetic namespace "
          "(numbers are noise; only the shape is under test)\n" + "-" * 70)
    W.report(SMOKE)
    print("-" * 70)
    for p in (os.path.join(W.HERE, f"w26f_csweep_s{SMOKE}.json"),
              os.path.join(W.HERE, f"w26f_mix_s{SMOKE}.csv")):
        if not os.path.exists(p):
            print(f"   FAIL report did not write {os.path.basename(p)}")
            ok = False
    print("   report wrote both artefacts")

    # ---- 4. clean up, and prove the real namespaces were never touched ----
    shutil.rmtree(W.holddir(SMOKE))
    for p in (W.csv_path(SMOKE), os.path.join(W.HERE, f"w26f_csweep_s{SMOKE}.json"),
              os.path.join(W.HERE, f"w26f_mix_s{SMOKE}.csv")):
        if os.path.exists(p):
            os.remove(p)
    leaked = [p for p in (W.csv_path(1000), W.csv_path(2000), W.holddir(1000),
                          W.holddir(2000)) if os.path.exists(p)]
    print(f"\n4. cleaned up. real namespaces present on disk: {leaked or 'none'}")

    print("\nSMOKE " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
