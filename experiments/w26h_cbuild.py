"""w26h — turn the w26f C grid into submittable files, because the send queue runs dry on 08-22.

WHY THIS EXISTS
---------------
w26 slot 4 found the send queue is 27 files deep, not the 46 the tooling claimed (the CLI's
50-row page hid 21 already-sent files). Ten a day empties it on 08-21 and the deadline is
08-31, so nine send days have nothing built for them. The brief is explicit that an unused
slot is pure waste here -- submissions never evict each other and the board shows best-of-all
-- so the constraint on the send strategy is now BUILDING candidates, not choosing among them.

The cheapest honest source of genuinely different files is the axis w26f is already sweeping.
Each (transform, C) cell is a different fit of the same 187 members, so it is a real variant
and not a re-send, and the marginal cost is ONE full-data fit: the cross-fitted OOF that
defends it on CV has already been computed by `w26f_csweep.py --cv`. 3 transforms x 7 C values
plus their 7 h3 mixes = 28 files, about three send days, for ~10 minutes of compute.

AND SENDING THEM IS AN EXPERIMENT, NOT FILLER. w26f measures the C curve on held-out rows and
on cross-fitted CV; those two disagree exactly insofar as the combiner's fold reuse leaks,
which is w25d's open question. The public LB is a THIRD reading of the same curve, on rows
from a different distribution of the same generator, and 21 points on it is far more than the
one-point comparisons this workspace has been making. Whatever the sweep concludes, the send
day costs nothing and buys that curve.

WHAT THIS IS NOT. Nothing here is a deadline candidate on its own. Prereg §D3's ship rule is
unchanged: a cell ships only if it wins on the HELD-OUT rows and its cross-fitted CV clears
0.9701182. Building a file is not selecting it, and selection is still on CV.

CONVENTION, AND THE GATE THAT MAKES IT SAFE
-------------------------------------------
The submission path scales columns by the sd over ALL training rows, because that is what
`blend_lab.build(std=True)` did for every standardised file already on the board, and a
candidate should be comparable to its predecessors. `w26f --cv` scales by the FOLD-TRAIN rows,
which is the clean version, and its OOF vectors are what defend these files on CV. The two
conventions differ by an sd computed over 691k vs 553k rows.

That difference is asserted to be negligible everywhere in this workspace and has never been
checked. Here it IS checked, and this script REFUSES TO RUN if it fails: w26f's C=1.0 cells
must reproduce the numbers `logs_w23f_stdbuild4.txt` recorded for the global-scale build, to
within GATE_E6. If they do not, the two conventions are not interchangeable, copying the OOF
across would misprice every file below, and the right move is to find out why rather than to
ship 28 files defended by the wrong number.

    .venv/bin/python experiments/w26h_cbuild.py --dry            # check the gate, build nothing
    .venv/bin/python experiments/w26h_cbuild.py --kind hybrid
    .venv/bin/python experiments/w26h_cbuild.py --mix
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402

import w26f_csweep as F  # noqa: E402  (__main__-guarded; safe to import, unlike most of experiments/)

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302
GATE_E6 = 2.0       # e-6 of AUC. The log records 6 dp, so 1e-6 is its own rounding; 2 is one
                    # rounding step of slack and still 4x under the 8.21e-6 slice-noise floor.


def gate():
    """w26f --cv at C=1.0 must reproduce the global-scale build. Returns (ok, lines)."""
    cv = F.load_cv()
    lines, ok = [], True
    if cv.empty:
        return False, ["  the w26f --cv arm has not run; there are no OOF vectors to defend "
                       "these files with. Run `w26f_csweep.py --cv KIND` first."]
    for kind in F.KINDS:
        row = cv[(cv.kind == kind) & (cv.C == 1.0)]
        if row.empty:
            ok = False
            lines.append(f"  {kind:8s} C=1.0 cell MISSING")
            continue
        got = float(row.cv_auc.iloc[0])
        want = F.W23F_STD_CV[kind]
        d = (got - want) * 1e6
        good = abs(d) <= GATE_E6
        ok &= good
        lines.append(f"  {kind:8s} cross-fitted {got:.10f} vs logs_w23f_stdbuild4 {want:.6f}"
                     f"   {d:+6.2f}e-6   {'PASS' if good else 'FAIL'}")
    return ok, lines


def name_for(kind, C):
    return f"w26h_c{C:g}_{kind}"


def write_sub(name, ids, pred, oof_src=None, oof=None):
    p = os.path.join(SUB, f"{name}.csv")
    d = pd.DataFrame({"id": ids, TARGET: pred})
    assert len(d) == N_TEST, f"{name}: {len(d)} rows"
    assert np.isfinite(pred).all(), f"{name}: non-finite predictions"
    d.to_csv(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    op = os.path.join(SUB, f"oof_{name}.npy")
    if oof is not None:
        np.save(op, np.asarray(oof, dtype=np.float64))
    elif oof_src:
        # w23b ranks candidates on fast_auc(y, oof); AUC is invariant to a monotone map, so
        # the cross-fitted DECISION FUNCTION is as good as a probability here.
        shutil.copyfile(oof_src, op)
    return p


def run_kind(kind):
    ok, lines = gate()
    print("\n".join(lines))
    if not ok:
        raise SystemExit("GATE FAILED — refusing to build files defended by an OOF vector "
                         "computed under a convention that does not reproduce the record.")
    todo = [C for C in F.GRID if not os.path.exists(os.path.join(SUB, name_for(kind, C) + ".csv"))]
    if not todo:
        print(f"{kind}: all {len(F.GRID)} files already built")
        return
    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ids = te["id"].to_numpy()
    del tr, te
    from stack import load_members, transform  # noqa: E402
    from common import DATA  # noqa: E402
    from blend_lab import HONEST_DROP  # noqa: E402
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, N_TEST, extra_dirs=extra, drop=set(HONEST_DROP))
    Z, Zt = transform(O, T, kind)
    Z, Zt = Z.astype("float64"), Zt.astype("float64")
    del O, T
    gc.collect()
    # blend_lab.build(std=True)'s convention: scale from ALL training rows, applied to both.
    s = Z.std(0)
    s[s <= 0] = 1.0
    Z /= s
    Zt /= s
    print(f"{len(names)} members, {kind} {Z.shape} / {Zt.shape}, {time.time()-t0:.0f}s",
          flush=True)

    for C in todo:
        t = time.time()
        m = LogisticRegression(max_iter=5000, C=C, tol=1e-4).fit(Z, y)
        pred = m.predict_proba(Zt)[:, 1]
        nm = name_for(kind, C)
        src = F.cv_cell_path(kind, C)
        if not os.path.exists(src):
            print(f"  C {C:<6g} SKIPPED: no cross-fitted OOF cell, the file would be "
                  f"unrankable on CV")
            continue
        cvauc = roc_auc_score(y, np.load(src))
        write_sub(nm, ids, pred, oof_src=src)
        print(f"  C {C:<6g} -> {nm}.csv  CV {cvauc:.10f}  mean {pred.mean():.4f}  "
              f"{time.time()-t:.0f}s", flush=True)
        del m
    print(f"{kind} done, {time.time()-t0:.0f}s")


def run_mix():
    """The h3 mix per C: rank-average of the three transform stacks, on test and on OOF.

    Rank-averaged the same way `make_h3.py` does for the files already on the board, so these
    are comparable to them and not a new family.
    """
    ok, lines = gate()
    print("\n".join(lines))
    if not ok:
        raise SystemExit("GATE FAILED — see run_kind.")
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    for C in F.GRID:
        nm = f"w26h_c{C:g}_h3"
        if os.path.exists(os.path.join(SUB, nm + ".csv")):
            continue
        subs = [os.path.join(SUB, name_for(k, C) + ".csv") for k in F.KINDS]
        oofs = [F.cv_cell_path(k, C) for k in F.KINDS]
        if not all(os.path.exists(q) for q in subs + oofs):
            print(f"  C {C:<6g} skipped: needs all three transforms built first")
            continue
        ds = [pd.read_csv(q) for q in subs]
        ids = ds[0]["id"].to_numpy()
        assert all((d["id"].to_numpy() == ids).all() for d in ds), f"{nm}: id order differs"
        pred = np.mean([rankdata(d[TARGET].to_numpy()) for d in ds], 0) / (len(ids) + 1)
        oof = np.mean([rankdata(np.load(q)) for q in oofs], 0)
        write_sub(nm, ids, pred, oof=oof)
        print(f"  C {C:<6g} -> {nm}.csv  CV {roc_auc_score(y, oof):.10f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=F.KINDS)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--dry", action="store_true",
                    help="check the reproduction gate and print the plan; build nothing")
    a = ap.parse_args()
    if a.dry:
        ok, lines = gate()
        print("REPRODUCTION GATE (w26f --cv C=1.0 vs the global-scale build):")
        print("\n".join(lines))
        print(f"gate {'PASS' if ok else 'FAIL'}")
        built = [f for f in os.listdir(SUB) if f.startswith("w26h_c") and f.endswith(".csv")]
        want = [name_for(k, C) + ".csv" for k in F.KINDS for C in F.GRID] + \
               [f"w26h_c{C:g}_h3.csv" for C in F.GRID]
        print(f"\nplan: {len(want)} files, {len(built)} already on disk, "
              f"{len(set(want) - set(built))} to build")
        print("  " + ", ".join(sorted(set(want) - set(built))[:6]) + " ...")
        # non-zero on a failed gate so a chain script stops HERE, with the reason printed,
        # rather than at the first --kind call with a stack trace
        return 0 if ok else 1
    if a.mix:
        run_mix()
    elif a.kind:
        run_kind(a.kind)
    else:
        ap.error("one of --kind, --mix, --dry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
