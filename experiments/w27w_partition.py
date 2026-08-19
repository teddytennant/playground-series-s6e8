"""w27w -- does the 190-vs-188 delta survive a change of STACKER fold partition?

Pre-registered in full at experiments/w27_prereg_slot6.txt §M11, appended BEFORE this file
was written and before any off-seed number for this pack existed. The decision rule R-M11d
is fixed there and is not revisited here.

WHY. Slot 7's handed angle is fold/seed diversity. Its naive form -- average the stack over
fold seeds -- is a structural null (blend_lab takes the test prediction from a full fit, so
no shipped file depends on the stacker partition) AND actively harmful as a CV estimator
(w27v: five of six off-seed partitions read +8.9..+13.3e-6 HIGH against seed 42 on 6 of 6
metrics, because only under seed 42 is the stacker's validation fold exactly one member
fold). So partitions are used here as a REPLICATION device, not a variance-reduction one:
partition noise cancels 17-82% in a paired contrast on identical partitions (w27v), which
is what makes a 4-partition replication of a +2.8e-6 effect worth running at all.

R-M11f. ONE load serves both arms: the 188 pack is the 190 matrix with the two
ext_members7pin columns deleted. Every transform in agent/stack.py is strictly per-column
and --standardize divides by a per-column std, so subsetting after transform+scale is
exactly equal to loading 188 members. That is asserted, not assumed -- R-M11e requires the
seed-42 cells to reproduce the on-disk shipped numbers to 6 dp.

    .venv/bin/python experiments/w27w_partition.py --seeds 42,101,13,7
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import DATA  # noqa: E402
from blend_lab import load_all  # noqa: E402

H3 = ("hybrid", "rankraw", "rescale")           # the deadline mix; it EXCLUDES logit
DROP = "golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,lat_ctfixte_r400"
# WARNING: load_all() takes ABSOLUTE paths -- blend_lab.main does the
# os.path.join(DATA, d) itself, and load_members SILENTLY SKIPS any dir that does
# not exist (`if not os.path.isdir(d): continue`). Passing the bare names loaded
# 165 members instead of 190 with no error at all. Same hazard class as R-M10f,
# different door: not a live directory, a relative path resolving to nothing.
XDIRS = tuple(os.path.join(DATA, d) for d in
              ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin"))
NEW = ("lat_ctdrop_r400", "lat_encdrop_r400")   # the two members that make 190 out of 188

# R-M11e gate. Shipped seed-42 numbers on disk, from w27h (188) and w27t (190).
GATE = {(188, "hybrid"): 0.9700993121, (188, "rankraw"): 0.9700930487,
        (188, "rescale"): 0.9700962686,
        (190, "hybrid"): 0.970103, (190, "rankraw"): 0.970095}


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def crossfit(Z, y, folds, C=1.0):
    mo = np.zeros(len(y))
    for itr, iva in folds:
        mo[iva] = (LogisticRegression(max_iter=5000, C=C)
                   .fit(Z[itr], y[itr]).decision_function(Z[iva]))
    return mo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,13,7")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    # fail fast: the count assert below is worthless if it only fires after the
    # transforms have been paid for, which is what happened on the first launch.
    for d in XDIRS:
        if not os.path.isdir(d):
            raise SystemExit(f"R-M11e: extra dir does not exist: {d}")
    t0 = time.time()
    names, y, mats, _te = load_all(H3, DROP.split(","), extra_dirs=XDIRS, dtype="float32")
    if len(names) != 190:
        raise SystemExit(f"R-M11e: expected 190 members, got {len(names)}")
    keep = np.array([i for i, n in enumerate(names) if n not in NEW])
    if len(keep) != 188:
        raise SystemExit(f"R-M11e: expected to drop 2 for the 188 arm, dropped "
                         f"{190-len(keep)}")
    print(f"190 members, 188-arm keeps {len(keep)}, loaded in {time.time()-t0:.0f}s",
          flush=True)

    # standardise exactly as blend_lab.build(std=True) does: scale from the OOF side,
    # per column, computed on the FULL 190-column matrix.
    # Free each transform's TEST matrix as soon as its OOF side is scaled. Nothing here
    # predicts test -- w27t already wrote those files -- and holding all three (Z, Zt)
    # pairs is what got w25d's first version OOM-killed with no traceback.
    Z = {}
    for k in H3:
        M = mats[k][0]
        s = M.std(0)
        s[s <= 0] = 1.0
        Z[k] = (M / s).astype(M.dtype)
        mats[k] = None
        del M
        gc.collect()
    del mats
    gc.collect()

    res = {}          # res[seed][arm][kind]
    for sd in seeds:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=sd)
                     .split(np.zeros(len(y)), y))
        res[sd] = {}
        for arm, cols in ((190, None), (188, keep)):
            oof, row = {}, {}
            for k in H3:
                t1 = time.time()
                M = Z[k] if cols is None else Z[k][:, cols]
                oof[k] = crossfit(M, y, folds, a.C)
                row[k] = roc_auc_score(y, oof[k])
                print(f"  seed {sd:>3d} arm {arm} {k:8s} {row[k]:.7f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                del M
                gc.collect()
            row["h3"] = roc_auc_score(y, np.mean([rk(oof[k]) for k in H3], 0))
            print(f"  seed {sd:>3d} arm {arm} {'h3':8s} {row['h3']:.7f}", flush=True)
            res[sd][arm] = row

    # ---- R-M11e gate ----------------------------------------------------------------
    print("\n== R-M11e gate: seed-42 cells vs the shipped on-disk numbers ==")
    bad = []
    if 42 in res:
        for (arm, k), want in GATE.items():
            got = res[42][arm][k]
            dp = len(str(want).split(".")[1])
            ok = abs(got - want) < 0.5 * 10 ** (-dp)
            print(f"  arm {arm} {k:8s} got {got:.7f}  want {want:.{dp}f}  "
                  f"{'OK' if ok else '*** MISMATCH ***'}")
            if not ok:
                bad.append((arm, k))
        print("  GATE PASSES -- the load reproduces the shipped builds" if not bad
              else f"  *** GATE FAILS on {bad}: every number below is VOID ***")
    else:
        print("  seed 42 not run, gate cannot be evaluated")

    # ---- M11(b): the paired delta, per partition -------------------------------------
    print("\n== M11(b) THE GATE: paired 190 - 188 delta on identical partitions (e-6) ==")
    print(f"  {'seed':>5s}  " + "  ".join(f"{k:>9s}" for k in (*H3, "h3")))
    dh3 = []
    for sd in seeds:
        d = {k: (res[sd][190][k] - res[sd][188][k]) * 1e6 for k in (*H3, "h3")}
        dh3.append(d["h3"])
        print(f"  {sd:>5d}  " + "  ".join(f"{d[k]:+9.2f}" for k in (*H3, "h3")))
    dh3 = np.array(dh3)
    npos = int((dh3 > 0).sum())
    print(f"\n  h3 delta: mean {dh3.mean():+.2f}e-6  sd {dh3.std(ddof=1):.2f}e-6  "
          f"se {dh3.std(ddof=1)/np.sqrt(len(dh3)):.2f}e-6  positive in {npos}/{len(dh3)}")
    print(f"  M11(b) requires positive in >=3/{len(dh3)}: "
          f"{'PASS -- R-M10b applies as written' if npos >= 3 else 'FAIL -- R-M11d fires, the 188 files stay WANTED'}")

    # ---- M11(c): off-seed optimism, replicated on a different member set --------------
    print("\n== M11(c) off-seed mean minus seed 42 (w14c's mechanism, new member set) ==")
    off = [s for s in seeds if s != 42]
    if 42 in res and off:
        for arm in (188, 190):
            for k in (*H3, "h3"):
                d = (np.mean([res[s][arm][k] for s in off]) - res[42][arm][k]) * 1e6
                print(f"  arm {arm} {k:8s} {d:+8.2f}e-6")

    out = os.path.join(ROOT, "experiments", "w27w_partition.json")
    json.dump({"seeds": seeds, "res": {str(s): {str(arm): r for arm, r in v.items()}
                                       for s, v in res.items()},
               "gate_bad": bad}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
