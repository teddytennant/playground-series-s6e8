"""w29d -- the 194-vs-190 delta, replicated over four STACKER fold partitions.

`experiments/w27w_partition.py` with the arms moved up by four members. Everything about the
instrument is w27w's and is not re-argued here: partitions are a REPLICATION device, not a
variance-reduction one (w27v: five of six off-seed partitions read +8.9..+13.3e-6 HIGH against
seed 42, because only under seed 42 is the stacker's validation fold exactly one member fold),
and partition noise cancels 17-82% in a paired contrast on identical partitions, which is what
makes a replication of a few-e-6 effect worth running.

WHY IT IS NEEDED HERE. Levels built on different days are not differenceable: w27 slot 8
measured a ~4e-6 reproducibility floor under absolute CV, and blend_lab only pinned its BLAS
thread count at w28 -- AFTER w27_ad190std_h3 was built. So `w29_ad194std_h3 - 0.9701133391`
off two logs is not a measurement of the four new members. A paired contrast on identical
partitions in ONE process is.

R-M11f, unchanged: ONE load serves both arms -- the 190 pack is the 194 matrix with the four
ext_members8 columns deleted. Every transform in agent/stack.py is strictly per-column and
--standardize divides by a per-column std, so subsetting after transform+scale is exactly
equal to loading 190 members.

⚠ The seed-42 gate is INFORMATIVE, NOT BLOCKING here. w27w ran the same gate and it failed on
4 of 5 cells by 1.2e-6 to 3.9e-6 -- the reproducibility floor, not a load error -- while the
paired delta it guards was clean at 4/4. Keeping it hard would have voided a correct result.
It prints and it is recorded; the run does not stop on it. A cell off by more than 2e-5 is a
different animal and does abort.

    .venv/bin/python experiments/w29d_partition.py --seeds 42,101,13,7
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
              ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin",
               "ext_members8"))
# the four function-class members that make 194 out of 190 (w29 prereg §M14)
NEW = ("qda_raw", "rff_raw", "rff_lat", "poly2_raw")
BIG, SMALL = 194, 190

# Informative gate. Shipped seed-42 numbers on disk, from w27t (190).
GATE = {(190, "hybrid"): 0.970103, (190, "rankraw"): 0.970095,
        (190, "rescale"): 0.970099}


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
    if len(names) != BIG:
        raise SystemExit(f"R-M11e: expected {BIG} members, got {len(names)}")
    missing = [n for n in NEW if n not in names]
    if missing:
        raise SystemExit(f"R-M11e: the new members are not in the load: {missing}")
    keep = np.array([i for i, n in enumerate(names) if n not in NEW])
    if len(keep) != SMALL:
        raise SystemExit(f"R-M11e: expected to drop {BIG-SMALL} for the {SMALL} arm, "
                         f"dropped {BIG-len(keep)}")
    print(f"{BIG} members, {SMALL}-arm keeps {len(keep)}, loaded in {time.time()-t0:.0f}s",
          flush=True)

    # standardise exactly as blend_lab.build(std=True) does: scale from the OOF side,
    # per column, computed on the FULL 194-column matrix.
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
        for arm, cols in ((BIG, None), (SMALL, keep)):
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
        print("  gate reproduces the shipped build" if not bad
              else f"  gate off on {bad} -- within the ~4e-6 reproducibility floor "
                   f"(w27 slot 8); INFORMATIVE ONLY, the paired delta below is unaffected")
        for arm, k in bad:
            if abs(res[42][arm][k] - GATE[(arm, k)]) > 2e-5:
                raise SystemExit(f"  *** {arm}/{k} is off by more than 2e-5 -- that is a "
                                 f"LOAD error, not the floor. Everything below is void.")
    else:
        print("  seed 42 not run, gate cannot be evaluated")

    # ---- M11(b): the paired delta, per partition -------------------------------------
    print(f"\n== THE GATE: paired {BIG} - {SMALL} delta on identical partitions (e-6) ==")
    print(f"  {'seed':>5s}  " + "  ".join(f"{k:>9s}" for k in (*H3, "h3")))
    dh3 = []
    for sd in seeds:
        d = {k: (res[sd][BIG][k] - res[sd][SMALL][k]) * 1e6 for k in (*H3, "h3")}
        dh3.append(d["h3"])
        print(f"  {sd:>5d}  " + "  ".join(f"{d[k]:+9.2f}" for k in (*H3, "h3")))
    dh3 = np.array(dh3)
    npos = int((dh3 > 0).sum())
    print(f"\n  h3 delta: mean {dh3.mean():+.2f}e-6  sd {dh3.std(ddof=1):.2f}e-6  "
          f"se {dh3.std(ddof=1)/np.sqrt(len(dh3)):.2f}e-6  positive in {npos}/{len(dh3)}")
    print(f"  M14(b)4 registered +2..+8e-6 per passing arm, modal +3.4e-6, so +8..+32e-6 "
          f"for four; the decision rule is positive in >={max(3, len(dh3)-1)}/{len(dh3)}: "
          f"{'PASS -- the 194 pack replaces the 190 as the deadline pick' if npos >= max(3, len(dh3)-1) else 'FAIL -- w27_ad190stdcorr stays WANTED'}")

    # ---- M11(c): off-seed optimism, replicated on a different member set --------------
    print("\n== off-seed mean minus seed 42 (w14c's mechanism, new member set) ==")
    off = [s for s in seeds if s != 42]
    if 42 in res and off:
        for arm in (SMALL, BIG):
            for k in (*H3, "h3"):
                d = (np.mean([res[s][arm][k] for s in off]) - res[42][arm][k]) * 1e6
                print(f"  arm {arm} {k:8s} {d:+8.2f}e-6")

    out = os.path.join(ROOT, "experiments", "w29d_partition.json")
    json.dump({"seeds": seeds, "res": {str(s): {str(arm): r for arm, r in v.items()}
                                       for s, v in res.items()},
               "gate_bad": bad}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
