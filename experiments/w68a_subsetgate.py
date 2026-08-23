"""w68a -- is the SUBSETTED 199 arm the same object as a NATIVELY LOADED 199 pack?

Registered in experiments/w68_prereg.txt, committed 4621590 BEFORE this file existed.

WHY. w65a's gate failed 7/8 (`arm 199 rankraw` -6.49e-6) with offsets that are SIGN-CONSISTENT
WITHIN ARM: the unsubsetted 202 arm reads +1.500e-6 mean against its shipped file, the
subsetted 199 arm reads -3.595e-6, a differential of -5.095e-6. An independent estimate from
the same run -- w65a's seed-42 h3 delta of +2.25e-6 against the stored -2.406e-6 -- puts it at
+4.656e-6 with a SIGN FLIP. The two agree to 0.44e-6.

w65a's R-M11f says subsetting after transform+scale is EXACTLY loading 199 members. Reading
agent/stack.py that should be true: rankraw is a per-j loop, hybrid's `bad` mask and its
`s = B[:, bad].std(0)` are per-column, rescale's lo/hi are .min(0)/.max(0), and w65a's own
`s = M.std(0)` is per-column. So the two matrices should be BITWISE equal.

This compares the MATRICES, not the AUCs. An AUC gap confounds the data question with the
lbfgs question; the data question is answerable exactly and is answered first.

    .venv/bin/python experiments/w68a_subsetgate.py
"""
from __future__ import annotations

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

H3 = ("hybrid", "rankraw", "rescale")
DROP = ("golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,"
        "lat_ctfixte_r400,om_cat")
# w38d_run.sh, verbatim (the 202 pack).
DIRS_202 = ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin",
            "ext_members8", "ext_members10", "ext_members11", "ext_members12",
            "ext_members14")
# w36b_run.sh, verbatim -- the 202 list MINUS ext_members14 (the native 199 pack).
DIRS_199 = DIRS_202[:-1]
NEW = ("lexb_cat_base", "lexb_lgb02", "lexb_xgb_base")
# w65a's own PLA1 draw, from its log -- the negative control.
PLA1 = ("imp_lgbm", "naji04", "bolt_cat_pair_evidence")
BIG, SMALL = 202, 199
SEED = 42

OUT = {"prereg": "experiments/w68_prereg.txt", "failures": []}


def fail(msg):
    OUT["failures"].append(msg)
    print("  !! " + msg, flush=True)


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def crossfit(Z, y, folds, C=1.0):
    mo = np.zeros(len(y))
    for itr, iva in folds:
        mo[iva] = (LogisticRegression(max_iter=5000, C=C)
                   .fit(Z[itr], y[itr]).decision_function(Z[iva]))
    return mo


def scaled(mats):
    """Exactly w65a's scaling: per column, from the OOF side. Test side dropped at once."""
    Z = {}
    for k in H3:
        M = mats[k][0]
        s = M.std(0)
        s[s <= 0] = 1.0
        Z[k] = (M / s).astype(M.dtype)
        mats[k] = None
        del M
        gc.collect()
    return Z


def main():
    t0 = time.time()
    print("== loading the 202 pack ==", flush=True)
    n202, y, m202, _ = load_all(H3, DROP.split(","),
                                extra_dirs=tuple(os.path.join(DATA, d) for d in DIRS_202),
                                dtype="float32")
    if len(n202) != BIG:
        raise SystemExit(f"expected {BIG} members, got {len(n202)}")
    Z202 = scaled(m202)
    del m202
    gc.collect()
    print(f"  202 pack scaled at {time.time()-t0:.0f}s", flush=True)

    print("\n== loading the 199 pack NATIVELY (ext_members14 omitted) ==", flush=True)
    n199, y2, m199, _ = load_all(H3, DROP.split(","),
                                 extra_dirs=tuple(os.path.join(DATA, d) for d in DIRS_199),
                                 dtype="float32")
    if len(n199) != SMALL:
        raise SystemExit(f"expected {SMALL} members natively, got {len(n199)}")
    Z199 = scaled(m199)
    del m199
    gc.collect()
    print(f"  199 pack scaled at {time.time()-t0:.0f}s", flush=True)

    # ---- P1 GATE: name lists, ELEMENTWISE and IN ORDER -----------------------------------
    print("\n== P1 GATE: the 202 name list minus lexb vs the native 199 name list ==")
    sub_names = [n for n in n202 if n not in NEW]
    same_order = sub_names == list(n199)
    same_set = set(sub_names) == set(n199)
    print(f"  |202 minus lexb| = {len(sub_names)}   |native 199| = {len(n199)}")
    print(f"  same SET   : {same_set}")
    print(f"  same ORDER : {same_order}")
    if same_set and not same_order:
        mism = [(i, a, b) for i, (a, b) in enumerate(zip(sub_names, n199)) if a != b]
        print(f"  *** ORDER DIFFERS at {len(mism)} positions; first 5: {mism[:5]}")
    if not np.array_equal(y, y2):
        fail("P1: the two loads disagree on y")
    OUT["P1"] = {"same_set": bool(same_set), "same_order": bool(same_order),
                 "n_sub": len(sub_names), "n_native": len(n199)}
    if not same_set:
        fail("P1 falsified: the two 199 member sets are not equal")
    if not same_order:
        fail("P1 falsified: same set, different ORDER")

    # Align the subset to the native order so P2 tests the DATA, not the ordering.
    pos = {n: i for i, n in enumerate(n202)}
    keep = np.array([pos[n] for n in n199])

    # ---- P2: the matrices, BITWISE ------------------------------------------------------
    print("\n== P2: subsetted 202 vs natively loaded 199, BITWISE ==")
    p2 = {}
    for k in H3:
        A = Z202[k][:, keep]
        B = Z199[k]
        md = float(np.abs(A - B).max())
        eq = bool(np.array_equal(A, B))
        p2[k] = {"max_abs_diff": md, "bitwise_equal": eq}
        print(f"  {k:8s} max|diff| {md:.6e}   bitwise equal: {eq}")
        del A
        gc.collect()
    OUT["P2"] = p2
    p2_ok = all(v["bitwise_equal"] for v in p2.values())
    print(f"  P2 -> {'CONFIRMED (bitwise equal)' if p2_ok else 'FALSIFIED'}")
    if not p2_ok:
        fail("P2 falsified: the subsetted matrix is not the native matrix")

    # ---- P5 NEGATIVE CONTROL: a different three columns must NOT match -------------------
    print("\n== P5 negative control: delete PLA1's three columns instead ==")
    pla_keep = np.array([pos[n] for n in n202 if n not in PLA1])
    p5 = {}
    for k in H3:
        A = Z202[k][:, pla_keep]
        md = float(np.abs(A - Z199[k]).max())
        p5[k] = md
        print(f"  {k:8s} max|diff| vs native 199: {md:.6e}")
        del A
        gc.collect()
    OUT["P5"] = p5
    p5_ok = all(v > 0 for v in p5.values())
    print(f"  P5 -> {'CONFIRMED (detectable)' if p5_ok else 'FALSIFIED -- P2 PASSES VACUOUSLY'}")
    if not p5_ok:
        fail("P5 falsified: the comparison cannot detect a wrong column set")

    # ---- P6 determinism, then P3 the AUCs -----------------------------------------------
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                 .split(np.zeros(len(y)), y))
    print(f"\n== P6 determinism control: crossfit twice on the SAME array (hybrid) ==")
    t1 = time.time()
    a1 = roc_auc_score(y, crossfit(Z199["hybrid"], y, folds))
    a2 = roc_auc_score(y, crossfit(Z199["hybrid"], y, folds))
    p6 = (a1 == a2)
    OUT["P6"] = {"a1": a1, "a2": a2, "identical": bool(p6)}
    print(f"  {a1:.9f} vs {a2:.9f}  diff {(a1-a2)*1e6:+.3e}e-6  -> "
          f"{'CONFIRMED' if p6 else 'FALSIFIED -- no e-6 AUC comparison is meaningful'}")
    if not p6:
        fail("P6 falsified: crossfit is not deterministic within a process")
    print(f"  ({time.time()-t1:.0f}s)", flush=True)

    print("\n== P3: seed-42 AUCs, subsetted arm vs natively loaded arm ==")
    oofA, oofB, rows = {}, {}, {}
    for k in H3:
        t1 = time.time()
        A = np.ascontiguousarray(Z202[k][:, keep])
        oofA[k] = crossfit(A, y, folds)
        del A
        gc.collect()
        oofB[k] = crossfit(Z199[k], y, folds)
        aa, bb = roc_auc_score(y, oofA[k]), roc_auc_score(y, oofB[k])
        rows[k] = {"subset": aa, "native": bb, "diff_e6": (aa - bb) * 1e6}
        print(f"  {k:8s} subset {aa:.9f}  native {bb:.9f}  "
              f"{(aa-bb)*1e6:+8.3f}e-6  ({time.time()-t1:.0f}s)", flush=True)
    aa = roc_auc_score(y, np.mean([rk(oofA[k]) for k in H3], 0))
    bb = roc_auc_score(y, np.mean([rk(oofB[k]) for k in H3], 0))
    rows["h3"] = {"subset": aa, "native": bb, "diff_e6": (aa - bb) * 1e6}
    print(f"  {'h3':8s} subset {aa:.9f}  native {bb:.9f}  {(aa-bb)*1e6:+8.3f}e-6")
    OUT["P3"] = rows
    worst = max(abs(v["diff_e6"]) for v in rows.values())
    p3 = worst < 0.5
    OUT["P3_worst_abs_e6"] = worst
    print(f"  P3: worst |diff| {worst:.3f}e-6, bar 0.5 -> "
          f"{'CONFIRMED' if p3 else 'FALSIFIED'}")
    if not p3:
        fail(f"P3 falsified: worst {worst:.3f}e-6")

    # ---- P4 the conditional reading, registered in the prereg ---------------------------
    print("\n== P4 (prereg section 4), the branch is chosen by P2/P3, not by me ==")
    if p2_ok and p3:
        verdict = ("R-M11f SURVIVES. The subsetted arm IS the native pack, bitwise. The "
                   "-5.095e-6 is NOT subsetting -- it is the cross-process rebuild floor, "
                   "w65a's 4e-6 P2 bar is too tight for it, and w65a's gate failure is a "
                   "FLOOR failure, not a LOAD failure. w29d's 190/194 promotions untouched.")
    elif not p2_ok:
        verdict = ("R-M11f IS FALSE. w65a's R1/R2 and w29d's 190-over-188 and 194-over-190 "
                   "promotions all inherit an uncontrolled term the size of the effect they "
                   "measure.")
    else:
        verdict = ("SPLIT: the matrices are bitwise equal but the FITS differ. The term is "
                   "purely numerical -- lbfgs on float32 at tol=1e-4 is not a function of "
                   "the data alone.")
    OUT["P4_verdict"] = verdict
    print("  " + verdict)

    out = os.path.join(ROOT, "experiments", "w68a_subsetgate.json")
    json.dump(OUT, open(out, "w"), indent=1)
    print(f"\nFAILURES {len(OUT['failures'])}\nwrote {out}")


if __name__ == "__main__":
    main()
