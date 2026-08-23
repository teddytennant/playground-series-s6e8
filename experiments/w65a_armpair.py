"""w65a -- the 202-vs-199 paired partition contrast, plus TWO matched placebo drops.

Registered in experiments/w65_prereg.txt, committed 7193479 BEFORE this file existed.

WHY. `check_selection.WANTED` slot 1 is `w36_ad199stdcorr`. Its nearest rival
`w38_ad202stdcorr` is -2.417e-6 away on the stored levels, and `check_selection.py` says of
exactly this situation: "Ranking these three files by their stored levels is not a
measurement." The 190-over-188 and 194-over-190 decisions were both made on w27w/w29d's
paired-partition instrument; the 199-over-202 comparison has never been run on it. This is
w29d_partition.py moved up eight members, with the placebo arms w29d did not have.

R-M11f, unchanged: ONE load serves all four arms -- ARM 199 is the ARM 202 matrix with the
three ext_members14 columns deleted, and every transform in agent/stack.py is strictly
per-column, so subsetting after transform+scale is exactly equal to loading 199 members. That
claim is GATED, not assumed (prereg P2), and the gate compares against AUCs RECOMPUTED from
the shipped .npy files in this process -- never against a hard-typed literal.

    .venv/bin/python experiments/w65a_armpair.py --seeds 42,101,13,7
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

H3 = ("hybrid", "rankraw", "rescale")          # the deadline mix; it EXCLUDES logit
# w36b_run.sh / w38d_run.sh, verbatim.
DROP = ("golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,lat_ctraw_r400,"
        "lat_ctfixte_r400,om_cat")
XDIRS = tuple(os.path.join(DATA, d) for d in
              ("ext_members3", "ext_members4", "ext_members6", "ext_members7pin",
               "ext_members8", "ext_members10", "ext_members11", "ext_members12",
               "ext_members14"))
NEW = ("lexb_cat_base", "lexb_lgb02", "lexb_xgb_base")   # ext_members14, the three under test
BIG, SMALL = 202, 199
PLACEBO_SEED = 65          # prereg section 2, fixed before any number existed
N_PLACEBO = 2

SUBS = os.path.join(ROOT, "submissions")
STEM = {BIG: "w38_ad202std", SMALL: "w36_ad199std"}       # for the gate only

BAR_GATE = 4e-6            # prereg P2
BAR_VOID = 2e-5            # a cell off by more than this is a LOAD error
R1_MEAN = 2.0              # prereg R1(ii), e-6
R1_POS = 3                 # prereg R1(i), of 4
P8_SPREAD = 0.5            # prereg P8, e-6

OUT = {"prereg": "experiments/w65_prereg.txt", "failures": []}


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


def shipped_auc(y):
    """The gate's reference values, recomputed from the .npy files this run. A COMPARISON,
    not a stamp: if the file moves or is rebuilt, this moves with it."""
    ref = {}
    for arm, stem in STEM.items():
        for k in H3:
            p = os.path.join(SUBS, f"oof_{stem}_{k}.npy")
            ref[(arm, k)] = roc_auc_score(y, np.load(p))
        p = os.path.join(SUBS, f"oof_{stem}_h3.npy")
        ref[(arm, "h3")] = roc_auc_score(y, np.load(p))
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,13,7")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    for d in XDIRS:
        if not os.path.isdir(d):
            raise SystemExit(f"R-M11e: extra dir does not exist: {d}")
    t0 = time.time()
    names, y, mats, _te = load_all(H3, DROP.split(","), extra_dirs=XDIRS, dtype="float32")
    if len(names) != BIG:
        raise SystemExit(f"R-M11e: expected {BIG} members, got {len(names)}")
    missing = [n for n in NEW if n not in names]
    if missing:
        raise SystemExit(f"R-M11e: the members under test are not in the load: {missing}")

    keep = np.array([i for i, n in enumerate(names) if n not in NEW])
    if len(keep) != SMALL:
        raise SystemExit(f"R-M11e: expected to drop {BIG-SMALL} for the {SMALL} arm, "
                         f"dropped {BIG-len(keep)}")

    # --- the placebo arms: the SAME operator on three columns that are not under test -------
    rng = np.random.default_rng(PLACEBO_SEED)
    pool = [i for i, n in enumerate(names) if n not in NEW]
    picks, taken = [], set()
    for _ in range(N_PLACEBO):
        while True:
            c = tuple(sorted(rng.choice(pool, size=len(NEW), replace=False).tolist()))
            if not (set(c) & taken):
                break
        taken |= set(c)
        picks.append(c)
    arms = {BIG: None, SMALL: keep}
    for j, c in enumerate(picks, 1):
        drop = set(c)
        arms[f"PLA{j}"] = np.array([i for i in range(len(names)) if i not in drop])
        print(f"  PLA{j} drops {[names[i] for i in c]}", flush=True)
        if len(arms[f"PLA{j}"]) != SMALL:
            raise SystemExit(f"placebo {j} kept {len(arms[f'PLA{j}'])}, expected {SMALL}")
    OUT["placebo_dropped"] = {f"PLA{j}": [names[i] for i in c]
                              for j, c in enumerate(picks, 1)}
    print(f"{BIG} members, loaded in {time.time()-t0:.0f}s; arms {list(arms)}", flush=True)

    ref = shipped_auc(y)
    OUT["shipped_ref"] = {f"{k[0]}/{k[1]}": v for k, v in ref.items()}

    # standardise exactly as blend_lab.build(std=True) does: per column, from the OOF side,
    # computed on the FULL 202-column matrix. Per-column scaling means the subset arms see
    # exactly the scale they would see if loaded alone.
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

    res = {}
    for sd in seeds:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=sd)
                     .split(np.zeros(len(y)), y))
        res[sd] = {}
        for arm, cols in arms.items():
            oof, row = {}, {}
            for k in H3:
                t1 = time.time()
                M = Z[k] if cols is None else Z[k][:, cols]
                oof[k] = crossfit(M, y, folds, a.C)
                row[k] = roc_auc_score(y, oof[k])
                print(f"  seed {sd:>3d} arm {str(arm):>4s} {k:8s} {row[k]:.7f} "
                      f"({time.time()-t1:.0f}s)", flush=True)
                del M
                gc.collect()
            row["h3"] = roc_auc_score(y, np.mean([rk(oof[k]) for k in H3], 0))
            print(f"  seed {sd:>3d} arm {str(arm):>4s} {'h3':8s} {row['h3']:.7f}", flush=True)
            res[sd][arm] = row
    OUT["res"] = {str(s): {str(arm): r for arm, r in v.items()} for s, v in res.items()}

    # ---- P2: the gate, BOTH arms, against values recomputed this process ------------------
    print("\n== P2 GATE: seed-42 cells of BOTH real arms vs the shipped OOF vectors ==")
    ok = 0
    if 42 in res:
        for arm in (BIG, SMALL):
            for k in (*H3, "h3"):
                got, want = res[42][arm][k], ref[(arm, k)]
                d = (got - want) * 1e6
                good = abs(got - want) < BAR_GATE
                ok += good
                print(f"  arm {arm} {k:8s} got {got:.7f}  shipped {want:.7f}  "
                      f"{d:+7.2f}e-6  {'OK' if good else '*** OVER 4e-6 ***'}")
                if abs(got - want) > BAR_VOID:
                    raise SystemExit(f"  *** {arm}/{k} off by more than {BAR_VOID:.0e} -- "
                                     f"that is a LOAD error, not the floor. All void.")
        OUT["p2_ok"] = int(ok)
        print(f"  P2: {ok}/8 cells inside {BAR_GATE:.0e}"
              f"  -> {'CONFIRMED' if ok == 8 else 'FALSIFIED'}")
        if ok != 8:
            fail(f"P2 falsified: {ok}/8")
    else:
        fail("seed 42 not run -- P2 cannot be evaluated and R1 is void")

    # ---- the paired deltas ---------------------------------------------------------------
    def deltas(arm):
        return np.array([(res[sd][BIG][k := "h3"] - res[sd][arm][k]) * 1e6 for sd in seeds])

    print(f"\n== paired {BIG} - arm delta on identical partitions (e-6) ==")
    print(f"  {'arm':>5s} {'seed':>5s}  " + "  ".join(f"{k:>9s}" for k in (*H3, "h3")))
    per_arm = {}
    for arm in arms:
        if arm == BIG:
            continue
        for sd in seeds:
            d = {k: (res[sd][BIG][k] - res[sd][arm][k]) * 1e6 for k in (*H3, "h3")}
            print(f"  {str(arm):>5s} {sd:>5d}  " + "  ".join(f"{d[k]:+9.2f}" for k in (*H3, "h3")))
        per_arm[arm] = deltas(arm)
        v = per_arm[arm]
        print(f"  {str(arm):>5s}  h3 mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  "
              f"se {v.std(ddof=1)/np.sqrt(len(v)):.3f}  positive {int((v>0).sum())}/{len(v)}\n")
    OUT["h3_delta"] = {str(k): v.tolist() for k, v in per_arm.items()}

    lex = per_arm[SMALL]
    mean, sd_, npos = float(lex.mean()), float(lex.std(ddof=1)), int((lex > 0).sum())
    OUT["lex"] = {"mean": mean, "sd": sd_, "npos": npos, "n": len(lex)}

    # ---- R1 ------------------------------------------------------------------------------
    r1 = (npos >= R1_POS) and (mean >= R1_MEAN)
    OUT["R1"] = {"fires": bool(r1), "npos": npos, "bar_pos": R1_POS,
                 "mean": mean, "bar_mean": R1_MEAN}
    print("== R1 (prereg section 3) ==")
    print(f"  (i)  positive in {npos}/{len(lex)}, needs >= {R1_POS}: {npos >= R1_POS}")
    print(f"  (ii) mean {mean:+.3f}e-6, needs >= {R1_MEAN:+.1f}: {mean >= R1_MEAN}")
    print(f"  R1 {'FIRES -- WANTED slot 1 moves to w38_ad202stdcorr' if r1 else 'DOES NOT FIRE -- WANTED slot 1 stays w36_ad199stdcorr'}")

    # ---- P8 then R2 ----------------------------------------------------------------------
    pm = {k: float(v.mean()) for k, v in per_arm.items() if str(k).startswith("PLA")}
    spread = abs(pm["PLA1"] - pm["PLA2"])
    p8 = spread > P8_SPREAD
    OUT["P8"] = {"spread": spread, "bar": P8_SPREAD, "pass": bool(p8)}
    print(f"\n== P8 power control: |PLA1 - PLA2| = {spread:.3f}e-6, "
          f"needs > {P8_SPREAD}: {'CONFIRMED' if p8 else 'FALSIFIED -- R2 IS VOID'}")
    lo, hi = min(pm.values()), max(pm.values())
    r2 = (lo <= mean <= hi) if p8 else None
    OUT["R2"] = {"fires": r2, "placebo": pm, "lo": lo, "hi": hi, "lex_mean": mean}
    print(f"== R2: placebo h3 means PLA1 {pm['PLA1']:+.3f}  PLA2 {pm['PLA2']:+.3f}  "
          f"range [{lo:+.3f}, {hi:+.3f}]; lexb {mean:+.3f}")
    print(f"  R2 {'VOID (P8 failed)' if r2 is None else ('FIRES -- the three lexb members are not distinguishable from deleting any three columns' if r2 else 'DOES NOT FIRE -- the lexb drop is outside the placebo range')}")

    # ---- P3/P3b/P4/P5/P6 -----------------------------------------------------------------
    ver = {
        "P3  mean < 0": mean < 0,
        "P3b mean in [-3.9,-0.9]": -3.9 <= mean <= -0.9,
        "P4  sd in [0.5,3.0]": 0.5 <= sd_ <= 3.0,
        "P5  R1 does not fire": not r1,
        "P6  R2 fires": bool(r2) if r2 is not None else None,
    }
    print("\n== registered predictions ==")
    for k, v in ver.items():
        print(f"  {k:28s} {'CONFIRMED' if v is True else ('VOID' if v is None else 'FALSIFIED')}")
    OUT["predictions"] = ver

    # ---- P7 load-integrity control -------------------------------------------------------
    print("\n== P7 off-seed mean minus seed 42 (w14c's mechanism) ==")
    off = [s for s in seeds if s != 42]
    worst, cells = None, 0
    if 42 in res and off:
        for arm in arms:
            for k in (*H3, "h3"):
                d = (np.mean([res[s][arm][k] for s in off]) - res[42][arm][k]) * 1e6
                cells += 1
                worst = d if worst is None else min(worst, d)
                print(f"  arm {str(arm):>4s} {k:8s} {d:+8.2f}e-6")
        p7 = worst > 8.0
        OUT["P7"] = {"worst": worst, "cells": cells, "pass": bool(p7)}
        print(f"  P7: worst cell {worst:+.2f}e-6 over {cells}, needs > +8.0: "
              f"{'CONFIRMED' if p7 else 'FALSIFIED'}")
        if not p7:
            fail(f"P7 falsified: worst off-seed lift {worst:+.2f}e-6 -- suspect the load")

    out = os.path.join(ROOT, "experiments", "w65a_armpair.json")
    json.dump(OUT, open(out, "w"), indent=1)
    print(f"\nFAILURES {len(OUT['failures'])}\nwrote {out}")


if __name__ == "__main__":
    main()
