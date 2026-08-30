"""w127a -- ANGLE INDEX row 7 (seed and fold diversity) against its artefacts and its UNITS.

THE HANDED ANGLE. "Seed and fold diversity: same models across multiple seeds and fold splits,
averaged" -- row 7, closed since w64 (2026-08-11) and re-verified at the artefact level by w118.
Nothing here re-opens it. This prices the row's own PRICE CELL in the units the cell claims.

🔴 THE DEFECT. Row 7's price cell reads `structural null (stacker) · **ENROLMENT** +2e-6
(member)`. Three things are wrong with the parenthetical, and the third decides a closure:

  (a) THE LAYER. +2e-6 is a STACK-layer number -- the 2026-08-13 reading `+138e-6 solo ->
      +2e-6 stack` for seed-averaging `xgb_latcat`, which is where the workspace's 1.4%
      pass-through constant was fitted. The MEMBER-layer number for the same manoeuvre is
      +138e-6, which the row does not publish and which is ABOVE the 50e-6 floor.
  (b) THE GENUS. Rows 3 and 4 define ENROLMENT as "the value of ADDING a member the pack does
      not hold". The +2e-6 measures arm B of w109 -- three seed twins REPLACED by their mean.
      Nothing is added; the pack loses two columns. That is a SUBSTITUTION price.
  (c) THE SCOPE, and it is the one that matters. Rows 3/4 put a rate denominator in exactly
      this position (`5.9e-6/member`, `+10.04e-6/member`, `+7.38e-6/member`); rows 5/6 put a
      LAYER word there (`MEMBER layer`, `TOP-LEVEL layer, k=4`). Read "+2e-6 (member)" with
      rows 3/4's convention, at the pack's k=104, and it multiplies out to +208e-6 -- FOUR
      TIMES the 50e-6 floor, which would re-open the row. Its real scope is k=1.

⚠ #55 `w126c_scopeguard` IS BLIND TO THIS BY ITS OWN DOCSTRING, WHICH SAYS SO: it only inspects
cells carrying both a magnitude and the word `SEARCH`, "so a wrongly-scoped ENROLMENT price
walks straight past it". w126 wrote that sentence naming this exact cell shape one run before
the harness handed this row. The prediction is now five-for-five.

WHAT THIS MEASURES, PRE-REGISTERED IN `w127_prereg.txt` BEFORE ANY NUMBER EXISTED. The ENROLMENT
price of seed diversity in rows 3/4's units, with rows 3/4's instrument (paired 50/50
StratifiedShuffleSplit at random_state 0,1,2; LogisticRegression C=1.0; transform `hybrid`), ⚠ ON THE
FULL LOADED PACK (167 members), NOT on rows 3/4's `base104`: base104 is the subset not in
`_vetting.csv` and it holds no `xgb_latcat`, so these arms cannot live there. The base104
CatBoost control anchors the INSTRUMENT to w123/w124; it does not make the arms' pool theirs,
and this run does not claim it does. Five nested arms P/Q/R/S/T; see the prereg for the design
and for the registered intervals and the falsifier.

THE CONTROL, and why it is the same one w124 used. `+cat_only` is re-measured in the SAME
process on the SAME splits and must reproduce w123's +10.0416e-6/member to 0.05e-6. If it does
not, this run's arms are not comparable to rows 3/4 and must not be published in their units.
A number in someone else's units needs their instrument to be verified in-process, not cited.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  R1 +   THE PUBLISHED ARTEFACTS. w118's member arm reproduces from the OOF arrays: three
         seed OOF AUCs, their probability mean, the +138.2e-6, and `avg3` IS that mean exactly.
  R2 +-  THE SPAN ARGUMENT, MEASURED NOT ASSERTED. `avg3` in the span of its three seeds is a
         claim about rank. Reported as the smallest singular value of the 4-column block and
         as the residual of a least-squares fit of `avg3` on the three seeds -- both must be
         at machine zero, and the same two statistics on a NON-derived 4-column block from the
         same pool must NOT be, else the statistic is measuring nothing.
  R3 +-  THE ENROLMENT ARMS, with the CatBoost control in-process (above).
  R4 +   THE ARITHMETIC OF THE MISREADING. 104 x the cell's magnitude against the 50e-6 floor,
         printed with both readings so the consequence is on the record either way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, TARGET, load_raw           # noqa: E402
from stack import DEFAULT_DROP, load_members, transform  # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
OOFDIR = os.path.join(ROOT, "oof")
OUT = os.path.join(HERE, "w127a_row7.json")

REPS = 3
CVAL = 1.0
TRANSFORM = "hybrid"
DECORR_MAX = 0.97

SEEDS = ["xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23"]
AVG = "xgb_latcat_avg3"

# Published, frozen as literals: this is a COMPARISON, not a recomputation that agrees with
# itself. Sources named per claim.
PUBLISHED = {
    # R1 -- w118's member arm, quoted from the "ROW 7 OF THE ANGLE INDEX RE-VERIFIED" section
    "solo": {"xgb_latcat": 0.967696, "xgb_latcat_s17": 0.967750, "xgb_latcat_s23": 0.967766},
    "probmean": 0.967904,
    "seedavg_solo": 138.2e-6,
    "solo_tol": 1e-6,
    # the row 7 PRICE cell, and the constants it is read against
    "row7_stack": 2e-6,
    "noise_floor": 50e-6,
    "pack_k": 104,
    # R3 -- the control, from w123a_row3.json via w124a_row4.py
    "cat_only_per_member": 10.041599941293389e-06,
    "cat_only_n": 8,
    "control_tol": 0.05e-6,
    # the registered intervals, from w127_prereg.txt
    "pred_span_max": 5e-6,
    "pred_enrol_lo": -5e-6,
    "pred_enrol_hi": 15e-6,
}


def paired(Z, y, idx, variants, reps=REPS):
    rows = []
    for rep in range(reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {}
        for k, mem in variants.items():
            cols = [idx[m] for m in mem]
            m = LogisticRegression(max_iter=3000, C=CVAL).fit(Z[np.ix_(iA, cols)], y[iA])
            r[k] = roc_auc_score(y[iB], m.predict_proba(Z[np.ix_(iB, cols)])[:, 1])
        rows.append(r)
        print("    split %d: " % rep + "  ".join(f"{k}={v:.6f}" for k, v in r.items()),
              flush=True)
    return pd.DataFrame(rows)


def span_stats(M, target_col):
    """(smallest singular value of M, lstsq residual of target on the other columns)."""
    sv = float(np.linalg.svd(M, compute_uv=False)[-1])
    others = np.delete(M, target_col, axis=1)
    coef, *_ = np.linalg.lstsq(others, M[:, target_col], rcond=None)
    resid = float(np.max(np.abs(others @ coef - M[:, target_col])))
    return sv, resid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the paired arms (R3)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w127a -- ANGLE INDEX row 7 (seed and fold diversity) against its artefacts\n")

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    # ---------------------------------------------------------------- R1, w118's member arm
    print("[R1] w118's member arm, recomputed from oof/")
    solo = {}
    for n in SEEDS:
        p = os.path.join(OOFDIR, f"oof_{n}.npy")
        if not os.path.exists(p):
            fail(f"row 7's evidence array oof_{n}.npy is not on disk")
            continue
        v = np.load(p)
        solo[n] = float(roc_auc_score(y, v))
        want = PUBLISHED["solo"][n]
        mark = "OK " if abs(solo[n] - want) <= PUBLISHED["solo_tol"] else "-> "
        print(f"    {mark}{n:<20s} {solo[n]:.6f}   (published {want:.6f})")
        if abs(solo[n] - want) > PUBLISHED["solo_tol"]:
            fail(f"{n} recomputes at {solo[n]:.6f}, w118 published {want:.6f}")
    if len(solo) == 3:
        V = np.column_stack([np.load(os.path.join(OOFDIR, f"oof_{n}.npy")) for n in SEEDS])
        pm = float(roc_auc_score(y, V.mean(axis=1)))
        gain = pm - max(solo.values())
        print(f"    probability mean {pm:.6f}   (published {PUBLISHED['probmean']:.6f})")
        print(f"    over the best single: {gain*1e6:+.1f}e-6   (published "
              f"{PUBLISHED['seedavg_solo']*1e6:+.1f}e-6)")
        if abs(pm - PUBLISHED["probmean"]) > PUBLISHED["solo_tol"]:
            fail(f"the probability mean recomputes at {pm:.6f}")
        if abs(gain - PUBLISHED["seedavg_solo"]) > PUBLISHED["solo_tol"]:
            fail(f"the +138.2e-6 recomputes at {gain*1e6:+.1f}e-6")
        pa = os.path.join(OOFDIR, f"oof_{AVG}.npy")
        if os.path.exists(pa):
            md = float(np.max(np.abs(np.load(pa) - V.mean(axis=1))))
            print(f"    max|{AVG} - mean(seeds)| = {md:.3e}   (w109a relation 2: exactly 0.0)")
            if md != 0.0:
                fail(f"{AVG} is not exactly the seed mean; max|diff| = {md:.3e}")
        else:
            fail(f"oof_{AVG}.npy is not on disk")
        out["member_arm"] = {"solo": solo, "probmean": pm, "gain": gain}
        note(f"the MEMBER-layer number for this manoeuvre is {gain*1e6:+.1f}e-6, which is "
             f"ABOVE the {PUBLISHED['noise_floor']*1e6:.0f}e-6 floor. The row's cell publishes "
             f"{PUBLISHED['row7_stack']*1e6:+.0f}e-6 under the word `(member)`. They are "
             f"different layers of the same manoeuvre and differ by "
             f"{gain/PUBLISHED['row7_stack']:.0f}x.")

    # ---------------------------------------------------------------- the pool
    # ⚠ THE POOL IS NOT base104 AND SAYING SO WOULD BE THIS RUN'S OWN DEFECT.
    # `load_members` returns the FULL loaded pack (167 today); rows 3/4's `base104` is the
    # subset NOT in `_vetting.csv`, and it holds no `xgb_latcat`, so these arms cannot live
    # there. The P..T arms run on the full pack -- the one that ships -- and ONLY the
    # catbase/catbase+cat control arms are base104. The control anchors the INSTRUMENT to
    # w123/w124; it does not make the arms' pool theirs.
    print("\n[pool] the FULL loaded pack via load_members(drop=DEFAULT_DROP); the base104 "
          "control group is carved out of it in R3")
    names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    idx = {n: i for i, n in enumerate(names)}
    print(f"    {len(names)} members loaded")
    out["pool_n"] = len(names)
    missing = [n for n in SEEDS + [AVG] if n not in idx]
    if missing:
        fail(f"the seed family is not in the loaded pool: {missing}")
        json.dump(out, open(OUT, "w"), indent=1)
        return 1

    # ---------------------------------------------------------------- R2, the span argument
    print("\n[R2] is `avg3` in the span of its three seeds? (rank, not rhetoric)")
    Z, _ = transform(O, T, TRANSFORM)
    blk = Z[:, [idx[n] for n in SEEDS + [AVG]]]
    sv, resid = span_stats(blk, 3)
    print(f"    seed block + avg3   smallest singular value {sv:.4e}   lstsq max|resid| "
          f"{resid:.4e}")
    # the negative direction: the same two statistics on a block that is NOT derived
    ctrl_names = [n for n in names if n not in SEEDS + [AVG]][:4]
    cblk = Z[:, [idx[n] for n in ctrl_names]]
    csv_, cresid = span_stats(cblk, 3)
    print(f"    control block       smallest singular value {csv_:.4e}   lstsq max|resid| "
          f"{cresid:.4e}   ({', '.join(ctrl_names)})")
    derived = sv < 1e-8 * np.linalg.norm(blk) and resid < 1e-6
    ctrl_not = csv_ > 1e-8 * np.linalg.norm(cblk) and cresid > 1e-6
    print(f"    {'OK ' if derived else '-> '}the avg3 block is rank-deficient")
    print(f"    {'OK ' if ctrl_not else '-> '}the control block is NOT (else the statistic "
          f"measures nothing)")
    if not derived:
        note("in the `hybrid` transform `avg3` is NOT an exact linear function of its three "
             "seeds -- the transform is applied per column, and rank-averaging is not linear. "
             "The span argument holds in PROBABILITY space (R1, max|diff| 0.0) and is "
             "approximate here; R3 is the measurement that decides, not this.")
    if not ctrl_not:
        fail("the control block is also rank-deficient; the span statistic is not "
             "discriminating and R2 proves nothing")
    out["span"] = {"sv": sv, "resid": resid, "ctrl_sv": csv_, "ctrl_resid": cresid,
                   "ctrl_names": ctrl_names}

    # ---------------------------------------------------------------- R3, the enrolment arms
    if a.quick:
        print("\n[R3] --quick: paired arms skipped")
    else:
        # the CatBoost control group, reconstructed exactly as w124 does
        vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
        new = [n for n in names if n in vet.index]
        bei = [n for n in new if n.startswith("bei_")]
        lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
        decorr = [n for n in new
                  if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
        rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]
        cat = sorted(n for n in rest if "cat" in set(n.split("_")))
        catbase = [n for n in names if n not in vet.index]
        if len(cat) != PUBLISHED["cat_only_n"]:
            fail(f"the CatBoost control group reconstructs at n={len(cat)}, w123 measured "
                 f"n={PUBLISHED['cat_only_n']}")

        P = [n for n in names if n not in SEEDS + [AVG]]
        variants = {
            "P": P,
            "Q": P + [SEEDS[0]],
            "R": P + SEEDS,
            "S": P + [SEEDS[0], AVG],
            "T": P + SEEDS + [AVG],
            "catbase": catbase,
            "catbase+cat": catbase + cat,
        }
        print(f"\n[R3] paired 50/50, {REPS} splits, C={CVAL}, transform={TRANSFORM} "
              f"(w123/w124's procedure and seeds).  P={len(P)}  T={len(variants['T'])}")
        rob = paired(Z, y, idx, variants)

        print("\n    THE CONTROL FIRST -- w123's CatBoost rate, in this process, these splits")
        dc = rob["catbase+cat"] - rob["catbase"]
        cpm = float(dc.mean() / len(cat))
        gap = cpm - PUBLISHED["cat_only_per_member"]
        print(f"      +cat_only n={len(cat)}  {cpm*1e6:+.4f}e-6/member  against w123's "
              f"{PUBLISHED['cat_only_per_member']*1e6:+.4f}e-6  (gap {gap*1e6:+.4f}e-6)")
        comparable = abs(gap) <= PUBLISHED["control_tol"]
        if not comparable:
            fail(f"the CatBoost control does not reproduce w123 (gap {gap*1e6:+.4f}e-6); the "
                 f"arms below are NOT in rows 3/4's units and must not be published in them")
        out["control"] = {"per_member": cpm, "gap": gap, "n": len(cat),
                          "comparable": comparable}

        print("\n    THE ENROLMENT ARMS (paired; split noise cancels)")
        arms = {
            "seeds added as members  (R-Q)/2": (rob["R"] - rob["Q"], 2),
            "the seed AVERAGE alone  (S-Q)/1": (rob["S"] - rob["Q"], 1),
            "the average ON TOP      (T-R)/1": (rob["T"] - rob["R"], 1),
            "the whole family        (T-P)/4": (rob["T"] - rob["P"], 4),
        }
        res = {}
        for label, (d, n) in arms.items():
            sign = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
            pm_ = float(d.mean() / n)
            res[label] = {"n": n, "delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                          "per_member": pm_, "sign": sign}
            print(f"      {label:<34s} {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
                  f"{pm_*1e6:+8.2f}e-6/member  [{sign}]")
        out["arms"] = res

        # the registered predictions, scored
        print("\n    THE REGISTERED PREDICTIONS, SCORED")
        top = res["the average ON TOP      (T-R)/1"]["delta"]
        p1 = abs(top) < PUBLISHED["pred_span_max"]
        print(f"      1  |T-R| < {PUBLISHED['pred_span_max']*1e6:.0f}e-6 : "
              f"{abs(top)*1e6:.2f}e-6  {'HELD' if p1 else 'MISSED'}")
        for key in ("seeds added as members  (R-Q)/2", "the seed AVERAGE alone  (S-Q)/1"):
            v = res[key]["per_member"]
            ok = PUBLISHED["pred_enrol_lo"] <= v <= PUBLISHED["pred_enrol_hi"]
            print(f"      2  {key} in [{PUBLISHED['pred_enrol_lo']*1e6:+.0f}, "
                  f"{PUBLISHED['pred_enrol_hi']*1e6:+.0f}]e-6/member : {v*1e6:+.2f}e-6  "
                  f"{'HELD' if ok else 'MISSED'}")
        reopen = res["seeds added as members  (R-Q)/2"]["per_member"] > PUBLISHED["noise_floor"]
        print(f"      3  DOES ROW 7 RE-OPEN?  (R-Q)/2 vs the "
              f"{PUBLISHED['noise_floor']*1e6:.0f}e-6 floor : "
              f"{'*** YES, SAY SO ***' if reopen else 'NO'}")
        out["reopen"] = bool(reopen)

    # ---------------------------------------------------------------- R4, the misreading
    print("\n[R4] the arithmetic of the misreading, on the record either way")
    k = PUBLISHED["pack_k"]
    as_rate = PUBLISHED["row7_stack"] * k
    print(f"    read as a RATE (rows 3/4's convention): {PUBLISHED['row7_stack']*1e6:+.0f}e-6 "
          f"x k={k} = {as_rate*1e6:+.0f}e-6, which is "
          f"{'ABOVE' if as_rate > PUBLISHED['noise_floor'] else 'below'} the "
          f"{PUBLISHED['noise_floor']*1e6:.0f}e-6 floor")
    print(f"    read at its real scope k=1: {PUBLISHED['row7_stack']*1e6:+.0f}e-6, which is "
          f"below the floor")
    note(f"the parenthetical decides which side of the floor the row lands on: "
         f"{as_rate*1e6:+.0f}e-6 against {PUBLISHED['row7_stack']*1e6:+.0f}e-6, a factor of {k}.")
    out["misreading"] = {"as_rate": as_rate, "as_total": PUBLISHED["row7_stack"], "k": k}

    out["fails"], out["notes"] = fails, notes
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")
    print(f"FAILURES: {len(fails)}")
    for f in fails:
        print("  - " + f)
    if notes:
        print(f"NOTES: {len(notes)}")
        for n in notes:
            print("  - " + n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
