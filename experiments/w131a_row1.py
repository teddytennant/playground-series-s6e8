"""w131a -- ANGLE INDEX row 1 (the original dataset) against its artefacts and its BASELINE.

THE HANDED ANGLE. "Original dataset: find the real source dataset this synthetic data was
generated from, and concatenate it as extra training rows. Historically the single biggest
edge in Playground Series." Row 1, closed 2026-08-11 on all three routes and re-verified at
the artefact level by w112. NOTHING HERE RE-OPENS IT. This prices the row's own PRICE CELL.

🔴 THE DEFECT, AND I WROTE IT ONE RUN AGO. The cell reads:

    a **CONCAT** price (extra training ROWS, not members). **0 measured against the same
    stack trained on `train.csv` alone (0× dose), and the usual Playground edge is INVERTED
    here: −58e-6 at 1× dose, −3,340e-6 at 50×; ...**

w130 added "measured against the same stack trained on `train.csv` alone (0× dose)" BY HAND,
in the same pass that registered #59, to turn #59 green on this row.

  (a) THE NAMED BASELINE IS THE ARM. `0× dose` IS "the same stack trained on train.csv
      alone". `orig_concat.py` makes this literal -- its dose loop is `if w:`, so at w=0 no
      frame is built at all. The headline therefore compares a frame with itself: zero for
      any model, any metric, any seed, with ZERO DEGREES OF FREEDOM, before anything is fit.
      #59 asks that a zero NAME a baseline. It does not ask that the baseline be a CONTRAST.
  (b) THE PRICE COLUMN IS READ BY ITS HEADLINE, and row 1 is the ONE row whose manoeuvre is
      measured to LOSE AUC at every setting anyone ran. Printed as `0` in a column where
      +10.04e-6 is published as a reason NOT to build, it ranks as the cheapest row there is.
  (c) ROW 8 GOT THIS TREATMENT ONE RUN AGO AND ROW 1 DID NOT. w130 demoted row 8's `0` to a
      labelled REPEAT price and published the ABSENCE numbers ahead of it; row 1's `0` was
      handed a baseline clause and nothing else. Same defect, one of them fixed.

⟹ THE FOURTH PLACE A GREEN MEANS "I DID NOT LOOK", after w130's three: A CELL EDITED BY HAND
TO SATISFY A GUARD, because the only reader it was ever tested against is the one it was
written for.

WHAT THIS MEASURES, pre-registered in `w131_prereg.txt` before any number existed, falsifier
included. Four arms, one per clause of the row's own elaboration.

  A  FRAME layer. Rows appended at dose 0/1/10/50 and max|frame(0) - train|. Not a price.
  B  MEMBER-TRAINING-SET layer, k=1, a per-competition TOTAL. The dose ladder from
     `logs_orig_concat.txt`, against the 50e-6 floor, with monotonicity MEASURED.
  C  STACK layer, per member. The separate-estimator route priced as an ENROLMENT in rows
     3/4/7/9's units for the first time -- paired 50/50 StratifiedShuffleSplit rs 0,1,2;
     LogisticRegression C=1.0; transform `hybrid`; the FULL loaded pack (⚠ NOT `base104`,
     which is only the control group carved out of it) -- with the base104 CatBoost control
     re-measured IN THIS PROCESS and required to reproduce w123's +10.0416e-6/member.
  D  IDENTITY layer. md5 of the original CSV against the frozen literal recorded when the
     deleted official dataset was identified. Not a price.

⛔ FOUR CURRENCIES -- a row count, OOF AUC at k=1, AUC per member, and a hash. THE ARMS DO
NOT ADD. Same genus as w130's row-8 arms and the standing DO-NOT on measured-alone deltas.

CONTROLS (w72 5.3: a control that can only fail is not a control; both directions or decoration)
  R1 +-  ARM A FIRES BOTH WAYS. Dose 0 must append exactly 0 rows and reproduce the training
         frame exactly; doses 1/10/50 must each append exactly w*7500 and NOT reproduce it.
         "Dose 0 changes nothing" alone would pass on a broken frame builder.
  R2 +-  ARM B's LADDER IS COMPARED TO FROZEN LITERALS, not recomputed against itself, and
         monotonicity is measured over the four rungs rather than asserted.
  R3 +-  ARM C carries the in-process CatBoost control. If it does not reproduce w123 to
         0.05e-6 the arm is not in rows 3/4's units and must not be published in them.
  R4 +   THE ARITHMETIC OF THE MISREADING, printed under both readings so the consequence is
         on the record either way.

    .venv/bin/python experiments/w131a_row1.py            # full, ~40m (R3 fits the pack)
    .venv/bin/python experiments/w131a_row1.py --quick    # A, B, D only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from common import CAT, DATA, NUM, TARGET, get_folds, load_raw          # noqa: E402
from stack import DEFAULT_DROP, load_members, transform                 # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
ORIGDIR = os.path.join(DATA, "orig")
ORIGCSV = os.path.join(ORIGDIR, "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
CONCAT_LOG = os.path.join(ROOT, "logs_orig_concat.txt")
OUT = os.path.join(HERE, "w131a_row1.json")

REPS, CVAL, TRANSFORM, DECORR_MAX = 3, 1.0, "hybrid", 0.97
DOSES = (0, 1, 10, 50)

# Frozen literals. Sources named per claim; this is a COMPARISON, not a recomputation that
# agrees with itself.
PUBLISHED = {
    # R2 -- logs_orig_concat.txt, quoted in the row 1 price cell
    "ladder": {0: 0.962639, 1: 0.962581, 10: 0.961653, 50: 0.959299},
    "cell_1x_e6": -58.0,
    "cell_50x_e6": -3340.0,
    "ladder_tol_e6": 1.0,
    # the constants the table is read against
    "noise_floor_e6": 50.0,
    "row3_catboost_e6": 10.041599941293389,   # w123a_row3.json, via w124/w127
    "control_tol_e6": 0.05,
    "orig_rows": 7500,
    # R4 -- the deleted official original, identified 2026-08-13
    "orig_md5": "d831a326bc6f0ab76056a12279cb0047",
    # the registered interval for arm C, from w131_prereg.txt prediction 4
    "pred_enrol_abs_max_e6": 50.0,
}


def as_frame(df, vocab):
    """orig_concat.py's frame builder, reproduced so arm A measures the real code path."""
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=vocab[c])
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the paired arm (R3)")
    a = ap.parse_args()

    fails, notes = [], []
    out = {"published": PUBLISHED}

    def fail(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    def note(msg):
        notes.append(msg)
        print("  note " + msg)

    print("w131a -- ANGLE INDEX row 1 (the original dataset) against its own baseline\n")

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    orig = pd.read_csv(ORIGCSV)

    # ------------------------------------------------------- R1 / ARM A, the construction zero
    print("[R1 / ARM A] FRAME layer -- what dose 0 actually appends")
    vocab = {c: sorted(set(orig[c].dropna()) | set(tr[c].dropna())) for c in CAT}
    X = as_frame(tr, vocab)
    Xo = as_frame(orig, vocab)
    yo = orig[TARGET].to_numpy(dtype=float)
    folds = get_folds(y)
    a_idx, _ = folds[0]
    base_Xa, base_ya = X.iloc[a_idx], y[a_idx]
    print(f"    fold 0 training half: {len(base_Xa):,} rows; the originals: {len(Xo):,} rows")
    arm_a = {}
    for w in DOSES:
        # orig_concat.py's own dose branch, verbatim: `if w:`
        if w:
            Xa = pd.concat([base_Xa] + [Xo] * w, ignore_index=True)
            ya = np.concatenate([base_ya] + [yo] * w)
        else:
            Xa, ya = base_Xa, base_ya
        appended = len(Xa) - len(base_Xa)
        head = Xa.iloc[:len(base_Xa)]
        num_gap = float(np.nanmax(np.abs(head[NUM].to_numpy(dtype=float)
                                         - base_Xa[NUM].to_numpy(dtype=float))))
        same_len = appended == 0
        identical = same_len and num_gap == 0.0 and len(ya) == len(base_ya)
        expect = w * len(Xo)
        ok = appended == expect
        print(f"    {'OK ' if ok else '-> '}dose {w:>2d}x  appended {appended:>7,} rows "
              f"(expected {expect:>7,})  frame identical to train.csv: {identical}")
        if not ok:
            fail(f"dose {w}x appended {appended} rows, expected {expect}")
        arm_a[w] = {"appended": int(appended), "identical": bool(identical)}
    zero_ok = arm_a[0]["identical"]
    nonzero_ok = all(not arm_a[w]["identical"] for w in DOSES if w)
    print(f"    {'OK ' if zero_ok else '-> '}dose 0 reproduces the training frame exactly")
    print(f"    {'OK ' if nonzero_ok else '-> '}every non-zero dose does NOT "
          f"(else arm A measures nothing)")
    if not zero_ok:
        fail("dose 0 does NOT reproduce the training frame; the headline baseline IS a "
             "contrast and prediction 1's falsifier has fired")
    if not nonzero_ok:
        fail("a non-zero dose also reproduces the training frame; arm A is not discriminating")
    if zero_ok:
        note("the headline `0` compares a frame with ITSELF. Degrees of freedom: 0. It is "
             "true before any model is fitted, for any metric and any seed, and it is not a "
             "measurement of the manoeuvre.")
    out["arm_a"] = arm_a

    # ------------------------------------------------------------ R2 / ARM B, the dose ladder
    print("\n[R2 / ARM B] MEMBER-TRAINING-SET layer, k=1, a per-competition TOTAL")
    txt = open(CONCAT_LOG, encoding="utf-8").read()
    read = {}
    for line in txt.splitlines():
        m = re.search(r"OOF AUC\s+([0-9.]+)", line)
        if not m:
            continue
        w = 0 if "baseline" in line else int(re.search(r"\+(\d+)x", line).group(1))
        read[w] = float(m.group(1))
    base = read.get(0)
    arm_b = {}
    for w in sorted(read):
        want = PUBLISHED["ladder"].get(w)
        d_e6 = (read[w] - base) * 1e6
        mark = "OK " if want is not None and abs(read[w] - want) < 1e-6 else "-> "
        floor_x = abs(d_e6) / PUBLISHED["noise_floor_e6"]
        print(f"    {mark}dose {w:>2d}x  OOF AUC {read[w]:.6f}  delta {d_e6:+10.1f}e-6  "
              f"|delta| / 50e-6 floor = {floor_x:6.2f}x")
        if want is None or abs(read[w] - want) >= 1e-6:
            fail(f"dose {w}x reads {read[w]:.6f}, frozen literal {want}")
        arm_b[w] = {"auc": read[w], "delta_e6": d_e6, "floor_multiple": floor_x}
    nz = [w for w in sorted(read) if w]
    deltas = [arm_b[w]["delta_e6"] for w in nz]
    all_neg = all(d < 0 for d in deltas)
    monotone = all(deltas[i] > deltas[i + 1] for i in range(len(deltas) - 1))
    print(f"    {'OK ' if all_neg else '-> '}every non-zero dose is NEGATIVE")
    print(f"    {'OK ' if monotone else '-> '}the ladder is MONOTONE DECREASING in dose "
          f"({' > '.join(f'{d:+.0f}' for d in deltas)})e-6")
    if not all_neg:
        fail("a non-zero dose is non-negative; prediction 2's falsifier has fired")
    if not monotone:
        fail("the ladder is not monotone in dose; 'monotone harm' is the wrong description")
    d1 = arm_b[1]["delta_e6"]
    above = abs(d1) > PUBLISHED["noise_floor_e6"]
    print(f"    {'OK ' if above else '-> '}the MINIMUM dose {d1:+.1f}e-6 is "
          f"{'ABOVE' if above else 'UNDER'} the {PUBLISHED['noise_floor_e6']:.0f}e-6 floor")
    if not above:
        fail("the 1x rung is under the floor; prediction 3's falsifier has fired and the "
             "headline `0` is defensible as written")
    # the cell publishes the 1x and 50x rungs and omits the 10x one
    for w, want in ((1, PUBLISHED["cell_1x_e6"]), (50, PUBLISHED["cell_50x_e6"])):
        if abs(arm_b[w]["delta_e6"] - want) > PUBLISHED["ladder_tol_e6"]:
            fail(f"the cell publishes {want:+.0f}e-6 at {w}x; the log gives "
                 f"{arm_b[w]['delta_e6']:+.1f}e-6")
    note(f"the cell publishes the 1x and 50x rungs and omits the 10x rung "
         f"({arm_b[10]['delta_e6']:+.0f}e-6), which is the one that makes the monotonicity "
         f"checkable rather than asserted.")
    out["arm_b"] = arm_b

    # ------------------------------------------------------------------- R4 / ARM D, identity
    print("\n[R4 / ARM D] IDENTITY layer -- 'there is nothing else to find'")
    md5 = hashlib.md5(open(ORIGCSV, "rb").read()).hexdigest()
    ok = md5 == PUBLISHED["orig_md5"]
    print(f"    {'OK ' if ok else '-> '}md5 {md5}  (frozen {PUBLISHED['orig_md5']})")
    if not ok:
        fail(f"the original CSV hashes to {md5}, not the frozen {PUBLISHED['orig_md5']}")
    nrows_ok = len(orig) == PUBLISHED["orig_rows"]
    print(f"    {'OK ' if nrows_ok else '-> '}{len(orig):,} rows x {orig.shape[1]} cols")
    if not nrows_ok:
        fail(f"the original has {len(orig)} rows, not {PUBLISHED['orig_rows']}")
    otr_p = os.path.join(ORIGDIR, "origmodel_train.npy")
    ote_p = os.path.join(ORIGDIR, "origmodel_test.npy")
    otr = np.load(otr_p)
    solo = float(roc_auc_score(y, otr))
    print(f"    origmodel_train solo AUC against the competition labels: {solo:.4f}")
    out["arm_d"] = {"md5": md5, "rows": int(len(orig)), "origmodel_solo_auc": solo}

    # --------------------------------------------------------------- R3 / ARM C, the enrolment
    if a.quick:
        print("\n[R3 / ARM C] --quick: the paired arm is skipped")
    else:
        print("\n[R3 / ARM C] STACK layer, per member -- the separate-estimator route in "
              "rows 3/4/7/9's units")
        names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
        # ⚠ THE POOL IS THE FULL LOADED PACK, NOT base104. base104 is the subset NOT in
        # `_vetting.csv` and it is carved out below ONLY as the CatBoost control group. The
        # control anchors the INSTRUMENT to w123/w124; it does not make this arm's pool theirs.
        print(f"    {len(names)} members loaded (the full pack); origmodel is the +1")
        ote = np.load(ote_p)
        names2 = list(names) + ["origmodel"]
        Z, _ = transform(np.column_stack([O, otr]), np.column_stack([T, ote]), TRANSFORM)
        idx = {n: i for i, n in enumerate(names2)}

        vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
        new = [n for n in names if n in vet.index]
        bei = [n for n in new if n.startswith("bei_")]
        lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
        decorr = [n for n in new
                  if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
        rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]
        cat = sorted(n for n in rest if "cat" in set(n.split("_")))
        catbase = [n for n in names if n not in vet.index]
        if len(cat) != 8:
            fail(f"the CatBoost control group reconstructs at n={len(cat)}, w123 measured n=8")

        variants = {"P": list(names), "P+orig": names2,
                    "catbase": catbase, "catbase+cat": catbase + cat}
        print(f"    paired 50/50, {REPS} splits, C={CVAL}, transform={TRANSFORM}  "
              f"P={len(names)}  catbase={len(catbase)}  cat={len(cat)}")
        rob = paired(Z, y, idx, variants)

        print("\n    THE CONTROL FIRST -- w123's CatBoost rate, in this process, these splits")
        dc = rob["catbase+cat"] - rob["catbase"]
        cpm_e6 = float(dc.mean() / len(cat)) * 1e6
        gap = abs(cpm_e6 - PUBLISHED["row3_catboost_e6"])
        print(f"    {'OK ' if gap <= PUBLISHED['control_tol_e6'] else '-> '}"
              f"+cat_only {cpm_e6:+.4f}e-6/member  (w123 published "
              f"{PUBLISHED['row3_catboost_e6']:+.4f}e-6/member, gap {gap:.4f}e-6)")
        if gap > PUBLISHED["control_tol_e6"]:
            fail(f"the in-process CatBoost control reads {cpm_e6:+.4f}e-6/member against "
                 f"w123's {PUBLISHED['row3_catboost_e6']:+.4f}e-6; this arm is NOT in rows "
                 f"3/4's units and must not be published in them")

        do = rob["P+orig"] - rob["P"]
        per = [float(v) * 1e6 for v in do]        # k=1: the rate IS the delta
        mean_e6 = float(np.mean(per))
        sd_e6 = float(np.std(per, ddof=1))
        signs = {np.sign(v) for v in per}
        flipping = len(signs) > 1
        print(f"\n    +origmodel  {mean_e6:+.2f}e-6/member  (sd {sd_e6:.2f}e-6 over "
              f"{REPS} splits: " + ", ".join(f"{v:+.2f}" for v in per) + ")")
        under = abs(mean_e6) < PUBLISHED["noise_floor_e6"]
        print(f"    {'OK ' if under else '-> '}|price| is "
              f"{'UNDER' if under else 'OVER'} the {PUBLISHED['noise_floor_e6']:.0f}e-6 floor")
        print(f"    {'OK ' if flipping else '-> '}the sign "
              f"{'FLIPS' if flipping else 'is CONSISTENT'} across the {REPS} splits")
        if not under or not flipping:
            note("prediction 4's falsifier has fired: the separate-estimator route prices as "
                 "a sign-consistent enrolment above the floor. This does NOT go in the price "
                 "column as a closure -- it goes to WANTED as a candidate, and this run must "
                 "say so.")
        out["arm_c"] = {"pool_n": len(names), "per_split_e6": per, "mean_e6": mean_e6,
                        "sd_e6": sd_e6, "sign_flipping": bool(flipping),
                        "control_e6": cpm_e6, "control_gap_e6": gap,
                        "cat_n": len(cat), "catbase_n": len(catbase)}

    # --------------------------------------------------------------- R4, the two readings
    print("\n[R4] THE TWO READINGS OF ROW 1's PRICE CELL, both on the record")
    print(f"    as printed  -- headline `0`, and 0 is the CHEAPEST value the price column "
          f"can carry; row 3's {PUBLISHED['row3_catboost_e6']:+.2f}e-6 is published there as "
          f"a reason NOT to build")
    print(f"    as measured -- {arm_b[1]['delta_e6']:+.0f}e-6 at the minimum dose "
          f"({arm_b[1]['floor_multiple']:.2f}x the floor) falling monotonically to "
          f"{arm_b[50]['delta_e6']:+.0f}e-6 at 50x ({arm_b[50]['floor_multiple']:.1f}x)")
    print(f"    row 1 is the ONLY row in the table whose manoeuvre is measured NEGATIVE at "
          f"every setting anyone ran, and it prints the column's cheapest headline")
    out["two_readings"] = {"headline": 0.0, "min_dose_e6": arm_b[1]["delta_e6"],
                           "max_dose_e6": arm_b[50]["delta_e6"]}

    out["failures"] = fails
    out["notes"] = notes
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nFAILURES: {len(fails)}")
    for f in fails:
        print("  - " + f)
    print(f"wrote {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
