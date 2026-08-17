"""w21b: is the algorithm FAMILY worth anything once the PIPELINE is held fixed?

THE HANDED ANGLE, AND WHY IT TAKES THIS FORM
--------------------------------------------
Slot 7's angle is "XGBoost: third leg of the ensemble, tuned on the same folds so the blend
weights mean something". Re-tuning our own XGBoost is closed -- RESEARCH's
"⚠ CLOSED 2026-08-13: tuning ANY GBDT is worth ~4e-7 into the stack", and solo->stack
pass-through is 1.4%. So the angle is taken in the one form the record says can pay, and it
is aimed at a claim the record made LAST SLOT and cannot actually support yet.

w20d measured four foreign CatBoosts at **+10.3e-6 per member**, against ~5.9e-6 for the
`rest` group and ~1.0e-6 for ten "redundant" TE-GBDT variants, and RESEARCH now says:

    "Whose pipeline built it is the dominant variable, and it is NOT observable from the
     family label."

That conclusion is drawn from a CONFOUNDED comparison. The four CatBoosts were foreign; every
"another GBDT is worth nothing" measurement it was weighed against was made on a GBDT WE
built, on OUR features. Pipeline and family move together in that table, so it cannot separate
"foreign beats ours" from "CatBoost beats XGBoost/LightGBM".

The adarsh library breaks the confound for free, because it contains all three families from
ONE pipeline on our exact frozen folds:

    xgb  5   ad_gxgbcs4, ad_xgbte, ad_gxgbd4, ad_gxgbd8, ad_gxgbnote
    cat  5   ad_catnative, ad_gcatlr02, ad_gcatd8, ad_gcatseed7, ad_gcatnote
    lgb  7   ad_glgbcs3, ad_glgbd4, ad_lgbte, ad_lgbs7, ad_glgb127, ad_glgbnote2, ad_lgbnote
    hgb  1   ad_hgbte
    nn   3   ad_gnn_te, ad_gnn_wide, ad_gnn_note
    lin  1   ad_logregte

**`xgb` and `cat` are SIZE-MATCHED at 5, from the same author, the same feature pipeline and
the same folds.** That pair is the clean contrast, and it is the whole point of this script:
no per-member normalisation, no cross-pipeline comparison, nothing to argue about.

WHAT EACH ARM ANSWERS
---------------------
  xgb   vs cat   size-matched 5 v 5: does the FAMILY label carry value within one pipeline?
  lgb7 / lgb5    lgb at its natural size and at a fixed size-5 prefix, so the 7-vs-5 count
                 difference cannot be mistaken for a family difference
  gbdt17         all three GBDT families together -- how much of the 22-member +47e-6 is GBDT
  nonGBDT5       hgb + nn + lin, the complement, as the arithmetic check on gbdt17
  cat4           w20d's EXACT four-member cat group, as a REPRODUCTION GATE. It must return
                 w20d's +0.000041 to within split noise or this harness is not the same
                 instrument and none of the rows above are comparable to the record.

Same estimator as `member_value2.py` and `w20d_value.py`: paired 50/50 stratified splits, the
same rows scored with and without the group, so the ~2e-4 split-to-split noise cancels and
only the group enters the difference. A gain whose sign flips across splits is noise whatever
its mean. Reps raised from 3 to 5 because the contrast of interest here is a DIFFERENCE OF
TWO group deltas, which carries roughly sqrt(2) times the noise of either.

PRE-REGISTERED, before the run
------------------------------
If `cat5` and `xgb5` land within ~1 sd of each other, RESEARCH's "the family label is not
informative" stands as written and the third leg is worth having. If `cat5` beats `xgb5` by
more than 2 sd, that sentence is overstated and must be rewritten: family would then carry
real signal WITHIN a pipeline, and the correct reading of w20d's +10.3e-6 would be part
pipeline, part CatBoost. Either outcome is written up. No file is shipped on the argmax of
this table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, ROOT, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402

HONEST_DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac")
EXP = os.path.join(ROOT, "experiments")

XGB = ["ad_gxgbcs4", "ad_xgbte", "ad_gxgbd4", "ad_gxgbd8", "ad_gxgbnote"]
CAT = ["ad_catnative", "ad_gcatlr02", "ad_gcatd8", "ad_gcatseed7", "ad_gcatnote"]
LGB = ["ad_glgbcs3", "ad_glgbd4", "ad_lgbte", "ad_lgbs7", "ad_glgb127",
       "ad_glgbnote2", "ad_lgbnote"]
OTH = ["ad_hgbte", "ad_gnn_te", "ad_gnn_wide", "ad_gnn_note", "ad_logregte"]
CAT4_W20D = ["ad_catnative", "ad_gcatlr02", "ad_gcatd8", "ad_gcatseed7"]

GROUPS = {
    "xgb5": XGB,
    "cat5": CAT,
    "lgb5": LGB[:5],
    "lgb7": LGB,
    "gbdt17": XGB + CAT + LGB,
    "nonGBDT5": OTH,
    "cat4": CAT4_W20D,          # reproduction gate against w20d
}


def fit_score(Z_tr, y_tr, Z_te, y_te, C, std=False):
    """`std` unit-sds each member column using the FITTING half's sd. See `w20d_value.py`
    for the mechanism: lbfgs's `tol=1e-4` stopping rule carries the columns' scale, the
    hybrid members span sd 1.82..27.59, and the fit terminates early on that slack. Off by
    default so the family table already in the journal reproduces exactly."""
    if std:
        s = Z_tr.std(0)
        s[s <= 0] = 1.0
        Z_tr, Z_te = Z_tr / s, Z_te / s
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--out", default="w21b_famvalue")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(
        y, len(te),
        extra_dirs=(os.path.join(DATA, "ext_members"),
                    os.path.join(DATA, "ext_members2"),
                    os.path.join(DATA, "ext_members3")),
        drop=set(HONEST_DROP))
    idx = {n: i for i, n in enumerate(names)}
    new = [n for n in names if n.startswith("ad_")]
    base = [n for n in names if not n.startswith("ad_")]
    print(f"{len(names)} members = {len(base)} base + {len(new)} adarsh", flush=True)
    assert len(new) == 22, new
    assert sorted(XGB + CAT + LGB + OTH) == sorted(new), "family cut is not a partition of the 22"

    t0 = time.time()
    Z, _ = transform(O, T, a.transform)
    del O, T
    print(f"transform {a.transform} in {time.time()-t0:.0f}s", flush=True)

    cfg = {"base": base}
    for g, mem in GROUPS.items():
        missing = [m for m in mem if m not in idx]
        assert not missing, missing
        cfg[g] = base + mem

    rows = []
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {"rep": rep}
        for cname, mem in cfg.items():
            cols = [idx[m] for m in mem]
            t1 = time.time()
            r[cname] = fit_score(Z[np.ix_(iA, cols)], y[iA],
                                 Z[np.ix_(iB, cols)], y[iB], a.C, a.standardize)
            print(f"  rep {rep} {cname:9s} n_mem {len(mem):3d} "
                  f"AUC {r[cname]:.6f}  ({time.time()-t1:.0f}s)", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP, a.out + ".csv"), index=False)
    print("\nheld-out AUC per 50/50 split")
    print(df.to_string(index=False, float_format="%.6f"))

    print("\nPAIRED vs base (same rows, split noise cancels)")
    out, d_by = {}, {}
    for c in df.columns:
        if c in ("rep", "base"):
            continue
        d = (df[c] - df["base"]).to_numpy()
        d_by[c] = d
        n_new = len(cfg[c]) - len(base)
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        out[c] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=n_new,
                      per_member=float(d.mean() / n_new), consistent=ok == "consistent")
        print(f"  {c:9s} n {n_new:2d}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}"
              f"  per member {d.mean()/n_new:+.2e}  [{ok}]")

    print("\nTHE SIZE-MATCHED CONTRASTS (paired within rep, so the base cancels too)")
    contrasts = [("cat5", "xgb5"), ("cat5", "lgb5"), ("lgb5", "xgb5")]
    ctr = {}
    for hi, lo in contrasts:
        d = d_by[hi] - d_by[lo]
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se > 0 else float("nan")
        ctr[f"{hi}-{lo}"] = dict(mean=float(d.mean()), se=float(se), t=float(t),
                                 wins=int((d > 0).sum()), reps=int(len(d)))
        print(f"  {hi} - {lo:5s}  {d.mean():+.6f}  se {se:.6f}  t {t:+5.2f}  "
              f"{int((d>0).sum())}/{len(d)} reps")

    print("\nARITHMETIC CHECK  (groups overlap in what they explain, so these need not add)")
    print(f"  gbdt17 {out['gbdt17']['mean']:+.6f}   "
          f"xgb5+cat5+lgb7 {out['xgb5']['mean']+out['cat5']['mean']+out['lgb7']['mean']:+.6f}")

    print("\nREPRODUCTION GATE vs w20d (its `cat` row was +0.000041 +/- 0.000001 on 3 reps)")
    print(f"  cat4 here {out['cat4']['mean']:+.6f} +/- {out['cat4']['sd']:.6f}  "
          f"-> {'PASS' if abs(out['cat4']['mean'] - 0.000041) < 3*max(out['cat4']['sd'],1e-6) else 'FAIL'}")

    json.dump(dict(transform=a.transform, C=a.C, reps=a.reps, standardize=a.standardize,
                   n_base=len(base), n_new=len(new),
                   families={k: v for k, v in GROUPS.items()},
                   paired=out, contrasts=ctr),
              open(os.path.join(EXP, a.out + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
