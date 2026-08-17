"""Paired value of the 22 adarsh members, and of the sub-groups the author claims matter.

Method is the workspace standard (`member_value2.py`): paired 50/50 stratified splits,
same rows with and without the group, so the ~2e-4 split noise cancels and only the group
enters the difference. A gain whose sign flips across splits is noise, whatever its mean.

The sub-groups are cut by the author's OWN claim, so this is a test of it rather than a
re-derivation. adarsh1077's README says the three `no target encoding` views (across XGB,
LGBM and CatBoost) took three of the top four stack coefficients in their 178-member
stack, while deeper/wider/reseeded variants of recipes already present went NEGATIVE:

  note      ad_gxgbnote, ad_glgbnote2, ad_gcatnote, ad_lgbnote   -- the no-TE views
  cat       ad_catnative, ad_gcatlr02, ad_gcatd8, ad_gcatseed7   -- the slot's angle
  nn        ad_gnn_te, ad_gnn_wide, ad_gnn_note                  -- their MLPs
  logreg    ad_logregte                                          -- maxcorr 0.9598,
                                                                    solo 0.9589
  redundant everything else (TE-GBDT variants of recipes we already hold)

Two of those cut against rules this workspace holds, which is why they are measured
separately rather than lumped in:

  * `ad_logregte` is solo 0.9589, i.e. BELOW the ~0.966 floor RESEARCH.md sets for
    judging a member on correlation. The rule says a member under that floor has to earn
    its place on a paired measurement. This is that measurement.
  * the `cat` group is four CatBoost variants of one recipe -- the exact shape the record
    calls worthless. The handed angle for this slot is CatBoost, so it gets its own row.
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

GROUPS = {
    "note": ["ad_gxgbnote", "ad_glgbnote2", "ad_gcatnote", "ad_lgbnote"],
    "cat": ["ad_catnative", "ad_gcatlr02", "ad_gcatd8", "ad_gcatseed7"],
    "nn": ["ad_gnn_te", "ad_gnn_wide", "ad_gnn_note"],
    "logreg": ["ad_logregte"],
}


def fit_score(Z_tr, y_tr, Z_te, y_te, C, std=False):
    """`std` divides each member column by its sd on the FITTING half, applied to both.

    w23 measured why this is not cosmetic: lbfgs stops when the max gradient falls under
    `tol=1e-4`, that rule carries the columns' scale, and the hybrid members span sd
    1.82..27.59 -- so the default rule is ~15x looser on the narrow columns than the wide
    ones and the fit terminates on that slack with no warning. Unstandardised, the
    187-member cell stops at 414 iterations; standardised it converges in 69 and reaches
    the SAME point an unstandardised fit only gets to at `tol=1e-7` after 8,000 iterations
    (residual gap -0.21e-6). Every per-member value in the w20d/w21b tables was measured
    with the under-converged fit, which is what this flag exists to re-cut.

    The scale comes from the fitting half rather than blend_lab's whole-OOF convention,
    because here the two halves are a held-out pair; a column sd is not target-dependent
    so the difference is immaterial, but the fitting-half version needs no argument.
    """
    if std:
        s = Z_tr.std(0)
        s[s <= 0] = 1.0
        Z_tr, Z_te = Z_tr / s, Z_te / s
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--standardize", action="store_true",
                    help="unit-sd the members before the meta-fit; default off so the "
                         "w20d numbers already in the journal reproduce exactly")
    ap.add_argument("--out", default="w20d_value.csv")
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

    t0 = time.time()
    Z, _ = transform(O, T, a.transform)
    del O, T
    print(f"transform {a.transform} in {time.time()-t0:.0f}s", flush=True)

    cfg = {"base": base, "all22": base + new}
    for g, mem in GROUPS.items():
        missing = [m for m in mem if m not in idx]
        assert not missing, missing
        cfg[g] = base + mem
    # the complement of the four named groups: TE-GBDT variants of recipes already held
    named = {m for mem in GROUPS.values() for m in mem}
    cfg["redundant"] = base + [n for n in new if n not in named]

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
            print(f"  rep {rep} {cname:10s} n_mem {len(mem):3d} "
                  f"AUC {r[cname]:.6f}  ({time.time()-t1:.0f}s)", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP, a.out), index=False)
    print("\nheld-out AUC per 50/50 split")
    print(df.to_string(index=False, float_format="%.6f"))

    print("\nPAIRED vs base (same rows, split noise cancels)")
    out = {}
    for c in df.columns:
        if c in ("rep", "base"):
            continue
        d = df[c] - df["base"]
        n_new = len(cfg[c]) - len(base)
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        out[c] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)), n=n_new,
                      per_member=float(d.mean() / n_new), consistent=ok == "consistent")
        print(f"  {c:10s} n {n_new:2d}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}"
              f"  per member {d.mean()/n_new:+.2e}  [{ok}]")

    # ⚠ the JSON follows --out too. It did NOT until 2026-08-17 (w24), and a re-cut run
    # with --out set silently overwrote the ORIGINAL run's json while writing its csv to
    # the new name. Restored from git that same slot; the flag exists so it cannot recur.
    json.dump(dict(transform=a.transform, C=a.C, reps=a.reps, standardize=a.standardize,
                   n_base=len(base), n_new=len(new), paired=out),
              open(os.path.join(EXP, os.path.splitext(a.out)[0] + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
