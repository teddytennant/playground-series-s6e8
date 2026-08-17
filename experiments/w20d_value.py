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


def fit_score(Z_tr, y_tr, Z_te, y_te, C):
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--transform", default="hybrid")
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
                                 Z[np.ix_(iB, cols)], y[iB], a.C)
            print(f"  rep {rep} {cname:10s} n_mem {len(mem):3d} "
                  f"AUC {r[cname]:.6f}  ({time.time()-t1:.0f}s)", flush=True)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP, "w20d_value.csv"), index=False)
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

    json.dump(dict(transform=a.transform, C=a.C, reps=a.reps,
                   n_base=len(base), n_new=len(new), paired=out),
              open(os.path.join(EXP, "w20d_value.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
