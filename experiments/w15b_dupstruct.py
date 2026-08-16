"""w15b step 1 -- what estimators does the data structure actually permit?

The mission asks for the Bayes ceiling inside the given columns. There are exactly two
families of estimator for that, and which one is available is a property of the data:

A. REPEATED-MEASUREMENT (duplicate) estimators. If the same exact feature vector x occurs
   n>=2 times, all those rows share the same true p(x), so the within-group label
   variability is PURE Bernoulli noise. That identifies E[p(1-p)] separately from
   E[(p-s)^2], which is the whole game -- see w15b_ceiling.py. Needs duplicates.

B. HONEST CELL-ORACLE estimators. Estimate P(y|cell) out of fold and score held-out rows.
   Always available, but only ever a LOWER bound (estimation noise costs ranking), and it
   degrades exactly where the cells get sparse.

This script measures which of the two we have, at every level of feature refinement, so
the ceiling script is not built on an assumption.

Nothing here fits a model. Pure counting.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, DAILY, NUM, TARGET, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
FEATS = NUM + CAT


def group_key(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Integer group id over the exact joint value of `cols`, NaN as its own level."""
    codes = []
    for c in cols:
        s = df[c]
        if s.dtype.kind in "fi":
            # exact float equality, NaN folded to a single extra level
            v = s.to_numpy(np.float64)
            _, inv = np.unique(np.where(np.isnan(v), np.inf, v), return_inverse=True)
        else:
            inv = pd.factorize(s, use_na_sentinel=False)[0]
        codes.append(np.asarray(inv, dtype=np.int64))
    # combine by successive re-factorisation so the width never overflows
    key = codes[0]
    for c in codes[1:]:
        key = pd.factorize(pd.Series(key) * (c.max() + 1) + c)[0].astype(np.int64)
    return key


def report(name: str, key: np.ndarray, y: np.ndarray) -> dict:
    n = len(key)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    ys = y[order]
    # group boundaries
    bnd = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
    sizes = np.diff(bnd)
    sums = np.add.reduceat(ys, bnd[:-1])
    ngroups = len(sizes)
    rows_in_dup = int(sizes[sizes >= 2].sum())
    out = {
        "name": name,
        "n_groups": ngroups,
        "rows_per_group": n / ngroups,
        "frac_rows_in_groups_ge2": rows_in_dup / n,
        "max_group": int(sizes.max()),
        "n_groups_ge2": int((sizes >= 2).sum()),
    }
    # base rate inside vs outside the duplicated subpopulation -- representativeness check
    big = sizes >= 2
    if big.any():
        out["rate_in_dup"] = float(sums[big].sum() / sizes[big].sum())
        out["rate_out_dup"] = float(
            (sums[~big].sum() / max(sizes[~big].sum(), 1)) if (~big).any() else np.nan)
    return out


def main() -> None:
    tr, te = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    n = len(tr)
    print(f"train {n:,} rows   base rate {y.mean():.7f}\n")

    print("--- lattice: distinct values per column (NaN excluded) ---")
    for c in FEATS:
        s = tr[c]
        print(f"  {c:26s} distinct {s.nunique():>7,}   NA {s.isna().mean():.4f}"
              f"   rows/level {(n * (1 - s.isna().mean())) / max(s.nunique(), 1):>9,.1f}")

    print("\n--- duplicate structure at increasing refinement ---")
    ladders = [
        ("social", [SOCIAL]),
        ("daily", [DAILY]),
        ("social+daily", [SOCIAL, DAILY]),
        ("social+daily+weekend", [SOCIAL, DAILY, "weekend_screen_time"]),
        ("rule4 = +gaming+work", [SOCIAL, DAILY, "weekend_screen_time",
                                  "gaming_hours", "work_study_hours"]),
        ("3 categoricals", CAT),
        ("all 9 numeric", NUM),
        ("ALL 12", FEATS),
    ]
    rows = []
    for name, cols in ladders:
        k = group_key(tr, cols)
        r = report(name, k, y)
        rows.append(r)
        print(f"  {name:24s} groups {r['n_groups']:>9,}  rows/grp {r['rows_per_group']:>7.2f}"
              f"  rows in dup-groups {r['frac_rows_in_groups_ge2']:>7.4f}"
              f"  max {r['max_group']:>7,}")
        if "rate_in_dup" in r:
            print(f"  {'':24s} base rate  in-dup {r['rate_in_dup']:.4f}"
                  f"   singleton {r['rate_out_dup']:.4f}")

    pd.DataFrame(rows).to_csv(
        os.path.join(ROOT, "experiments", "w15b_dupstruct.csv"), index=False)
    print("\nwrote experiments/w15b_dupstruct.csv")


if __name__ == "__main__":
    main()
