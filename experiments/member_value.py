"""Does adding a candidate member group actually buy anything?

The only honest way to answer this is a PAIRED comparison: fit the combiner on the same
half of the rows with and without the candidates, score on the same other half, and take
the difference. Split noise (~0.0002) is an order of magnitude larger than the effects
being measured (~0.00001), so an unpaired before/after comparison is worthless here.

Reports mean +/- std of the paired delta and whether the sign is consistent across
splits. A gain that flips sign across splits is noise, no matter how good the mean looks.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, LIB, OOF, TARGET, load_raw  # noqa: E402
from stack import load_members, to_logit  # noqa: E402

EXT = os.path.join(DATA, "ext_members")

GROUPS = {
    "fm": ["fmplr", "fmnum", "fmdeep", "fmwide", "fmpure"],
    # golem a and f early-stop on the held-out validation fold, which the author
    # discloses as mildly optimistic. Kept separable so their effect can be isolated.
    "golem_clean": ["golem_b", "golem_c", "golem_d", "golem_e", "golem_g"],
    "golem_es": ["golem_a", "golem_f"],
    "mine": ["lgbm_tuned_lat", "lgbm_tuned_lat_frac"],
}


def fit_score(Z_tr, y_tr, Z_te, y_te, C=1.0):
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(y, len(te), extra_dirs=(EXT,))
    idx = {n: i for i, n in enumerate(names)}
    print(f"{len(names)} members total")

    base = [n for n in names if not any(n in g for g in GROUPS.values())]
    print(f"base library: {len(base)}")
    for g, mem in GROUPS.items():
        got = [m for m in mem if m in idx]
        print(f"  group {g:<12s} {len(got)}/{len(mem)}  "
              + " ".join(f"{m}={roc_auc_score(y, O[:, idx[m]]):.5f}" for m in got))

    Z = to_logit(O)
    pub = base + GROUPS["fm"] + GROUPS["golem_clean"]
    variants = {"base": base}
    variants["base+pub_ext"] = pub
    variants["base+mine"] = base + GROUPS["mine"]
    variants["pub_ext+mine_notuneonly"] = pub + ["lgbm_tuned_lat"]
    variants["pub_ext+mine_fraconly"] = pub + ["lgbm_tuned_lat_frac"]
    variants["pub_ext+mine"] = pub + GROUPS["mine"]
    variants = {k: [m for m in v if m in idx] for k, v in variants.items()}

    rows = []
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        r = {}
        for k, mem in variants.items():
            cols = [idx[m] for m in mem]
            r[k] = fit_score(Z[np.ix_(iA, cols)], y[iA], Z[np.ix_(iB, cols)], y[iB], a.C)
        rows.append(r)
        print(f"  split {rep}: " + "  ".join(f"{k}={v:.6f}" for k, v in r.items()),
              flush=True)

    rob = pd.DataFrame(rows)
    print("\nPAIRED DELTA vs base (same rows; split noise cancels)")
    for c in rob.columns:
        if c == "base":
            continue
        d = rob[c] - rob["base"]
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  {c:<22s} {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")


if __name__ == "__main__":
    main()
