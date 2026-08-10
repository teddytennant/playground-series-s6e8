"""Paired value of the 51 new members from boltuzamaki + mohankrishnathalla.

Two days of measurement said adding members is worth ~1e-5 and adding a new function
class ~5e-6. Both of those were measured on candidates that correlate 0.99+ with the
pack. The new library is different in one specific way: it contains members at maxcorr
0.81-0.96 against everything we stack, where the current diversity prize (`lookup`) sits
at 0.9869. So the groups here are cut by CORRELATION, not by family, because correlation
is the variable the journal says matters.

  decorr   maxcorr < 0.97 -- extratrees, gandalf, dcnv2, ft-transformer, ebm (a GAM),
                             tabr (retrieval), deepfm, lookup_v3, neural, mkt_nn
  lookup2  the six seeds of the author's second Lookup-Transformer implementation
  rest     the remaining GBDT-shaped members, which the journal predicts are worthless

Everything is judged by paired 50/50 splits: same rows, with and without the group, so
the ~0.0002 split noise cancels. A gain whose sign flips across splits is noise.
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
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, transform  # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
DECORR_MAX = 0.97


def fit_score(Z_tr, y_tr, Z_te, y_te, C=1.0):
    m = LogisticRegression(max_iter=3000, C=C).fit(Z_tr, y_tr)
    return roc_auc_score(y_te, m.predict_proba(Z_te)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--transform", default="hybrid")
    a = ap.parse_args()

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, T = load_members(y, len(te), extra_dirs=(EXT, EXT2),
                               drop=set(DEFAULT_DROP))
    idx = {n: i for i, n in enumerate(names)}

    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    new = [n for n in names if n in vet.index]
    base = [n for n in names if n not in vet.index]
    bei = [n for n in new if n.startswith("bei_")]
    lookup2 = [n for n in new if n.startswith("bolt_lookup_v2")]
    decorr = [n for n in new
              if n not in lookup2 and n not in bei and vet.loc[n, "maxcorr"] < DECORR_MAX]
    rest = [n for n in new if n not in lookup2 and n not in decorr and n not in bei]
    print(f"base {len(base)} | new {len(new)} = decorr {len(decorr)} + "
          f"lookup2 {len(lookup2)} + rest {len(rest)} + bei {len(bei)}")
    print("  decorr:", " ".join(f"{n}({vet.loc[n,'maxcorr']:.3f})" for n in decorr))

    Z, _ = transform(O, T, a.transform)

    variants = {
        "base86": base,
        "+decorr": base + decorr,
        "+lookup2": base + lookup2,
        "+rest": base + rest,
        "+bei": base + bei,
        "+all_new": base + new,
    }

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
    print("\nPAIRED DELTA vs base86 (same rows; split noise cancels)")
    for c in rob.columns:
        if c == "base86":
            continue
        d = rob[c] - rob["base86"]
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  {c:<18s} {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  [{ok}]")


if __name__ == "__main__":
    main()
