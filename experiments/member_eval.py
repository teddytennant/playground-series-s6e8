"""What is a candidate member actually worth on top of the 149 already in the stack?

Three numbers, in the order the journal says to read them:

1. `maxcorr`  the largest Pearson correlation, on the stacker's scale, against any member
   already held. This is the leading indicator: the slot-3 attribution measured
   decorrelated members at ~1.5x the value per member of GBDT-shaped ones, and the pack
   already sits at 0.987-0.999 with each other.
2. `sd_ratio` sd(logit(test)) / sd(logit(oof)), the train/test scale diagnostic. Our own
   honest 5-fold models sit at 1.000-1.002; anything far from that is a broken transform
   and would earn distorted stacker weight.
3. `paired`   the held-out AUC delta from adding the candidate to the base, fit and
   scored on identical 50/50 row splits so split noise cancels. This is what selection
   uses. A cross-fit cannot resolve differences at this size (noise floor ~5e-5).

Usage:  member_eval.py --add et_lat_frac,linlat --reps 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OOF  # noqa: E402
from stack import to_logit, transform  # noqa: E402
from stack_lab import build, crossfit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", required=True, help="comma-separated member names in oof/")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--crossfit", action="store_true")
    a = ap.parse_args()

    names, Z, Zt, y = build("hybrid", set())
    print(f"base matrix {Z.shape}")
    cand = [c for c in a.add.split(",") if c]

    # candidates are dropped from the cached base if they were already picked up there
    keep = [i for i, n in enumerate(names) if n not in cand]
    if len(keep) < len(names):
        print(f"[base] excluding {len(names)-len(keep)} candidate column(s) already cached")
        Z, Zt = Z[:, keep], Zt[:, keep]
        names = [names[i] for i in keep]

    cols_o, cols_t = [], []
    for c in cand:
        o = np.load(os.path.join(OOF, f"oof_{c}.npy"))
        t = np.load(os.path.join(OOF, f"test_{c}.npy"))
        zo, zt = transform(o[:, None], t[:, None], "hybrid")
        lo, lt = to_logit(o), to_logit(t)
        corr = np.abs(np.corrcoef(np.column_stack([zo[:, 0], Z]).T)[0, 1:])
        j = int(np.argmax(corr))
        print(f"\n[{c}] solo AUC {roc_auc_score(y, o):.5f}   "
              f"sd_ratio {lt.std()/lo.std():.4f}   "
              f"maxcorr {corr[j]:.4f} vs {names[j]}   "
              f"median corr {np.median(corr):.4f}")
        cols_o.append(zo[:, 0].astype("float32"))
        cols_t.append(zt[:, 0].astype("float32"))

    # each candidate on its own, plus all of them together, sharing one base fit per
    # split -- the base is the expensive part and refitting it per candidate would triple
    # the cost while adding nothing (it is the same fit on the same rows).
    variants = {c: np.hstack([Z, cols_o[i][:, None]]) for i, c in enumerate(cand)}
    if len(cand) > 1:
        variants["ALL"] = np.hstack([Z, np.column_stack(cols_o)])

    print(f"\npaired 50/50, {a.reps} splits: base {Z.shape[1]} members")
    ds = {k: [] for k in variants}
    for rep in range(a.reps):
        iA, iB = next(StratifiedShuffleSplit(1, test_size=0.5, random_state=rep)
                      .split(np.zeros(len(y)), y))
        b = roc_auc_score(y[iB], LogisticRegression(max_iter=3000, C=a.C)
                          .fit(Z[iA], y[iA]).decision_function(Z[iB]))
        line = [f"  split {rep}: base {b:.6f}"]
        for k, Za in variants.items():
            w = roc_auc_score(y[iB], LogisticRegression(max_iter=3000, C=a.C)
                              .fit(Za[iA], y[iA]).decision_function(Za[iB]))
            ds[k].append(w - b)
            line.append(f"{k} {w-b:+.6f}")
        print("   ".join(line), flush=True)

    print()
    for k, v in ds.items():
        d = np.array(v)
        n = len(cand) if k == "ALL" else 1
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"PAIRED DELTA {k:>16s}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
              f"[{ok}]  per member {d.mean()/n:+.6f}")

    if a.crossfit:
        cv0, _ = crossfit(Z, y, a.C)
        print(f"\ncross-fitted base {cv0:.6f}")
        for k, Za in variants.items():
            cv1, _ = crossfit(Za, y, a.C)
            print(f"cross-fitted {k:>16s} {cv1:.6f}  ({cv1-cv0:+.6f})", flush=True)


if __name__ == "__main__":
    main()
