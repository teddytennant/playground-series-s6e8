"""Repeated cross-fitting of the combiner over many fold splits.

Every candidate ranking in this journal -- blend156w 0.970047, blend156_h3 0.970046,
blend156 0.970042 -- comes from ONE cross-fit over ONE fold partition. `auc_boot.py`
priced the row noise on those gaps and found it small (paired sd ~2e-6). It could not
price the other half: the combiner's fold assignment is itself a random draw, and nobody
here has ever resampled it.

This does. The member OOF matrix is fixed on disk, so a combiner fold split is free to
change -- 5 logistic fits per transform per split, ~5s each. Split 0 is the frozen
StratifiedKFold(5, seed=42) that every number in the journal used; splits 1..R-1 are new
seeds. Each split produces a full set of candidates, so the output is both

  * the fold-split sd of the CV LEVEL (how much a headline number could have been luck)
  * the fold-split sd of each paired DIFFERENCE, which is what selection rests on

and, as a by-product, a repeated-CV estimate: averaging the per-split AUCs is a lower-
variance selection statistic than any single cross-fit.

LEAKAGE, STATED PLAINLY. The members' own OOF came from the frozen member folds, so a
combiner trained on rows outside its eval set still sees features built by models that
saw eval-set labels. That is true of the shipped cross-fit too -- member fold g's OOF is
produced by a model trained on fold f -- so re-splitting the COMBINER's folds adds no new
kind of optimism, only the same kind. It inflates every candidate together and cancels in
the paired differences, which is all this bench reads.

    repcv.py --splits 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import N_SPLITS, SEED  # noqa: E402
from blend_lab import HONEST_DROP, KINDS, load_all, rk  # noqa: E402

# the full-data simplex weights transform_weights.py settled on; this is the vector
# blend156w actually ships, so applying it per split scores that candidate and not a
# refitted cousin of it.
W_SEARCHED = {"logit": 0.05, "hybrid": 0.15, "rankraw": 0.50, "rescale": 0.25}


def candidates(oof, y, kinds):
    """Every deadline candidate, rebuilt from one split's per-transform stack OOF."""
    r = {f"stack_{k}": roc_auc_score(y, oof[k]) for k in kinds}
    ranks = {k: rk(oof[k]) for k in kinds}
    r["ens4"] = roc_auc_score(y, np.mean([ranks[k] for k in kinds], 0))
    h3 = [k for k in kinds if k != "logit"]
    r["ens3_h3"] = roc_auc_score(y, np.mean([ranks[k] for k in h3], 0))
    w = np.array([W_SEARCHED[k] for k in kinds])
    r["ens_w"] = roc_auc_score(y, sum(wi * ranks[k] for wi, k in zip(w, kinds)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--drop", default=",".join(HONEST_DROP) + ",bolt_xgb_d7_alt2")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    kinds = [k for k in a.kinds.split(",") if k]
    t0 = time.time()
    names, y, mats, _ = load_all(tuple(kinds), set(filter(None, a.drop.split(","))))
    print(f"loaded {len(names)} members in {time.time()-t0:.0f}s", flush=True)

    rows, oof_acc = [], {k: np.zeros(len(y)) for k in kinds}
    for s in range(a.splits):
        # split 0 IS the frozen scheme, so its row reproduces the journal's numbers and
        # acts as a check that this bench agrees with blend_lab before anything is read
        # off the other splits.
        seed = SEED if s == 0 else 1000 + s
        folds = list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=seed).split(np.zeros(len(y)), y))
        oof = {}
        for k in kinds:
            Z = mats[k][0]
            mo = np.zeros(len(y))
            for itr, iva in folds:
                mo[iva] = (LogisticRegression(max_iter=5000, C=a.C)
                           .fit(Z[itr], y[itr]).decision_function(Z[iva]))
            oof[k] = mo
            oof_acc[k] += rk(mo) / a.splits
        r = candidates(oof, y, kinds)
        r["split"] = s
        r["seed"] = seed
        rows.append(r)
        print(f"  split {s} (seed {seed})  " +
              "  ".join(f"{c} {r[c]:.6f}" for c in ("ens4", "ens3_h3", "ens_w")) +
              f"   ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows).set_index("split")
    print("\nper-split cross-fitted AUC")
    print(df.to_string(float_format="%.6f"))

    cols = [c for c in df.columns if c not in ("seed",)]
    print("\nFOLD-SPLIT NOISE of the CV level (sd across splits)")
    for c in sorted(cols, key=lambda z: -df[z].mean()):
        print(f"  {c:12s} mean {df[c].mean():.6f}  sd {df[c].std(ddof=1):.6f}  "
              f"frozen-split {df[c].iloc[0]:.6f}")

    print("\nPAIRED differences vs ens4, per split (this is what selection rests on)")
    for c in sorted(cols, key=lambda z: -df[z].mean()):
        if c == "ens4":
            continue
        d = df[c] - df["ens4"]
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        print(f"  {c:12s} {d.mean():+.6f} +/- {d.std(ddof=1):.6f}  "
              f"frozen {d.iloc[0]:+.6f}  [{ok}]")

    print("\nREPEATED-CV candidates (per-split OOF ranks averaged over all splits)")
    for c, v in sorted(candidates(oof_acc, y, kinds).items(), key=lambda z: -z[1]):
        print(f"  {c:12s} {v:.6f}")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
