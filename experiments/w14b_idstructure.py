"""Is there id-ordered structure that a NON-random public slice could pick up?

`w14b_slicenoise2.py` prices the public slice as a uniform random subsample of the test
set.  That is the standard Playground split and it is what the sd it reports assumes.  If
Kaggle instead cut the public slice contiguously in id order -- or if the synthetic
generator drifted with id -- then the real slice's draw could be much wider than a random
subsample's, and the +97e-6 `logit` displacement would stop being a 3-sigma event.

Nobody has checked.  It is cheap to check: cut the 691,369 labelled OOF rows into K
contiguous id-ordered blocks, score each transform on each block, and compare the
block-to-block spread of the paired difference against the random-subsample spread at the
same block size.  Equal spread => no id structure, the random-slice null stands.  A wider
contiguous spread => the null understates the draw and the displacement is cheaper than it
looks.

    w14b_idstructure.py --blocks 24
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w14b_slicenoise2 import FILES, SIX, prep, subset_auc  # noqa: E402

SUB = os.path.join(ROOT, "submissions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=24)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=14)
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    ids = tr["id"].to_numpy()
    assert (np.diff(ids) > 0).all(), "train is not in ascending id order -- fix the cut"
    print(f"{n:,} rows, ids {ids[0]:,}..{ids[-1]:,}, ascending: yes")

    V = {k: np.load(os.path.join(SUB, f"oof_{FILES[k]}.npy")) for k in SIX}
    PRE = {k: prep(v) for k, v in V.items()}
    YS = {k: y[PRE[k][0]] for k in SIX}

    bs = n // a.blocks
    print(f"{a.blocks} contiguous id blocks of {bs:,} rows each\n")

    pairs = [("logit", "hybrid"), ("logit", "h3"), ("rankraw", "hybrid"), ("h3", "ens4")]

    # contiguous blocks
    cont = {p: [] for p in pairs}
    base = {}
    for k in SIX:
        base[k] = subset_auc(*PRE[k], YS[k], np.ones(n, bool))
    for b in range(a.blocks):
        m = np.zeros(n, bool)
        m[b * bs:(b + 1) * bs] = True
        auc_b = {k: subset_auc(*PRE[k], YS[k], m) for k in SIX}
        for aa, bb in pairs:
            cont[(aa, bb)].append(auc_b[aa] - auc_b[bb])

    # matched random subsamples of the SAME size
    rng = np.random.default_rng(a.seed)
    rand = {p: [] for p in pairs}
    for _ in range(a.reps):
        m = np.zeros(n, bool)
        m[rng.permutation(n)[:bs]] = True
        auc_b = {k: subset_auc(*PRE[k], YS[k], m) for k in SIX}
        for aa, bb in pairs:
            rand[(aa, bb)].append(auc_b[aa] - auc_b[bb])

    print(f"{'pair':16s} {'pooled':>9s} {'sd contiguous':>14s} {'sd random':>11s} "
          f"{'ratio':>7s}   (all e-6)")
    for aa, bb in pairs:
        c = np.array(cont[(aa, bb)])
        r = np.array(rand[(aa, bb)])
        pooled = base[aa] - base[bb]
        print(f"{aa+'-'+bb:16s} {pooled*1e6:+9.1f} {c.std(ddof=1)*1e6:14.1f} "
              f"{r.std(ddof=1)*1e6:11.1f} {c.std(ddof=1)/r.std(ddof=1):7.2f}")

    print("\nper-block logit-h3 deviation from pooled (e-6), in id order:")
    d = (np.array(cont[("logit", "h3")]) - (base["logit"] - base["h3"])) * 1e6
    print("  " + "  ".join(f"{x:+.0f}" for x in d))
    # trend test: correlation of the deviation with block index
    rho = float(np.corrcoef(np.arange(a.blocks), d)[0, 1])
    print(f"\n  correlation with block index (drift test): {rho:+.3f}")
    print("  ratio ~1.0 and no drift => the public slice behaves like a random subsample")
    print("  and w14b_slicenoise2's sd is the right one.")


if __name__ == "__main__":
    main()
