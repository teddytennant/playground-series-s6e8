"""A fold-partition gate that needs neither `fold_id.npy` nor published per-fold AUCs.

WHY THIS EXISTS
---------------
RESEARCH.md "Always verify the split before stacking anyone's OOF" lists three gates,
weakest to strongest:

  1. published-vs-computed pooled OOF AUC   -- proves ROW ORDER only, nothing about folds
  2. credibility (< 0.9720)                 -- catches gross leakage only
  3. the fold-id gate                       -- exact, but needs the author to ship fold_id

`adarsh1077/s6e8-adarsh-oof-library` (22 members, published 2026-08-15) ships neither a
fold id nor per-fold AUCs. It only *claims* StratifiedKFold(5, shuffle=True,
random_state=42) over train.csv in file order. Under gate 1 it is indistinguishable from a
library built on a different partition, and a foreign partition is exactly the failure mode
that inflates a stack's CV: a training-fold row's member value then comes from a base model
that saw our validation fold's labels.

THE TEST
--------
A K-fold OOF vector is a MOSAIC of K different fitted models. Two models fitted on 80%
overlapping data are not identically calibrated, so the fold-level mean of the member's
score carries a signature of the partition that produced it.

  H1 (their partition IS ours):  our folds each isolate one of their base models
                                 -> fold means differ by model-to-model calibration
  H0 (foreign partition):        our folds are a stratified random subsample of their
                                 mixture -> fold means differ by sampling noise alone

Statistic: one-way ANOVA F of the member's logit score across OUR five folds.
Null: 200 stratified random 5-way partitions of the same sizes, same statistic.
Reported as log10(F_obs / median F_null), i.e. how many orders of magnitude of
between-fold structure the claimed partition explains over a random one.

WHY THE PERMUTATION NULL IS THE RIGHT NEGATIVE CONTROL, not merely a convenient one:
a member cross-validated on a FOREIGN stratified 5-fold partition is exchangeable with a
random stratified partition *with respect to ours* -- same fold sizes, same class balance
per fold (both stratified on the same y), and independent fold membership. So the null
here is not an approximation to the alternative-hypothesis distribution; it IS it.

ASYMMETRY OF THE VERDICT -- read before using it
------------------------------------------------
A PASS is strong evidence: only the true partition can concentrate between-fold variance.
A FAIL is inconclusive for members whose base models are near-identical across folds
(a heavily-regularised linear model on 553k rows barely moves). So the run below prints
positive controls -- our own members and the ten fold-id-gated beicicc libraries, all of
which are on our partition by construction -- to establish the test's power before any
candidate is judged by it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from common import DATA, OOF, ROOT, TARGET, get_folds  # noqa: E402
from stack import to_logit  # noqa: E402

N_TR = 691369
OUT = os.path.join(ROOT, "experiments")


def fold_labels(folds, n):
    f = np.empty(n, np.int8)
    for k, (_, va) in enumerate(folds):
        f[va] = k
    return f


def anova_f(z, lab, k=5):
    """One-way ANOVA F of z across the k groups in lab. O(n), no allocation per call."""
    n = len(z)
    cnt = np.bincount(lab, minlength=k).astype(np.float64)
    s = np.bincount(lab, weights=z, minlength=k)
    m = s / cnt
    gm = z.mean()
    ssb = float((cnt * (m - gm) ** 2).sum())
    ssw = float(((z - gm) ** 2).sum() - ssb)
    return (ssb / (k - 1)) / (ssw / (n - k))


def null_labels(y, n_draw, seed0=90210):
    """Stratified random 5-way partitions -- the foreign-partition distribution."""
    out = []
    for b in range(n_draw):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed0 + b)
        out.append(fold_labels(list(skf.split(np.zeros(len(y)), y)), len(y)))
    return out


def score(z, ours, nulls):
    fo = anova_f(z, ours)
    fn = np.array([anova_f(z, l) for l in nulls])
    return dict(
        F_obs=fo,
        F_null_med=float(np.median(fn)),
        F_null_max=float(fn.max()),
        log10_ratio=float(np.log10(fo / np.median(fn))),
        exceeds_null_max=bool(fo > fn.max()),
    )


def gather(paths_dirs, y):
    """{name: oof path} over (dir, prefix) pairs."""
    out = {}
    for d, pref in paths_dirs:
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "oof_*.npy"))):
            nm = pref + os.path.basename(p)[4:-4]
            if nm not in out:
                out[nm] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--max-controls", type=int, default=25)
    a = ap.parse_args()

    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
    assert len(y) == N_TR
    ours = fold_labels(get_folds(y), N_TR)
    print(f"building {a.draws} stratified null partitions ...", flush=True)
    nulls = null_labels(y, a.draws)

    groups = {
        # positive controls: on our partition by construction / by the fold-id gate
        "ctl_ours": gather([(OOF, "")], y),
        "ctl_beicicc": gather([(os.path.join(DATA, "ext_members2"), "")], y),
        "ctl_lib74": gather([(os.path.join(DATA, "oof"), "")], y),
        # the candidates this slot is actually judging
        "adarsh": gather([(os.path.join(DATA, "w20_new"), "")], y),
        "catstr": gather([(os.path.join(DATA, "w20_new", "catstr"), "")], y),
    }
    # beicicc dir holds the imported ext2 pool, which is a MIX of libraries; keep the
    # bei_ ones (fold-id gated) as the control and label the rest honestly.
    groups["ctl_beicicc"] = {k: v for k, v in groups["ctl_beicicc"].items()
                             if k.startswith("bei_")}
    for g in ("ctl_ours", "ctl_beicicc", "ctl_lib74"):
        ks = sorted(groups[g])[: a.max_controls]
        groups[g] = {k: groups[g][k] for k in ks}

    rows = []
    for g, mem in groups.items():
        for nm, p in sorted(mem.items()):
            o = np.load(p).ravel()
            if o.shape != (N_TR,):
                print(f"[skip] {g}/{nm}: shape {o.shape}")
                continue
            z = to_logit(o)
            r = score(z, ours, nulls)
            r.update(group=g, member=nm)
            rows.append(r)
            print(f"{g:14s} {nm:34s} F {r['F_obs']:12.2f}  nullmed "
                  f"{r['F_null_med']:6.2f}  log10 {r['log10_ratio']:+6.2f}"
                  f"  {'PASS' if r['exceeds_null_max'] else 'weak'}", flush=True)

    df = pd.DataFrame(rows)[["group", "member", "F_obs", "F_null_med", "F_null_max",
                             "log10_ratio", "exceeds_null_max"]]
    df.to_csv(os.path.join(OUT, "w20a_foldgate.csv"), index=False)

    print("\n=== by group ===")
    for g, sub in df.groupby("group"):
        print(f"{g:14s} n {len(sub):3d}  pass {int(sub.exceeds_null_max.sum()):3d}"
              f"  log10 ratio min {sub.log10_ratio.min():+.2f}"
              f"  med {sub.log10_ratio.median():+.2f}"
              f"  max {sub.log10_ratio.max():+.2f}")

    summary = {g: dict(n=int(len(s)), n_pass=int(s.exceeds_null_max.sum()),
                       log10_min=float(s.log10_ratio.min()),
                       log10_med=float(s.log10_ratio.median()),
                       log10_max=float(s.log10_ratio.max()))
               for g, s in df.groupby("group")}
    json.dump(dict(draws=a.draws, summary=summary),
              open(os.path.join(OUT, "w20a_foldgate.json"), "w"), indent=1)
    print("\nwrote w20a_foldgate.csv / .json")


if __name__ == "__main__":
    main()
