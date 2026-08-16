"""w15g -- the decisive test of the lead: does fold-safe lattice TE inflate OOF AUC?

THE LEAD (w15b, next-run item 5b)
    "our OOF score partly tracks realised cell counts, which the test predictions cannot
     do.  That is a candidate mechanism for part of the CV->LB offset."

The claim has a completely clean experimental form that needs no model and no assumption
about the mechanism.  Take the REAL lattice cells and the REAL frozen folds, but PERMUTE
the labels.  Under permuted labels the cells carry exactly zero true signal, so the
out-of-fold cell mean is a pure realised-count tracker and nothing else.  Score the
permuted labels with it and read the pooled OOF AUC:

    if fold-safe TE inflates OOF AUC by tracking realised cell counts, this number is
    above 0.5.  If it is 0.5, the mechanism does not exist and the lead is closed.

POSITIVE CONTROL (mandatory -- a null is worthless without one).  The same statistic with
the encoder fitted on ALL five folds, i.e. each row's own label inside its own encoding.
That is textbook target leakage and must read far above 0.5.  If it does not, the
instrument is broken and the null means nothing.

Two smoothing arms, because smoothing is what damps count tracking:
    smooth=20  -- agent/features.py:te_block's actual setting
    smooth=0   -- no damping at all, the strongest form the mechanism can take.

Three key sets spanning the cell-size range measured by w15b_dupstruct.py:
    social+daily  221,008 cells, 3.13 rows/cell   (tiny cells, maximal per-row tracking)
    daily          ~1,390 cells, ~500 rows/cell   (large cells, maximal pair coverage)
    social            722 cells, ~958 rows/cell

Usage:  python experiments/w15g_teleak.py [n_reps]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DAILY, TARGET, get_folds, load_raw  # noqa: E402

SOCIAL = "social_media_hours"
OUT = os.path.join(ROOT, "experiments", "w15g_teleak.json")
SEED = 20260815


def codes(tr: pd.DataFrame, cols) -> np.ndarray:
    """Integer cell id at full lattice resolution, NaN its own level."""
    # pandas' str dtype propagates NA through concatenation, which would send those rows
    # to factorize's -1 sentinel and silently merge every missing pattern; NaN must be
    # its own lattice level, so it is materialised as a token first.
    def col(c):
        return tr[c].astype(str).fillna("NA").astype(object)

    s = col(cols[0])
    for c in cols[1:]:
        s = s + "__" + col(c)
    code = pd.factorize(s)[0]
    assert (code >= 0).all()
    return code


def te_oof(cell: np.ndarray, y: np.ndarray, fold: np.ndarray, ncell: int,
           smooth: float, leaky: bool) -> np.ndarray:
    """Leave-fold-out (or, if `leaky`, in-sample) smoothed cell mean."""
    gm = y.mean()
    tot_s = np.bincount(cell, weights=y, minlength=ncell)
    tot_c = np.bincount(cell, minlength=ncell).astype(np.float64)
    out = np.empty(len(y))
    for k in np.unique(fold):
        m = fold == k
        if leaky:
            s, c = tot_s, tot_c
        else:
            s = tot_s - np.bincount(cell[m], weights=y[m], minlength=ncell)
            c = tot_c - np.bincount(cell[m], minlength=ncell).astype(np.float64)
        num = s[cell[m]] + smooth * gm
        den = c[cell[m]] + smooth
        out[m] = np.where(den > 0, num / np.where(den > 0, den, 1.0), gm)
    return out


def main() -> None:
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    n = len(y)
    fold = np.empty(n, dtype=np.int64)
    for k, (_, va) in enumerate(get_folds(y)):
        fold[va] = k

    keysets = [("social+daily", [SOCIAL, DAILY]), ("daily", [DAILY]), ("social", [SOCIAL])]
    rng = np.random.default_rng(SEED)
    out = {}
    for label, cols in keysets:
        cell = codes(tr, cols)
        ncell = cell.max() + 1
        sizes = np.bincount(cell)
        print(f"\n### {label}: {ncell} cells, mean size {sizes.mean():.2f}, "
              f"max {sizes.max()}")
        for smooth in (20.0, 0.0):
            for leaky in (False, True):
                # the leaky arm is a ~0.3 AUC effect; it needs no replication
                reps = 5 if leaky else nrep
                aucs = []
                for r in range(reps):
                    yp = y.copy()
                    rng.shuffle(yp)          # destroys ALL true cell signal
                    t = te_oof(cell, yp, fold, ncell, smooth, leaky)
                    aucs.append(roc_auc_score(yp, t))
                a = np.asarray(aucs)
                tag = "IN-SAMPLE (positive control)" if leaky else "fold-safe"
                key = f"{label}|smooth{smooth:g}|{'leaky' if leaky else 'safe'}"
                se = a.std(ddof=1) / np.sqrt(len(a))
                out[key] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                                se=float(se), n=len(a), lo=float(a.min()),
                                hi=float(a.max()))
                print(f"  smooth {smooth:4.0f}  {tag:28s} AUC {a.mean():.6f} "
                      f"+- {a.std(ddof=1):.6f}  n={len(a):4d}  dev from 0.5: "
                      f"{(a.mean() - 0.5) * 1e6:+8.1f}e-6 +- {se * 1e6:.1f}e-6 "
                      f"(z {(a.mean() - 0.5) / se:+.2f})")
        # and the real labels, for scale
        for smooth in (20.0, 0.0):
            t = te_oof(cell, y, fold, ncell, smooth, False)
            out[f"{label}|smooth{smooth:g}|REAL"] = float(roc_auc_score(y, t))
            print(f"  smooth {smooth:4.0f}  REAL labels, fold-safe        "
                  f"AUC {roc_auc_score(y, t):.6f}")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
