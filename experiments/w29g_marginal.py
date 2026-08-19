"""w29g -- does a DECORRELATED member carry orthogonal SIGNAL, or just orthogonal noise?

w29d returned a clean null: four members spanning maxcorr 0.888..0.984 -- one of them below
the 190-pack's previous MINIMUM of 0.91797 -- are jointly worth +0.30e-6 +/- 0.67 on
identical stacker partitions, positive in 2 of 4. That contradicts the standing
recommendation this workspace has been operating on since w20d ("what a saturated stack pays
for is decorrelation"), so the reading matters more than the number.

The candidate explanation is that `maxcorr` cannot tell the two reasons a member can be
decorrelated apart:

    (a) it computes a DIFFERENT function of x   -- a new direction, worth something
    (b) it computes the same function BADLY     -- its disagreement is estimation noise

At solo AUC 0.82..0.96 against a pack median of 0.966, (b) is available in quantity, and a
rank correlation cannot see the difference: noise decorrelates exactly like signal does.

THE INSTRUMENT. Collapse the whole pack to its own stack output S and ask what one member
adds ON TOP of it: cross-fit logistic(y ~ [S]) against logistic(y ~ [S, c]) on identical
partitions and take the paired AUC delta. Two columns, so it costs seconds rather than the
hour a pack refit costs, and it is the same question the combiner asks -- the combiner is
free to give c a negative coefficient here exactly as it is in the full fit.

⚠ WHAT THIS IS NOT. S is itself a cross-fitted stack OOF, so the base is mildly optimistic
and these deltas are NOT comparable to a full-pack refit's absolute level. They are
comparable to EACH OTHER, which is the whole use: the ranking of candidates by marginal
value, and its relation to maxcorr, is what is being read. The absolute check on the four
w29 members already exists and is w29d's +0.30e-6.

The control matters as much as the candidates. `linlat`, `logreg` and `orig_binm` are
already IN the pack, so dropping them from S is not possible here -- instead the controls
are pack members re-offered to a stack that already contains them, which must read ~0 and
calibrates the noise floor of the instrument.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET, load_raw  # noqa: E402

CAND_DIRS = ("ext_members8", "ext_members9")
# already inside S -- the instrument must read ~0 on these
CONTROLS = {"linlat": os.path.join(ROOT, "oof"),
            "logreg": os.path.join(DATA, "oof", "oof"),
            "orig_binm": os.path.join(ROOT, "oof")}


def rk(v):
    return ((rankdata(v) - 0.5) / len(v)).astype(np.float64)


def gauss(v):
    """rank -> normal quantile: the `rankraw` transform, minus its clip."""
    from scipy.special import ndtri
    return ndtri(rk(v))


def crossfit(Z, y, folds):
    o = np.zeros(len(y))
    for itr, iva in folds:
        o[iva] = (LogisticRegression(max_iter=5000)
                  .fit(Z[itr], y[itr]).decision_function(Z[iva]))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="w27_ad190std_h3",
                    help="the stack whose OOF is collapsed to the single column S")
    ap.add_argument("--seeds", default="42,101,13,7")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    S = np.load(os.path.join(SUB, f"oof_{a.base}.npy"))
    assert S.shape == (len(y),)
    print(f"base {a.base}: OOF AUC {roc_auc_score(y, S):.10f}", flush=True)

    cands = {}
    for d in CAND_DIRS:
        p = os.path.join(DATA, d)
        for f in sorted(os.listdir(p)):
            if f.startswith("oof_") and f.endswith(".npy"):
                cands[f[4:-4]] = os.path.join(p, f)
    for n, p in CONTROLS.items():
        f = os.path.join(p, f"oof_{n}.npy")
        if os.path.exists(f):
            cands[f"[control] {n}"] = f

    Sg = gauss(S)[:, None]
    prep = {}
    for n, f in cands.items():
        v = np.load(f)
        prep[n] = np.column_stack([Sg[:, 0], gauss(v)])
        print(f"  loaded {n:>18s}  solo AUC {roc_auc_score(y, v):.6f}", flush=True)

    rows = {n: [] for n in prep}
    base = []
    for sd in seeds:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=sd)
                     .split(np.zeros(len(y)), y))
        b = roc_auc_score(y, crossfit(Sg, y, folds))
        base.append(b)
        print(f"\nseed {sd}: base {b:.10f}", flush=True)
        for n, Z in prep.items():
            t0 = time.time()
            d = (roc_auc_score(y, crossfit(Z, y, folds)) - b) * 1e6
            rows[n].append(d)
            print(f"  {n:>18s}  {d:+8.2f}e-6   ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n== marginal value on top of the whole {a.base} stack, paired over "
          f"{len(seeds)} partitions (e-6 of AUC) ==")
    print(f"{'member':>18s} {'mean':>9s} {'se':>7s} {'pos':>5s}   per-seed")
    order = sorted(rows, key=lambda n: -np.mean(rows[n]))
    for n in order:
        v = np.array(rows[n])
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f"{n:>18s} {v.mean():+9.2f} {se:7.2f} {int((v>0).sum())}/{len(v)}   "
              + " ".join(f"{x:+7.2f}" for x in v))
    json.dump({"base": a.base, "base_auc": base, "seeds": seeds,
               "delta": {n: list(map(float, v)) for n, v in rows.items()}},
              open(os.path.join(ROOT, "experiments", "w29g_marginal.json"), "w"), indent=1)
    print(f"\nwrote experiments/w29g_marginal.json")


if __name__ == "__main__":
    main()
