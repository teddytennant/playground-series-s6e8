"""w15e: how large an AUC swing can a perturbation of THIS size produce with no signal?

The graft scored 0.97104 against the base's 0.97105 -- one quantisation unit down. Before
calling that a refutation or dismissing it as noise, the question has to be answered
quantitatively: what is the null distribution of the AUC change produced by adding
0.10 * (a correction with exactly this magnitude and value distribution) to a blend, when
the correction carries no signal at all?

The control is the one this workspace uses everywhere else: the SAME correction values with
their rows permuted. Same sd, same heavy tails, same weight, zero information.

Applied to labelled data, so there is an actual AUC to measure: draw 296,302 rows of our
own cross-fitted OOF for `blend159av_h3` (matching the test size exactly, so the per-draw
noise is the right size), add 0.10 * permuted anti_student to its percentile ranks, and
record the AUC change. 400 draws.

This yields two numbers the journal needs:
  * the null sd -- how far a signal-free correction of this size moves the score, i.e. the
    resolution of the paired LB read;
  * the null range -- an upper bound on what the whole construction could ever have been
    worth at this weight, which prices any future attempt to rebuild it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET, load_raw  # noqa: E402

N_TEST = 296_302
W = 0.10
N_DRAW = 400


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    oof = np.load(os.path.join(SUB, "oof_blend159av_h3.npy")).astype(np.float64)
    assert oof.shape == y.shape
    print(f"base OOF AUC (full 691,369 rows) = {roc_auc_score(y, oof):.6f}")

    z = np.load(os.path.join(DATA, "w15e",
                             "raykkretzschmar_s6e8-transductive-anti-student-signals",
                             "transductive_signals.npz"))
    reference = z["reference_contrast"].astype(np.float64)
    residual = rankdata(z["test_teacher"]) / N_TEST - rankdata(z["test_student"]) / N_TEST
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    ref_sq = np.sign(reference) * np.abs(reference) ** 2
    test_sq = np.sign(scaled) * np.abs(scaled) ** 2
    anti = (test_sq - ref_sq.mean()) * reference.std() / ref_sq.std()
    print(f"anti_student: sd {anti.std():.6f}  "
          f"[{anti.min():+.4f}, {anti.max():+.4f}]  weight {W}")

    rng = np.random.default_rng(15)
    deltas = np.empty(N_DRAW)
    rho = np.empty(N_DRAW)
    for i in range(N_DRAW):
        idx = rng.choice(len(y), N_TEST, replace=False)
        yy, vv = y[idx], oof[idx]
        base_rank = rankdata(vv) / N_TEST
        pert = base_rank + W * rng.permutation(anti)
        deltas[i] = roc_auc_score(yy, pert) - roc_auc_score(yy, base_rank)
        if i < 25:
            a, b = rankdata(pert), rankdata(base_rank)
            rho[i] = np.corrcoef(a, b)[0, 1]

    sd = deltas.std(ddof=1)
    print(f"\n=== null distribution, {N_DRAW} draws of {N_TEST:,} labelled rows ===")
    print(f"  mean  {deltas.mean():+.3e}   sd {sd:.3e}")
    print(f"  q01 {np.quantile(deltas, .01):+.3e}   q05 {np.quantile(deltas, .05):+.3e}   "
          f"q50 {np.quantile(deltas, .50):+.3e}   "
          f"q95 {np.quantile(deltas, .95):+.3e}   q99 {np.quantile(deltas, .99):+.3e}")
    print(f"  min {deltas.min():+.3e}   max {deltas.max():+.3e}")
    print(f"  rank-corr(perturbed, base) over the first 25 draws: "
          f"{rho[:25].mean():.7f}  (real file vs base: 0.9999720)")

    obs = -1e-5   # the LB read, 0.97104 - 0.97105, quantised at 1e-5
    print(f"\n=== the observed LB paired difference ===")
    print(f"  0.97104 - 0.97105 = {obs:+.1e}, but the LB quantises at 1e-5, so the true")
    print(f"  difference lies in (-2e-5, 0). It is NOT positive.")
    print(f"  |obs| / null sd = {abs(obs) / sd:.1f}   "
          f"P(null <= {obs:+.1e}) = {(deltas <= obs).mean():.3f}")
    print(f"  a signal-free correction of this exact size reaches {obs:+.1e} or worse "
          f"{100 * (deltas <= obs).mean():.1f}% of the time")

    np.save(os.path.join(ROOT, "experiments", "w15e_nullband.npy"), deltas)


if __name__ == "__main__":
    main()
