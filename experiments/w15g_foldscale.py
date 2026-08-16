"""w15g -- split the pooled-vs-within gap into its two CV-only components.

w15g_coupling.py measures `A_within - A_pooled`: how much a member's pooled OOF AUC
differs from the same AUC restricted to pairs that sit inside one fold.  Two DIFFERENT
CV-only artefacts can produce that gap, and they have completely different consequences:

  (1) LABEL COUPLING -- the lead this run was sent to test.  For a cross-fold pair
      (i in k, j in l), s_i's model was trained on folds != k, which contains y_j, and
      s_j's model on folds != l, which contains y_i.  Nothing like this exists at test
      time.  It is a property of the LABELS and cannot be transformed away.

  (2) INTER-FOLD SCALE MISMATCH -- five different fitted models emit five slightly
      different score scales.  Comparing a row scored by M_1 against a row scored by M_2
      therefore compares two rulers, and at test time there is only one ruler.  This one
      is a property of the SCORES and IS removable: rank-normalise each member's OOF
      inside each fold.

Telling them apart is exact, because A_within is invariant to any per-fold monotone map
(both rows of a within-fold pair get the same map, so their order cannot change) while
pooled AUC is not.  So:

    pooled_raw  --(fold-normalise)-->  pooled_fnorm  --(drop cross-fold pairs)-->  A_within
                 \_________________/                  \_______________________/
                   scale mismatch                      label coupling + composition

Controls: the same three quantities under a stratified re-partition of the rows into five
pseudo-folds.  Under that control the five pseudo-folds are exchangeable, so
fold-normalisation must be a no-op and the whole ladder must collapse; whatever spread it
shows is the instrument's resolution.

If (2) turns out to carry the gap, there is something to BUILD: the workspace's `rankraw`
transform ranks each member GLOBALLY over all 691,369 OOF rows
(agent/stack.py:transform), which matches the OOF scale to the test scale but does
nothing about the five rulers inside the OOF itself.  A per-fold rank would.

Usage:  python experiments/w15g_foldscale.py [n_control_draws]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, get_folds, load_raw  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15g_coupling import MEMBERS, NFOLD, FoldPairAUC, decompose, stratified_permute  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "w15g_foldscale.json")
SEED = 20260816


def fold_normalise(s: np.ndarray, fold_of_row: np.ndarray) -> np.ndarray:
    """Rank-gauss each fold's scores onto a common scale. Ties averaged, as `rankraw`."""
    out = np.empty_like(s)
    for k in range(NFOLD):
        m = fold_of_row == k
        out[m] = ndtri((rankdata(s[m]) - 0.5) / m.sum())
    return out


def three(s: np.ndarray, y: np.ndarray, fold_of_row, npos, nneg):
    """(pooled_raw, pooled_fnorm, A_within) for one score vector."""
    u = FoldPairAUC(s, y).u_matrix(fold_of_row)
    pooled_raw, within, _ = decompose(u, npos, nneg)
    sn = fold_normalise(s, fold_of_row)
    un = FoldPairAUC(sn, y).u_matrix(fold_of_row)
    pooled_fn, within_n, _ = decompose(un, npos, nneg)
    # invariance check: a per-fold monotone map cannot move a within-fold pair
    assert abs(within - within_n) < 1e-12, (within, within_n)
    return pooled_raw, pooled_fn, within


def main() -> None:
    ndraw = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    n = len(y)
    fold_of_row = np.empty(n, dtype=np.int64)
    for k, (_, va) in enumerate(get_folds(y)):
        fold_of_row[va] = k
    npos = np.array([(y[fold_of_row == k] > 0.5).sum() for k in range(NFOLD)], float)
    nneg = np.array([(y[fold_of_row == k] < 0.5).sum() for k in range(NFOLD)], float)

    rng = np.random.default_rng(SEED)
    ctrl = [stratified_permute(y, npos.astype(int), nneg.astype(int), rng)
            for _ in range(ndraw)]

    print(f"{'member':22s} {'recipe':26s} {'pooled':>10} "
          f"{'scale e-6':>10} {'coupling e-6':>13} {'total e-6':>10} | "
          f"{'ctrl scale':>11} {'ctrl coup':>11}")
    results = []
    for name, path, recipe, dose in MEMBERS:
        if not os.path.exists(path):
            continue
        s = np.load(path).astype(np.float64)
        p_raw, p_fn, w = three(s, y, fold_of_row, npos, nneg)
        scale = p_fn - p_raw          # what fold-normalisation recovers
        coup = w - p_fn               # what is left, and cannot be transformed away
        cs, cc = [], []
        for cf in ctrl:
            a, b, c = three(s, y, cf, npos, nneg)
            cs.append(b - a)
            cc.append(c - b)
        cs, cc = np.asarray(cs), np.asarray(cc)
        print(f"{name:22s} {recipe:26s} {p_raw:10.6f} "
              f"{scale * 1e6:+10.2f} {coup * 1e6:+13.2f} {(w - p_raw) * 1e6:+10.2f} | "
              f"{cs.mean() * 1e6:+6.2f}+-{cs.std(ddof=1) * 1e6:4.2f} "
              f"{cc.mean() * 1e6:+6.2f}+-{cc.std(ddof=1) * 1e6:4.2f}")
        results.append(dict(member=name, recipe=recipe, dose=dose, pooled=p_raw,
                            pooled_fnorm=p_fn, within=w, scale=scale, coupling=coup,
                            ctrl_scale_mu=float(cs.mean()), ctrl_scale_sd=float(cs.std(ddof=1)),
                            ctrl_coup_mu=float(cc.mean()), ctrl_coup_sd=float(cc.std(ddof=1))))

    with open(OUT, "w") as fh:
        json.dump(dict(ndraw=ndraw, seed=SEED, members=results), fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
