"""w16g: the deadline pick, re-run with the arm-average file in the candidate set.

w16c ran the paired private-slice simulation over the 11-file zero-parameter CV-top cluster
plus the three arms of the corrected object, and concluded the first pick should move from
`blend159av_h3` to `w16b_cellweight` (+6.49e-6 +/- 0.20 paired, P 0.912). It then found, by
re-running w16b's own selection rule leave-one-fold-out, that +1.78e-6 of w16b's headline is
ARM-SELECTION OPTIMISM.

w16f removes the selection step by rank-averaging the three arms, and lands at cross-fitted CV
0.97005536 against per-cell's 0.97005561 — the same number, minus the optimism. That makes it a
different and better-founded candidate for the first pick, and it was not in w16c's candidate
set because it did not exist yet.

This re-runs only the decision, on the same protocol: same seed, same f, the SAME simulated
private slice for every file each rep, so the sd of the difference is the sd of a paired
comparison and not of two independent draws.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402

from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import CELL_A, fast_auc, pct, rule_cells  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_TEST = 296_302
REPS = 500
SEED = 1616          # w16c's seed, so the two runs share their slice draws
F = 0.20
NPAR = {"blend159av_h3": 0, "blend160origm_h3": 0, "blendtop3": 0,
        "w15f_antistudent_avg": 1, "w16b_cellweight": 7, "w16f_armavg": 10}


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    j = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    assigns = {"glob": np.array(["0"] * n, dtype=object),
               "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
               "per_cell": cell}
    arm_oof = {}
    for key in ("glob", "a_only", "per_cell"):
        z = br.copy()
        for (_, iva), w in zip(folds, j["fold_weights"][key]):
            for lvl, wv in w.items():
                m = iva[assigns[key][iva] == lvl]
                z[m] = br[m] + float(wv) * c[m]
        arm_oof[key] = z

    vecs = {
        "blend159av_h3": np.load(os.path.join(SUB, "oof_blend159av_h3.npy")),
        "blend160origm_h3": np.load(os.path.join(SUB, "oof_blend160origm_h3.npy")),
        "blendtop3": np.load(os.path.join(SUB, "oof_blendtop3.npy")),
        "w15f_antistudent_avg": arm_oof["glob"],
        "w16b_cellweight": arm_oof["per_cell"],
        "w16f_armavg": np.mean([rankdata(arm_oof[k]) for k in
                                ("glob", "a_only", "per_cell")], axis=0),
    }
    names = list(vecs)
    print("candidate set (cross-fitted OOF for every corrected file):")
    for nm in names:
        print(f"  {nm:24s} p{NPAR[nm]:<2d} CV {fast_auc(y, vecs[nm].astype(np.float64)):.8f}")

    pk = {nm: prep(vecs[nm].astype(np.float64), y) for nm in names}
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    A = np.zeros((REPS, len(names)))
    for i in range(REPS):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[n_pub:]] = True
        for k, nm in enumerate(names):
            A[i, k] = subset_auc(pk[nm], m)
    mean = A.mean(axis=0)

    inc = A[:, names.index(BASE)]
    print(f"\nmean simulated private AUC, paired against the incumbent first pick {BASE}:")
    for k, nm in enumerate(names):
        d = A[:, k] - inc
        print(f"  {nm:24s} mean {mean[k]:.8f}  d {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P(better) {(d > 0).mean():.3f}")

    print("\nhead-to-head, paired, among the corrected files:")
    for a, b in itertools.combinations(
            ("w15f_antistudent_avg", "w16b_cellweight", "w16f_armavg"), 2):
        d = A[:, names.index(a)] - A[:, names.index(b)]
        print(f"  {a:22s} - {b:22s} {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P({a[:8]} better) {(d > 0).mean():.3f}")

    print("\nE[max] of every candidate PAIR, best first (this is what Kaggle scores):")
    pairs = []
    for i, k in itertools.combinations(range(len(names)), 2):
        mx = np.maximum(A[:, i], A[:, k])
        pairs.append((float(mx.mean()), names[i], names[k]))
    pairs.sort(reverse=True)
    incpair = np.maximum(A[:, names.index(BASE)], A[:, names.index("blend160origm_h3")])
    for e, a, b in pairs:
        d = np.maximum(A[:, names.index(a)], A[:, names.index(b)]) - incpair
        star = "  <-- 2026-08-13..15 incumbent" if {a, b} == {BASE, "blend160origm_h3"} else ""
        print(f"  E[max] {e:.8f}  vs incumbent {d.mean()*1e6:+6.2f}e-6 "
              f"+/-{d.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P {(d > 0).mean():.3f}  "
              f"p{NPAR[a]}+p{NPAR[b]}  {a} + {b}{star}")

    json.dump(dict(names=names, mean=[float(x) for x in mean],
                   pairs=[dict(e_max=e, a=a, b=b) for e, a, b in pairs]),
              open(os.path.join(HERE, "w16g_pickcheck.json"), "w"), indent=1)
    print("\nwrote experiments/w16g_pickcheck.json")


if __name__ == "__main__":
    main()
