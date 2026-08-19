"""M13 follow-up: is the cross-process CV gap caused by BLAS thread count?

w27z proved the matrix and the in-process fit are bit-exact, and that float64 agrees
with float32 to 0.28e-6 -- so neither R-M11f nor precision explains the +0.9..+3.3e-6
by which today's runs sit ABOVE the shipped on-disk numbers on identical code. The two
remaining candidates are BLAS thread count (summation order) and a code change since
the shipped build. This isolates the first: same process is impossible, so the thread
count is set from the environment and the cell is refitted.

    OMP_NUM_THREADS=n .venv/bin/python experiments/w27z2_threads.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import DATA, get_folds  # noqa: E402
from blend_lab import load_all  # noqa: E402

DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "lat_ctraw_r400", "lat_ctfixte_r400")
D346 = tuple(os.path.join(DATA, d) for d in
             ("ext_members3", "ext_members4", "ext_members6"))
SHIPPED = 0.9700993121

names, y, mats, _ = load_all(("hybrid",), DROP, extra_dirs=D346, dtype="float32", quiet=True)
assert len(names) == 188
Z = mats["hybrid"][0]
s = Z.std(0); s[s <= 0] = 1.0
Z = (Z / s).astype(Z.dtype)
mo = np.zeros(len(y)); iters = []
for itr, iva in get_folds(y):
    m = LogisticRegression(max_iter=5000, C=1.0).fit(Z[itr], y[itr])
    iters.append(int(np.ravel(m.n_iter_)[0]))
    mo[iva] = m.decision_function(Z[iva])
a = roc_auc_score(y, mo)
print(f"OMP={os.environ.get('OMP_NUM_THREADS','unset'):>5s}  AUC {a:.10f}  "
      f"vs shipped {(a-SHIPPED)*1e6:+7.3f}e-6  n_iter {iters}", flush=True)
