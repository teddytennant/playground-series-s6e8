"""w126b -- is the simplex searcher's NEGATIVE optimism at k=8 a property of the search, or a
property of its ITERATION BUDGET? (2026-08-30)

w126a's R5 ran w36d's `fit_w` VERBATIM on member columns and got optimisms
+1.08 +1.81 +3.71 +4.41 -2.70 e-6 at k=2,3,4,6,8. The last one is negative, which is not a
thing selection optimism can be: `insample` is the score of a maximiser on the rows it
maximised over, so it cannot honestly sit BELOW the cross-fitted score.

The suspicion is mechanical. `fit_w` passes `maxiter=150` to Nelder-Mead, a budget that does
not scale with k while the simplex it has to walk has k vertices. At k=3-4, where w36d measured,
150 iterations converge. At k=8 they may not, and then `insample` is not a maximum at all --
it is wherever the optimiser stopped, which can be below a cross-fitted average of five other
stopping points.

⚠ w125 5 caught this exact shape in its own script: an assertion printed where a measurement
was available. So this measures it. IN-SAMPLE ONLY -- one fit per budget, no folds -- because
the claim is entirely about whether the in-sample number is a maximum.

  If AUC(maxiter=150) < AUC(maxiter=2400), the k=8 in-sample number is budget-limited and the
  negative optimism is an artefact of the optimiser, not a reading about search cost.
  If they agree to ~1e-9, the budget is not the cause and the negative optimism is real noise.

    .venv/bin/python experiments/w126b_budget.py

Deterministic: same columns, same seed-42 nested order, same objective as w126a/w36d.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, TARGET, load_raw            # noqa: E402
from stack import DEFAULT_DROP, load_members          # noqa: E402

EXT = os.path.join(DATA, "ext_members")
EXT2 = os.path.join(DATA, "ext_members2")
KS = (4, 8)
BUDGETS = (150, 600, 2400)      # 150 is w36d's, verbatim
MEMBER_SEED = 42


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def fast_auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def fit_w(R, y, w0, maxiter):
    """w36d's `fit_w` with the ONE constant this file varies exposed as an argument."""
    def neg(w):
        w = np.abs(w)
        s = w.sum()
        if s <= 0:
            return 0.0
        return -fast_auc(y, R @ (w / s))
    r = minimize(neg, w0, method="Nelder-Mead",
                 options=dict(xatol=2e-4, fatol=2e-10, maxiter=maxiter, disp=False))
    w = np.abs(r.x)
    return w / w.sum(), r.nit, r.nfev


def main() -> int:
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, Om, _ = load_members(y, len(te), extra_dirs=(EXT, EXT2), drop=set(DEFAULT_DROP))
    vet = pd.read_csv(os.path.join(EXT2, "_vetting.csv")).set_index("member")
    base = [n for n in names if n not in vet.index]
    idx = {n: i for i, n in enumerate(names)}
    order = list(np.random.default_rng(MEMBER_SEED).permutation(base))
    Rm = np.column_stack([rk(Om[:, idx[n]]) for n in order])
    print(f"base {len(base)}, nested order seed={MEMBER_SEED} -- w126a's columns exactly\n")

    out, verdicts = {}, []
    for k in KS:
        R = Rm[:, :k]
        w0 = np.full(k, 1.0 / k)
        eq = roc_auc_score(y, R @ w0)
        print(f"k={k}   equal-weight in-sample AUC {eq:.10f}")
        rows = {}
        for b in BUDGETS:
            t = time.time()
            w, nit, nfev = fit_w(R, y, w0, b)
            auc = roc_auc_score(y, R @ w)
            rows[b] = {"auc": auc, "nit": int(nit), "nfev": int(nfev)}
            print(f"    maxiter={b:<5d} in-sample {auc:.10f}  "
                  f"({(auc-eq)*1e6:+8.3f}e-6 vs equal)  nit={nit} nfev={nfev}  "
                  f"{'BUDGET-LIMITED' if nit >= b else 'converged'}  ({time.time()-t:.0f}s)",
                  flush=True)
        gain = rows[max(BUDGETS)]["auc"] - rows[150]["auc"]
        v = ("budget-limited" if gain > 1e-9 else "not budget-limited")
        verdicts.append((k, gain, v))
        print(f"    ** k={k}: maxiter {max(BUDGETS)} finds {gain*1e6:+.3f}e-6 more in-sample "
              f"AUC than w36d's 150 -> {v} **\n")
        out[k] = rows

    print("VERDICT")
    for k, gain, v in verdicts:
        print(f"  k={k:<3d} {gain*1e6:+8.3f}e-6  {v}")
    json.dump(out, open(os.path.join(HERE, "w126b_budget.json"), "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
