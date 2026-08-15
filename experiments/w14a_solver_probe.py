"""Why the stack does not reproduce: is it the solver, and is the difference real?

`w14a_repro.py` rebuilt `blend159av` from the frozen folds and the member library and got
CVs 1.4e-6 to 3.1e-6 BELOW the stored ones, with test decision functions differing by up to
1.06 and 99.9% of test rows changing rank. No member file was written after the original
build and sklearn logged no ConvergenceWarning, so neither "the inputs moved" nor "it ran
out of iterations" explains it.

The remaining candidate is the conditioning of the stack itself. 159 members correlating
0.987-0.999 give a near-singular Hessian, so the L2-logistic loss has a long flat valley;
at `C=1` over 553,095 rows the penalty is ~0, so nothing picks a point in that valley.
Two runs can both satisfy lbfgs's gradient tolerance at coefficient vectors that are far
apart and yet score the same to 6dp.

Three measurements, each with its own control:

1.  **In-process determinism.** Fit fold 0 twice in the same process. Identical coefficients
    mean the code path is deterministic and the run-to-run difference is environmental.
2.  **Thread sensitivity.** The same fit under a different BLAS/OpenMP thread count, run as
    a separate process. Different reduction order -> different rounding -> a different point
    in the valley. (Capped at 4 threads per the workspace concurrency rule; the contrast is
    1 vs 4, which is sufficient.)
3.  **Is the difference real or is the valley flat?** Compare the two solutions' training
    log-loss and held-out fold AUC. If the coefficients move a lot while the loss and the
    AUC move by ~0, the disagreement is reparametrisation inside a flat valley and the
    reported CV is stable even though the file is not.

    OMP_NUM_THREADS=4 w14a_solver_probe.py 4
    OMP_NUM_THREADS=1 w14a_solver_probe.py 1
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from common import TARGET, get_folds, load_raw  # noqa: E402
import blend_lab  # noqa: E402
from w14a_repro import DROP  # noqa: E402

KIND = "rescale"  # cheapest transform to build; the conditioning story is not kind-specific
OUT = os.path.join(HERE, "w14a_solver_probe.json")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    names, y, mats, _ = blend_lab.load_all((KIND, "rankraw"), set(DROP), quiet=True)
    Z = mats[KIND][0]
    itr, iva = get_folds(y)[0]
    print(f"{len(names)} members, fold 0: {len(itr):,} train / {len(iva):,} val", flush=True)

    # conditioning of the design the stacker actually sees
    s = np.linalg.svd(Z[itr[:60000]].astype(np.float64), compute_uv=False)
    print(f"condition number (60k-row subsample): {s[0] / s[-1]:.3e}", flush=True)

    res = []
    for rep in (0, 1):
        m = LogisticRegression(max_iter=5000, C=1.0).fit(Z[itr], y[itr])
        c = m.coef_.ravel().astype(np.float64)
        dv = m.decision_function(Z[iva])
        res.append({
            "coef": c,
            "n_iter": int(m.n_iter_[0]),
            "auc": roc_auc_score(y[iva], dv),
            "trloss": log_loss(y[itr], m.predict_proba(Z[itr])[:, 1]),
            "dv": dv,
        })
        print(f"  rep{rep}: n_iter {res[-1]['n_iter']:4d}  fold-0 AUC {res[-1]['auc']:.6f}  "
              f"train logloss {res[-1]['trloss']:.9f}  ||coef|| {np.linalg.norm(c):.4f}",
              flush=True)

    a, b = res
    print(f"\n[1] in-process determinism: coef identical = "
          f"{np.array_equal(a['coef'], b['coef'])}   "
          f"max|dcoef| {np.abs(a['coef'] - b['coef']).max():.3e}")

    # persist rep0 so a run under a different thread count can be compared to it
    store = {}
    if os.path.exists(OUT):
        store = json.load(open(OUT))
    store[tag] = {"coef": a["coef"].tolist(), "n_iter": a["n_iter"],
                  "auc": a["auc"], "trloss": a["trloss"],
                  "dv_head": a["dv"][:2000].tolist()}
    json.dump(store, open(OUT, "w"))

    others = [k for k in store if k != tag]
    for o in others:
        oc = np.array(store[o]["coef"])
        odv = np.array(store[o]["dv_head"])
        cos = float(oc @ a["coef"] / (np.linalg.norm(oc) * np.linalg.norm(a["coef"])))
        print(f"\n[2] vs run '{o}' ({o} thread(s)):")
        print(f"    max|dcoef| {np.abs(oc - a['coef']).max():.3e}   "
              f"cosine {cos:.9f}   n_iter {store[o]['n_iter']} vs {a['n_iter']}")
        print(f"    max|d decision_function| (first 2000 val rows) "
              f"{np.abs(odv - a['dv'][:2000]).max():.3e}")
        print(f"[3] and does it matter?  d fold-0 AUC {a['auc'] - store[o]['auc']:+.3e}   "
              f"d train logloss {a['trloss'] - store[o]['trloss']:+.3e}")

    if not others:
        print("\n[2/3] no other thread count on file yet -- run this again with a "
              "different OMP_NUM_THREADS to get the cross-process contrast.")


if __name__ == "__main__":
    main()
