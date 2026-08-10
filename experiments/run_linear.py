"""An additive log-odds member over the full-resolution lattice encodings.

WHY THIS SHAPE
--------------
Every one of the 149 members in the stack is a tree ensemble, a factorization machine or
a neural net. Not one is linear. That matters because the target encodings ARE the signal
here -- the public ablation record puts full-resolution target + frequency encoding at
+0.0023, larger than tuning and model selection combined -- and a target encoding is
already an estimate of P(y | cell). The natural way to combine several such estimates is
to add their log-odds, which is exactly a linear model in logit(TE) space and exactly what
no member in the pack is.

So the features are logit(TE_c) rather than TE_c. On the TE scale a linear model has to
spend capacity undoing the link function; on the logit scale the correct combination of
independent cell estimates is literally a sum, and the fitted coefficient per key becomes
an interpretable reliability weight. log1p(CT_c) rides alongside so the model can discount
thin cells, which is the same reason the count is emitted next to the mean in te_block.

This is a deliberately weak member -- it cannot represent an interaction the lattice keys
do not already name. It is here for decorrelation: a linear stacker can hand a weak member
a NEGATIVE coefficient and use it as a correction, which is how `pub_ryota` (solo 0.9637)
earns a coefficient of -0.10 in the shipped stack.

Reads the per-fold caches from experiments/build_cache.py -- the target encodings are
already fold-safe there, so nothing about leakage changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import CACHE, N_SPLITS, OOF, TARGET, get_folds, load_raw, save_preds  # noqa: E402
from run_lgbm import lattice  # noqa: E402

EPS = 1e-6


def design(X, cols, te_idx, ct_idx, base_idx, med):
    """logit(TE) + log1p(CT) + median-filled base columns."""
    te = np.clip(X[:, te_idx], EPS, 1 - EPS)
    out = [np.log(te / (1 - te)), np.log1p(np.maximum(X[:, ct_idx], 0.0))]
    b = X[:, base_idx].copy()
    bad = ~np.isfinite(b)
    b[bad] = np.take(med, np.where(bad)[1])
    out.append(b)
    return np.hstack(out).astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--frac", action="store_true")
    ap.add_argument("--folds", default="")
    a = ap.parse_args()

    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    te_idx = np.array([i for i, c in enumerate(cols) if c.startswith("TE_")])
    ct_idx = np.array([i for i, c in enumerate(cols) if c.startswith("CT_")])
    base_idx = np.array([i for i, c in enumerate(cols)
                         if not c.startswith(("TE_", "CT_"))])
    print(f"[{a.name}] {len(te_idx)} TE + {len(ct_idx)} CT + {len(base_idx)} base "
          f"-> {len(cols)} cols, C={a.C}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    Ftr = Fte = None
    if a.frac:
        Ftr, Fte = lattice(tr), lattice(te)

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    t0 = time.time()
    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        if a.frac:
            Xa = np.hstack([Xa, Ftr[itr]])
            Xb = np.hstack([Xb, Ftr[iva]])
            Xt = np.hstack([Xt, Fte])
            extra = np.arange(len(cols), Xa.shape[1])
            bidx = np.concatenate([base_idx, extra])
        else:
            bidx = base_idx
        # medians from the TRAINING part only, so the fill carries no held-out information
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(Xa[:, bidx], axis=0)
        med = np.nan_to_num(med)
        Da = design(Xa, cols, te_idx, ct_idx, bidx, med)
        mu, sd = Da.mean(0), Da.std(0) + 1e-9
        Da = (Da - mu) / sd
        del Xa
        m = LogisticRegression(max_iter=2000, C=a.C).fit(Da, ya)
        del Da
        oof[iva] = m.decision_function((design(Xb, cols, te_idx, ct_idx, bidx, med) - mu) / sd)
        tp += m.decision_function((design(Xt, cols, te_idx, ct_idx, bidx, med) - mu) / sd) / N_SPLITS
        print(f"[{a.name}] fold {f}: AUC {roc_auc_score(yb, oof[iva]):.5f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb, Xt, m

    if len(want) < N_SPLITS:
        print(f"[{a.name}] partial run, nothing saved", flush=True)
        return
    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] FULL OOF AUC = {cv:.5f}  ({time.time()-t0:.0f}s)", flush=True)
    # stored as probabilities: load_members/transform expect the member scale, and the
    # stacker's own to_logit inverts this exactly for a decision function.
    save_preds(a.name, 1 / (1 + np.exp(-oof)), 1 / (1 + np.exp(-tp)), len(y), len(te))
    json.dump(dict(name=a.name, cv=float(cv), C=a.C, frac=a.frac),
              open(os.path.join(OOF, f"summary_{a.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
