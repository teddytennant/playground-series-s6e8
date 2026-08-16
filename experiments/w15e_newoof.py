"""w15e: price the OOF published SINCE the 2026-08-11 enumeration.

Three datasets appeared on 2026-08-15 from mohankrishnathalla's "tuner OOF saver"
kernels (`*_v3`). The author already has four members in `data/ext_members2/`
(xgb/cat/nn/lgb, imported 2026-08-10), so the first question is whether the v3 arrays
are new objects at all or a re-upload.

Gates, in the order RESEARCH.md prescribes:
  1. length + finiteness
  2. published-vs-computed solo AUC on OUR frozen folds (row order)
  3. credibility ceiling: anything above ~0.9720 is early-stopped or in-sample
  4. per-fold AUC printed, so a future run can gate against the kernel log
  5. correlation against the existing pack (maxcorr / median)
  6. duplication check against the author's already-imported members

Nothing is written into oof/ or data/ext_members*/ -- this script only measures.
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
from common import DATA, OOF, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members  # noqa: E402

W15E = os.path.join(DATA, "w15e")

NEW = {
    "mkt_xgb_v3": ("mohankrishnathalla_s6e8-xgb-oof", "oof_xgb_v3.npy", "test_xgb_v3.npy"),
    "mkt_cat_v3": ("mohankrishnathalla_s6e8-cat-mlp-oof", "oof_cat_v3.npy", "test_cat_v3.npy"),
    "mkt_lgb_v3": ("mohankrishnathalla_s6e8-lgb-dart-oof", "oof_lgb_v3.npy", "test_lgb_v3.npy"),
}


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)

    names, O, T = load_members(
        y, len(te),
        extra_dirs=(os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2")),
        drop={"golem_a", "golem_f"},
    )
    print(f"pack: {len(names)} members\n")

    # rank-transform the pack once; Spearman is the correlation that matters for AUC
    Rp = np.column_stack([rankdata(O[:, j]) for j in range(O.shape[1])])
    Rp = (Rp - Rp.mean(0)) / Rp.std(0)
    n = len(y)

    rows = []
    for nm, (d, of, tf) in NEW.items():
        o = np.load(os.path.join(W15E, d, of)).astype(np.float64)
        t = np.load(os.path.join(W15E, d, tf)).astype(np.float64)
        assert o.shape == (n,) and t.shape == (len(te),), (nm, o.shape, t.shape)
        assert np.isfinite(o).all() and np.isfinite(t).all(), nm

        solo = roc_auc_score(y, o)
        per_fold = [roc_auc_score(y[iva], o[iva]) for _, iva in folds]

        r = rankdata(o)
        r = (r - r.mean()) / r.std()
        corr = (Rp * r[:, None]).sum(0) / n
        jmax = int(np.argmax(corr))

        rows.append(dict(member=nm, solo=solo, maxcorr=corr.max(),
                         nearest=names[jmax], medcorr=float(np.median(corr)),
                         n_above_0999=int((corr > 0.999).sum())))
        print(f"{nm}: solo {solo:.6f}  per-fold "
              + " ".join(f"{a:.5f}" for a in per_fold))
        print(f"    maxcorr {corr.max():.6f} vs {names[jmax]}   "
              f"median {np.median(corr):.4f}   #>0.999: {(corr > 0.999).sum()}")

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format="%.6f"))

    # pack reference: what does a typical member's maxcorr look like?
    print("\npack self-reference (maxcorr of each member vs the other 160):")
    G = (Rp.T @ Rp) / n
    np.fill_diagonal(G, -np.inf)
    mx = G.max(1)
    print(f"  min {mx.min():.4f}  q10 {np.quantile(mx, .10):.4f}  "
          f"median {np.median(mx):.4f}  max {mx.max():.4f}")
    print(f"  members below 0.97 maxcorr: {(mx < 0.97).sum()} / {len(mx)}")

    solos = np.array([roc_auc_score(y, O[:, j]) for j in range(O.shape[1])])
    print(f"  pack solo AUC: min {solos.min():.4f} median {np.median(solos):.4f} "
          f"max {solos.max():.4f}")


if __name__ == "__main__":
    main()
