"""w15f stage 9: the honest best construction, its cross-fitted CV, and the test file.

Stage 6 showed the author's additive signed-square at his published 0.10 is a null on our
base (-3.4e-6, z -0.54) while smaller weights are mildly positive. Reading the peak off that
five-point curve and shipping it would be fitting a weight to the same 691,369 rows it is
scored on -- the exact move this workspace's own rules forbid.

So the weight is chosen CROSS-FITTED in both functional forms: for each frozen fold, search
the weight on the other four folds and score it on the held-out one. That number is honest,
it is the one the submission decision is made on, and it is reported whether or not it is
positive.

  form A  additive : base_rank + w * c        (the author's, c = sign(r)|r|^2 per fold)
  form B  rankblend: (1-w)*rank(base) + w*rank(r)   (w15d's perfectly-conditioned instrument)

Note rank(c) == rank(r) exactly, since sign(r)|r|^2 is monotone in r, so form B does not
depend on the signed square at all -- the square only matters in form A, where it decides how
the additive move is DISTRIBUTED across rows.

The student is stage 2's `smooth`, fixed before any of this was measured. The smoothness
ladder in stage 7 is a sensitivity check, NOT a menu to pick the best-scoring student from.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15f_eval import pct  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
FALLBACK = "blend159"
N_TEST = 296_302
GRID_ADD = np.linspace(0.0, 0.20, 101)
GRID_RNK = np.linspace(0.0, 0.02, 101)


def main():
    z = np.load(os.path.join(HERE, "w15f_nested.npz"))
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    base_auc = roc_auc_score(y, base_p)
    fb_auc = roc_auc_score(y, np.load(os.path.join(SUB, f"oof_{FALLBACK}.npy")))
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_trans.npy"))
    r = z["teacher_r"] - z["student_t"]
    mr = rankdata(r) / n

    print(f"base     {BASE:>16}  cross-fitted CV {base_auc:.7f}")
    print(f"fallback {FALLBACK:>16}  cross-fitted CV {fb_auc:.7f}\n")

    out = {"base_auc": float(base_auc), "fallback_auc": float(fb_auc)}
    forms = {
        "A_additive": (GRID_ADD, lambda w, i: br[i] + w * c[i]),
        "B_rankblend": (GRID_RNK, lambda w, i: (1 - w) * br[i] + w * mr[i]),
    }
    best = None
    for nm, (grid, f) in forms.items():
        ws, deltas = [], []
        for itr, iva in folds:
            a = np.array([roc_auc_score(y[itr], f(w, itr)) for w in grid])
            wk = grid[int(a.argmax())]
            ws.append(wk)
            deltas.append(roc_auc_score(y[iva], f(wk, iva))
                          - roc_auc_score(y[iva], f(0.0, iva)))
        deltas = np.array(deltas)
        full = np.array([roc_auc_score(y, f(w, np.arange(n))) for w in grid])
        w_full = grid[int(full.argmax())]
        print(f"{nm:>12}: cross-fitted w per fold {np.round(ws, 4)}")
        print(f"{'':>12}  cross-fitted dAUC {deltas.mean():+.3e}  "
              f"({' '.join(f'{d:+.1e}' for d in deltas)})  "
              f"{int((deltas > 0).sum())}/5 folds")
        print(f"{'':>12}  full-data w* {w_full:.4f}  in-sample gain "
              f"{full.max()-full[0]:+.3e}")
        out[nm] = dict(xfit_w=[float(x) for x in ws], xfit=float(deltas.mean()),
                       folds_pos=int((deltas > 0).sum()), w_full=float(w_full),
                       insample=float(full.max() - full[0]))
        if best is None or deltas.mean() > best[1]:
            best = (nm, deltas.mean(), w_full)

    nm, xfit, w_full = best
    cand_cv = base_auc + xfit
    print(f"\nbetter cross-fitted form: {nm}  dAUC {xfit:+.3e}  -> candidate CV "
          f"{cand_cv:.7f}")
    print(f"vs fallback {fb_auc:.7f}: {cand_cv - fb_auc:+.3e}")
    out["chosen"] = dict(form=nm, xfit=float(xfit), w=float(w_full),
                         candidate_cv=float(cand_cv))

    # ---------------- build the test file ------------------------------------
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    assert (test["id"].to_numpy() == ids).all()
    bt = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id").reindex(ids)
    bp = bt[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test.npy"))
    assert ct.shape == (N_TEST,)

    if nm == "A_additive":
        pred = btr + w_full * ct
    else:
        pred = (1 - w_full) * btr + w_full * (rankdata(ct) / N_TEST)

    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})

    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w15f_antistudent_cv.csv")
    sub.to_csv(path, index=False)

    rho = float(np.corrcoef(rankdata(strict), rankdata(bp))[0, 1])
    moved = int((rankdata(pred) != rankdata(bp)).sum())
    print(f"\nwrote {path}  rows {len(sub):,}  range "
          f"[{strict.min():.3e}, {strict.max():.6f}]")
    print(f"  spearman vs {BASE}: {rho:.7f}   rows whose rank moved: {moved:,}")
    out["file"] = dict(path=path, spearman_vs_base=rho, moved=moved)

    json.dump(out, open(os.path.join(HERE, "w15f_final.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
