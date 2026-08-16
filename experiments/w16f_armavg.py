"""w16f: average the three arms instead of selecting one — the fix for w16c's optimism finding.

THE PROBLEM THIS SOLVES
-----------------------
w16b built one object (base + a fitted weight on `c_avg`) at three resolutions and shipped the
arm with the highest cross-fitted delta:

    GLOBAL    1 weight   +3.338e-6      A-ONLY  2 weights  +5.031e-6      PER-CELL 7  +6.195e-6

w16c re-ran that selection rule leave-one-fold-out — choose the arm on four folds, read it on
the fifth. It picks A-ONLY in 3 of 5 folds and returns **+4.417e-6 (se 2.865e-6)**, not
+6.195e-6. **The +1.78e-6 gap is arm-selection optimism**, about double w16b's own estimate,
and the honest conclusion is that the three arms are not separated by the data: the nested rule
cannot even decide which one it wants.

When a selection rule cannot separate its candidates, averaging them is strictly better than
picking one. It removes the selection step entirely, so there is no optimism left to correct,
and it lowers the variance of the fitted correction without adding a parameter. This is the
same argument the workspace already accepted for seeds and folds (w14c, blend159av); it has
simply never been applied to the *arm* dimension.

WHAT IS AVERAGED
----------------
The three cross-fitted OOF vectors, each rebuilt from w16b's stored per-fold weights so that
fold f's rows carry weights fitted without fold f, then rank-averaged. On the test side the
three shipped/derived test vectors are rank-averaged the same way. No new weight is fitted
anywhere in this script, and the arm ensemble has no free parameter of its own.

Reported against the honest baseline: the nested +4.417e-6, not the selected +6.195e-6.

Not a deadline pick — it inherits the arms' fitted parameters.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402

from w16b_cellweight import CELL_A, apply_w, ascend, fast_auc, pct, rule_cells  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_TEST = 296_302
ARMS = (("glob", "global"), ("a_only", "a_only"), ("per_cell", "per_cell"))


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    j = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    one = np.array(["0"] * n, dtype=object)
    assigns = {"glob": one, "a_only": a_only, "per_cell": cell}
    base_auc = fast_auc(y, br)
    print(f"base {BASE} OOF AUC {base_auc:.8f}")

    # ---- cross-fitted OOF for each arm, then their rank average
    oofs = {}
    for key, armname in ARMS:
        z = br.copy()
        for (_, iva), w in zip(folds, j["fold_weights"][key]):
            for lvl, wv in w.items():
                m = iva[assigns[key][iva] == lvl]
                z[m] = br[m] + float(wv) * c[m]
        oofs[key] = z
        print(f"  arm {armname:9s} cross-fitted CV {fast_auc(y, z):.8f}")
    avg_oof = np.mean([rankdata(oofs[k]) for k, _ in ARMS], axis=0)
    cv_avg = fast_auc(y, avg_oof)
    nested = 4.417e-06
    print(f"\n  ARM AVERAGE   cross-fitted CV {cv_avg:.8f}")
    print(f"    vs base            {(cv_avg - base_auc)*1e6:+.3f}e-6")
    print(f"    vs nested-honest   {(cv_avg - base_auc - nested)*1e6:+.3f}e-6   "
          f"(the +4.417e-6 an unselected rule actually delivers)")
    print(f"    vs per_cell as sent{(cv_avg - fast_auc(y, oofs['per_cell']))*1e6:+.3f}e-6")

    # per-fold, so the comparison carries an uncertainty rather than a point
    per_fold = []
    for _, iva in folds:
        per_fold.append(fast_auc(y[iva], avg_oof[iva]) - fast_auc(y[iva], br[iva]))
    per_fold = np.array(per_fold)
    se = per_fold.std(ddof=1) / np.sqrt(5)
    print(f"    per fold {' '.join(f'{x*1e6:+.2f}' for x in per_fold)}  "
          f"mean {per_fold.mean()*1e6:+.3f}e-6  se {se*1e6:.3f}e-6  "
          f"t(4df) {per_fold.mean()/se:+.2f}  {int((per_fold > 0).sum())}/5")

    # ---------------------------------------------------------------- the test file
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test_avg.npy")).astype(np.float64)
    cell_te = rule_cells(te)
    assigns_te = {"glob": np.array(["0"] * N_TEST, dtype=object),
                  "a_only": np.where(cell_te == CELL_A, "A", "rest").astype(object),
                  "per_cell": cell_te}

    ranks = []
    weights_used = {}
    for key, armname in ARMS:
        w_full = ascend(y, br, c, assigns[key], np.arange(n))
        weights_used[armname] = {str(k): float(v) for k, v in w_full.items()}
        assert set(assigns_te[key]) == set(w_full)
        p = apply_w(btr, ct, assigns_te[key], w_full, np.arange(N_TEST))
        ranks.append(rankdata(p))
        print(f"  full-data weights {armname:9s} "
              f"{ {k: round(v, 4) for k, v in w_full.items()} }")
    avg_rank = np.mean(ranks, axis=0)

    order = np.lexsort((ids, btr, avg_rank))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16f_armavg.csv")
    sub.to_csv(path, index=False)

    rk = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16f_armavg.csv":
            continue
        try:
            o = pd.read_csv(os.path.join(SUB, f))
        except Exception:
            continue
        if list(o.columns) != ["id", TARGET] or len(o) != N_TEST:
            continue
        ov = o.set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        if np.array_equal(rankdata(ov), rk):
            dupes.append(f)
    print(f"\n  rank-identical to an existing submission file? {dupes or 'no'}")
    for other in ("w16b_cellweight", "w15f_antistudent_avg", BASE):
        p = os.path.join(SUB, f"{other}.csv")
        if os.path.exists(p):
            ov = pd.read_csv(p).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
            print(f"    spearman vs {other:22s} {np.corrcoef(rk, rankdata(ov))[0,1]:.7f}   "
                  f"rows differing {int((rankdata(ov) != rk).sum()):,}")
    print(f"  wrote {path}  rows {len(sub):,}  range "
          f"[{strict.min():.3e}, {strict.max():.3f}]")

    json.dump(dict(cv=float(cv_avg), delta_vs_base=float(cv_avg - base_auc),
                   per_fold=[float(x) for x in per_fold], se=float(se),
                   arm_cv={a: float(fast_auc(y, oofs[k])) for k, a in ARMS},
                   full_weights=weights_used, dupes=dupes),
              open(os.path.join(HERE, "w16f_armavg.json"), "w"), indent=1)
    print("  wrote experiments/w16f_armavg.json")


if __name__ == "__main__":
    main()
