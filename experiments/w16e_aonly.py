"""w16e: build the 2-parameter A-only file — the arm w16b measured, scored, and did not ship.

WHY THIS FILE EXISTS
--------------------
w16b enumerated three arms of the same object (base + a fitted weight on `c_avg`) and shipped
the 7-parameter one because it had the highest cross-fitted delta:

    GLOBAL    1 weight   +3.338e-6   -> w15f_antistudent_avg, sent, LB 0.97107
    A-ONLY    2 weights  +5.031e-6   -> never built as a file
    PER-CELL  7 weights  +6.195e-6   -> w16b_cellweight, sent, LB 0.97107

w16c's nested check says that ranking is not as clean as it looks. Re-running w16b's OWN
selection rule leave-one-fold-out — choose the arm on four folds, read it on the fifth — picks
A-ONLY in 3 of 5 folds and returns +4.417e-6 (se 2.865e-6) rather than +6.195e-6. The
difference, **+1.78e-6, is the arm-selection optimism**, roughly double the 0.5-1e-6 w16b
estimated in its own honest caveat. So the middle arm is the one the honest version of w16b's
rule prefers more often than not, it costs 5 fewer fitted parameters, and it has never been
built. Building it is the cheapest way to put a third rung on the corrected-file ladder.

The weights are re-fitted on the full training set with the same coordinate ascent, same
0..0.02 grid, same 2 passes, same frozen folds for the cross-fitted number. The test side
re-derives its A membership from test.csv by the same rule function, so nothing is transferred
by row index.

Not a deadline pick: 2 fitted parameters above the stack.
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
    base_auc = fast_auc(y, br)

    d = np.array(j["arms"]["a_only"]["per_fold"])
    print(f"base {BASE} OOF AUC {base_auc:.8f}")
    print(f"A-only cross-fitted dAUC {d.mean():+.3e}  ({' '.join(f'{x:+.1e}' for x in d)})  "
          f"{int((d > 0).sum())}/5 folds   CV {base_auc + d.mean():.8f}")
    print(f"  vs its own permuted control "
          f"{d.mean() - np.array(j['arms']['ctrl2']['per_fold']).mean():+.3e}")

    w_full = ascend(y, br, c, a_only, np.arange(n))
    print(f"full-data weights: { {k: round(v, 4) for k, v in w_full.items()} }")

    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test_avg.npy")).astype(np.float64)
    assert ct.shape == (N_TEST,)

    assign_te = np.where(rule_cells(te) == CELL_A, "A", "rest").astype(object)
    assert set(assign_te) == set(w_full)
    shares = {str(v): float((assign_te == v).mean()) for v in sorted(set(assign_te))}
    print(f"test-side shares: { {k: round(v, 4) for k, v in shares.items()} }")

    pred = apply_w(btr, ct, assign_te, w_full, np.arange(N_TEST))
    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16e_aonly.csv")
    sub.to_csv(path, index=False)

    dupes = []
    rk = rankdata(strict)
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16e_aonly.csv":
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
    print(f"rank-identical to an existing submission file? {dupes or 'no'}")
    for other in ("w16b_cellweight", "w15f_antistudent_avg", BASE):
        p = os.path.join(SUB, f"{other}.csv")
        if os.path.exists(p):
            ov = pd.read_csv(p).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
            print(f"  spearman vs {other:22s} {np.corrcoef(rk, rankdata(ov))[0,1]:.7f}   "
                  f"rows differing {int((rankdata(ov) != rk).sum()):,}")
    print(f"wrote {path}  rows {len(sub):,}")

    json.dump(dict(cv=float(base_auc + d.mean()), xfit=float(d.mean()),
                   full_weights={str(k): float(v) for k, v in w_full.items()},
                   test_shares=shares, dupes=dupes),
              open(os.path.join(HERE, "w16e_aonly.json"), "w"), indent=1)
    print("wrote experiments/w16e_aonly.json")


if __name__ == "__main__":
    main()
