"""w16j: average the BASE dimension as well as the arm dimension.

Every corrected file this workspace has shipped -- w15f_antistudent_avg, w16b_cellweight,
w16f_armavg -- is `blend159av_h3` plus a fitted weight on `c_avg`. w16f removed the *arm*
selection by averaging the three arms. The *base* was still a single file, chosen because it
was the argmax of the six zero-parameter h3 stacks.

`w16h_pickavg.py` swept that dimension. Two results matter here:

  * the base argmax is STABLE -- the leave-one-fold-out rule picks `blend159av_h3` in 5/5
    folds, so unlike the arm choice it carries no measurable selection optimism; and
  * on the top-k sweep over the same six candidates, k=3 has the best nested fold-mean
    (0.97005312) -- better than k=1 (0.97005270, the pure pick) and better than k=6
    (0.97005238, the pure average). All three sit inside 0.8e-6 of each other, so the
    dimension is flat, but the ordering is what `blendtop3` already is: the rank-average of
    the three joint-top h3 files.

So this file is the same correction with both dimensions averaged rather than picked: base =
`blendtop3` (three bases averaged, zero fitted parameters, the highest-CV zero-parameter file
in the workspace at 0.97004946 against `blend159av_h3`'s 0.97004917), correction = the three
arms averaged, exactly as `w16f_armavg` does. Nine (base, arm) combinations collapse to one
object with no argmax anywhere in it.

Honest up front: the base half of this is worth about +0.3e-6, which is UNDER the workspace's
~2e-6 reproducibility floor. This is not a discovery. It is the highest-CV file the workspace
holds, built by a rule with no selection step left in it, and it costs nothing.

Arm weights are refitted here rather than reused, because the base changed and the weights are
fitted against the base's ranks. Same frozen SKF5 seed42 folds, same 0..0.02 grid, same 2-pass
coordinate ascent, same cross-fitting (fold f's rows carry weights fitted without fold f).
Nothing here is chosen with reference to the public leaderboard.
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
BASE = "blendtop3"
N_TEST = 296_302
ARMS = ("glob", "a_only", "per_cell")


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    cell = rule_cells(tr)
    assigns = {
        "glob": np.array(["0"] * n, dtype=object),
        "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
        "per_cell": cell,
    }
    base_auc = fast_auc(y, br)
    ref = fast_auc(y, pct(np.load(os.path.join(SUB, "oof_blend159av_h3.npy"))))
    print(f"base {BASE} OOF AUC {base_auc:.8f}   (blend159av_h3 {ref:.8f})", flush=True)

    oofs, per_fold, fw = {}, {}, {}
    for k in ARMS:
        z = br.copy()
        d, ws = [], []
        for itr, iva in folds:
            w = ascend(y, br, c, assigns[k], itr)
            ws.append({str(a): float(b) for a, b in w.items()})
            zv = apply_w(br, c, assigns[k], w, iva)
            z[iva] = zv
            d.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
        oofs[k], per_fold[k], fw[k] = z, np.array(d), ws
        se = per_fold[k].std(ddof=1) / np.sqrt(5)
        print(f"  arm {k:<9} xfit {per_fold[k].mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {per_fold[k].mean()/se:+5.2f}  {int((per_fold[k] > 0).sum())}/5  "
              f"CV {fast_auc(y, z):.8f}", flush=True)

    avg = np.mean([rankdata(oofs[k]) for k in ARMS], axis=0)
    cv = fast_auc(y, avg)
    d = np.array([fast_auc(y[iva], avg[iva]) - fast_auc(y[iva], br[iva]) for _, iva in folds])
    se = d.std(ddof=1) / np.sqrt(5)
    print(f"\n  ARM AVERAGE on {BASE}  cross-fitted CV {cv:.8f}")
    print(f"    vs base            {(cv - base_auc)*1e6:+.3f}e-6")
    print(f"    vs w16f_armavg     {(cv - 0.97005536)*1e6:+.3f}e-6  (w16f = same arms on "
          f"blend159av_h3, CV 0.97005536, LB 0.97107)")
    print(f"    per fold {' '.join(f'{x*1e6:+.2f}' for x in d)}  mean {d.mean()*1e6:+.3f}e-6  "
          f"se {se*1e6:.3f}  t(4df) {d.mean()/se:+.2f}  {int((d > 0).sum())}/5")

    # paired against w16f's cross-fitted OOF, rebuilt here from w16b's stored fold weights
    stored = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))
    br_ref = pct(np.load(os.path.join(SUB, "oof_blend159av_h3.npy")).astype(np.float64))
    ref_ranks = []
    for k in ARMS:
        z = br_ref.copy()
        for (_, iva), w in zip(folds, stored["fold_weights"][k]):
            z[iva] = apply_w(br_ref, c, assigns[k], {a: float(b) for a, b in w.items()}, iva)
        ref_ranks.append(rankdata(z))
    w16f_oof = np.mean(ref_ranks, axis=0)
    print(f"    w16f_armavg rebuilt CV {fast_auc(y, w16f_oof):.8f} (published 0.97005536)")
    dd = np.array([fast_auc(y[iva], avg[iva]) - fast_auc(y[iva], w16f_oof[iva])
                   for _, iva in folds])
    s2 = dd.std(ddof=1) / np.sqrt(5)
    print(f"    paired vs w16f per fold {' '.join(f'{x*1e6:+.2f}' for x in dd)}  "
          f"mean {dd.mean()*1e6:+.3f}e-6 se {s2*1e6:.3f} t {dd.mean()/s2:+.2f}  "
          f"{int((dd > 0).sum())}/5")

    # ---------------------------------------------------------------- the test file
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    bp = pd.read_csv(os.path.join(SUB, f"{BASE}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    btr = pct(bp)
    ct = np.load(os.path.join(HERE, "w15f_c_test_avg.npy")).astype(np.float64)
    cell_te = rule_cells(te)
    assigns_te = {
        "glob": np.array(["0"] * N_TEST, dtype=object),
        "a_only": np.where(cell_te == CELL_A, "A", "rest").astype(object),
        "per_cell": cell_te,
    }
    ranks, full_w = [], {}
    for k in ARMS:
        w = ascend(y, br, c, assigns[k], np.arange(n))
        full_w[k] = {str(a): float(b) for a, b in w.items()}
        assert set(assigns_te[k]) == set(w)
        ranks.append(rankdata(apply_w(btr, ct, assigns_te[k], w, np.arange(N_TEST))))
        print(f"  full-data weights {k:<9} { {a: round(b,4) for a, b in w.items()} }", flush=True)
    avg_rank = np.mean(ranks, axis=0)

    order = np.lexsort((ids, btr, avg_rank))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all() and sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16j_basearm.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, "oof_w16j_basearm.npy"), avg / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16j_basearm.csv":
            continue
        try:
            o = pd.read_csv(os.path.join(SUB, f))
        except Exception:
            continue
        if list(o.columns) != ["id", TARGET] or len(o) != N_TEST:
            continue
        ov = o.set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        if np.array_equal(rankdata(ov), r):
            dupes.append(f)
    print(f"\n  wrote {path}  rows {len(sub):,}  range [{strict.min():.3e}, {strict.max():.3f}]")
    print(f"  rank-identical to an existing submission? {dupes or 'no'}")
    for other in ("w16f_armavg", "w16b_cellweight", "w15f_antistudent_avg", "blendtop3",
                  "blend159av_h3"):
        ov = pd.read_csv(os.path.join(SUB, f"{other}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<22} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    json.dump(dict(base=BASE, base_auc=float(base_auc), cv=float(cv),
                   arms={k: dict(xfit=float(v.mean()), per_fold=[float(x) for x in v],
                                 cv=float(fast_auc(y, oofs[k]))) for k, v in per_fold.items()},
                   avg_per_fold=[float(x) for x in d], fold_weights=fw,
                   full_weights=full_w, dupes=dupes),
              open(os.path.join(HERE, "w16j_basearm.json"), "w"), indent=1)
    print("  wrote experiments/w16j_basearm.json")


if __name__ == "__main__":
    main()
