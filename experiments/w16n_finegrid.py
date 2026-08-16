"""w16n: the grid's CEILING was not binding (w16m). Its RESOLUTION has never been tested.

WHERE THIS COMES FROM
---------------------
`experiments/w16m_widegrid.py`, run immediately before this one, answered the question w16i §6
item 2 and w16l §4 item 1 both put at the top of the board: is the `c_avg` correction's
coordinate-ascent grid binding at the top? The answer is an exact no. Widening the box from
`linspace(0, 0.02, 41)` to `linspace(0, 0.05, 101)` returned the **identical weight vector in
all 30 fits** (5 schemes x 5 folds + 5 full-data). The decile arm's `q6 = 0.0200` in fold 0 was
never a constrained optimum -- it is the argmax over 0..0.05 as well, and it merely coincides
with the old ceiling. Nothing is pinned at 0.05 anywhere.

That closes the ceiling but not the box. `linspace(0, 0.02, 41)` is TWO fixed choices, and
w16a set both at once: a ceiling of 0.02 and a **step of 5e-4**. The ceiling is now measured at
zero. The step has never been varied, and it is the half that actually binds on every fit --
every weight this workspace has ever fitted for this correction is a multiple of 5e-4 by
construction, and the decile arm's fitted values (0.0055, 0.0065, 0.0105, 0.0155, 0.0180) sit
in a range where one step is 3-9% of the weight.

WHAT THIS RUNS
--------------
The identical object -- base `blend159av_h3`, `c_avg`, the same five partitions, the frozen
SKF5 seed42 folds, the same 2-pass coordinate ascent, the same 5-arm rank average -- with the
grid refined to `linspace(0, 0.02, 201)`, step **1e-4**. That is a strict superset of w16a's
grid (5e-4 = 5 x 1e-4), so the fine fit cannot be worse in sample, and the narrow fit is an
exact matched control for the same reason w16m's was: the two runs differ in the candidate
weight set and in nothing else.

The narrow weights are not refitted here. They are loaded from `w16m_widegrid.json`, which
produced them with this same `ascend` and asserted them equal to `w16i_schemeavg.json`'s stored
per-fold AND full-data weights for all five arms. This script re-asserts that chain before
using them.

PRE-REGISTRATION -- fixed before any number in this script existed
------------------------------------------------------------------
  1. **SHIP the FINE-grid 5-arm scheme average, unconditionally, whatever its CV**, including
     if it is a regression against `w16i_schemeavg`, and including if it comes out
     rank-identical (in which case nothing is sent, because scores are deterministic and
     resubmitting an identical file is forbidden -- that is a fact about the file, not a
     decision about its score).
  2. **The deadline pick moves only if** the fine arm beats the narrow one on plain
     cross-fitted CV AND the paired per-fold difference is positive in at least 4 of 5 folds.
     Evaluated mechanically and printed. Anything short of that leaves `check_selection.py`
     alone.
  3. No sub-combination of the five arms is shipped, and no arm is dropped. The 5-arm set is
     fixed, for w16i's reason.

There is no permuted-membership control here for the same reason as in w16m: the narrow grid
IS the control, and it is exact.

Nothing here is chosen with reference to the public leaderboard.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, get_folds, load_raw  # noqa: E402
from w16b_cellweight import CELL_A, apply_w, fast_auc, pct, rule_cells  # noqa: E402
from w16i_schemeavg import decile_levels, mask_levels  # noqa: E402
from w16m_widegrid import ARMS, ascend  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_TEST = 296_302
FINE = np.linspace(0.0, 0.02, 201)          # step 1e-4; w16a's step was 5e-4

_G = {}


def _task(job):
    arm, f = job
    idx = _G["folds"][f][0] if f >= 0 else np.arange(len(_G["y"]))
    w = ascend(_G["y"], _G["br"], _G["c"], _G["assigns"][arm], idx, FINE)
    return arm, f, {str(k): float(v) for k, v in w.items()}


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    print(f"base {BASE} OOF AUC {fast_auc(y, br):.8f}", flush=True)

    cell = rule_cells(tr)
    assigns = {
        "glob": np.array(["0"] * n, dtype=object),
        "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
        "rule": cell,
        "mask": mask_levels(tr),
        "decile": decile_levels(br),
    }
    _G.update(y=y, br=br, c=c, assigns=assigns, folds=folds)

    # ---- the matched control: w16m's narrow weights, re-verified against w16i's
    wm = json.load(open(os.path.join(HERE, "w16m_widegrid.json")))["weights"]
    wi = json.load(open(os.path.join(HERE, "w16i_schemeavg.json")))

    def same(a, b):
        return set(a) == set(b) and all(abs(float(a[k]) - float(b[k])) < 1e-12 for k in a)

    NAR = {}
    for arm in ARMS:
        for f in list(range(5)) + [-1]:
            NAR[(arm, f)] = {k: float(v) for k, v in wm[f"narrow|{arm}|{f}"].items()}
            want = wi["fold_weights"][arm][f] if f >= 0 else wi["full_weights"][arm]
            assert same(NAR[(arm, f)], want), (arm, f)
    print("  VERIFIED: w16m's narrow weights == w16i's shipped weights, all 5 arms, 6 fits each",
          flush=True)

    jobs = [(a, f) for a in ARMS for f in list(range(5)) + [-1]]
    print(f"fitting {len(jobs)} fine-grid ascents (step 1e-4, {len(FINE)} points)", flush=True)
    FIN = {}
    with ProcessPoolExecutor(max_workers=15, mp_context=mp.get_context("fork")) as ex:
        for arm, f, w in ex.map(_task, jobs):
            FIN[(arm, f)] = w
            off = sum(1 for v in w.values() if abs(v * 2000 - round(v * 2000)) > 1e-9)
            print(f"  fine {arm:<7} fold {f:>2}  {off}/{len(w)} off the 5e-4 lattice  "
                  f"{ {k: round(v, 5) for k, v in w.items()} }", flush=True)

    def oof_of(store, arm):
        z = br.copy()
        d = []
        for f, (_, iva) in enumerate(folds):
            zv = apply_w(br, c, assigns[arm], store[(arm, f)], iva)
            z[iva] = zv
            d.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
        return z, np.array(d)

    print("\n=== per-arm cross-fitted delta vs base, narrow (5e-4) vs fine (1e-4), paired ===")
    oofs, arm_rows = {}, {}
    for arm in ARMS:
        zn, dn = oof_of(NAR, arm)
        zf, df = oof_of(FIN, arm)
        oofs[("narrow", arm)], oofs[("fine", arm)] = zn, zf
        diff = df - dn
        se = diff.std(ddof=1) / np.sqrt(5)
        moved = sum(1 for f in list(range(5)) + [-1] if not same(FIN[(arm, f)], NAR[(arm, f)]))
        print(f"  {arm:<7} narrow {dn.mean()*1e6:+7.3f}e-6  fine {df.mean()*1e6:+7.3f}e-6  "
              f"diff {diff.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  {int((diff > 0).sum())}/5  "
              f"fine-CV {fast_auc(y, zf):.8f}  fits that moved {moved}/6")
        arm_rows[arm] = dict(narrow=float(dn.mean()), fine=float(df.mean()),
                             diff=float(diff.mean()), se=float(se),
                             diff_per_fold=[float(x) for x in diff],
                             folds_pos=int((diff > 0).sum()),
                             cv_narrow=float(fast_auc(y, zn)), cv_fine=float(fast_auc(y, zf)),
                             fits_moved=int(moved), fine_weights_full=FIN[(arm, -1)])

    def avg(gtag):
        v = np.mean([rankdata(oofs[(gtag, a)]) for a in ARMS], axis=0)
        d = np.array([fast_auc(y[iva], v[iva]) - fast_auc(y[iva], br[iva]) for _, iva in folds])
        return fast_auc(y, v), d, v

    print("\n=== the 5-arm scheme average (the shipped object), narrow vs fine ===")
    res = {}
    for gtag in ("narrow", "fine"):
        cv, d, v = avg(gtag)
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"  {gtag:<6} CV {cv:.8f}  xfit {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se:+5.2f}  {int((d > 0).sum())}/5")
        res[gtag] = dict(cv=float(cv), d=d, v=v)

    ref = np.load(os.path.join(SUB, "oof_w16i_schemeavg.npy")).astype(np.float64)
    print(f"  narrow reproduces oof_w16i_schemeavg.npy: rank corr "
          f"{np.corrcoef(rankdata(ref), rankdata(res['narrow']['v']))[0, 1]:.8f}")

    diff = res["fine"]["d"] - res["narrow"]["d"]
    se = diff.std(ddof=1) / np.sqrt(5)
    print(f"\n  FINE - NARROW  {diff.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
          f"t(4df) {diff.mean()/se if se > 0 else float('nan'):+5.2f}  "
          f"{int((diff > 0).sum())}/5  [{' '.join(f'{x*1e6:+6.2f}' for x in diff)}]")

    move = bool(res["fine"]["cv"] > res["narrow"]["cv"] and int((diff > 0).sum()) >= 4)
    print(f"\n  PRE-REGISTERED PICK RULE (fine CV > narrow CV AND >=4/5 folds positive): {move}")
    print("  -> deadline pick MOVES to w16n_finegrid" if move
          else "  -> deadline pick UNCHANGED: {w16i_schemeavg.csv, blend159av_h3.csv}")

    # ------------------------------------------------------------------ the test file
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
        "rule": cell_te,
        "mask": mask_levels(te),
        "decile": decile_levels(btr),
    }
    ranks = []
    for a in ARMS:
        w = FIN[(a, -1)]
        assert set(assigns_te[a]) == set(w), (a, set(assigns_te[a]), set(w))
        ranks.append(rankdata(apply_w(btr, ct, assigns_te[a], w, np.arange(N_TEST))))
    avg_rank = np.mean(ranks, axis=0)

    order = np.lexsort((ids, btr, avg_rank))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all() and sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16n_finegrid.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, "oof_w16n_finegrid.npy"), res["fine"]["v"] / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16n_finegrid.csv":
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
    print(f"\n  wrote {path}  rows {len(sub):,}  "
          f"range [{strict.min():.3e}, {strict.max():.3f}]  distinct {sub[TARGET].nunique():,}")
    print(f"  rank-identical to an existing submission? {dupes or 'no'}  "
          f"{'-> DO NOT SEND' if dupes else '-> sendable'}")
    for other in ("w16i_schemeavg", "w16f_armavg", "w16b_cellweight", BASE):
        ov = pd.read_csv(os.path.join(SUB, f"{other}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<20} {np.corrcoef(r, rankdata(ov))[0, 1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    json.dump(dict(arms=arm_rows,
                   avg={g: dict(cv=res[g]["cv"], xfit=float(res[g]["d"].mean()),
                                per_fold=[float(x) for x in res[g]["d"]])
                        for g in ("narrow", "fine")},
                   fine_minus_narrow=dict(mean=float(diff.mean()), se=float(se),
                                          per_fold=[float(x) for x in diff],
                                          folds_pos=int((diff > 0).sum())),
                   pick_rule_move=move,
                   fine_weights={f"{a}|{f}": FIN[(a, f)] for (a, f) in FIN},
                   dupes=dupes),
              open(os.path.join(HERE, "w16n_finegrid.json"), "w"), indent=1)
    print("  wrote experiments/w16n_finegrid.json")


if __name__ == "__main__":
    main()
