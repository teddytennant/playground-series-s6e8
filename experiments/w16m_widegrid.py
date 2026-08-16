"""w16m: the correction's weight grid was a fixed box and the decile arm is pinned against it.

WHERE THIS COMES FROM
---------------------
w16a set the coordinate-ascent grid for the `c_avg` correction to `linspace(0, 0.02, 41)` when
the only partition being fitted was the seven generator rule cells, whose fitted weights all
land at 0.0000-0.0090, i.e. deep inside the box. w16i then added two more schemes and the
decile arm does NOT fit inside it: `q6` (the 7th octile of the base score) returned exactly
0.0200 -- the ceiling -- in fold 0, and 0.0115 / 0.0155 / 0.0180 / 0.0155 in the other four,
with the full-data fit at 0.0170. One of six fits is against the wall and the rest are in the
top quarter of the box. The optimum for that level is plausibly outside the searched region,
so every number the decile arm contributes to `w16i_schemeavg` is a constrained optimum.

w16i §6 item 2 and w16l §4 item 1 both name this as the cheapest untested thing on the board,
and w16i named the replacement ceiling: **0.05**. That number is taken from the journal rather
than chosen here, deliberately, so that no discretion of mine enters the grid definition. The
step stays 5e-4, so the narrow grid is a strict SUBSET of the wide one and the wide fit can
never be worse in sample.

PRE-REGISTRATION -- written and committed to before any number in this script existed
--------------------------------------------------------------------------------------
This is the fourth time in this wave that a fixed choice has been relaxed and the best point
picked afterwards (arms: +1.78e-6 of optimism, schemes: +1.55e-6, sub-combinations: caught
inside w16i's own audit before it shipped). Relaxing the grid ceiling and then shipping
whichever of {narrow, wide} scores better is that bug a fourth time. So:

  1. **SHIP the WIDE-grid 5-arm scheme average, unconditionally, whatever its CV** -- including
     if it is a regression against `w16i_schemeavg`. The file is named before the numbers
     exist. This is w16l's discipline and it is the only thing that makes the comparison
     readable afterwards.
  2. **The deadline pick moves only if** the wide arm beats the narrow one on plain
     cross-fitted CV AND the paired per-fold difference is positive in at least 4 of 5 folds.
     Both conditions are evaluated mechanically at the bottom of this script and printed.
     Anything short of that leaves `check_selection.py` alone.
  3. No sub-combination of the five arms is shipped, for w16i's reason. The 5-arm set is fixed.

WHY NO PERMUTED CONTROL
-----------------------
Because the narrow grid IS the control, and it is an exact one. Narrow and wide differ in
nothing but the set of candidate weights: same base, same `c_avg`, same five partitions, same
frozen SKF5 seed42 folds, same 2-pass coordinate ascent, same rank-average. The paired
per-fold difference therefore isolates the grid ceiling and nothing else. A permuted-membership
control answers a different question (is the partition real?), which w16i already answered for
both new schemes.

The narrow refit is not loaded from w16i's JSON -- it is recomputed here by this script's own
ascent and then ASSERTED equal to w16i's stored per-fold weights for all five arms, so the
harness is verified against the shipped object before the wide numbers are believed.

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

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
N_TEST = 296_302
PASSES = 2                                   # w16b's, unchanged
GRIDS = {
    "narrow": np.linspace(0.0, 0.02, 41),    # w16a's box, what w16i shipped
    "wide": np.linspace(0.0, 0.05, 101),     # ceiling named by w16i §6 item 2
}
ARMS = ["glob", "a_only", "rule", "mask", "decile"]

_G = {}


def ascend(y, br, c, assign, idx, grid):
    """w16b's coordinate ascent, verbatim except the grid is a parameter."""
    lv = sorted(set(assign[idx]))
    w = {v: 0.0 for v in lv}
    cur = br[idx].copy()
    for _ in range(PASSES):
        for v in lv:
            mv = assign[idx] == v
            if mv.sum() < 500:
                continue
            z = cur.copy()
            best, bw = -1.0, w[v]
            for g in grid:
                z[mv] = br[idx][mv] + g * c[idx][mv]
                a = fast_auc(y[idx], z)
                if a > best:
                    best, bw = a, g
            w[v] = bw
            cur[mv] = br[idx][mv] + bw * c[idx][mv]
    return w


def _task(job):
    gtag, arm, f = job
    idx = _G["folds"][f][0] if f >= 0 else np.arange(len(_G["y"]))
    w = ascend(_G["y"], _G["br"], _G["c"], _G["assigns"][arm], idx, GRIDS[gtag])
    return gtag, arm, f, {str(k): float(v) for k, v in w.items()}


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)

    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    base_auc = fast_auc(y, br)
    print(f"base {BASE} OOF AUC {base_auc:.8f}", flush=True)

    cell = rule_cells(tr)
    assigns = {
        "glob": np.array(["0"] * n, dtype=object),
        "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
        "rule": cell,
        "mask": mask_levels(tr),
        "decile": decile_levels(br),
    }
    _G.update(y=y, br=br, c=c, assigns=assigns, folds=folds)

    jobs = [(g, a, f) for g in GRIDS for a in ARMS for f in list(range(5)) + [-1]]
    print(f"fitting {len(jobs)} ascents "
          f"({len(GRIDS)} grids x {len(ARMS)} arms x 6)", flush=True)
    W = {}
    with ProcessPoolExecutor(max_workers=15, mp_context=mp.get_context("fork")) as ex:
        for gtag, arm, f, w in ex.map(_task, jobs):
            W[(gtag, arm, f)] = w
            print(f"  {gtag:<6} {arm:<7} fold {f:>2}  "
                  f"{ {k: round(v, 4) for k, v in w.items()} }", flush=True)

    # ---- harness verification: narrow must reproduce w16i's stored weights exactly
    stored = json.load(open(os.path.join(HERE, "w16i_schemeavg.json")))

    def same(a, b):
        return set(a) == set(b) and all(abs(a[k] - float(b[k])) < 1e-12 for k in a)

    for arm in ARMS:
        for f in range(5):
            got, want = W[("narrow", arm, f)], stored["fold_weights"][arm][f]
            assert same(got, want), (arm, f, got, want)
        gotf, wantf = W[("narrow", arm, -1)], stored["full_weights"][arm]
        assert same(gotf, wantf), (arm, "full", gotf, wantf)
    print("  VERIFIED: narrow-grid refit reproduces w16i's per-fold AND full-data weights "
          "for all 5 arms, exactly", flush=True)

    # ---- cross-fitted OOF per (grid, arm)
    per_fold, oofs = {}, {}
    for gtag in GRIDS:
        for arm in ARMS:
            z = br.copy()
            d = []
            for f, (_, iva) in enumerate(folds):
                zv = apply_w(br, c, assigns[arm], W[(gtag, arm, f)], iva)
                z[iva] = zv
                d.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
            per_fold[(gtag, arm)] = np.array(d)
            oofs[(gtag, arm)] = z

    print("\n=== per-arm cross-fitted delta vs base, narrow vs wide (paired) ===")
    arm_rows = {}
    for arm in ARMS:
        dn, dw = per_fold[("narrow", arm)], per_fold[("wide", arm)]
        diff = dw - dn
        se = diff.std(ddof=1) / np.sqrt(5)
        bound = sum(1 for f in list(range(5)) + [-1]
                    if max(W[("wide", arm, f)].values()) >= GRIDS["wide"][-1] - 1e-12)
        print(f"  {arm:<7} narrow {dn.mean()*1e6:+7.3f}e-6  wide {dw.mean()*1e6:+7.3f}e-6  "
              f"diff {diff.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"{int((diff > 0).sum())}/5  wide-CV {fast_auc(y, oofs[('wide', arm)]):.8f}  "
              f"fits pinned at 0.05: {bound}/6")
        arm_rows[arm] = dict(narrow=float(dn.mean()), wide=float(dw.mean()),
                             diff=float(diff.mean()), se=float(se),
                             diff_per_fold=[float(x) for x in diff],
                             folds_pos=int((diff > 0).sum()),
                             cv_wide=float(fast_auc(y, oofs[("wide", arm)])),
                             cv_narrow=float(fast_auc(y, oofs[("narrow", arm)])),
                             pinned_at_ceiling=int(bound),
                             wide_weights_full=W[("wide", arm, -1)])

    # ---- the 5-arm scheme average under each grid
    def avg(gtag):
        v = np.mean([rankdata(oofs[(gtag, a)]) for a in ARMS], axis=0)
        d = np.array([fast_auc(y[iva], v[iva]) - fast_auc(y[iva], br[iva]) for _, iva in folds])
        return fast_auc(y, v), d, v

    print("\n=== the 5-arm scheme average (the shipped object), narrow vs wide ===")
    res = {}
    for gtag in GRIDS:
        cv, d, v = avg(gtag)
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"  {gtag:<6} CV {cv:.8f}  xfit {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se:+5.2f}  {int((d > 0).sum())}/5")
        res[gtag] = dict(cv=float(cv), d=d, v=v)

    ref = np.load(os.path.join(SUB, "oof_w16i_schemeavg.npy")).astype(np.float64)
    print(f"  narrow reproduces oof_w16i_schemeavg.npy: rank corr "
          f"{np.corrcoef(rankdata(ref), rankdata(res['narrow']['v']))[0, 1]:.8f}  "
          f"CV of stored file {fast_auc(y, ref):.8f}")

    diff = res["wide"]["d"] - res["narrow"]["d"]
    se = diff.std(ddof=1) / np.sqrt(5)
    print(f"\n  WIDE - NARROW  {diff.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
          f"t(4df) {diff.mean()/se if se > 0 else float('nan'):+5.2f}  "
          f"{int((diff > 0).sum())}/5  "
          f"[{' '.join(f'{x*1e6:+6.2f}' for x in diff)}]")

    move = bool(res["wide"]["cv"] > res["narrow"]["cv"] and int((diff > 0).sum()) >= 4)
    print(f"\n  PRE-REGISTERED PICK RULE (wide CV > narrow CV AND >=4/5 folds positive): {move}")
    print("  -> deadline pick MOVES to w16m_widegrid" if move
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
        w = W[("wide", a, -1)]
        assert set(assigns_te[a]) == set(w), (a, set(assigns_te[a]), set(w))
        ranks.append(rankdata(apply_w(btr, ct, assigns_te[a], w, np.arange(N_TEST))))
    avg_rank = np.mean(ranks, axis=0)

    order = np.lexsort((ids, btr, avg_rank))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all() and sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16m_widegrid.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, "oof_w16m_widegrid.npy"), res["wide"]["v"] / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16m_widegrid.csv":
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
    print(f"  rank-identical to an existing submission? {dupes or 'no'}")
    for other in ("w16i_schemeavg", "w16f_armavg", "w16b_cellweight", BASE):
        ov = pd.read_csv(os.path.join(SUB, f"{other}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<20} {np.corrcoef(r, rankdata(ov))[0, 1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    json.dump(dict(base_auc=float(base_auc),
                   arms=arm_rows,
                   avg={g: dict(cv=res[g]["cv"], xfit=float(res[g]["d"].mean()),
                                per_fold=[float(x) for x in res[g]["d"]]) for g in GRIDS},
                   wide_minus_narrow=dict(mean=float(diff.mean()), se=float(se),
                                          per_fold=[float(x) for x in diff],
                                          folds_pos=int((diff > 0).sum())),
                   pick_rule_move=move,
                   weights={f"{g}|{a}|{f}": W[(g, a, f)]
                            for (g, a, f) in W},
                   dupes=dupes),
              open(os.path.join(HERE, "w16m_widegrid.json"), "w"), indent=1)
    print("  wrote experiments/w16m_widegrid.json")


if __name__ == "__main__":
    main()
