"""w16i: the segmentation SCHEME was picked too. Average over schemes, not just over arms.

WHERE THIS COMES FROM
---------------------
w16c found that w16b had *selected* one arm of the `c_avg` correction (global / A-only /
per-cell) on the same cross-fitted folds it scored on, worth +1.78e-6 of optimism, and fixed it
by averaging the three arms (`w16f_armavg`, CV 0.9700554, LB 0.97107). It then asked slot 3 to
sweep the workspace for the same shape.

`w16h_pickavg.py` did that sweep on the three dimensions slot 2 named. All three stay closed,
and the reason they stay closed is the useful part -- see that script's log. The short version:
when the leave-one-fold-out rule picks the SAME candidate in 5/5 folds, the argmax carries no
measurable optimism and averaging is pure dilution. w16c's arm rule was unstable (3/5 A-only,
2/5 per-cell) and that instability is exactly what made averaging the right move.

That test points at one place nobody has looked. w16a measured the `c_avg` residual under THREE
segmentation schemes and reported a chi-squared for each:

    generator rule cells   chi2 52.21 / 7 df
    base-score decile      chi2 42.41 / 7 df
    per-row missing count  chi2 15.05 / 4 df

It then built the correction on the rule cells alone, and w16b/w16c/w16f all inherited that
choice without ever pricing it. **The arms w16f averaged are three resolutions WITHIN one
scheme; the scheme itself was picked by argmax chi-squared.** That is the same failure shape
one level up, and it has never been nested.

WHAT THIS RUNS
--------------
Five arms of the same object -- base + a fitted weight on `c_avg`, one weight per level --
differing only in how the rows are partitioned:

    glob      1 level                                    (= w15f_antistudent_avg)
    a_only    2 levels   cell A vs rest                  (w16b's 2-parameter arm)
    rule      7 levels   the generator's rule cells      (= w16b_cellweight)
    mask      5 levels   per-row missing count 0..4+
    decile    8 levels   base-score octile

glob/a_only/rule reuse w16b's stored per-fold weights verbatim (asserted to reproduce its
printed cross-fitted deltas to the digit); mask and decile are fitted here on the same grid,
same 2-pass coordinate ascent, same frozen SKF5 seed42 folds, searched on four folds and read
on the fifth, each against a size-matched permuted-membership control.

Then the two questions, in order:
  1. Is the SCHEME choice stable under leave-one-fold-out? If it is, the scheme pick costs
     nothing and w16f stands as the honest object.
  2. If it is not, the 5-arm rank average is the object with no selection left in it, and its
     cross-fitted CV is honest as it stands.

WHAT GETS SHIPPED, fixed here before the run and not revisited afterwards
------------------------------------------------------------------------
**The 5-arm average, always.** Sub-combinations (drop mask, drop decile, ...) are computed and
printed because their spread is informative, but the argmax over them is NOT shipped and must
not be, for exactly the reason this whole script exists: choosing the best-CV combination on
the same cross-fitted folds that scored it re-introduces the selection step being audited, one
level up. The first version of this script shipped that argmax; that was the same bug in the
audit of the bug, and it is fixed here. If a future run wants to drop an arm it must do so on
a criterion fixed before seeing the combination CVs -- the permuted-membership control is the
obvious candidate, and w16c's disclosure about needing an `xfit > 0` floor applies to it.

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
BASE = "blend159av_h3"
N_TEST = 296_302
CTRL_SEED = 20260816


def mask_levels(df):
    nm = df.isna().sum(axis=1).to_numpy()
    return np.array([f"m{min(int(v), 4)}" for v in nm], dtype=object)


def decile_levels(base_rank, k=8):
    q = np.floor(base_rank * k).astype(int)
    q[q == k] = k - 1
    return np.array([f"q{int(v)}" for v in q], dtype=object)


def permuted(assign, rng):
    """Size-matched control: same level sizes, membership shuffled."""
    out = np.empty(len(assign), dtype=object)
    out[rng.permutation(len(assign))] = assign
    return out


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    rng = np.random.default_rng(CTRL_SEED)

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
    for k, a in assigns.items():
        print(f"  scheme {k:<7} {len(set(a))} levels  "
              f"sizes {sorted(np.unique(a, return_counts=True)[1].tolist(), reverse=True)}")

    stored = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))
    reuse = {"glob": "glob", "a_only": "a_only", "rule": "per_cell"}
    fold_w, per_fold, oofs = {}, {}, {}

    for name in assigns:
        if name in reuse:
            fold_w[name] = [{k: float(v) for k, v in w.items()}
                            for w in stored["fold_weights"][reuse[name]]]
        else:
            fold_w[name] = []
            for itr, _ in folds:
                fold_w[name].append(ascend(y, br, c, assigns[name], itr))
                print(f"    fitted {name} fold {len(fold_w[name])-1} "
                      f"{ {k: round(v,4) for k, v in fold_w[name][-1].items()} }", flush=True)
        z = br.copy()
        d = []
        for (_, iva), w in zip(folds, fold_w[name]):
            zv = apply_w(br, c, assigns[name], w, iva)
            z[iva] = zv
            d.append(fast_auc(y[iva], zv) - fast_auc(y[iva], br[iva]))
        per_fold[name] = np.array(d)
        oofs[name] = z
        se = per_fold[name].std(ddof=1) / np.sqrt(5)
        print(f"  arm {name:<7} xfit {per_fold[name].mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {per_fold[name].mean()/se:+5.2f}  {int((per_fold[name] > 0).sum())}/5  "
              f"CV {fast_auc(y, z):.8f}", flush=True)

    # reproduction check against w16b's printed numbers
    for name, key in reuse.items():
        got, want = per_fold[name].mean(), stored["arms"][key if key != "glob" else "global"]["xfit"]
        assert abs(got - want) < 1e-12, (name, got, want)
    print("  reuse check: glob/a_only/rule reproduce w16b's xfit deltas exactly", flush=True)

    # ---- size-matched permuted-membership controls for the two NEW schemes
    ctrl = {}
    for name in ("mask", "decile"):
        pa = permuted(assigns[name], rng)
        d = []
        for itr, iva in folds:
            w = ascend(y, br, c, pa, itr)
            d.append(fast_auc(y[iva], apply_w(br, c, pa, w, iva)) - fast_auc(y[iva], br[iva]))
        ctrl[name] = np.array(d)
        print(f"  CTRL permuted {name:<7} xfit {ctrl[name].mean()*1e6:+7.3f}e-6  "
              f"-> real minus control {(per_fold[name].mean()-ctrl[name].mean())*1e6:+7.3f}e-6",
              flush=True)

    # ------------------------------------------------------------------ Q1: is the pick stable?
    names = list(assigns)
    print("\n=== Q1  leave-one-fold-out over the FIVE schemes ===")
    picks, held = [], []
    for f in range(5):
        others = [g for g in range(5) if g != f]
        best = max(names, key=lambda nm: per_fold[nm][others].mean())
        picks.append(best)
        held.append(per_fold[best][f])
    held = np.array(held)
    naive_arm = max(names, key=lambda nm: per_fold[nm].mean())
    naive = per_fold[naive_arm]
    print(f"  naive argmax arm = {naive_arm}  xfit {naive.mean()*1e6:+.3f}e-6")
    print(f"  nested picks     = {picks}")
    print(f"  nested delta     = {held.mean()*1e6:+.3f}e-6  se "
          f"{held.std(ddof=1)/np.sqrt(5)*1e6:.3f}  "
          f"[{' '.join(f'{x*1e6:+6.2f}' for x in held)}]")
    print(f"  SCHEME-SELECTION OPTIMISM = {(naive.mean()-held.mean())*1e6:+.3f}e-6   "
          f"stability {max(picks.count(p) for p in set(picks))}/5")

    # ------------------------------------------------------------------ Q2: the averages
    def avg_cv(keys):
        v = np.mean([rankdata(oofs[k]) for k in keys], axis=0)
        d = np.array([fast_auc(y[iva], v[iva]) - fast_auc(y[iva], br[iva]) for _, iva in folds])
        return fast_auc(y, v), d, v

    print("\n=== Q2  averaged objects (no selection inside them) ===")
    combos = {
        "w16f 3-arm (glob+a_only+rule)": ["glob", "a_only", "rule"],
        "5-arm (all schemes)": names,
        "4-arm (drop decile)": ["glob", "a_only", "rule", "mask"],
        "4-arm (drop mask)": ["glob", "a_only", "rule", "decile"],
    }
    out = {}
    for tag, keys in combos.items():
        cv, d, v = avg_cv(keys)
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"  {tag:<32} CV {cv:.8f}  xfit {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se:+5.2f}  {int((d > 0).sum())}/5")
        out[tag] = dict(cv=float(cv), xfit=float(d.mean()), se=float(se),
                        per_fold=[float(x) for x in d], vec=v, keys=keys)

    best_tag = max(out, key=lambda t: out[t]["cv"])
    SHIP = "5-arm (all schemes)"
    print(f"\n  highest cross-fitted CV: {best_tag}  {out[best_tag]['cv']:.8f}")
    if best_tag != SHIP:
        print(f"  ** NOT SHIPPING THE ARGMAX. ** Shipping the pre-registered {SHIP} "
              f"(CV {out[SHIP]['cv']:.8f}) instead: picking the best sub-combination here "
              f"would re-introduce exactly the selection step this script exists to price.")

    # paired against w16f's object
    ref = out["w16f 3-arm (glob+a_only+rule)"]
    for tag in out:
        if tag == "w16f 3-arm (glob+a_only+rule)":
            continue
        d = np.array(out[tag]["per_fold"]) - np.array(ref["per_fold"])
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"    {tag:<32} - w16f  {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se if se > 0 else float('nan'):+5.2f}  {int((d > 0).sum())}/5")

    # ------------------------------------------------------------------ the test file
    keys = out[SHIP]["keys"]
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
    ranks, full_w = [], {}
    for k in keys:
        w = ascend(y, br, c, assigns[k], np.arange(n))
        full_w[k] = {str(a): float(b) for a, b in w.items()}
        assert set(assigns_te[k]) == set(w), (k, set(assigns_te[k]), set(w))
        ranks.append(rankdata(apply_w(btr, ct, assigns_te[k], w, np.arange(N_TEST))))
        print(f"  full-data weights {k:<7} { {a: round(b,4) for a, b in w.items()} }", flush=True)
    avg_rank = np.mean(ranks, axis=0)

    order = np.lexsort((ids, btr, avg_rank))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all() and sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16i_schemeavg.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, "oof_w16i_schemeavg.npy"), out[SHIP]["vec"] / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16i_schemeavg.csv":
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
          f"range [{strict.min():.3e}, {strict.max():.3f}]")
    print(f"  rank-identical to an existing submission? {dupes or 'no'}")
    for other in ("w16f_armavg", "w16b_cellweight", "w15f_antistudent_avg", BASE):
        ov = pd.read_csv(os.path.join(SUB, f"{other}.csv")).set_index("id") \
            .reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<22} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    json.dump(dict(base_auc=float(base_auc),
                   arms={k: dict(xfit=float(v.mean()), per_fold=[float(x) for x in v],
                                 cv=float(fast_auc(y, oofs[k]))) for k, v in per_fold.items()},
                   ctrl={k: float(v.mean()) for k, v in ctrl.items()},
                   nested_picks=picks, nested_delta=float(held.mean()),
                   naive_arm=naive_arm, naive_delta=float(naive.mean()),
                   scheme_optimism=float(naive.mean() - held.mean()),
                   combos={t: {kk: vv for kk, vv in d.items() if kk != "vec"}
                           for t, d in out.items()},
                   shipped=SHIP, cv_argmax_combo=best_tag,
                   fold_weights={k: [{a: float(b) for a, b in w.items()} for w in v]
                                 for k, v in fold_w.items()},
                   full_weights=full_w, dupes=dupes),
              open(os.path.join(HERE, "w16i_schemeavg.json"), "w"), indent=1)
    print("  wrote experiments/w16i_schemeavg.json")


if __name__ == "__main__":
    main()
