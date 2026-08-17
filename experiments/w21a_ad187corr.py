"""w21a: the 5-arm scheme-average `c_avg` correction, rebuilt on the 187-member base.

WHY THIS RUNS NOW
-----------------
w20 imported 22 members from `adarsh1077/s6e8-adarsh-oof-library` and the resulting
`w20_ad187_h3` took CV 0.9701008150, +51.6e-6 on `blend159av_h3` and +45.2e-6 on the standing
deadline pick `w16i_schemeavg`. But `w16i_schemeavg` is not the bare 159-member stack -- it is
that stack PLUS the 5-arm scheme-average `c_avg` correction, worth +6.4e-6 of cross-fitted
gain on its own base. That correction has never been applied to the new base, so the two
objects currently in the option set are not comparable on equal terms: the CV leader has no
correction and the corrected file has an inferior base.

This is the cheapest remaining CV gain in the workspace and it makes the presumptive new
deadline pick strictly better than the one it replaces on the SAME construction.

WHAT THIS IS
------------
`w16q_ens4base.py`'s Part 2 verbatim, with BASE swapped from `blend159av` to `w20_ad187_h3`.
Five arms of the same object -- base rank plus a fitted weight on `c_avg`, one weight per
level -- differing only in the row partition:

    glob      1 level                                    (= w15f_antistudent_avg's scheme)
    a_only    2 levels   cell A vs rest
    rule      7 levels   the generator's rule cells
    mask      5 levels   per-row missing count 0..4+
    decile    8 levels   base-score octile

ALL FIVE ARMS ARE REFIT FROM SCRATCH. w16b's stored per-fold weights were fitted against the
h3 base of the 159-member stack and do not transfer; w16q established that by refitting them
for the ens4 base, and the same applies here with more force since the base has changed by
+51.6e-6 rather than by a transform.

WHAT GETS SHIPPED, fixed here before the run and not revisited afterwards
------------------------------------------------------------------------
**The 5-arm average, unconditionally.** Sub-combinations (drop mask, drop decile, ...) are
computed and printed because their spread is informative, but the argmax over them is NOT
shipped, for the reason w16i exists: choosing the best-CV sub-combination on the same
cross-fitted folds that scored it re-introduces the selection step being priced.

PRE-REGISTERED EXPECTATION, written before the run
--------------------------------------------------
w16i's 5-arm average bought +6.494e-6 (se 3.504) over `blend159av_h3`; w16q's bought
+6.4e-6-ish over `blend159av`. The correction is a fixed residual signal, so the honest prior
is that it buys LESS on a stronger base, not more -- 22 new members may already span part of
the direction `c_avg` points in. Registered range: **+2 to +7e-6**, with a POSITIVE point
estimate and a real possibility of the arms coming in flat. A NEGATIVE cross-fitted delta
would be the interesting outcome and would say the new pack has absorbed the correction.

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
from w16i_schemeavg import decile_levels, mask_levels, permuted  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# BASE/TAG are overridable so the SAME code can be pointed at the all-four ("ens4") base of
# the same pack, which is w16q's move one pack later. Defaults reproduce the h3 run exactly.
BASE = os.environ.get("W21A_BASE", "w20_ad187_h3")   # 187-member h3 mix, CV 0.9701008150
TAG = os.environ.get("W21A_TAG", "w21_ad187corr")
REF = "blend159av_h3"        # the base w16i corrected
N_TEST = 296_302
CTRL_SEED = 20260817


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
    ref_auc = fast_auc(y, np.load(os.path.join(SUB, f"oof_{REF}.npy")).astype(np.float64))
    print(f"base {BASE} OOF AUC {base_auc:.10f}   (w16i's base {REF} {ref_auc:.10f}, "
          f"delta {(base_auc-ref_auc)*1e6:+.2f}e-6)", flush=True)

    cell = rule_cells(tr)
    assigns = {
        "glob": np.array(["0"] * n, dtype=object),
        "a_only": np.where(cell == CELL_A, "A", "rest").astype(object),
        "rule": cell,
        "mask": mask_levels(tr),
        "decile": decile_levels(br),
    }
    names = list(assigns)
    for k, a in assigns.items():
        print(f"  scheme {k:<7} {len(set(a))} levels  "
              f"sizes {sorted(np.unique(a, return_counts=True)[1].tolist(), reverse=True)}")

    fold_w, per_fold, oofs = {}, {}, {}
    for name in names:
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
              f"CV {fast_auc(y, z):.10f}", flush=True)

    print()
    ctrl = {}
    for name in names:
        if len(set(assigns[name])) < 2:
            continue
        pa = permuted(assigns[name], rng)
        d = []
        for itr, iva in folds:
            w = ascend(y, br, c, pa, itr)
            d.append(fast_auc(y[iva], apply_w(br, c, pa, w, iva)) - fast_auc(y[iva], br[iva]))
        ctrl[name] = np.array(d)
        print(f"  CTRL permuted {name:<7} xfit {ctrl[name].mean()*1e6:+7.3f}e-6  "
              f"-> real minus control {(per_fold[name].mean()-ctrl[name].mean())*1e6:+7.3f}e-6",
              flush=True)

    print("\n=== leave-one-fold-out over the FIVE schemes (187 base) ===")
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
          f"{held.std(ddof=1)/np.sqrt(5)*1e6:.3f}")
    print(f"  SCHEME-SELECTION OPTIMISM = {(naive.mean()-held.mean())*1e6:+.3f}e-6   "
          f"stability {max(picks.count(p) for p in set(picks))}/5")

    def avg_cv(keys):
        v = np.mean([rankdata(oofs[k]) for k in keys], axis=0)
        d = np.array([fast_auc(y[iva], v[iva]) - fast_auc(y[iva], br[iva]) for _, iva in folds])
        return fast_auc(y, v), d, v

    print("\n=== averaged objects (no selection inside them) ===")
    combos = {
        "3-arm (glob+a_only+rule)": ["glob", "a_only", "rule"],
        "5-arm (all schemes)": names,
        "4-arm (drop decile)": ["glob", "a_only", "rule", "mask"],
        "4-arm (drop mask)": ["glob", "a_only", "rule", "decile"],
    }
    out = {}
    for tag, keys in combos.items():
        cv, d, v = avg_cv(keys)
        se = d.std(ddof=1) / np.sqrt(5)
        print(f"  {tag:<28} CV {cv:.10f}  xfit {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se:+5.2f}  {int((d > 0).sum())}/5")
        out[tag] = dict(cv=float(cv), xfit=float(d.mean()), se=float(se),
                        per_fold=[float(x) for x in d], vec=v, keys=keys)

    SHIP = "5-arm (all schemes)"
    best_tag = max(out, key=lambda t: out[t]["cv"])
    print(f"\n  highest cross-fitted CV: {best_tag}  {out[best_tag]['cv']:.10f}")
    if best_tag != SHIP:
        print(f"  ** NOT SHIPPING THE ARGMAX. ** Shipping the pre-registered {SHIP} "
              f"(CV {out[SHIP]['cv']:.10f}).")

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
    path = os.path.join(SUB, f"{TAG}.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(SUB, f"oof_{TAG}.npy"), out[SHIP]["vec"] / n)

    r = rankdata(strict)
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == f"{TAG}.csv":
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
    print(f"  rank-identical to an existing submission file? {dupes or 'no'}")
    for other in (BASE, "w16i_schemeavg", REF, "w16q_ens4avg"):
        op = os.path.join(SUB, f"{other}.csv")
        if not os.path.exists(op):
            continue
        ov = pd.read_csv(op).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<22} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    W16I_CV = 0.9700556663
    ship_cv = out[SHIP]["cv"]
    print("\n=== deadline-pick arithmetic (decided in the journal, not here) ===")
    print(f"  {TAG:<22} CV {ship_cv:.10f}")
    print(f"  {BASE:<22} CV {base_auc:.10f}   ({(ship_cv-base_auc)*1e6:+.3f}e-6)")
    print(f"  w16i_schemeavg         CV {W16I_CV:.10f}   ({(ship_cv-W16I_CV)*1e6:+.3f}e-6)")

    json.dump(dict(base=BASE, base_auc=float(base_auc), ref_base_auc=float(ref_auc),
                   arms={k: dict(xfit=float(v.mean()), per_fold=[float(x) for x in v],
                                 cv=float(fast_auc(y, oofs[k]))) for k, v in per_fold.items()},
                   ctrl={k: float(v.mean()) for k, v in ctrl.items()},
                   nested_picks=picks, nested_delta=float(held.mean()),
                   naive_arm=naive_arm, naive_delta=float(naive.mean()),
                   scheme_optimism=float(naive.mean() - held.mean()),
                   combos={k: dict(cv=v["cv"], xfit=v["xfit"], se=v["se"],
                                   per_fold=v["per_fold"]) for k, v in out.items()},
                   shipped=SHIP, cv_argmax_combo=best_tag,
                   fold_weights={k: [{str(a): float(b) for a, b in w.items()} for w in v]
                                 for k, v in fold_w.items()},
                   full_weights=full_w, tag=TAG),
              open(os.path.join(HERE, f"w21a_{TAG}.json"), "w"), indent=1)
    print(f"\nwrote experiments/w21a_{TAG}.json")


if __name__ == "__main__":
    main()
