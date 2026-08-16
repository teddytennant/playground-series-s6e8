"""w16d: per-cell MEMBER weights — item 1 of w16a's ranked open list.

WHAT w16a MEASURED AND DID NOT HAVE THE SLOT TO BUILD
-----------------------------------------------------
Its cond-AUC instrument (pooled within-bin Mann-Whitney against the label, bins on the base
score, re-binned inside each segment, 8-seed permuted control) applied to pack MEMBERS:

    member              cell A                cell B   cell BAND   cell D
    xgb_cat_lattice     cond 0.556267 z +6.88   +0.44     +0.71      -1.45
    cat_native          cond 0.539967 z +3.41   -0.73     -4.80      -0.20
    c_avg  (for scale)              z +5.16     +4.42     +3.50      +0.32

`xgb_cat_lattice` carries MORE information the 159-member stack is not using in cell A than
`c_avg` does, and it is null-to-negative everywhere else. The stack weights it GLOBALLY, which
averages a strong cell-A signal against nothing elsewhere. blend153 already showed the
decorrelated pair is worth a sign-flipping +/-1-5e-6 globally; nobody asked whether that null
is a regional cancellation. This run asks.

CONSTRUCTION, AND WHY IT IS NOT JUST "SWAP ONE PATH"
----------------------------------------------------
w16b's correction `c_avg` is a residual standardised to unit sd WITHIN FOLD, which is what
makes its 0..0.02 grid mean "perturbation size in percentile units". `oof_xgb_cat_lattice.npy`
is a raw probability on a completely different scale, so pointing w16b at it directly would
compare a weight grid against the wrong ruler. The matched object is

    c_mem = pct(member) - pct(base),  mean-centred and scaled to unit sd within each fold

which is monotone-identical to a rank blend of base and member and therefore reads as "let the
stack's weight on this one member vary by rule cell". Test side is built the same way from
`test_xgb_cat_lattice.npy` and the base's own test file, standardised on the test side, exactly
as w15f built `c_test_avg`.

TWO ARM SETS
------------
SET 1  base = blend159av_h3 (the zero-parameter stack). The clean form of the question.
SET 2  base = w16b_cellweight, the per-cell c_avg file already sent today. Does the member's
       cell-A signal ADD on top of the correction that already exploits cell A, or is it the
       same signal twice? This is the part that decides whether item 1 is a new finding or a
       re-discovery, and w16a could not know which.

Arms inside each set are w16b's, unchanged: GLOBAL 1 weight, A-ONLY 2 weights, PER-CELL 7
weights, each against a size-matched PERMUTED-membership control, coordinate ascent over the
same 0..0.02 grid in 41 steps, 2 passes, weights searched on four of the frozen SKF5 seed42
folds and scored on the fifth.

SHIPPING RULE, FIXED BEFORE THE NUMBERS WERE SEEN
-------------------------------------------------
Ship the highest cross-fitted arm across both sets, provided it beats its own matched permuted
control by more than +1.0e-6; ties inside 1e-6 go to the arm with fewer fitted parameters, and
SET 1 breaks a remaining tie because it carries fewer parameters overall. If no arm clears its
control by 1e-6, ship SET 2's GLOBAL arm — one extra parameter on top of a file already sent,
the minimal honest variation — and say in the journal that the finding did not replicate.

Nothing here refits a model; every input is a saved vector. Nothing is chosen with reference to
the public leaderboard. This is NOT a deadline pick: it fits parameters above the stack where
blend159av_h3 and blend160origm_h3 fit zero.
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

from common import DATA, OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402

from w16b_cellweight import CELL_A, ascend, apply_w, fast_auc, pct, rule_cells, xfit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "blend159av_h3"
MEMBER = "xgb_cat_lattice"
STACKED_BASE = "w16b_cellweight"
N_TEST = 296_302


def unit_sd_within_fold(v, folds, n):
    """Mean-centre and scale to unit sd inside each fold, as w15f built c_avg."""
    out = np.zeros(n, dtype=np.float64)
    for _, iva in folds:
        x = v[iva]
        out[iva] = (x - x.mean()) / x.std()
    return out


def run_set(label, y, br, c, cell, a_only, one, folds, n, base_auc, store):
    print("\n" + "-" * 78)
    print(f"{label}   base OOF AUC {base_auc:.8f}")
    print("-" * 78)
    d_g, w_g = xfit(y, br, c, one, folds, "GLOBAL   1 weight")
    d_a, w_a = xfit(y, br, c, a_only, folds, "A-ONLY   2 weights")
    d_c, w_c = xfit(y, br, c, cell, folds, "PER-CELL 7 weights")
    rng = np.random.default_rng(4242)
    d_c7, _ = xfit(y, br, c, cell[rng.permutation(n)], folds, "CTRL permuted 7")
    rng2 = np.random.default_rng(909)
    d_c2, _ = xfit(y, br, c, a_only[rng2.permutation(n)], folds, "CTRL permuted 2")
    print(f"  CV(global)   {base_auc + d_g.mean():.8f}")
    print(f"  CV(A-only)   {base_auc + d_a.mean():.8f}   minus its ctrl "
          f"{d_a.mean()-d_c2.mean():+.3e}")
    print(f"  CV(per-cell) {base_auc + d_c.mean():.8f}   minus its ctrl "
          f"{d_c.mean()-d_c7.mean():+.3e}   minus global {d_c.mean()-d_g.mean():+.3e}")
    print("  full-data-equivalent per-fold weights (per-cell arm):")
    for k, w in enumerate(w_c):
        print(f"    fold {k}: " + "  ".join(f"{v[:1]}={w[v]:.4f}" for v in sorted(w)))
    store[label] = {
        k: dict(xfit=float(v.mean()), folds_pos=int((v > 0).sum()),
                per_fold=[float(x) for x in v], cv=float(base_auc + v.mean()))
        for k, v in (("global", d_g), ("a_only", d_a), ("per_cell", d_c),
                     ("ctrl7", d_c7), ("ctrl2", d_c2))
    }
    store[label]["fold_weights"] = dict(
        glob=[{str(a): float(b) for a, b in w.items()} for w in w_g],
        a_only=[{str(a): float(b) for a, b in w.items()} for w in w_a],
        per_cell=[{str(a): float(b) for a, b in w.items()} for w in w_c])
    return dict(global_=(d_g, one, 1, d_g), a_only=(d_a, a_only, 2, d_c2),
                per_cell=(d_c, cell, 7, d_c7))


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    cell = rule_cells(tr)
    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    one = np.zeros(n, dtype=object)

    # ---------------------------------------------------------------- the correction
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    mem = np.load(os.path.join(OOF, f"oof_{MEMBER}.npy")).astype(np.float64)
    br = pct(base_p)
    raw = pct(mem) - br
    c_mem = unit_sd_within_fold(raw, folds, n)
    rho = float(np.corrcoef(rankdata(mem), rankdata(base_p))[0, 1])
    print(f"correction c_mem from {MEMBER}: raw sd {raw.std():.5f}; on the unit-sd version "
          f"a weight of 0.02 is a 0.02-percentile perturbation, which corresponds to an "
          f"effective member blend weight of {0.02/raw.std():.4f}")
    print(f"  spearman(member, base) {rho:.6f}   solo member OOF AUC "
          f"{fast_auc(y, mem):.7f}   base {fast_auc(y, base_p):.7f}")
    print(f"  c_mem sd by cell: " + "  ".join(
        f"{v[:1]} {c_mem[cell == v].std():.3f}" for v in sorted(set(cell))))

    store = {}
    r1 = run_set("SET1 base=blend159av_h3", y, br, c_mem, cell, a_only, one, folds, n,
                 fast_auc(y, br), store)

    # ---------------------------- SET 2: on top of the per-cell c_avg file already sent
    j = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))
    c_avg = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    br2 = br.copy()
    for (_, iva), w in zip(folds, j["fold_weights"]["per_cell"]):
        for lvl, wv in w.items():
            m = iva[cell[iva] == lvl]
            br2[m] = br[m] + float(wv) * c_avg[m]
    br2 = pct(br2)          # back onto the percentile scale the grid is calibrated for
    r2 = run_set("SET2 base=w16b_cellweight (cross-fitted)", y, br2, c_mem, cell, a_only,
                 one, folds, n, fast_auc(y, br2), store)

    # ------------------------------------------------------------------ the shipping rule
    print("\n" + "=" * 78)
    print("SHIPPING RULE (fixed in the docstring before the run)")
    print("=" * 78)
    cands = []
    for setname, r, brs, tag in (("SET1", r1, br, "s1"), ("SET2", r2, br2, "s2")):
        for arm in ("global_", "a_only", "per_cell"):
            d, assign, npar, ctrl = r[arm]
            net = d.mean() - ctrl.mean() if arm != "global_" else d.mean()
            cands.append(dict(set=setname, arm=arm, npar=npar, xfit=float(d.mean()),
                              net=float(net), assign=assign, br=brs, tag=tag))
            print(f"  {setname} {arm:9s} p{npar}  xfit {d.mean():+.3e}  "
                  f"vs own control {net:+.3e}")
    ok = [c for c in cands if c["net"] > 1.0e-6]
    if ok:
        ok.sort(key=lambda c: (-c["xfit"],))
        top = ok[0]
        for c in ok[1:]:
            if top["xfit"] - c["xfit"] < 1e-6 and c["npar"] < top["npar"]:
                top = c
        replicated = True
    else:
        top = [c for c in cands if c["set"] == "SET2" and c["arm"] == "global_"][0]
        replicated = False
        print("  NO ARM CLEARS ITS CONTROL BY 1e-6 -> falling back to SET2 global, "
              "and the finding did not replicate.")
    print(f"\n  chosen: {top['set']} {top['arm']} ({top['npar']} params, "
          f"xfit {top['xfit']:+.3e}, net {top['net']:+.3e})")
    cv = fast_auc(y, top["br"]) + top["xfit"]
    print(f"  cross-fitted CV of the chosen arm: {cv:.8f}")

    w_full = ascend(y, top["br"], c_mem, top["assign"], np.arange(n))
    print(f"  full-data weights: { {str(k): round(v, 4) for k, v in w_full.items()} }")
    if any(abs(v - 0.02) < 1e-9 for v in w_full.values()):
        print("  ** WARNING: a full-data weight pinned at the top of the grid **")

    # ------------------------------------------------------------------- the test file
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    ids = sample["id"].to_numpy()
    assert (te["id"].to_numpy() == ids).all()
    src = BASE if top["set"] == "SET1" else STACKED_BASE
    bp = pd.read_csv(os.path.join(SUB, f"{src}.csv")).set_index("id") \
        .reindex(ids)[TARGET].to_numpy(np.float64)
    assert np.isfinite(bp).all() and len(bp) == N_TEST
    btr = pct(bp)
    memt = np.load(os.path.join(OOF, f"test_{MEMBER}.npy")).astype(np.float64)
    assert memt.shape == (N_TEST,)
    raw_t = pct(memt) - btr
    ct = (raw_t - raw_t.mean()) / raw_t.std()
    print(f"\n  test side: base {src}, raw sd {raw_t.std():.5f} "
          f"(train {raw.std():.5f}, ratio {raw_t.std()/raw.std():.3f})")

    cell_te = rule_cells(te)
    assign_te = (np.zeros(N_TEST, dtype=object) if top["arm"] == "global_"
                 else np.where(cell_te == CELL_A, "A", "rest").astype(object)
                 if top["arm"] == "a_only" else cell_te)
    assert set(assign_te) == set(w_full), (sorted(set(assign_te)), sorted(w_full))
    shares = {str(v): float((assign_te == v).mean()) for v in sorted(set(assign_te))}
    print(f"  test-side segment shares: { {k: round(v, 4) for k, v in shares.items()} }")

    pred = apply_w(btr, ct, assign_te, w_full, np.arange(N_TEST))
    order = np.lexsort((ids, btr, pred))
    strict = np.empty(N_TEST, np.float64)
    strict[order] = (np.arange(N_TEST) + 1) / N_TEST
    sub = pd.DataFrame({"id": ids, TARGET: strict})
    assert sub.shape == (N_TEST, 2)
    assert sub["id"].equals(sample["id"])
    assert np.isfinite(sub[TARGET]).all()
    assert sub[TARGET].nunique() == N_TEST
    path = os.path.join(SUB, "w16d_membercell.csv")
    sub.to_csv(path, index=False)

    # never resend an identical file
    dupes = []
    for f in sorted(os.listdir(SUB)):
        if not f.endswith(".csv") or f == "w16d_membercell.csv":
            continue
        try:
            o = pd.read_csv(os.path.join(SUB, f))
        except Exception:
            continue
        if list(o.columns) != ["id", TARGET] or len(o) != N_TEST:
            continue
        ov = o.set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        if np.array_equal(rankdata(ov), rankdata(strict)):
            dupes.append(f)
    print(f"  rank-identical to an existing submission file? {dupes or 'no'}")
    rho_b = float(np.corrcoef(rankdata(strict), rankdata(bp))[0, 1])
    ndiff = int((rankdata(strict) != rankdata(bp)).sum())
    print(f"  wrote {path}  rows {len(sub):,}  spearman vs its base {rho_b:.7f}  "
          f"rows differing {ndiff:,}")

    json.dump(dict(arms=store, chosen=dict(set=top["set"], arm=top["arm"],
                                           npar=top["npar"], xfit=top["xfit"],
                                           net=top["net"], cv=float(cv),
                                           replicated=replicated,
                                           full_weights={str(k): float(v)
                                                         for k, v in w_full.items()}),
                   member=MEMBER, spearman_member_base=rho, test_shares=shares,
                   dupes=dupes, spearman_vs_base=rho_b, rows_differing=ndiff),
              open(os.path.join(HERE, "w16d_membercell.json"), "w"), indent=1)
    print("  wrote experiments/w16d_membercell.json")


if __name__ == "__main__":
    main()
