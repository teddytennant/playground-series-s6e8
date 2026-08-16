"""w16q: the h3/ens4 separability claim, and the corrected file that has never existed on the
ens4 side.

WHERE THIS COMES FROM
---------------------
Slot 7's assignment was slot 6's second recommendation: sweep the workspace for numbers quoted
as a property of the PACK that were actually measured on a SINGLE member, and for nulls
measured POOLED over a strong categorical. w16o found one of the first kind (`gap_bagging`,
+662.6e-6 on one member, +116.0e-6 on the pack) and it overturned a claim three entries had
built on.

The sweep turns up a second one, in `RESEARCH.md` rather than `JOURNAL.md`, and it is the most
load-bearing single claim in the workspace because BOTH deadline picks sit on one side of it:

    "Standing position: `h3` and `ens4` are not separable. Do not claim either is better,
     and do not resolve it on the public LB."

That position rests on the MIX-GAP ESTIMATOR (a mix's CV->LB gap = the mean of its components'
gaps), which produced `ens4 - h3` = +21e-6 in the gap against `h3 - ens4` = +5e-6 in CV, i.e.
"the bias is 4.2x the margin it would overturn, in the opposite direction". `RESEARCH.md` states
the estimator's validation in full:

    "validated on 150fx where all four components *and* the mix are scored, error -7e-6"

ONE member set. One. And it is used to characterise the whole 160-member pack's transform
decision. That is exactly w16o's pattern, and unlike w16o's case it is checkable for free: the
account has since scored six member sets where the ens4 mix AND the h3 mix are both on the
board. No model fitting, no new file, pure arithmetic on the submission history.

PART 1 -- the audit (arithmetic only, no fitting, no new object)
----------------------------------------------------------------
For every member set with both `<set>.csv` (ens4) and `<set>_h3.csv` scored, print the PAIRED
h3-minus-ens4 difference in CV and in LB. Pairing within member set is the workspace's own
prescription for this comparison (`RESEARCH.md`, "for a two-file comparison on a fixed slice,
that means pairing on the slice") -- the two files share ~156 of 160 members and correlate at
~0.9999, so the shared slice noise cancels and what is left is the transform contrast.

The six member sets are NOT independent of each other and this script must not pretend they
are. They are nested builds sharing almost all members, so a sign test over them is a test of
BUILD reproducibility, not six independent draws from the slice. What replication across them
does establish is that the contrast is not an artefact of one particular member list. It is
reported that way and no p-value is computed from n=6.

PART 2 -- the build
-------------------
Every corrected file this workspace has ever produced -- `w15f_antistudent_avg`,
`w16b_cellweight`, `w16f_armavg`, `w16i_schemeavg`, `w16n_finegrid` -- is built on the base
`blend159av_h3`. The ens4 side of the board is empty above CV 0.9700449, and the deadline pick
is {`w16i_schemeavg.csv`, `blend159av_h3.csv`}: two h3 objects, one of them derived from the
other. Whatever Part 1 says, the workspace has no ens4-side corrected candidate to choose
between, which is a gap in the option set rather than a gap in the analysis.

This builds it: `w16i_schemeavg.py` verbatim with BASE swapped from `blend159av_h3` to
`blend159av`. Same frozen SKF5 seed42 folds, same `c_avg` correction vector, same five
partitions (glob / a_only / rule / mask / decile), same 0..0.02 grid at step 5e-4 -- w16m
measured that grid's ceiling at an exact zero and w16n measured its resolution at +0.046e-6, so
holding it fixed costs nothing and changes one thing only. All five arms are refitted from
scratch, because w16b's stored per-fold weights were fitted against the h3 base and are not
valid here; consequently all five get their own size-matched permuted-membership control rather
than the two w16i needed.

WHAT GETS SHIPPED -- fixed here before the run and not revisited afterwards
--------------------------------------------------------------------------
1.  **The 5-arm average on the ens4 base, unconditionally, whatever its CV.** Same
    pre-registration w16i used, for the same reason: sub-combinations are printed because their
    spread is informative and the argmax over them must not be shipped. If a CV regression comes
    out, it goes out anyway -- naming the file before the numbers exist is the only defence
    against the argmax bug that cost w16c +1.778e-6 and w16i +1.546e-6.
2.  **Fallback, and only for this reason:** if the file comes out rank-identical to one this
    account has already sent, ship `submissions/w14a_repro159av_h3.csv` instead (CV 0.9700472,
    verified unsent). Rank-identical resubmission is the one thing the run-context file forbids
    outright, and it is a mechanical check, not a judgement.
3.  **Deadline picks.** This run moves `WANTED` in `check_selection.py` if and ONLY if the
    ens4-base 5-arm average's cross-fitted CV exceeds `w16i_schemeavg`'s **0.97005567**, in
    which case it replaces `w16i_schemeavg` in the FIRST slot -- the corrected slot -- on CV
    alone.

    It will NOT touch the second slot under any outcome. w16i §4 fixed that slot to a
    ZERO-PARAMETER file deliberately: `w16b`/`w16f`/`w16i` are the same base perturbed by the
    same `c_avg` correction, so if that correction reverses on the private rows they lose
    together, and the hedge costs only 0.77e-6 of E[max]. A 23-parameter ens4-base file built
    on the same `c_avg` cannot serve that role no matter what its CV is.

    Part 1 is an LB measurement and an LB measurement does not move a deadline pick here,
    however clean it is; that is the Rogii rule and the brief restates it. If Part 1 says the
    zero-parameter hedge would be better served by `blend159av` (ens4) than by `blend159av_h3`,
    that is RECORDED with its CV price and left for the human, not acted on by me.
4.  Part 1 cannot produce a file and cannot move anything on its own. It is a read of the
    existing submission history.

Nothing in Part 2 is chosen with reference to the public leaderboard. The BASE swap is motivated
by the empty half of the option set, and Part 1 is reported whichever way it comes out.
"""
from __future__ import annotations

import csv
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
BASE = "blend159av"          # the ens4 mix. w16i used blend159av_h3.
H3_BASE = "blend159av_h3"    # for the CV comparison in ship rule 3
TAG = "w16q_ens4avg"
N_TEST = 296_302
CTRL_SEED = 20260816
SUBS_CSV = os.environ.get("W16Q_SUBS", "")


# --------------------------------------------------------------------------- PART 1
def part1(y):
    """Paired h3-minus-ens4 on every member set where BOTH mixes have a public score."""
    print("=" * 78)
    print("PART 1  h3 vs ens4, paired within member set, CV recomputed and LB from the API")
    print("=" * 78)
    if not SUBS_CSV or not os.path.exists(SUBS_CSV):
        print("  no submission csv supplied via W16Q_SUBS; skipping")
        return {}
    lb = {}
    for r in csv.DictReader(open(SUBS_CSV)):
        if r.get("publicScore") and r["fileName"].endswith(".csv"):
            lb[r["fileName"][:-4]] = float(r["publicScore"])

    def cv_of(name):
        p = os.path.join(SUB, f"oof_{name}.npy")
        return fast_auc(y, np.load(p).astype(np.float64)) if os.path.exists(p) else None

    sets = ["blend150fx", "blend150sx", "blend153", "blend156", "blend158", "blend159",
            "blend159av", "blend160orig", "blend160origm"]
    rows = []
    print(f"  {'member set':16s} {'ens4 CV':>11s} {'ens4 LB':>9s} {'h3 CV':>11s} {'h3 LB':>9s}"
          f" {'dCV(h3-e)':>10s} {'dLB(h3-e)':>10s} {'dGAP':>9s}")
    for s in sets:
        h = s + "_h3"
        if s not in lb or h not in lb:
            continue
        ce, ch = cv_of(s), cv_of(h)
        if ce is None or ch is None:
            continue
        dcv, dlb = (ch - ce) * 1e6, (lb[h] - lb[s]) * 1e6
        rows.append(dict(set=s, cv_e=ce, cv_h=ch, lb_e=lb[s], lb_h=lb[h],
                         dcv=dcv, dlb=dlb, dgap=dlb - dcv))
        print(f"  {s:16s} {ce:11.7f} {lb[s]:9.5f} {ch:11.7f} {lb[h]:9.5f}"
              f" {dcv:+10.2f} {dlb:+10.2f} {dlb-dcv:+9.2f}")
    if not rows:
        print("  no paired member sets found")
        return {}
    dcv = np.array([r["dcv"] for r in rows])
    dlb = np.array([r["dlb"] for r in rows])
    dgap = dlb - dcv
    n_pos_cv = int((dcv > 0).sum())
    n_neg_lb = int((dlb < 0).sum())
    print(f"\n  n paired member sets = {len(rows)}")
    print(f"  h3 - ens4 in CV   mean {dcv.mean():+7.2f}e-6   range [{dcv.min():+.2f}, "
          f"{dcv.max():+.2f}]   h3 higher in {n_pos_cv}/{len(rows)}")
    print(f"  h3 - ens4 in LB   mean {dlb.mean():+7.2f}e-6   range [{dlb.min():+.2f}, "
          f"{dlb.max():+.2f}]   h3 LOWER  in {n_neg_lb}/{len(rows)}")
    print(f"  h3 - ens4 in the CV->LB GAP  mean {dgap.mean():+7.2f}e-6  "
          f"range [{dgap.min():+.2f}, {dgap.max():+.2f}]")
    print("\n  RESEARCH.md's mix-gap estimator, validated on ONE member set (150fx, error")
    print("  -7e-6), predicted ens4 - h3 = +21e-6 in the gap, i.e. h3 - ens4 = -21e-6.")
    print(f"  MEASURED DIRECTLY on {len(rows)} member sets: {dgap.mean():+.2f}e-6.")
    print("  CAVEAT, stated before the numbers: these member sets are NESTED builds sharing")
    print("  almost all 160 members, so this is one highly-replicated contrast and NOT n")
    print("  independent draws from the public slice. No p-value is computed from it.")
    return dict(rows=rows, dcv_mean=float(dcv.mean()), dlb_mean=float(dlb.mean()),
                dgap_mean=float(dgap.mean()), n=len(rows),
                cv_h3_higher=n_pos_cv, lb_h3_lower=n_neg_lb)


# --------------------------------------------------------------------------- PART 2
def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    rng = np.random.default_rng(CTRL_SEED)

    p1 = part1(y)

    print("\n" + "=" * 78)
    print(f"PART 2  the 5-arm scheme average on the ENS4 base ({BASE})")
    print("=" * 78)
    base_p = np.load(os.path.join(SUB, f"oof_{BASE}.npy")).astype(np.float64)
    br = pct(base_p)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)
    base_auc = fast_auc(y, br)
    h3_auc = fast_auc(y, np.load(os.path.join(SUB, f"oof_{H3_BASE}.npy")).astype(np.float64))
    print(f"base {BASE} OOF AUC {base_auc:.8f}   (h3 base {H3_BASE} {h3_auc:.8f})", flush=True)

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
              f"CV {fast_auc(y, z):.8f}", flush=True)

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

    print("\n=== leave-one-fold-out over the FIVE schemes (ens4 base) ===")
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
        print(f"  {tag:<28} CV {cv:.8f}  xfit {d.mean()*1e6:+7.3f}e-6  se {se*1e6:5.3f}  "
              f"t {d.mean()/se:+5.2f}  {int((d > 0).sum())}/5")
        out[tag] = dict(cv=float(cv), xfit=float(d.mean()), se=float(se),
                        per_fold=[float(x) for x in d], vec=v, keys=keys)

    SHIP = "5-arm (all schemes)"
    best_tag = max(out, key=lambda t: out[t]["cv"])
    print(f"\n  highest cross-fitted CV: {best_tag}  {out[best_tag]['cv']:.8f}")
    if best_tag != SHIP:
        print(f"  ** NOT SHIPPING THE ARGMAX. ** Shipping the pre-registered {SHIP} "
              f"(CV {out[SHIP]['cv']:.8f}).")

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
    for other in ("w16i_schemeavg", "w16n_finegrid", "w15f_antistudent_avg", BASE, H3_BASE):
        op = os.path.join(SUB, f"{other}.csv")
        if not os.path.exists(op):
            continue
        ov = pd.read_csv(op).set_index("id").reindex(ids)[TARGET].to_numpy(np.float64)
        print(f"    spearman vs {other:<22} {np.corrcoef(r, rankdata(ov))[0,1]:.7f}  "
              f"rows differing {int((rankdata(ov) != r).sum()):,}")

    # ------------------------------------------------------------------ ship rule 3
    ship_cv = out[SHIP]["cv"]
    W16I_CV = 0.97005567
    move = bool(ship_cv > W16I_CV)
    print("\n=== PRE-REGISTERED DEADLINE-PICK RULE ===")
    print(f"  replace w16i_schemeavg in slot 1 only if ens4-base 5-arm CV {ship_cv:.8f} > "
          f"w16i_schemeavg CV {W16I_CV:.8f}  ->  {move}")
    print(f"  slot 2 (the zero-parameter hedge, {H3_BASE}) is not touched under any outcome.")
    print(f"  for the record, zero-parameter hedge CV price of the ens4 sibling: "
          f"{H3_BASE} {h3_auc:.8f} vs {BASE} {base_auc:.8f} = "
          f"{(base_auc-h3_auc)*1e6:+.2f}e-6")
    print("  (Part 1 is an LB measurement and does not move a deadline pick here.)")

    json.dump(dict(part1=p1, base=BASE, base_auc=float(base_auc), h3_base_auc=float(h3_auc),
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
                   full_weights=full_w, dupes=dupes, move_wanted=move),
              open(os.path.join(HERE, "w16q_ens4base.json"), "w"), indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
