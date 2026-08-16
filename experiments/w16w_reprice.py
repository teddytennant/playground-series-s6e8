"""w16w: reprice the un-made click for the FOURTH time in one day. Slot 10 caused it too.

`w16e_aonly` returned 0.97108, so auto-slot 1 is now a THREE-way tie:

    auto-slot 1: 0.97108, 3-way - w16e_aonly, w16q_ens4avg, w16t_cellens4
    auto-slot 2: 0.97107, 5-way - w15f_antistudent_avg, w16b_cellweight, w16f_armavg,
                                  w16i_schemeavg, w16n_finegrid

⚠ THIS RUN EXISTS BECAUSE SLOT 10's OWN PRE-REGISTRATION WAS WRONG. w16v_prereg_lb.txt
argued the send "provably cannot make the click more expensive", because an h3-side file
could not reach 0.97108 (it would need CV >= ~0.9700607 against a best h3 CV of 0.9700557).
That argument rested entirely on the h3-corrected CV->LB ladder, and the ladder is what the
result falsified. The file reached 0.97108 at CV 0.9700542. So the send DID move the tier
structure, in the one direction slot 10 told itself it could not, and the honest thing is to
re-derive the price rather than leave the stale w16u figure standing.

At limit 2 the auto-pick was DETERMINED under w16u's 2-way tie. With three files tied it is
ambiguous again and the answer is a RANGE over the three unordered pairs, exactly the shape
w16s had and w16u thought it had removed for good.

The one thing that improves: `w16e_aonly` is H3-side, while `w16q_ens4avg` and `w16t_cellens4`
are both ens4-side and share the same `c_avg`. So two of the three limit-2 branches now pair
an h3 file with an ens4 file instead of two correlated ens4 files. That is a partial hedge
Kaggle might hand us by accident. It is still not the hedge `WANTED` buys, which is a
ZERO-PARAMETER second file, and this instrument structurally cannot price that (w16s: it only
ever draws worlds where the CV ordering is right).

Same instrument throughout: 500 reps, seed 1616, f 0.20, every file scored on the SAME
simulated slice each rep so every +/- is a paired standard error. Gated on reproducing w16s's
two pair readings AND w16u's limit-2 figure before anything below is believed.

PRE-REGISTERED: this run creates no submission (the day's quota is spent, 0 remaining) and it
CANNOT move `WANTED`. It reports a price for a human action. Nothing here is a selector.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, get_folds, load_raw  # noqa: E402
from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_TEST = 296_302
REPS, SEED, F = 500, 1616, 0.20
WANTED = ("w16i_schemeavg", "blend159av_h3")
AUTO1 = ("w16e_aonly", "w16q_ens4avg", "w16t_cellens4")   # the 0.97108 tie, live
SIDE = {"w16e_aonly": "h3", "w16q_ens4avg": "ens4", "w16t_cellens4": "ens4"}

GATE_PAIRS = {("blend159av_h3", "blend159av"): (4.1900467953939204e-06, 0.912),
              ("w16i_schemeavg", "w16q_ens4avg"): (4.051418625711678e-06, 0.908)}


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    get_folds(y)
    names = ["blend159av_h3", "blend159av", "w16i_schemeavg", "w16q_ens4avg",
             "w16t_cellens4", "w16e_aonly"]
    vecs = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, v) for k, v in vecs.items()}
    for k in names:
        print(f"  {k:20s} CV {cv[k]:.10f}")

    pk = {k: prep(vecs[k], y) for k in names}
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    A = np.zeros((REPS, len(names)))
    for i in range(REPS):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[n_pub:]] = True
        for k, nm in enumerate(names):
            A[i, k] = subset_auc(pk[nm], m)
    col = {nm: k for k, nm in enumerate(names)}

    print("\nGATE against w16s/w16u (same seed, same protocol, one extra column)")
    ok = True
    for (a, b), (sd, pp) in GATE_PAIRS.items():
        d = A[:, col[a]] - A[:, col[b]]
        dr = d.mean() - sd
        ok &= abs(dr) < 1e-12
        print(f"  {a:18s} - {b:18s} sim d {d.mean()*1e6:+6.2f}e-6  drift {dr*1e12:+.3f}e-12"
              f"   P {(d > 0).mean():.3f} (w16s {pp:.3f})")
    cur = np.maximum(A[:, col[WANTED[0]]], A[:, col[WANTED[1]]])
    ju = json.load(open(os.path.join(HERE, "w16u_autoprice.json")))
    auto2 = np.maximum(A[:, col["w16q_ens4avg"]], A[:, col["w16t_cellens4"]])
    dr2 = (auto2 - cur).mean() - ju["d"]
    ok &= abs(dr2) < 1e-12
    print(f"  w16u limit-2 (the OLD 2-way tie)  d {(auto2-cur).mean()*1e6:+6.3f}e-6  "
          f"drift {dr2*1e12:+.3f}e-12")
    print(f"  GATE {'PASSED' if ok else '*** FAILED ***'}")
    if not ok:
        print("  downstream numbers INVALID; not writing json")
        return

    out = {"cv": {k: float(v) for k, v in cv.items()}, "gate_passed": True,
           "e_wanted": float(cur.mean()), "limit2": {}, "limit1": {}}

    print(f"\n  WANTED  {' + '.join(WANTED)}   E[max] {cur.mean():.8f}")
    print("\n=== limit 2: auto takes TWO of the three 0.97108 files, tiebreak undocumented ===")
    rows = []
    for a, b in itertools.combinations(AUTO1, 2):
        e = np.maximum(A[:, col[a]], A[:, col[b]])
        d = e - cur
        se = d.std(ddof=1) / np.sqrt(REPS)
        sides = f"{SIDE[a]}+{SIDE[b]}"
        rows.append((-d.mean(), a, b, sides, e.mean(), se, (d > 0).mean()))
        out["limit2"][f"{a}+{b}"] = dict(sides=sides, e_max=float(e.mean()),
                                         d=float(d.mean()), se=float(se),
                                         p_auto_better=float((d > 0).mean()),
                                         cost_of_not_clicking=float(-d.mean()))
    for cost, a, b, sides, em, se, p in sorted(rows):
        print(f"  {a:14s} + {b:16s} [{sides:9s}]  E[max] {em:.8f}  "
              f"cost of NOT clicking {cost*1e6:+6.3f}e-6 +/-{se*1e6:.3f}  P(auto better) {p:.3f}")
    lo, hi = min(r[0] for r in rows), max(r[0] for r in rows)
    print(f"  -> RANGE over the undocumented tiebreak: {lo*1e6:+.3f}e-6 to {hi*1e6:+.3f}e-6")
    print(f"     (w16u's determinate figure on the 2-way tie was +3.326e-6; that branch is "
          f"still one of the three)")
    out["limit2_range"] = [float(lo), float(hi)]

    print("\n=== limit 1: auto takes ONE of the three ===")
    for x in AUTO1:
        d1 = A[:, col[x]] - cur
        se = d1.std(ddof=1) / np.sqrt(REPS)
        print(f"  auto = {x:16s} [{SIDE[x]:4s}]  cost of not clicking "
              f"{-d1.mean()*1e6:+6.3f}e-6 +/-{se*1e6:.3f}  P(auto better) {(d1 > 0).mean():.3f}")
        out["limit1"][x] = dict(side=SIDE[x], cost_of_not_clicking=float(-d1.mean()),
                                se=float(se), p_auto_better=float((d1 > 0).mean()))
    l1 = [v["cost_of_not_clicking"] for v in out["limit1"].values()]
    print(f"  -> RANGE {min(l1)*1e6:+.3f}e-6 to {max(l1)*1e6:+.3f}e-6")

    print("\n  ⚠ READ THIS BEFORE QUOTING ANY NUMBER ABOVE.")
    print("  Two of the three limit-2 branches now pair an h3-side file with an ens4-side one")
    print("  instead of two correlated ens4 files, which is a partial hedge Kaggle would be")
    print("  handing us by accident. It is NOT the hedge WANTED buys: WANTED's second slot is")
    print("  a ZERO-PARAMETER file, insurance against the whole fitted-correction family")
    print("  failing, and all three auto-slot-1 files carry the same c_avg correction. This")
    print("  instrument draws only worlds in which the CV ordering is RIGHT, so it cannot see")
    print("  that failure mode at all. Every figure above is a LOWER bound on the click.")
    print("  The click has now been repriced FOUR times in one Kaggle day. Re-run")
    print("  check_selection.py before quoting it a fifth.")

    json.dump(out, open(os.path.join(HERE, "w16w_reprice.json"), "w"), indent=1)
    print("\nwrote experiments/w16w_reprice.json")


if __name__ == "__main__":
    main()
