"""w16u: reprice the un-made human click, because w16t changed the tier structure.

w16s §5 priced the click on tiers where auto-slot 1 was a ONE-way tie (`w16q_ens4avg`) and
auto-slot 2 was a five-way tie, so the cost depended on which tier-2 file Kaggle's undocumented
tiebreak happened to take, and it came out between -0.24e-6 and +2.14e-6 at limit 2.

`w16t_cellens4` scored 0.97108 and auto-slot 1 is now a TWO-way tie:

    auto-slot 1: 0.97108, 2-way - w16q_ens4avg, w16t_cellens4
    auto-slot 2: 0.97107, 5-way - w15f_antistudent_avg, w16b_cellweight, w16f_armavg,
                                  w16i_schemeavg, w16n_finegrid

That changes the shape of the question, not just the number. At limit 2 the auto-pick is now
DETERMINED - it is exactly those two files - so the tiebreak ambiguity that produced w16s's
range disappears. And both of them are ens4-side corrected files built on the same `c_avg`,
which is the pairing `WANTED` was deliberately constructed to avoid: w16i §4 spends the second
slot on a ZERO-PARAMETER file as insurance against the correction family failing, and w16s
settled that h3 beats ens4 on CV at P 0.91 on a single private-sized draw.

Same instrument as w16k/w16s: 500 reps, seed 1616, f 0.20, every file scored on the SAME
simulated slice each rep so every +/- is a paired standard error. Gated on reproducing w16s's
p0 and p23 pair readings exactly before anything else is believed.

This run creates no file and moves no pick. `WANTED` is untouched.
"""
from __future__ import annotations

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
AUTO2 = ("w16q_ens4avg", "w16t_cellens4")
GATE = {("blend159av_h3", "blend159av"): (4.1900467953939204e-06, 0.912),
        ("w16i_schemeavg", "w16q_ens4avg"): (4.051418625711678e-06, 0.908)}


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    get_folds(y)
    names = ["blend159av_h3", "blend159av", "w16i_schemeavg", "w16q_ens4avg", "w16t_cellens4"]
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

    print("\ngate against w16s (same seed, same protocol):")
    ok = True
    for (a, b), (sd, pp) in GATE.items():
        d = A[:, names.index(a)] - A[:, names.index(b)]
        dr = d.mean() - sd
        ok &= abs(dr) < 1e-12
        print(f"  {a:18s} - {b:18s} sim d {d.mean()*1e6:+6.2f}e-6 (w16s {sd*1e6:+6.2f})  "
              f"drift {dr*1e12:+.3f}e-12   P {(d > 0).mean():.3f} (w16s {pp:.3f})")
    print(f"  gate {'PASSED' if ok else 'FAILED'}")

    cur = np.maximum(A[:, names.index(WANTED[0])], A[:, names.index(WANTED[1])])
    auto = np.maximum(A[:, names.index(AUTO2[0])], A[:, names.index(AUTO2[1])])
    d = auto - cur
    se = d.std(ddof=1) / np.sqrt(REPS)
    print("\n=== limit 2: the auto-pick is now DETERMINED (auto-slot 1 is a 2-way tie) ===")
    print(f"  WANTED  {' + '.join(WANTED)}   E[max] {cur.mean():.8f}")
    print(f"  AUTO    {' + '.join(AUTO2)}   E[max] {auto.mean():.8f}")
    print(f"  auto - WANTED {d.mean()*1e6:+.3f}e-6 +/-{se*1e6:.3f}   "
          f"P(auto better on one draw) {(d > 0).mean():.3f}")
    print(f"  -> cost of NOT clicking, limit 2: {-d.mean()*1e6:+.3f}e-6 "
          f"(positive = clicking wins)")

    print("\n  for comparison, w16s's limit-2 range on the OLD tiers: -0.24e-6 to +2.14e-6,")
    print("  and it was a RANGE because the tier-2 tiebreak was undocumented. It is a point")
    print("  now: at limit 2 the auto-pick is exactly the two 0.97108 files.")

    out = {"cv": {k: float(v) for k, v in cv.items()},
           "gate_passed": bool(ok),
           "e_wanted": float(cur.mean()), "e_auto2": float(auto.mean()),
           "d": float(d.mean()), "se": float(se), "p_auto_better": float((d > 0).mean()),
           "cost_of_not_clicking_limit2": float(-d.mean())}

    print("\n=== limit 1: auto takes one of the two 0.97108 files, tiebreak undocumented ===")
    for x in AUTO2:
        d1 = A[:, names.index(x)] - cur
        print(f"  auto = {x:16s}  d {d1.mean()*1e6:+6.2f}e-6 "
              f"+/-{d1.std(ddof=1)/np.sqrt(REPS)*1e6:4.2f}  P(auto better) {(d1 > 0).mean():.3f}"
              f"  -> cost of not clicking {-d1.mean()*1e6:+.2f}e-6")
        out[f"limit1_{x}"] = float(-(A[:, names.index(x)] - cur).mean())

    print("\n  ⚠ Both files in auto-slot 1 are ens4-side AND built on the same c_avg. WANTED")
    print("  pairs a corrected file with a ZERO-PARAMETER one precisely so they cannot fail")
    print("  together, and w16s showed this instrument structurally cannot price that hedge:")
    print("  it only ever draws worlds in which the CV ordering is right. The numbers above")
    print("  are therefore a LOWER bound on what the click is worth.")

    json.dump(out, open(os.path.join(HERE, "w16u_autoprice.json"), "w"), indent=1)
    print("\nwrote experiments/w16u_autoprice.json")


if __name__ == "__main__":
    main()
