"""w16v: the LAST single-draw control in the correction family, and the 2-vs-7 question.

WHY THIS RUN EXISTS
-------------------
w16t (slot 9) caught both published "real minus control" margins for the SEVEN-weight
per-cell arm resting on ONE permuted-membership draw each, and re-measured them with
seven draws per base. It did not touch the TWO-weight `a_only` arm, whose controls are
still one draw apiece and are quoted in exactly the same way:

    w16b (h3 base)    real +5.031e-6   ctrl2 +3.078e-6 (rng 909)        margin +1.953e-6   1 draw
    w16q (ens4 base)  real +4.987e-6   ctrl a_only +2.687e-6 (rng 2026..) margin +2.300e-6   1 draw

Slot 10 is shipping the `a_only` arm as a file (`w16e_aonly.csv`), so its control is
now load-bearing rather than incidental. Seven draws per base, same protocol.

THE PAIRED DESIGN, which is the part worth more than the correction
-------------------------------------------------------------------
`w16i_schemeavg.permuted()` is the SCATTER form: `out[rng.permutation(n)] = assign`.
So for a fixed seed the permutation array is identical whatever `assign` is, and row
`perm[i]` receives `assign[i]`. Feeding it `a_only` instead of `cell` therefore yields
EXACTLY the A/rest collapse of the same 7-level control draw -- the 2-level control is
a coarsening of the 7-level control, not an independent shuffle.

w16t drew its 7-level controls at seeds 601..606. Reusing those seeds here makes the
2-level and 7-level controls PAIRED on identical shuffles, which removes the draw noise
from the comparison and lets one number be measured that nothing in this workspace has
measured: the cost of splitting a RANDOM 2-way into a RANDOM 7-way, on matched draws.

That prices the decision this workspace actually made. w16b, w16q and w16t all shipped
the 7-parameter per-cell arm over the 2-parameter a_only arm on a raw gap of +1.164e-6
(h3) and +1.412e-6 (ens4). If splitting a random 2-way into a random 7-way buys about
that much for free, the extra five parameters bought nothing and the raw gap is selection
noise. If it buys less, they are real. Neither entry asked.

PRE-REGISTERED, before the script was run once:
  * This run CANNOT move `WANTED` and cannot change what slot 10 sends. `w16e_aonly.csv`
    is already built and its send is pre-registered unconditionally in w16v_prereg_lb.txt.
    This is a measurement, not a selector.
  * Gates first. If either published single draw fails to reproduce exactly, everything
    downstream is invalid and the run says so.
  * Report the margin with its standard error and state plainly whether it is significant.
    w16t's per-cell margins came out at t ~ 1.8, i.e. NOT individually significant, and
    this run is not expected to do better -- the error is dominated by the real arm's
    fold-to-fold spread, which no amount of control draws can shrink.
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
from w16b_cellweight import CELL_A, apply_w, ascend, fast_auc, pct, rule_cells  # noqa: E402
from w16i_schemeavg import permuted  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ENS4 = "blend159av"
H3 = "blend159av_h3"
CTRL_SEEDS = (601, 602, 603, 604, 605, 606)   # identical to w16t's, on purpose

# published quantities this run gates on, all read off the stored json
W16B_CTRL2_RNG = 909
W16Q_CTRL_SEED = 20260816

# w16t's seven-draw 7-level control means, for the paired 2-vs-7 comparison
W16T_CTRL7_MEAN = {ENS4: 1.929e-06, H3: 1.545e-06}


def arm_xfit(y, br, c, assign, folds):
    """Cross-fitted delta over the base. The shared protocol, unchanged."""
    d = []
    for itr, iva in folds:
        w = ascend(y, br, c, assign, itr)
        d.append(fast_auc(y[iva], apply_w(br, c, assign, w, iva))
                 - fast_auc(y[iva], br[iva]))
    return np.array(d)


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    folds = get_folds(y)
    cell = rule_cells(tr)
    a_only = np.where(cell == CELL_A, "A", "rest").astype(object)
    c = np.load(os.path.join(HERE, "w15f_c_avg.npy")).astype(np.float64)

    jb = json.load(open(os.path.join(HERE, "w16b_cellweight.json")))
    jq = json.load(open(os.path.join(HERE, "w16q_ens4base.json")))
    pub = {
        (H3, "real"): float(np.mean(jb["arms"]["a_only"]["per_fold"])),
        (H3, "ctrl"): float(np.mean(jb["arms"]["ctrl2"]["per_fold"])),
        (H3, "real7"): float(np.mean(jb["arms"]["per_cell"]["per_fold"])),
        (ENS4, "real"): float(jq["arms"]["a_only"]["xfit"]),
        (ENS4, "ctrl"): float(jq["ctrl"]["a_only"]),
        (ENS4, "real7"): float(jq["naive_delta"]),
    }
    print(f"n {n:,}   folds SKF5 seed42 (frozen)   grid/protocol unchanged")
    print(f"published a_only controls being re-measured: "
          f"h3 {pub[(H3,'ctrl')]*1e6:+.4f}e-6, ens4 {pub[(ENS4,'ctrl')]*1e6:+.4f}e-6\n")

    base = {}
    for b in (ENS4, H3):
        p = np.load(os.path.join(SUB, f"oof_{b}.npy")).astype(np.float64)
        base[b] = pct(p)
        print(f"base {b:16s} OOF AUC {fast_auc(y, base[b]):.8f}")

    out = {"gates": {}, "real": {}, "ctrl2": {}, "ctrl7_w16t": W16T_CTRL7_MEAN,
           "margin": {}, "paired_2v7": {}}

    # ---------------- GATES: reproduce both published single draws exactly ------------
    print("\nGATE  re-derive the published single-draw quantities in this process")
    # w16b's ctrl2: GATHER form, rng(909), on the h3 base
    pb2 = a_only[np.random.default_rng(W16B_CTRL2_RNG).permutation(n)]
    g_h3 = arm_xfit(y, base[H3], c, pb2, folds).mean()
    # w16q's a_only ctrl: SCATTER form, first permuted() call off rng(20260816), ens4 base
    pq2 = permuted(a_only, np.random.default_rng(W16Q_CTRL_SEED))
    g_e4 = arm_xfit(y, base[ENS4], c, pq2, folds).mean()
    # the real arms, both bases
    r_h3 = arm_xfit(y, base[H3], c, a_only, folds)
    r_e4 = arm_xfit(y, base[ENS4], c, a_only, folds)

    gates = {
        "w16b ctrl2 (h3, rng 909)": (g_h3, pub[(H3, "ctrl")]),
        "w16q ctrl a_only (ens4)": (g_e4, pub[(ENS4, "ctrl")]),
        "w16b real a_only (h3)": (r_h3.mean(), pub[(H3, "real")]),
        "w16q real a_only (ens4)": (r_e4.mean(), pub[(ENS4, "real")]),
    }
    ok = True
    for name, (got, want) in gates.items():
        if want is None:
            print(f"  {name:34s} {got*1e6:+9.4f}e-6   (no stored value to gate on)")
            continue
        drift = got - want
        ok &= abs(drift) < 1e-15
        out["gates"][name] = float(drift)
        print(f"  {name:34s} {got*1e6:+9.4f}e-6   drift {drift*1e12:+.3f}e-12")
    print(f"  ALL GATES {'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        print("  downstream numbers are INVALID; not writing json")
        return

    out["real"] = {ENS4: r_e4.tolist(), H3: r_h3.tolist()}

    # ---------------- the control ENSEMBLE, 7 draws per base, paired to w16t ----------
    print("\nPART 1  a_only control ENSEMBLE: 7 size-matched 2-level permuted draws per base")
    print("        (draw 1 is the published one, reproduced above; 601-606 are w16t's seeds)")
    ctrl = {ENS4: [float(g_e4)], H3: [float(g_h3)]}
    src = {ENS4: [f"w16q draw (rng {W16Q_CTRL_SEED})"], H3: [f"w16b draw (rng {W16B_CTRL2_RNG})"]}
    paired = {ENS4: {}, H3: {}}
    for s in CTRL_SEEDS:
        pa = permuted(a_only, np.random.default_rng(s))
        for b in (ENS4, H3):
            d = arm_xfit(y, base[b], c, pa, folds).mean()
            ctrl[b].append(float(d))
            src[b].append(f"rng {s}")
            paired[b][s] = float(d)
            print(f"  ctrl2 {b:16s} seed {s}  {d*1e6:+8.4f}e-6", flush=True)

    print()
    for b, real in ((ENS4, r_e4), (H3, r_h3)):
        a = np.array(ctrl[b])
        se_real = real.std(ddof=1) / np.sqrt(len(real))
        se_ctrl = a.std(ddof=1) / np.sqrt(len(a))
        margin = real.mean() - a.mean()
        se = float(np.hypot(se_real, se_ctrl))
        pubm = real.mean() - pub[(b, "ctrl")]
        print(f"  {b:16s} real {real.mean()*1e6:+7.3f}e-6 (se {se_real*1e6:.3f})   "
              f"ctrl mean {a.mean()*1e6:+7.3f} sd {a.std(ddof=1)*1e6:.3f} "
              f"range [{a.min()*1e6:+.3f}, {a.max()*1e6:+.3f}]")
        print(f"                   REAL - CONTROL {margin*1e6:+7.3f}e-6 +/-{se*1e6:.3f}  "
              f"t {margin/se:+5.2f}   (published single-draw figure {pubm*1e6:+7.3f}e-6)")
        rank = int((a > pub[(b, "ctrl")] - 1e-18).sum())
        print(f"                   the published draw ranks {rank}/{len(a)} from the top "
              f"of its own control distribution")
        out["ctrl2"][b] = dict(draws=ctrl[b], src=src[b], mean=float(a.mean()),
                               sd=float(a.std(ddof=1)))
        out["margin"][b] = dict(real=float(real.mean()), se_real=float(se_real),
                                margin=float(margin), se=se, t=float(margin / se),
                                published_single_draw=float(pubm))

    # ---------------- PART 2  the 2-vs-7 question, on matched shuffles ----------------
    print("\nPART 2  cost of splitting a RANDOM 2-way into a RANDOM 7-way, PAIRED per seed")
    print("        w16t's 7-level controls used the same scatter form at seeds 601-606, so")
    print("        its draw at seed s is the exact refinement of this run's draw at seed s.")
    print("        Draw 1 is NOT paired on either base (different rng / different call")
    print("        position), so it is excluded and only the 6 matched seeds are used.")
    jt = json.load(open(os.path.join(HERE, "w16t_cellens4.json")))["control"]
    for b in (ENS4, H3):
        src7 = jt[b]["src"]
        d7 = {s: v for s, v in zip(src7, jt[b]["draws"]) if s.startswith("rng ")}
        pairs = [(s, paired[b][s], d7[f"rng {s}"]) for s in CTRL_SEEDS]
        diffs = np.array([p7 - p2 for _, p2, p7 in pairs])
        toll = diffs.mean()
        se_toll = diffs.std(ddof=1) / np.sqrt(len(diffs))
        real_gap = pub[(b, "real7")] - out["margin"][b]["real"]
        print(f"\n  {b}")
        for s, p2, p7 in pairs:
            print(f"    seed {s}  2-way {p2*1e6:+7.3f}e-6   7-way {p7*1e6:+7.3f}e-6   "
                  f"refine {(p7-p2)*1e6:+7.3f}e-6")
        print(f"    PAIRED random 2->7 toll {toll*1e6:+7.3f}e-6 +/-{se_toll*1e6:.3f}  "
              f"t {toll/se_toll:+5.2f}   ({int((diffs<0).sum())}/6 draws lose from refining)")
        print(f"    the REAL 2->7 gap (per_cell - a_only) is {real_gap*1e6:+7.3f}e-6")
        print(f"    real refinement clears its random toll by "
              f"{(real_gap - toll)*1e6:+7.3f}e-6")
        out["paired_2v7"][b] = dict(seeds=list(CTRL_SEEDS),
                                    ctrl2=[p2 for _, p2, _ in pairs],
                                    ctrl7=[p7 for _, _, p7 in pairs],
                                    random_toll=float(toll), se=float(se_toll),
                                    real_gap=float(real_gap),
                                    clears_by=float(real_gap - toll))

    json.dump(out, open(os.path.join(HERE, "w16v_a2ctrl.json"), "w"), indent=1, default=float)
    print("\nwrote experiments/w16v_a2ctrl.json")


if __name__ == "__main__":
    main()
