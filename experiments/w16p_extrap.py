"""w16p -- read w16o_blendbag.json and project the blend-size curve out to the real pack.

MODEL, stated before it was fitted to the 400-round run (it was chosen on the 25-round smoke
run, whose numbers are in logs only and are not quoted anywhere as a result).

A fold model's prediction carries an idiosyncratic error on top of the signal.  Split that
error into a part COMMON to every member at the same fold index -- the members share the
fold's 80% training subsample, which is exactly the mechanism w15g named -- and a part
SPECIFIC to the member.  Blending k members at the SAME fold index averages the specific
part and leaves the common part intact; bagging over folds removes both.  If the AUC penalty
is locally linear in the error variance, then

    gap_aligned(k)    = A + B / k          A = the common floor, B = the member-specific part
    gap_misaligned(k) = (A + B) / k        misalignment makes the common part average too

so the fraction of the single-member bagging gap that survives an arbitrarily large blend is
A / (A + B), and the projection at the real pack size is A + B/160.

Both forms are fitted per outer split by ordinary least squares in 1/k, so the spread across
splits gives the uncertainty rather than an assumed one.  The misaligned form is a PREDICTION
of the model with no free parameter beyond A+B (which the aligned fit already supplies), so
comparing it to the measured misaligned curve is a check on the model, not another fit.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = os.path.join(ROOT, "experiments", "w16o_blendbag.json")


def main() -> None:
    d = json.load(open(J))
    M = len(d["names"])
    ks = np.arange(1, M + 1)
    print(f"members {d['names']}   outer splits {d['n_outer']}   rounds {d['n_est']}\n")

    A_s, B_s, proj_s, surv_s = [], [], [], []
    for sp in d["splits"]:
        g = np.array([sp["curve"][str(k)]["gap_aligned_mean"] for k in ks])
        # OLS of g on 1/k
        Xd = np.vstack([np.ones_like(ks, float), 1.0 / ks]).T
        A, B = np.linalg.lstsq(Xd, g, rcond=None)[0]
        pred = Xd @ np.array([A, B])
        rms = float(np.sqrt(np.mean((g - pred) ** 2)))
        A_s.append(A)
        B_s.append(B)
        proj_s.append(A + B / 160.0)
        surv_s.append(A / (A + B))
        print(f"  seed {sp['seed']}: measured " +
              " ".join(f"k{k}={x * 1e6:+.1f}" for k, x in zip(ks, g)) +
              f"   ->  A {A * 1e6:+.1f}  B {B * 1e6:+.1f}  fit rms {rms * 1e6:.2f}e-6")
        gm = [sp["curve"][str(k)]["gap_misaligned_mean"] for k in ks[1:]]
        predm = (A + B) / ks[1:]
        print("        misaligned measured " +
              " ".join(f"k{k}={x * 1e6:+.1f}" for k, x in zip(ks[1:], gm)) +
              "   model (A+B)/k " +
              " ".join(f"k{k}={x * 1e6:+.1f}" for k, x in zip(ks[1:], predm)))

    def s(v):
        v = np.array(v)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        return v.mean(), se

    print("\n" + "=" * 78)
    for lab, v, sc in (("A  (common floor, k->inf)", A_s, 1e6),
                       ("B  (member-specific)", B_s, 1e6),
                       ("A + B/160  (projection at the real pack size)", proj_s, 1e6)):
        m, se = s(v)
        print(f"  {lab:46s} {m * sc:+9.1f}e-6   se {se * sc:6.1f}")
    m, se = s(surv_s)
    print(f"  {'SURVIVAL FRACTION A/(A+B)':46s} {m * 100:9.1f}%    se {se * 100:6.1f}")

    g1, _ = s([sp["curve"]["1"]["gap_aligned_mean"] for sp in d["splits"]])
    print(f"\n  single-member gap measured here            {g1 * 1e6:+9.1f}e-6")
    print(f"  w15g_cvgap.py single-member reference      +662.6e-6 se 6.9")
    pm, pse = s(proj_s)
    print(f"  => of w15g's +662.6e-6, the 160-member pack keeps "
          f"{pm / g1 * 100:.1f}% -> {662.6e-6 * pm / g1 * 1e6:+.1f}e-6 "
          f"(rescaled to w15g's baseline)")
    print(f"  the residual CV->LB gap this had to explain (w15c/w15g): +98e-6")

    # The full confound-free decomposition at the blend level, k by k.
    # gap_total = bagged(HOLD) - CV(TRAIN);  gap_honesty = aligned(HOLD) - CV(TRAIN)
    print("\n" + "=" * 78)
    print("CONFOUND-FREE GAP DECOMPOSITION AT THE BLEND LEVEL, e-6")
    print(f"{'k':>2} {'gap_bagging':>13} {'gap_honesty':>13} {'gap_total':>13}")
    for k in ks:
        tot, bag, hon = [], [], []
        for sp in d["splits"]:
            b = sp["curve"][str(k)]["gap_aligned_mean"]
            # rebuild total for this k from the k=M total, which is the only one stored
            bag.append(b)
        b_, bse = s(bag)
        if k == M:
            t_, tse = s([sp["gap_total_blend"] for sp in d["splits"]])
            h_ = t_ - b_
            print(f"{k:>2} {b_ * 1e6:+10.1f}+-{bse * 1e6:<4.1f} {h_ * 1e6:+13.1f} "
                  f"{t_ * 1e6:+10.1f}+-{tse * 1e6:<4.1f}")
        else:
            print(f"{k:>2} {b_ * 1e6:+10.1f}+-{bse * 1e6:<4.1f} {'-':>13} {'-':>13}")
    print("  w15g single member: bagging +662.6  honesty -230.2  total +432.4")


if __name__ == "__main__":
    main()
