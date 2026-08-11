"""Propagate the logit CV-bias onto the h3-vs-ens4 contrast, and predict tomorrow's scores.

`logit_bias.py` established that the `logit` stack's (LB - CV) gap sits +9.7e-5 above
`hybrid`'s, replicated 2/2 within member set, with rankraw/rescale controls at +1.8e-5 /
+3.1e-5 and an exact permutation p of 0.0085. This script asks the only question that
matters: **does that bias, propagated onto the four-transform mix, exceed the +5e-6 OOF
margin that made `h3` the deadline pick?**

Two confounds are handled explicitly, because ignoring either gives a confident wrong
answer:

1.  **The gap shrinks as CV rises.** stack_pub74/88/86 sit at CV 0.9696 with gaps of
    +0.00112 to +0.00117, far above the ~+0.00100 of the 150-member era. Pooling them into
    a "logit family mean" inflates the logit gap by ~4e-5 and would predict
    `blend158_logit` at 0.97108 -- a leaderboard-topping score -- purely from the
    contamination. Everything here is restricted to the >=150-member sets.
2.  **h3 has never been scored.** Its gap must be estimated, and the only available
    estimator is "the mix's gap is the mean of its components' gaps". That estimator is
    validated on the one member set where all four components AND the mix are scored
    (150fx); the validation error is reported next to the effect it is being used to
    measure, so the reader can see whether the instrument is sharp enough.

    predict_lb.py
"""
from __future__ import annotations

import numpy as np

# (CV, LB) for >=150-member stacks only. The 74/88/86-member entries are deliberately
# excluded: see confound 1.
G = {
    ("150fx", "logit"):   (0.969950, 0.97103),
    ("150fx", "hybrid"):  (0.970014, 0.97099),
    ("150fx", "rankraw"): (0.970024, 0.97102),
    ("150fx", "rescale"): (0.970013, 0.97102),
    ("150fx", "ens4"):    (0.970032, 0.97104),
    ("150sx", "logit"):   (0.969955, 0.97104),
    ("150sx", "hybrid"):  (0.970016, 0.97101),
    ("150sx", "rankraw"): (0.970022, 0.97102),
    ("150sx", "ens4"):    (0.970033, 0.97104),
    ("151",   "hybrid"):  (0.970024, 0.97099),
    ("151",   "rankraw"): (0.970023, 0.97102),
    ("156",   "rankraw"): (0.970036, 0.97104),
    ("156",   "rescale"): (0.970025, 0.97105),
    ("156",   "ens4"):    (0.970042, 0.97106),
}
gap = {k: lb - cv for k, (cv, lb) in G.items()}

# CV of the unscored 158-member candidates, from audit_results.csv.
CV158 = {"h3": 0.970048, "ens4": 0.970043, "rankraw": 0.970036, "hybrid": 0.970028,
         "rescale": 0.970027, "logit": 0.969961}


def main():
    print("=== the mix-gap estimator, validated on 150fx (all four components scored) ===")
    comp = ("logit", "hybrid", "rankraw", "rescale")
    pred = np.mean([gap[("150fx", t)] for t in comp])
    act = gap[("150fx", "ens4")]
    print(f"  mean of the four component gaps {pred:+.6f}")
    print(f"  actual ens4 gap                 {act:+.6f}")
    print(f"  estimator error                 {act - pred:+.6f}")

    print("\n=== per-transform gap, averaged over the >=150-member sets ===")
    for t in ("logit", "hybrid", "rankraw", "rescale", "ens4"):
        v = [gap[k] for k in gap if k[1] == t]
        print(f"  {t:>8s} n={len(v)}  {np.mean(v):+.6f}   " +
              " ".join(f"{x:+.6f}" for x in sorted(v)))

    g_h = np.mean([gap[k] for k in gap if k[1] == "hybrid"])
    g_r = np.mean([gap[k] for k in gap if k[1] == "rankraw"])
    g_s = np.mean([gap[k] for k in gap if k[1] == "rescale"])
    g_l = np.mean([gap[k] for k in gap if k[1] == "logit"])
    g_h3 = np.mean([g_h, g_r, g_s])
    g_e4 = np.mean([g_h, g_r, g_s, g_l])

    print("\n=== the propagation: does the bias beat the margin? ===")
    print(f"  estimated h3   gap  {g_h3:+.6f}  (mean of hybrid/rankraw/rescale)")
    print(f"  estimated ens4 gap  {g_e4:+.6f}  (mean of all four)")
    print(f"  ens4 - h3 in GAP    {g_e4 - g_h3:+.6f}   <-- favours ens4 on test")
    print(f"  h3 - ens4 in CV     {CV158['h3'] - CV158['ens4']:+.6f}   <-- favours h3 on OOF")
    net = (CV158["ens4"] + g_e4) - (CV158["h3"] + g_h3)
    print(f"  net predicted LB, ens4 - h3     {net:+.6f}")
    print(f"\n  the OOF margin that selected h3 is {CV158['h3'] - CV158['ens4']:.6f};")
    print(f"  the bias running the other way is  {g_e4 - g_h3:.6f}  "
          f"({(g_e4 - g_h3) / (CV158['h3'] - CV158['ens4']):.1f}x larger)")
    print(f"  estimator error from the 150fx check is {abs(act - pred):.6f}, which is "
          f"{abs(act - pred) / (g_e4 - g_h3):.0%} of the effect -- so the SIGN is")
    print("  informative and the magnitude is not.")

    print("\n=== falsifiable predictions for the 158-member files (all unsent) ===")
    gm = {"h3": g_h3, "ens4": g_e4, "hybrid": g_h, "rankraw": g_r, "rescale": g_s,
          "logit": g_l}
    print(f"{'file':>22s} {'CV':>9s} {'gap':>10s} {'predicted LB':>13s}")
    for t, cv in sorted(CV158.items(), key=lambda kv: -(kv[1] + gm[kv[0]])):
        print(f"{'blend158_' + t:>22s} {cv:9.6f} {gm[t]:+10.6f} {cv + gm[t]:13.5f}")
    print("\n  The CV ranking puts h3 first and logit last. This model puts ens4 first and")
    print("  logit level with hybrid. Sending all six settles it against a real slice.")


if __name__ == "__main__":
    main()
