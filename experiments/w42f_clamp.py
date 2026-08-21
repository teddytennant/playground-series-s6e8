"""w42f -- WHICH correction does the w41b held-out failure actually license?

w41 §5 measured the four held-out STACK files landing a mean 29.2e-6 BELOW the w30b
prediction (z = -8.1 against an in-sample residual sd of 7.2e-6) and described it as the
CV->LB slope FLATTENING at the top of our range. Two different repairs are consistent with
that sentence and they do NOT agree about a higher-CV arm:

  LEVEL  a constant offset applied above the fitted range. A better arm keeps earning LB
         at the fitted rate 1.856e-6 per 1e-6 of CV; it just starts one step lower.
  CLAMP  CV above the fitted maximum converts at ZERO. A better arm earns nothing at all
         above 0.9701183, and the whole 197/202/211/217 programme is priced against a
         conversion that has stopped.

They are separable on the four files, because their CV excesses over the fitted maximum
differ by 6x (3.6e-6 to 21.7e-6). CLAMP predicts a shortfall PROPORTIONAL to that excess;
LEVEL predicts the same shortfall for all four. This script asks which one the data says,
and reports the answer even when it is "neither, at this n".
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))["coefs"]
SLOPE = C["cv_e6"]
HI = 0.9701183          # the fitted sample's CV maximum, frozen in w41b
RESID_IN = 7.2          # in-sample residual sd, e-6

h = pd.read_csv(os.path.join(HERE, "w41b_heldout.csv"))
s = h[h.cv > HI].copy()                    # the STACK group; the 5 singles are below the min
s["excess_e6"] = (s.cv - HI) * 1e6
s["clamp_pred"] = -SLOPE * s.excess_e6     # shortfall if excess CV converts at zero
s["level_pred"] = s.resid_e6.mean()

print(f"slope {SLOPE:.4f}e-6 LB per 1e-6 CV; fitted CV max {HI}\n")
print(s[["file", "cv", "excess_e6", "resid_e6", "clamp_pred", "level_pred"]]
      .round(2).to_string(index=False))

for lab, col in (("CLAMP", "clamp_pred"), ("LEVEL", "level_pred")):
    e = s.resid_e6 - s[col]
    print(f"\n{lab}: residual-after-correction mean {e.mean():+7.2f}e-6  "
          f"rms {np.sqrt((e**2).mean()):6.2f}e-6   per-file {np.round(e.values,1)}")

r = np.corrcoef(s.excess_e6, s.resid_e6)[0, 1]
print(f"\ncorr(excess, shortfall) = {r:+.3f} on n={len(s)}  "
      f"(CLAMP wants strongly NEGATIVE, LEVEL wants 0)")

# The rescale file is the one that separates them; report the fit with and without it, and
# say plainly that dropping a point at n=4 is not evidence, only a sensitivity.
m = ~s.file.str.contains("rescale")
e_c = (s.resid_e6 - s.clamp_pred)[m]
print(f"\nSENSITIVITY, excluding the one `rescale` file (n={m.sum()}, NOT a result):")
print(f"  CLAMP residual mean {e_c.mean():+.2f}e-6  rms {np.sqrt((e_c**2).mean()):.2f}e-6"
      f"  per-file {np.round(e_c.values,1)}")
print(f"  corr(excess, shortfall) = "
      f"{np.corrcoef(s.excess_e6[m], s.resid_e6[m])[0,1]:+.3f}")

# effective n: the four files are near-twins (w29 slot 10 measured test-set rho 0.999979
# between the two h3 siblings), so these are NOT four independent readings.
print(f"\n⚠ EFFECTIVE n. These four are near-twin stacks over one 195-199 member pack; w29 "
      f"\n  measured rho 0.999979 between two of them on the test set. Treat n=4 as ~2 "
      f"\n  independent readings. Neither model is established at this n; what IS established"
      f"\n  is that the four came in low, so both repairs deflate and NEITHER inflates.")

out = dict(slope_e6=SLOPE, cv_fitted_max=HI, n=int(len(s)),
           level_offset_e6=float(s.resid_e6.mean()),
           clamp_resid_rms_e6=float(np.sqrt(((s.resid_e6 - s.clamp_pred) ** 2).mean())),
           level_resid_rms_e6=float(np.sqrt(((s.resid_e6 - s.level_pred) ** 2).mean())),
           corr_excess_shortfall=float(r), resid_sd_in_sample_e6=RESID_IN)
json.dump(out, open(os.path.join(HERE, "w42f_clamp.json"), "w"), indent=1)
print(f"\nwrote experiments/w42f_clamp.json")
