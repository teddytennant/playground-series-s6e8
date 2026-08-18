"""w25c — a DEFENSIBLE LB predictor at last: LB ~ CV + transform family.

w17a set the goal explicitly: "until a residual-scatter estimate exists, no LB prediction in
this workspace is defensible." w17a produced the slice-noise half. w25a/w25b supply the piece
it was missing -- the scatter about a single CV->LB line is NOT slice noise, it is dominated by
a fixed per-family offset (h3 sits -8.7e-6 below the pooled line, ens4 +4.5e-6 above it, and
h3's WITHIN-family residual sd is only 3.0e-6 over nine files).

That matters twice over:

  (a) Every LB prediction registered in this workspace has used a single pooled slope, quoted
      as "roughly 2x". That is the CV>=0.970040 pooled fit (1.825). Today's send1 was
      registered at modal 0.97119 on it and printed 0.97116. This script asks what the slope
      is once the family offset is taken out, and how wide the honest interval is.

  (b) If the family offset is real and fixed, the h3-vs-ens4 LB pattern needs no slice story
      at all -- but it also does not become a reason to move the pick, because a fixed offset
      measured only on the public slice is exactly as unidentifiable as a slice draw. The
      script prices it; it does not act on it.

    .venv/bin/python experiments/w25c_ancova.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

t = pd.read_csv("experiments/w25b_families.csv")
h = t[t.cv >= 0.97].copy()            # the decision region; the sub-0.97 public forks are out
h = h[h.fam.isin(["h3", "ens4", "rankraw", "hybrid", "rescale", "w", "wh3"])]

fams = sorted(h.fam.unique())
base = "h3"                            # both WANTED files are h3, so make it the reference
X = [np.ones(len(h)), (h.cv.values - h.cv.mean()) * 1e6]
names = ["const", "cv_e6"]
for f in fams:
    if f == base:
        continue
    X.append((h.fam.values == f).astype(float))
    names.append(f"fam[{f}]")
X = np.column_stack(X)
yv = h.lb.values * 1e6

beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
resid = yv - X @ beta
dof = len(h) - X.shape[1]
s2 = resid @ resid / dof
cov = s2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov))

print(f"n {len(h)}   dof {dof}   residual sd {np.sqrt(s2):.2f}e-6   "
      f"(pooled no-family fit: {12.16:.2f}e-6)")
print(f"\ncoefficient (units: e-6 of LB; reference family = {base})")
for nm, b, s in zip(names, beta, se):
    star = "  *" if abs(b) > 2 * s else ""
    print(f"  {nm:14s} {b:+9.3f}  se {s:6.3f}   t {b/s:+6.2f}{star}")

slope = beta[1]
print(f"\nCV->LB SLOPE with the family offset removed: {slope:+.3f} "
      f"(95% CI {slope-1.96*se[1]:+.3f} .. {slope+1.96*se[1]:+.3f})")
print(f"the workspace has been registering predictions at 'roughly 2x' (pooled 1.825).")

print("\n=== within-family residual sd about the fitted line, e-6 ===")
h["r6"] = resid
print(h.groupby("fam").r6.agg(n="size", sd="std", mean="mean").round(2).to_string())

# What this says about today's send1, after the fact and stated as such.
d = 8.20    # dCV of w23_ad187stdcorr over w21_ad187corr, e-6; both are family h3
pred = slope * d
print(f"\n--- applied to today's send1, POST HOC (both files are family h3, so the offset "
      f"cancels exactly) ---")
print(f"dCV +{d:.2f}e-6  ->  predicted dLB {pred:+.1f}e-6.  Observed dLB -10.0e-6 "
      f"(0.97116 vs 0.97117).")
print(f"residual {-10.0-pred:+.1f}e-6 against w17a's leaders paired slice sd 8.21e-6: "
      f"z {(-10.0-pred)/8.21:+.2f}")
print("Both differences are under ONE 1e-5 reporting step, so the observed -10.0 is itself")
print("a rounding artefact of a true difference somewhere in (-20, 0)e-6. The registered")
print("modal 0.97119 was too confident: the honest interval always spanned 2-3 steps.")

# The h3-vs-ens4 offset, priced against the slice-noise null.
off = beta[names.index("fam[ens4]")]
off_se = se[names.index("fam[ens4]")]
print(f"\n--- the h3/ens4 question, priced ---")
print(f"ens4 offset vs h3: {off:+.2f}e-6 (se {off_se:.2f}, t {off/off_se:+.2f})")
print(f"against w17a's leaders paired slice sd 8.21e-6: z {off/8.21:+.2f}")
print("The ten matched pairs are NOT ten measurements -- they share member sets and are all")
print("read off ONE fixed public slice, so they are one measurement of about +13e-6 with a")
print("slice-noise sd of 8.2e-6, i.e. z ~ +1.6. That is not evidence, and the pick does not")
print("move on it. This reproduces w16s's closure with a proper effect size attached.")

json.dump(dict(n=int(len(h)), slope=float(slope), slope_se=float(se[1]),
               resid_sd=float(np.sqrt(s2)),
               coefs={n: float(b) for n, b in zip(names, beta)},
               ens4_offset=float(off), ens4_offset_se=float(off_se)),
          open("experiments/w25c_ancova.json", "w"), indent=1)
print("\nwrote experiments/w25c_ancova.json")
