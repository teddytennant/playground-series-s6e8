"""w25f — refit w25c's LB model on the enlarged sample, with a STANDARDISATION term.

w25c fitted LB ~ CV + transform family on the 51 files scored before this slot's sends and
got a residual sd of 8.67e-6, essentially equal to w17a's independently simulated 8.21e-6
paired slice sd -- i.e. the relation looked fully accounted for. That sample contained exactly
ONE standardised file. Today's ten sends added six more, and they are the model's out-of-sample
test. It fails it: on the enlarged sample the CV>=0.97 pooled residual sd rises from 12.16e-6
to 19.83e-6.

So the honest model needs a third term. This script fits it and reports whether the fit
returns to the slice-noise floor once the standardisation is accounted for -- which is the
difference between "the CV->LB relation is understood" and "there is still something loose".

    .venv/bin/python experiments/w25f_ancova2.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

STD_FILES = {"w23_ad187stdcorr", "w23_ad187std_h3", "w23_ad187std", "w23_ad187std_logit",
             "w23_ad187std_h3_hybrid", "w23_ad187std_h3_rankraw", "w23_ad187std_h3_rescale"}
PAIRED_SD = 8.21     # w17a simulated leaders paired slice sd, e-6


def family(s):
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit"):
        if s.endswith("_" + f):
            return f
    if s in ("blend156w", "blend156w2") or s.endswith("_w") or s.endswith("_w2"):
        return "w"
    return "ens4"


t = pd.read_csv("experiments/w25a_cvlb_full.csv").dropna(subset=["cv"]).copy()
t["fam"] = t.stem.map(family)
t["std"] = t.stem.isin(STD_FILES).astype(float)
h = t[t.cv >= 0.97].copy()
print(f"n {len(h)}   standardised {int(h['std'].sum())}   families {sorted(h.fam.unique())}")


def fit(h, use_std, label):
    fams = sorted(h.fam.unique())
    mu = h.cv.mean()
    X, names = [np.ones(len(h)), (h.cv.values - mu) * 1e6], ["const", "cv_e6"]
    for f in fams:
        if f == "h3":
            continue
        X.append((h.fam.values == f).astype(float))
        names.append(f"fam[{f}]")
    if use_std:
        X.append(h["std"].values)
        names.append("standardised")
    X = np.column_stack(X)
    yv = h.lb.values * 1e6
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b
    dof = len(h) - X.shape[1]
    cov = (r @ r / dof) * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    sd = np.sqrt(r @ r / dof)
    print(f"\n--- {label}:  residual sd {sd:.2f}e-6   (slice-noise floor {PAIRED_SD:.2f}e-6) ---")
    for n_, bb, ss in zip(names, b, se):
        if n_ == "const":
            continue
        star = "  *" if abs(bb) > 2 * ss else ""
        print(f"  {n_:14s} {bb:+8.3f}  se {ss:5.3f}   t {bb/ss:+6.2f}{star}")
    return dict(zip(names, b)), dict(zip(names, se)), float(sd), r


b0, s0, sd0, _ = fit(h, False, "w25c's model refitted on the enlarged sample (no std term)")
b1, s1, sd1, r1 = fit(h, True, "with a standardisation indicator")

print(f"\nadding the standardisation term takes the residual sd from {sd0:.2f} to {sd1:.2f}e-6")
print(f"the slice-noise floor is {PAIRED_SD:.2f}e-6, so the fit is now "
      f"{'AT' if sd1 < PAIRED_SD * 1.25 else 'STILL ABOVE'} it.")
print(f"\nstandardisation coefficient {b1['standardised']:+.2f}e-6 "
      f"(se {s1['standardised']:.2f}, t {b1['standardised']/s1['standardised']:+.2f}) — "
      f"that is {abs(b1['standardised'])/PAIRED_SD:.1f}x the paired slice sd.")
print(f"CV->LB slope: {b0['cv_e6']:+.3f} without the term, {b1['cv_e6']:+.3f} with it "
      f"(se {s1['cv_e6']:.3f}, 95% CI {b1['cv_e6']-1.96*s1['cv_e6']:+.3f} .. "
      f"{b1['cv_e6']+1.96*s1['cv_e6']:+.3f})")

h = h.assign(resid=r1)
print("\n=== the five largest remaining residuals, e-6 ===")
print(h.reindex(h.resid.abs().sort_values(ascending=False).index)
      .head(5)[["stem", "fam", "cv", "lb", "resid"]]
      .assign(cv=lambda d: d.cv.map("{:.7f}".format),
              resid=lambda d: d.resid.map("{:+.1f}".format)).to_string(index=False))

json.dump(dict(n=int(len(h)), resid_sd_no_std=sd0, resid_sd_with_std=sd1,
               slice_floor=PAIRED_SD, coefs=b1, ses=s1),
          open("experiments/w25f_ancova2.json", "w"), indent=1)
print("\nwrote experiments/w25f_ancova2.json")
