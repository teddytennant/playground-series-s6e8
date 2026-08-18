"""w26e — the w25f family classifier MISLABELS the two `corr` files, including WANTED slot 1.

w25b/w25f assign a transform family by stem suffix, and the fallthrough is "a bare stem is the
rank-average of all four transforms", i.e. ens4. Two stems fall through that they should not:

    w21_ad187corr      -> labelled ens4
    w23_ad187stdcorr   -> labelled ens4

Both are h3 mixes carrying the 5-arm `c_avg` scheme-average correction. The journal is
unambiguous on this: w21 §2 builds `w21_ad187corr` as the corrected h3-side file and w21 §9
contrasts it against `w21_ad187corr_ens4`, "the corrected ens4", as a matched h3-vs-ens4 pair;
w25 §4's pair table lists them under "corrected h3". So the classifier puts the h3 member of a
matched h3/ens4 pair into the ens4 group, alongside its own counterpart.

This is not cosmetic. `w23_ad187stdcorr` is slot 1 of `WANTED`, and w25f's ens4 term is
+14.0e-6 against h3's 0. Any LB prediction for the deadline pick has been carrying a family
offset belonging to the mix it was built NOT to be. w25 §1's registered forecast for that very
file missed 3 reporting steps low, and this is a candidate contributor to that miss.

Refits w25f exactly -- same rows, same CV >= 0.97 filter, same centring, same standardised
indicator, same dof -- changing ONLY the two labels, and reports every coefficient side by
side. NOTHING here can move `WANTED`: w24 R3 and w25 §4 S4 both forbid it, and this is an
LB-side model in any case. It changes what this workspace should PREDICT, not what it ships.

    .venv/bin/python experiments/w26e_famfix.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STD_FILES = {"w23_ad187stdcorr", "w23_ad187std_h3", "w23_ad187std", "w23_ad187std_logit",
             "w23_ad187std_h3_hybrid", "w23_ad187std_h3_rankraw", "w23_ad187std_h3_rescale"}
# The correction: both are corrected h3, not ens4. Kept as an explicit map rather than a
# smarter suffix rule, because "corr" is not a transform and a rule that guessed it would be
# guessing. Any future `*corr` file must be added here by hand.
CORR_H3 = {"w21_ad187corr": "h3", "w23_ad187stdcorr": "h3"}
PAIRED_SD = 8.21


def family(s):
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit"):
        if s.endswith("_" + f):
            return f
    if s.endswith("_w") or s.endswith("_w2") or s in ("blend156w", "blend156w2"):
        return "w"
    return "ens4"


def fit(h, famcol):
    fams = sorted(h[famcol].unique())
    mu = h.cv.mean()
    X, names = [np.ones(len(h)), (h.cv.values - mu) * 1e6], ["const", "cv_e6"]
    for f in fams:
        if f == "h3":
            continue
        X.append((h[famcol].values == f).astype(float))
        names.append(f"fam[{f}]")
    X.append(h["std"].values)
    names.append("standardised")
    X = np.column_stack(X)
    yv = h.lb.values * 1e6
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b
    dof = len(h) - X.shape[1]
    s2 = (r ** 2).sum() / dof
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * s2)
    return dict(zip(names, b)), dict(zip(names, se)), float(np.sqrt(s2)), mu


t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"]).copy()
t["std"] = t.stem.isin(STD_FILES).astype(float)
t["fam_old"] = t.stem.map(family)
t["fam_new"] = [CORR_H3.get(s, f) for s, f in zip(t.stem, t.fam_old)]
h = t[t.cv >= 0.97].copy()

moved = h[h.fam_old != h.fam_new]
print(f"n {len(h)}   rows relabelled: {len(moved)}")
print(moved[["stem", "fam_old", "fam_new", "cv", "lb"]].to_string(index=False))

b0, s0, r0, mu = fit(h, "fam_old")
b1, s1, r1, _ = fit(h, "fam_new")

print(f"\n{'term':16s} {'w25f (old)':>12s} {'corrected':>12s} {'shift':>9s} {'se(new)':>8s}")
for k in b1:
    print(f"{k:16s} {b0.get(k, float('nan')):+12.3f} {b1[k]:+12.3f} "
          f"{b1[k]-b0.get(k, float('nan')):+9.3f} {s1[k]:8.3f}")
print(f"\nresidual sd   {r0:.3f}e-6 -> {r1:.3f}e-6   "
      f"(simulated slice-noise floor {PAIRED_SD:.2f}e-6)")
print("A LOWER residual sd is evidence the corrected labels are the right ones: the same "
      "rows,\nthe same number of parameters, and the fit improves purely from moving two "
      "files to the\nfamily the journal says they belong to.")

# What the relabel does to the deadline pick's own predicted LB.
print("\n=== consequence for the files that matter ===")
for stem in ("w23_ad187stdcorr", "w21_ad187corr", "w21_ad187corr_ens4"):
    row = h[h.stem == stem]
    if not len(row):
        continue
    r = row.iloc[0]
    def pred(b, fam):
        v = b["const"] + b["cv_e6"] * (r.cv - mu) * 1e6 + b.get(f"fam[{fam}]", 0.0)
        return (v + b["standardised"] * r["std"]) * 1e-6
    p0, p1 = pred(b0, r.fam_old), pred(b1, r.fam_new)
    print(f"{stem:22s} actual LB {r.lb:.5f}   old-label pred {p0:.6f} "
          f"(resid {(r.lb-p0)*1e6:+6.1f}e-6)   corrected pred {p1:.6f} "
          f"(resid {(r.lb-p1)*1e6:+6.1f}e-6)")

json.dump(dict(n=len(h), relabelled=moved.stem.tolist(), mu=float(mu),
               resid_sd_old=r0, resid_sd_new=r1, coefs_old=b0, coefs_new=b1, ses_new=s1),
          open(os.path.join(HERE, "w26e_famfix.json"), "w"), indent=1)
print("\nwrote experiments/w26e_famfix.json")
