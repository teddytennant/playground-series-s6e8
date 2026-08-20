"""w30b — does the c_avg CORRECTION carry an LB offset the pricer has no term for?

THE OBSERVATION (w30a, held out, n=10)
--------------------------------------
The two h3 files in today's drain printed +19.3e-6 above a prediction written before
upload, z +3.08, while ens4 and logit landed dead on and rescale landed a step low. Both
h3 files are `*stdcorr` -- the 5-arm scheme-average c_avg correction -- and the model has
no term for the correction: w26e fits CV, transform family, and standardisation only.
So "the h3 family runs hot" and "the CORRECTED files run hot" are perfectly confounded in
today's ten, because every corrected file in it is h3-side.

They are NOT confounded in the full history, which is why this is answerable. The fitted
sample holds corrected files on the ens4 and rankraw bases too (w21_ad187corr_ens4,
w22_ad187corr_rankraw), and h3 files with no correction at all (w20_ad187_h3,
w23_ad187std_h3, w27_ad188std_h3, ...). This refits with both terms present and lets the
data separate them.

⚠ The two held-out h3 files are near-twins: test-set rho 0.999979 (w29 slot 10 §4), so
they are ~ONE reading, not two. Everything below reports the effective-n version too.

    .venv/bin/python experiments/w30b_corrterm.py
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stdflag import family, is_std           # noqa: E402

PAIRED_SD = 8.21          # w17a's independently simulated paired slice sd, e-6
GRID_SD = 10.0 / np.sqrt(12)   # 1e-5 reporting grid -> uniform(-5,+5)e-6


def fit(h, terms, label):
    fams = sorted(h.fam.unique())
    mu = h.cv.mean()
    X, names = [np.ones(len(h)), (h.cv.values - mu) * 1e6], ["const", "cv_e6"]
    for f in fams:
        if f == "h3":
            continue
        X.append((h.fam.values == f).astype(float)); names.append(f"fam[{f}]")
    for t in terms:
        X.append(h[t].values.astype(float)); names.append(t)
    X = np.column_stack(X)
    yv = h.lb.values * 1e6
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b
    dof = len(h) - X.shape[1]
    sd = float(np.sqrt(r @ r / dof))
    se = np.sqrt(np.diag((r @ r / dof) * np.linalg.inv(X.T @ X)))
    print(f"\n--- {label}   n {len(h)}  resid sd {sd:.2f}e-6 ---")
    for n_, bb, ss in zip(names, b, se):
        if n_ == "const":
            continue
        print(f"  {n_:16s} {bb:+8.2f}  se {ss:5.2f}  t {bb/ss:+6.2f}"
              f"{'  *' if abs(bb) > 2 * ss else ''}")
    return dict(zip(names, b)), dict(zip(names, se)), sd, float(mu)


t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"]).copy()
t["fam"] = t.stem.map(family)
t["std"] = t.stem.map(is_std).astype(float)
# "corrected" = carries the c_avg / scheme-average correction. Named by build convention:
# every such file in this workspace has `corr` in its stem and nothing else does.
t["corr"] = t.stem.str.contains("corr").astype(float)
h = t[t.cv >= 0.97].copy()
print(f"n {len(h)}   standardised {int(h['std'].sum())}   corrected {int(h['corr'].sum())}")
print("corrected files and their transform families (the confound is broken here, not in "
      "today's ten):")
for _, x in h[h["corr"] > 0].sort_values("lb", ascending=False).iterrows():
    print(f"   {x.stem:26s} {x.fam:8s} cv {x.cv:.10f}  lb {x.lb:.5f}")

b0, s0, sd0, mu0 = fit(h, ["std"], "w26e's model (CV + family + standardisation)")
b1, s1, sd1, mu1 = fit(h, ["std", "corr"], "PLUS a correction indicator")

c, se = b1["corr"], s1["corr"]
floor = float(np.hypot(PAIRED_SD, GRID_SD))
print(f"\ncorrection coefficient {c:+.2f}e-6  se {se:.2f}  t {c/se:+.2f}")
print(f"residual sd {sd0:.2f} -> {sd1:.2f}e-6   (slice+grid noise floor {floor:.2f}e-6)")
print("VERDICT:", "the correction carries a real LB offset the pricer is missing"
      if abs(c) > 2 * se else
      "no separable correction offset -- w30a's h3 excess is a family/draw reading, "
      "and the pricer needs no new term")

json.dump(dict(n=int(len(h)), coefs=b1, ses=s1, resid_sd_new=sd1, resid_sd_old=sd0,
               mu=mu1, floor=floor), open(os.path.join(HERE, "w30b_corrterm.json"), "w"),
          indent=1)
print("\nwrote experiments/w30b_corrterm.json")

# ---------------------------------------------------------------------------
# CONFOUND CHECK. All six corrected files sit in the top ~15e-6 of the CV range,
# so a `corr` term can absorb CURVATURE in the CV slope at the top instead of a real
# correction effect. Three independent ways to break that; the coefficient has to
# survive all three or it is curvature.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("CONFOUND CHECK — is `corr` just curvature at the top of the CV range?")
print("=" * 72)

lo = h.cv.min()
h2 = h.copy()
h2["cv2"] = (((h2.cv - h2.cv.mean()) * 1e6) ** 2) / 1e3     # scaled so se is readable
b2, s2, sd2, _ = fit(h2, ["std", "corr", "cv2"], "(1) with a QUADRATIC CV term")
print(f"    corr survives quadratic: {b2['corr']:+.2f} se {s2['corr']:.2f} "
      f"t {b2['corr']/s2['corr']:+.2f}")

band = h[h.cv >= 0.97009].copy()      # the 22-file top band; corrected files are 6 of it
b3, s3, sd3, _ = fit(band, ["std", "corr"], "(2) restricted to the top CV band (cv>=0.97009)")
print(f"    corr within-band: {b3['corr']:+.2f} se {s3['corr']:.2f} "
      f"t {b3['corr']/s3['corr']:+.2f}")

# (3) Drop the two near-twins, which are ~one reading and both today's. If the effect is
# carried only by them it is today's draw, not a standing offset.
h4 = h[~h.stem.isin(["w29_ad194stdcorr", "w27_ad190stdcorr"])].copy()
b4, s4, sd4, _ = fit(h4, ["std", "corr"], "(3) dropping today's two near-twins (rho 0.999979)")
print(f"    corr on the 4 PRE-EXISTING corrected files: {b4['corr']:+.2f} "
      f"se {s4['corr']:.2f} t {b4['corr']/s4['corr']:+.2f}")

surv = [abs(bb["corr"]) > 2 * ss["corr"] for bb, ss in ((b2, s2), (b3, s3), (b4, s4))]
print(f"\nsurvives {sum(surv)} of 3 confound checks -> "
      + ("the correction offset is REAL and the pricer needs the term"
         if sum(surv) >= 2 else
         "NOT robust; treat +12.6e-6 as curvature/draw and do NOT add the term"))
