"""w48c -- is the -29.82e-6 "era shift" just an over-steep slope read at the top of the range?

Registered in experiments/w48_prereg.txt, committed 9a76f76 BEFORE this file existed.
S1 refit at floor 0.9699 / S2 the five era residuals under it / S3 a quadratic inside support
/ S4 the residual-sd control that stops S2 being satisfied by wrecking the fit.

    .venv/bin/python experiments/w48c_slope.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stdflag import family, is_std                      # noqa: E402
import w46c_predlb as W                                 # noqa: E402

SD_LOO = 8.36
ERA = W.ERA_ROWS

t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"]).copy()
t["fam"] = t.stem.map(family)
t["std"] = t.stem.map(is_std).astype(float)
t["corr"] = t.stem.str.contains("corr").astype(float)
# the five era files are HELD OUT of every fit here, exactly as they were from w30b
t = t[~t.stem.isin([s for s, _, _ in ERA])].copy()


def fit(h, quad=False):
    fams = sorted(h.fam.unique()); mu = h.cv.mean()
    x = (h.cv.values - mu) * 1e6
    X, names = [np.ones(len(h)), x], ["const", "cv_e6"]
    if quad:
        X.append(x ** 2 / 100.0); names.append("cv2_e6")     # /100 for conditioning
    for f in fams:
        if f == "h3":
            continue
        X.append((h.fam.values == f).astype(float)); names.append(f"fam[{f}]")
    for c in ("std", "corr"):
        X.append(h[c].values.astype(float)); names.append(c)
    X = np.column_stack(X); yv = h.lb.values * 1e6
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b; dof = len(h) - X.shape[1]
    sd = float(np.sqrt(r @ r / dof))
    se = np.sqrt(np.diag((r @ r / dof) * np.linalg.inv(X.T @ X)))
    C = dict(zip(names, b))

    def pred(cv, stem):
        xx = (cv - mu) * 1e6
        v = C["const"] + C["cv_e6"] * xx + C.get(f"fam[{family(stem)}]", 0.0)
        if quad:
            v += C["cv2_e6"] * xx ** 2 / 100.0
        if is_std(stem):
            v += C["std"]
        if "corr" in stem:
            v += C["corr"]
        return v * 1e-6
    return C, dict(zip(names, se)), sd, pred, len(h)


def era_resid(pred):
    r = np.array([(lb - pred(cv, s)) * 1e6 for s, cv, lb in ERA])
    return r


print("=" * 88)
print("w48c  OVER-STEEP SLOPE vs ERA SHIFT")
print("=" * 88)

MODELS = [("w30b as fitted   (floor 0.97)", t[t.cv >= 0.97], False),
          ("S1 refit         (floor 0.9699)", t[t.cv >= 0.9699], False),
          ("S3 quadratic     (floor 0.97)", t[t.cv >= 0.97], True),
          ("     quadratic   (floor 0.9699)", t[t.cv >= 0.9699], True)]

res = {}
print(f"\n  {'model':32s} {'n':>3s} {'slope':>7s} {'se':>5s} {'sd':>6s} "
      f"{'fam[logit]':>11s}  {'era mean':>9s} {'z':>7s}")
for lbl, h, q in MODELS:
    C, se, sd, pred, n = fit(h, q)
    r = era_resid(pred)
    z = r.mean() / (SD_LOO / np.sqrt(len(r)))
    res[lbl.strip()] = dict(n=n, slope=C["cv_e6"], se=se["cv_e6"], sd=sd,
                            fam_logit=C.get("fam[logit]", float("nan")),
                            era_mean=float(r.mean()), era_sd=float(r.std(ddof=1)), z=float(z),
                            era_each={s: float(v) for (s, _, _), v in zip(ERA, r)})
    print(f"  {lbl:32s} {n:3d} {C['cv_e6']:7.3f} {se['cv_e6']:5.3f} {sd:6.2f} "
          f"{C.get('fam[logit]', float('nan')):11.2f}  {r.mean():+9.2f} {z:+7.2f}")

print("\n  --- the five era files, residual under each model (e-6) ---")
print(f"  {'stem':24s} " + " ".join(f"{l.split()[0]:>10s}" for l, _, _ in MODELS))
for i, (s, cv, lb) in enumerate(ERA):
    vals = [res[l.strip()]["era_each"][s] for l, _, _ in MODELS]
    print(f"  {s:24s} " + " ".join(f"{v:+10.2f}" for v in vals))

print("\n" + "=" * 88)
print("READING THE REGISTERED RULE (w48_prereg.txt)")
print("=" * 88)
s1 = res["S1 refit         (floor 0.9699)"]
print(f"\n  S1  slope {s1['slope']:.3f} (w30b {res['w30b as fitted   (floor 0.97)']['slope']:.3f})")
print(f"      rule: <=1.70 supports over-steep, >=1.80 does not  ->  "
      f"{'SUPPORTS' if s1['slope'] <= 1.70 else ('DOES NOT' if s1['slope'] >= 1.80 else 'between')}")
print(f"\n  S4  refit residual sd {s1['sd']:.2f}e-6, rule: >12e-6 downgrades S2  ->  "
      f"{'⛔ DOWNGRADE' if s1['sd'] > 12 else 'fit is sound, S2 reads normally'}")
m = s1["era_mean"]
call = ("ERA SHIFT IS A SLOPE ARTEFACT" if abs(m) <= 10 else
        "ERA SHIFT IS INDEPENDENT — w46c STANDS" if m <= -20 else "PARTIAL — change nothing")
print(f"\n  S2  era mean residual under the S1 refit {m:+.2f}e-6  (z {s1['z']:+.2f})")
print(f"      rule: |m|<=10 artefact / m<=-20 independent / else partial  ->  {call}")
q = res["S3 quadratic     (floor 0.97)"]
mq = q["era_mean"]
callq = ("ARTEFACT" if abs(mq) <= 10 else "INDEPENDENT" if mq <= -20 else "PARTIAL")
print(f"\n  S3  era mean under a quadratic inside support {mq:+.2f}e-6  ->  {callq}")

print(f"\n  REGISTERED CALL was: slope 1.75-1.85, era between -22 and -30e-6.")
print(f"  OUTCOME:             slope {s1['slope']:.3f}, era {m:+.2f}e-6  ->  "
      f"{'call correct' if 1.75 <= s1['slope'] <= 1.85 and -30 <= m <= -22 else 'CALL WRONG on at least one leg'}")

json.dump(res, open(os.path.join(HERE, "w48c_slope.json"), "w"), indent=1)
print("\nwrote w48c_slope.json")
