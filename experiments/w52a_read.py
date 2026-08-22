"""w52a -- the 08-22 read, executing w47b_prereg.txt verbatim.

Three readings were fixed in advance, before any of the ten was sent:

  p  = mean over the FIVE PROBES of (actual LB - w30b prediction), units 1e-6.
       p <= -18 -> ERA CONFIRMED (w46c stands).
       p >= -12 -> CV-REGION CONFIRMED (w46c's gate is wrong, replace with a cv threshold).
       in between -> MIXED (fit both terms jointly on the pooled 15, report both).

  r5 = mean over the FIVE DRAIN files of (actual - w30b prediction).
       r5 <= -20 -> H1 (the era correction) confirmed on the high-CV side.
       r5 >= -10 -> H1 rejected there, revert to w30b.
       in between -> undecided.  Report incl. AND excl. the two *stdcorr files (w46d).

  E4 = OLS of (actual - w30b prediction) on cv, over the pooled 15 scored era files.
       slope <= -0.60 per e-6 -> bad CV slope, re-estimate the slope, not a dummy.
       within +/-0.40         -> a level shift, the dummy is the right form.

ALL THREE ARE READ.  The prereg is explicit that the one which suits the day's argument
may not be picked out of the three.
"""
from __future__ import annotations

import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w46c_predlb as W46C

# --- the ten sent on 2026-08-22, with the public scores the API returned. ------------
# (stem, cv, actual public LB).  CVs are the values w48e_order.py verified against the
# stored OOF vectors immediately before the send.
PROBES = [("w40_ad211std_rankraw", 0.9701044022, 0.97114),
          ("w38_ad202std_rankraw", 0.9701067554, 0.97114),
          ("w34_ad196std_hybrid",  0.9701091250, 0.97113),
          ("w34_ad195std_hybrid",  0.9701108333, 0.97113),
          ("w36_ad197std_hybrid",  0.9701124257, 0.97114)]

DRAINS = [("w36_ad199std_h3",  0.9701354276, 0.97116),
          ("w38_ad202std",     0.9701305726, 0.97116),
          ("w40_ad211std",     0.9701309541, 0.97117),
          ("w40_ad211stdcorr", 0.9701374733, 0.97118),
          ("w38_ad202stdcorr", 0.9701375891, 0.97117)]

# the five 08-21 rows w46c fitted ERA_SHIFT on; pooled with today's ten for E4.
PRIOR = [(s, cv, lb) for s, cv, lb in W46C.ERA_ROWS]


def resid6(stem, cv, lb):
    """actual - w30b(uncorrected) prediction, in units of 1e-6."""
    return (lb - W46C._raw(cv, stem)) / 1e-6


def table(name, rows):
    out = []
    print(f"\n{name}")
    print(f"  {'stem':24s} {'cv':>14s} {'pred_w30b':>10s} {'actual':>9s} {'resid e-6':>10s}")
    for s, cv, lb in rows:
        r = resid6(s, cv, lb)
        out.append(r)
        print(f"  {s:24s} {cv:.10f} {W46C._raw(cv,s):10.6f} {lb:9.5f} {r:+10.2f}")
    a = np.array(out)
    print(f"  {'mean':24s} {'':14s} {'':10s} {'':9s} {a.mean():+10.2f}   (sd {a.std(ddof=1):.2f}, n {len(a)})")
    return a


print("=" * 92)
print("w52a -- the 08-22 read.  w47b_prereg.txt, all three rules.")
print("=" * 92)
print(f"w30b: n=78, resid sd {W46C.SD_OLD:.2f}e-6, MU {W46C.MU:.10f}")
print(f"w46c: ERA_SHIFT {W46C.ERA_SHIFT:+.2f}e-6 (se {W46C.ERA_SE:.2f}, n {W46C.ERA_N})")

pr = table("READING 1 -- THE FIVE PROBES (inside CV support, ad>=195)", PROBES)
dr = table("READING 2 -- THE FIVE DRAIN FILES (above CV support, ad>=195)", DRAINS)
po = table("(context) THE FIVE 08-21 ROWS w46c WAS FITTED ON", PRIOR)

SEM = 3.74   # w47a leave-one-out per-file held-out sd 8.36e-6 / sqrt(5)

# ---------------- READING 1 -----------------------------------------------------------
p = float(pr.mean())
if p <= -18:
    v1, act1 = "ERA CONFIRMED", "w46c stands exactly as written; `new_era` stays the gate."
elif p >= -12:
    v1, act1 = "CV-REGION CONFIRMED", ("w46c's gate is WRONG -- replace with a cv>0.970118 "
                                       "threshold, mark w46c superseded, re-price the 19 "
                                       "in-support ad>=195 queue files UP by 29.8e-6.")
else:
    v1, act1 = "MIXED", "fit an era term and a cv-region term jointly on the pooled 15; report both."

# ---------------- READING 2 -----------------------------------------------------------
r5 = float(dr.mean())
nocorr = np.array([resid6(s, cv, lb) for s, cv, lb in DRAINS if "corr" not in s])
r3 = float(nocorr.mean())
if r5 <= -20:
    v2 = "H1 CONFIRMED on the high-CV side"
elif r5 >= -10:
    v2 = "H1 REJECTED there -- revert to w30b"
else:
    v2 = "UNDECIDED"

# ---------------- READING 3 (E4) ------------------------------------------------------
pool = PROBES + DRAINS + PRIOR
x = np.array([(cv - W46C.MU) * 1e6 for _, cv, _ in pool])
y = np.array([resid6(s, cv, lb) for s, cv, lb in pool])
X = np.column_stack([np.ones(len(x)), x])
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X @ beta
s2 = float(res @ res) / (len(x) - 2)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
slope, slope_se = float(beta[1]), float(se[1])
level, level_se = float(beta[0]), float(se[0])
if slope <= -0.60:
    v3 = "BAD CV SLOPE -- the fix is a re-estimated slope, not a dummy"
elif abs(slope) <= 0.40:
    v3 = "LEVEL SHIFT -- the dummy is the right form"
else:
    v3 = "IN BETWEEN the registered thresholds -- neither branch fires"

print("\n" + "=" * 92)
print("VERDICTS -- all three, as registered")
print("=" * 92)
print(f"\nR1  p  = {p:+.2f}e-6   (sem {SEM:.2f}, z vs 0 = {p/SEM:+.2f})")
print(f"    thresholds: ERA p<=-18 | MIXED | CV-REGION p>=-12")
print(f"    VERDICT: {v1}")
print(f"    ACTION : {act1}")
print(f"\nR2  r5 = {r5:+.2f}e-6 over all five   (excl. the two *stdcorr: r3 = {r3:+.2f}e-6, n=3)")
print(f"    thresholds: H1 r5<=-20 | UNDECIDED | reject r5>=-10")
print(f"    VERDICT: {v2}")
print(f"\nR3  E4 on the pooled {len(pool)}:  slope {slope:+.3f} per e-6 of CV (se {slope_se:.3f}, "
      f"t {slope/slope_se:+.2f})")
print(f"    level {level:+.2f}e-6 (se {level_se:.2f})")
print(f"    thresholds: slope<=-0.60 bad-slope | |slope|<=0.40 level-shift")
print(f"    VERDICT: {v3}")

# the cleanest single contrast, named in w47b before the send
c_above = resid6("w36_ad199std_hybrid", 0.9701231283, 0.97114)
c_below = resid6("w36_ad197std_hybrid", 0.9701124257, 0.97114)
print(f"\nTHE NAMED PAIR (same era, same transform, differ only in CV region):")
print(f"    w36_ad199std_hybrid  ABOVE support  resid {c_above:+.2f}e-6")
print(f"    w36_ad197std_hybrid  INSIDE support resid {c_below:+.2f}e-6")
print(f"    split = {c_below - c_above:+.2f}e-6  (CV-REGION predicts a large positive split; "
      f"ERA predicts ~0)")

json.dump(dict(
    day="2026-08-22", n_sent=10,
    probes=[dict(stem=s, cv=cv, lb=lb, resid_e6=resid6(s, cv, lb)) for s, cv, lb in PROBES],
    drains=[dict(stem=s, cv=cv, lb=lb, resid_e6=resid6(s, cv, lb)) for s, cv, lb in DRAINS],
    R1=dict(p=p, sem=SEM, verdict=v1, action=act1),
    R2=dict(r5=r5, r3_excl_stdcorr=r3, verdict=v2),
    R3=dict(slope=slope, slope_se=slope_se, level=level, level_se=level_se,
            n=len(pool), verdict=v3),
    named_pair=dict(above=c_above, below=c_below, split=c_below - c_above),
), open(os.path.join(HERE, "w52a_read.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(HERE, 'w52a_read.json')}")
