"""w52d -- the CV->LB predictor, SLOPE-corrected.  SUPERSEDES w46c's level dummy.

⚠⚠ WHY THIS MODULE EXISTS.  w46c corrected w30b with a flat era LEVEL dummy of -29.82e-6 for
every ad>=195 file.  The 08-22 ten were designed by w47b to test exactly that, and they refute
the flat form.  The miss is NOT flat -- it grows with CV:

    the five PROBES (inside CV support, cv6 +47..+55)   mean residual -13.02e-6
    the five DRAINS (above CV support, cv6 +73..+80)    mean residual -31.91e-6

w47b's registered R3 read that as slope -0.675 per e-6 of CV (t -4.27) -> "BAD CV SLOPE, the
fix is a re-estimated slope, not a dummy".  w52b then fitted four forms on all 93 scored files
and compared them by LEAVE-ONE-DAY-OUT, which is honest held-out error:

    M0  no era term          full LODO RMSE 12.92e-6    era-slice RMSE 18.83e-6
    M1  era level dummy       "          9.59            "          12.82   <- w46c's form
    M2  era x cv interaction  "          8.69            "           8.77   <- THIS MODULE
    M3  both terms            "          9.21            "          11.30

M2 wins the full 12-day LODO outright and wins 08-22 (7.91 vs 11.86) clearly.

⚠ IT IS PROVISIONAL, AND THE HONEST CAVEAT IS THIS.  On the 08-21 day alone M3 edges M2,
10.11 vs 10.28e-6 -- a 0.17e-6 gap on n=5, which is a tie, not a disagreement.  w52c's
pre-set binary test still fired "the two days disagree", and that is reported rather than
smoothed away.  What is NOT in doubt across every cut: w46c's M1 dummy is beaten by both
slope-carrying forms, so the dummy form is wrong whichever of M2/M3 is right.

⚠ THE SEND PATH STILL PRICES UNDER w46c, DELIBERATELY.  `w26d_queueprice.predict()` adds
`W46.ERA_SHIFT` and then ASSERTS it reproduces w30b's residual sd; delegating it to M2 trips
that assert and would hard-fail the ritual inside `w48e_order.py`, which is the exact failure
w51 section 3 had to fix.  Rewiring it is a deliberate change for a run that has slots to
verify with.  It costs almost nothing to defer: the 08-23 order is a HARDCODED list, so the
pricer cannot reorder it, and w46b priced the whole free-rider ordering at +0.47e-6 total.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w46c_predlb as W46C

MU = W46C.MU
ERA_MIN_AD = W46C.ERA_MIN_AD
new_era, wave, family, is_std, is_corr = (W46C.new_era, W46C.wave, W46C.family,
                                          W46C.is_std, W46C.is_corr)

_t = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
_t = _t[_t.fam != "wh3"].reset_index(drop=True)          # singleton family, see w52c
FAMS = sorted(f for f in _t.fam.unique() if f != "h3")   # h3 is the reference level


def _design(cv6, fam, std, corr, era):
    row = [1.0, cv6] + [1.0 if fam == f else 0.0 for f in FAMS] + [float(std), float(corr)]
    row.append(float(era) * cv6)                          # M2: era x cv, NO level dummy
    return np.array(row)


_X = np.column_stack([np.ones(len(_t)), _t.cv6.values]
                     + [(_t.fam == f).astype(float).values for f in FAMS]
                     + [_t["std"].values, _t["corr"].values, (_t.era * _t.cv6).values])
_y = _t.lb6.values
BETA, *_ = np.linalg.lstsq(_X, _y, rcond=None)
NAMES = ["const", "cv6"] + [f"fam[{f}]" for f in FAMS] + ["std", "corr", "era:cv6"]
COEFS = dict(zip(NAMES, BETA))
_r = _y - _X @ BETA
RESID_SD = float(np.sqrt(float(_r @ _r) / (len(_t) - _X.shape[1])))
SD_ERA = 8.77      # w52c's HELD-OUT era-slice RMSE -- the honest predictive sd for era files
SD_OLD = W46C.SD_OLD

BASE_SLOPE = float(COEFS["cv6"])
ERA_SLOPE = float(COEFS["cv6"] + COEFS["era:cv6"])       # what an era file actually converts at


def predict_lb(cv, stem):
    cv6 = (cv - MU) * 1e6
    x = _design(cv6, family(stem), is_std(stem), is_corr(stem), new_era(stem))
    return float(x @ BETA) * 1e-6


def pred_sd(stem):
    return (SD_ERA if new_era(stem) else SD_OLD) * 1e-6


def cv_needed(target_lb, stem):
    """The CV an era file of this stem's family/flags would need to predict `target_lb`."""
    x0 = _design(0.0, family(stem), is_std(stem), is_corr(stem), new_era(stem))
    intercept = float(x0 @ BETA)
    slope = ERA_SLOPE if new_era(stem) else BASE_SLOPE
    return MU + ((target_lb * 1e6 - intercept) / slope) * 1e-6


if __name__ == "__main__":
    print(f"w52d M2 pricer: n={len(_t)}, in-sample resid sd {RESID_SD:.2f}e-6, "
          f"held-out era sd {SD_ERA:.2f}e-6")
    print(f"  base cv slope {BASE_SLOPE:+.3f} per e-6   era cv slope {ERA_SLOPE:+.3f} "
          f"({100*ERA_SLOPE/BASE_SLOPE:.0f}% of it)")

    ref = "w36_ad199stdcorr"                 # the WANTED deadline pick: h3, std, corr, era
    BEST_SENT_CV, BEST_LB = 0.9701400060, 0.97118
    print(f"\n  check: {ref} cv {BEST_SENT_CV:.10f} -> pred {predict_lb(BEST_SENT_CV, ref):.6f}"
          f"   actual {BEST_LB:.5f}")

    print(f"\n  WHAT THE BOARD NOW COSTS, in CV, for a file like {ref}")
    print(f"  (board read 2026-08-22 ~12:5x UTC: us 0.97118 at rank 61; "
          f"gold cut rank 14 = 0.97124; leader 0.97141)")
    print(f"  {'target LB':>10s} {'rank':>6s} {'CV needed':>14s} {'vs best sent CV':>16s} "
          f"{'vs best BUILT':>14s}")
    BEST_BUILT_CV = 0.9701500880             # w50_ad216stdcorr, the highest CV ever built here
    for tgt, rank in [(0.97119, "~58"), (0.97121, "~30"), (0.97122, "~20"),
                      (0.97124, "14 GOLD"), (0.97128, "10"), (0.97141, "1")]:
        need = cv_needed(tgt, ref)
        print(f"  {tgt:10.5f} {rank:>6s} {need:14.10f} {(need-BEST_SENT_CV)*1e6:+15.1f}e-6 "
              f"{(need-BEST_BUILT_CV)*1e6:+13.1f}e-6")
    print(f"\n  ⚠ Every row above the first is an EXTRAPOLATION beyond the fitted CV range "
          f"(era files span cv6 +47..+80, i.e. cv {MU+47e-6:.7f}..{MU+80e-6:.7f}).")
    print(f"  ⚠ Read it as 'the gap is this large in CV units', NOT as a promise that reaching "
          f"that CV would deliver that LB.")

    json.dump(dict(n=len(_t), coefs=COEFS, resid_sd=RESID_SD, sd_era=SD_ERA,
                   base_slope=BASE_SLOPE, era_slope=ERA_SLOPE, mu=MU,
                   supersedes="w46c_predlb (M1 level dummy)", provisional=True,
                   cv_needed={f"{t:.5f}": cv_needed(t, ref)
                              for t in (0.97119, 0.97121, 0.97122, 0.97124, 0.97128, 0.97141)},
                   best_sent_cv=BEST_SENT_CV, best_built_cv=BEST_BUILT_CV),
              open(os.path.join(HERE, "w52d_predlb.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE,'w52d_predlb.json')}")
