"""w32b -- does the FITTED model re-rank the account's files against raw CV, and does that
touch the final pick?

w32a's inversion makes one thing unavoidable: the w30b model does not just add a constant to
CV, it adds a FAMILY term worth up to +147e-6 of LB (logit) -- 27 standard errors, and
validated out of sample by w30a on ten held-out files. If that term is a train->TEST property
it applies to the private slice as much as the public one, and the workspace's final pick,
which is made on RAW CV, is then ranking files on a quantity the model says is biased.

This file does NOT move the pick. It states the size of the disagreement, and pre-registers
what would have to be true for the pick to move.

    .venv/bin/python experiments/w32b_famrank.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stdflag import family, is_std  # noqa: E402
from check_selection import WANTED  # noqa: E402

M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C, RESID = M["coefs"], M["resid_sd_new"]
t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
fit = t[t.cv >= 0.97].copy()
MU = fit.cv.mean()

fit["fam"] = fit.stem.map(family)
fit["std"] = fit.stem.map(is_std)
fit["corr"] = fit.stem.str.contains("corr")
fit["fit_lb"] = (C["const"] + C["cv_e6"] * (fit.cv - MU) * 1e6
                 + fit.fam.map(lambda f: C.get(f"fam[{f}]", 0.0))
                 + fit["std"] * C["std"] + fit["corr"] * C["corr"]) * 1e-6
fit["resid_e6"] = (fit.lb - fit.fit_lb) * 1e6

print(f"{len(fit)} scored files with CV >= 0.97, ranked two ways\n")
print("TOP 12 BY RAW CV  (what check_selection.WANTED is chosen on)")
print(f"  {'#':>2s} {'stem':30s} {'fam':8s} {'cv':>14s} {'LB':>8s} {'fit LB':>9s} {'resid':>7s}")
by_cv = fit.sort_values("cv", ascending=False)
for i, r in enumerate(by_cv.head(12).itertuples(), 1):
    star = " <-WANTED" if f"{r.stem}.csv" in WANTED else ""
    print(f"  {i:2d} {r.stem:30s} {r.fam:8s} {r.cv:.10f} {r.lb:8.5f} {r.fit_lb:9.5f} "
          f"{r.resid_e6:+6.1f}{star}")

print("\nTOP 12 BY FITTED TEST AUC  (CV + family + std + corr; residual = slice draw)")
print(f"  {'#':>2s} {'stem':30s} {'fam':8s} {'cv':>14s} {'LB':>8s} {'fit LB':>9s} {'dCV':>8s}")
by_fit = fit.sort_values("fit_lb", ascending=False)
lead_cv = by_cv.cv.iloc[0]
for i, r in enumerate(by_fit.head(12).itertuples(), 1):
    star = " <-WANTED" if f"{r.stem}.csv" in WANTED else ""
    print(f"  {i:2d} {r.stem:30s} {r.fam:8s} {r.cv:.10f} {r.lb:8.5f} {r.fit_lb:9.5f} "
          f"{(r.cv-lead_cv)*1e6:+7.1f}{star}")

# Do the two orderings agree at the top?
top_cv = list(by_cv.head(2).stem)
top_fit = list(by_fit.head(2).stem)
print(f"\n  CV top-2   : {top_cv}")
print(f"  fitted top-2: {top_fit}")
print(f"  AGREE: {set(top_cv) == set(top_fit)}")
print(f"  spearman(cv rank, fitted rank) over all {len(fit)} = "
      f"{fit.cv.rank().corr(fit.fit_lb.rank(), method='spearman'):+.4f}")

# ------------------------------------------------------------------ the family terms
print("\nFAMILY TERMS, and how many files each is measured on")
print(f"  {'family':>8s} {'n':>3s} {'coef e-6':>9s} {'se':>6s} {'t':>6s} "
      f"{'= e-6 of CV':>12s} {'cv range of the fam':>22s}")
for fam, g in fit.groupby("fam"):
    k = C.get(f"fam[{fam}]", 0.0)
    se = M["ses"].get(f"fam[{fam}]", 0.0)
    tt = k / se if se else float("nan")
    print(f"  {fam:>8s} {len(g):3d} {k:+9.2f} {se:6.2f} {tt:+6.2f} "
          f"{k/C['cv_e6']:+12.1f} {(g.cv.min()-0.97)*1e6:8.1f}..{(g.cv.max()-0.97)*1e6:.1f}e-6")

print("""
  ⚠ HOW TO READ THIS, and why it is NOT licence to move the pick on public feedback.

  A family term CANNOT be a public-slice draw. The slice noise floor is 8.70e-6 and a term
  measured on n files has a slice-draw sd of ~8.7/sqrt(n); logit's +147e-6 on n=%d is %.0f
  such sds. So the family effect is a property of the TRAIN->TEST map, not of which rows
  Kaggle put in the public half, and it therefore applies to the private half too.

  What that means concretely: raw CV mis-ranks ACROSS families. It is fine WITHIN one.

  ⛔ AND THE PICK STILL DOES NOT MOVE THIS SLOT. Two reasons, both pre-registered here:
    (1) The mechanism is unidentified. "CV understates logit by 79e-6 of CV" is a fitted
        coefficient, not an explanation. Until there is a story for WHY, acting on it is
        curve-fitting to 78 points, which is the Rogii shape even though the arithmetic is
        different from Rogii's.
    (2) The test that would identify it is CV-SIDE and has never been run: refit each
        transform family on a genuine held-out TRAIN split (not the frozen folds the member
        OOF vectors were produced on) and see whether the family ordering flips there too.
        If it does, the effect is combiner overfitting to the shared folds -- a train
        artefact visible without ever touching the LB -- and the fitted ranking is right.
        If it does not, the effect is test-set specific and must not be acted on.

  REGISTERED CRITERION for a future slot: the pick moves to the fitted ranking only if that
  held-out-train experiment reproduces the family ordering with the same SIGN and at least
  half the magnitude. Nothing on the public LB, at any score, is sufficient on its own.
""" % (len(fit[fit.fam == "logit"]), 147.1 / (8.70 / np.sqrt(len(fit[fit.fam == "logit"])))))

json.dump(dict(n=len(fit), top_cv=top_cv, top_fit=top_fit,
               agree=bool(set(top_cv) == set(top_fit)),
               spearman=float(fit.cv.rank().corr(fit.fit_lb.rank(), method="spearman")),
               wanted=sorted(WANTED)),
          open(os.path.join(HERE, "w32b_famrank.json"), "w"), indent=1)
print("wrote experiments/w32b_famrank.json")
