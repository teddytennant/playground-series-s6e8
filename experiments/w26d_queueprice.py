"""w26d — price the ENTIRE unsent send queue under the w25f model, before spending a day on it.

The brief says an unused slot is pure waste and to send all ten every day, because submissions
here never evict each other and the public LB shows best-of-all. That is still true. What it
does NOT say is that the queue is inexhaustible, and this slot noticed that it very nearly is
exhausted: `w23b_sendqueue` reports the best CV among ALREADY-SENT files as 0.9701150809 and
the best CV among the 48 unsent ones as 0.9700483189 -- 67e-6 BELOW it. Every remaining
candidate is materially worse on CV than something already on the board.

w25f now lets that be priced rather than guessed. It fits, over the 60 scored files with
CV >= 0.97,  LB = const + 1.909*CV_e6 + family + standardised,  with residual sd 8.41e-6
against an independently simulated slice-noise floor of 8.21e-6 -- i.e. the fit is complete
and the residual is pure slice draw. So for each unsent file we can state a predicted LB and,
with the residual as the only remaining uncertainty, P(this file beats the account best).

That probability is the number that decides how to spend the 08-19 Kaggle day. It cannot make
sending harmful -- the brief's economics are sound and a send still cannot hurt public rank --
but it distinguishes "drain the queue" from "build something above 0.9701150809 first", and
only one of those is worth a run's compute.

    .venv/bin/python experiments/w26d_queueprice.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


# NOT imported from w25b_gapfamily. That module runs its whole analysis at import time and
# rewrites w25b_{pairs,families}.csv and w25b_gapfamily.json as a side effect -- the first
# version of this script did import it and silently overwrote all three. Caught by
# `git status`, restored with `git checkout --`. Same class of near-miss as w24's hard-coded
# JSON path, and the second time this workspace has had it. Copied verbatim instead; if the
# classifier ever changes, BOTH copies must change.
def family(s):
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit"):
        if s.endswith("_" + f):
            return f
    if s.endswith("_w") or s.endswith("_w2") or s in ("blend156w", "blend156w2"):
        return "w"
    return "ens4"          # a bare stem is the rank-average of all four transforms


STD_FILES = {"w23_ad187stdcorr", "w23_ad187std_h3", "w23_ad187std", "w23_ad187std_logit",
             "w23_ad187std_h3_hybrid", "w23_ad187std_h3_rankraw", "w23_ad187std_h3_rescale"}

# w25f fits on CENTRED CV: cv_e6 = (cv - mu) * 1e6, mu the mean over its own 60 rows. mu is
# not in the JSON, so it is recomputed here from the same table under the same CV >= 0.97
# filter. Feeding the model raw CV instead gave a predicted LB of 2.82 on the first run --
# absurd enough to be caught on sight, which is exactly why the gate below now exists: the
# next parameterisation error will not be six orders of magnitude out and will not be obvious.
_t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
_fit = _t[_t.cv >= 0.97].copy()
MU = _fit.cv.mean()

M = json.load(open(os.path.join(HERE, "w25f_ancova2.json")))
C, RESID = M["coefs"], M["resid_sd_with_std"]
BEST_LB = 0.97118          # account best, w21_ad187corr_ens4
STEP = 1e-5                # the LB reports to 5 decimals


def predict(cv, fam, standardised):
    """Predicted public LB, in absolute AUC. `standardised` is the combiner-scaling flag."""
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{fam}]", 0.0)
    if standardised:
        lb6 += C["standardised"]
    return lb6 * 1e-6


# GATE. Re-predict the 60 files the model was fitted on and require its residual sd back.
_fit["fam"] = _fit.stem.map(family)
_fit["std"] = _fit.stem.isin(STD_FILES)
_r6 = (_fit.lb.values - np.array([predict(r.cv, r.fam, r.std)
                                  for r in _fit.itertuples()])) * 1e6
# w25f divides the residual sum of squares by its DOF (n - p), not by n. Using np.std's
# default ddof=0 here read 7.68e-6 against its 8.41e-6 and failed this gate on the first
# run -- a real 9% discrepancy, not a rounding one, and precisely the kind of quiet
# parameterisation slip the gate exists to catch. sqrt(60/50) = 1.095 accounts for it exactly.
_dof = len(_fit) - len(C)
_rsd = float(np.sqrt((_r6 ** 2).sum() / _dof))
print(f"GATE: refitted-sample residual sd {_rsd:.2f}e-6 (dof {_dof}) vs w25f's "
      f"{RESID:.2f}e-6 -- {'PASS' if abs(_rsd - RESID) < 0.5 else 'FAIL'}")
assert abs(_rsd - RESID) < 0.5, "predict() does not reproduce w25f; nothing below is readable"


q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
q = q[~q.sent].dropna(subset=["cv"]).copy()
q["stem"] = q.file.str.replace(".csv", "", regex=False)
q["fam"] = q.stem.map(family)
q["std"] = q.stem.isin(STD_FILES)      # the same explicit list w25f fitted on
q["pred_lb"] = [predict(r.cv, r.fam, r.std) for r in q.itertuples()]

# P(beat the account best). The file must print STRICTLY above 0.97118, and the LB rounds to
# 1e-5, so the target on the underlying scale is BEST_LB + STEP/2.
q["p_beat"] = 1.0 - norm.cdf((BEST_LB + STEP / 2 - q.pred_lb) / (RESID * 1e-6))
q = q.sort_values("pred_lb", ascending=False)

print(f"{len(q)} unsent files with a stored OOF vector, priced under w25f "
      f"(resid sd {RESID:.2f}e-6, centred at mu={MU:.10f})\n")
print("TOP 12 BY PREDICTED PUBLIC LB")
print(f"{'file':32s} {'fam':8s} {'std':5s} {'cv':>14s} {'pred LB':>9s} {'P(>best)':>9s}")
for r in q.head(12).itertuples():
    print(f"{r.file:32s} {r.fam:8s} {str(r.std):5s} {r.cv:.10f} {r.pred_lb:9.5f} "
          f"{r.p_beat:9.2e}")

pbest = q.p_beat.max()
# P(at least one of the ten best beats it), treating the residuals as independent draws.
# They are NOT independent -- all ten are read off ONE fixed public slice and share member
# sets -- so this OVERSTATES the chance, which is the safe direction for the conclusion.
ten = q.head(10)
p_any = 1.0 - np.prod(1.0 - ten.p_beat.values)

print(f"\nbest single file          P(beat {BEST_LB}) = {pbest:.3e}")
print(f"best TEN sent together    P(at least one)  = {p_any:.3e}   "
      f"(independence assumed -> an OVERSTATEMENT)")
print(f"\npredicted LB of the best unsent file: {q.pred_lb.iloc[0]:.5f}   "
      f"= {(q.pred_lb.iloc[0]-BEST_LB)*1e6:+.1f}e-6 vs the account best "
      f"({(q.pred_lb.iloc[0]-BEST_LB)/STEP:+.1f} reporting steps)")

# What CV would a NEW file need for an even-money shot at the record?
need = (BEST_LB + STEP / 2) * 1e6
for fam in ("h3", "ens4"):
    cv_need = (need - C["const"] - C.get(f"fam[{fam}]", 0.0)) / C["cv_e6"] * 1e-6 + MU
    print(f"CV needed for P(beat)=0.50 in family {fam:5s}: {cv_need:.10f}  "
          f"({(cv_need - 0.9701150809)*1e6:+.1f}e-6 on the current CV leader)")

q.to_csv(os.path.join(HERE, "w26d_queueprice.csv"), index=False)
json.dump(dict(n=len(q), resid_sd=RESID, best_lb=BEST_LB, p_best_single=float(pbest),
               p_any_of_ten=float(p_any), pred_lb_best=float(q.pred_lb.iloc[0])),
          open(os.path.join(HERE, "w26d_queueprice.json"), "w"), indent=1)
print("\nwrote experiments/w26d_queueprice.{csv,json}")
