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


# The classifiers now live in `experiments/stdflag.py` and are IMPORTED, not copied.
#
# ⚠⚠ SECOND BUG IN THIS FLAG, FOUND 2026-08-19 (w28). w27 slot 3 replaced a hard-coded
# whitelist with the substring test `"std" in stem`, which was right on every row w25f fitted
# and WRONG on the six `w27_ad188raw*` files -- built by `w27k_ctrawctl.sh`, which passes
# `--standardize`; the "raw" is the CT-raw MEMBER, not a raw combiner. Each was handed the
# +27.43e-6 the `standardised` coefficient withholds, and `w27_ad188raw` consequently priced
# at P(beat the account best) = 0.548, the best number on disk. Corrected: 8.4e-4, and the
# whole ten-file day drops from P = 0.647 to 0.083. A filename is not provenance; `stdflag`
# derives the flag from the WAVE NUMBER, because `--standardize` entered the chain at w23 and
# every build since passes it. That module gates itself against w25f's own 60 rows.
sys.path.insert(0, HERE)
import stdflag  # noqa: E402
from stdflag import STD_FILES, family, is_std  # noqa: E402,F401

# w25f fits on CENTRED CV: cv_e6 = (cv - mu) * 1e6, mu the mean over its own 60 rows. mu is
# not in the JSON, so it is recomputed here from the same table under the same CV >= 0.97
# filter. Feeding the model raw CV instead gave a predicted LB of 2.82 on the first run --
# absurd enough to be caught on sight, which is exactly why the gate below now exists: the
# next parameterisation error will not be six orders of magnitude out and will not be obvious.
_t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
_fit = _t[_t.cv >= 0.97].copy()
MU = _fit.cv.mean()

# ⚠ THE MODEL IS w26e's REFIT, NOT w25f (changed w28, 2026-08-19). w26e found that the two
# `*corr` files -- one of which is `WANTED` slot 1 -- were labelled `ens4` by the suffix rule
# when they are corrected **h3** mixes, refitted with only those two labels changed, and got a
# LOWER residual sd (8.349e-6 vs 8.408e-6) with no extra parameters. That refit then sat unused
# for two days while the pricer kept the wrong labels. It matters most exactly where the
# decision is: the queue's top two files are BOTH `*corr`, so under the suffix rule each was
# collecting the +13.9e-6 `ens4` term on top of an h3 base.
# ⚠ THE MODEL IS w30b, NOT w26e (changed w30, 2026-08-20). The 08-20 ten-file drain was the
# first genuinely HELD-OUT test this predictor has ever had -- ten files priced before upload
# under a fit containing none of them (w30a). It came back unbiased overall (mean residual
# +1.67e-6, z +0.60) but with the two `*stdcorr` files +19.3e-6 high, z +3.08. w30b refits with
# a c_avg-CORRECTION indicator, which w26e had no term for, and gets +12.61e-6 (se 4.07,
# t +3.10) with the residual sd falling 8.23 -> 7.76e-6 -- BELOW the 8.70e-6 slice+grid noise
# floor, i.e. the CV->LB relation is now fully accounted for. Family coefficients barely move,
# so this is not the h3 term in disguise: the six corrected files span h3, ens4 AND rankraw.
# ⚠ CAVEAT, the reason to read the corr term as ~+10e-6 rather than +12.6: it survives a
# quadratic-CV guard (+9.92, t +2.12) and the top-CV band (+15.71, t +3.37) but NOT the drop of
# today's two near-twins (+7.57, t +1.59). Positive in all four cuts, significant in three.
M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C, RESID = M["coefs"], M["resid_sd_new"]
BEST_LB = 0.97118          # account best, w21_ad187corr_ens4
STEP = 1e-5                # the LB reports to 5 decimals


def is_corr(stem):
    """Carries the c_avg / scheme-average correction. Every such build in this workspace has
    `corr` in its stem and nothing else does; stdflag.CORR_MAP enforces registration."""
    return "corr" in stem


def predict(cv, fam, standardised, corrected=False):
    """Predicted public LB, in absolute AUC. `standardised` is the combiner-scaling flag,
    `corrected` the c_avg-correction flag (w30b)."""
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{fam}]", 0.0)
    if standardised:
        lb6 += C["std"]
    if corrected:
        lb6 += C["corr"]
    return lb6 * 1e-6


# GATE. Re-predict the 60 files the model was fitted on and require its residual sd back.
_fit["fam"] = _fit.stem.map(family)
_fit["std"] = _fit.stem.map(is_std)
_fit["corr"] = _fit.stem.map(is_corr)
_r6 = (_fit.lb.values - np.array([predict(r.cv, r.fam, r.std, r.corr)
                                  for r in _fit.itertuples()])) * 1e6
# w25f divides the residual sum of squares by its DOF (n - p), not by n. Using np.std's
# default ddof=0 here read 7.68e-6 against its 8.41e-6 and failed this gate on the first
# run -- a real 9% discrepancy, not a rounding one, and precisely the kind of quiet
# parameterisation slip the gate exists to catch. sqrt(60/50) = 1.095 accounts for it exactly.
_dof = len(_fit) - len(C)
_rsd = float(np.sqrt((_r6 ** 2).sum() / _dof))
print(f"GATE: refitted-sample residual sd {_rsd:.2f}e-6 (dof {_dof}) vs w25f's "
      f"{RESID:.2f}e-6 -- {'PASS' if abs(_rsd - RESID) < 0.5 else 'FAIL'}")
assert abs(_rsd - RESID) < 0.5, "predict() does not reproduce w30b; nothing below is readable"


q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
q = q[~q.sent].dropna(subset=["cv"]).copy()
q["stem"] = q.file.str.replace(".csv", "", regex=False)
stdflag.require_corr_registered(q.stem)   # a new *corr file must be classified BY HAND
q["fam"] = q.stem.map(family)
q["std"] = q.stem.map(is_std)          # see is_std: a whitelist here was a BUG
q["corr"] = q.stem.map(is_corr)
q["pred_lb"] = [predict(r.cv, r.fam, r.std, r.corr) for r in q.itertuples()]

# P(beat the account best). The file must print STRICTLY above 0.97118, and the LB rounds to
# 1e-5, so the target on the underlying scale is BEST_LB + STEP/2.
q["p_beat"] = 1.0 - norm.cdf((BEST_LB + STEP / 2 - q.pred_lb) / (RESID * 1e-6))

# PRIORITY. The queue is ranked on predicted PUBLIC LB, which is the right order for chasing
# public rank and the WRONG order for the one thing a submission is actually needed for here:
# Kaggle's final-selection dialog lists SUBMITTED entries only, so a deadline pick that is
# never sent cannot be ticked. `w27_ad188stdcorr` sat unsent for four slots behind three
# journal entries saying "send it first" because the sender reads this file and this file did
# not rank it first. Anything in `check_selection.WANTED` now goes to the head of the queue.
from check_selection import WANTED  # noqa: E402
q["priority"] = q.file.isin(WANTED).astype(int)
q = q.sort_values(["priority", "pred_lb"], ascending=[False, False])
_pin = q[q.priority == 1].file.tolist()
print(f"PINNED to the head of the queue (check_selection.WANTED, unsent): {_pin or 'none'}")

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
#
# ⚠ This used to omit the `standardised` term, so every bar it printed was the bar for an
# UNSTANDARDISED file — and every build this workspace has made since w23 is standardised. The
# quoted "ens4 bar is 0.9701108460, 4.2e-6 BELOW the current CV leader" is that omission: the
# real bar for the standardised ens4 files we actually build is 0.9701252, +10.1e-6 ABOVE the
# leader. Both are printed now so the assumption can never be dropped again.
need = (BEST_LB + STEP / 2) * 1e6
# The CV leader is read LIVE off the queue, never quoted -- it was hard-coded at 0.9701150809
# (the w23 leader) for three waves after two better files existed, so every "vs leader" figure
# printed here between w27 and w29 was stale by 3.0e-6.
LEADER = float(max(q.cv.max(), _fit.cv.max()))
print(f"\nCV needed for an even-money shot at {BEST_LB:.5f}, BY FAMILY AND BY SCALING")
print(f"  (leader {LEADER:.10f})")
print(f"  {'family':>7s}  {'unstd':>16s}  {'std':>16s}  {'std+corr':>16s}   {'gap':>10s}")
for fam in ("h3", "ens4", "hybrid", "rankraw", "rescale", "logit"):
    base = (need - C["const"] - C.get(f"fam[{fam}]", 0.0)) / C["cv_e6"] * 1e-6 + MU
    std = base - C["std"] / C["cv_e6"] * 1e-6
    stdc = std - C["corr"] / C["cv_e6"] * 1e-6
    print(f"  {fam:>7s}  {base:16.10f}  {std:16.10f}  {stdc:16.10f}   "
          f"{(stdc - LEADER)*1e6:+8.1f}e-6")
print("  Every file this workspace builds is standardised, and the ones worth building are")
print("  ALSO corrected -- read the std+corr column. `gap` is what a NEW build must add to")
print("  the CV leader to be an even-money shot at the board; negative means already there.")

q.to_csv(os.path.join(HERE, "w26d_queueprice.csv"), index=False)
json.dump(dict(n=len(q), resid_sd=RESID, best_lb=BEST_LB, p_best_single=float(pbest),
               p_any_of_ten=float(p_any), pred_lb_best=float(q.pred_lb.iloc[0])),
          open(os.path.join(HERE, "w26d_queueprice.json"), "w"), indent=1)
print("\nwrote experiments/w26d_queueprice.{csv,json}")
