"""w32a -- what CV does a MEDAL cost?  The bar table, inverted against the BOARD.

w26d inverts the CV->LB model at exactly one target: `BEST_LB = 0.97118`, the account's own
best.  That is the right target for "is this queue file worth believing", and it is the wrong
target for "is another week of building worth it", which is the question 11 days from the
deadline.  This file inverts the SAME model (w30b, the one w30a validated out-of-sample on
ten held-out files) at the board's own thresholds.

⚠ IT ALSO CORRECTS A UNITS ERROR THAT HAS BEEN IN LEADERBOARD.md SINCE 08-19.  Two entries
say "the gap to first is 16e-6" and one adds "which the CV->LB model says needs a CV of
0.9701307114".  Both are wrong and they are wrong in different ways:

    0.97134 - 0.97118 = 0.00016 = 160e-6 = SIXTEEN reporting steps, not 1.6.
    0.9701307114 is +12.4e-6 of CV on the leader, and at the fitted slope 1.856 that buys
    +23e-6 of LB -- i.e. it is the bar for ~0.97120, the GOLD cutoff, not for first place.

The two numbers were never consistent with each other.  Nothing downstream depended on them
(w26d prices against the account's own 0.97118 and is self-consistent), so no build decision
was made on the error -- but "we are 1.6 steps off the lead" and "we are 16 steps off the
lead" are different strategies, and only one of them is true.

    .venv/bin/python experiments/w32a_goldbar.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stdflag  # noqa: E402
from stdflag import family, is_std  # noqa: E402

STEP = 1e-5          # the public LB reports to 5 decimals
OURS = 0.97118       # account best, held three ways, all c_avg-corrected

M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C, RESID = M["coefs"], M["resid_sd_new"]

# Same centring as w26d: mu is recomputed from the same table under the same CV >= 0.97
# filter, because it is not stored in the JSON.
_t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
_fit = _t[_t.cv >= 0.97].copy()
MU = _fit.cv.mean()


def is_corr(stem):
    return "corr" in stem


def predict(cv, fam, standardised, corrected=False):
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{fam}]", 0.0)
    if standardised:
        lb6 += C["std"]
    if corrected:
        lb6 += C["corr"]
    return lb6 * 1e-6


def invert(target_lb, fam, standardised, corrected):
    """CV at which predict() equals `target_lb`."""
    lb6 = target_lb * 1e6 - C.get(f"fam[{fam}]", 0.0)
    if standardised:
        lb6 -= C["std"]
    if corrected:
        lb6 -= C["corr"]
    return (lb6 - C["const"]) / C["cv_e6"] * 1e-6 + MU


# ---------------------------------------------------------------- GATE (copied from w26d)
# The model must reproduce its own fitted residual sd, or every bar below is unreadable.
_fit["fam"] = _fit.stem.map(family)
_fit["std"] = _fit.stem.map(is_std)
_fit["corr"] = _fit.stem.map(is_corr)
_r6 = (_fit.lb.values - np.array([predict(r.cv, r.fam, r.std, r.corr)
                                  for r in _fit.itertuples()])) * 1e6
_dof = len(_fit) - len(C)
_rsd = float(np.sqrt((_r6 ** 2).sum() / _dof))
ok = abs(_rsd - RESID) < 0.5
print(f"GATE predict(): residual sd {_rsd:.2f}e-6 (dof {_dof}) vs stored {RESID:.2f}e-6 "
      f"-- {'PASS' if ok else 'FAIL'}")
assert ok, "predict() does not reproduce w30b"

# GATE 2, the inversion is the exact inverse of the forward map.
for fam in ("h3", "ens4", "logit"):
    for s in (True, False):
        for c in (True, False):
            cv = invert(0.97125, fam, s, c)
            assert abs(predict(cv, fam, s, c) - 0.97125) < 1e-12
print("GATE invert(): exact inverse of predict() on 12 (family, std, corr) cells -- PASS")

LEADER_CV = float(_fit.cv.max())
print(f"\nCV leader on disk: {LEADER_CV:.10f}  "
      f"({_fit.loc[_fit.cv.idxmax(), 'stem']})")

# ---------------------------------------------------------------- the board
lb = pd.read_csv(os.path.join(HERE, "w32_lb_top.csv"), skiprows=1)
lb["rank"] = np.arange(1, len(lb) + 1)
ours = lb[lb.teamName.str.contains("Teddy", case=False, na=False)]
our_rank = int(ours["rank"].iloc[0]) if len(ours) else -1
print(f"board read: {len(lb)} rows, we are rank {our_rank} at {OURS:.5f}, "
      f"leader {lb.score.iloc[0]:.5f}")


def rank_at(score):
    """Rank this score would hold on the board as read (ties broken in our favour)."""
    return int((lb.score > score + 1e-12).sum()) + 1


# ---------------------------------------------------------------- THE TABLE
# Every build this workspace makes is standardised, and the ones worth building are also
# c_avg-corrected, so std+corr is the only column that describes a real candidate. The
# unstandardised column is printed only so the assumption can never be silently dropped.
print("\n" + "=" * 78)
print("WHAT A MEDAL COSTS IN CV -- w30b inverted at the board's thresholds")
print("=" * 78)
print(f"  slope dLB/dCV = {C['cv_e6']:.4f} +/- {M['ses']['cv_e6']:.4f}  "
      f"=> one reporting step (10e-6 of LB) = {10/C['cv_e6']:.2f}e-6 of CV")
print(f"  residual sd {RESID:.2f}e-6 of LB = {RESID/C['cv_e6']:.2f}e-6 of CV\n")

targets = [
    (0.97119, "one step up"),
    (0.97120, "GOLD cutoff (rank ~14 today)"),
    (0.97121, "gold, one clear"),
    (0.97125, "rank ~4"),
    (0.97134, "MILANFX, rank 1"),
]
print(f"  {'target LB':>10s} {'rank':>5s}  {'dLB':>8s}  "
      f"{'CV needed (h3 std+corr)':>24s} {'vs leader':>11s}  what it is")
for t, label in targets:
    need = (t + STEP / 2)          # must PRINT at t, so clear the rounding boundary
    cv_h3 = invert(need, "h3", True, True)
    cv_e4 = invert(need, "ens4", True, True)
    best = min(cv_h3, cv_e4)
    print(f"  {t:10.5f} {rank_at(t):5d}  {(t-OURS)*1e6:+7.0f}e-6  {best:24.10f} "
          f"{(best - LEADER_CV)*1e6:+9.1f}e-6  {label}")
print("  (CV column is the cheaper of the h3 and ens4 families; ens4 carries +13.7e-6 of")
print("   family term, so it is the cheaper one everywhere.)")

print("\n  FULL FAMILY BREAKDOWN, std+corr, at the gold cutoff 0.97120 and at rank 1")
print(f"  {'family':>8s}  {'CV for 0.97120':>16s} {'gap':>10s}   "
      f"{'CV for 0.97134':>16s} {'gap':>10s}")
for fam in ("ens4", "rankraw", "h3", "hybrid", "rescale", "logit"):
    a = invert(0.97120 + STEP / 2, fam, True, True)
    b = invert(0.97134 + STEP / 2, fam, True, True)
    print(f"  {fam:>8s}  {a:16.10f} {(a-LEADER_CV)*1e6:+8.1f}e-6   "
          f"{b:16.10f} {(b-LEADER_CV)*1e6:+8.1f}e-6")

# ---------------------------------------------------------------- extrapolation honesty
span = (_fit.cv.max() - _fit.cv.min()) * 1e6
gold = invert(0.97120 + STEP / 2, "ens4", True, True)
first = invert(0.97134 + STEP / 2, "ens4", True, True)
print("\n  EXTRAPOLATION CHECK. The fit spans %.0fe-6 of CV." % span)
print("    gold  is %+.1fe-6 past the CV leader = %.0f%% of the fitted span -- interpolative."
      % ((gold - LEADER_CV) * 1e6, (gold - LEADER_CV) * 1e6 / span * 100))
print("    rank1 is %+.1fe-6 past the CV leader = %.0f%% of the fitted span -- and the"
      % ((first - LEADER_CV) * 1e6, (first - LEADER_CV) * 1e6 / span * 100))
print("    slope's own +/-%.4f (%.1f%%) puts a +/-%.0fe-6 band on that CV alone."
      % (M['ses']['cv_e6'], M['ses']['cv_e6'] / C['cv_e6'] * 100,
         abs((first - LEADER_CV) * 1e6) * M['ses']['cv_e6'] / C['cv_e6']))

# ---------------------------------------------------------------- what has ever bought that
print("\n  FOR SCALE -- the largest CV gains this workspace has ever realised, from JOURNAL:")
for what, val in [("the 22-member adarsh import (w20)", 47.0),
                  ("the c_avg correction (8 readings)", 5.1),
                  ("standardising the combiner (w23)", 8.0),
                  ("adding one good new member", 3.4)]:
    print(f"    {what:38s} {val:+6.1f}e-6")
print(f"    gold needs {(gold - LEADER_CV)*1e6:+.1f}e-6 -- reachable: it is 2-4 corrections,")
print(f"    or one good member family.  Rank 1 needs {(first - LEADER_CV)*1e6:+.1f}e-6 --")
print("    ~1.8x the ENTIRE adarsh import, the single biggest jump in this workspace's")
print("    history.  No stacking tweak reaches it; it needs a better base model.")

out = dict(
    slope=C["cv_e6"], slope_se=M["ses"]["cv_e6"], resid_sd=RESID, leader_cv=LEADER_CV,
    our_lb=OURS, our_rank=our_rank, leader_lb=float(lb.score.iloc[0]),
    cv_per_step=10 / C["cv_e6"],
    gold_lb=0.97120, gold_cv_ens4=gold, gold_gap_e6=(gold - LEADER_CV) * 1e6,
    first_lb=0.97134, first_cv_ens4=first, first_gap_e6=(first - LEADER_CV) * 1e6,
    fitted_cv_span_e6=span,
)
json.dump(out, open(os.path.join(HERE, "w32a_goldbar.json"), "w"), indent=1)
print("\nwrote experiments/w32a_goldbar.json")
