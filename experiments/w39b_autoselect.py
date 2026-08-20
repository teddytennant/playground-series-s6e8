"""w39 consolidation -- WHAT WILL KAGGLE AUTO-SELECT, once the 08-21 queue drains?

The workspace has called the final-selection toggle "the highest-value open item" since
w13. It is a manual browser action, re-verified blocked again this run (no logged-in
profile anywhere on this box). So the question that actually matters is not "can I click
it" but: **given that Kaggle will auto-select the best PUBLIC submissions, how far from
the CV pick will that land?**

That was a live danger when the exposure was `blend158_logit` at -88e-6 of CV sitting in
the best-public tie. It may not be one now: every priority-1 file in the send queue is a
high-CV stack, and the CV LEADER itself is about to be sent for the first time.

Instrument: the w30b CV->LB model already in w26d_queueprice (fitted on the 60 >=0.97
sent files, residual sd RESID). Sent files contribute their OBSERVED public score; queued
files contribute a draw. Because the public slice is FIXED, its common component is
absorbed by the fitted intercept, so a new file's deviation from the fit is idiosyncratic
only. w17a's paired reading (within-family rms(dLB-dCV) = 7.4e-6 against a marginal
8.41e-6) implies rho ~ 0.61 between file residuals, i.e. an idiosyncratic sd of
RESID*sqrt(1-rho) ~ 5.2e-6. Both are reported; the truth is bracketed by them.

Read-out: the CV of whatever Kaggle would auto-select, versus the CV of the deadline pick.
That gap -- not the toggle -- is the real exposure.
"""
from __future__ import annotations

import contextlib
import io
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.sys.path.insert(0, HERE)

# w26d prints a lot on import; it also runs its own reproduction GATE, which we want.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import w26d_queueprice as QP  # noqa: E402
gate = [l for l in buf.getvalue().splitlines() if l.startswith("GATE")]
print(gate[0] if gate else "GATE line not found")

RESID = QP.RESID
RHO = 1.0 - (7.4 ** 2) / (2.0 * RESID ** 2)          # w17a paired reading
SD_IDIO = RESID * np.sqrt(max(1.0 - RHO, 0.0))
print(f"marginal resid sd {RESID:.2f}e-6 | implied rho {RHO:.3f} | idiosyncratic sd "
      f"{SD_IDIO:.2f}e-6\n")

full = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv", "lb"])
q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
q = q[(q.priority == 1) & (~q.sent.astype(str).str.lower().eq("true"))].copy()
q = q.sort_values("send_rank")
print(f"sent files with a public score: {len(full)}   |   queued for the 08-21 drain: {len(q)}")

CV = dict(zip(full.stem, full.cv))
LB = dict(zip(full.stem, full.lb))
for r in q.itertuples():
    CV[r.stem] = r.cv

pred = {r.stem: QP.predict(r.cv, r.fam, bool(r.std), bool(r.corr)) for r in q.itertuples()}
LEADER = max(CV, key=lambda s: CV[s])
print(f"CV leader overall: {LEADER}  cv {CV[LEADER]:.10f}  "
      f"({'SENT' if LEADER in LB else 'UNSENT -> not selectable until it is sent'})\n")

# best public today, and its CV
best_lb = max(LB.values())
holders = sorted([s for s in LB if LB[s] == best_lb], key=lambda s: -CV[s])
print(f"best public today: {best_lb:.5f} held by {len(holders)} file(s):")
for s in holders:
    print(f"   {s:28s} cv {CV[s]:.10f}   ({(CV[s]-CV[LEADER])*1e6:+7.2f}e-6 vs the CV leader)")

REPS = 20000
# Kaggle's final-submission limit is NOT exposed by the API -- GetCompetition returns
# deadline / maxDailySubmissions / maxTeamSize / teamCount and nothing about selection
# (re-checked w39). 2 is Kaggle's standing default for Playground; 1 is run as the
# pessimistic sensitivity so the conclusion does not rest on an unconfirmed constant.
LIMITS = (2, 1)
rng = np.random.default_rng(3939)
stems_q = list(q.stem)
mu = np.array([pred[s] for s in stems_q])
sent_stems = list(LB)
sent_lb = np.array([LB[s] for s in sent_stems])

out = {}
for LIMIT in LIMITS:
  print(f"\n{'='*72}\nFINAL-SUBMISSION LIMIT = {LIMIT}\n{'='*72}")
  for tag, sd in (("idiosyncratic", SD_IDIO), ("marginal", RESID)):
      draws = mu[None, :] + rng.normal(0.0, sd * 1e-6, size=(REPS, len(stems_q)))
      draws = np.round(draws, 5)                       # the board prints 5 dp
      lead_top = 0
      lead_top2 = 0
      picked_cv_gap = np.empty(REPS)
      worst_cv_gap = np.empty(REPS)
      for i in range(REPS):
          allst = sent_stems + stems_q
          alllb = np.concatenate([sent_lb, draws[i]])
          # ties broken by CV is OPTIMISTIC and by anti-CV is PESSIMISTIC; Kaggle's real
          # tiebreak is undocumented, so break at random -- the honest middle.
          order = np.lexsort((rng.random(len(allst)), -alllb))
          top = [allst[j] for j in order[:LIMIT]]
          lead_top += top[0] == LEADER
          lead_top2 += LEADER in top
          gaps = [(CV[s] - CV[LEADER]) * 1e6 for s in top]
          picked_cv_gap[i] = max(gaps)
          worst_cv_gap[i] = min(gaps)
      out[f"limit{LIMIT}_{tag}"] = dict(
          sd_e6=float(sd),
          p_leader_best_public=lead_top / REPS,
          p_leader_in_auto_set=lead_top2 / REPS,
          mean_best_picked_cv_gap_e6=float(picked_cv_gap.mean()),
          p5_best_picked_cv_gap_e6=float(np.percentile(picked_cv_gap, 5)),
          p1_best_picked_cv_gap_e6=float(np.percentile(picked_cv_gap, 1)),
          min_best_picked_cv_gap_e6=float(picked_cv_gap.min()),
          p_best_pick_worse_than_5e6=float((picked_cv_gap < -5.0).mean()),
          p_best_pick_worse_than_10e6=float((picked_cv_gap < -10.0).mean()),
          mean_worst_picked_cv_gap_e6=float(worst_cv_gap.mean()),
          p5_worst_picked_cv_gap_e6=float(np.percentile(worst_cv_gap, 5)),
      )
      o = out[f"limit{LIMIT}_{tag}"]
      print(f"\n--- residual sd {sd:.2f}e-6 ({tag}), {REPS} reps, limit {LIMIT} ---")
      print(f"  P(CV leader is the single best public file)   {o['p_leader_best_public']:.4f}")
      print(f"  P(CV leader lands in the auto-selected set)  {o['p_leader_in_auto_set']:.4f}")
      print(f"  CV of the BETTER auto-pick, vs the CV leader  "
            f"{o['mean_best_picked_cv_gap_e6']:+7.2f}e-6 mean, "
            f"{o['p5_best_picked_cv_gap_e6']:+7.2f}e-6 at the 5th pct")
      print(f"  CV of the WORSE  auto-pick, vs the CV leader  "
            f"{o['mean_worst_picked_cv_gap_e6']:+7.2f}e-6 mean, "
            f"{o['p5_worst_picked_cv_gap_e6']:+7.2f}e-6 at the 5th pct")
      # Kaggle scores private as the BEST of the selected files, so the WORSE pick costs
      # nothing but a wasted slot. The exposure is entirely in the BETTER pick's tail.
      print(f"  >> exposure that actually bites (better pick): "
            f"p1 {o['p1_best_picked_cv_gap_e6']:+7.2f}e-6, "
            f"worst of {REPS} {o['min_best_picked_cv_gap_e6']:+7.2f}e-6")
      print(f"     P(better pick is >5e-6 below the CV leader)  "
            f"{o['p_best_pick_worse_than_5e6']:.4f}")
      print(f"     P(better pick is >10e-6 below the CV leader) "
            f"{o['p_best_pick_worse_than_10e6']:.4f}")

with open(os.path.join(HERE, "w39b_autoselect.json"), "w") as fh:
    json.dump(dict(leader=LEADER, leader_cv=CV[LEADER], best_lb=best_lb,
                   holders=holders, resid_sd=RESID, rho=RHO, sd_idio=float(SD_IDIO),
                   reps=REPS, limits=list(LIMITS), results=out), fh, indent=1)
print("\nwrote experiments/w39b_autoselect.json")
