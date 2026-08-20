"""w39 -- THE BRIEF'S "an extra submission can never hurt" IS FALSE HERE. Pricing the one
file in the 08-21 queue where it fails.

The brief's submission economics say the public board shows best-of-all, so a spare slot is
free and an extra send can only help. That is true for PUBLIC RANK. It is NOT true for the
FINAL SCORE, because nothing is selected and Kaggle then auto-selects **on public score**:
a file that prints high publicly while sitting far below the CV leader can DISPLACE the CV
pick out of the auto-selected set. Sending it is then strictly negative.

The queue has exactly one such file. `w36_ad199std_logit` is 82.9e-6 of CV below the leader
-- the worst in the priority-1 set by a factor of four -- yet w30b prices it at 0.97119,
inside a reporting step of the top, because the `logit` family term is +1121e-6 of LB-CV gap
against +1025e-6 for h3 (w39c). It is the same shape as `blend158_logit`, the file the w13
audit identified as the whole of the auto-selection exposure.

This re-runs w39b with that one file removed and prices the difference. Nothing else changes.
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
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    import w26d_queueprice as QP
assert "left untouched" in buf.getvalue(), "w26d wrote on import -- the w39 guard is gone"
RESID = QP.RESID
SD_IDIO = RESID * np.sqrt(1.0 - (1.0 - (7.4 ** 2) / (2.0 * RESID ** 2)))

full = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv", "lb"])
q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
q = q[q.priority == 1].sort_values("send_rank")
CV = dict(zip(full.stem, full.cv)) | dict(zip(q.stem, q.cv))
LB = dict(zip(full.stem, full.lb))
LEADER = max(CV, key=lambda s: CV[s])
DROP = "w36_ad199std_logit"

sent_stems, sent_lb = list(LB), np.array(list(LB.values()))
REPS = 20000


def price(stems_q, limit, sd, seed):
    rng = np.random.default_rng(seed)
    sub = q[q.stem.isin(stems_q)]
    mu = np.array([QP.predict(r.cv, r.fam, bool(r.std), bool(r.corr)) for r in sub.itertuples()])
    names = list(sub.stem)
    draws = np.round(mu[None, :] + rng.normal(0.0, sd * 1e-6, (REPS, len(names))), 5)
    allst = sent_stems + names
    inset, gap = 0, np.empty(REPS)
    for i in range(REPS):
        alllb = np.concatenate([sent_lb, draws[i]])
        order = np.lexsort((rng.random(len(allst)), -alllb))
        top = [allst[j] for j in order[:limit]]
        inset += LEADER in top
        gap[i] = max((CV[s] - CV[LEADER]) * 1e6 for s in top)
    return dict(p_leader_selected=inset / REPS, mean_gap_e6=float(gap.mean()),
                p1_gap_e6=float(np.percentile(gap, 1)), min_gap_e6=float(gap.min()),
                p_gap_worse_10e6=float((gap < -10.0).mean()))


keep = [s for s in q.stem if s != DROP]
print(f"CV leader {LEADER} ({CV[LEADER]:.10f}); dropping {DROP} "
      f"({(CV[DROP]-CV[LEADER])*1e6:+.1f}e-6 of CV)\n")
res = {}
for limit in (2, 1):
    for tag, sd in (("idio", SD_IDIO), ("marg", RESID)):
        a = price(list(q.stem), limit, sd, 3939)
        b = price(keep, limit, sd, 3939)
        res[f"limit{limit}_{tag}"] = dict(with_logit=a, without_logit=b)
        print(f"limit {limit}, sd {sd:.2f}e-6 ({tag}):")
        print(f"   P(CV leader auto-selected)      {a['p_leader_selected']:.4f} -> "
              f"{b['p_leader_selected']:.4f}   ({b['p_leader_selected']-a['p_leader_selected']:+.4f})")
        print(f"   E[CV of the best auto-pick]  {a['mean_gap_e6']:+7.2f} -> "
              f"{b['mean_gap_e6']:+7.2f} e-6   ({b['mean_gap_e6']-a['mean_gap_e6']:+.2f})")
        print(f"   1st pct                      {a['p1_gap_e6']:+7.2f} -> {b['p1_gap_e6']:+7.2f} e-6")
        print(f"   worst of {REPS}             {a['min_gap_e6']:+7.2f} -> {b['min_gap_e6']:+7.2f} e-6")
        print(f"   P(best pick >10e-6 below)       {a['p_gap_worse_10e6']:.4f} -> "
              f"{b['p_gap_worse_10e6']:.4f}\n")

json.dump(dict(leader=LEADER, dropped=DROP, reps=REPS, results=res),
          open(os.path.join(HERE, "w39d_logithazard.json"), "w"), indent=1)
print("wrote experiments/w39d_logithazard.json")
