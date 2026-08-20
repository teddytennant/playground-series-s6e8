"""w39 consolidation -- the CV->LB gap across EVERY scored submission, and a pre-registered
held-out test for the 08-21 drain.

The angle for this slot is "check the CV-to-LB gap across every experiment so far". w25a built
that table; this re-reads it under the CURRENT model (w30b, the one w26d actually prices with)
and asks three things the table alone does not answer:

  1. Is the fit still unbiased, family by family, on all 87 scored files?
  2. Is any SENT file anomalously high for its CV? That matters directly: Kaggle auto-selects
     on public score, so an inflated file is exactly the one that gets picked over the CV pick.
  3. The 13 queued files already carry a `pred_lb` written before upload. Freeze them here so
     the 08-21 drain is a genuine held-out test of w30b, the way w30a was of w26e.
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
    import w26d_queueprice as QP  # side-effect free since w39
    import stdflag
print([l for l in buf.getvalue().splitlines() if l.startswith("GATE")][0])
assert "left untouched" in buf.getvalue(), "w26d wrote on import -- the w39 guard is gone"

t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv", "lb"])
t = t[t.cv >= 0.97].copy()
t["fam"] = t.stem.map(stdflag.family)
t["std"] = t.stem.map(stdflag.is_std)
t["corr"] = t.stem.map(QP.is_corr)
t["pred"] = [QP.predict(r.cv, r.fam, r.std, r.corr) for r in t.itertuples()]
t["resid_e6"] = (t.lb - t.pred) * 1e6
print(f"\nscored files with CV >= 0.97: {len(t)}   "
      f"mean residual {t.resid_e6.mean():+.2f}e-6   sd {t.resid_e6.std(ddof=1):.2f}e-6")

print("\n--- gap and residual by family (gap = LB - CV, the raw quantity the angle names) ---")
g = t.groupby("fam").agg(n=("lb", "size"), cv=("cv", "mean"), lb=("lb", "mean"),
                         gap_e6=("lb", lambda s: np.nan), resid=("resid_e6", "mean"),
                         resid_sd=("resid_e6", lambda s: s.std(ddof=1)))
g["gap_e6"] = [(t[t.fam == f].lb - t[t.fam == f].cv).mean() * 1e6 for f in g.index]
print(g.sort_values("resid", ascending=False).to_string(
    float_format=lambda v: f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"))

print("\n--- the 8 files most INFLATED for their CV (auto-selection picks these first) ---")
top = t.nlargest(8, "resid_e6")[["stem", "fam", "cv", "lb", "pred", "resid_e6"]]
top["cv_vs_best"] = (top.cv - t.cv.max()) * 1e6
print(top.to_string(index=False,
                    formatters={"cv": "{:.10f}".format, "pred": "{:.6f}".format,
                                "resid_e6": "{:+.2f}".format, "cv_vs_best": "{:+.2f}".format}))

# Is the board-best file one of them? That is the question that decides whether auto-selection
# is dangerous or merely imprecise.
best_lb = t.lb.max()
hold = t[t.lb == best_lb]
print(f"\nbest public {best_lb:.5f} is held by {len(hold)} file(s); their standardised "
      f"residuals: " + ", ".join(f"{r.stem} {r.resid_e6/t.resid_e6.std(ddof=1):+.2f}sd"
                                 for r in hold.itertuples()))

q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
q = q[q.priority == 1].sort_values("send_rank")
print(f"\n--- PRE-REGISTERED, frozen {pd.Timestamp.utcnow():%Y-%m-%d %H:%M}Z: the 08-21 drain "
      f"as a held-out test of w30b ---")
print(f"{'send':>4}  {'file':28s} {'fam':8s} {'cv':>14s} {'pred LB':>9s}")
for r in q.itertuples():
    print(f"{int(r.send_rank):>4}  {r.file[:-4]:28s} {r.fam:8s} {r.cv:.10f} {r.pred_lb:9.5f}")
print("\nAfter the drain, re-run w25a_cvlb_full.py and compare the realised LB against the")
print("`pred LB` column above BEFORE refitting anything. Mean residual and its z are the test;")
print("w30a is the precedent and it is what caught the +19.3e-6 *stdcorr bias.")

json.dump(dict(n=len(t), mean_resid_e6=float(t.resid_e6.mean()),
               sd_resid_e6=float(t.resid_e6.std(ddof=1)),
               by_family={k: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                          for k, row in g.iterrows()},
               most_inflated=top.stem.tolist(),
               prereg_0821=[{"send_rank": int(r.send_rank), "file": r.file,
                             "cv": float(r.cv), "pred_lb": float(r.pred_lb)}
                            for r in q.itertuples()]),
          open(os.path.join(HERE, "w39c_gapaudit.json"), "w"), indent=1)
t.to_csv(os.path.join(HERE, "w39c_gapaudit.csv"), index=False)
print("\nwrote experiments/w39c_gapaudit.{csv,json}")
