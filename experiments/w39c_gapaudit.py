"""w39 consolidation -- the CV->LB gap across EVERY scored submission, and a pre-registered
held-out test for the 08-21 drain.

The angle for this slot is "check the CV-to-LB gap across every experiment so far". w25a built
that table; this re-reads it under the CURRENT model (w30b, the one w26d actually prices with)
and asks three things the table alone does not answer:

  1. Is the fit still unbiased, family by family, on all 87 scored files?
  2. Is any SENT file anomalously high for its CV? That matters directly: Kaggle auto-selects
     on public score, so an inflated file is exactly the one that gets picked over the CV pick.
  3. The 13 queued files already carry a `pred_lb` written before upload. Freeze them here so
     the drain is a genuine held-out test of w30b, the way w30a was of w26e.

⚠⚠ TWO DEFECTS FOUND AND FIXED BY w57 (2026-08-22), both of which let this script exit 0
while printing confident numbers that were wrong:

  (a) IT FITTED A STALE TABLE. `w25a_cvlb_full.csv` is a frozen snapshot, not the live board.
      By 08-22 it was two board-days behind (87 rows against 111 scored submissions) and its
      top-LB tie held THREE files -- while the live tie held FIVE, including the CV leader
      `w36_ad199stdcorr` itself. So the auto-selection paragraph below, whose entire job is to
      say whether auto-selection is dangerous, was computed on a table that did not contain
      the file the danger is about. `GATE S` now re-derives the tie from the LIVE board and
      REFUSES to run when the table is behind. This is the same failure RESEARCH already
      records once for `w39b_autoselect` (it built its CV dict from the queue and took max(),
      so "every figure in the auto-selection report was measured against" the wrong leader).

  (b) ITS "FROZEN" PRE-REGISTRATION WAS NOT FROZEN. The block printed `frozen {utcnow()}` and
      re-read the CURRENT queue, so every run silently rewrote the predictions AND re-stamped
      them with the present time -- including into `w39c_gapaudit.json`'s `prereg_0821` key.
      A later run comparing realised LB against "the frozen prediction" would have been
      comparing against a prediction regenerated after the outcome was knowable, which is the
      exact opposite of a held-out test. The block is now written ONCE to
      `w39c_prereg_frozen.json` and thereafter re-read and reprinted with its ORIGINAL
      timestamp; it is never overwritten. It was also MISLABELLED "the 08-21 drain" while
      reading the 08-23 plan, so the frozen record is now keyed to the queue's own `plan_day`.

⛔ SUPERSEDED FOR THE AUTO-SELECTION QUESTION -- USE `w57a_tierprice2.py`.
GATE S will keep this script at exit 2 until someone DELIBERATELY re-centres the pricer (see
the gate's own message and `w57c_muguard.py`), and that is the correct resting state: the
auto-selection paragraph here cannot be computed honestly from a frozen table. `w57a` answers
the same question from the LIVE board, needs no table, and records the tier membership it ran
on into its JSON so a future run can tell at a glance whether its numbers still apply. What
this script still uniquely offers -- the family-by-family residual decomposition -- is worth
having, but only after a deliberate refit, never as a drive-by refresh.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys

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

COMP = "playground-series-s6e8"


def live_top_tie():
    """The stems holding the best PUBLIC score on the live board, and the scored count.

    w57 GATE S. Everything below prices auto-selection, and auto-selection runs on the LIVE
    board -- so a frozen table is not merely out of date here, it is answering about a
    different set of files than the one Kaggle will pick from.
    """
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    # w54: the API defaults to 50 rows and truncates SILENTLY.
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    best = sub.groupby("stem")["publicScore"].max()
    return set(best[best == best.max()].index), int(len(sub)), float(best.max())


t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv", "lb"])
t = t[t.cv >= 0.97].copy()

# --- w57 GATE S: refuse to price auto-selection off a table the board has moved past.
_live_tie, _n_live, _live_best = live_top_tie()
_tab_tie = set(t[t.lb == t.lb.max()].stem)
print(f"GATE S: live board {_n_live} scored, best public {_live_best:.5f} held by "
      f"{len(_live_tie)} file(s); table has {len(t)} rows, best {t.lb.max():.5f} held by "
      f"{len(_tab_tie)}")
if _live_tie != _tab_tie:
    print("*** GATE S FAILED -- w25a_cvlb_full.csv IS STALE. ***")
    print(f"    live tie:  {sorted(_live_tie)}")
    print(f"    table tie: {sorted(_tab_tie)}")
    print(f"    missing from the table: {sorted(_live_tie - _tab_tie)}")
    print("    Do NOT read the family residuals or the auto-selection paragraph below: they")
    print("    would be computed on a file set Kaggle is not choosing from.")
    print()
    print("    🔴 AND DO NOT 'JUST REFRESH IT'. `w25a_cvlb_full.py` IS NOT A READ-ONLY REFRESH.")
    print("       `w46c_predlb.MU` is defined as the MEAN CV of this table's cv>=0.97 rows:")
    print("           w46c_predlb.py:59   MU = _t[_t.cv >= 0.97].cv.mean()")
    print("       so rewriting the table RE-CENTRES the pricer's design. `w52b_cvlb93.csv`")
    print("       carries a PRECOMPUTED `cv6` column built at the old MU, and w53a_pricer")
    print("       fits on that column while predicting through the new one -- they desync and")
    print("       w53a's own gate fires (`predict_flags() does not reproduce the M2 fit`).")
    print("       w57 did exactly this and had to `git checkout` the table to restore the")
    print("       send path. Measured: the 08-22 refresh moved MU by +10.76e-6 (78 -> 93 rows).")
    print("       Refreshing is a DELIBERATE send-path change: refit M2, regenerate")
    print("       w52b_cvlb93.csv's cv6, and re-derive w52b_joint/w52d_predlb in the SAME run.")
    print("       Never as a side effect of running an audit. See experiments/w57c_muguard.py.")
    sys.exit(2)
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
_day = str(q.plan_day.iloc[0]) if "plan_day" in q.columns and len(q) else "unknown"

# --- w57: the pre-registration is written ONCE and never overwritten. A prereg that
# re-stamps itself on every run is not a prereg.
FROZEN = os.path.join(HERE, "w39c_prereg_frozen.json")
if os.path.exists(FROZEN):
    _fz = json.load(open(FROZEN))
    _fresh = False
else:
    _fz = dict(frozen_utc=f"{pd.Timestamp.utcnow():%Y-%m-%d %H:%M}Z", plan_day=_day,
               rows=[{"send_rank": int(r.send_rank), "file": r.file, "fam": r.fam,
                      "cv": float(r.cv), "pred_lb": float(r.pred_lb)} for r in q.itertuples()])
    json.dump(_fz, open(FROZEN, "w"), indent=1)
    _fresh = True
print(f"\n--- PRE-REGISTERED, frozen {_fz['frozen_utc']}: the {_fz['plan_day']} drain "
      f"as a held-out test of w30b ---")
print(f"    ({'written this run' if _fresh else 're-read from w39c_prereg_frozen.json, NOT regenerated'})")
if not _fresh and _fz["plan_day"] != _day:
    print(f"    ⚠ the queue has moved on to {_day}. The frozen block below still covers "
          f"{_fz['plan_day']} and is LEFT ALONE -- that is the point of freezing it.")
    print(f"    To register a NEW day, add a day-keyed file; do NOT delete this one.")
print(f"{'send':>4}  {'file':28s} {'fam':8s} {'cv':>14s} {'pred LB':>9s}")
for r in _fz["rows"]:
    print(f"{r['send_rank']:>4}  {r['file'][:-4]:28s} {r['fam']:8s} {r['cv']:.10f} {r['pred_lb']:9.5f}")
print("\nAfter the drain, re-run w25a_cvlb_full.py and compare the realised LB against the")
print("`pred LB` column above BEFORE refitting anything. Mean residual and its z are the test;")
print("w30a is the precedent and it is what caught the +19.3e-6 *stdcorr bias.")

json.dump(dict(n=len(t), mean_resid_e6=float(t.resid_e6.mean()),
               sd_resid_e6=float(t.resid_e6.std(ddof=1)),
               by_family={k: {c: (None if pd.isna(v) else float(v)) for c, v in row.items()}
                          for k, row in g.iterrows()},
               most_inflated=top.stem.tolist(),
               prereg_frozen=_fz,
               gate_s=dict(live_scored=_n_live, live_best=_live_best,
                           tie=sorted(_live_tie))),
          open(os.path.join(HERE, "w39c_gapaudit.json"), "w"), indent=1)
t.to_csv(os.path.join(HERE, "w39c_gapaudit.csv"), index=False)
print("\nwrote experiments/w39c_gapaudit.{csv,json}")
