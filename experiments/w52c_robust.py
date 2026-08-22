"""w52c -- is M2's win real, or is it one day?

w52b's leave-one-day-out over all 11 days reported UNIDENTIFIED for every form. The cause is
benign and worth recording so nobody re-debugs it: family `wh3` has exactly ONE file in the
whole scored set (08-15), so holding out 08-15 leaves `fam[wh3]` with no rows. It is not a
problem with the era terms. Dropping that single row makes the full LODO run.

Two robustness questions, both of which can kill M2:
  A. Does M2 still win the full 12-day LODO, not just the era slice?
  B. The era slice has only TWO days. Report each separately -- if M2's win is carried by
     08-21 alone (5 rows, and the days are the unit of dependence) it is not a result.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w46c_predlb as W46C

t = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
drop = t[t.fam == "wh3"]
print(f"dropping the {len(drop)} singleton-family row(s) so LODO is identified: "
      f"{list(drop.stem)} (day {list(drop.day)})")
t = t[t.fam != "wh3"].reset_index(drop=True)
FAMS = sorted(f for f in t.fam.unique() if f != "h3")
FORMS = ["M0", "M1", "M2", "M3"]


def design(d, form):
    cols = [np.ones(len(d)), d.cv6.values]
    for f in FAMS:
        cols.append((d.fam == f).astype(float).values)
    cols += [d["std"].values, d["corr"].values]
    if form in ("M1", "M3"):
        cols.append(d.era.values)
    if form in ("M2", "M3"):
        cols.append((d.era * d.cv6).values)
    return np.column_stack(cols)


def heldout(form, test_days):
    errs = []
    for d0 in test_days:
        tr, te = t[t.day != d0], t[t.day == d0]
        Xtr = design(tr, form)
        if np.linalg.matrix_rank(Xtr) < Xtr.shape[1]:
            return None
        b, *_ = np.linalg.lstsq(Xtr, tr.lb6.values, rcond=None)
        errs.extend(list(te.lb6.values - design(te, form) @ b))
    return np.array(errs)


days = sorted(t.day.unique())
print(f"\nn={len(t)}, {len(days)} send days\n")

print("A. FULL LEAVE-ONE-DAY-OUT, all 12 days")
print(f"   {'form':5s} {'RMSE':>7s} {'MAE':>7s} {'bias':>7s}")
full = {}
for f in FORMS:
    e = heldout(f, days)
    full[f] = None if e is None else dict(rmse=float(np.sqrt((e**2).mean())),
                                          mae=float(np.abs(e).mean()), bias=float(e.mean()))
    if e is None:
        print(f"   {f:5s} UNIDENTIFIED"); continue
    print(f"   {f:5s} {full[f]['rmse']:7.2f} {full[f]['mae']:7.2f} {full[f]['bias']:+7.2f}")
bestA = min((f for f in FORMS if full.get(f)), key=lambda f: full[f]["rmse"])
print(f"   best: {bestA}")

print("\nB. THE TWO ERA DAYS, HELD OUT SEPARATELY (the days are the unit of dependence)")
print(f"   {'form':5s} {'08-21 RMSE (n=5)':>18s} {'08-22 RMSE (n=10)':>19s} {'pooled':>8s}")
per = {}
for f in FORMS:
    e21, e22 = heldout(f, ["2026-08-21"]), heldout(f, ["2026-08-22"])
    if e21 is None or e22 is None:
        print(f"   {f:5s} UNIDENTIFIED"); per[f] = None; continue
    r21, r22 = float(np.sqrt((e21**2).mean())), float(np.sqrt((e22**2).mean()))
    ep = np.concatenate([e21, e22])
    per[f] = dict(rmse_0821=r21, rmse_0822=r22, rmse_pooled=float(np.sqrt((ep**2).mean())))
    print(f"   {f:5s} {r21:18.2f} {r22:19.2f} {per[f]['rmse_pooled']:8.2f}")

w21 = min((f for f in FORMS if per.get(f)), key=lambda f: per[f]["rmse_0821"])
w22 = min((f for f in FORMS if per.get(f)), key=lambda f: per[f]["rmse_0822"])
print(f"   best on 08-21: {w21}      best on 08-22: {w22}")
print(f"   -> {'M2 WINS BOTH DAYS SEPARATELY -- not carried by one day.' if w21 == w22 == 'M2' else 'THE TWO DAYS DISAGREE -- M2 is not established; report as provisional.'}")

json.dump(dict(n=len(t), dropped=list(drop.stem), full_lodo=full, best_full=bestA,
               era_days=per, best_0821=w21, best_0822=w22,
               m2_wins_both=bool(w21 == w22 == "M2")),
          open(os.path.join(HERE, "w52c_robust.json"), "w"), indent=1)
print(f"\nwrote {os.path.join(HERE,'w52c_robust.json')}")
