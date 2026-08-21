"""w48b -- AUDIT THE LOGIT VETO, the most consequential standing decision on this disk.

WHAT THE VETO IS. `fam[logit] = +147.14e-6` is by far the largest family term in w30b -- six
times the next largest (`fam[rescale]` +34.6). It says a logit file lands 147e-6 of LB ABOVE
where its cross-fitted CV alone would put it. From that one number the workspace derived:

  * the five-file send veto (w29_ad194stdcorr_ens4/_rescale plus the three *_logit files),
    binding on every send list since 08-19 and re-asserted in w46d and w47b;
  * w39's replacement of `w36_ad199std_logit` by `w36_ad199std_hybrid` in slot 9, argued in
    the permanent Kaggle submission log as moving the worst auto-selection from -82.9e-6 to
    -18.1e-6 of CV.

WHAT IT RESTS ON. w48a's coverage check (part D) found w30b's sample is cut by a hard floor,
`h = t[t.cv >= 0.97]` in w30b_corrterm.py line 68. That floor is family-neutral in intent but
NOT in effect: a logit file scores ~10e-6 LESS cross-fitted CV than its own h3 sibling, so the
floor removes logit files preferentially. Of the twelve logit files with a CV on disk, EIGHT
fall below the floor and are cut. `fam[logit]` is fitted on the surviving FOUR, all of them
ad187-ad194, all inside a 15e-6 window of CV.

Truncating on a regressor does not bias OLS, so this is not an accusation that the term is
wrong. It is the observation that eight relevant, already-scored observations have never been
used, and that they sit 300-400e-6 of CV away -- an enormous lever arm on exactly the linear
form the term is defined against. This file holds the four-point term fixed and PREDICTS the
eight, which is a genuine out-of-sample test and costs nothing.

READING RULE, fixed before the numbers are printed:
  * mean held-out residual within +/- 2 * (LOO sd 8.36e-6) = +/- 16.7e-6   -> the term holds;
    the veto is sound and stays.
  * outside that                                                          -> `fam[logit]` is
    absorbing misspecification rather than measuring a family offset, and the veto -- which
    is a real cost, five files blocked from every send list -- has to be re-derived.

    .venv/bin/python experiments/w48b_logitveto.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stdflag import family, is_std                      # noqa: E402
import w46c_predlb as W                                 # noqa: E402

SD_LOO = 8.36            # w47a leave-one-out held-out residual sd, e-6
FLOOR = 0.97             # w30b_corrterm.py line 68

t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"]).copy()
t["fam"] = t.stem.map(family)
t["std"] = t.stem.map(is_std).astype(float)
t["corr"] = t.stem.str.contains("corr").astype(float)
inn, out = t[t.cv >= FLOOR].copy(), t[t.cv < FLOOR].copy()

print("=" * 88)
print("w48b  THE LOGIT VETO, AUDITED AGAINST THE EIGHT FILES THE CV FLOOR CUT")
print("=" * 88)
print(f"\n  w30b fits on cv >= {FLOOR}: n {len(inn)} in, n {len(out)} cut.")
print(f"  fam[logit] {W.C['fam[logit]']:+.2f}e-6 se {json.load(open(os.path.join(HERE,'w30b_corrterm.json')))['ses']['fam[logit]']:.2f}"
      f"  -- the largest family term, {W.C['fam[logit]']/W.C['fam[rescale]']:.1f}x the next.")
print(f"\n  logit files IN the fit  : {(inn.fam=='logit').sum()}")
print(f"  logit files CUT by floor: {(out.fam=='logit').sum()}")
print(f"  the floor is family-neutral in intent; in effect it cut "
      f"{(out.fam=='logit').sum()}/{(t.fam=='logit').sum()} of the logit evidence "
      f"and {(out.fam!='logit').sum()}/{(t.fam!='logit').sum()} of everything else.")

print("\n  --- HELD-OUT: the four-point term, applied to the eight it never saw ---")
print(f"  {'stem':26s} {'cv':>12s} {'d_floor':>8s} {'fam':>8s} {'pred':>9s} {'actual':>8s} {'resid':>8s}")
rows = []
for _, r in out.sort_values("cv").iterrows():
    p = W._raw(r.cv, r.stem)               # w30b raw: none of these is ad>=195, so no era term
    res = (r.lb - p) * 1e6
    rows.append(dict(stem=r.stem, cv=r.cv, fam=r.fam, pred=p, lb=r.lb, res6=res))
    print(f"  {r.stem:26s} {r.cv:.7f} {(r.cv-FLOOR)*1e6:+8.1f} {r.fam:>8s} "
          f"{p:.6f} {r.lb:8.5f} {res:+8.2f}")
H = pd.DataFrame(rows)
L = H[H.fam == "logit"]

for lbl, D in (("ALL EIGHT cut files", H), ("the LOGIT subset", L)):
    m, n = D.res6.mean(), len(D)
    sem = SD_LOO / np.sqrt(n)
    print(f"\n  {lbl:22s} n {n}  mean residual {m:+7.2f}e-6  sem {sem:.2f}  z {m/sem:+6.2f}")
    print(f"  {'':22s} reading rule +/- {2*SD_LOO:.1f}e-6  ->  "
          f"{'HOLDS' if abs(m) <= 2*SD_LOO else '⛔ FAILS'}")

# what fam[logit] would have to be for the eight to land on prediction
need = W.C["fam[logit]"] + L.res6.mean()
print(f"\n  For the eight cut files to land on prediction, fam[logit] would have to be "
      f"{need:+.2f}e-6")
print(f"  instead of {W.C['fam[logit]']:+.2f}e-6 -- a shift of {L.res6.mean():+.2f}e-6.")

# refit with the floor removed, purely as a second view
print("\n  --- SECOND VIEW: refit with no CV floor (reported, NOT adopted) ---")
def fit(h):
    fams = sorted(h.fam.unique()); mu = h.cv.mean()
    X, names = [np.ones(len(h)), (h.cv.values - mu) * 1e6], ["const", "cv_e6"]
    for f in fams:
        if f == "h3":
            continue
        X.append((h.fam.values == f).astype(float)); names.append(f"fam[{f}]")
    for c in ("std", "corr"):
        X.append(h[c].values.astype(float)); names.append(c)
    X = np.column_stack(X); yv = h.lb.values * 1e6
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X @ b; dof = len(h) - X.shape[1]
    se = np.sqrt(np.diag((r @ r / dof) * np.linalg.inv(X.T @ X)))
    return dict(zip(names, b)), dict(zip(names, se)), float(np.sqrt(r @ r / dof))

b_all, se_all, sd_all = fit(t)
b_in, se_in, sd_in = fit(inn)
print(f"  {'term':16s} {'floor 0.97 (w30b)':>22s} {'no floor':>22s}")
for k in ("cv_e6", "fam[logit]", "fam[rescale]", "fam[ens4]", "std", "corr"):
    print(f"  {k:16s} {b_in.get(k,float('nan')):+10.2f} se {se_in.get(k,float('nan')):5.2f}   "
          f"{b_all.get(k,float('nan')):+10.2f} se {se_all.get(k,float('nan')):5.2f}")
print(f"  {'resid sd':16s} {sd_in:10.2f}      {'':6s}{sd_all:10.2f}")
print(f"  {'n':16s} {len(inn):10d}      {'':6s}{len(t):10d}")

json.dump(dict(n_in=len(inn), n_cut=len(out),
               heldout=H.to_dict("records"),
               mean_all=float(H.res6.mean()), mean_logit=float(L.res6.mean()),
               n_logit=int(len(L)), sd_loo=SD_LOO,
               fam_logit_w30b=W.C["fam[logit]"], fam_logit_implied=float(need),
               refit_no_floor=b_all, refit_floor=b_in,
               sd_no_floor=sd_all, sd_floor=sd_in),
          open(os.path.join(HERE, "w48b_logitveto.json"), "w"), indent=1)
print("\nwrote w48b_logitveto.json")
