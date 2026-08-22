"""w52b -- the joint fit that w47b's R1=MIXED branch and R3's BAD-CV-SLOPE verdict both order.

w52a returned:
    R1  p  = -13.02e-6  -> MIXED    ("fit an era term AND a cv-region term jointly on the
                                      pooled 15 and report BOTH; do not round it to one story")
    R2  r5 = -31.91e-6  -> H1 confirmed on the high-CV side
    R3  slope -0.675/e-6 (t -4.27) -> BAD CV SLOPE ("the fix is a re-estimated slope, not a dummy")

An era DUMMY is not identified inside the pooled 15 -- every one of those files is ad>=195, so
the dummy is collinear with the intercept there. The joint fit therefore has to run on the FULL
scored set: w30b's 78 pre-era rows + the 5 from 08-21 + the 10 from 08-22 = 93.

Four forms, all on the same 93 rows and the same base design:
    M0  w30b's form, refit                       (no era term at all)
    M1  + era level dummy                        (w46c's form)
    M2  + era x cv interaction                   (R3's form: the slope changes, not the level)
    M3  + era dummy AND era x cv                 (R1-MIXED's form: report both)

⚠ In-sample residual sd always falls as parameters are added, so it decides nothing. The forms
are compared by LEAVE-ONE-DAY-OUT prediction over the 11 send days -- honest held-out error,
and it respects the fact that files sent on the same day are not independent draws.
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w46c_predlb as W46C
from stdflag import family, is_std

TODAY = [("w40_ad211std_rankraw", 0.9701044022, 0.97114),
         ("w38_ad202std_rankraw", 0.9701067554, 0.97114),
         ("w34_ad196std_hybrid",  0.9701091250, 0.97113),
         ("w34_ad195std_hybrid",  0.9701108333, 0.97113),
         ("w36_ad197std_hybrid",  0.9701124257, 0.97114),
         ("w36_ad199std_h3",      0.9701354276, 0.97116),
         ("w38_ad202std",         0.9701305726, 0.97116),
         ("w40_ad211std",         0.9701309541, 0.97117),
         ("w40_ad211stdcorr",     0.9701374733, 0.97118),
         ("w38_ad202stdcorr",     0.9701375891, 0.97117)]

t = pd.read_csv(os.path.join(HERE, "w46c_cvlb_live.csv"))[["stem", "cv", "lb", "day"]]
t = pd.concat([t, pd.DataFrame([dict(stem=s, cv=cv, lb=lb, day="2026-08-22")
                                for s, cv, lb in TODAY])], ignore_index=True)
assert len(t) == 93 and t.stem.is_unique, (len(t), t.stem.duplicated().sum())

t["fam"] = t.stem.map(family)
t["std"] = t.stem.map(is_std).astype(float)
t["corr"] = t.stem.str.contains("corr").astype(float)
t["era"] = t.stem.map(W46C.new_era).astype(float)
t["cv6"] = (t.cv - W46C.MU) * 1e6
t["lb6"] = t.lb * 1e6

FAMS = sorted(f for f in t.fam.unique() if f != "h3")           # h3 is the reference level


def design(d, form):
    cols, names = [np.ones(len(d)), d.cv6.values], ["const", "cv6"]
    for f in FAMS:
        cols.append((d.fam == f).astype(float).values); names.append(f"fam[{f}]")
    cols += [d["std"].values, d["corr"].values];        names += ["std", "corr"]
    if form in ("M1", "M3"):
        cols.append(d.era.values);                       names.append("era")
    if form in ("M2", "M3"):
        cols.append((d.era * d.cv6).values);             names.append("era:cv6")
    return np.column_stack(cols), names


def fit(d, form):
    X, names = design(d, form)
    b, *_ = np.linalg.lstsq(X, d.lb6.values, rcond=None)
    r = d.lb6.values - X @ b
    dof = len(d) - np.linalg.matrix_rank(X)
    s2 = float(r @ r) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return dict(zip(names, b)), dict(zip(names, se)), float(np.sqrt(float(r @ r) / dof)), names


FORMS = ["M0", "M1", "M2", "M3"]
print("=" * 96)
print(f"w52b -- joint fit on the full scored set, n={len(t)} "
      f"({int(t.era.sum())} era ad>=195, {int((1-t.era).sum())} pre-era)")
print("=" * 96)

fits = {}
for form in FORMS:
    b, se, sd, names = fit(t, form)
    fits[form] = dict(coefs=b, ses=se, resid_sd=sd)
    extra = " ".join(f"{k} {b[k]:+.3f}({se[k]:.3f})" for k in ("era", "era:cv6") if k in b)
    print(f"\n{form}: in-sample resid sd {sd:5.2f}e-6   cv6 slope {b['cv6']:+.3f} "
          f"(se {se['cv6']:.3f})   {extra}")

print("\n" + "-" * 96)
print("LEAVE-ONE-DAY-OUT held-out error over the 11 send days (this is what decides it)")
print("-" * 96)
days = sorted(t.day.unique())
lodo = {}
for form in FORMS:
    errs = []
    for d0 in days:
        tr, te = t[t.day != d0], t[t.day == d0]
        Xtr, names = design(tr, form)
        if np.linalg.matrix_rank(Xtr) < Xtr.shape[1]:
            errs = None; break                      # form unidentified without that day
        b, *_ = np.linalg.lstsq(Xtr, tr.lb6.values, rcond=None)
        Xte, _ = design(te, form)
        errs.extend(list(te.lb6.values - Xte @ b))
    if errs is None:
        lodo[form] = None
        print(f"{form}: UNIDENTIFIED on at least one fold")
        continue
    e = np.array(errs)
    lodo[form] = dict(rmse=float(np.sqrt((e ** 2).mean())), mae=float(np.abs(e).mean()),
                      bias=float(e.mean()))
    print(f"{form}: held-out RMSE {lodo[form]['rmse']:5.2f}e-6   MAE {lodo[form]['mae']:5.2f}"
          f"   bias {lodo[form]['bias']:+5.2f}")

# --- era-day-only held-out error: the 15 rows that actually matter for pricing the queue ---
print("\n" + "-" * 96)
print("Held-out error RESTRICTED to the 15 era rows (08-21 + 08-22) -- the pricing-relevant slice")
print("-" * 96)
era_lodo = {}
for form in FORMS:
    errs = []
    ok = True
    for d0 in ["2026-08-21", "2026-08-22"]:
        tr, te = t[t.day != d0], t[t.day == d0]
        Xtr, _ = design(tr, form)
        if np.linalg.matrix_rank(Xtr) < Xtr.shape[1]:
            ok = False; break
        b, *_ = np.linalg.lstsq(Xtr, tr.lb6.values, rcond=None)
        Xte, _ = design(te, form)
        errs.extend(list(te.lb6.values - Xte @ b))
    if not ok:
        era_lodo[form] = None; print(f"{form}: UNIDENTIFIED"); continue
    e = np.array(errs)
    era_lodo[form] = dict(rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
    print(f"{form}: held-out RMSE {era_lodo[form]['rmse']:5.2f}e-6   bias {era_lodo[form]['bias']:+6.2f}")

best = min((f for f in FORMS if era_lodo.get(f)), key=lambda f: era_lodo[f]["rmse"])
print(f"\nBEST FORM on the era slice: {best}")

b3, se3 = fits["M3"]["coefs"], fits["M3"]["ses"]
print("\n" + "=" * 96)
print("M3 -- BOTH TERMS, as R1's MIXED branch requires them reported")
print("=" * 96)
print(f"  era level      {b3['era']:+8.2f}e-6   se {se3['era']:5.2f}   t {b3['era']/se3['era']:+6.2f}")
print(f"  era:cv6 slope  {b3['era:cv6']:+8.3f}     se {se3['era:cv6']:5.3f}   "
      f"t {b3['era:cv6']/se3['era:cv6']:+6.2f}")
print(f"  base cv6 slope {b3['cv6']:+8.3f}     se {se3['cv6']:5.3f}")
print(f"  => era files convert CV at {b3['cv6'] + b3['era:cv6']:+.3f} per e-6, "
      f"vs {b3['cv6']:+.3f} pre-era "
      f"({100*(b3['cv6']+b3['era:cv6'])/b3['cv6']:.0f}% of the old rate)")

json.dump(dict(n=len(t), n_era=int(t.era.sum()), fams=FAMS, mu=W46C.MU,
               fits={k: dict(coefs=v["coefs"], ses=v["ses"], resid_sd=v["resid_sd"])
                     for k, v in fits.items()},
               lodo=lodo, era_lodo=era_lodo, best_form=best),
          open(os.path.join(HERE, "w52b_joint.json"), "w"), indent=1)
t.to_csv(os.path.join(HERE, "w52b_cvlb93.csv"), index=False)
print(f"\nwrote w52b_joint.json and w52b_cvlb93.csv (n={len(t)})")
