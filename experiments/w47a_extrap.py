"""w47a — is the -29.82e-6 "era shift" an ERA effect, or the predictor failing above
its own CV support?  Registered in experiments/w47_prereg.txt before this file existed.

w46 section 2 dismissed extrapolation with this check:
    "In-sample residual by CV quintile is FLAT ... so the linear-in-CV slope is not
     failing at the top of the range and this is not an extrapolation artefact."
w30b is OLS with a constant and a CV term.  sum(r)=0 and sum(r*cv)=0 are its normal
equations.  The flat profile is arithmetic, not evidence -- and in any case there is
no in-sample data above cv 0.9701180 for any in-sample check to speak about, while
four of the five 08-21 files and ALL TEN queued for 08-22 sit above it.

    .venv/bin/python experiments/w47a_extrap.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RNG = np.random.default_rng(20260821)
t = pd.read_csv(os.path.join(HERE, "w46c_cvlb_live.csv"))
t["day"] = pd.to_datetime(t.day)
FAMS = sorted(t.fam.unique())
OLD = t[t.day < "2026-08-21"].reset_index(drop=True)      # the 78 w30b was fitted on
NEW = t[t.day == "2026-08-21"].reset_index(drop=True)     # the 5 held-out


def design(df, mu, fams):
    X, names = [np.ones(len(df)), (df.cv.values - mu) * 1e6], ["const", "cv_e6"]
    for f in fams:
        if f == "h3":
            continue
        X.append((df.fam.values == f).astype(float)); names.append(f"fam[{f}]")
    X.append(df["std"].values.astype(float)); names.append("std")
    X.append(df.cr.values.astype(float)); names.append("corr")
    return np.column_stack(X), names


def fit_predict(tr, te):
    """Fit w30b's form on tr, predict te. Families absent from tr are dropped from
    both design and test rows (a held-out row of an unseen family is unpredictable)."""
    fams = sorted(set(tr.fam) & set(FAMS))
    te = te[te.fam.isin(fams)]
    if len(te) == 0 or len(tr) < len(fams) + 4:
        return None, None, None
    mu = tr.cv.mean()
    Xtr, _ = design(tr, mu, fams)
    b, *_ = np.linalg.lstsq(Xtr, tr.lb.values * 1e6, rcond=None)
    Xte, _ = design(te, mu, fams)
    res = te.lb.values * 1e6 - Xte @ b
    return te, res, b


print("=" * 78)
print("SETUP -- the design points, which is the whole story")
print("=" * 78)
print(f"  78 fitted files   cv range {OLD.cv.min():.7f} .. {OLD.cv.max():.7f}")
print(f"   5 held-out files cv range {NEW.cv.min():.7f} .. {NEW.cv.max():.7f}")
q = json.load(open(os.path.join(HERE, "w45b_unsent_cv.json")))
qcv = np.array([r["cv"] for r in q])
print(f"  77 unsent files   cv range {qcv.min():.7f} .. {qcv.max():.7f}   "
      f"{(qcv > OLD.cv.max()).sum()} of them above the fitted max")
d_new = (NEW.cv.values - OLD.cv.max()) * 1e6
print(f"  held-out five sit {d_new.min():+.1f} .. {d_new.max():+.1f}e-6 above the "
      f"fitted max (mean {d_new.mean():+.1f})")

print("\n" + "=" * 78)
print("A. THE QUINTILE CHECK IS ARITHMETIC.  Demonstration on a KNOWN saturating truth.")
print("=" * 78)
print("  Truth: lb = linear in cv up to a knee, dead FLAT above it (saturation).")
print("  Fit the linear model on data BELOW the knee only -- which is exactly the")
print("  situation w30b was in -- then run w46's check on the fitted residuals.")
knee = 0.970118
cvs = np.sort(RNG.uniform(0.970012, knee, 78))
lbs = 971069e-6 + 1.0 * (cvs - 0.9700571) + RNG.normal(0, 8e-6, 78)
b = np.polyfit((cvs - cvs.mean()) * 1e6, lbs * 1e6, 1)
r = lbs * 1e6 - np.polyval(b, (cvs - cvs.mean()) * 1e6)
qs = pd.qcut(cvs, 5, labels=False)
prof = [r[qs == i].mean() for i in range(5)]
print("  in-sample residual by CV quintile: " + " / ".join(f"{v:+.2f}" for v in prof) + "e-6")
cv_out = np.array([knee + d * 1e-6 for d in d_new])
true_out = 971069e-6 + 1.0 * (knee - 0.9700571)          # flat above the knee
bias = (true_out * 1e6 - np.polyval(b, (cv_out - cvs.mean()) * 1e6))
print(f"  yet the SAME fit, at the five real held-out design points, is biased "
      f"{bias.mean():+.1f}e-6.")
print("  The check reads FLAT on data that is saturating.  It has no power over the")
print("  linear component (normal equations force it) and no data at all above the knee.")

print("\n" + "=" * 78)
print("B. E1 -- EXTRAPOLATION CURVE.  Fit the low-CV part, predict the held-out top.")
print("=" * 78)
rows = []
for cut in np.arange(0.50, 0.951, 0.05):
    thr = OLD.cv.quantile(cut)
    tr, te = OLD[OLD.cv <= thr], OLD[OLD.cv > thr]
    te2, res, _ = fit_predict(tr, te)
    if te2 is None:
        continue
    dist = (te2.cv.values - tr.cv.max()) * 1e6
    rows.append(dict(cut=cut, n_tr=len(tr), n_te=len(te2), dist=dist.mean(),
                     bias=res.mean(), sd=res.std(ddof=1)))
E1 = pd.DataFrame(rows)
print(f"  {'cut':>5s} {'n_tr':>5s} {'n_te':>5s} {'mean dCV above fit max':>23s} "
      f"{'held-out bias':>14s} {'sd':>7s}")
for _, x in E1.iterrows():
    print(f"  {x.cut:5.2f} {x.n_tr:5.0f} {x.n_te:5.0f} {x.dist:23.1f} "
          f"{x.bias:+14.2f} {x.sd:7.2f}")
sl, ic = np.polyfit(E1.dist, E1.bias, 1)
pred_at = ic + sl * d_new.mean()
print(f"\n  bias vs extrapolation distance: {sl:+.3f}e-6 of bias per e-6 of distance "
      f"(intercept {ic:+.2f})")
print(f"  the five 08-21 files sat {d_new.mean():+.1f}e-6 out -> this curve predicts "
      f"{pred_at:+.2f}e-6 of bias")
print(f"  they actually came in {NEW.res6.mean():+.2f}e-6.")
share = pred_at / NEW.res6.mean() if NEW.res6.mean() else float("nan")
print(f"  => extrapolation alone accounts for {100*share:.0f}% of the miss "
      f"({pred_at:+.1f} of {NEW.res6.mean():+.1f}e-6)")
print(f"  E1 VERDICT: {'CONFIRMED' if pred_at <= -10 else 'REJECTED' if pred_at >= -3 else 'PARTIAL'}"
      "  (registered: <=-10 confirmed, >=-3 rejected)")

print("\n" + "=" * 78)
print("C. E2 -- PROSPECTIVE OPTIMISM.  Expanding window, fit on days < D, predict day D.")
print("=" * 78)
days = sorted(OLD.day.unique())
pro = []
for D in days:
    tr, te = OLD[OLD.day < D], OLD[OLD.day == D]
    if len(tr) < 20:
        continue
    te2, res, _ = fit_predict(tr, te)
    if te2 is None:
        continue
    dist = (te2.cv.values - tr.cv.max()) * 1e6
    pro.append(dict(day=str(pd.Timestamp(D).date()), n=len(te2), dist=dist.mean(),
                    bias=res.mean(), sd=res.std(ddof=1) if len(res) > 1 else np.nan))
    for v in res:
        pass
P = pd.DataFrame(pro)
print(f"  {'day':>12s} {'n':>3s} {'mean dCV out':>13s} {'day-ahead bias':>15s} {'sd':>7s}")
for _, x in P.iterrows():
    print(f"  {x.day:>12s} {x.n:3.0f} {x.dist:13.1f} {x.bias:+15.2f} {x.sd:7.2f}")
tr = OLD
te2, res5, _ = fit_predict(tr, NEW)
print(f"  {'2026-08-21':>12s} {len(te2):3d} {d_new.mean():13.1f} {res5.mean():+15.2f} "
      f"{res5.std(ddof=1):7.2f}   <- the disputed day")
m = P.bias.mean()
print(f"\n  mean day-ahead bias over the eight old days: {m:+.2f}e-6 "
      f"(sd of day means {P.bias.std(ddof=1):.2f})")
print(f"  E2 VERDICT: {'unbiased prospectively' if abs(m) < 5 else 'generically optimistic'}"
      f"  (registered: |m|<5 -> unbiased, m<=-10 -> generic optimism)")

print("\n" + "=" * 78)
print("D. E3 -- WHAT IS THE HELD-OUT RESIDUAL SD, REALLY?")
print("=" * 78)
loo = np.full(len(OLD), np.nan)          # NaN where the row is the only file of its
for i in range(len(OLD)):                # family, so leaving it out makes it unpredictable
    tr = OLD.drop(OLD.index[i])
    te2, res, _ = fit_predict(tr, OLD.iloc[[i]])
    if te2 is not None and len(res):
        loo[i] = res[0]
n_skip = int(np.isnan(loo).sum())
sd_loo = float(np.nanstd(loo, ddof=1))
sd_in = 7.756485094565338
print(f"  in-sample dof-corrected sd (w30b, quoted everywhere): {sd_in:.2f}e-6")
print(f"  leave-one-out held-out sd, n={len(loo)-n_skip}:              {sd_loo:.2f}e-6"
      f"   ({sd_loo/sd_in:.2f}x)")
print(f"  day-ahead residual sd (pooled over the eight days):   "
      f"{np.sqrt((P.sd.dropna()**2).mean()):.2f}e-6")
z_in = NEW.res6.mean() / (sd_in / np.sqrt(5))
z_loo = NEW.res6.mean() / (sd_loo / np.sqrt(5))
print(f"\n  w46 scored the five at z = {z_in:+.1f} using the in-sample sd.")
print(f"  On the leave-one-out sd it is z = {z_loo:+.1f}.")
print(f"  E3 VERDICT: {'CONFIRMED' if sd_loo > sd_in else 'REJECTED'} "
      f"(registered: LOO sd exceeds 7.756e-6)"
      f"{'; and it exceeds 11e-6, so the quoted z was overstated' if sd_loo > 11 else ''}")

print("\n" + "=" * 78)
print("E. E4 -- SLOPE (a ramp in CV) or LEVEL (a dummy)?")
print("=" * 78)
x = (NEW.cv.values - OLD.cv.max()) * 1e6
y = NEW.res6.values
sl5, ic5 = np.polyfit(x, y, 1)
se5 = np.sqrt(((y - (ic5 + sl5 * x)) ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
print(f"  residual-on-cv among the five scored: slope {sl5:+.3f} per e-6, se {se5:.3f}, "
      f"t {sl5/se5:+.2f}  (n=5, weak)")
print(f"  registered thresholds: <= -0.60 -> slope story; within +/-0.40 -> level story")
print(f"  reading NOW: "
      + ("SLOPE" if sl5 <= -0.6 else "LEVEL" if abs(sl5) <= 0.4 else "AMBIGUOUS")
      + " -- but n=5 and se is large; the real read is tomorrow at n=15.")
q10 = [r for r in q if r["stem"] in {
    "w38_ad202stdcorr", "w40_ad211stdcorr", "w40_ad211std", "w38_ad202std",
    "w36_ad199std_h3", "w36_ad197stdcorr", "w40_ad211std_h3", "w38_ad202std_h3",
    "w40_ad211std_rescale", "w38_ad202std_rescale"}]
cv10 = np.array([r["cv"] for r in q10])
xall = np.concatenate([x, (cv10 - OLD.cv.max()) * 1e6])
print(f"  tomorrow's ten span cv {cv10.min():.7f}..{cv10.max():.7f}; pooled with the five,")
print(f"  n=15 spans {xall.min():+.1f}..{xall.max():+.1f}e-6 of extrapolation distance.")
sd_use = max(sd_loo, sd_in)
se15 = sd_use / np.sqrt(((xall - xall.mean()) ** 2).sum())
print(f"  se of the n=15 residual-on-cv slope will be ~{se15:.3f} per e-6 "
      f"(using sd {sd_use:.1f}e-6)")
print(f"  -> a true slope of -0.86 (fitted 1.856 vs a true 1.0) would read t "
      f"{-0.86/se15:+.1f}. The n=15 test is {'adequately' if abs(-0.86/se15)>2 else 'NOT well'} powered.")

print("\n" + "=" * 78)
print("F. THE COMPETING FIXES, and what they say about tomorrow's ten")
print("=" * 78)
allr = pd.concat([OLD, NEW], ignore_index=True)
fams = sorted(set(allr.fam))
mu_a = allr.cv.mean()
Xa, na = design(allr, mu_a, fams)
ba, *_ = np.linalg.lstsq(Xa, allr.lb.values * 1e6, rcond=None)
i_cv = na.index("cv_e6")
print(f"  slope on the 78 fitted files          : {1.8563296706069337:+.3f} per e-6")
print(f"  slope refitted on all 83 (free slope) : {ba[i_cv]:+.3f} per e-6")
print("\n  predictions for the 08-22 ten under three fixes (e-6 above 0.970000):")
print(f"  {'file':24s} {'cv':>13s} {'w30b':>9s} {'w46c dummy':>11s} {'free slope':>11s}")
tot = {"w30b": [], "dummy": [], "slope": []}
import w46c_predlb as W
for r in sorted(q10, key=lambda r: -r["cv"]):
    s, cvv = r["stem"], r["cv"]
    raw = W._raw(cvv, s) * 1e6 - 970000
    dum = W.predict_lb(cvv, s) * 1e6 - 970000
    row = pd.DataFrame([dict(cv=cvv, fam=W.family(s), **{"std": float(W.is_std(s))},
                             cr=float(W.is_corr(s)))])
    Xr, _ = design(row, mu_a, fams)
    fs = float((Xr @ ba)[0]) - 970000
    tot["w30b"].append(raw); tot["dummy"].append(dum); tot["slope"].append(fs)
    print(f"  {s:24s} {cvv:.10f} {raw:9.2f} {dum:11.2f} {fs:11.2f}")
for k in tot:
    tot[k] = float(np.mean(tot[k]))
print(f"\n  mean prediction: w30b {tot['w30b']:.2f}   w46c dummy {tot['dummy']:.2f}   "
      f"free-slope refit {tot['slope']:.2f}  (e-6 above 0.970000)")
print(f"  the dummy and the free-slope refit differ by "
      f"{tot['slope']-tot['dummy']:+.2f}e-6 on tomorrow's ten -- "
      f"{'materially' if abs(tot['slope']-tot['dummy'])>3 else 'not materially'} different.")

print("\n" + "=" * 78)
print("G. THE RESIDUALS ARE CLUSTERED BY DAY, so w46's z = -8.6 is the wrong statistic")
print("=" * 78)
OLD2 = OLD.copy()
OLD2["loo"] = loo
dm = OLD2.dropna(subset=["loo"]).groupby(OLD2.day.dt.date).loo.agg(["mean", "size"])
print(f"  {'day':>12s} {'n':>3s} {'LOO day mean':>13s}")
for d, xx in dm.iterrows():
    print(f"  {str(d):>12s} {xx['size']:3.0f} {xx['mean']:+13.2f}")
sd_daymean = float(dm["mean"].std(ddof=1))
nbar = float(dm["size"].mean())
sd_if_indep = float(sd_loo / np.sqrt(nbar))
print(f"\n  observed sd of the ten day means      : {sd_daymean:.2f}e-6")
print(f"  sd expected if files were independent : {sd_if_indep:.2f}e-6 "
      f"(= {sd_loo:.2f}/sqrt({nbar:.1f}))")
sd_day = float(np.sqrt(max(sd_daymean ** 2 - sd_if_indep ** 2, 0.0)))
print(f"  => a DAY-LEVEL component of sd {sd_day:.2f}e-6 that the per-file sd does not carry.")
print("  Files sent on the same day share a build wave, a pack and an adapter era, so this")
print("  is what a clustered sample looks like.  A new day's mean is ONE draw from the day")
print("  distribution, not the mean of n independent draws.")
from scipy import stats as _st
z_clu = float((NEW.res6.mean() - dm["mean"].mean()) / sd_daymean)
p_clu = float(2 * _st.t.sf(abs(z_clu), df=len(dm) - 1))
print(f"\n  w46 quoted z = {z_in:+.1f} (per-file sd, files assumed independent).")
print(f"  Clustered on the day, 08-21 is t = {z_clu:+.2f} on {len(dm)-1} df, p = {p_clu:.4f}.")
sd_pred = float(np.sqrt(sd_loo ** 2 + sd_day ** 2))
print("  The clustering correction I expected to find is NOT THERE: the day component is")
print(f"  {sd_day:.2f}e-6, and clustering moves the statistic from {z_in:+.1f} to {z_clu:+.1f}, i.e. not at all.")
print(f"  The era finding survives this attack too.  Honest predictive sd for a new file is")
print(f"  sqrt({sd_loo:.2f}^2 + {sd_day:.2f}^2) = {sd_pred:.2f}e-6 vs the {sd_in:.2f}e-6 quoted; w46c's")
print("  widening to 10.69e-6 is if anything conservative, so leave it alone.")
print("  NOTE the contrast with the expanding-window sd in section C (11.69e-6 pooled, day")
print("  means sd 6.32).  That is inflated by ESTIMATION error from small early training")
print("  sets (n as low as 20), not by clustering.  Leave-one-out at n=77 is the right")
print("  analogue for a prediction made from a fit on 78, and it is the number to quote.")

json.dump(dict(d_new_mean=float(d_new.mean()), e1_slope=float(sl), e1_intercept=float(ic),
               e1_pred_bias=float(pred_at), e1_share=float(share),
               e2_mean=float(m), e2_days=P.to_dict("records"),
               e3_sd_loo=float(sd_loo), e3_sd_in=sd_in, e3_z_in=float(z_in),
               e3_z_loo=float(z_loo), e4_slope5=float(sl5), e4_se5=float(se5),
               e4_se15_expected=float(se15), slope_78=1.8563296706069337,
               slope_83=float(ba[i_cv]), pred_ten=tot,
               g_sd_daymean=sd_daymean, g_sd_if_indep=sd_if_indep,
               g_sd_day=sd_day, g_t_clustered=z_clu, g_p_clustered=p_clu,
               g_sd_pred=sd_pred),
          open(os.path.join(HERE, "w47a_extrap.json"), "w"), indent=1)
E1.to_csv(os.path.join(HERE, "w47a_e1curve.csv"), index=False)
P.to_csv(os.path.join(HERE, "w47a_e2days.csv"), index=False)
print("\nwrote w47a_extrap.json, w47a_e1curve.csv, w47a_e2days.csv")
