#!/usr/bin/env python3
"""
w44b -- score w44a's per-member law against ARM 211, then re-fit and PRE-REGISTER ARM 217.

HONESTY NOTE, stated first because it is the whole value of the exercise:
  * ARM 211's h3 base was written to w40f_build.log at 00:57 UTC; w44a ran at 01:01 UTC.
    The number was ON DISK before the fit. It is therefore a HELD-OUT test, NOT a
    pre-registration -- w44a's EVENTS list provably contains only 190->202 events, so the
    fit cannot have used it, but "I did not look" is a weaker claim than "it did not exist".
  * ARM 217's h3 base does NOT exist anywhere yet (w42e is still blocked on w40f). The
    prediction printed at the bottom IS a genuine pre-registration.
"""
import numpy as np, csv, pathlib, math

ROOT = pathlib.Path(__file__).resolve().parent.parent
cv = {}
with open(ROOT / "experiments/w23b_sendqueue.csv") as fh:
    for r in csv.DictReader(fh):
        if r.get("cv"):
            cv[r["file"].replace(".csv", "")] = float(r["cv"])

def from_log(log, pat):
    for line in (ROOT / "experiments" / log).read_text().splitlines():
        if pat in line and "OOF AUC" in line:
            return float(line.split("OOF AUC")[1].split()[0])
    raise SystemExit(f"no base AUC for {pat} in {log}")

ARMS = {
    190: cv["w27_ad190std_h3"], 194: cv["w29_ad194std_h3"],
    196: cv["w34_ad196std_h3"], 195: cv["w34_ad195std_h3"],
    197: from_log("w36g_build.log", "w36_ad197std_h3"),
    199: from_log("w36b_build.log", "w36_ad199std_h3"),
    202: from_log("w38d_build.log", "w38_ad202std_h3"),
    211: from_log("w40f_build.log", "w40_ad211std_h3"),   # the held-out point
}
FIT = [(190,194,4), (194,196,2), (195,197,2), (197,199,2), (199,202,3)]
FLOOR = 4.0

def origin_fit(ev):
    n = np.array([e[2] for e in ev], float)
    d = np.array([(ARMS[e[1]] - ARMS[e[0]]) * 1e6 for e in ev])
    slope = (n * d).sum() / (n ** 2).sum()
    resid = d - slope * n
    se = np.sqrt((resid ** 2).sum() / (len(n) - 1) / (n ** 2).sum())
    return slope, se, n, d

slope, se, n, d = origin_fit(FIT)
print("=" * 78); print("w44b  HELD-OUT TEST of the w44a per-member law"); print("=" * 78)
print(f"\nw44a fit (190->202 only): slope {slope:+.2f}e-6/member  se {se:.2f}")

pred, pse = slope * 9, se * 9
act = (ARMS[211] - ARMS[202]) * 1e6
lo, hi = pred - 1.96 * pse, pred + 1.96 * pse
print(f"\nARM 211  (+9 members, ext_members15) -- the largest single dose ever added here")
print(f"  predicted delta   {pred:+.2f}e-6   95% CI [{lo:+.2f}, {hi:+.2f}]e-6")
print(f"  ACTUAL delta      {act:+.2f}e-6      (h3 base {ARMS[211]:.10f})")
print(f"  per member        {act/9:+.3f}e-6")
print(f"  inside 95% CI?    {'YES' if lo <= act <= hi else 'NO -- law falsified'}")
print(f"  z vs prediction   {(act-pred)/pse:+.2f}")
print(f"\n  >> The law is NOT falsified, but the point prediction was {pred:+.1f} and the")
print(f"     best-powered arm in the record returned {act:+.2f}. w44a gave this arm")
print(f"     P(clear the {FLOOR:.0f}e-6 floor) = 0.72. It cleared {act/FLOOR:.2f} floors.")

# ---- re-fit with 211 in ----------------------------------------------------------
FIT2 = FIT + [(202, 211, 9)]
slope2, se2, n2, d2 = origin_fit(FIT2)
print("\n" + "=" * 78); print("RE-FIT including ARM 211"); print("=" * 78)
print(f"\n{'event':>12} {'n':>3} {'delta e-6':>10} {'per-mem':>9}")
for (a, b, k), dd in zip(FIT2, d2):
    print(f"{a:>4}->{b:<6} {k:>3} {dd:>+10.2f} {dd/k:>+9.2f}")
print(f"\norigin slope  {slope:+.2f} -> {slope2:+.2f}e-6/member   se {se2:.2f}   t {slope2/se2:+.2f}")
print(f"  the 9-member point carries {81/ (n2**2).sum():.0%} of the leverage and halves the slope")
per = d2 / n2
print(f"unweighted per-member mean {per.mean():+.2f}e-6  se {per.std(ddof=1)/np.sqrt(len(per)):.2f}"
      f"  t {per.mean()/(per.std(ddof=1)/np.sqrt(len(per))):+.2f}")
cum = (ARMS[211] - ARMS[190]) * 1e6
print(f"\ncumulative 190->211  {cum:+.2f}e-6 over {int(n2.sum())} members "
      f"= {cum/n2.sum():+.2f}e-6/member  ({cum/FLOOR:.1f} rebuild floors)")

# ---- the genuine prereg ----------------------------------------------------------
print("\n" + "=" * 78)
print("*** PRE-REGISTRATION: ARM 217 (+6 members, ext_members16) ***")
print("    w42e has not started building. This number exists nowhere on disk.")
print("=" * 78)
p, ps = slope2 * 6, se2 * 6
cvp = ARMS[211] + p * 1e-6
print(f"\n  point delta          {p:+.2f}e-6   95% CI [{p-1.96*ps:+.2f}, {p+1.96*ps:+.2f}]e-6")
print(f"  predicted h3 base    {cvp:.10f}")
print(f"  P(delta > {FLOOR:.0f}e-6 floor)  {1-0.5*(1+math.erf((FLOOR-p)/(ps*math.sqrt(2)))):.2f}")
print(f"  bar to become WANTED 0.9701440  -> needs {(0.9701440-ARMS[211])*1e6:+.2f}e-6, "
      f"P = {1-0.5*(1+math.erf(((0.9701440-ARMS[211])*1e6-p)/(ps*math.sqrt(2)))):.2f}")
print("\n  REGISTERED CALL: ARM 217 lands as another null. If it does, the import line is")
print("  closed on six consecutive events, and no further arm should be built.")
