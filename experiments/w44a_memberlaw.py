#!/usr/bin/env python3
"""
w44a -- What is one imported member actually worth?

Every import arm on record is a NESTED superset of the one before it, built by the same
script (blend_lab.py --standardize --build) with the same --drop list and one more
--extra-dirs entry. That makes the arm sequence a controlled dose-response experiment
that nobody here has ever read as one: dose = members added, response = h3-base CV.

Measured on the h3 BASE (not the w21a-corrected object), because the correction step adds
a near-constant +4.5e-6 and would only add its own noise to the contrast.

Prints a per-member slope with an honest error bar, and the resulting predictions for the
two arms that have NOT landed yet (211, 217). Run before they finish -> a real prereg.
"""
import numpy as np, csv, sys, pathlib, math

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---- arm h3-base CVs, each read from the artefact that produced it -------------------
# sendqueue holds the ones that reached a submission file; the two newest are only in logs.
QUEUE = ROOT / "experiments/w23b_sendqueue.csv"
cv = {}
with open(QUEUE) as fh:
    for r in csv.DictReader(fh):
        if r.get("cv"):
            cv[r["file"].replace(".csv", "")] = float(r["cv"])

def from_log(log, pat):
    for line in (ROOT / "experiments" / log).read_text().splitlines():
        if pat in line and "OOF AUC" in line:
            return float(line.split("OOF AUC")[1].split()[0])
    raise SystemExit(f"no base AUC for {pat} in {log}")

ARMS = {
    190: cv["w27_ad190std_h3"],
    194: cv["w29_ad194std_h3"],
    196: cv["w34_ad196std_h3"],          # w34a pack, om_cat still IN
    195: cv["w34_ad195std_h3"],          # same pack, om_cat dropped -> the w36+ basis
    197: from_log("w36g_build.log", "w36_ad197std_h3"),
    199: from_log("w36b_build.log", "w36_ad199std_h3"),
    202: from_log("w38d_build.log", "w38_ad202std_h3"),
}

# ---- the group events: (from, to, dir added, n members in that dir) ------------------
# nesting verified from the X= lines of the run scripts; n from `ls data/ext_membersN/oof_*.npy`
EVENTS = [
    (190, 194, "ext_members8",  4),
    (194, 196, "ext_members10", 2),
    (195, 197, "ext_members12", 2),
    (197, 199, "ext_members11", 2),
    (199, 202, "ext_members14", 3),
]
PENDING = [
    (202, 211, "ext_members15", 9),
    (211, 217, "ext_members16", 6),
]

FLOOR = 4.0   # w29 rebuild reproducibility floor, in units of 1e-6 AUC

print("=" * 78)
print("w44a  DOSE-RESPONSE: what one imported member is worth on the h3 base")
print("=" * 78)
print(f"\n{'event':>12}  {'dir':<15} {'n':>2}  {'delta e-6':>10}  {'per-mem':>8}  {'floors':>7}")
rows = []
for a, b, d, n in EVENTS:
    delta = (ARMS[b] - ARMS[a]) * 1e6
    rows.append((n, delta))
    print(f"{a:>4}->{b:<6}  {d:<15} {n:>2}  {delta:>+10.2f}  {delta/n:>+8.2f}  {delta/FLOOR:>+7.2f}")

n_arr = np.array([r[0] for r in rows], float)
d_arr = np.array([r[1] for r in rows], float)

# per-member values, treated as 5 independent readings of one quantity
per = d_arr / n_arr
m, sd = per.mean(), per.std(ddof=1)
se = sd / np.sqrt(len(per))
print(f"\nper-member readings   : {np.array2string(per, precision=2, floatmode='fixed')}")
print(f"mean per member       : {m:+.2f}e-6   sd {sd:.2f}   se {se:.2f}   t {m/se:+.2f}   n={len(per)}")

# slope through the origin (a member is worth what it is worth; no free intercept)
slope = (n_arr * d_arr).sum() / (n_arr ** 2).sum()
resid = d_arr - slope * n_arr
s_slope = np.sqrt((resid ** 2).sum() / (len(n_arr) - 1) / (n_arr ** 2).sum())
print(f"origin slope (wtd)    : {slope:+.2f}e-6 per member   se {s_slope:.2f}   t {slope/s_slope:+.2f}")

cum_n, cum_d = n_arr.sum(), (ARMS[202] - ARMS[190]) * 1e6
print(f"\ncumulative 190->202   : {cum_d:+.2f}e-6 over {int(cum_n)} members "
      f"({cum_d/cum_n:+.2f}e-6/member, {cum_d/FLOOR:+.1f} rebuild floors)")
print(f"  (includes the one-off om_cat drop at 196->195: "
      f"{(ARMS[195]-ARMS[196])*1e6:+.2f}e-6, not a member add)")

print(f"\nNOT ONE single group event clears the {FLOOR:.0f}e-6 rebuild floor except "
      f"{'ext_members11' if max(d_arr) == d_arr[3] else '?'} "
      f"at {d_arr[3]:+.2f}e-6 ({d_arr[3]/FLOOR:.1f} floors).")

# ---- the prereg ---------------------------------------------------------------------
print("\n" + "=" * 78)
print("PRE-REGISTERED PREDICTIONS for the two arms that have NOT landed")
print("=" * 78)
base = ARMS[202]
out = {}
for a, b, d, n in PENDING:
    pred_delta = slope * n
    pred_se = s_slope * n
    base_cv = base if a == 202 else out[a]["cv"]
    cvpred = base_cv + pred_delta * 1e-6
    out[b] = {"cv": cvpred}
    lo, hi = pred_delta - 1.96 * pred_se, pred_delta + 1.96 * pred_se
    print(f"\nARM {b}  (+{n} members from {d}, on top of ARM {a})")
    print(f"  point delta   {pred_delta:+.2f}e-6   95% CI [{lo:+.2f}, {hi:+.2f}]e-6")
    print(f"  predicted h3-base CV  {cvpred:.10f}")
    print(f"  P(delta > +{FLOOR:.0f}e-6 rebuild floor) = "
          f"{1 - 0.5*(1+math.erf((FLOOR-pred_delta)/(pred_se*np.sqrt(2)))):.2f}")

print("\n" + "-" * 78)
print("FALSIFIABLE: if ARM 211's h3 base lands outside its 95% CI, the linear")
print("per-member law is wrong and the import line should be closed on that evidence.")
print("-" * 78)
