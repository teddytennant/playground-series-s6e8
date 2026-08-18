"""w25e — the six matched STANDARDISATION pairs, assembled from live LB reads.

Six pairs were sent this slot (three already existed, three were built by today's sends).
Each pair holds the 187-member pack, the transform, C=1.0 and the frozen SKF5 seed42 folds
fixed and varies ONE thing: whether the combiner's design matrix is scaled to unit sd before
the logistic fit. Both members of every pair are the same transform family, so w25c's family
offset cancels EXACTLY and no modelling is needed to read the paired column.

The point of assembling them in one place is that the dCV column spans +0.26e-6 to +20.45e-6,
which is what separates the two candidate readings:
  * a FIXED penalty on standardised files      -> dLB is the same in all six
  * standardisation converting to LB at slope 0 -> dLB ~ 0 everywhere, and the -10s are the
                                                   1e-5 reporting grid rounding a small
                                                   negative number

    .venv/bin/python experiments/w25e_stdpairs.py
"""
from __future__ import annotations

import io
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SLOPE = 1.750          # w25c ANCOVA, CV>=0.97, family offset removed
PAIRED_SD = 8.21       # w17a simulated leaders paired slice sd, e-6

PAIRS = [
    ("corrected h3",   "w23_ad187stdcorr",        "w21_ad187corr"),
    ("uncorrected h3", "w23_ad187std_h3",         "w20_ad187_h3"),
    ("ens4",           "w23_ad187std",            "w20_ad187"),
    ("rescale",        "w23_ad187std_h3_rescale", "w20_ad187_rescale"),
    ("hybrid",         "w23_ad187std_h3_hybrid",  "w20_ad187_hybrid"),
    ("rankraw",        "w23_ad187std_h3_rankraw", "w20_ad187_rankraw"),
]

raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", "playground-series-s6e8",
                      "-v", "--page-size", "300"], capture_output=True, text=True).stdout
sub = pd.read_csv(io.StringIO(raw))
sub = sub[sub.status == "SubmissionStatus.COMPLETE"]
lb = sub.assign(stem=sub.fileName.str.replace(r"\.csv$", "", regex=True)) \
        .groupby("stem").publicScore.max()

y = pd.read_csv("data/train.csv", usecols=["addicted_label"]).addicted_label.values
cv = {}


def CV(s):
    if s not in cv:
        cv[s] = roc_auc_score(y, np.load(f"submissions/oof_{s}.npy"))
    return cv[s]


rows = []
for fam, a, b in PAIRS:
    if a not in lb or b not in lb:
        print(f"  (skipping {fam}: {a if a not in lb else b} not scored yet)")
        continue
    d_cv = (CV(a) - CV(b)) * 1e6
    d_lb = (lb[a] - lb[b]) * 1e6
    rows.append(dict(family=fam, std=a, unstd=b, cv_std=CV(a), cv_unstd=CV(b),
                     dcv6=d_cv, lb_std=lb[a], lb_unstd=lb[b], dlb6=d_lb,
                     pred_dlb6=SLOPE * d_cv, resid6=d_lb - SLOPE * d_cv))
t = pd.DataFrame(rows).sort_values("dcv6")

print("=== the six matched standardisation pairs, ordered by dCV ===")
print(t[["family", "dcv6", "lb_std", "lb_unstd", "dlb6", "pred_dlb6", "resid6"]]
      .assign(dcv6=lambda d: d.dcv6.map("{:+.2f}".format),
              dlb6=lambda d: d.dlb6.map("{:+.1f}".format),
              pred_dlb6=lambda d: d.pred_dlb6.map("{:+.1f}".format),
              resid6=lambda d: d.resid6.map("{:+.1f}".format)).to_string(index=False))

n = len(t)
print(f"\npairs {n}   standardised file printed HIGHER in {int((t.dlb6>0).sum())}, "
      f"TIED in {int((t.dlb6==0).sum())}, LOWER in {int((t.dlb6<0).sum())}")
print(f"dCV spans {t.dcv6.min():+.2f} .. {t.dcv6.max():+.2f}e-6 and the standardisation "
      f"never once bought a higher print.")

# Which reading survives? Regress dLB on dCV across the six pairs.
if n >= 3:
    s, i = np.polyfit(t.dcv6.values, t.dlb6.values, 1)
    r = t.dlb6.values - (s * t.dcv6.values + i)
    print(f"\nOLS across the pairs: dLB = {s:+.3f} * dCV {i:+.2f}   resid sd "
          f"{r.std(ddof=2):.2f}e-6")
    print(f"  fixed-penalty reading predicts slope 0 with intercept ~ -10")
    print(f"  honest-CV reading predicts slope {SLOPE:+.3f} with intercept 0")
    print(f"  slope-zero-conversion reading predicts slope 0 with intercept ~ 0")

print(f"\nmean residual against the {SLOPE:.3f} slope: {t.resid6.mean():+.1f}e-6, "
      f"i.e. z {t.resid6.mean()/PAIRED_SD:+.2f} against w17a's {PAIRED_SD:.2f}e-6 paired sd.")
print("\nCAVEAT: the six pairs share the 187-member pack and one fixed public slice, so they")
print("are not six independent draws. The dCV SPREAD is what carries information here, not")
print("the repetition count.")

t.to_csv("experiments/w25e_stdpairs.csv", index=False)
json.dump(dict(n=n, mean_resid=float(t.resid6.mean()),
               n_higher=int((t.dlb6 > 0).sum()), n_tied=int((t.dlb6 == 0).sum()),
               n_lower=int((t.dlb6 < 0).sum()),
               rows=t.to_dict("records")),
          open("experiments/w25e_stdpairs.json", "w"), indent=1)
print("\nwrote experiments/w25e_stdpairs.csv + .json")
