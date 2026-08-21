"""w43c -- why 164 of 249 members "fail" the w43a KS null, and why KS must NOT become a gate.

w43a compared each member's OOF and test MARGINAL distributions with a two-sample KS test,
on the reasoning that train and test are iid draws from one generator so a correctly-exported
member's two sides are two samples from the same marginal. Under that null the 99.99% point
is 0.00428. **164 of 249 members exceed it.** A screen that rejects two thirds of a pack that
has been scoring 0.9701 for weeks is not detecting 164 defects; the null is wrong.

This script tests the specific reason it is wrong, so a later run does not either re-derive
this from scratch or -- far worse -- act on w43a's KS column as if it were a defect count.

THE HYPOTHESIS.  The two sides are NOT produced the same way, even for a perfect export:

  OOF   row i is predicted by the ONE fold model that did not train on row i.
        The OOF vector is therefore a MIXTURE of 5 models' outputs.
  test  is almost always the MEAN of all 5 fold models' predictions.

A mixture of 5 correlated predictors and the mean of those same 5 have different marginals
by construction -- the mean is shrunk toward the centre and has thinner tails -- even when
every fold model is identical in distribution. So KS > 0 is the EXPECTED signature of a
correct 5-fold export, not evidence of anything.

THE TESTABLE PREDICTION.  Averaging 5 lattices SMOOTHS: if a member's per-fold output lives
on a coarse lattice of k values (lookup tables, shallow trees, quantised scores), the OOF
side stays on that coarse lattice while the mean of 5 lands on a far finer one. The
distributional gap the averaging opens is therefore LARGEST exactly where the OOF side is
most discrete. Prediction: KS should rise with OOF coarseness, measured as the fraction of
distinct values on the OOF side. If instead KS were tracking broken imports, it would have
no reason to care about lattice resolution at all.

A defect like w42 s3's (test side 5x the OOF side) is a SCALE fault and `ratio` catches it.
KS conflates that with an averaging artefact it cannot separate. Hence: ratio gates, KS does not.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, LIB, OOF  # noqa: E402

N_TR, N_TE = 691369, 296302
aud = pd.read_csv(os.path.join(HERE, "w43a_scaleaudit.csv"))

DIRS = [os.path.join(LIB, "oof"), OOF] + sorted(
    (os.path.join(DATA, d) for d in os.listdir(DATA) if d.startswith("ext_members")),
    key=lambda p: (len(p), p))
loc = {}
for d in DIRS:
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        if fn.startswith("oof_") and fn.endswith(".npy"):
            loc.setdefault(fn[4:-4], d)

rows = []
for _, r in aud.iterrows():
    d = loc.get(r.member)
    if d is None:
        continue
    o = np.load(os.path.join(d, f"oof_{r.member}.npy")).astype(np.float64)
    t = np.load(os.path.join(d, f"test_{r.member}.npy")).astype(np.float64)
    rows.append(dict(member=r.member, ks=r.ks, ratio=r.ratio,
                     res_oof=len(np.unique(o)) / N_TR, res_test=len(np.unique(t)) / N_TE))
    print(f"  {len(rows):3d}/{len(aud)} {r.member:44s} res_oof {rows[-1]['res_oof']:.4f} "
          f"res_test {rows[-1]['res_test']:.4f}  ks {r.ks:.5f}", flush=True)

df = pd.DataFrame(rows)
df["smoothing"] = df.res_test - df.res_oof     # >0 == the test side is FINER than the OOF side
df.to_csv(os.path.join(HERE, "w43c_ksconfound.csv"), index=False)

from scipy.stats import spearmanr
print("\n" + "=" * 78)
print(f"n = {len(df)} members")
print(f"  spearman(ks, res_oof)     = {spearmanr(df.ks, df.res_oof).statistic:+.3f}   "
      "(prediction: NEGATIVE -- coarser OOF lattice => bigger KS)")
print(f"  spearman(ks, smoothing)   = {spearmanr(df.ks, df.smoothing).statistic:+.3f}   "
      "(prediction: POSITIVE -- averaging refines the test side)")
print(f"  spearman(ks, |ratio - 1|) = {spearmanr(df.ks, (df.ratio - 1).abs()).statistic:+.3f}   "
      "(if KS tracked SCALE faults this would dominate)")

q = pd.qcut(df.res_oof, 4, labels=["coarsest OOF", "q2", "q3", "finest OOF"])
print("\nmedian KS by OOF-lattice-resolution quartile:")
print(df.groupby(q, observed=True).ks.agg(["median", "max", "size"]).to_string(float_format="%.5f"))
print("\n--- the 10 coarsest-OOF members (the lookup/quantised tier) ---")
print(df.nsmallest(10, "res_oof")[["member", "res_oof", "res_test", "ks", "ratio"]]
      .to_string(index=False, float_format="%.5f"))
