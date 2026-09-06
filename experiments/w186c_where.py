"""Where does the cdfd_* gain land? Tests the mechanism w186a pre-registered.

w186a predicted near-null and named the one channel that could still pay: `cdfd_c` is
estimated from a DIFFERENT sample of the generator, so it is a denoised prior where
`TE_c` -- estimated on the competition frame at exact-value resolution -- is noisiest.
If that is the mechanism, the gain concentrates on rows whose exact value is RARE in
train, and it should be flat in the number of missing fields.

Free: reads the two saved OOF arrays, refits nothing.
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT, TARGET, load_raw  # noqa: E402

OOF = os.path.join(ROOT, "oof_w186")   # the side dir, not the curated pack
from sklearn.metrics import roc_auc_score  # noqa: E402

tr, _ = load_raw()
y = tr[TARGET].astype(int).to_numpy()
a = np.load(os.path.join(OOF, "oof_w186_lgbmfrac_ctrl.npy"))
b = np.load(os.path.join(OOF, "oof_w186_lgbmfrac_cdf.npy"))


def band(mask, label):
    if mask.sum() < 2000 or y[mask].min() == y[mask].max():
        return
    ca, cb = roc_auc_score(y[mask], a[mask]), roc_auc_score(y[mask], b[mask])
    # AUC deltas are not scale-free: a band sitting at 0.928 has three times the
    # headroom of one at 0.975, so a raw gradient across bands can be pure headroom.
    # `hr` divides by (1 - ctrl AUC) to take that out.
    print(f"  {label:26s} n={mask.sum():>7,}  ctrl {ca:.6f}  cdf {cb:.6f}  "
          f"{(cb - ca) * 1e6:+8.1f}e-6   hr {(cb - ca) / (1 - ca) * 1e3:+6.2f}e-3")


print(f"overall  ctrl {roc_auc_score(y, a):.7f}  cdf {roc_auc_score(y, b):.7f}  "
      f"{(roc_auc_score(y, b) - roc_auc_score(y, a)) * 1e6:+.1f}e-6\n")

feat = [c for c in tr.columns if c not in ("id", TARGET)]
nm = tr[feat].isna().sum(axis=1).to_numpy()
print("by number of missing fields")
for k in range(4):
    band(nm == k, f"n_missing={k}")
band(nm >= 4, "n_missing>=4")

print("\nby train-frequency of the row's daily_screen_time_hours value")
d = tr["daily_screen_time_hours"]
cnt = d.map(d.value_counts()).to_numpy()
cnt = np.where(np.isnan(d.to_numpy()), -1, cnt)
band(cnt < 0, "daily missing")
edges = [0, 200, 400, 600, 900, 10 ** 9]
for lo, hi in zip(edges[:-1], edges[1:]):
    band((cnt >= lo) & (cnt < hi), f"value seen {lo}-{hi}x")

print("\nby age (the one column with 18 distinct values -- TE is never short of data)")
for lo, hi in [(0, 25), (25, 35), (35, 45), (45, 200)]:
    band((tr["age"] >= lo) & (tr["age"] < hi), f"age {lo}-{hi}")
