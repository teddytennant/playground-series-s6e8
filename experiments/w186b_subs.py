"""Turn the w186 paired arms into submission CSVs.

Both files are MEASUREMENTS on a graded board, not picks: the competition closed
2026-08-31 and nothing sent now can be selected. Sending the pair prices the `cdfd_*`
channel on the real 296,302-row test set instead of on CV alone, exactly as w185b did
one level down.
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import OOF as PACK, ROOT, SUB, TARGET, get_folds, load_raw  # noqa: E402

OOF = os.path.join(ROOT, "oof_w186")   # the side dir, not the curated pack
from sklearn.metrics import roc_auc_score  # noqa: E402

tr, te = load_raw()
y = tr[TARGET].astype(int).to_numpy()
folds = get_folds(y)
rows = []
for arm in ("ctrl", "cdf"):
    n = f"w186_lgbmfrac_{arm}"
    oof = np.load(os.path.join(OOF, f"oof_{n}.npy"))
    tp = np.load(os.path.join(OOF, f"test_{n}.npy"))
    cv = roc_auc_score(y, oof)
    per = [roc_auc_score(y[v], oof[v]) for _, v in folds]
    out = os.path.join(SUB, f"{n}.csv")
    pd.DataFrame({"id": te["id"].values, TARGET: tp}).to_csv(out, index=False)
    rows.append((n, cv, per, out))
    print(f"{n:24s} CV {cv:.7f}  folds " + " ".join(f"{p:.6f}" for p in per))
    print(f"  -> {out}  n={len(tp)}  range [{tp.min():.6f}, {tp.max():.6f}]")

(nc, cc, pc, _), (nd, cd, pd_, _) = rows
print(f"\ndelta CV = {(cd - cc) * 1e6:+.1f}e-6")
print("per-fold  " + "  ".join(f"{(b - a) * 1e6:+.1f}" for a, b in zip(pc, pd_)) + "  e-6")
print(f"folds won by cdf: {sum(b > a for a, b in zip(pc, pd_))}/5")

# rank correlation between the two arms: how much of a distinct member is this?
from scipy.stats import spearmanr  # noqa: E402
o1 = np.load(os.path.join(OOF, "oof_w186_lgbmfrac_ctrl.npy"))
o2 = np.load(os.path.join(OOF, "oof_w186_lgbmfrac_cdf.npy"))
print(f"spearman(ctrl oof, cdf oof) = {spearmanr(o1, o2).statistic:.6f}")
stored = np.load(os.path.join(PACK, "oof_lgbm_tuned_lat_frac.npy"))
print(f"max |ctrl - stored lgbm_tuned_lat_frac| = {np.abs(o1 - stored).max():.3e}")
