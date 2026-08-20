"""w40c — cheap gates over `yadoy666/94-verified-oof-gpu-accelerated-meta-stack`.

The w40a sweep's largest find by far: a single kernel shipping union94_oof_matrix.npy
(691369, 94) and union94_test_matrix.npy (296302, 94) with a names file. Both row counts are
ours exactly and the kernel log prints folds=[138274 x4, 138273], a 5-fold split of our train.

⚠ WHAT THIS IS NOT. This is an AGGREGATION notebook -- its own log says "Loading 82 base
prediction pairs / 12 additional", so the 94 streams come from other people's notebooks and
the es-on-val provenance is per-source and unknown here. w34 3.1's rule (a foreign partition
is worse than es-on-val) applies to each stream separately. Nothing here imports anything.

This script runs only the cheap gates, because they are the ones that can DISQUALIFY:
  G1 solo AUC under our y, and the per-fold AUC spread under OUR get_folds().
  G2 an implausibly high solo AUC is the es-on-val / foreign-partition signature.
  G3 correlation against members we ALREADY hold, which is how a repackaged duplicate is
     caught -- many names (pub_tabm, pub_rmlp, realmlp, lat_*, lookup) look like streams
     this workspace has imported under its own names.
The expensive gate 4 (per-fold reproduction against the author's printed numbers) cannot be
run: an aggregator prints no per-member fold AUCs. That is recorded as a LIMITATION, not
waved through.
"""
from __future__ import annotations
import os, sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402

SRC = os.path.join(ROOT, "notebooks", "w40", "out",
                   "yadoy666_94-verified-oof-gpu-accelerated-meta-stack")
N_TR, N_TE = 691369, 296302

names = json.load(open(os.path.join(SRC, "union94_member_names.json")))
O = np.load(os.path.join(SRC, "union94_oof_matrix.npy"), mmap_mode="r")
T = np.load(os.path.join(SRC, "union94_test_matrix.npy"), mmap_mode="r")
assert O.shape == (N_TR, len(names)) and T.shape == (N_TE, len(names)), (O.shape, T.shape)

tr, _te = load_raw()
y = tr[TARGET].values
folds = get_folds(y)
print(f"y {y.shape} base rate {y.mean():.6f};  {len(folds)} folds "
      f"{[len(v) for _, v in folds]}")

# --- our own pack, for the duplicate check -------------------------------------------
ours = {}
for d in sorted(os.listdir(DATA)):
    p = os.path.join(DATA, d)
    if not (d.startswith("ext_members") or d == "oof") or not os.path.isdir(p):
        continue
    for f in os.listdir(p):
        if f.startswith("oof_") and f.endswith(".npy"):
            ours.setdefault(f[4:-4], os.path.join(p, f))
for f in os.listdir(os.path.join(ROOT, "oof")):
    if f.startswith("oof_") and f.endswith(".npy"):
        ours.setdefault(f[4:-4], os.path.join(ROOT, "oof", f))
print(f"{len(ours)} of our own oof vectors on disk for the duplicate check")

OURS = {}
for k, p in ours.items():
    try:
        v = np.load(p, mmap_mode="r")
        if v.shape == (N_TR,):
            OURS[k] = np.asarray(v, dtype=np.float64)
    except Exception:
        pass
print(f"{len(OURS)} of them are ({N_TR},) and usable\n")
OM = np.array([OURS[k] for k in OURS]) if OURS else np.zeros((0, N_TR))
OK = list(OURS)
OMr = np.array([pd.Series(v).rank().values for v in OM]) if len(OM) else OM

rows = []
for j, nm in enumerate(names):
    v = np.asarray(O[:, j], dtype=np.float64)
    if not np.isfinite(v).all():
        rows.append(dict(member=nm, solo=np.nan, note="non-finite")); continue
    solo = roc_auc_score(y, v)
    per = [roc_auc_score(y[va], v[va]) for _, va in folds]
    vr = pd.Series(v).rank().values
    if len(OMr):
        c = np.array([np.corrcoef(vr, w)[0, 1] for w in OMr])
        k = int(np.argmax(c)); mx, near = float(c[k]), OK[k]
    else:
        mx, near = np.nan, ""
    rows.append(dict(member=nm, solo=solo, fold_sd=float(np.std(per)),
                     fold_min=min(per), fold_max=max(per), maxcorr=mx, nearest=near))
    print(f"  {nm:34s} solo {solo:.6f}  foldsd {np.std(per):.2e}  "
          f"maxcorr {mx:.5f} vs {near}")

df = pd.DataFrame(rows).sort_values("solo", ascending=False)
df.to_csv(os.path.join(HERE, "w40c_union94_vet.csv"), index=False)
print(f"\nwrote experiments/w40c_union94_vet.csv")
print(f"\nsolo AUC: max {df.solo.max():.6f}  median {df.solo.median():.6f}  "
      f"min {df.solo.min():.6f}")
print(f"members with maxcorr < 0.99 against everything we hold: "
      f"{int((df.maxcorr < 0.99).sum())} of {len(df)}")
print(df.head(12).to_string(index=False))
