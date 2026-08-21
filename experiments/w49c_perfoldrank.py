"""w49c -- THE CONTROL. Does per-fold rank normalisation alone explain hboyang_mix's +886e-6?

w49b established the mechanism beyond reasonable doubt. `oof_hboyang_mix` is not a raw
prediction vector: its 691,369 values form 138,274 tie-groups of EXACTLY five (plus one group
of four, because 691,369 = 5*138,273 + 4), the values sit exactly at (5j+2.5)/691369 -- the
signature of (rankdata(average) - 0.5)/n -- and every single group of five contains exactly
one row from each of our five folds (20,000 of 20,000 groups checked; the chance rate is
5!/5^5 = 3.8%). The only procedure that produces that is RANKING EACH FOLD'S PREDICTIONS
SEPARATELY and concatenating: a row's value is its WITHIN-FOLD rank.

⚠ AND THE TEST VECTOR IS NOT LIKE THAT. `test_hboyang_mix` has 296,302 distinct values over
296,302 rows -- a single global rank, no ties. The two partitions were normalised differently,
which is exactly the asymmetry w49b flagged before knowing the mechanism.

WHY THAT INFLATES THE OOF AUC AND CANNOT INFLATE THE LB. Global AUC over a concatenated OOF is
dragged down whenever the five folds' predictions are on slightly different scales, because a
high-scoring row in a "cold" fold is ranked below a low-scoring row in a "hot" one. Ranking
within each fold removes that mismatch and lifts the pooled AUC toward the average of the
per-fold AUCs. It is not leakage -- no target is used -- but it IS a free gain that the test
set cannot receive, because the test set has no folds to normalise within. So a member treated
this way is scored on a different, more favourable footing than the 176 members it was
compared against, all of which are raw.

THE CONTROL, and the reading, fixed before it is run:
Take strong members whose OOF vectors are raw, apply the SAME per-fold rank transform, and
measure the lift. Let L = median lift over the controls.
  L >= 400e-6  -> the +886e-6 outlier is substantially a NORMALISATION ARTEFACT. hboyang's OOF
                  AUC is not comparable to the other 176, ARM 217's +41.7e-6 is measured on an
                  inflated member, and the honest comparison is hboyang-vs-controls AFTER both
                  are transformed the same way.
  L <= 100e-6  -> the transform is nearly free and hboyang really is an outlier on merit;
                  w48d's leakage-shaped suspicion survives on other grounds.
  in between   -> partial; report the residual outlier gap after transforming everything.
Also reported: hboyang's rank AFTER all controls get the same treatment. That is the only
apples-to-apples number in this whole comparison.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, N_SPLITS, SEED                 # noqa: E402

y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].values
n = len(y)
fold = np.empty(n, np.int8)
for f, (_, v) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
                           .split(np.zeros(n), y)):
    fold[v] = f

def perfold_rank(v):
    out = np.empty(n, np.float64)
    for f in range(N_SPLITS):
        m = fold == f
        out[m] = (pd.Series(v[m]).rank().values - 0.5) / m.sum()
    return out

print("=" * 92)
print("w49c  CONTROL: what is per-fold rank normalisation worth, in AUC, on RAW members?")
print("=" * 92)

files = sorted(set(glob.glob(os.path.join(DATA, "ext_members*", "oof_*.npy"))))
rows, seen = [], set()
for f in files:
    nm = os.path.basename(f)[4:-4]
    if nm in seen or nm == "hboyang_mix":
        continue
    try:
        v = np.load(f).ravel()
    except Exception:
        continue
    if len(v) != n:
        continue
    seen.add(nm)
    rows.append((nm, v))

# rank the raw members, keep the strongest 12 as controls
base = [(nm, float(roc_auc_score(y, v)), v) for nm, v in rows]
base.sort(key=lambda r: -r[1])
CTRL = base[:12]
print(f"\n  {len(base)} raw members available; using the 12 strongest as controls\n")
print(f"  {'member':34s} {'raw AUC':>12s} {'per-fold ranked':>16s} {'lift e-6':>10s}   ties?")
lifts = []
for nm, a0, v in CTRL:
    a1 = float(roc_auc_score(y, perfold_rank(v)))
    lifts.append((a1 - a0) * 1e6)
    d = len(np.unique(v))
    print(f"  {nm:34s} {a0:12.8f} {a1:16.8f} {(a1-a0)*1e6:10.1f}   "
          f"{'RAW (all distinct)' if d==n else f'{d:,} distinct'}")

L = float(np.median(lifts))
hb = np.load(os.path.join(DATA, "ext_members16", "oof_hboyang_mix.npy")).ravel()
hb_auc = float(roc_auc_score(y, hb))
best_ctrl_raw = CTRL[0][1]
best_ctrl_tr = max(float(roc_auc_score(y, perfold_rank(v))) for _, _, v in CTRL)

print(f"\n  median lift over the 12 controls: L = {L:.1f}e-6   "
      f"(min {min(lifts):.1f}, max {max(lifts):.1f})")
print(f"\n  APPLES TO APPLES -- everything per-fold ranked:")
print(f"    hboyang_mix (already ranked)      {hb_auc:.8f}")
print(f"    best control AFTER the transform  {best_ctrl_tr:.8f}")
print(f"    residual gap                      {(hb_auc-best_ctrl_tr)*1e6:+.1f}e-6")
print(f"  versus w48d's comparison, which was raw-vs-ranked:")
print(f"    best control BEFORE the transform {best_ctrl_raw:.8f}   gap "
      f"{(hb_auc-best_ctrl_raw)*1e6:+.1f}e-6")

verdict = ("NORMALISATION ARTEFACT" if L >= 400 else
           "not the explanation" if L <= 100 else "PARTIAL")
print(f"\n  READING: L = {L:.1f}e-6 -> {verdict}")
json.dump(dict(L_median_e6=L, lifts_e6=lifts, controls=[c[0] for c in CTRL],
               hboyang_auc=hb_auc, best_ctrl_raw=best_ctrl_raw,
               best_ctrl_transformed=best_ctrl_tr,
               residual_gap_e6=(hb_auc - best_ctrl_tr) * 1e6,
               w48d_gap_e6=(hb_auc - best_ctrl_raw) * 1e6, verdict=verdict),
          open(os.path.join(HERE, "w49c_perfoldrank.json"), "w"), indent=1)
