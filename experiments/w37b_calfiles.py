"""w37b — build the es-bias CALIBRATION submission files.

WHY THESE ARE NOT FILLER, AND NOT LB-CHASING.
Each file is one member's own test-prediction vector, submitted to read back its exact public
LB. That LB is a MEASUREMENT of a bias, not an attempt on the board. Every one of them will
score ~0.958-0.969 against our 0.97118 best; the public LB shows best-of-all-submissions, so
they cannot cost us a place, and they are not selection candidates under any circumstance.

WHAT THEY BUY. w36f priced es-on-val at +2.7e-4 of fake OOF AUC off ONE usable dirty reading
(`zwr_realmlp`); the second (`tam_lkup`) came back with the wrong SIGN because its title's LB
was the notebook's blend, not that member. Submitting the vector ourselves removes exactly
that failure mode. Five such readings turn n=1 into n=5 and, more to the point, give the first
estimate of the SPREAD of the inflation across members -- which is the quantity that decides
whether the 15 quarantined members can be rescued by deflation or stay quarantined.

`mkt_realmlp` is CLEAN and goes FIRST: it is the enabling send. It anchors the line at 0.9581
and drops the line's se at the low-AUC quarantined members (dkv_*, ravi_*, kava_*) from
1.0-1.4e-4 -- 37-53% of the effect being measured, i.e. useless -- to ~2e-5. Without it, five
of the dirty readings cannot be interpreted.
"""
import numpy as np, pandas as pd, hashlib, json, os
from sklearn.metrics import roc_auc_score

W = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"
SUB, TARGET = W + "submissions/", "addicted_label"

# (name, dir, role, oof_auc_expected, es_mechanism)
PLAN = [
 ("mkt_realmlp",   "ext_members13",   "CLEAN-ANCHOR", 0.9581337349, "none -- max_epochs=512 reached on all 5 folds"),
 ("om_xgb2",       "ext_members10es", "DIRTY",        0.9687331700, "XGB early stopping on the scored fold"),
 ("omid_tabm",     "ext_members13es", "DIRTY",        0.9675077199, "NN 'New best epoch!' + 'Restoring best model' per fold"),
 ("dm_cat",        "ext_members11es", "DIRTY",        0.9667001254, "CatBoost od_type + best-iteration on the scored fold"),
 ("ravi_realmlp1c","ext_members11es", "DIRTY",        0.9646676405, "RealMLP additive patience 15 on the scored fold"),
 ("dkv_xgb",       "ext_members12es", "DIRTY",        0.9644659266, "XGB early_stopping_rounds=150 -- the MILDEST form here"),
 ("ram_hgb",       "ext_members12",   "CLEAN-AUDIT",  0.9680258266, "none -- audits the title-LB the whole line rests on"),
]

te = pd.read_csv(W + "data/test.csv", usecols=["id"])
y  = pd.read_csv(W + "data/train.csv", usecols=["addicted_label"])["addicted_label"].to_numpy()
print(f"test rows {len(te):,}   train rows {len(y):,}\n")

# every submission already on disk, so we can prove none of these duplicates one
seen = {}
for f in os.listdir(SUB):
    if f.endswith(".csv"):
        seen[hashlib.md5(open(SUB + f, "rb").read()).hexdigest()] = f

rows = []
for name, d, role, auc_exp, mech in PLAN:
    oof  = np.load(f"{W}data/{d}/oof_{name}.npy")
    test = np.load(f"{W}data/{d}/test_{name}.npy")
    auc  = roc_auc_score(y, oof)
    assert len(test) == len(te),  f"{name}: test len {len(test)} != {len(te)}"
    assert len(oof)  == len(y),   f"{name}: oof len {len(oof)} != {len(y)}"
    assert np.isfinite(test).all(), f"{name}: non-finite test values"
    assert abs(auc - auc_exp) < 1e-9, f"{name}: oof AUC {auc:.10f} != expected {auc_exp:.10f}"

    out = f"w37_cal_{name}.csv"
    pd.DataFrame({"id": te["id"].to_numpy(), TARGET: test}).to_csv(SUB + out, index=False)
    md5 = hashlib.md5(open(SUB + out, "rb").read()).hexdigest()
    dup = seen.get(md5, "")
    ndist = len(np.unique(test))
    print(f"{role:13s} {name:15s} oof {auc:.10f}  rows {len(test):,}  distinct {ndist:,}"
          f"  range [{test.min():.3e}, {test.max():.6f}]  dup:{dup or 'NONE'}")
    assert not dup, f"{name} duplicates {dup} -- an identical file scores identically, pointless"
    rows.append(dict(order=len(rows)+1, file=out, member=name, role=role,
                     oof_auc=auc, n_distinct=int(ndist), md5=md5, es_mechanism=mech))

pd.DataFrame(rows).to_csv(W + "experiments/w37b_calfiles.csv", index=False)
print(f"\nwrote {len(rows)} calibration files + experiments/w37b_calfiles.csv")
