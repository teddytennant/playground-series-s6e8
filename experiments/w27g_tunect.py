"""w27g -- does correcting the CT_ train/serve skew MOVE the LightGBM optimum?

THE QUESTION, AND WHY IT IS NOT THE DAY-1 SWEEP AGAIN
------------------------------------------------------
Tuning LightGBM against these features is a priced null here: the day-1 two-stage sweep
(JOURNAL 2026-08-10) found +500e-6 on fold 0 and it collapsed to +30e-6 on the full OOF,
under the 50e-6 noise floor. Its one durable conclusion was "less capacity, more
regularisation wins, because 144 of the 184 features are target-derived".

Every number in that sweep was measured on features carrying the CT_ skew (§G/§J): 72 of the
184 columns arrive at serve time 4/3 too large, on every fold, because `te_block` builds the
train side of the count with an inner StratifiedKFold(4). A model is RIGHT to lean away from a
block that is systematically 33% off at serve time. So the day-1 "regularise harder" finding
may be, in part, the model correctly defending itself against a defect -- and once the defect
is corrected, leaning away is no longer right and the optimum should move back toward capacity.

That is the hypothesis. It is registered at w26_prereg.txt §K, in full, before this ran.

WHY IT COSTS ONE FIT PER CONFIG AND NOT TWO
---------------------------------------------
Rescaling one input of a tree by s is the same function as rescaling that feature's split
thresholds by s, so "fit with CT x s" == "the same booster served rows whose CT is divided by
s" -- verified in w26 slot 7 to maxdiff 0.000e+00, spearman 1.00000000. One booster therefore
yields both arms, on identical rows and identical folds. The interaction test is free relative
to the sweep it sits inside.

  ⚠ This equivalence covers SCALE ONLY. It does not extend to dropping CT_ (w27c) or to the
  clean full-map rebuild (w27f); those change which splits get chosen and need their own fits.

READ AGAINST (w26l, pooled 5 folds, 400 rounds, control config, same seed and folds):
    ct1.0000  0.9654813306
    ct1.3333  0.9657751945    +293.86e-6

Per-config checkpointed; re-running the identical command resumes.

    .venv/bin/python experiments/w27g_tunect.py --name tunect --rounds 400 --jobs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import CACHE, TARGET, get_folds, load_raw  # noqa: E402

CKPT = os.path.join(os.path.dirname(HERE), "cache", "tunectckpt")

# lgbm_fixed_lat, exactly as in oof/summary_lgbm_fixed_lat.json (CV 0.9677108350).
CONTROL = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
               min_child_samples=250, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.6, reg_lambda=80.0)

# One knob at a time off the control, on the four axes the angle names. The capacity and
# regularisation axes are the ones §K2(c) makes a directional prediction about; lr and
# colsample are there so the table is not purely a capacity sweep.
GRID = [
    ("control",        {}),
    # -- capacity
    ("leaves31_d6",    dict(num_leaves=31, max_depth=6)),
    ("leaves127_d9",   dict(num_leaves=127, max_depth=9)),
    ("leaves255_dinf", dict(num_leaves=255, max_depth=-1)),
    # -- leaf size (the other capacity knob)
    ("mcs40",          dict(min_child_samples=40)),
    ("mcs800",         dict(min_child_samples=800)),
    # -- L2
    ("l2_5",           dict(reg_lambda=5.0)),
    ("l2_300",         dict(reg_lambda=300.0)),
    # -- the two knobs that regularise the FEATURE choice, i.e. how much the model is
    #    allowed to lean on the 144 target-derived columns at all
    ("colsample0.4",   dict(colsample_bytree=0.4)),
    ("colsample0.9",   dict(colsample_bytree=0.9)),
    ("bynode0.7",      dict(feature_fraction_bynode=0.7)),
    ("pathsmooth20",   dict(path_smooth=20.0)),
    # -- learning rate, at fixed rounds so it is also a capacity proxy
    ("lr0.05",         dict(learning_rate=0.05)),
    # -- the corner K2(c) predicts should improve MOST under correction: more capacity and
    #    less penalty at once. Registered as a prediction, not chosen after the fact.
    ("cap_corner",     dict(num_leaves=127, max_depth=9, min_child_samples=40,
                            reg_lambda=5.0)),
]
SCALES = [1.0, 4.0 / 3.0]


def save_atomic(path, **arrs):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:      # np.savez appends .npz to a NAME, not to a handle
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tunect")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    ct = np.array([i for i, c in enumerate(cols) if c.startswith("CT_")])
    print(f"[{a.name}] {len(cols)} cols, {len(ct)} CT_ cols, {len(GRID)} configs, "
          f"arms {[round(s,4) for s in SCALES]}", flush=True)
    print(f"[{a.name}] fold={a.fold} rounds={a.rounds} seed={a.seed} jobs={a.jobs}",
          flush=True)

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    itr, iva = get_folds(y)[a.fold]

    g = lambda k: np.load(os.path.join(CACHE, f"f{a.fold}_{k}.npy"))
    Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
    assert len(yb) == len(iva), "fold order does not match the cache"

    # the §J R-J3 style sanity gate, free here: the cached train CT_ must sit ~3/4 of the
    # cached valid CT_ in the mean, or the whole premise of the arms is wrong.
    ratio = Xb[:, ct].mean(0) / np.maximum(Xa[:, ct].mean(0), 1e-9)
    med = float(np.median(ratio))
    print(f"[{a.name}] GATE  median valid/train CT_ ratio = {med:.4f} "
          f"(expected 1.3333)", flush=True)
    if not (1.28 <= med <= 1.39):
        raise SystemExit(f"[{a.name}] GATE FAILED at {med:.4f} -- the cache is not the one "
                         f"§G measured. Every arm below would be meaningless. STOP.")

    b0 = Xb[:, ct].copy()
    rows, t0 = [], time.time()

    for tag, delta in GRID:
        p = os.path.join(CKPT, f"{a.name}_f{a.fold}_{tag}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["auc"].shape == (len(SCALES),):
                rows.append((tag, float(z["auc"][0]), float(z["auc"][1]),
                             float(z["share"])))
                print(f"[{a.name}] {tag:16s} resumed  "
                      f"1.0 {z['auc'][0]:.6f}  4/3 {z['auc'][1]:.6f}", flush=True)
                continue
            print(f"[{a.name}] {tag:16s} STALE checkpoint, refitting", flush=True)

        params = dict(CONTROL)
        params.update(delta)
        m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                               verbose=-1, n_jobs=a.jobs, **params)
        m.fit(Xa, ya)
        gain = m.booster_.feature_importance("gain")
        share = float(gain[ct].sum() / max(gain.sum(), 1e-9))

        auc = np.zeros(len(SCALES))
        for j, s in enumerate(SCALES):
            Xb[:, ct] = b0 / s
            auc[j] = roc_auc_score(yb, m.predict_proba(Xb)[:, 1])
        Xb[:, ct] = b0

        save_atomic(p, auc=auc, share=np.float64(share))
        rows.append((tag, float(auc[0]), float(auc[1]), share))
        print(f"[{a.name}] {tag:16s} 1.0 {auc[0]:.6f}  4/3 {auc[1]:.6f}  "
              f"d_c {(auc[1]-auc[0])*1e6:+8.2f}e-6  CTshare {share*100:5.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del m

    if len(rows) < len(GRID):
        print(f"[{a.name}] only {len(rows)}/{len(GRID)} configs done -- re-run to resume",
              flush=True)
        return

    print(f"\n[{a.name}] FOLD {a.fold} TABLE ({a.rounds} rounds). Every d_c is against that "
          f"config's OWN s=1.0 number (R-K2).", flush=True)
    print(f"  {'config':16s} {'AUC@1.0':>10s} {'AUC@4/3':>10s} {'d_c':>11s} "
          f"{'vs ctl@1.0':>11s} {'vs ctl@4/3':>11s} {'CTshare':>8s}")
    c10 = dict((r[0], r[1]) for r in rows)["control"]
    c43 = dict((r[0], r[2]) for r in rows)["control"]
    for tag, a10, a43, sh in rows:
        print(f"  {tag:16s} {a10:10.6f} {a43:10.6f} {(a43-a10)*1e6:+9.2f}e-6 "
              f"{(a10-c10)*1e6:+9.2f}e-6 {(a43-c43)*1e6:+9.2f}e-6 {sh*100:7.1f}%")

    d = np.array([r[2] - r[1] for r in rows])
    sh = np.array([r[3] for r in rows])
    rho, pv = spearmanr(sh, d)
    print(f"\n  K2(a)  d_c > 0 for all configs?  "
          f"{'YES' if (d > 0).all() else 'NO -- ' + str([rows[i][0] for i in np.where(d<=0)[0]])}")
    print(f"  K2(b)  spearman(CT_ gain share, d_c) = {rho:+.3f}  (p={pv:.3f}, n={len(d)}) "
          f"-- registered POSITIVE, modal +0.5")
    i10 = int(np.argmax([r[1] for r in rows]))
    i43 = int(np.argmax([r[2] for r in rows]))
    print(f"  K2(c)  argmax @1.0 = {rows[i10][0]}   argmax @4/3 = {rows[i43][0]}   "
          f"{'DIFFER' if i10 != i43 else 'SAME'}")
    head = (rows[i43][2] - rows[i10][1]) * 1e6 - 293.86
    print(f"  K2(d)  HEADLINE best(4/3) - best(1.0) - 293.86e-6 = {head:+.2f}e-6  "
          f"-- registered 0 to +80e-6, modal +15e-6")
    print(f"\n  R-K1/R-K3: fold-0 screen. Nothing here is exported, blended or shipped; if "
          f"the headline clears +80e-6 the follow-on is a 5-fold re-run of {rows[i43][0]} "
          f"against the control.")
    with open(os.path.join(HERE, f"w27g_{a.name}.json"), "w") as fh:
        json.dump({"fold": a.fold, "rounds": a.rounds, "rows": rows,
                   "spearman": [float(rho), float(pv)], "headline_e6": float(head),
                   "argmax_10": rows[i10][0], "argmax_43": rows[i43][0]}, fh, indent=1)
    print(f"  saved w27g_{a.name}.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
