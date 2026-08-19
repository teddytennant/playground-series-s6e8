"""w27r -- the feature-block ablation ladder, pre-registered at w27_prereg_slot5.txt §M7.

Slot 5's angle is "measure every feature on CV, keep only what pays". `cache/cols.json` is 184
columns and splits exactly three ways -- 72 `TE_` (in-fold smoothed target encoding), 72 `CT_`
(in-fold cell count), 40 raw/derived. w27c measured ONE cell of that table (drop CT_, pooled
0.9656895129). This measures the rest.

It does two jobs with one set of fits, which is why it is worth the compute on a saturated box:

  1. the ablation table the angle asks for, on the frozen folds, matched to w26l on everything
     except the column set;
  2. three CHEAP MEMBERS on feature sets no other member of the 188-pack has. Journal §9 item 4
     needs five of those (the pack pays +3.37e-6 per added member, se 0.24, 3/3 reps), and a
     feature-set variant is how a new member is made DECORRELATED rather than redundant --
     which w20d and adarsh1077's leave-one-author-out both say is what a saturated stack pays
     for. §M7(b) registers the prediction that the WEAKEST arm is the most valuable one here.

MATCHED TO w27c/w26l ON EVERYTHING EXCEPT THE COLUMNS
  Same PARAMS (imported from w26k_ctscale, so `lgbm_fixed_lat`'s), same 400 rounds, same seed
  42, same `get_folds` folds, same `cache/f{k}_*` arrays. ⚠ Read w27c's own caveat: n_jobs is
  not guaranteed to give a bit-identical LightGBM histogram reduction, so this is a
  matched-config comparison and differences under ~10e-6 must not be read.

Per-(arm, fold) checkpointed to `cache/ctckpt/<arm>_f<k>.npz` via temp-file + os.replace, the
same idiom as w26k/w26l/w27c: re-running the identical command resumes and refits nothing.

⚠ The arm named `ctdrop` is DELIBERATELY ABSENT from ARMS below. w27c already owns that name
and its five checkpoints; adding it here would either refit it or -- worse -- silently resume
w27c's checkpoints under a config this file might later change. It is quoted as a constant.

    .venv/bin/python experiments/w27r_blockdrop.py --arm encdrop --rounds 400 --jobs 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import CACHE, N_SPLITS, TARGET, get_folds, load_raw  # noqa: E402
sys.path.insert(0, HERE)
from w26k_ctscale import PARAMS, save_atomic  # noqa: E402

CKPT = os.path.join(os.path.dirname(HERE), "cache", "ctckpt")
OUT = os.path.join(os.path.dirname(HERE), "data", "ext_members7")

# arm -> prefixes to DROP. Ordered cheapest-fit-first, which is also §M7(e)'s launch order.
ARMS = {
    "encdrop": ("TE_", "CT_"),   #  40 cols -- the raw frame alone
    "tedrop":  ("TE_",),         # 112 cols -- counts kept, means gone
    "rawdrop": (),               # 144 cols -- encoding only; handled by KEEP_ONLY below
}
KEEP_ONLY = {"rawdrop": ("TE_", "CT_")}

# w26l pooled, 5 folds, 400 rounds, all 184 columns -- the control every arm is read against.
W26L_CT1 = 0.9654813306
W27C_CTDROP = 0.9656895129


def select(cols, arm):
    """Column indices this arm keeps. Two mutually exclusive modes so an arm can be stated
    either way round without an ambiguous double negative in the arm table."""
    if arm in KEEP_ONLY:
        pre = KEEP_ONLY[arm]
        return np.array([i for i, c in enumerate(cols) if c.startswith(pre)])
    pre = ARMS[arm]
    return np.array([i for i, c in enumerate(cols) if not c.startswith(pre)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    keep = select(cols, a.arm)
    assert len(keep) > 0, f"{a.arm} selected no columns"
    print(f"[{a.arm}] {len(cols)} cols -> {len(keep)} kept, {len(cols)-len(keep)} dropped",
          flush=True)
    print(f"[{a.arm}] rounds={a.rounds} seed={a.seed} jobs={a.jobs}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = ([int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS)))

    oof = np.zeros(len(y))
    tp = np.zeros(len(te))
    got = []
    t0 = time.time()

    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        p = os.path.join(CKPT, f"{a.arm}_f{f}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(iva),):
                oof[iva] = z["oof"]
                tp += z["test"] / N_SPLITS
                got.append(f)
                print(f"[{a.arm}] fold {f}: resumed from checkpoint", flush=True)
                continue
            print(f"[{a.arm}] fold {f}: STALE checkpoint, ignored", flush=True)

        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                               verbose=-1, n_jobs=a.jobs, **PARAMS)
        m.fit(Xa[:, keep], ya)
        del Xa
        ob = m.predict_proba(Xb[:, keep])[:, 1]
        ot = m.predict_proba(Xt[:, keep])[:, 1]
        save_atomic(p, oof=ob, test=ot)
        oof[iva] = ob
        tp += ot / N_SPLITS
        got.append(f)
        print(f"[{a.arm}] fold {f}: {roc_auc_score(yb, ob):.6f}  ({time.time()-t0:.0f}s)",
              flush=True)
        del Xb, Xt, m

    if len(got) < N_SPLITS:
        print(f"[{a.arm}] only folds {sorted(got)} done -- nothing pooled, re-run to resume",
              flush=True)
        return

    cv = roc_auc_score(y, oof)
    print(f"\n[{a.arm}] POOLED OOF over {len(y):,} rows")
    print(f"  {a.arm:>8s}  {cv:.10f}")
    print(f"  control 184col   {W26L_CT1:.10f}   {a.arm} - control  {(cv-W26L_CT1)*1e6:+9.2f}e-6")
    print(f"  w27c ctdrop      {W27C_CTDROP:.10f}   {a.arm} - ctdrop   {(cv-W27C_CTDROP)*1e6:+9.2f}e-6")

    # R-M7a: export is UNCONDITIONAL and was decided before any number above was seen.
    name = f"lat_{a.arm}_r400"
    np.save(os.path.join(OUT, f"oof_{name}.npy"), oof)
    np.save(os.path.join(OUT, f"test_{name}.npy"), tp)
    print(f"  exported data/ext_members7/{{oof,test}}_{name}.npy  ({time.time()-t0:.0f}s)",
          flush=True)
    print(f"  next: .venv/bin/python experiments/w27q_cand.py --gate {name}={cv:.10f}",
          flush=True)


if __name__ == "__main__":
    main()
