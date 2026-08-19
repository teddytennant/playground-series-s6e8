"""w27c -- the missing control for the CT_ scale fix, pre-registered at w26_prereg.txt §G8.

THE QUESTION
------------
w26l measured, pooled over all 5 folds at 400 rounds, that dividing the 72 CT_ columns by
4/3 at serve time is worth +293.9e-6. The obvious rival explanation is not "the scale is
wrong" but "the CT_ block is simply NET-HARMFUL as built, and anything that neutralises it
would score the same". §G8 named the control that separates those and registered its prior
BEFORE any number existed:

    ct_drop ~ ct1.0      -> the block was inert; the fix repairs a channel that carries signal
    ct_drop ~ ct1.3333   -> the block as built is net-harmful; DELETE it, do not rescale it
    ct_drop > ct1.3333   -> the fix is a partial repair of something better removed

    registered prior: ct_drop lands BETWEEN ct1.0 and ct1.3333, nearer ct1.0.
    modal guess ct_drop - ct1.0 = +100e-6, about a third of the fix.

WHY IT NEEDS ITS OWN FIT
-------------------------
Every arm in w26l came off one booster per fold, because rescaling an input is the same
function as rescaling that feature's thresholds. Dropping a feature is NOT such a transform:
it changes which splits get chosen. §G8 also forbids the cheap substitute of "set CT to a
constant at serve time" -- that keeps the split structure and merely sends every row down
one side, which answers a different question.

MATCHED TO w26l ON EVERYTHING EXCEPT THE DROP
----------------------------------------------
Same PARAMS (lgbm_fixed_lat), same 400 rounds, same seed 42, same folds from get_folds, same
cache arrays. The only difference is the 72 columns removed from X. The comparison numbers
are w26l's pooled ct1.0000 = 0.9654813306 and ct1.3333 = 0.9657751945.

⚠ n_jobs differs from w26l's 3 (the box is free now). §G5b applies: LightGBM's histogram
reduction is not guaranteed bit-identical across thread counts, so this is NOT a bit-exact
pairing with w26l -- it is a matched-config comparison, and a difference under ~10e-6 should
not be read. The effect being looked for is ~100-300e-6, comfortably above that.

Per-fold checkpointed exactly as w26k/w26l are; re-running the identical command resumes.

    .venv/bin/python experiments/w27c_ctdrop.py --name ctdrop --rounds 400 --jobs 6
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
# w26l pooled, 5 folds, 400 rounds -- what this is read against
W26L_CT1 = 0.9654813306
W26L_CTFIX = 0.9657751945


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ctdrop")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--folds", default=None)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    keep = np.array([i for i, c in enumerate(cols) if not c.startswith("CT_")])
    print(f"[{a.name}] {len(cols)} cols -> {len(keep)} kept, "
          f"{len(cols)-len(keep)} CT_ dropped", flush=True)
    print(f"[{a.name}] rounds={a.rounds} seed={a.seed} jobs={a.jobs}", flush=True)

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
        p = os.path.join(CKPT, f"{a.name}_f{f}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(iva),):
                oof[iva] = z["oof"]
                tp += z["test"] / N_SPLITS
                got.append(f)
                print(f"[{a.name}] fold {f}: resumed from checkpoint", flush=True)
                continue
            print(f"[{a.name}] fold {f}: STALE checkpoint, ignored", flush=True)

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
        print(f"[{a.name}] fold {f}: {roc_auc_score(yb, ob):.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        del Xb, Xt, m

    if len(got) < N_SPLITS:
        print(f"[{a.name}] only folds {sorted(got)} done -- nothing pooled, "
              f"re-run the same command to resume", flush=True)
        return

    cv = roc_auc_score(y, oof)
    print(f"\n[{a.name}] POOLED OOF over {len(y):,} rows")
    print(f"  ct_drop   {cv:.10f}")
    print(f"  w26l ct1.0000  {W26L_CT1:.10f}   ct_drop - ct1.0    {(cv-W26L_CT1)*1e6:+9.2f}e-6")
    print(f"  w26l ct1.3333  {W26L_CTFIX:.10f}   ct_drop - ctfix    {(cv-W26L_CTFIX)*1e6:+9.2f}e-6")
    frac = (cv - W26L_CT1) / (W26L_CTFIX - W26L_CT1)
    print(f"  ct_drop recovers {frac*100:.1f}% of the fix (§G8: registered ~33%)")
    np.save(os.path.join(HERE, f"w27c_{a.name}_oof.npy"), oof)
    np.save(os.path.join(HERE, f"w27c_{a.name}_test.npy"), tp)
    print(f"  saved w27c_{a.name}_{{oof,test}}.npy  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
