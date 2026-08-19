"""w27p -- the XGBoost and CatBoost legs of the pre-registered CT trio.

WHY THIS EXISTS
---------------
RESEARCH.md:6562 was written before any paired Delta landed, precisely so a null could not
afterwards be used as an excuse to drop the thread:

    "A null paired-Delta for ONE corrected member does NOT close the CT thread ... the paired
     Delta answers 'what is one un-skewed member worth INSIDE a span made of 187 uniformly-
     skewed ones?', which is a LOWER BOUND on, and not an estimate of, the thing that would
     actually pay ... If the Delta is null, the correct next move is the rebuild, not
     abandoning the thread."

and it names the probe to run before committing to ~33 refits: rebuild THREE lattice members
of different function classes and measure the paired Delta of the trio. The lgbm leg already
exists as `lat_ctfix_r400` / `lat_ctraw_r400`. This builds the other two.

Pre-registered at experiments/w27_prereg_slot4.txt SS M5 before it was ever run.

WHY IT IS TEN FITS AND NOT TWENTY
---------------------------------
The CT correction is a pure serve-time rescale and needs no refit: rescaling a tree's input
feature by s is the same function as rescaling that feature's thresholds by s (RESEARCH SS3,
gated there). So the skewed CONTROL member and the corrected member come off the SAME fitted
booster, and the pair is exactly matched -- identical trees, identical seed, identical rows.
Five folds x two classes = ten fits for four members.

    ct1.0000   status quo    -> the control member, `{cls}_ctraw_r400`
    ct1.3333   CT_ / (4/3)   -> the corrected member, `{cls}_ctfix_r400`

WHAT IS IMPORTED RATHER THAN REWRITTEN, AND WHY
-----------------------------------------------
`apply_arm` comes from w26l_serve, so the arms are byte-identical to every LightGBM run in
this thread. `fit_xgboost` / `fit_catboost` come from w27j_ctclass, so the hyperparameters are
identical to the d_c values SS M3 measured (+291.06e-6 xgb, +525.49e-6 cat on fold 0) and
M5(a) can gate on reproducing them exactly. Nothing here re-states a parameter that already
has a home.

⚠ Xb/Xt are COPIED per arm instead of mutated-and-restored. CatBoost clears the writeable flag
on an array it has predicted from, so w26l's mutate-and-restore raises AFTER the fit has cost
its full runtime. That safety is a property of LightGBM, not of `apply_arm`.

⚠ Members land in data/ext_members7/, NEVER in oof/. `load_members` scans oof/ by default, so
saving there moves the pack under blend_lab, w26f, w26h, w26i and w26j at once, and every
reproduction gate in this repo is stated against a fixed member COUNT.

    .venv/bin/python experiments/w27p_serveclass.py --cls xgb --rounds 400
    .venv/bin/python experiments/w27p_serveclass.py --cls cat --rounds 400
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import CACHE, DATA, N_SPLITS, TARGET, get_folds, load_raw, save_preds  # noqa: E402
sys.path.insert(0, HERE)
from w26l_serve import apply_arm, blocks, save_atomic  # noqa: E402
from w27j_ctclass import FITTERS  # noqa: E402

CKPT = os.path.join(os.path.dirname(HERE), "cache", "serveclassckpt")
OUT = os.path.join(DATA, "ext_members7")
ARMS = ["ct1.0000", "ct1.3333"]                 # control, corrected
SUFFIX = {"ct1.0000": "ctraw", "ct1.3333": "ctfix"}

# w27j fold 0, 400 rounds, seed 42 -- the M5(a) gate. Same fitter, same matrix, same arms.
GATE = {
    "xgb": {"ct1.0000": 0.965438622483465,  "ct1.3333": 0.9657296816076262},
    "cat": {"ct1.0000": 0.9651228505635709, "ct1.3333": 0.9656483450687335},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=sorted(FITTERS))
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--folds", default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)

    cols, ti, ci = blocks()
    tag = f"{a.cls}_r{a.rounds}"
    print(f"[{tag}] {len(cols)} cols | {len(ti)} TE_/CT_ pairs | arms {ARMS}", flush=True)
    print(f"[{tag}] rounds={a.rounds} seed={a.seed} threads={a.threads}", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    want = [int(x) for x in a.folds.split(",")] if a.folds else list(range(N_SPLITS))

    oof = np.zeros((len(y), len(ARMS)))
    tp = np.zeros((len(te), len(ARMS)))
    got, t0 = [], time.time()

    for f, (itr, iva) in enumerate(folds):
        if f not in want:
            continue
        p = os.path.join(CKPT, f"{tag}_f{f}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(iva), len(ARMS)):
                oof[iva] = z["oof"]; tp += z["test"] / N_SPLITS; got.append(f)
                print(f"[{tag}] fold {f}: resumed", flush=True)
                continue
            print(f"[{tag}] fold {f}: STALE checkpoint, ignored", flush=True)

        g = lambda k: np.load(os.path.join(CACHE, f"f{f}_{k}.npy"))
        Xa, ya, Xb, yb, Xt = g("Xa"), g("ya"), g("Xb"), g("yb"), g("Xt")
        gm = float(ya.mean())

        tf = time.time()
        predict, _imp = FITTERS[a.cls](Xa, ya, a.rounds, a.seed, a.threads)
        print(f"[{tag}] fold {f}: fitted in {time.time()-tf:.0f}s", flush=True)
        del Xa

        ob = np.zeros((len(yb), len(ARMS)))
        ot = np.zeros((len(Xt), len(ARMS)))
        for j, arm in enumerate(ARMS):
            Xb2, Xt2 = Xb.copy(), Xt.copy()      # copy: CatBoost clears the writeable flag
            apply_arm(Xb2, ti, ci, arm, gm)
            apply_arm(Xt2, ti, ci, arm, gm)
            ob[:, j] = predict(Xb2)
            ot[:, j] = predict(Xt2)
            del Xb2, Xt2

        save_atomic(p, oof=ob, test=ot)
        oof[iva] = ob; tp += ot / N_SPLITS; got.append(f)
        aucs = {arm: roc_auc_score(yb, ob[:, j]) for j, arm in enumerate(ARMS)}
        line = "  ".join(f"{arm} {v:.6f}" for arm, v in aucs.items())
        print(f"[{tag}] fold {f}: {line}  d_c {1e6*(aucs['ct1.3333']-aucs['ct1.0000']):+8.2f}e-6"
              f"  ({time.time()-t0:.0f}s)", flush=True)

        if f == 0 and (a.rounds, a.seed, a.threads) == (400, 42, 3):
            # The gate is only meaningful under w27j's exact settings. `hist` reduction
            # order depends on the thread count, so threads is part of the identity here.
            print(f"[{tag}] === M5(a) GATE, fold 0 vs w27j ===", flush=True)
            ok = True
            for arm, exp in GATE[a.cls].items():
                d = aucs[arm] - exp
                ok &= abs(d) < 1e-9
                print(f"[{tag}]   {arm} got {aucs[arm]:.16f}  w27j {exp:.16f}  "
                      f"diff {d:+.3e}  {'PASS' if abs(d) < 1e-9 else 'FAIL'}", flush=True)
            if not ok:
                raise SystemExit(f"[{tag}] M5(a) GATE FAILED -- w27p and w27j are not the "
                                 f"same instrument, nothing downstream is readable")
        elif f == 0:
            print(f"[{tag}] M5(a) gate SKIPPED: settings "
                  f"(rounds={a.rounds}, seed={a.seed}, threads={a.threads}) are not w27j's "
                  f"(400, 42, 3). This run is a smoke test, not a result.", flush=True)
        del Xb, Xt

    if len(got) < N_SPLITS:
        print(f"[{tag}] only folds {sorted(got)} -- re-run to resume", flush=True)
        return

    os.makedirs(a.out, exist_ok=True)
    print(f"\n[{tag}] === POOLED over {N_SPLITS} folds ===", flush=True)
    for j, arm in enumerate(ARMS):
        auc = roc_auc_score(y, oof[:, j])
        name = f"{a.cls}_{SUFFIX[arm]}_r{a.rounds}"
        save_preds(name, oof[:, j], tp[:, j], len(y), len(te), out=a.out)
        print(f"[{tag}]   {arm:10s} OOF {auc:.10f}   -> {a.out}/oof_{name}.npy", flush=True)
    d_c = roc_auc_score(y, oof[:, 1]) - roc_auc_score(y, oof[:, 0])
    print(f"[{tag}]   d_c (pooled, M5(b)) {1e6*d_c:+8.2f}e-6", flush=True)
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
