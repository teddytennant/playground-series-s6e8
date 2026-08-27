"""The windowed-local TE prior at the SHIPPED preset, over ALL FIVE FOLDS, with a null arm.

w94c screened this at a cheap operating point (lr 0.08, 64 leaves, 60% of the rows) and
came back +123.872e-6 on fold 0. That is 51x the +2.417e-6 margin the whole selection rests
on, which is exactly why it cannot be believed as it stands: it is ONE fold, ONE seed, and
it is measured against a learner 440e-6 weaker than the one we ship. w94's own history says
the number shrinks every time the representation gets stronger -- +520.3e-6 on the 12-column
TE-only frame became +123.9e-6 once the other 172 columns went back.

Two things are missing before this can decide anything, and this file adds both.

1. THE SHIPPED OPERATING POINT. lr 0.035, 96 leaves, 8000 trees / 200 rounds early stopping,
   every row -- `run_lgbm.PRESETS["control"]`, which is what every member in the blend was
   trained at. A feature edge measured against a weak learner is the edge that learner could
   not get elsewhere; at the shipped preset there is less "elsewhere" left.

2. A NULL ARM. The two arms share rows, params and seed, so the only thing separating them
   apart from the prior is LightGBM's own stochasticity (row subsample, colsample, tie
   breaking). `global_seed43` refits the CONTROL arm with random_state=SEED+1 and nothing
   else changed, so `global_seed43 - global` is a paired difference drawn from pure noise at
   this exact operating point. Five of those give the noise floor the signal must clear.
   Without it, "+124e-6" is a number with no scale attached to it.

Per fold the three fits run in the order global / windowed / global_seed43, and EVERY arm is
written to JSON the moment it exists -- w94b was killed twice and cost the whole measurement
both times.

⚠ Absolute AUCs here ARE at the shipped preset but they are still single-fold single-model
numbers on a 184-column frame, not the blended pipeline's CV. Do not quote them as a CV.
Only the paired differences mean anything.

  C1  WIN=1e7 must collapse the variant onto te_block EXACTLY (fold 0, TE/CT block only --
      see w94c for why the assembled Xa cannot carry this assertion: the base block keeps
      raw NaNs and a max-abs-diff over it is `nan` by construction).
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from sklearn.metrics import roc_auc_score  # noqa: E402
import lightgbm as lgb  # noqa: E402
from common import TARGET, get_folds, load_raw, SEED  # noqa: E402
from features import NUM, make_frames, te_block  # noqa: E402
from w94b_teprior_full import SMOOTH, WIN, _axis, te_block_win  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w96a_teprior_folds.json")

# run_lgbm.PRESETS["control"] verbatim -- what every blend member was trained at.
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.035, num_leaves=96,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.6, reg_lambda=5.0, max_depth=7,
              verbosity=-1, n_jobs=16)
N_EST, STOPPING = 8000, 200

# What the members in the blend were ACTUALLY trained at -- see oof/summary_lgbm_tuned_lat.json
# and the w96b prereg addendum. --tuned swaps to it. The replication is confirmatory only.
TUNED = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
             min_child_samples=250, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.6, reg_lambda=80.0)

# global and global_seed43 differ in the LIGHTGBM SEED ALONE and share one frame byte for
# byte, so their paired difference is the noise floor for the windowed-vs-global one. They
# run back to back precisely so the frame cannot drift between them.


def say(*a):
    print(*a, flush=True)


def write(res):
    json.dump(res, open(OUT, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--tuned", action="store_true",
                    help="the vector the blend members were really trained at. CONFIRMATORY "
                         "ONLY -- it can withdraw a BUILD, never create one (w96b prereg).")
    ap.add_argument("--smoke", action="store_true",
                    help="60 trees, 20 rounds -- exercises every path in ~2 min. "
                         "Its AUCs are meaningless; it exists to prove the plumbing.")
    a = ap.parse_args()
    want = [int(s) for s in a.folds.split(",")]
    global N_EST, STOPPING, OUT, PARAMS
    if a.tuned:
        PARAMS = dict(objective="binary", metric="auc", verbosity=-1, n_jobs=16, **TUNED)
        OUT = OUT.replace(".json", "_tuned.json")
    if a.smoke:
        N_EST, STOPPING = 60, 20
        OUT = OUT.replace(".json", "_smoke.json")

    t0 = time.time()
    res = {"preset": "tuned (oof/summary_lgbm_tuned_lat.json)" if a.tuned
           else "run_lgbm.PRESETS[control]", "params": dict(PARAMS),
           "n_estimators": N_EST, "stopping": STOPPING, "smooth": SMOOTH, "win": WIN,
           "folds_requested": want, "folds": {}}
    tr, te = load_raw()
    y = tr[TARGET].astype(int)
    Xtr, _Xte, Ktr, Kte = make_frames(tr, te)
    ntr = len(Ktr)
    say(f"frame: {Xtr.shape[1]} base feats, {Ktr.shape[1]} lattice keys ({time.time()-t0:.0f}s)")

    full_keys = pd.concat([Ktr, Kte], ignore_index=True)
    axes_all = {}
    for c in NUM:
        cd, n_lev = _axis(full_keys, c)
        axes_all[c] = (cd[:ntr], cd[ntr:], n_lev)
    wk = [c for c in axes_all if axes_all[c][2] > 3 * WIN]
    say(f"windowed keys ({len(wk)} of {len(axes_all)} single-column NUM, >{3*WIN} levels): {wk}")
    res["windowed_keys"] = {c: int(axes_all[c][2]) for c in axes_all}
    write(res)

    folds = get_folds(y)
    for fi in want:
        itr, iva = folds[fi]
        ya, yb = y.iloc[itr], y.iloc[iva].to_numpy()
        ax = {c: (axes_all[c][0][itr], axes_all[c][0][iva], axes_all[c][1][:1], axes_all[c][2])
              for c in axes_all}
        slot = res["folds"].setdefault(str(fi), {"n_train": len(itr), "n_valid": len(iva)})
        first = fi == want[0]
        say(f"\n=== fold {fi}: train {len(itr)}, valid {len(iva)} "
            f"({time.time()-t0:.0f}s elapsed) ===")
        base_tr = Xtr.iloc[itr].reset_index(drop=True)
        base_va = Xtr.iloc[iva].reset_index(drop=True)

        def build(win):
            t1 = time.time()
            if win is None:
                t_tr, t_va, _ = te_block(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1], SMOOTH)
            else:
                t_tr, t_va, _ = te_block_win(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1],
                                             ax, SMOOTH, win)
            Xa = pd.concat([base_tr, t_tr.reset_index(drop=True)], axis=1).to_numpy("float32")
            Xb = pd.concat([base_va, t_va.reset_index(drop=True)], axis=1).to_numpy("float32")
            cols = sorted(t_tr.columns)
            blk = t_tr[cols].to_numpy("float32") if first else None
            del t_tr, t_va
            gc.collect()
            say(f"  {'global' if win is None else 'win=' + str(win)} frame built "
                f"{Xa.shape} ({time.time()-t1:.0f}s)")
            return Xa, Xb, cols, blk

        def fit(tag, Xa, Xb, seed):
            t2 = time.time()
            m = lgb.LGBMClassifier(n_estimators=N_EST, random_state=seed, **PARAMS)
            m.fit(Xa, ya.to_numpy(), eval_set=[(Xb, yb)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(STOPPING, verbose=False)])
            auc = float(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))
            slot[tag] = {"auc": auc, "trees": int(m.best_iteration_),
                         "seed": seed, "secs": round(time.time() - t2, 1)}
            write(res)
            say(f"  fold-{fi} AUC {tag:13s} {auc:.10f}  ({m.best_iteration_} trees, "
                f"{time.time()-t2:.0f}s)  [written]")
            del m
            gc.collect()

        # --- control frame, fitted twice: the shipped seed and the null seed ---
        Xa, Xb, cols, te_global = build(None)
        fit("global", Xa, Xb, SEED)
        fit("global_seed43", Xa, Xb, SEED + 1)
        del Xa, Xb
        gc.collect()

        # --- variant frame ---
        Xa, Xb, _, blk = build(WIN)
        if first:
            shift = float(np.max(np.abs(blk - te_global)))
            slot["max_te_shift"] = shift
            say(f"  max|TE shift| windowed vs global = {shift:.6f}  "
                f"({'the arms DIFFER' if shift > 0 else 'IDENTICAL -- variant is a no-op'})")
            del blk
            gc.collect()
            tc = time.time()
            t_c1, _, _ = te_block_win(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte.iloc[:1],
                                      ax, SMOOTH, 10 ** 7)
            d = float(np.max(np.abs(t_c1[cols].to_numpy("float32") - te_global)))
            res["C1_max_abs_diff"], res["C1"] = d, "PASS" if d == 0.0 else "FAIL"
            say(f"  C1 WIN=1e7 vs te_block, TE/CT block only: max|diff| = {d:.3e}  "
                f"{res['C1']} ({time.time()-tc:.0f}s)")
            write(res)
            del t_c1
            gc.collect()
            if d != 0.0:
                say("C1 FAILED -- refusing to report arms off an unverified variant.")
                return 1
        del te_global
        gc.collect()
        fit("windowed", Xa, Xb, SEED)
        del Xa, Xb
        gc.collect()

        for a_, b_, key in (("windowed", "global", "signal_e6"),
                            ("global_seed43", "global", "null_e6")):
            if a_ in slot and b_ in slot:
                slot[key] = (slot[a_]["auc"] - slot[b_]["auc"]) * 1e6
        write(res)
        say(f"  fold {fi}: SIGNAL (windowed-global) {slot.get('signal_e6', float('nan')):+.3f}e-6"
            f"   NULL (seed43-global) {slot.get('null_e6', float('nan')):+.3f}e-6")

    sig = [f["signal_e6"] for f in res["folds"].values() if "signal_e6" in f]
    nul = [f["null_e6"] for f in res["folds"].values() if "null_e6" in f]

    def sd(v):
        return float(np.std(v, ddof=1)) if len(v) > 1 else None

    def fmt(v):
        return f"{v:.3f}e-6" if v is not None else "n/a (one fold)"

    if sig:
        res["signal_mean_e6"], res["signal_sd_e6"] = float(np.mean(sig)), sd(sig)
        res["null_mean_e6"], res["null_sd_e6"] = float(np.mean(nul)) if nul else None, sd(nul)
        say(f"\nSHIPPED PRESET, {len(sig)} fold(s), paired:")
        say(f"  SIGNAL windowed - global    mean {np.mean(sig):+.3f}e-6  sd {fmt(sd(sig))}  "
            f"per fold {[round(v, 1) for v in sig]}")
        say(f"  NULL   seed43   - global    mean {np.mean(nul):+.3f}e-6  sd {fmt(sd(nul))}  "
            f"per fold {[round(v, 1) for v in nul]}")
        say("  w94c's cheap-preset fold-0 number was +123.872e-6")
        say("  the selection margin this would have to be worth something against is +2.417e-6")
    write(res)
    say(f"done ({time.time()-t0:.0f}s) -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
