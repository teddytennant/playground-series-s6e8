"""Build a REAL 5-fold OOF for the windowed-TE prior, and its matched global-prior twin.

This is the "BUILD" branch of `experiments/w96b_prereg.txt` and it refuses to run unless
`w96b_gate.py` returns BUILD (`--force` overrides, and says so loudly in the log).

TWO ARMS, BOTH BUILT HERE. The obvious shortcut is to compare the new windowed member
against an incumbent already sitting in oof/. That comparison would be confounded by every
difference of provenance between them -- seed lists, `--frac`, the exact param JSON that was
passed on the day, whatever the cache held at the time. Building the global twin in the same
loop, off the same frames, at the same seed, costs one extra pass and makes the OOF-CV delta
mean the same thing the fold-level delta in w96a meant.

NO CACHE DIRECTORY. `build_cache.py`'s layout would need ~3.6 GB and this disk has 3.7 GB
free (100% used). Every frame here is built, used and dropped inside its fold.

Params are the TUNED vector -- what the blend members were actually trained at
(oof/summary_lgbm_tuned_lat.json), not run_lgbm.PRESETS["control"]. See the w96b addendum.

Writes oof/oof_{name}.npy, oof/test_{name}.npy and oof/summary_{name}.json via
`common.save_preds`, and nothing else. No submission, no blend weight, no registration.
Whether either arm then ENTERS the blend is a later decision under the existing member
gates; this only produces the evidence for it.
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
from common import TARGET, get_folds, load_raw, save_preds, SEED  # noqa: E402
from features import NUM, make_frames, te_block  # noqa: E402
from w94b_teprior_full import SMOOTH, WIN, _axis, te_block_win  # noqa: E402
import w96b_gate  # noqa: E402

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.025, num_leaves=63,
              max_depth=7, max_bin=511, min_child_samples=250, subsample=0.9,
              subsample_freq=1, colsample_bytree=0.6, reg_lambda=80.0,
              verbosity=-1, n_jobs=16, random_state=SEED)
N_EST, STOPPING = 8000, 200
SD_RATIO_GATE = (0.95, 1.45)   # RESEARCH w43: the member side-agreement gate


def say(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="windowed,global")
    ap.add_argument("--force", action="store_true", help="build even if the w96b gate says no")
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(",")]

    rc = w96b_gate.main()
    if rc != 0:
        if not a.force:
            say("\nw96b GATE DID NOT SAY BUILD -- refusing. Re-read experiments/w96b_prereg.txt.")
            return rc
        say("\n⚠ --force: building against a gate that did NOT say BUILD. Whatever this "
            "produces is NOT a pre-registered result and must not be reported as one.")

    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int)
    Xtr, Xte, Ktr, Kte = make_frames(tr, te)
    ntr, nte = len(Ktr), len(Kte)
    say(f"frame: {Xtr.shape[1]} base feats, {Ktr.shape[1]} keys, "
        f"train {ntr} test {nte} ({time.time()-t0:.0f}s)")

    full_keys = pd.concat([Ktr, Kte], ignore_index=True)
    axes_all = {}
    for c in NUM:
        cd, n_lev = _axis(full_keys, c)
        axes_all[c] = (cd[:ntr], cd[ntr:], n_lev)

    folds = get_folds(y)
    for arm in arms:
        name = f"lgbm_teprior_{arm}"
        oof = np.zeros(ntr)
        tp = np.zeros(nte)
        iters, aucs = [], []
        for fi, (itr, iva) in enumerate(folds):
            t1 = time.time()
            ya = y.iloc[itr]
            if arm == "global":
                t_tr, t_va, t_te = te_block(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte, SMOOTH)
            else:
                ax = {c: (axes_all[c][0][itr], axes_all[c][0][iva], axes_all[c][1],
                          axes_all[c][2]) for c in axes_all}
                t_tr, t_va, t_te = te_block_win(Ktr.iloc[itr], ya, Ktr.iloc[iva], Kte,
                                                ax, SMOOTH, WIN)
            Xa = pd.concat([Xtr.iloc[itr].reset_index(drop=True),
                            t_tr.reset_index(drop=True)], axis=1).to_numpy("float32")
            Xb = pd.concat([Xtr.iloc[iva].reset_index(drop=True),
                            t_va.reset_index(drop=True)], axis=1).to_numpy("float32")
            Xt = pd.concat([Xte.reset_index(drop=True),
                            t_te.reset_index(drop=True)], axis=1).to_numpy("float32")
            del t_tr, t_va, t_te
            gc.collect()
            m = lgb.LGBMClassifier(n_estimators=N_EST, **PARAMS)
            m.fit(Xa, ya.to_numpy(), eval_set=[(Xb, y.iloc[iva].to_numpy())],
                  eval_metric="auc", callbacks=[lgb.early_stopping(STOPPING, verbose=False)])
            oof[iva] = m.predict_proba(Xb)[:, 1]
            tp += m.predict_proba(Xt)[:, 1] / len(folds)
            iters.append(int(m.best_iteration_))
            aucs.append(float(roc_auc_score(y.iloc[iva], oof[iva])))
            say(f"  [{arm}] fold {fi}: AUC {aucs[-1]:.10f}  ({iters[-1]} trees, "
                f"{time.time()-t1:.0f}s)")
            del Xa, Xb, Xt, m
            gc.collect()

        cv = float(roc_auc_score(y, oof))
        ratio = float(np.std(tp) / np.std(oof))
        lo, hi = SD_RATIO_GATE
        say(f"[{arm}] OOF CV {cv:.10f}   per-fold mean {np.mean(aucs):.10f}   "
            f"sd_test/sd_oof {ratio:.4f} "
            f"{'PASS' if lo <= ratio <= hi else 'FAIL'} (gate [{lo}, {hi}])")
        save_preds(name, oof, tp, ntr, nte)
        json.dump({"name": name, "cv": cv, "params": {k: v for k, v in PARAMS.items()},
                   "seeds": [SEED], "frac": False, "iters": iters, "fold_aucs": aucs,
                   "sd_test_over_sd_oof": ratio, "smooth": SMOOTH,
                   "win": WIN if arm == "windowed" else None,
                   "built_by": "experiments/w96c_build_teprior_member.py"},
                  open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "oof", f"summary_{name}.json"), "w"), indent=2)
        say(f"[{arm}] saved as {name} ({time.time()-t0:.0f}s)")

    say(f"done ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
