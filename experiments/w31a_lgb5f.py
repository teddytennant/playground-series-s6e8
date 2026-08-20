"""w31a -- the §K follow-on: 5 folds, 2000 rounds, three configs, two CT arms, four stages.

WHY THIS EXISTS
---------------
w27g_tunect.py's fold-0 table fired its own escape clause: K2(d) came in at +146.75e-6
against a registered 0..+80e-6, and R-K3 says verbatim "if the headline clears +80e-6, the
follow-on is a 5-fold re-run of that one config against the control, not a member export."
This is that re-run. The grid is frozen at three configs and no search happens here.

THE CONFOUND THE FOLD-0 TABLE DID NOT CONTROL
----------------------------------------------
Every §K number was fitted at 400 rounds. The control config IS `lgbm_fixed_lat`, which is
shipped at 2000 rounds for CV 0.9677108350; the same config at 400 rounds pools to
0.9654813306. Rounds 400 -> 2000 are worth +2229.5e-6, which is 1.5x the entire 14-config
fold-0 spread. At fixed rounds `num_leaves` and `learning_rate` are speed-of-convergence
knobs as much as capacity knobs, and the two fold-0 argmaxes (lr0.05, leaves255_dinf) are
exactly the two configs that reading predicts. So the follow-on runs at the operating point.

TWO FREE AXES, ONE FIT
-----------------------
  rounds: boosting is sequential, so the first r trees of a 2000-round fit ARE the r-round
          fit. `num_iteration=r` gives every stage off one booster. Gate (a) checks this
          against w26l_r400 rather than assuming it.
  CT arm: rescaling a tree's input by s is the same function as rescaling that feature's
          thresholds by s (§G3, verified maxdiff 0.000e+00). So "fit with CT x s" == "this
          booster served CT / s", and both arms come off one booster on identical rows.

Registered in full at experiments/w31_prereg_slot2.txt §N before this was started.

    .venv/bin/python experiments/w31a_lgb5f.py --fold 0 --jobs 3      # one process per fold
    .venv/bin/python experiments/w31a_lgb5f.py --report              # pooled table, R-N1
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

CKPT = os.path.join(os.path.dirname(HERE), "cache", "lgb5fckpt")

# lgbm_fixed_lat, byte-identical to w27g_tunect.CONTROL and w26l_serve.PARAMS.
CONTROL = dict(learning_rate=0.025, num_leaves=63, max_depth=7, max_bin=511,
               min_child_samples=250, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.6, reg_lambda=80.0)
GRID = [
    ("control",        {}),
    ("leaves255_dinf", dict(num_leaves=255, max_depth=-1)),   # R-K3's config
    ("lr0.05",         dict(learning_rate=0.05)),             # fold-0 argmax at s=1.0
]
STAGES = [400, 800, 1400, 2000]
SCALES = [1.0, 4.0 / 3.0]

# gate (a)/(b) targets, from w26l_r400.log and oof/summary_lgbm_fixed_lat.json
GATE_R400 = {1.0: 0.9654813306, 4.0 / 3.0: 0.9657751945}
GATE_R2000_S1 = 0.9677106709      # w26k_run.log, the 2000-round refit of the control
GATE_R2000_S43 = 0.9682861183     # ditto, s=4/3 -- the known constant §N5(b2)


def save_atomic(path, **arrs):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:      # np.savez appends .npz to a NAME, not to a handle
        np.savez(fh, **arrs)
    os.replace(tmp, path)


def ct_index():
    cols = json.load(open(os.path.join(CACHE, "cols.json")))
    return cols, np.array([i for i, c in enumerate(cols) if c.startswith("CT_")])


def run_fold(a):
    cols, ct = ct_index()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    itr, iva = get_folds(y)[a.fold]

    g = lambda k: np.load(os.path.join(CACHE, f"f{a.fold}_{k}.npy"))
    Xa, ya, Xb, yb = g("Xa"), g("ya"), g("Xb"), g("yb")
    assert len(yb) == len(iva), "fold order does not match the cache"

    # w27g's premise gate, free: the cached valid CT_ must sit ~4/3 of the cached train CT_.
    med = float(np.median(Xb[:, ct].mean(0) / np.maximum(Xa[:, ct].mean(0), 1e-9)))
    print(f"[w31a f{a.fold}] GATE median valid/train CT_ ratio = {med:.4f} (expect 1.3333)",
          flush=True)
    if not (1.28 <= med <= 1.39):
        raise SystemExit(f"[w31a f{a.fold}] GATE FAILED at {med:.4f} -- the cache is not the "
                         f"one §G measured. Every arm below would be meaningless. STOP.")

    t0 = time.time()
    for tag, delta in GRID:
        p = os.path.join(CKPT, f"{tag}_f{a.fold}.npz")
        if os.path.exists(p):
            z = np.load(p)
            if z["oof"].shape == (len(yb), len(STAGES), len(SCALES)):
                print(f"[w31a f{a.fold}] {tag:15s} resumed", flush=True)
                continue
            print(f"[w31a f{a.fold}] {tag:15s} STALE checkpoint, refitting", flush=True)

        mp = os.path.join(CKPT, f"{tag}_f{a.fold}.txt")
        if os.path.exists(mp):
            bst = lgb.Booster(model_file=mp)
            print(f"[w31a f{a.fold}] {tag:15s} booster loaded from disk", flush=True)
        else:
            params = dict(CONTROL)
            params.update(delta)
            m = lgb.LGBMClassifier(n_estimators=a.rounds, random_state=a.seed,
                                   verbose=-1, n_jobs=a.jobs, **params)
            m.fit(Xa, ya)
            bst = m.booster_
            bst.save_model(mp + ".tmp")
            os.replace(mp + ".tmp", mp)
            del m
            print(f"[w31a f{a.fold}] {tag:15s} fitted ({time.time()-t0:.0f}s)", flush=True)

        gain = bst.feature_importance("gain")
        share = float(gain[ct].sum() / max(gain.sum(), 1e-9))

        Xt = np.load(os.path.join(CACHE, f"f{a.fold}_Xt.npy"))
        ob = np.zeros((len(yb), len(STAGES), len(SCALES)), dtype="float64")
        ot = np.zeros((len(Xt), len(STAGES), len(SCALES)), dtype="float64")
        b0, t0c = Xb[:, ct].copy(), Xt[:, ct].copy()
        for j, s in enumerate(SCALES):
            Xb[:, ct] = b0 / s
            Xt[:, ct] = t0c / s
            for i, r in enumerate(STAGES):
                ob[:, i, j] = bst.predict(Xb, num_iteration=r)
                ot[:, i, j] = bst.predict(Xt, num_iteration=r)
        Xb[:, ct] = b0
        del Xt, t0c

        save_atomic(p, oof=ob, test=ot, share=np.float64(share))
        line = "  ".join(f"r{r}/{s:.3f} {roc_auc_score(yb, ob[:, i, j]):.6f}"
                         for i, r in enumerate(STAGES) for j, s in enumerate(SCALES))
        print(f"[w31a f{a.fold}] {tag:15s} CTshare {share*100:.1f}%  {line}  "
              f"({time.time()-t0:.0f}s)", flush=True)


def report():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    have = {}
    for tag, _ in GRID:
        oof = np.zeros((len(y), len(STAGES), len(SCALES)))
        tp = np.zeros((len(te), len(STAGES), len(SCALES)))
        ok = True
        for f, (_, iva) in enumerate(folds):
            p = os.path.join(CKPT, f"{tag}_f{f}.npz")
            if not os.path.exists(p):
                ok = False
                break
            z = np.load(p)
            oof[iva] = z["oof"]
            tp += z["test"] / N_SPLITS
        if ok:
            have[tag] = (oof, tp)
            np.save(os.path.join(HERE, f"w31a_{tag}_oof.npy"), oof.astype("float32"))
            np.save(os.path.join(HERE, f"w31a_{tag}_test.npy"), tp.astype("float32"))
        else:
            print(f"[w31a] {tag}: incomplete, skipped (R-N5)", flush=True)
    if not have:
        return

    cv = {}
    print(f"\n[w31a] POOLED OOF over {len(y):,} rows -- R-N1, the only currency\n")
    print(f"  {'config':16s} {'stage':>6s} {'AUC@1.0':>14s} {'AUC@4/3':>14s} {'d_c':>12s}"
          f" {'vs ctl@1.0':>12s} {'vs ctl@4/3':>12s}")
    for tag, _ in GRID:
        if tag not in have:
            continue
        for i, r in enumerate(STAGES):
            a10 = roc_auc_score(y, have[tag][0][:, i, 0])
            a43 = roc_auc_score(y, have[tag][0][:, i, 1])
            cv[(tag, r, 1.0)] = a10
            cv[(tag, r, 4 / 3)] = a43
            c10 = cv.get(("control", r, 1.0))
            c43 = cv.get(("control", r, 4 / 3))
            vs = (f"{(a10-c10)*1e6:+10.2f}e-6 {(a43-c43)*1e6:+10.2f}e-6"
                  if c10 is not None else " " * 27)
            print(f"  {tag:16s} {r:6d} {a10:14.10f} {a43:14.10f} "
                  f"{(a43-a10)*1e6:+10.2f}e-6 {vs}")

    print("\n  -- N2 readouts --")
    if ("control", 400, 1.0) in cv:
        for s, want in ((1.0, GATE_R400[1.0]), (4 / 3, GATE_R400[4 / 3])):
            got = cv[("control", 400, s)]
            print(f"  N2(a) GATE control@400 s={s:.3f}: {got:.10f} vs w26l {want:.10f} "
                  f"-> {(got-want)*1e6:+.2f}e-6  {'PASS' if abs(got-want) < 20e-6 else 'FAIL'}")
    if ("control", 2000, 1.0) in cv:
        got = cv[("control", 2000, 1.0)]
        print(f"  N2(b) GATE control@2000 s=1.0: {got:.10f} vs lgbm_fixed_lat "
              f"{GATE_R2000_S1:.10f} -> {(got-GATE_R2000_S1)*1e6:+.2f}e-6  "
              f"{'PASS' if abs(got-GATE_R2000_S1) < 20e-6 else 'FAIL'}")
        g43 = cv[("control", 2000, 4 / 3)]
        print(f"  N5(b2) GATE control@2000 s=4/3: {g43:.10f} vs w26k {GATE_R2000_S43:.10f} "
              f"-> {(g43-GATE_R2000_S43)*1e6:+.2f}e-6  "
              f"{'PASS' if abs(g43-GATE_R2000_S43) < 20e-6 else 'FAIL'}")
        print(f"        rounds 400->2000 on the control @1.0: "
              f"{(cv[('control',2000,1.0)]-cv[('control',400,1.0)])*1e6:+.1f}e-6")
    if ("leaves255_dinf", 2000, 4 / 3) in cv:
        d = (cv[("leaves255_dinf", 2000, 4 / 3)] - cv[("control", 2000, 4 / 3)]) * 1e6
        print(f"  N2(c) HEADLINE d_2000 = leaves255_dinf - control, both @4/3 @2000 = "
              f"{d:+.2f}e-6   (fold-0 @400 was +1466.40e-6, i.e. {100*d/1466.4:+.1f}% of it; "
              f"registered point +150, range -100..+500)")
    if ("leaves255_dinf", 2000, 1.0) in cv:
        dl = (cv[("leaves255_dinf", 2000, 4 / 3)] - cv[("leaves255_dinf", 2000, 1.0)]) * 1e6
        dc = (cv[("control", 2000, 4 / 3)] - cv[("control", 2000, 1.0)]) * 1e6
        print(f"  N5(g) d_c on leaves255_dinf @2000 = {dl:+.2f}e-6, ratio to the control's "
              f"{dc:+.1f}e-6 is {dl/dc:.2f}x   (registered +600..+1100e-6, point +800, ratio "
              f"compressing from 2.09x toward ~1.4x)")
    if ("control", 2000, 1.0) in cv:
        b10 = max((cv[(t, 2000, 1.0)], t) for t, _ in GRID if (t, 2000, 1.0) in cv)
        b43 = max((cv[(t, 2000, 4 / 3)], t) for t, _ in GRID if (t, 2000, 4 / 3) in cv)
        dc = cv[("control", 2000, 4 / 3)] - cv[("control", 2000, 1.0)]
        print(f"  N2(d) headline at the operating point: best(4/3)={b43[1]} - "
              f"best(1.0)={b10[1]} - ctl d_c({dc*1e6:.2f}e-6) = "
              f"{((b43[0]-b10[0])-dc)*1e6:+.2f}e-6   (registered 0..+80, modal +15)")
        print(f"  N2(e) control d_c by stage: " + "  ".join(
            f"r{r} {(cv[('control',r,4/3)]-cv[('control',r,1.0)])*1e6:+.1f}e-6"
            for r in STAGES if ("control", r, 1.0) in cv) +
            f"   (w26l @400 pooled: +293.86e-6; registered still positive, 0.5x-1.5x, "
            f"trend INCREASING)")
        print(f"  N2(f) argmax @1.0 = {b10[1]}   argmax @4/3 = {b43[1]}   "
              f"{'DIFFER' if b10[1] != b43[1] else 'SAME'}   (registered P(differ)=0.35)")
    print("\n  R-N3: nothing here is exported or shipped without beating the control at the "
          "SAME stage and scale. R-N4: at cap today; nothing here is submitted or moves "
          "WANTED.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    os.makedirs(CKPT, exist_ok=True)
    if a.report:
        report()
        return
    if a.fold is None:
        raise SystemExit("--fold k, or --report")
    run_fold(a)


if __name__ == "__main__":
    main()
