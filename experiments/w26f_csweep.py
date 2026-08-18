"""w26f — sweep the L2 penalty on the STANDARDISED (converged) 187-member combiner.

Pre-registered in full at experiments/w26_prereg.txt ADDENDUM D1 BEFORE this ran once. The
grid, the hypotheses H-D1/H-D2/H-D3, the selection geometry and the ship rule are fixed there
and are not renegotiated here.

Short version. w23a tested "an L2 penalty helps at 187 members" and falsified it flat -- every
lam from 1e-8 to 1e-5 read -1.41 … +3.38e-6 and every one sign-flipped across reps. But that
sweep ran on the UNSTANDARDISED design, which w23 §1 showed in the same wave is (i) not at its
optimum, because tol=1e-4 is a max-gradient rule and the hybrid columns have sd 1.82 … 27.59,
and (ii) anisotropically shrunk, the widest member penalised ~230x less than the narrowest.
The isotropic converged design has never been swept. That is what this does.

SELECTION IS ON HELD-OUT ROWS, NOT ON CROSS-FITTED CV (prereg §D3). The combiner is cross-
fitted on the same frozen folds that produced its member OOF columns, and whether that leak
rewards a better-converged fit is exactly what w25d is testing; picking a hyperparameter on
the suspect instrument would be the rogii-wellbore failure with a different axis label. This
reuses w25d's splits exactly, so w25d's std arm IS the C=1.0 row and costs nothing to check.

Checkpoints after every (transform, C, rep) cell and resumes from disk: background jobs do not
survive the end of a run's session here and `setsid` is not installed.

    .venv/bin/python experiments/w26f_csweep.py --kind hybrid
    .venv/bin/python experiments/w26f_csweep.py --kind rankraw
    .venv/bin/python experiments/w26f_csweep.py --kind rescale
    .venv/bin/python experiments/w26f_csweep.py --report
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402
from blend_lab import HONEST_DROP  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HOLD = 0.20
KINDS = ("hybrid", "rankraw", "rescale")
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)   # prereg §D2 H-D3, FIXED, not to be widened
REPS = 5
SEED0 = 1000                 # w25d's splits exactly, so its std arm is the C=1.0 row
CV_GATE = 0.9701182          # w26d: CV needed for an even-money shot at the record, family h3
CSV = os.path.join(HERE, "w26f_csweep.csv")


def splits_for(y, reps, seed0=SEED0):
    out = []
    for r in range(reps):
        ip, ih = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=seed0 + r)
                      .split(np.zeros(len(y)), y))
        out.append((np.sort(ip), np.sort(ih)))
    return out


def load_done():
    if not os.path.exists(CSV):
        return pd.DataFrame(columns=["kind", "C", "rep", "hold_auc", "n_iter", "secs"])
    return pd.read_csv(CSV)


def append_row(row, store, kind):
    """One row + one score vector per cell, written atomically so a kill mid-write cannot
    leave a truncated artefact the resume path then trusts."""
    df = pd.concat([load_done(), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV + ".tmp", index=False)
    os.replace(CSV + ".tmp", CSV)
    npz = os.path.join(HERE, f"w26f_hold_{kind}.npz")
    np.savez_compressed(npz + ".tmp.npz", **store)
    os.replace(npz + ".tmp.npz", npz)


def run_kind(kind, seed0):
    t0 = time.time()
    done = load_done()
    done = {(r.C, r.rep) for r in done[done.kind == kind].itertuples()}
    npz = os.path.join(HERE, f"w26f_hold_{kind}.npz")
    store = dict(np.load(npz)) if os.path.exists(npz) else {}
    todo = [(c, r) for c in GRID for r in range(REPS) if (c, r) not in done]
    if not todo:
        print(f"{kind}: all {len(GRID)*REPS} cells already on disk")
        return
    print(f"{kind}: {len(todo)} of {len(GRID)*REPS} cells to run", flush=True)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    extra = tuple(os.path.join(DATA, d) for d in
                  ("ext_members", "ext_members2", "ext_members3"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    Z, _t = transform(O, T, kind)
    Z = Z.astype("float64")
    del O, T, _t, te
    gc.collect()
    print(f"{len(names)} members, {kind} Z {Z.shape}, {time.time()-t0:.0f}s", flush=True)

    for rep, (ip, ih) in enumerate(splits_for(y, REPS, seed0)):
        if all((c, rep) in done for c in GRID):
            continue
        Ztr, Zho, ytr, yho = Z[ip], Z[ih], y[ip], y[ih]
        s = Ztr.std(0)
        s[s <= 0] = 1.0
        Ztr /= s            # standardised design: the penalty is isotropic and lbfgs converges
        Zho /= s
        for C in GRID:
            if (C, rep) in done:
                continue
            t = time.time()
            m = LogisticRegression(max_iter=5000, C=C, tol=1e-4).fit(Ztr, ytr)
            el = time.time() - t
            d = m.decision_function(Zho)
            auc = roc_auc_score(yho, d)
            store[f"C{C}_rep{rep}"] = d.astype("float32")
            ni = int(np.ravel(m.n_iter_)[0])
            append_row(dict(kind=kind, C=C, rep=rep, hold_auc=auc, n_iter=ni, secs=el),
                       store, kind)
            print(f"  C {C:<6g} rep{rep}  auc {auc:.10f}  iters {ni:5d}  {el:6.1f}s",
                  flush=True)
            del m
        del Ztr, Zho
        gc.collect()
    print(f"{kind} done, {time.time()-t0:.0f}s")


def report():
    df = load_done()
    if df.empty:
        print("nothing on disk yet")
        return
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    sp = splits_for(y, REPS)

    print("=== per-transform holdout AUC by C, paired against C=1.0 (e-6) ===")
    per = {}
    for kind in KINDS:
        k = df[df.kind == kind]
        if k.empty:
            continue
        p = k.pivot(index="rep", columns="C", values="hold_auc")
        if 1.0 not in p.columns:
            print(f"  {kind}: no C=1.0 baseline yet, skipped")
            continue
        d6 = p.sub(p[1.0], axis=0) * 1e6
        per[kind] = {float(c): float(d6[c].mean()) for c in d6.columns}
        print(f"\n  {kind}")
        for c in sorted(d6.columns):
            v = d6[c].dropna()
            if not len(v):
                continue
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            print(f"    C {c:<6g}  D {v.mean():+7.3f}e-6  se {se:5.3f}  "
                  f"{int((v > 0).sum())}/{len(v)} positive")

    have = [k for k in KINDS if not df[df.kind == k].empty]
    if len(have) == len(KINDS):
        print("\n=== the h3 MIX (rank-average of the three stacks) on held-out rows ===")
        z = {k: np.load(os.path.join(HERE, f"w26f_hold_{k}.npz")) for k in KINDS}
        rows = []
        for C in GRID:
            ds = []
            for rep, (_ip, ih) in enumerate(sp):
                keys = [f"C{C}_rep{rep}" for _ in KINDS]
                if not all(k in z[kk].files for kk, k in zip(KINDS, keys)):
                    ds = None
                    break
                e = np.mean([rankdata(z[kk][f"C{C}_rep{rep}"]) for kk in KINDS], 0)
                ds.append(roc_auc_score(y[ih], e))
            if ds is not None:
                rows.append(dict(C=C, mix_auc=float(np.mean(ds)),
                                 per_rep=[float(v) for v in ds]))
        if rows:
            mx = pd.DataFrame(rows)
            base = mx[mx.C == 1.0].mix_auc
            if len(base):
                mx["D6"] = (mx.mix_auc - base.iloc[0]) * 1e6
            print(mx[["C", "mix_auc", "D6"]].to_string(index=False))
            best = mx.loc[mx.mix_auc.idxmax()]
            print(f"\nbest cell C={best.C:g} at {best.D6:+.3f}e-6 vs C=1.0")
            print("PREREG §D2 H-D1 registered -2 to +5e-6 and EXPECTED FAILURE; anything "
                  "above +8e-6\nis to be disbelieved and re-run on fresh splits "
                  "(random_state 2000+r) before it is built on.")
            print(f"PREREG §D3 SHIP RULE: a cell ships only if it wins HERE *and* its "
                  f"cross-fitted CV\nclears {CV_GATE:.7f}. This script measures the first "
                  f"half only -- the second needs a\nfull cross-fitted build and is NOT "
                  f"implied by anything above.")
            mx.to_csv(os.path.join(HERE, "w26f_mix.csv"), index=False)
    json.dump(dict(grid=list(GRID), reps=REPS, per_transform=per, cv_gate=CV_GATE),
              open(os.path.join(HERE, "w26f_csweep.json"), "w"), indent=1)
    print("\nwrote experiments/w26f_{csweep.json,mix.csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=KINDS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seed0", type=int, default=SEED0,
                    help="1000 = w25d's splits (default). 2000 = the FRESH splits prereg "
                         "§D5 requires before believing any winner.")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        run_kind(a.kind, a.seed0)


if __name__ == "__main__":
    main()
