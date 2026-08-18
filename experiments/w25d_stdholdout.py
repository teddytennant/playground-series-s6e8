"""w25d — does the standardisation's +8.46e-6 of CROSS-FITTED CV survive an honest holdout?

Pre-registered in full at experiments/w25_prereg.txt §2-§5 BEFORE this ran once. The decision
rule S1/S2/S3 is fixed there and is not revisited here.

Short version. Four matched public-LB pairs landed this slot, each isolating the
standardisation, each printing ONE reporting step BELOW its unstandardised twin despite being
+8.0 to +15.9e-6 above it on cross-fitted CV. w24 R3 forbids moving the deadline pick on an LB
measurement and it is not being moved. R3 does NOT forbid asking whether the CV number is
honest, and there is a named mechanism that predicts this: the combiner is cross-fitted on the
SAME frozen folds that produced its 187 member OOF vectors, so a better-converged combiner can
exploit that leak harder -- and letting lbfgs actually converge is the whole documented effect
of standardising.

The test is w23c's geometry done properly: R reps of an 80/20 StratifiedShuffleSplit, combiner
fit on the pool and scored on rows it never saw, both arms, all three h3 transforms,
rank-averaged into the h3 mix on the holdout rows. Paired within rep.

Run ONE TRANSFORM PER PROCESS. The first version of this script held all three transform
matrices' worth of intermediates in one process and was killed by the OS partway through
transform 1 with no traceback, after correctly reproducing the gate value. Splitting the work
means a kill costs one transform, not the run, and the peak footprint is ~2.1 GB: the std arm
now scales Ztr/Zho IN PLACE after the unstd fit rather than allocating a second copy.

  .venv/bin/python experiments/w25d_stdholdout.py --kind hybrid  --reps 5
  .venv/bin/python experiments/w25d_stdholdout.py --kind rankraw --reps 5
  .venv/bin/python experiments/w25d_stdholdout.py --kind rescale --reps 5
  .venv/bin/python experiments/w25d_stdholdout.py --combine     --reps 5
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
KINDS = ("hybrid", "rankraw", "rescale")          # the h3 mix, exactly as shipped
W23C = {"unstd": 0.9703668630916813, "std": 0.9703699239648432}   # rep-0 hybrid gate
CROSSFIT = 8.46      # w23_ad187std_h3 0.9701092751 - w20_ad187_h3 0.9701008150, e-6


def splits_for(y, reps):
    out = []
    for r in range(reps):
        ip, ih = next(StratifiedShuffleSplit(1, test_size=HOLD, random_state=1000 + r)
                      .split(np.zeros(len(y)), y))
        out.append((np.sort(ip), np.sort(ih)))
    return out


def checkpoint(kind, store, rows):
    """Write both artefacts after EVERY rep, atomically, so a kill costs one rep.

    w25d has now been killed mid-run twice (w25 §8, w26 §1a), both times losing every
    completed rep because the writes only happened after the last one. Reps 0-3 of hybrid
    had already been computed and printed to a log when the second kill landed. Write via
    a temp file + os.replace so a kill DURING the write cannot leave a truncated artefact
    that the resume path would then trust.
    """
    npz, csv = (os.path.join(HERE, f"w25d_hold_{kind}.npz"),
                os.path.join(HERE, f"w25d_arms_{kind}.csv"))
    np.savez_compressed(npz + ".tmp.npz", **store)
    os.replace(npz + ".tmp.npz", npz)
    pd.DataFrame(rows).to_csv(csv + ".tmp", index=False)
    os.replace(csv + ".tmp", csv)


def resume(kind, reps):
    """Return (store, rows, done) from a previous partial run, or empty if there is none.

    A rep counts as done only when BOTH arms are on disk in BOTH artefacts -- a rep killed
    between its unstd and std fits is recomputed whole, since the two must share the split
    and the paired D is meaningless otherwise.
    """
    npz, csv = (os.path.join(HERE, f"w25d_hold_{kind}.npz"),
                os.path.join(HERE, f"w25d_arms_{kind}.csv"))
    if not (os.path.exists(npz) and os.path.exists(csv)):
        return {}, [], set()
    z = np.load(npz)
    df = pd.read_csv(csv)
    done = {r for r in range(reps)
            if all(f"{a}_rep{r}" in z.files for a in ("unstd", "std"))
            and len(df[(df.rep == r) & (df.arm.isin(["unstd", "std"]))]) == 2}
    store = {k: z[k] for k in z.files if int(k.split("rep")[1]) in done}
    rows = df[df.rep.isin(done)].to_dict("records")
    if done:
        print(f"  resuming {kind}: reps {sorted(done)} already on disk", flush=True)
    return store, rows, done


def run_kind(kind, reps):
    t0 = time.time()
    store, rows, done = resume(kind, reps)
    if len(done) >= reps:
        print(f"{kind}: all {reps} reps already on disk, nothing to do")
        return
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
    print(f"{len(names)} members, {kind} Z {Z.shape} {Z.dtype}, {time.time()-t0:.0f}s",
          flush=True)

    for r, (ip, ih) in enumerate(splits_for(y, reps)):
        if r in done:
            continue
        Ztr, Zho, ytr, yho = Z[ip], Z[ih], y[ip], y[ih]
        s = Ztr.std(0)
        s[s <= 0] = 1.0
        for arm in ("unstd", "std"):
            if arm == "std":
                # in place: the unstd fit is done with these buffers and a second copy is
                # what killed the first version of this script.
                Ztr /= s
                Zho /= s
            t = time.time()
            m = LogisticRegression(max_iter=5000, C=1.0, tol=1e-4).fit(Ztr, ytr)
            el = time.time() - t
            d = m.decision_function(Zho)
            auc = roc_auc_score(yho, d)
            store[f"{arm}_rep{r}"] = d.astype("float32")
            ni = int(np.ravel(m.n_iter_)[0])
            rows.append(dict(kind=kind, rep=r, arm=arm, hold_auc=auc, n_iter=ni, secs=el))
            print(f"  rep{r} {arm:5s} auc {auc:.10f}  iters {ni:5d}  {el:6.1f}s", flush=True)
            del m
        del Ztr, Zho
        gc.collect()
        checkpoint(kind, store, rows)

    print(f"wrote w25d_hold_{kind}.npz + w25d_arms_{kind}.csv   {time.time()-t0:.0f}s")


def combine(reps):
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr
    sp = splits_for(y, reps)

    df = pd.concat([pd.read_csv(os.path.join(HERE, f"w25d_arms_{k}.csv")) for k in KINDS])
    df.to_csv(os.path.join(HERE, "w25d_arms.csv"), index=False)
    z = {k: np.load(os.path.join(HERE, f"w25d_hold_{k}.npz")) for k in KINDS}

    print("=== P2 REPRODUCTION GATE (rep 0, hybrid, vs w23c) ===")
    gate_ok = True
    for arm in ("unstd", "std"):
        got = df[(df.kind == "hybrid") & (df.rep == 0) & (df.arm == arm)].hold_auc.iloc[0]
        d6 = (got - W23C[arm]) * 1e6
        ok = abs(d6) < 0.5
        gate_ok &= ok
        print(f"  {arm:5s} got {got:.10f}  w23c {W23C[arm]:.10f}  diff {d6:+.3f}e-6  "
              f"{'PASS' if ok else 'FAIL'}")
    if not gate_ok:
        print("\n!! GATE FAILED -- per prereg P2 nothing below is read and no rule fires.")

    print("\n=== the h3 MIX (rank-average of the three stacks) on held-out rows ===")
    mix = []
    for r, (_ip, ih) in enumerate(sp):
        yho, rec = y[ih], {"rep": r}
        for arm in ("unstd", "std"):
            e = np.mean([rankdata(z[k][f"{arm}_rep{r}"]) for k in KINDS], 0)
            rec[arm] = roc_auc_score(yho, e)
        rec["D6"] = (rec["std"] - rec["unstd"]) * 1e6
        mix.append(rec)
        print(f"  rep{r}  unstd {rec['unstd']:.10f}  std {rec['std']:.10f}  "
              f"D {rec['D6']:+.3f}e-6")
    mx = pd.DataFrame(mix)
    mx.to_csv(os.path.join(HERE, "w25d_mix.csv"), index=False)

    D = mx.D6.mean()
    se = mx.D6.std(ddof=1) / np.sqrt(len(mx))
    npos = int((mx.D6 > 0).sum())
    print(f"\nD = {D:+.3f}e-6   se {se:.3f}   {npos}/{len(mx)} reps positive")
    print(f"cross-fitted figure under test: +{CROSSFIT}e-6      "
          f"ratio holdout/cross-fitted: {D/CROSSFIT:.2f}")

    print("\n=== per-transform D, e-6 ===")
    per = {}
    for kind in KINDS:
        k = df[df.kind == kind].pivot(index="rep", columns="arm", values="hold_auc")
        d6 = (k["std"] - k["unstd"]) * 1e6
        per[kind] = float(d6.mean())
        print(f"  {kind:8s} {d6.mean():+7.3f}  se {d6.std(ddof=1)/np.sqrt(len(d6)):5.3f}  "
              f"{int((d6>0).sum())}/{len(d6)} positive")

    print("\n=== P3: iteration counts by arm ===")
    print(df.groupby("arm").n_iter.agg(["mean", "min", "max"]).round(1).to_string())

    if not gate_ok:
        verdict = "GATE FAILED -- no rule fires"
    elif D >= 6.0:
        verdict = ("S1 -- cross-fitted gain corroborated by an independent holdout; "
                   "WANTED UNCHANGED; the leak hypothesis is FALSIFIED")
    elif D >= 2.0:
        verdict = ("S2 -- gain inflated but still positive; WANTED UNCHANGED; "
                   "restate the standardisation's value as D, not +8.46e-6")
    else:
        verdict = ("S3 -- gain does not survive an honest holdout; SWAP slots 1 and 2 of "
                   "WANTED to {w21_ad187corr.csv, w23_ad187stdcorr.csv}")
    print(f"\n=== PRE-REGISTERED VERDICT (w25_prereg.txt §4): {verdict} ===")

    json.dump(dict(reps=reps, D=float(D), se=float(se), n_pos=npos, gate_ok=bool(gate_ok),
                   crossfitted=CROSSFIT, per_transform=per, verdict=verdict,
                   per_rep=mx.to_dict("records")),
              open(os.path.join(HERE, "w25d_stdholdout.json"), "w"), indent=1)
    print("wrote experiments/w25d_{arms,mix}.csv + w25d_stdholdout.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default=None, choices=KINDS)
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    if a.combine:
        combine(a.reps)
    else:
        run_kind(a.kind, a.reps)


if __name__ == "__main__":
    main()
