"""w15h: the weight ladder -- give the anti-student class its BEST SHOT before killing it.

WHY THIS IS NECESSARY FOR A FAIR AUDIT
--------------------------------------
Every measurement of this correction so far, here and in w15e, has been taken at
raykkretzschmar's published weight 0.10. That weight was fixed on HIS anchors. Judging the
mechanism only at 0.10 on OUR anchors would be attacking a strawman: a correction carrying
real signal `S` but costing a toll `T(w)` has a net of roughly `w*S - T(w)`, and if `T`
grows faster in `w` than the signal does, then 0.10 can be net-negative while some smaller
weight is net-positive.

So sweep the weight and report the BEST the class can do on our stack, with its own
matched permutation null at every rung. If the maximum over `w` of the real delta is still
under the workspace's 2e-6 reproducibility floor (w14a), the class is dead for us at every
weight, which is a far stronger statement than "0.10 did not work".

Weight 0 is included as a hard control: it must read exactly 0.000e-6 in both arms.

INPUTS
------
  * `w15h_anti_hat_train.npy`   -- rayk's own published correction, transferred to the
    691,369 labelled rows by label-free function approximation (see `w15h_transfer.py`).
  * `w15h_rebuild_lookup_anti_train.npy` -- the mechanism rebuilt on our folds, if present.
  * the strongest sharp-minus-smooth member pair found by `w15h_pairs.py`.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, LIB, OOF, SUB, TARGET, get_folds  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
OUT = os.path.join(EXP, "w15h_weight.json")
NPZ = os.path.join(DATA, "w15e", "raykkretzschmar_s6e8-transductive-anti-student-signals",
                   "transductive_signals.npz")
DIRS = {"lib": os.path.join(LIB, "oof"), "own": OOF,
        "ext": os.path.join(DATA, "ext_members"), "ext2": os.path.join(DATA, "ext_members2")}
WEIGHTS = [0.0, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.25]
ANCHOR = "blend159av_h3"
N_PERM = 40


def pct(v):
    return rankdata(np.asarray(v, np.float64)) / len(v)


def build_anti(residual, reference):
    scale = reference.std() / residual.std()
    scaled = np.clip(residual * scale, -np.abs(reference).max(), np.abs(reference).max())
    rs = np.sign(reference) * np.abs(reference) ** 2
    ts = np.sign(scaled) * np.abs(scaled) ** 2
    return (ts - rs.mean()) * reference.std() / rs.std()


def main():
    t0 = time.time()
    y = pd.read_csv(os.path.join(DATA, "train.csv"))[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    base = pct(np.load(os.path.join(SUB, f"oof_{ANCHOR}.npy")))
    a0 = roc_auc_score(y, base)
    print(f"anchor {ANCHOR} cross-fitted OOF {a0:.7f}", flush=True)

    z = np.load(NPZ)
    reference = z["reference_contrast"].astype(np.float64)

    cands = {}
    p = os.path.join(EXP, "w15h_anti_hat_train.npy")
    if os.path.exists(p):
        cands["rayk_transferred"] = np.load(p)
    p = os.path.join(EXP, "w15h_rebuild_lookup_anti_train.npy")
    if os.path.exists(p):
        cands["rebuilt_transductive"] = np.load(p)
    # strongest member pair from w15h_pairs
    pj = os.path.join(EXP, "w15h_pairs.json")
    if os.path.exists(pj):
        best = json.load(open(pj))["best"]
        t = np.load(os.path.join(DIRS[best["teacher"].split(":")[0]],
                                 f"oof_{best['teacher'].split(':',1)[1]}.npy"))
        s = np.load(os.path.join(DIRS[best["student"].split(":")[0]],
                                 f"oof_{best['student'].split(':',1)[1]}.npy"))
        cands[f"pair_{best['teacher'].split(':')[1]}_{best['student'].split(':')[1]}"] = \
            build_anti(pct(t) - pct(s), reference)

    rows = []
    rg = np.random.default_rng(99)
    for nm, anti in cands.items():
        print(f"\n=== {nm}  (sd {anti.std():.6f}) ===", flush=True)
        for w in WEIGHTS:
            real = roc_auc_score(y, base + w * anti) - a0
            if w == 0.0:
                nm_, ns_ = 0.0, 0.0
            else:
                d = np.array([roc_auc_score(y, base + w * rg.permutation(anti)) - a0
                              for _ in range(N_PERM)])
                nm_, ns_ = float(d.mean()), float(d.std(ddof=1))
            per = [roc_auc_score(y[va], base[va] + w * anti[va])
                   - roc_auc_score(y[va], base[va]) for _, va in folds]
            rows.append(dict(correction=nm, w=w, real=float(real), null_mean=nm_,
                             null_sd=ns_, z=float((real - nm_) / ns_) if ns_ else 0.0,
                             folds_pos=int(sum(x > 0 for x in per))))
            print(f"  w={w:<6} real {real*1e6:+8.2f}e-6   null {nm_*1e6:+8.2f}"
                  f" +/- {ns_*1e6:.2f}   z {rows[-1]['z']:+6.2f}   folds+ "
                  f"{rows[-1]['folds_pos']}/5   {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP, "w15h_weight.csv"), index=False)
    summary = {}
    for nm in cands:
        sub = df[df.correction == nm]
        b = sub.loc[sub["real"].idxmax()]
        summary[nm] = dict(best_w=float(b["w"]), best_real=float(b["real"]),
                           best_z=float(b["z"]),
                           real_at_0p10=float(sub[sub.w == 0.10]["real"].iloc[0]))
        print(f"\n{nm}: best real delta {b['real']*1e6:+.2f}e-6 at w={b['w']} "
              f"(z {b['z']:+.2f});  at rayk's 0.10 it is "
              f"{summary[nm]['real_at_0p10']*1e6:+.2f}e-6")
    json.dump(summary, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}   {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
