"""Second pass of the one-parameter instrument, on the REPAIRED original member.

`w15d_twoway.py` returned w* = 0.000 exactly for `orig_binm` against four packs, in-sample
and cross-fitted, while the pure-noise control took w* = 0.002. So the stack's condition
number was not hiding anything: the member's contribution really is zero.

`w15d_origrepair.py` then lifted the original-trained transfer from 0.8853 to 0.9215 by
mapping the original's budget ratio onto the competition's. This asks the same question of
the stronger member -- and of the pair, since `none` and `r` correlate only +0.815 and may
be two channels rather than one.

    .venv/bin/python experiments/w15d_twoway2.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, OOF, SUB, TARGET, get_folds  # noqa: E402

RNG = np.random.default_rng(20260815)
PACKS = ["blend159av_h3", "blend160origm_h3"]


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def search_w(y, a, b, grid):
    sc = np.array([roc_auc_score(y, (1.0 - w) * a + w * b) for w in grid])
    i = int(np.argmax(sc))
    return grid[i], sc[i]


def main():
    y = pd.read_csv(os.path.join(DATA, "train.csv"),
                    usecols=[TARGET])[TARGET].astype(int).to_numpy()
    folds = get_folds(y)
    grid = np.round(np.concatenate([np.arange(0.0, 0.301, 0.002),
                                    np.arange(0.32, 1.001, 0.02)]), 4)

    mems = {
        "origrep_r": rk(np.load(os.path.join(OOF, "oof_w15d_origrep_r.npy"))),
        "orig_binm": rk(np.load(os.path.join(OOF, "oof_orig_binm.npy"))),
    }
    mems["NOISE"] = rk(RNG.random(len(y)))
    # the two original-trained members averaged: rho 0.815, so possibly two channels
    mems["origrep+binm"] = rk(mems["origrep_r"] + mems["orig_binm"])

    for m, v in mems.items():
        print(f"  member {m:14s} solo AUC {roc_auc_score(y, v):.6f}")
    print()

    rows = []
    for p in PACKS:
        pv = rk(np.load(os.path.join(SUB, f"oof_{p}.npy")))
        base = roc_auc_score(y, pv)
        print(f"=== pack {p}   OOF AUC {base:.6f} ===")
        for m, mv in mems.items():
            w_in, s_in = search_w(y, pv, mv, grid)
            d, ws = [], []
            for tr_i, va_i in folds:
                w_cf, _ = search_w(y[tr_i], pv[tr_i], mv[tr_i], grid)
                d.append(roc_auc_score(y[va_i], (1 - w_cf) * pv[va_i] + w_cf * mv[va_i])
                         - roc_auc_score(y[va_i], pv[va_i]))
                ws.append(w_cf)
            d = np.array(d)
            print(f"  {m:14s} in-sample w*={w_in:.3f} gain {(s_in-base)*1e6:+8.2f}e-6   |   "
                  f"cross-fitted w={np.mean(ws):.3f} dAUC {d.mean()*1e6:+8.2f}e-6 "
                  f"+-{d.std(ddof=1)/np.sqrt(5)*1e6:.2f}  ({(d>0).sum()}/5 positive)")
            rows.append(dict(pack=p, member=m, base=base, w_in=w_in, gain_in=s_in - base,
                             w_cf=float(np.mean(ws)), cf_mean=d.mean(),
                             cf_se=d.std(ddof=1) / np.sqrt(5), cf_pos=int((d > 0).sum())))
        print()

    pd.DataFrame(rows).to_csv("experiments/w15d_twoway2.csv", index=False)
    print("wrote experiments/w15d_twoway2.csv")


if __name__ == "__main__":
    main()
