"""Rank-average the joint-top zero-parameter h3 files into one candidate.

Motivation is operational, not modelling. Kaggle's default final selection takes the best
*public* submissions, and `blend158_logit` is currently tied for best public (0.97106) while
sitting -88e-6 +/- 9e-6 below the CV pick on the frozen folds -- a ~10 sigma gap. Until the
selection toggle is set by hand, the only lever this box has is to put CV-good files strictly
above 0.97106 so the default cannot land on that file. That makes a tenth slot spent on a
CV-top file worth more than one spent on a CV-poor leftover (`blend153_rankraw`, 0.970027).

This costs no refitting: every input is a saved cross-fitted OOF vector plus its submission
CSV, so the average is a pure function of files already on disk and fits zero parameters --
the same property that makes `h3` the deadline pick in the first place.

Inputs are the three joint-top h3 files that fit no weights. `blend159av_wh3` ties them on CV
but spends two fitted parameters, so it is deliberately excluded.

    make_topavg.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import SUB, TARGET, load_raw  # noqa: E402

PARTS = ("blend159av_h3", "blend160origm_h3", "blend158_h3")
NAME = "blendtop3"


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    oof, test, ids = [], [], None
    for p in PARTS:
        o = np.load(os.path.join(SUB, f"oof_{p}.npy"))
        d = pd.read_csv(os.path.join(SUB, f"{p}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"{p}.csv id order differs -- refusing to average")
        oof.append(rk(o))
        test.append(rk(d[TARGET].to_numpy()))
        print(f"  {p:20s} cross-fitted CV {roc_auc_score(y, o):.6f}")

    o_ens, t_ens = np.mean(oof, 0), np.mean(test, 0)
    cv = roc_auc_score(y, o_ens)

    pd.DataFrame({"id": ids, TARGET: t_ens}).to_csv(os.path.join(SUB, f"{NAME}.csv"),
                                                    index=False)
    np.save(os.path.join(SUB, f"oof_{NAME}.npy"), o_ens)

    best = max(roc_auc_score(y, np.load(os.path.join(SUB, f"oof_{p}.npy"))) for p in PARTS)
    print(f"\n{NAME}: cross-fitted CV {cv:.6f}   (best part {best:.6f}, "
          f"delta {cv - best:+.6f})")
    print(f"  wrote {os.path.join(SUB, NAME + '.csv')}  rows={len(ids):,}")


if __name__ == "__main__":
    main()
