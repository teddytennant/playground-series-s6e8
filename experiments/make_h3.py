"""Assemble the drop-logit (`h3`) rank-ensemble from per-transform files already on disk.

`blend_lab --build` writes each transform stack's submission CSV and its cross-fitted OOF
vector as a by-product of the ensemble build. The h3 variant -- equal rank-average of
hybrid/rankraw/rescale, dropping `logit` -- is a pure function of those, so it costs no
refitting at all: read four files, rank-average three, write one.

Dropping `logit` was worth +4e-6 cross-fitted at 156 members and is the one blend decision
here with zero fitted parameters (see the 2026-08-11 journal entry).

    make_h3.py blend158
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

KEEP = ("hybrid", "rankraw", "rescale")


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "blend158"
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    oof, test, ids = [], [], None
    for k in KEEP:
        o = np.load(os.path.join(SUB, f"oof_{base}_{k}.npy"))
        d = pd.read_csv(os.path.join(SUB, f"{base}_{k}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"{base}_{k}.csv id order differs -- refusing to average")
        oof.append(rk(o))
        test.append(rk(d[TARGET].to_numpy()))
        print(f"  {k:8s} oof AUC {roc_auc_score(y, o):.6f}")

    o_ens, t_ens = np.mean(oof, 0), np.mean(test, 0)
    cv = roc_auc_score(y, o_ens)
    name = f"{base}_h3"
    pd.DataFrame({"id": ids, TARGET: t_ens}).to_csv(os.path.join(SUB, f"{name}.csv"),
                                                    index=False)
    np.save(os.path.join(SUB, f"oof_{name}.npy"), o_ens)

    full = np.load(os.path.join(SUB, f"oof_{base}.npy"))
    print(f"\n{name}: cross-fitted CV {cv:.6f}   ({base} all-four: "
          f"{roc_auc_score(y, full):.6f})")
    print(f"  wrote {os.path.join(SUB, name + '.csv')}  rows={len(ids):,}")


if __name__ == "__main__":
    main()
