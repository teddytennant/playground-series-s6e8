"""w26n -- how novel is a corrected member? maxcorr against the pack, and against its own
uncorrected twin.

The pack's standing closure (RESEARCH, 'What this proved about the pack') is that a member
decorrelates from the pack's span only by being WORSE -- `orig_binm` at maxcorr 0.879 was
worth -1e-6 to -2e-6, `xgb_cat_lattice` at 0.9746 was worth 0. A corrected lattice member is
the one case that argument does not obviously cover: it is the same function class on the
same features, differing only by a displacement that EVERY member in the pack shares. So its
correlation with its own uncorrected twin is the number that matters, not its correlation
with the pack minimum.

    .venv/bin/python experiments/w26n_corr.py --src experiments/w26l_r400 --arms ct1.0000,ct1.3333
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members  # noqa: E402
sys.path.insert(0, HERE)
from w26l_serve import ARMS as L_ARMS  # noqa: E402
from w26k_ctscale import SCALES  # noqa: E402

K_ARMS = [f"ct{s:.4f}" for s in SCALES]
EXTRA = [os.path.join(DATA, d) for d in
         ("ext_members", "ext_members2", "ext_members3", "ext_members4", "ext_members5")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--arms", default="ct1.0000,ct1.3333")
    a = ap.parse_args()

    oof = np.load(a.src + "_oof.npy")
    arms = L_ARMS if oof.shape[1] == len(L_ARMS) else K_ARMS
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    names, O, _ = load_members(y, len(te), extra_dirs=[d for d in EXTRA if os.path.isdir(d)])
    print(f"pack: {len(names)} members")

    R = np.column_stack([rankdata(O[:, i]) for i in range(O.shape[1])])
    R = (R - R.mean(0)) / R.std(0)
    want = [w.strip() for w in a.arms.split(",")]
    cols = {}
    for w in want:
        j = arms.index(w)
        v = rankdata(oof[:, j]); v = (v - v.mean()) / v.std()
        cols[w] = v
        c = np.abs(R.T @ v) / len(y)
        k = int(np.argmax(c))
        print(f"  {w:>10s}  solo {roc_auc_score(y, oof[:, j]):.10f}  "
              f"maxcorr {c[k]:.6f} vs {names[k]}   (pack median {np.median(c):.6f})")
    if len(want) == 2:
        u, v = cols[want[0]], cols[want[1]]
        print(f"\n  spearman({want[0]}, {want[1]}) = {float(u @ v) / len(y):.8f}")
        print("  -- the pack holds the FIRST of these and nothing like the second; that "
              "correlation is the ceiling on how much new direction the fix can carry.")


if __name__ == "__main__":
    main()
