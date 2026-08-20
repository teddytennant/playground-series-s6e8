"""w31b -- slice one (config, stage, scale) cell out of a w31a run and write it as a member.

R-N3, binding: a cell is only exportable if its POOLED 5-fold OOF beats the control at the
SAME stage and the SAME scale. This script re-checks that itself rather than trusting the
caller, and refuses otherwise unless --force is passed (which must then be justified in the
journal entry that ships it).

⚠ Members land in data/ext_members10/, NEVER in oof/. `load_members` scans oof/ by default,
so saving there moves the pack under blend_lab, w26f, w26h, w26i and w26j at once, and every
reproduction gate in this repo is stated against a fixed member COUNT (w26m's warning).

    .venv/bin/python experiments/w31b_export.py --tag leaves255_dinf --stage 2000 \
        --scale 1.3333 --name lat_l255_ctfix_r2000
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import DATA, TARGET, load_raw, save_preds  # noqa: E402
sys.path.insert(0, HERE)
from w31a_lgb5f import SCALES, STAGES  # noqa: E402

OUT = os.path.join(DATA, "ext_members10")


def cell(tag, i, j):
    oof = np.load(os.path.join(HERE, f"w31a_{tag}_oof.npy"))
    test = np.load(os.path.join(HERE, f"w31a_{tag}_test.npy"))
    return oof[:, i, j].astype("float64"), test[:, i, j].astype("float64")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    i = STAGES.index(a.stage)
    j = int(np.argmin([abs(s - a.scale) for s in SCALES]))
    if abs(SCALES[j] - a.scale) > 1e-3:
        raise SystemExit(f"scale {a.scale} is not one of {SCALES}")

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    o, t = cell(a.tag, i, j)
    cv = roc_auc_score(y, o)
    co, _ = cell("control", i, j)
    ccv = roc_auc_score(y, co)
    print(f"{a.tag} stage {a.stage} scale {SCALES[j]:.4f}: pooled OOF {cv:.10f}\n"
          f"  control, same cell:                 {ccv:.10f}   "
          f"delta {(cv-ccv)*1e6:+.2f}e-6")
    if cv <= ccv and not a.force:
        raise SystemExit("R-N3: this cell does not beat the control at the same stage and "
                         "scale. Not exported. Pass --force only with a journal argument.")
    assert o.shape == (len(y),) and t.shape == (len(te),), "row counts do not match"
    assert np.isfinite(o).all() and np.isfinite(t).all(), "non-finite predictions"
    os.makedirs(a.out, exist_ok=True)
    save_preds(a.name, o, t, len(y), len(te), out=a.out)
    print(f"  wrote {a.name} to {a.out}")


if __name__ == "__main__":
    main()
