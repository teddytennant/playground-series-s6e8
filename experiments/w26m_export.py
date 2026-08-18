"""w26m -- turn one arm of a w26k/w26l run into a stackable member.

The arm arrays come out of w26k/w26l as (n_rows, n_arms). This slices one arm and writes it
in the pack's own convention (float64, original row order, oof_/test_ pair).

⚠ Members land in data/ext_members6/, NEVER in oof/. `load_members` scans oof/ by default, so
saving there moves the pack under blend_lab, w26f, w26h, w26i and w26j at once, and every
reproduction gate in this repo is stated against a fixed member COUNT.

    .venv/bin/python experiments/w26m_export.py --src experiments/w26l_r400 --arm both \
        --name lgbm_lat_ctfix
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
from common import DATA, TARGET, load_raw, save_preds  # noqa: E402
sys.path.insert(0, HERE)
from w26l_serve import ARMS as L_ARMS  # noqa: E402
from w26k_ctscale import SCALES  # noqa: E402

K_ARMS = [f"ct{s:.4f}" for s in SCALES]
OUT = os.path.join(DATA, "ext_members6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="path stem, e.g. experiments/w26l_r400")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    oof = np.load(a.src + "_oof.npy")
    test = np.load(a.src + "_test.npy")
    arms = L_ARMS if oof.shape[1] == len(L_ARMS) else K_ARMS
    if oof.shape[1] != len(arms):
        raise SystemExit(f"{oof.shape[1]} arms in {a.src}, matches neither w26k nor w26l")
    if a.arm not in arms:
        raise SystemExit(f"arm {a.arm!r} not in {arms}")
    j = arms.index(a.arm)

    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    os.makedirs(a.out, exist_ok=True)
    from sklearn.metrics import roc_auc_score
    print(f"{a.src} arms={arms}\n  arm {a.arm} (col {j}) OOF AUC "
          f"{roc_auc_score(y, oof[:, j]):.10f}")
    save_preds(a.name, oof[:, j], test[:, j], len(y), len(te), out=a.out)


if __name__ == "__main__":
    main()
