"""Profile najiama's 18_blend, the only new public artefact since the last pool sweep.

The dataset was re-versioned 2026-08-12; every other file byte-matches what we already
hold, so 18_blend is the single new object. najiama's 01-05 are already in the 74-model
library as naji01..naji05 and are its two best members (naji03/naji05, OOF 0.9688), so a
further blend of them is the author's own combiner output rather than a new base model.
This measures that rather than assuming it: solo AUC, and maximum correlation against the
pack in the space the stacker actually sees.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import DEFAULT_DROP, load_members, to_logit, transform  # noqa: E402

NJM = os.path.join(DATA, "ext", "najiama_predicting-smartphone-addiction-oof-submission-csv")


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    oo = pd.read_csv(os.path.join(NJM, "18_blend_oof_predictions.csv"))
    tt = pd.read_csv(os.path.join(NJM, "18_blend_submission.csv"))
    oc = [c for c in oo.columns if c != "id"][-1]
    tc = [c for c in tt.columns if c != "id"][-1]

    # row alignment is the one thing that can silently poison a stack: the library's
    # contract is original row order on the frozen folds, so check the ids literally.
    assert (oo["id"].to_numpy() == tr["id"].to_numpy()).all(), "oof ids misaligned"
    assert (tt["id"].to_numpy() == te["id"].to_numpy()).all(), "test ids misaligned"

    o, t = oo[oc].to_numpy(np.float64), tt[tc].to_numpy(np.float64)
    print(f"naji18  solo OOF AUC {roc_auc_score(y, o):.6f}")
    print(f"        oof range [{o.min():.6g}, {o.max():.6g}]  "
          f"test range [{t.min():.6g}, {t.max():.6g}]")

    names, O, T = load_members(
        y, len(te),
        extra_dirs=(os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2")),
        drop=set(DEFAULT_DROP))
    print(f"\npack: {len(names)} members")

    for kind in ("hybrid", "rankraw"):
        Z, _ = transform(np.column_stack([O, o]), np.column_stack([T, t]), kind)
        Z = (Z - Z.mean(0)) / Z.std(0)
        c = np.abs(Z[:, :-1].T @ Z[:, -1]) / len(y)
        order = np.argsort(-c)[:5]
        print(f"\n[{kind}] maxcorr {c.max():.4f} against {len(names)} members")
        for i in order:
            print(f"    {c[i]:.4f}  {names[i]}")

    solos = np.array([roc_auc_score(y, O[:, i]) for i in range(len(names))])
    j = int(np.argmax(solos))
    print(f"\npack best solo: {names[j]} {solos[j]:.6f}   "
          f"median {np.median(solos):.6f}   n above naji18: "
          f"{int((solos > roc_auc_score(y, o)).sum())}")

    out = os.path.join(DATA, "ext_members2")
    np.save(os.path.join(out, "oof_naji18.npy"), o)
    np.save(os.path.join(out, "test_naji18.npy"), t)
    print(f"\nsaved as member naji18 in {out}")


if __name__ == "__main__":
    main()
