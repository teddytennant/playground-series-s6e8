"""w38c -- gate 4 (foreign-partition test) and the import for the `zhukovoleksiy` members.

w38b's cheap gates passed three streams from `zhukovoleksiy/ps6e8-eda-feature-engineering-
pipeline`, and the source read says they are clean:

  * `train_cv_lattice` fits with a bare `m.fit(X_tr, y_tr)` -- NO eval_set is ever passed on
    the lattice path, so early-stopping-on-validation is structurally impossible. This is the
    first source in six sweeps where the clearance is by ABSENCE of the mechanism rather than
    by reading a log for whether it fired.
  * the shipped `oof/manifest.csv` declares StratifiedKFold(5, shuffle=True, random_state=42),
    which is our frozen partition.
  * test predictions are the mean over the five fold models, the honest construction.

A declaration is not a measurement. This script is the measurement: recompute each stream's
per-fold AUC under OUR get_folds() and compare against the author's own printed per-fold
numbers. If the partition is ours, every one of the 15 cells matches to the log's 5-dp
printing floor. If it is foreign, the fold AUCs are shuffled and the max abs diff blows up.
w34 3.1's rule -- a foreign partition is worse than es-on-val -- is what makes this the
load-bearing gate.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_D = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT_D, "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402

SRC = os.path.join(ROOT_D, "notebooks", "w38", "out",
                   "zhukovoleksiy_ps6e8-eda-feature-engineering-pipeline", "oof")
OUT = os.path.join(DATA, "ext_members14")
N_TR, N_TE = 691369, 296302

# transcribed BY HAND from the kernel log, in the author's fold order
LOG = {
    "lexb_xgb_base": ([0.96730, 0.96798, 0.96808, 0.96857, 0.96765], 0.967914126742773),
    "lexb_cat_base": ([0.96728, 0.96823, 0.96818, 0.96851, 0.96774], 0.9679867951811431),
    "lexb_lgb02":    ([0.96766, 0.96849, 0.96861, 0.96894, 0.96814], 0.9683696067936562),
}
FILES = {"lexb_xgb_base": "xgb_base", "lexb_cat_base": "cat_base", "lexb_lgb02": "lgb02"}


def main():
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    assert len(y) == N_TR and len(te) == N_TE
    folds = list(get_folds(y))

    rows = []
    for nm, tag in sorted(FILES.items()):
        o = np.load(os.path.join(SRC, f"oof_lexB_{tag}.npy")).ravel().astype(np.float64)
        t = np.load(os.path.join(SRC, f"test_lexB_{tag}.npy")).ravel().astype(np.float64)
        assert o.shape == (N_TR,) and t.shape == (N_TE,), f"{nm}: {o.shape} {t.shape}"
        assert np.isfinite(o).all() and np.isfinite(t).all(), f"{nm}: non-finite"

        ours = [roc_auc_score(y[va], o[va]) for _, va in folds]
        log_folds, log_overall = LOG[nm]
        d = max(abs(a - b) for a, b in zip(ours, log_folds))
        overall = roc_auc_score(y, o)
        rows.append(dict(member=nm, solo_auc=overall, log_overall=log_overall,
                         overall_diff=abs(overall - log_overall),
                         g4_maxdiff=d, g4_ok=bool(d < 1e-4),
                         sd_ratio=float(t.std() / o.std())))
        print(f"{nm:16s} solo {overall:.9f} (manifest {log_overall:.9f}, "
              f"d {abs(overall - log_overall):.2e})")
        print(f"   ours [{' '.join(f'{x:.5f}' for x in ours)}]")
        print(f"   log  [{' '.join(f'{x:.5f}' for x in log_folds)}]   max|d| {d:.2e}  "
              f"{'OUR FOLDS' if d < 1e-4 else '*** FOREIGN PARTITION ***'}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "w38c_lexvet.csv"), index=False)
    print("\n" + df.to_string(index=False))

    if bool(df["g4_ok"].all()):
        os.makedirs(OUT, exist_ok=True)
        for nm, tag in sorted(FILES.items()):
            np.save(os.path.join(OUT, f"oof_{nm}.npy"),
                    np.load(os.path.join(SRC, f"oof_lexB_{tag}.npy")).ravel().astype(np.float64))
            np.save(os.path.join(OUT, f"test_{nm}.npy"),
                    np.load(os.path.join(SRC, f"test_lexB_{tag}.npy")).ravel().astype(np.float64))
        print(f"\nwrote {len(FILES)} members to {OUT}")
    else:
        print("\ngate 4 FAILED -- nothing written")


if __name__ == "__main__":
    main()
