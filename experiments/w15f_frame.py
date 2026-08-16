"""w15f stage 1: the target-free feature frame the transductive teacher/student runs on.

WHY A NEW FRAME RATHER THAN cache/f*_Xa.npy
-------------------------------------------
The cached matrices carry fold-fitted TARGET encoding. The object being rebuilt here is
raykkretzschmar's teacher/student contrast, and its whole point is that the STUDENT is
fitted on the teacher's ranks while carrying the unlabeled outer-validation rows. Any
target-derived column would put outer-fold label information into that fit through the
back door and the measurement would be worthless.

So every column here is a function of the raw values alone, computed once on train+test
together. Frequency of an exact lattice value is target-free (it is a property of the
feature marginal, not of y), which is what makes it legitimate transductive preprocessing
in the same sense as the constrained imputation already in `agent/features.py`.

The author's description of the teacher is "a raw-feature LightGBM with exact-value
frequency features, missing indicators and a few generator identities". That maps onto:

  * `make_frames`'s X block -- raw NUM/CAT, the constrained-imputation reconstruction,
    the `*_was_missing` indicators, and the generator identity terms
    (`other_screen_imp`, `comp_share_imp`, `identity_violation`, `n_screen_missing`).
  * CNT_<col> -- how many of the 987,671 rows share this row's exact value in that column.
  * CNT_P_<a>__<b> -- the same at four joint lattice cells.

Writes experiments/w15f_X.npy (float32, n_train+n_test rows) and w15f_cols.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import CAT, DAILY, NUM, TARGET, load_raw  # noqa: E402
from features import make_frames  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(HERE, "w15f_X.npy")
COLS_PATH = os.path.join(HERE, "w15f_cols.json")

PAIRS = [(DAILY, "weekend_screen_time"),
         ("social_media_hours", "gaming_hours"),
         (DAILY, "social_media_hours"),
         ("notifications_per_day", "app_opens_per_day")]


def main():
    t0 = time.time()
    tr, te = load_raw()
    ntr, nte = len(tr), len(te)
    print(f"train {ntr:,}  test {nte:,}  ({time.time()-t0:.0f}s)", flush=True)

    # X is target-free by construction; the lattice keys it also returns are only used
    # here to spell exact values consistently (NaN -> its own level).
    Xtr, Xte, _, _ = make_frames(tr, te, wide_pairs=False, triples=False)
    X = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    print(f"base block {X.shape}  ({time.time()-t0:.0f}s)", flush=True)

    full = pd.concat([tr.drop(columns=[TARGET]), te], axis=0, ignore_index=True)

    # --- exact-value frequency, over train+test together (target-free) ---
    lev = {}
    for c in NUM + CAT:
        s = full[c].astype(str)          # NaN becomes the literal 'nan' level, which is
        lev[c] = s                       # exactly how the tree should see missingness here
        X[f"CNT_{c}"] = s.map(s.value_counts()).astype("float32")
    for a, b in PAIRS:
        s = lev[a] + "__" + lev[b]
        X[f"CNT_P_{a}__{b}"] = s.map(s.value_counts()).astype("float32")

    X = X.replace([np.inf, -np.inf], np.nan)
    arr = X.to_numpy("float32")
    assert arr.shape[0] == ntr + nte
    np.save(X_PATH, arr)
    json.dump(list(X.columns), open(COLS_PATH, "w"))
    print(f"wrote {X_PATH}  {arr.shape}  "
          f"nan cols {int(np.isnan(arr).any(0).sum())}  ({time.time()-t0:.0f}s)")
    print("columns:", ", ".join(X.columns))


if __name__ == "__main__":
    main()
