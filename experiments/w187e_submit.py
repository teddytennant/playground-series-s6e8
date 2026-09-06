"""Turn an oof_w187/test_*.npy into a submission CSV in the workspace's usual layout."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agent"))
from common import DATA, SUB, TARGET  # noqa: E402

name = sys.argv[1]
p = np.load(os.path.join(HERE, "..", "oof_w187", f"test_{name}.npy"))
sub = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
assert len(sub) == len(p), (len(sub), len(p))
assert np.isfinite(p).all()
sub[TARGET] = p
out = os.path.join(SUB, f"{name}.csv")
sub.to_csv(out, index=False)
print(f"{out}  n={len(sub)}  min={p.min():.6f} max={p.max():.6f} mean={p.mean():.6f}")
