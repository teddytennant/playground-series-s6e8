"""Why the original-trained member transfers at only 0.886: it is off-manifold.

Two hard constraints separate the two frames (see `w15d_screen_source.py`):

  * COMPETITION: `daily >= social + gaming + work_study`, 0 violations in 421,427 train
    and 182,287 test complete rows, min slack exactly 0.00.  The ORIGINAL violates it in
    60.7% of its 7,500 rows.
  * ORIGINAL: `weekend - daily` in [0.50, 3.00] in 100.0% of rows (min 0.50, max 3.00
    exactly). The COMPETITION satisfies that in 52.0% of train / 52.1% of test.

So `orig_binm` -- LightGBM fitted on the original alone -- is asked to score a frame whose
joint distribution it has never seen. This prices that: its AUC on competition rows inside
vs outside the original's own support, against the pack measured on the same rows.

    .venv/bin/python experiments/w15d_support.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, OOF, SUB, TARGET  # noqa: E402

ORIG = os.path.join(DATA, "orig", "Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
BOX = ["daily_screen_time_hours", "social_media_hours", "gaming_hours",
       "work_study_hours", "sleep_hours", "weekend_screen_time",
       "notifications_per_day", "app_opens_per_day", "age"]


def main():
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    o = pd.read_csv(ORIG)
    y = tr[TARGET].astype(int).to_numpy()
    m = np.load(os.path.join(OOF, "oof_orig_binm.npy"))
    p = np.load(os.path.join(SUB, "oof_blend159av_h3.npy"))

    lo = {c: o[c].min() for c in BOX}
    hi = {c: o[c].max() for c in BOX}

    # a row is "in support" if every OBSERVED value sits inside the original's range.
    inbox = np.ones(len(tr), dtype=bool)
    for c in BOX:
        v = tr[c].to_numpy(dtype=float)
        bad = np.isfinite(v) & ((v < lo[c] - 1e-9) | (v > hi[c] + 1e-9))
        inbox &= ~bad
        print(f"  {c:26s} orig range [{lo[c]:6.2f}, {hi[c]:6.2f}]   competition rows outside: "
              f"{bad.mean()*100:5.2f}%")

    # the original's own weekend-daily construction, applied to competition rows
    wd = (tr["weekend_screen_time"] - tr["daily_screen_time_hours"]).to_numpy(dtype=float)
    wd_ok = ~np.isfinite(wd) | ((wd >= 0.5 - 1e-9) & (wd <= 3.0 + 1e-9))

    print("\n=== AUC by whether the competition row is inside the ORIGINAL's support ===")
    print(f"{'subset':44s} {'n':>9s} {'rate':>7s} {'orig_binm':>10s} {'pack h3':>9s}")
    subsets = [
        ("all rows", np.ones(len(tr), bool)),
        ("inside every marginal range", inbox),
        ("outside at least one marginal range", ~inbox),
        ("weekend-daily inside orig's [0.5,3.0]", wd_ok),
        ("weekend-daily outside orig's [0.5,3.0]", ~wd_ok),
        ("inside box AND weekend-daily in band", inbox & wd_ok),
        ("outside box OR weekend-daily out of band", ~(inbox & wd_ok)),
    ]
    rows = []
    for name, sel in subsets:
        if sel.sum() < 100 or len(np.unique(y[sel])) < 2:
            continue
        a_m, a_p = roc_auc_score(y[sel], m[sel]), roc_auc_score(y[sel], p[sel])
        print(f"{name:44s} {sel.sum():9,d} {y[sel].mean():7.4f} {a_m:10.6f} {a_p:9.6f}")
        rows.append(dict(subset=name, n=int(sel.sum()), rate=float(y[sel].mean()),
                         orig_binm=a_m, pack_h3=a_p))

    out = os.path.join("experiments", "w15d_support.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
