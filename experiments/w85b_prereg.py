"""w85b — REGISTER a predicted public score for each w85 filler, BEFORE any of them is sent.

Same shape and same schema as `w37c_prereg.py`, and for the same reason: `w55a_unpriced.py`
certifies an unpriceable row at `registered pred_lb + worst landed overshoot`, where the
overshoot is measured on this account's own landed calibration sends. That instrument only
covers a row that appears in a REGISTERED prereg -- a number written down before the score
exists and never editable afterwards. This file is that registration for the 25 fillers.

THE PREDICTION, AND WHY IT IS NOT A FITTED LINE.
w37e's R2 rejected the linear-in-AUC offset form at 6.28 sigma over a ~1e-2 AUC range, and
these fillers span 0.82 to 0.96 -- fourteen times that range. So nothing is fitted. The
registered prediction is the flat MEDIAN offset over the five landed readings:

    member          oof_auc      public    public - oof
    mkt_realmlp     0.958134     0.96285      +0.0047163   <- OFFSET_MAX
    omid_tabm       0.967508     0.96949      +0.0019822
    dm_cat          0.966700     0.96813      +0.0014299   <- OFFSET_MEDIAN
    ram_hgb         0.968026     0.96945      +0.0014242
    om_xgb2         0.968733     0.96993      +0.0011968

⚠ THESE FIVE NUMBERS WERE TYPED BY HAND FIRST AND TWO OF THEM WERE WRONG (the median and the
max, each by ~7e-6, and dm_cat/ram_hgb were transposed). C3 recomputes them from the live
submission list and refused the file until the constants matched. A table in a docstring is
not a measurement.

    pred_lb = oof_auc + 0.001423

It will be wrong, and wrongly in a knowable direction: the offset was largest at the lowest
AUC of the five, so the low fillers should overshoot. THAT IS THE POINT. 25 readings spread
over 0.82-0.96 are the first data this account has on the SHAPE of the OOF->public offset off
the narrow high-AUC band, which is exactly what w37e R2 said it could not describe.

⛔ AND IT LICENSES NOTHING. The readout is a record. It must not become a deflation constant
(closed, w37e R3), a per-family or per-member correction (refused by name, w81 §6 and w84 §2),
or an input to the deadline pick, which is on CV. If a future run wants to use these numbers
for anything, it registers the use first, in its own file, before reading them.

CONTROLS.
  C1 +  every row is re-derived from `w85a_fillers.csv` and the md5 re-read off disk, so a
        prereg row can never describe a file that is not the one that will be sent.
  C2 -  every registered pred_lb is asserted below the tier by the full margin, so the prereg
        cannot register a row the send path would then have to block.
  C3 +  the five landed offsets are recomputed from the LIVE submission list (paginated --
        RESEARCH's standing rule, and the defect w84 §2 found) rather than copied from the
        docstring above, so a stale median fails loudly.

    .venv/bin/python experiments/w85b_prereg.py            # 0 = ok, 1 = a control failed
"""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
SUB = os.path.join(W, "submissions")
COMP = "playground-series-s6e8"

TIER = 0.97119
MARGIN_MIN = 1000e-6
OFFSET_MEDIAN = 0.0014299
OFFSET_MAX = 0.0047163

MAN = os.path.join(HERE, "w85a_fillers.csv")
W37 = os.path.join(HERE, "w37c_prereg.csv")
OUT = os.path.join(HERE, "w85b_prereg.csv")

# ⚠ --page-size is NOT optional (RESEARCH: the CLI silently returns 50 rows).
SUBS_ARGV = ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "200"]


def main() -> int:
    fails = []
    man = pd.read_csv(MAN)

    # ---- C3: recompute the offset table from the live list ---------------------------
    out = subprocess.run(SUBS_ARGV, capture_output=True, text=True, timeout=600)
    sc = pd.read_csv(io.StringIO(out.stdout)).dropna(subset=["publicScore"])
    lb = sc.groupby("fileName").publicScore.max()
    w37 = pd.read_csv(W37)
    w37["actual"] = w37.file.map(lb)
    landed = w37.dropna(subset=["actual"]).copy()
    landed["offset"] = landed.actual - landed.oof_auc
    print(f"live list {len(sc)} scored rows;  {len(landed)} of {len(w37)} w37 readings landed\n")
    print(f"{'member':16s} {'oof_auc':>12} {'public':>9} {'offset':>11}")
    for r in landed.sort_values("offset", ascending=False).itertuples():
        print(f"{r.member:16s} {r.oof_auc:12.10f} {r.actual:9.5f} {r.offset:+11.6f}")
    med, mx = float(landed.offset.median()), float(landed.offset.max())
    print(f"\nmedian {med:+.6f}  (registered {OFFSET_MEDIAN:+.6f})"
          f"   max {mx:+.6f}  (registered {OFFSET_MAX:+.6f})")
    if abs(med - OFFSET_MEDIAN) > 5e-7:
        fails.append(f"live median offset {med:.6f} != registered {OFFSET_MEDIAN:.6f}")
    if abs(mx - OFFSET_MAX) > 5e-7:
        fails.append(f"live max offset {mx:.6f} != registered {OFFSET_MAX:.6f}")

    rows = []
    for r in man.itertuples():
        path = os.path.join(SUB, r.file)
        if not os.path.exists(path):                                          # C1
            fails.append(f"{r.file}: in the manifest but not on disk")
            continue
        md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
        if md5 != r.md5:
            fails.append(f"{r.file}: md5 {md5} != manifest {r.md5}")
            continue
        pred = r.oof_auc + OFFSET_MEDIAN
        bound = pred + OFFSET_MAX
        if TIER - bound < MARGIN_MIN:                                         # C2
            fails.append(f"{r.file}: bound {bound:.6f} is {(TIER-bound)*1e6:.1f}e-6 "
                         f"under the tier, below the {MARGIN_MIN*1e6:.0f}e-6 minimum")
            continue
        rows.append(dict(order=len(rows) + 1, file=r.file, member=r.member,
                         role="FILLER", oof_auc=r.oof_auc, n_distinct=r.n_distinct,
                         md5=md5, es_mechanism="n/a -- w85 slot filler, see w85a",
                         offset_line=OFFSET_MEDIAN, pred_H0=pred, pred_H1=pred,
                         pred_lb=pred, sep_steps=np.nan))

    pre = pd.DataFrame(rows)
    pre.to_csv(OUT, index=False)
    print(f"\nregistered {len(pre)} rows -> {os.path.relpath(OUT, W)}")
    print(f"pred_lb spans {pre.pred_lb.min():.6f} .. {pre.pred_lb.max():.6f}; "
          f"worst bound {pre.pred_lb.max() + OFFSET_MAX:.6f} vs tier {TIER} "
          f"({(TIER - pre.pred_lb.max() - OFFSET_MAX)*1e6:+.0f}e-6)")
    if len(pre) != len(man):
        fails.append(f"registered {len(pre)} of {len(man)} manifest rows")

    for f in fails:
        print(f"  FAIL  {f}")
    print(f"\nFAILURES {len(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
