"""w15b step 5b -- pin the MECHANISM behind the lattice under-dispersion.

w15b_lackfit.py found that inside real (social, daily) lattice cells the pack's residuals
are ~2x LESS variable than independent Bernoulli sampling permits (T/df 0.50-0.67 against a
size-matched permuted control sitting at exactly 1.000, z = -10). I attributed that to
full-resolution target encoding: a row's OOF score is built from the 4 training folds, which
contain the OTHER rows of its own lattice cell, so E_c tracks the realised O_c.

That is an explanation, not a measurement. The control it demands is obvious and cheap:

    members that use lattice TARGET ENCODING should show T/df << 1.
    members fitted on RAW features only should show T/df ~ 1.

If both families come out the same, my explanation is wrong and the under-dispersion means
something else. The workspace's own naming convention makes the split available for free --
`*_lat*` members carry the lattice/TE recipe, `*_raw*` and `*_native` do not.

Each member is calibrated with the same cross-fitted binned-PAVA used everywhere in w15b
(B=100) so that E_c and V_c are on a probability scale and the statistic is comparable
across members. Every member also gets its OWN size-matched permuted control, because a
weaker member has a different score distribution and must not be judged against the pack's
null.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DAILY, OOF, SUB, TARGET, get_folds, load_raw  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15b_calib import binned_calibrator  # noqa: E402
from w15b_lackfit import cell_key, zstats  # noqa: E402

SOCIAL = "social_media_hours"
NPERM = 12
SEED = 20260815

# (label, path, does the recipe use full-resolution lattice target encoding?)
MEMBERS = [
    ("PACK blend159av_h3", os.path.join(SUB, "oof_blend159av_h3.npy"), "stack"),
    ("xgb_latcat", os.path.join(OOF, "oof_xgb_latcat.npy"), "TE"),
    ("xgb_lat", os.path.join(OOF, "oof_xgb_lat.npy"), "TE"),
    ("lgbm_tuned_lat", os.path.join(OOF, "oof_lgbm_tuned_lat.npy"), "TE"),
    ("cat_lat", os.path.join(OOF, "oof_cat_lat.npy"), "TE"),
    ("xgb_raw_nan", os.path.join(OOF, "oof_xgb_raw_nan.npy"), "raw"),
    ("cat_raw", os.path.join(OOF, "oof_cat_raw.npy"), "raw"),
    ("cat_native", os.path.join(OOF, "oof_cat_native.npy"), "raw"),
    ("orig_binm", os.path.join(OOF, "oof_orig_binm.npy"), "raw (orig-trained)"),
]


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    n = len(y)
    key = cell_key(tr, [SOCIAL, DAILY])
    keyd = cell_key(tr, [DAILY])
    rng = np.random.default_rng(SEED)

    print(f"{'member':24s} {'recipe':18s} {'solo AUC':>9}  "
          f"{'T/df sd (n>=20)':>16}  {'T/df daily (n>=20)':>19}  {'perm T/df':>10}")
    rows = []
    for name, path, recipe in MEMBERS:
        if not os.path.exists(path):
            print(f"  (missing {path})")
            continue
        raw = np.load(path)
        p = np.empty(n)
        for tr_i, va_i in folds:
            f = binned_calibrator(raw[tr_i], y[tr_i], 100)
            p[va_i] = f(raw[va_i])
        p = np.clip(p, 1e-6, 1 - 1e-6)

        out = {}
        for tag, k in (("sd", key), ("daily", keyd)):
            T, df, _ = zstats(k, y, p, 20)
            out[tag] = T / df
        # own size-matched permuted control on the social+daily lattice
        pt = []
        for _ in range(NPERM):
            pk = key[rng.permutation(n)]
            Tp, dfp, _ = zstats(pk, y, p, 20)
            pt.append(Tp / dfp)
        pm = float(np.mean(pt))
        a = roc_auc_score(y, raw)
        rows.append(dict(member=name, recipe=recipe, auc=float(a),
                         tdf_sd=out["sd"], tdf_daily=out["daily"], tdf_perm=pm))
        print(f"{name:24s} {recipe:18s} {a:9.6f}  {out['sd']:16.4f}  "
              f"{out['daily']:19.4f}  {pm:10.4f}", flush=True)

    df = pd.DataFrame(rows)
    te = df[df.recipe == "TE"]
    rawm = df[df.recipe.str.startswith("raw")]
    print("\n=== verdict ===")
    if len(te) and len(rawm):
        print(f"  TE members   : T/df(daily) mean {te.tdf_daily.mean():.4f}  "
              f"range {te.tdf_daily.min():.4f}-{te.tdf_daily.max():.4f}")
        print(f"  raw members  : T/df(daily) mean {rawm.tdf_daily.mean():.4f}  "
              f"range {rawm.tdf_daily.min():.4f}-{rawm.tdf_daily.max():.4f}")
        print(f"  permuted null: {df.tdf_perm.mean():.4f} (must be ~1.000 for all)")
        if te.tdf_daily.max() < rawm.tdf_daily.min():
            print("  => CLEAN SEPARATION. The under-dispersion is caused by target")
            print("     encoding, exactly as claimed. Raw-feature members do not show it.")
        else:
            print("  => NO clean separation; the TE explanation does not survive.")
    df.to_csv(os.path.join(ROOT, "experiments", "w15b_tecause.csv"), index=False)
    with open(os.path.join(ROOT, "experiments", "w15b_tecause.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote experiments/w15b_tecause.csv")


if __name__ == "__main__":
    main()
