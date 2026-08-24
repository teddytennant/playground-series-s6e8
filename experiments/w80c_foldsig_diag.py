"""w80c -- POST-HOC diagnosis of w80b's INDETERMINATE. ⛔ NOT REGISTERED. LICENSES NOTHING.

w80b ran the registered fold-signature test and returned INDETERMINATE: the pack's median
S = 2.736 sits strictly between max(NEG) = 1.966 and min(POS) = 46.913. Per w80_prereg2 that
licenses NOTHING, and this file does not and cannot change that. The verdict is not amended.

What this file asks is a different question, and it is the question a median hides: the 50
per-column S values are not one population. Three columns sit at S = 0.00, 0.32, 0.35 --
BELOW the chi2(4) null mean of 4 -- while others sit at 66, 260, 486, squarely in POS
territory. A value far below the null is not evidence of a foreign partition; nothing about a
foreign partition pushes S toward zero. It is the fingerprint of a column whose per-fold
means have been CENTRED, which would delete the very signature this instrument reads.

So: run each pack column under BOTH partitions. That separates the two explanations the
median conflates.

    S_ours >> S_foreign ~ 4     -> that column carries OUR signature. Matched.
    S_ours ~ S_foreign ~ 4      -> no signature either way. Foreign, or scrubbed.
    S_ours ~ S_foreign << 4     -> SCRUBBED. The instrument is BLIND on that column and its
                                   contribution to the median is not evidence of anything.

⛔ Nothing in w80c_foldsig_diag.json may be quoted as a registered finding. It is a
diagnosis of an instrument, not a verdict on a pack.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import common  # noqa: E402
from w80b_foldsig import fold_signature, read_prereg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACK = os.path.join(ROOT, "data", "ext_members17")
OUT = os.path.join(ROOT, "experiments", "w80c_foldsig_diag.json")
N_TR = 691369
CHI2_4_MEAN = 4.0
# chi2(4) quantiles, for reading a single S against the null it is supposed to follow.
CHI2_4_Q05, CHI2_4_Q95 = 0.7107, 9.4877


def main() -> int:
    reg = read_prereg()
    print("=" * 78)
    print("w80c -- POST-HOC diagnosis of w80b's INDETERMINATE")
    print("⛔ NOT REGISTERED. The w80b verdict stands at INDETERMINATE and licenses NOTHING.")
    print("=" * 78)

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[common.TARGET])
    y = tr[common.TARGET].values
    folds_ours = common.get_folds(y)
    folds_neg = list(StratifiedKFold(n_splits=reg["neg_k"], shuffle=True,
                                     random_state=reg["neg_seed"]).split(np.zeros(y.size), y))

    oof = np.load(os.path.join(PACK, "oof.npy"), mmap_mode="r")
    members = pd.read_csv(os.path.join(PACK, "members.csv"))
    auc_col = [c for c in members.columns if "auc" in c.lower()]
    solo = members[auc_col[0]].values if auc_col else np.full(reg["n_col"], np.nan)

    rows = []
    for j in range(reg["n_col"]):
        c = np.asarray(oof[:, j], dtype=np.float64)
        s_o = fold_signature(c, folds_ours)
        s_f = fold_signature(c, folds_neg)
        if s_o < CHI2_4_Q05 and s_f < CHI2_4_Q05:
            cls = "SCRUBBED"
        elif s_o > CHI2_4_Q95 and s_o > 5.0 * max(s_f, 1e-9):
            cls = "OURS"
        else:
            cls = "NOSIG"
        rows.append({"col": str(members.iloc[j, 0]), "solo_auc": float(solo[j]),
                     "S_ours": s_o, "S_foreign": s_f, "class": cls})

    df = pd.DataFrame(rows).sort_values("S_ours", ascending=False)
    print(f"\n  chi2(4) null: mean {CHI2_4_MEAN}, 5% {CHI2_4_Q05}, 95% {CHI2_4_Q95}")
    print(f"\n  {'col':>6s} {'solo_auc':>9s} {'S_ours':>10s} {'S_foreign':>10s}  class")
    for _, r in df.iterrows():
        print(f"  {r['col']:>6s} {r['solo_auc']:9.6f} {r['S_ours']:10.3f} "
              f"{r['S_foreign']:10.3f}  {r['class']}")

    counts = df["class"].value_counts().to_dict()
    print(f"\n  classes: {counts}")

    scrub = df[df["class"] == "SCRUBBED"]
    ours = df[df["class"] == "OURS"]
    nosig = df[df["class"] == "NOSIG"]

    print("\n" + "-" * 78)
    print("  WHAT THE SPLIT SAYS -- and what it does NOT")
    print(f"    {len(ours):2d} columns carry OUR fold signature (S_ours >> S_foreign).")
    print(f"    {len(scrub):2d} columns are SCRUBBED: BOTH S values sit below the chi2(4) 5%")
    print("       point. No partition makes S small; only per-fold centring does. On these")
    print("       columns the instrument is BLIND, not negative.")
    print(f"    {len(nosig):2d} columns show no signature under either partition.")
    print("\n    ⚠ A blind column still votes in w80b's median. That is why the median landed")
    print("      at 2.736 -- in the gap -- rather than on either side. INDETERMINATE is the")
    print("      HONEST reading of a statistic computed over a mixed population, and this")
    print("      diagnosis explains it; it does not overturn it.")
    print("\n    ⛔ AND IT CANNOT BE FIXED BY DROPPING THE BLIND COLUMNS AND RE-READING.")
    print("      The prereg fixed t = median over ALL 50. Re-reading a subset chosen AFTER")
    print("      seeing the values is the exact move the prereg exists to forbid. If a later")
    print("      run wants a per-column read it must REGISTER one -- and note in advance that")
    print("      a scrubbed column is UNTESTABLE by this instrument, so no registration can")
    print("      rescue those columns at all.")

    res = {
        "script": "w80c_foldsig_diag.py",
        "REGISTERED": False,
        "note": ("POST-HOC. w80b's registered verdict is INDETERMINATE and is not amended. "
                 "Nothing here may be quoted as a registered finding or license any adoption."),
        "chi2_4": {"mean": CHI2_4_MEAN, "q05": CHI2_4_Q05, "q95": CHI2_4_Q95},
        "counts": counts,
        "per_col": rows,
        "summary": {
            "n_ours": int(len(ours)),
            "n_scrubbed": int(len(scrub)),
            "n_nosig": int(len(nosig)),
            "median_S_ours_all": float(df["S_ours"].median()),
            "median_S_foreign_all": float(df["S_foreign"].median()),
        },
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
