"""w80d -- POST-HOC. What S does a SINGLE model show under a matched partition?
⛔ NOT REGISTERED. It does not re-read the pack and it does not amend w80b's INDETERMINATE.

WHY THIS EXISTS -- a defect in w80_prereg2's CONTROL SELECTION, found by looking at the
controls rather than at the verdict.

prereg2 chose POS as "the 20 lexicographically first stems in submissions/oof_*.npy". That
rule is reproducible, which is what it was chosen for, and it is NOT a sample. The 20 it
selects are:

    blend150fx{,_hybrid,_logit,_rankraw,_rescale}   blend150sx{...}
    blend153{...}                                   blend156{...}

Four blend bases x five transforms. Every one is a blend of 150+ members. Not one single
model. The pack under test is fifty SINGLE weak models (solo AUC 0.917..0.957).

That matters because Q3's MATCHED branch is `t >= min(POS)`, so the bar for calling the pack
matched was set by 150-member blends. A blend averages 150 per-fold biases that all point the
same way within a fold, so its fold signature is inflated relative to any one member. If a
single matched model shows S of order 10 and a blend shows S of order 100, then min(POS) was
never a reachable bar for this pack and INDETERMINATE was close to determined the moment the
control rule was written -- before any pack data was read.

⚠ Q1's control AUC of 1.0000 does not protect against this. Q1 asks whether the instrument
can tell a matched partition from a foreign one. It can, overwhelmingly. It says nothing
about the MAGNITUDE a single weak model should reach, and Q3's bar is a magnitude.
(w80 §7 wrote this same lesson about P2: check what a passing gate actually tests.)

THE CONTROL prereg2 SHOULD HAVE USED
------------------------------------
oof/oof_*.npy -- this workspace's own 20 INDIVIDUAL models (lgbm/xgb/cat/et/linear variants),
every one built by agent/ under common.get_folds. Known matched, and single models.

⛔ ALSO CORRECTS w80c. w80c classified pack columns against a chi2(4) null (mean 4, q05
0.711, q95 9.488). That null is WRONG and this arm shows why: S under a foreign partition
does not follow chi2(4). Both partitions are STRATIFIED ON y, and these columns are strongly
predictive of y, so stratification removes the y-attributable component of the between-fold
variance; a finite-population correction removes more. The observed foreign values sit at
~1-3, not ~4 with a 9.5 upper tail. ⛔ Do not quote w80c's chi2(4) row, its q05/q95, or its
"SCRUBBED" class (which found zero members using a threshold from that wrong null). The
EMPIRICAL null is the NEG arm, and prereg2 registered it.
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
OUT = os.path.join(ROOT, "experiments", "w80d_poscalib.json")
B = json.load(open(os.path.join(ROOT, "experiments", "w80b_foldsig.json")))
C = json.load(open(os.path.join(ROOT, "experiments", "w80c_foldsig_diag.json")))


def main() -> int:
    reg = read_prereg()
    print("=" * 78)
    print("w80d -- POST-HOC: the S magnitude of a SINGLE matched model")
    print("⛔ NOT REGISTERED. w80b's verdict stays INDETERMINATE. Nothing here licenses a send.")
    print("=" * 78)

    tr = pd.read_csv(os.path.join(ROOT, "data", "train.csv"), usecols=[common.TARGET])
    y = tr[common.TARGET].values
    folds_ours = common.get_folds(y)
    folds_neg = list(StratifiedKFold(n_splits=reg["neg_k"], shuffle=True,
                                     random_state=reg["neg_seed"]).split(np.zeros(y.size), y))

    singles = sorted(f for f in os.listdir(os.path.join(ROOT, "oof"))
                     if f.startswith("oof_") and f.endswith(".npy"))
    rows = []
    print(f"\n  OUR OWN {len(singles)} SINGLE MODELS -- known matched, built by agent/ on our folds")
    print(f"  {'model':>28s} {'S_ours':>10s} {'S_foreign':>10s}")
    for f in singles:
        c = np.load(os.path.join(ROOT, "oof", f))
        if c.size != len(y):
            print(f"  {f:>28s}   skipped ({c.size} rows)")
            continue
        s_o, s_f = fold_signature(c, folds_ours), fold_signature(c, folds_neg)
        rows.append({"model": f[4:-4], "S_ours": s_o, "S_foreign": s_f})
        print(f"  {f[4:-4]:>28s} {s_o:10.3f} {s_f:10.3f}")

    sing = np.array([r["S_ours"] for r in rows])
    sing_f = np.array([r["S_foreign"] for r in rows])

    blend_pos = np.array([r["S_ours"] for r in B["per_stem"]])
    blend_neg = np.array([r["S_foreign"] for r in B["per_stem"]])
    pack = np.array([r["S_ours"] for r in C["per_col"]])
    pack_f = np.array([r["S_foreign"] for r in C["per_col"]])

    def line(tag, a):
        print(f"  {tag:<34s} n={a.size:3d}  min {a.min():9.3f}  med {np.median(a):9.3f}"
              f"  max {a.max():9.3f}")

    print("\n" + "-" * 78)
    print("  S UNDER OUR FOLDS")
    line("prereg2 POS (20 blends, matched)", blend_pos)
    line("OUR single models (matched)", sing)
    line("the 50 pack columns", pack)
    print("\n  S UNDER THE FOREIGN PARTITION -- the empirical null")
    line("prereg2 NEG (20 blends)", blend_neg)
    line("OUR single models", sing_f)
    line("the 50 pack columns", pack_f)

    print("\n" + "-" * 78)
    print("  1. THE NULL IS NOT chi2(4).")
    null = np.r_[blend_neg, sing_f, pack_f]
    print(f"     Pooled foreign-partition S over {null.size} columns of three unrelated kinds:")
    print(f"       min {null.min():.3f}  median {np.median(null):.3f}  max {null.max():.3f}")
    print("     chi2(4) would give median 3.36 and a 95th percentile of 9.49. Stratification")
    print("     on y plus the finite-population correction shrink it. ⛔ w80c's chi2(4)")
    print("     thresholds and its 'SCRUBBED' class are void; the NEG arm is the null.")

    print("\n  2. A SINGLE MATCHED MODEL DOES NOT REACH min(POS).")
    below = int((sing < blend_pos.min()).sum())
    print(f"     {below} of {sing.size} of our OWN single models -- every one KNOWN matched --")
    print(f"     score BELOW min(POS) = {blend_pos.min():.3f}, the bar Q3 required the pack's")
    print("     MEDIAN to clear. Under prereg2's own rule those known-matched models would")
    print("     have read INDETERMINATE or FOREIGN.")
    ratio = np.median(blend_pos) / np.median(sing)
    print(f"     Blend median / single median = {ratio:.1f}x. A 150-member blend averages 150")
    print("     per-fold biases that agree in sign within a fold; one member does not.")

    print("\n  3. WHAT THIS DOES AND DOES NOT SAY ABOUT THE PACK.")
    hi = int((pack > sing.max()).sum())
    mid = int(((pack >= np.median(sing)) & (pack <= sing.max())).sum())
    lo = int((pack < blend_neg.max()).sum())
    print(f"     Against the SINGLE-model scale: {hi} pack columns exceed even the largest")
    print(f"     single matched model we own ({sing.max():.3f}); {mid} sit inside the matched")
    print(f"     single-model range; {lo} sit at or under max(NEG) = {blend_neg.max():.3f}.")
    print("     ⚠ Read the last group carefully. A LOW S is NOT evidence of a foreign")
    print("       partition for a single column: our own known-matched singles reach as low")
    print(f"       as {sing.min():.3f}. Low S means the instrument is UNINFORMATIVE on that")
    print("       column, in both directions. Only the high tail carries information.")
    print("     ⛔ AND THIS IS NOT A RE-READ OF THE VERDICT. Substituting a control set")
    print("       chosen after seeing the answer is precisely the move prereg2 forbids. This")
    print("       arm prices the DEFECT in the registered design. Acting on it requires a")
    print("       NEW registration -- see experiments/w80_prereg3.txt.")

    res = {
        "script": "w80d_poscalib.py",
        "REGISTERED": False,
        "note": ("POST-HOC instrument calibration. w80b's registered verdict INDETERMINATE is "
                 "NOT amended. Corrects w80c's chi2(4) null, which is wrong."),
        "corrects": {"file": "w80c_foldsig_diag.json",
                     "what": "its chi2(4) null, q05/q95 thresholds and SCRUBBED class are void"},
        "singles": rows,
        "scales": {
            "blend_pos": {"n": int(blend_pos.size), "min": float(blend_pos.min()),
                          "median": float(np.median(blend_pos)), "max": float(blend_pos.max())},
            "single_pos": {"n": int(sing.size), "min": float(sing.min()),
                           "median": float(np.median(sing)), "max": float(sing.max())},
            "pack": {"n": int(pack.size), "min": float(pack.min()),
                     "median": float(np.median(pack)), "max": float(pack.max())},
            "null_pooled": {"n": int(null.size), "min": float(null.min()),
                            "median": float(np.median(null)), "max": float(null.max())},
        },
        "singles_below_minPOS": below,
        "blend_over_single_ratio": float(ratio),
        "pack_vs_single_scale": {"above_max_single": hi, "inside_single_range": mid,
                                 "at_or_below_maxNEG": lo},
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
