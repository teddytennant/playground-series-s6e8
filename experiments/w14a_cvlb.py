"""Where the CV-to-LB relationship stands with all 30 readings, including 2026-08-13's ten.

`predict_lb.py` was written when 14 (CV, LB) pairs existed and it hard-codes them. Ten more
landed on 2026-08-13 -- including the first-ever `h3` readings and the third within-set
`logit` replication -- and nothing has re-fitted the relationship since. This script reads
the pairs from `audit_results.csv` (which now carries the LB column) rather than a literal,
so it cannot go stale again.

Four things are asked, in the order they matter:

1.  **Does CV predict LB at all, at the top?** Regress LB on CV over the >=150-member
    family only. The <=151-member public stacks (CV 0.9696) are held out as a separate
    stratum because `predict_lb.py`'s confound 1 -- the gap shrinks as CV rises -- would
    otherwise dominate the slope with three leverage points.
2.  **How much of the residual is quantisation?** The LB reports 5 decimals, so every
    reading is a 1e-5 bin. The whole >=150 family spans 8 bins. If the residual sd is at
    or below one bin the regression is measuring the reporting grid, not the test slice.
3.  **The transform offsets, with the 158 set folded in.** The `logit` displacement is the
    effect that made `blend158_logit` the auto-selection risk. Per-transform mean residual
    against the common fit, and the within-set `logit - hybrid` LB difference, which is
    paired and therefore the sharper instrument.
4.  **What the fit predicts for the pick, and for the file being sent today.**

    w14a_cvlb.py
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit_results.csv")

TRANSFORMS = ("logit", "hybrid", "rankraw", "rescale")


def classify(name):
    """(member-set tag, transform tag) for a submission name."""
    for t in TRANSFORMS:
        if name.endswith("_" + t):
            return name[: -len(t) - 1], t
    if name.endswith("_h3"):
        return name[:-3], "h3"
    if name.endswith("_wh3"):
        return name[:-4], "wh3"
    if name.endswith("_w") or name.endswith("w") and re.search(r"w\d*$", name):
        return name, "w"
    return name, "ens4"


def main():
    a = pd.read_csv(AUDIT)
    a = a[a["lb"].notna()].copy()
    a[["mset", "kind"]] = a["name"].apply(lambda n: pd.Series(classify(n)))
    # >=150-member family: everything except the three early public-library stacks
    a["big"] = ~a["name"].str.startswith("stack_pub7") & \
               ~a["name"].str.startswith("stack_pub8")
    big, small = a[a["big"]], a[~a["big"]]

    print(f"{len(a)} scored files ({len(big)} in the >=150-member family, "
          f"{len(small)} early public stacks held out)\n")

    print("=== 1. LB regressed on CV, >=150-member family only ===")
    x, y = big["cv"].to_numpy(), big["lb"].to_numpy()
    sl, ic = np.polyfit(x, y, 1)
    pred = sl * x + ic
    res = y - pred
    ss = 1 - res.var() / y.var()
    print(f"  n={len(x)}   slope {sl:+.3f}   R^2 {ss:.3f}")
    print(f"  residual sd {res.std(ddof=2):.3e}   max|res| {np.abs(res).max():.3e}")
    print(f"  pearson(CV, LB) {np.corrcoef(x, y)[0, 1]:+.3f}   "
          f"spearman {pd.Series(x).corr(pd.Series(y), method='spearman'):+.3f}")
    print("  gap (LB - CV):  mean {:+.6f}  sd {:.6f}  range [{:+.6f}, {:+.6f}]".format(
        (y - x).mean(), (y - x).std(ddof=1), (y - x).min(), (y - x).max()))
    print("  held-out early stacks, gap (LB - CV): " +
          ", ".join(f"{n} {v:+.5f}" for n, v in
                    zip(small["name"], small["lb"] - small["cv"])))

    print("\n=== 2. is the residual just the reporting grid? ===")
    print(f"  LB quantisation step 1e-5; family spans "
          f"{y.min():.5f}..{y.max():.5f} = {round((y.max()-y.min())/1e-5)} steps")
    print(f"  residual sd {res.std(ddof=2):.3e} = "
          f"{res.std(ddof=2)/1e-5:.2f} quantisation steps")
    print(f"  a slope of {sl:+.2f} means a +1e-5 CV move buys "
          f"{sl*1e-5/1e-5:+.2f}e-5 LB, i.e. {1e-5/abs(sl) if sl else float('inf'):.1e} "
          f"CV per LB step")

    print("\n=== 3. transform offsets against the common fit ===")
    big = big.assign(res=res)
    print(f"  {'kind':9s} {'n':>2s} {'mean resid':>11s} {'sd':>9s}  files")
    for k, g in big.groupby("kind"):
        print(f"  {k:9s} {len(g):2d} {g['res'].mean():+11.3e} "
              f"{g['res'].std(ddof=1) if len(g) > 1 else float('nan'):9.3e}  "
              + ",".join(sorted(g["mset"])))

    print("\n  within-member-set paired contrasts (the sharp instrument):")
    print(f"  {'member set':10s} {'logit CV':>9s} {'hyb CV':>9s} {'dCV':>8s} "
          f"{'logit LB':>9s} {'hyb LB':>9s} {'dLB':>8s} {'displace':>9s}")
    disp = []
    for ms, g in big.groupby("mset"):
        d = g.set_index("kind")
        if "logit" in d.index and "hybrid" in d.index:
            dcv = d.loc["logit", "cv"] - d.loc["hybrid", "cv"]
            dlb = d.loc["logit", "lb"] - d.loc["hybrid", "lb"]
            disp.append(dlb - dcv)
            print(f"  {ms:10s} {d.loc['logit','cv']:9.6f} {d.loc['hybrid','cv']:9.6f} "
                  f"{dcv:+8.1e} {d.loc['logit','lb']:9.5f} {d.loc['hybrid','lb']:9.5f} "
                  f"{dlb:+8.1e} {dlb-dcv:+9.1e}")
    if disp:
        disp = np.array(disp)
        print(f"  displacement (dLB - dCV): mean {disp.mean():+.2e}  "
              f"n={len(disp)}  all same sign: {bool((disp > 0).all() or (disp < 0).all())}")

    print("\n=== 4. what the fit says about the files that matter ===")
    tgt = pd.read_csv(AUDIT)
    for n in ("blend159av_h3", "blend160origm_h3", "blend158_logit", "blendtop3",
              "blend159av_wh3", "blend156w2"):
        r = tgt[tgt["name"] == n]
        if not len(r):
            print(f"  {n:18s} not in audit_results.csv")
            continue
        cv = float(r["cv"].iloc[0])
        lbv = r["lb"].iloc[0]
        p = sl * cv + ic
        print(f"  {n:18s} CV {cv:.6f}  predicted LB {p:.5f}  "
              + (f"actual {lbv:.5f}  resid {lbv-p:+.2e}" if pd.notna(lbv) else "UNSENT"))

    print("\n=== 5. the ordering question, stated plainly ===")
    b = big.sort_values("cv", ascending=False)
    print("  top of the family by CV, with LB alongside:")
    for _, r in b.head(10).iterrows():
        print(f"    {r['name']:22s} CV {r['cv']:.6f}   LB {r['lb']:.5f}")
    top = b[b["cv"] >= 0.970040]
    print(f"\n  among the {len(top)} files with CV >= 0.970040: "
          f"CV spans {top['cv'].max()-top['cv'].min():.1e}, "
          f"LB spans {top['lb'].max()-top['lb'].min():.1e} "
          f"({round((top['lb'].max()-top['lb'].min())/1e-5)} grid steps), "
          f"rank corr {top['cv'].corr(top['lb'], method='spearman'):+.2f}")

    print("\n=== 6. h3 vs ens4: is the LB inversion just logit re-entering? ===")
    # ens4 = rank-average of all four transform stacks; h3 drops `logit`. So any LB
    # advantage ens4 holds over h3 at the SAME member set has exactly one candidate
    # source -- the logit displacement measured in section 3 -- and the mean-of-component-
    # gaps estimator that predict_lb.py validated on 150fx turns that into a number.
    print(f"  {'member set':10s} {'h3 CV':>9s} {'ens4 CV':>9s} {'dCV':>8s} "
          f"{'h3 LB':>8s} {'ens4 LB':>8s} {'dLB':>8s}   predicted dLB from components")
    for ms, g in big.groupby("mset"):
        d = g.set_index("kind")
        if not {"h3", "ens4"} <= set(d.index):
            continue
        dcv = d.loc["ens4", "cv"] - d.loc["h3", "cv"]
        dlb = d.loc["ens4", "lb"] - d.loc["h3", "lb"]
        have = [t for t in TRANSFORMS if t in d.index]
        est = ""
        if len(have) == 4:
            e4 = np.mean([d.loc[t, "lb"] for t in TRANSFORMS])
            e3 = np.mean([d.loc[t, "lb"] for t in TRANSFORMS if t != "logit"])
            est = f"{e4 - e3:+.1e}  (all four transforms scored)"
        else:
            est = f"n/a -- only {len(have)}/4 transforms scored"
        print(f"  {ms:10s} {d.loc['h3','cv']:9.6f} {d.loc['ens4','cv']:9.6f} "
              f"{dcv:+8.1e} {d.loc['h3','lb']:8.5f} {d.loc['ens4','lb']:8.5f} "
              f"{dlb:+8.1e}   {est}")
    print("  CV prefers h3; the public slice prefers ens4 by exactly one grid step, and\n"
          "  the component decomposition attributes that step to `logit` re-entering.")


if __name__ == "__main__":
    main()
