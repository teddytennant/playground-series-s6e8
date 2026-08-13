"""Pool `oofsim.py` across outer splits: is the displacement replicated, and how big?

One split gives one number. The claim under test -- that the OOF instrument systematically
under-rates `logit` -- is a claim about the *mechanism*, so it has to survive re-drawing the
hold-out. This pools the per-seed JSON and reports, per dose:

    mean and sd across seeds of gap(logit) - gap(hybrid), and of gap(ens4) - gap(h3)

sd ACROSS SEEDS is the honest error bar here, strictly larger than the within-split paired
bootstrap (which holds the member models and the split fixed). Quote this one.

    oofsim_summary.py 7 11 13
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import ROOT  # noqa: E402

CACHE = os.path.join(ROOT, "cache", "oofsim")
KINDS = ("logit", "hybrid", "rankraw", "rescale", "ens4", "h3")
FULL_INNER = 553_095            # 80% of train, the only inner size a real run has


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [7, 11, 13]
    frames = []
    for s in seeds:
        p = os.path.join(CACHE, f"results_s{s}.json")
        if not os.path.exists(p):
            print(f"[skip] no results for seed {s}")
            continue
        with open(p) as f:
            d = json.load(f)
        # A `--frac` smoke run writes the same schema with a fifth of the rows and AUCs a
        # point and a half lower. Pooling one silently would poison the answer, so refuse
        # anything that is not a full-size run and refuse anything that will not say.
        meta = d.get("meta")
        if meta is None:
            print(f"[!] seed {s}: no meta block -- written before the size stamp existed. "
                  f"Check by hand that its dose-0 CV is ~0.962 (a 5% smoke run reads "
                  f"~0.947) before trusting this pool.")
        elif meta["n_inner"] < 0.9 * FULL_INNER:
            print(f"[SKIP] seed {s}: n_inner {meta['n_inner']:,} vs {FULL_INNER:,} "
                  f"expected -- this is a subsample, not a run.")
            continue
        df = pd.DataFrame(d["results"])
        df["seed"] = s
        frames.append(df)
    if not frames:
        raise SystemExit("no results")
    df = pd.concat(frames, ignore_index=True)
    seeds = sorted(df.seed.unique())
    print(f"pooling {len(seeds)} outer splits: {seeds}\n")

    sub = df[df.variant.isin(KINDS)]

    print("=== per-transform gap (test - cv), mean +/- sd across splits ===")
    print("the LEVEL is not interpretable (it absorbs each split's own shift);")
    print("only differences between transforms on the same split are.\n")
    g = sub.pivot_table(index="dose", columns="variant", values="gap",
                        aggfunc=["mean", "std"])
    print(g.to_string(float_format="%+.6f"))

    print("\n=== the two contrasts the deadline pick turns on ===")
    rows = []
    for dose, d in sub.groupby("dose"):
        w = d.pivot(index="seed", columns="variant", values="gap")
        lh = w["logit"] - w["hybrid"]
        lr = w["logit"] - w["rankraw"]
        eh = w["ens4"] - w["h3"]
        rows.append(dict(
            dose=dose,
            logit_hybrid=lh.mean(), lh_sd=lh.std(ddof=1) if len(lh) > 1 else np.nan,
            lh_signs=f"{int((lh > 0).sum())}/{len(lh)}",
            logit_rankraw=lr.mean(), lr_sd=lr.std(ddof=1) if len(lr) > 1 else np.nan,
            ens4_h3=eh.mean(), eh_sd=eh.std(ddof=1) if len(eh) > 1 else np.nan,
            eh_signs=f"{int((eh > 0).sum())}/{len(eh)}"))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format="%+.6f"))

    print("\nreal-data comparison: logit-hybrid claimed +0.000097 (3 LB sets, 3/3),")
    print("                      ens4-h3     claimed +0.000021 (mix-gap estimator)")

    print("\n=== does the OOF-searched blend weight generalise? (today's angle) ===")
    ws = df[df.variant == "wsearch"]
    for _, r in ws.sort_values(["dose", "seed"]).iterrows():
        gainT = r["test_at_w_true"] - r["test"]
        print(f"  dose {int(r['dose'])} seed {int(r['seed'])}: "
              f"w_oof {r['w_oof']} -> hold {r['test']:.6f}   "
              f"w_true {r['w_true']} -> {r['test_at_w_true']:.6f}  "
              f"(search left {gainT:+.6f} on the table)")

    print("\n=== equal weights vs the searched weights, on the hold-out ===")
    for dose, d in sub.groupby("dose"):
        w = d.pivot(index="seed", columns="variant", values="test")
        wsd = ws[ws.dose == dose].set_index("seed")["test"]
        dd = (wsd - w["ens4"]).dropna()
        if len(dd):
            print(f"  dose {dose}: searched - ens4 = {dd.mean():+.6f} "
                  f"+/- {dd.std(ddof=1) if len(dd) > 1 else float('nan'):.6f}  "
                  f"[{int((dd > 0).sum())}/{len(dd)} positive]")


if __name__ == "__main__":
    main()
