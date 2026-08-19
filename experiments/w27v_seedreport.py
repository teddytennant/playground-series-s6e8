"""w27v -- read out w14c's stacker-fold-seed data, which was computed 2026-08-14 and never
written up. Analysis only: loads the cached OOF vectors from experiments/w14c_out/ directly
and never touches the member matrix, so it costs seconds rather than the ~500s load +
several GB that `w14c_seedstack.py --stage seeds` would pay to reach the same cache.

THE QUESTION. Every stack CV in this workspace is cross-fitted on ONE partition,
StratifiedKFold(5, shuffle=True, random_state=42). The members must share it. The STACKER's
own cross-fit need not, and never has been varied in any reported number. So every gap this
workspace has argued over -- h3 vs ens4, 188 vs 190, corr vs raw, all of them 1-7e-6 -- has
been quoted against a partition-noise error bar that was never measured. Final selection is
on CV. If the ordering is not stable across partitions, the selection rule is reading noise.

WHAT IS AND IS NOT MEASURED HERE. The member OOF vectors are frozen on seed 42 throughout;
only the stacker's partition moves. That isolates stacker partition noise and nothing else.
The member set is blend159av (159 members), not the current 188/190 pack, so the LEVELS are
not comparable to today's files -- the SPREADS and the PAIRED CONTRAST STABILITY are what
transfers, and they are what is reported.

Two numbers with very different meanings, and conflating them is the trap:
  * MARGINAL sd across seeds -- how much one file's reported CV moves. Large.
  * PAIRED contrast sd -- how much the DIFFERENCE between two files moves when both are
    scored on the identical partition. This is the one the selection rule actually uses,
    and the partition noise largely cancels in it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import TARGET, load_raw  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "w14c_out")
KINDS = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    tr, _te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    seeds = sorted({int(f.split("_seed")[1].split("_")[0])
                    for f in os.listdir(OUT) if f.endswith(".npy")})
    print(f"stacker fold seeds found: {seeds}   (members frozen on seed 42 throughout)")

    rows = {}
    for s in seeds:
        oof = {}
        for k in KINDS:
            p = os.path.join(OUT, f"w14c_oof_seed{s}_{k}.npy")
            if not os.path.exists(p):
                break
            oof[k] = np.load(p)
        if len(oof) != len(KINDS):
            print(f"  seed {s}: incomplete, skipped")
            continue
        r = {k: roc_auc_score(y, oof[k]) for k in KINDS}
        r["h3"] = roc_auc_score(y, np.mean([rk(oof[k]) for k in H3], 0))
        r["ens4"] = roc_auc_score(y, np.mean([rk(oof[k]) for k in KINDS], 0))
        rows[s] = r
        print(f"  seed {s:>3d} read", flush=True)

    df = pd.DataFrame(rows).T
    df.index.name = "fold_seed"
    print("\n== cross-fitted stack AUC by STACKER fold seed ==")
    print(df.to_string(float_format="%.7f"))

    print("\n== MARGINAL spread across seeds (what one file's reported CV does) ==")
    for c in df.columns:
        v = df[c].to_numpy()
        print(f"  {c:>8s}  mean {v.mean():.7f}  sd {v.std(ddof=1)*1e6:7.2f}e-6  "
              f"range {(v.max()-v.min())*1e6:7.2f}e-6")

    if 42 in df.index and len(df) > 1:
        d = df.drop(index=42).mean() - df.loc[42]
        print("\n== pre-registered check: off-seed mean minus seed 42 ==")
        print("   (w14c registered POSITIVE = the member-model-sharing optimism showing)")
        for c in df.columns:
            print(f"  {c:>8s}  {d[c]*1e6:+8.2f}e-6")

    print("\n== PAIRED contrasts on identical partitions -- the number the 1e-6 "
          "arguments never had ==")
    pairs = [("h3", "ens4"), ("h3", "rankraw"), ("h3", "hybrid"), ("h3", "rescale"),
             ("hybrid", "logit"), ("rankraw", "hybrid"), ("ens4", "logit"),
             ("rankraw", "rescale")]
    print(f"  {'contrast':>20s}  {'mean':>10s}  {'sd':>9s}  {'se':>9s}  verdict")
    for a, b in pairs:
        d = (df[a] - df[b]).to_numpy() * 1e6
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a:>7s} - {b:<10s}  {d.mean():+9.2f}  {d.std(ddof=1):8.2f}  {se:8.2f}  "
              f"[{ok}]  per-seed {np.array2string(d, precision=1)}")

    print("\n== the ratio that matters ==")
    for a, b in pairs:
        d = (df[a] - df[b]).to_numpy() * 1e6
        marg = np.mean([df[a].std(ddof=1), df[b].std(ddof=1)]) * 1e6
        print(f"  {a:>7s} - {b:<10s}  paired sd {d.std(ddof=1):7.2f}e-6  vs marginal sd "
              f"{marg:7.2f}e-6  -> partition noise cancels "
              f"{100*(1-d.std(ddof=1)/max(marg,1e-12)):5.1f}%")

    df.to_csv(os.path.join(ROOT, "experiments", "w27v_seedstack.csv"))
    print(f"\nwrote experiments/w27v_seedstack.csv")


if __name__ == "__main__":
    main()
