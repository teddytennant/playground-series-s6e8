"""w43a -- side-agreement audit over EVERY member the builds can load.

w42 s3 found two imported members whose test vector was 4x and 5x its own OOF vector.
Both w42b gates (solo AUC, maxcorr rank correlation) are blind to that BY CONSTRUCTION:
both are rank statistics on the OOF side alone, and no monotone rescale of the TEST side
can move either. The defect is therefore invisible to CV and fatal on LB, and w42 s3
closed with "every earlier import should be re-checked on this axis". This is that check.

It audits the axis two ways, because sd_test/sd_oof alone is a blunt instrument:

  RATIO  sd_test / sd_oof -- what w42d's import gate uses.
  KS     the Kolmogorov-Smirnov distance between the two sides' MARGINAL distributions.

KS is the sharper statistic and it is the reason this script exists rather than a one-line
ratio print. Train and test are iid draws from one generator over the same covariates, so a
correctly-exported member's OOF and test predictions are two samples from the SAME marginal.
KS then has a known null: with n=691369 and m=296302 the asymptotic 99.99% point is
1.95*sqrt(1/n+1/m) = 0.0042. Anything far above that is a statement that the two sides did
not come out of the same pipeline -- and it fires on shifts a variance ratio cannot see
(a clip, a per-side rescale, a different seed's calibration), while NOT firing on the one
legitimate asymmetry we know about.

That legitimate asymmetry matters and it is why RATIO alone must not be read as a defect
count. OOF rows are predicted by ONE fold model; the test vector is usually the MEAN of 5
fold models. Averaging 5 correlated predictors shrinks variance, so sd_test/sd_oof slightly
BELOW 1 is the expected, healthy signature of a correct 5-fold export -- not a fault. The
pack's 0.9923 floor is that effect. Ratios far ABOVE 1 have no such excuse.
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import DATA, LIB, OOF, load_raw, TARGET  # noqa: E402

N_TR, N_TE = 691369, 296302
# the exact dir list load_members walks, in its order, plus every ext_members* on disk so
# a dir that is not currently in any --extra-dirs still gets audited before it ever is.
DIRS = [os.path.join(LIB, "oof"), OOF] + sorted(
    (os.path.join(DATA, d) for d in os.listdir(DATA) if d.startswith("ext_members")),
    key=lambda p: (len(p), p))

# KS null: asymptotic critical value at alpha, two-sample, n=691369 m=296302.
KS_NULL = np.sqrt(1.0 / N_TR + 1.0 / N_TE)
KS_9999 = 1.9495 * KS_NULL   # ~0.00417


def ks_two_sample(a, b):
    """Exact two-sample KS distance via a merged-order walk. O((n+m) log(n+m))."""
    a = np.sort(a); b = np.sort(b)
    allv = np.concatenate([a, b])
    cdfa = np.searchsorted(a, allv, side="right") / len(a)
    cdfb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdfa - cdfb)))


def main():
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    del tr

    seen, rows = set(), []
    for d in DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("oof_") and fn.endswith(".npy")):
                continue
            nm = fn[4:-4]
            if nm in seen:
                continue          # load_members keeps the FIRST dir's copy; mirror that
            tp = os.path.join(d, f"test_{nm}.npy")
            if not os.path.exists(tp):
                continue
            o = np.load(os.path.join(d, fn)).astype(np.float64)
            t = np.load(tp).astype(np.float64)
            if o.shape != (len(y),) or t.shape != (N_TE,):
                continue
            seen.add(nm)
            so, st = float(o.std()), float(t.std())
            rows.append(dict(
                member=nm, dir=os.path.basename(d),
                sd_oof=so, sd_test=st, ratio=(st / so if so > 0 else np.inf),
                mean_oof=float(o.mean()), mean_test=float(t.mean()),
                min_oof=float(o.min()), max_oof=float(o.max()),
                min_test=float(t.min()), max_test=float(t.max()),
                ks=ks_two_sample(o, t),
            ))
            print(f"  {len(rows):3d} {nm:44s} {os.path.basename(d):16s} "
                  f"ratio {rows[-1]['ratio']:7.4f}  ks {rows[-1]['ks']:.5f}", flush=True)

    df = pd.DataFrame(rows).sort_values("ks", ascending=False)
    out = os.path.join(HERE, "w43a_scaleaudit.csv")
    df.to_csv(out, index=False)

    print(f"\n{len(df)} members audited -> {out}")
    print(f"KS null scale sqrt(1/n+1/m) = {KS_NULL:.5f}; 99.99% point {KS_9999:.5f}")
    print(f"\nRATIO span [{df.ratio.min():.4f}, {df.ratio.max():.4f}]")
    print("\n--- 20 worst by KS (marginal disagreement between the two sides) ---")
    print(df.head(20)[["member", "dir", "ratio", "ks", "sd_oof", "sd_test",
                       "mean_oof", "mean_test"]].to_string(index=False,
                                                           float_format="%.5f"))
    print("\n--- outside the w42d import envelope [0.95, 1.45] on RATIO ---")
    bad = df[(df.ratio < 0.95) | (df.ratio > 1.45)]
    print(bad.to_string(index=False, float_format="%.5f") if len(bad) else "  none")
    print(f"\n--- KS above the 99.99% null point {KS_9999:.5f}: "
          f"{(df.ks > KS_9999).sum()} of {len(df)} ---")


if __name__ == "__main__":
    main()
