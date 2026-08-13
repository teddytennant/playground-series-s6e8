"""Re-price the transform question with the ten CV->LB points collected 2026-08-13.

Before today the transform->LB map was assembled across member sets, so every contrast
was confounded by the set. Today's six `blend158_*` files are one member set scored under
all six transforms, which is the matched design the earlier entries kept asking for, plus
the first `h3` readings ever (158, 159av, 160origm).

Two pre-registered questions:

  Q1  does logit's (LB - CV) gap sit above hybrid's a third time, within set?
  Q2  h3 excludes logit from the mix and ens4 includes it. If Q1 is real, ens4 should beat
      h3 on LB while losing to it on CV. Two sets now carry both.

The honest caveat, stated up front so no later run forgets it: **the LB prints five
decimals, so its resolution is 1e-5 and every contrast below is 1-3 units in the last
place.** No single comparison here is worth anything. Only the replication count is.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

# (member set, transform) -> (cross-fitted CV, public LB). >=150-member stacks only; the
# 74/88/86-member entries are excluded because the gap shrinks as CV rises and pooling
# them inflates the logit gap by ~4e-5 (confound 1 in predict_lb.py).
G = {
    ("150fx", "logit"):   (0.969950, 0.97103),
    ("150fx", "hybrid"):  (0.970014, 0.97099),
    ("150fx", "rankraw"): (0.970024, 0.97102),
    ("150fx", "rescale"): (0.970013, 0.97102),
    ("150fx", "ens4"):    (0.970032, 0.97104),
    ("150sx", "logit"):   (0.969955, 0.97104),
    ("150sx", "hybrid"):  (0.970016, 0.97101),
    ("150sx", "rankraw"): (0.970022, 0.97102),
    ("150sx", "ens4"):    (0.970033, 0.97104),
    ("151",   "hybrid"):  (0.970024, 0.97099),
    ("151",   "rankraw"): (0.970023, 0.97102),
    ("156",   "rankraw"): (0.970036, 0.97104),
    ("156",   "rescale"): (0.970025, 0.97105),
    ("156",   "ens4"):    (0.970042, 0.97106),
    # --- new 2026-08-13: the first fully-crossed member set, and the first h3 readings ---
    ("158",   "logit"):   (0.969961, 0.97106),
    ("158",   "hybrid"):  (0.970028, 0.97103),
    ("158",   "rankraw"): (0.970036, 0.97104),
    ("158",   "rescale"): (0.970027, 0.97105),
    ("158",   "ens4"):    (0.970043, 0.97106),
    ("158",   "h3"):      (0.970048, 0.97105),
    ("159av", "ens4"):    (0.970045, 0.97106),
    ("159av", "h3"):      (0.970049, 0.97105),
    ("160origm", "h3"):   (0.970049, 0.97105),
    ("156w",  "w4"):      (0.970047, 0.97105),
}
gap = {k: lb - cv for k, (cv, lb) in G.items()}
ULP = 1e-5


def spearman(a, b):
    def rank(v):
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties, which the LB column is full of
        for u in np.unique(v):
            m = v == u
            r[m] = r[m].mean()
        return r
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def within_set(t_hi, t_lo):
    """Paired (t_hi - t_lo) on CV and on LB, restricted to sets carrying both."""
    sets = sorted({k[0] for k in G if (k[0], t_hi) in G and (k[0], t_lo) in G})
    rows = []
    for s in sets:
        dcv = G[(s, t_hi)][0] - G[(s, t_lo)][0]
        dlb = G[(s, t_hi)][1] - G[(s, t_lo)][1]
        rows.append((s, dcv, dlb))
    return rows


def report(t_hi, t_lo):
    rows = within_set(t_hi, t_lo)
    if not rows:
        print(f"  {t_hi} vs {t_lo}: no set carries both")
        return
    print(f"\n  {t_hi} - {t_lo}, within member set:")
    print(f"    {'set':>9s}  {'dCV':>10s}  {'dLB':>10s}  {'dLB(ulp)':>9s}")
    for s, dcv, dlb in rows:
        print(f"    {s:>9s}  {dcv:+.6f}  {dlb:+.5f}  {dlb / ULP:+8.0f}")
    dcv = np.array([r[1] for r in rows])
    dlb = np.array([r[2] for r in rows])
    agree = int(np.sum(np.sign(dcv) == np.sign(dlb)))
    print(f"    mean       {dcv.mean():+.6f}  {dlb.mean():+.5f}   "
          f"sign agreement with CV {agree}/{len(rows)}")
    # exact sign test on the LB direction, two-sided; ties on the LB are dropped
    nz = dlb[dlb != 0]
    n, k = len(nz), int(np.sum(nz > 0))
    p = sum(math.comb(n, m) for m in range(n + 1)
            if abs(m - n / 2) >= abs(k - n / 2)) / 2 ** n if n else float("nan")
    print(f"    LB direction {k}/{n} positive ({len(dlb) - n} tied), "
          f"exact two-sided sign-test p = {p:.3f}")


def main():
    print("=== Q0: on the ONE fully-crossed member set (158), does CV order the LB? ===")
    six = ["h3", "ens4", "rankraw", "hybrid", "rescale", "logit"]
    cv = [G[("158", t)][0] for t in six]
    lb = [G[("158", t)][1] for t in six]
    print(f"    {'transform':>9s}  {'CV':>9s}  {'LB':>8s}")
    for t, c, l in sorted(zip(six, cv, lb), key=lambda r: -r[1]):
        print(f"    {t:>9s}  {c:.6f}  {l:.5f}")
    print(f"\n    CV spread {max(cv) - min(cv):.6f}   LB spread {max(lb) - min(lb):.5f}"
          f"  ({(max(lb) - min(lb)) / ULP:.0f} ulp)")
    print(f"    spearman(CV, LB) over the six = {spearman(cv, lb):+.3f}")

    print("\n=== Q1: the logit bias, now three within-set replications ===")
    report("logit", "hybrid")
    report("logit", "rankraw")
    report("logit", "rescale")

    print("\n=== Q2: ens4 (includes logit) vs h3 (excludes it) ===")
    report("ens4", "h3")

    print("\n=== per-transform mean (LB - CV) gap, >=150-member sets ===")
    for t in ("logit", "ens4", "rescale", "rankraw", "h3", "hybrid", "w4"):
        v = [gap[k] for k in gap if k[1] == t]
        if not v:
            continue
        print(f"    {t:>8s} n={len(v)}  mean {np.mean(v):+.6f}   " +
              " ".join(f"{x:+.6f}" for x in sorted(v)))

    print("\n=== all 24 points: does CV predict LB at all? ===")
    cvs = np.array([v[0] for v in G.values()])
    lbs = np.array([v[1] for v in G.values()])
    print(f"    spearman over all {len(G)} = {spearman(cvs, lbs):+.3f}")
    hi = [k for k in G if G[k][0] >= 0.97002]
    print(f"    restricted to CV >= 0.97002 (n={len(hi)}): "
          f"{spearman([G[k][0] for k in hi], [G[k][1] for k in hi]):+.3f}")

    print("\n=== how far apart must two candidates be for the LB to order them? ===")
    print("    split by whether a `logit` entry is one of the two, because if the bias is")
    print("    real the ordering failures should live entirely in the logit column.")
    ks = list(G)

    def bucket(lo, hi_, want_logit):
        agree = tot = 0
        for a, b in itertools.combinations(ks, 2):
            if (("logit" in (a[1], b[1]))) != want_logit:
                continue
            d = G[a][0] - G[b][0]
            if not (lo <= abs(d) < hi_):
                continue
            dl = G[a][1] - G[b][1]
            if dl == 0:
                continue
            tot += 1
            agree += int(np.sign(d) == np.sign(dl))
        return agree, tot

    print(f"    {'CV gap':>18s}  {'no logit':>14s}  {'logit involved':>16s}")
    for lo, hi_ in ((0, 2e-5), (2e-5, 5e-5), (5e-5, 1e-4), (1e-4, 1.0)):
        a0, t0 = bucket(lo, hi_, False)
        a1, t1 = bucket(lo, hi_, True)
        s0 = f"{a0}/{t0} = {100 * a0 / t0:3.0f}%" if t0 else "-"
        s1 = f"{a1}/{t1} = {100 * a1 / t1:3.0f}%" if t1 else "-"
        print(f"    [{lo:.0e}, {hi_:.0e}):  {s0:>14s}  {s1:>16s}")

    nl = [k for k in G if k[1] != "logit"]
    print(f"\n    spearman(CV, LB) with the 3 logit points dropped (n={len(nl)}): "
          f"{spearman([G[k][0] for k in nl], [G[k][1] for k in nl]):+.3f}"
          f"   (all 24: {spearman(cvs, lbs):+.3f})")

    print("\n=== Q3: bias-corrected CV, and what it does to the deadline pick ===")
    # The bias is a property of the CV side, not of the public slice: logit's clip pins
    # the tails of ~29-49 saturating members, and it pins MORE oof rows than test rows
    # because an oof row is one model's output while a test row is a 5-fold average and so
    # is less extreme. That depresses logit's CV relative to its true test score, on ALL
    # test rows. Estimate the depression as logit's gap excess over the non-logit mean.
    base = np.mean([gap[k] for k in G if k[1] in ("hybrid", "rankraw", "rescale")])
    bias = np.mean([gap[k] for k in G if k[1] == "logit"]) - base
    print(f"    non-logit mean gap {base:+.6f}   logit mean gap "
          f"{np.mean([gap[k] for k in G if k[1] == 'logit']):+.6f}")
    print(f"    => logit's CV is depressed by {bias:.6f}")
    print("    ens4 is a rank-average of 4 transforms, one of which is logit, so it")
    print(f"    carries 1/4 of that: {bias / 4:.6f}. h3 excludes logit and carries none.")
    for s in ("158", "159av"):
        cv_e, cv_h = G[(s, "ens4")][0], G[(s, "h3")][0]
        corr = cv_e + bias / 4
        print(f"    {s:>6s}  h3 {cv_h:.6f}   ens4 {cv_e:.6f} -> corrected {corr:.6f}   "
              f"corrected ens4 - h3 {corr - cv_h:+.6f}   "
              f"(LB said {G[(s, 'ens4')][1] - G[(s, 'h3')][1]:+.5f})")

    # The correction was calibrated on the gap column, so re-scoring the gap with it would
    # be circular. The non-circular check is the ORDERING on the fully-crossed 158 set:
    # nothing about spearman was used to build the correction, so if the correction is
    # real it should repair an ordering it was never fitted to.
    w = {"logit": 1.0, "ens4": 0.25, "h3": 0.0, "hybrid": 0.0, "rankraw": 0.0,
         "rescale": 0.0}
    ccv = [G[("158", t)][0] + w[t] * bias for t in six]
    print(f"\n    corrected CV on the 158 set, against LB:")
    for t, c, l in sorted(zip(six, ccv, lb), key=lambda r: -r[1]):
        print(f"      {t:>9s}  {c:.6f}  {l:.5f}")
    print(f"    spearman(CV, LB)           = {spearman(cv, lb):+.3f}")
    print(f"    spearman(corrected CV, LB) = {spearman(ccv, lb):+.3f}")


if __name__ == "__main__":
    main()
