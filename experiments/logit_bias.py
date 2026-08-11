"""Is the drop-logit decision an artefact of a transform-specific CV bias?

The workspace's deadline pick is `blend158_h3` = rank-average of hybrid/rankraw/rescale,
i.e. the four-transform ensemble with `logit` DROPPED. That decision rests on a +5e-6 OOF
gain, confirmed by four instruments (paired 50/50, row bootstrap, 8 resampled fold splits,
exhaustive subset enumeration). Every one of those four instruments is computed on
**out-of-fold predictions**. If OOF is differentially unfair to `logit`, all four share one
bias and they are one measurement, not four.

There is a documented mechanism for exactly that (`agent/stack.py:58-66`): `to_logit`'s
clip pins values at or outside the unit interval onto a constant, and the affected members
put 5.79% of their OOF rows on that plateau against 0.92% of their test rows -- because an
OOF prediction comes from one fold model while a test prediction is averaged over five and
is therefore less extreme. So the `logit` stack is scored on a damaged input at CV time and
applied to a much cleaner one at test time. Its CV should understate its test AUC, while
`rankraw` (which re-ranks both sides identically) should be unbiased.

Two measurements:

1.  **The paired displacement.** For each member set that has both a `logit` and a
    `hybrid` file on the leaderboard, take (LB - CV) for each and difference them. Pairing
    within member set removes everything except the transform. Controls: rankraw and
    rescale differenced against hybrid the same way, since those have no clip asymmetry
    and should show no displacement.

2.  **The clip census at the current member count.** Fraction of OOF vs test cells that
    `to_logit` pins, per member, for the 158-member library actually in use -- so the
    mechanism is confirmed at today's scale rather than quoted from an 86-member note.

    logit_bias.py
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, TARGET, load_raw  # noqa: E402
from stack import load_members  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HONEST_DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac")

# (member set, transform) -> (CV, LB), read off audit_results.csv. Only sets with at least
# two scored transforms are usable, because the pairing is what removes the member effect.
SCORED = {
    ("150fx", "logit"):   (0.969950, 0.97103),
    ("150fx", "hybrid"):  (0.970014, 0.97099),
    ("150fx", "rankraw"): (0.970024, 0.97102),
    ("150fx", "rescale"): (0.970013, 0.97102),
    ("150fx", "ens4"):    (0.970032, 0.97104),
    ("150sx", "logit"):   (0.969955, 0.97104),
    ("150sx", "hybrid"):  (0.970016, 0.97101),
    ("150sx", "rankraw"): (0.970022, 0.97102),
    ("150sx", "ens4"):    (0.970033, 0.97104),
    ("156",   "rankraw"): (0.970036, 0.97104),
    ("156",   "rescale"): (0.970025, 0.97105),
    ("156",   "ens4"):    (0.970042, 0.97106),
    ("151",   "hybrid"):  (0.970024, 0.97099),
    ("151",   "rankraw"): (0.970023, 0.97102),
}


def displacement():
    print("=== 1. paired (LB - CV) displacement, differenced within member set ===")
    gaps = {k: lb - cv for k, (cv, lb) in SCORED.items()}
    sets = sorted({s for s, _ in SCORED})
    print(f"{'set':>6s} " + " ".join(f"{t:>10s}" for t in
                                     ("logit", "hybrid", "rankraw", "rescale", "ens4")))
    for s in sets:
        cells = []
        for t in ("logit", "hybrid", "rankraw", "rescale", "ens4"):
            g = gaps.get((s, t))
            cells.append(f"{g:+10.6f}" if g is not None else f"{'-':>10s}")
        print(f"{s:>6s} " + " ".join(cells))

    print("\nwithin-set differences against hybrid (the reference with a clip repair):")
    print(f"{'contrast':>18s} {'n':>2s}  per-set values                    mean")
    for t in ("logit", "rankraw", "rescale", "ens4"):
        d = [(s, gaps[(s, t)] - gaps[(s, "hybrid")]) for s in sets
             if (s, t) in gaps and (s, "hybrid") in gaps]
        if not d:
            continue
        vals = "  ".join(f"{s}:{v:+.6f}" for s, v in d)
        print(f"{t + ' - hybrid':>18s} {len(d):2d}  {vals:<34s} {np.mean([v for _, v in d]):+.6f}")

    print("\n  Reading: `logit - hybrid` is the contrast with a mechanism. rankraw/rescale")
    print("  are the controls -- no clip asymmetry, so they should sit near zero.")

    # Exact permutation over family labels on the residual, using every scored file.
    df = pd.read_csv(os.path.join(HERE, "cvlb_residuals.csv"))
    lg = df[df.fam == "logit"].resid.to_numpy()
    ot = df[df.fam != "logit"].resid.to_numpy()
    allv = np.concatenate([lg, ot])
    obs = lg.mean() - ot.mean()
    k = len(lg)
    cnt = tot = 0
    for c in itertools.combinations(range(len(allv)), k):
        m = np.zeros(len(allv), bool)
        m[list(c)] = True
        if allv[m].mean() - allv[~m].mean() >= obs - 1e-15:
            cnt += 1
        tot += 1
    print(f"\n  exact permutation over family labels (logit n={k} vs rest n={len(ot)}):")
    print(f"    observed mean residual difference {obs:+.6f}, one-sided p = {cnt}/{tot} "
          f"= {cnt / tot:.4f}")


def clip_census():
    print("\n=== 2. clip census at the current member count ===")
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = (os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2"))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(HONEST_DROP))
    print(f"{len(names)} members, OOF {O.shape}, test {T.shape}")

    # to_logit pins anything at or outside the unit interval.
    po = ((O <= 0) | (O >= 1)).mean(0)
    pt = ((T <= 0) | (T >= 1)).mean(0)
    sd_ratio = T.std(0) / (O.std(0) + 1e-12)
    d = pd.DataFrame(dict(member=names, oof_pinned=po, test_pinned=pt,
                          ratio=po / np.maximum(pt, 1e-9), sd_ratio=sd_ratio))
    aff = d[(d.oof_pinned > 0) | (d.test_pinned > 0)].sort_values("oof_pinned",
                                                                 ascending=False)
    print(f"\n{len(aff)} of {len(names)} members put at least one cell on the clip plateau")
    print(f"{'member':>28s} {'OOF%':>7s} {'test%':>7s} {'OOF/test':>9s} {'sd_t/sd_o':>10s}")
    for _, r in aff.head(25).iterrows():
        print(f"{r['member']:>28s} {r['oof_pinned']:7.3%} {r['test_pinned']:7.3%} "
              f"{r['ratio']:9.2f} {r['sd_ratio']:10.4f}")
    print(f"\n  affected members: OOF pinned {aff.oof_pinned.mean():.3%} mean vs test "
          f"{aff.test_pinned.mean():.3%} mean")
    print(f"  total cells pinned: OOF {po.sum() / len(names):.3%} of all cells, "
          f"test {pt.sum() / len(names):.3%}")
    print("  => the logit stack is FIT and CV-SCORED on the damaged side and PREDICTS on")
    print("     the clean side. rankraw re-ranks both sides identically and cannot do this.")
    d.to_csv(os.path.join(HERE, "clip_census.csv"), index=False)


if __name__ == "__main__":
    displacement()
    if "--census" in sys.argv:
        clip_census()
