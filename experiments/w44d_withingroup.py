#!/usr/bin/env python3
"""
w44d -- WHY the import line went flat: the gate never measured within-group redundancy.

The w40d/w42b import rule is `maxcorr < 0.99 AND solo > 0.966319`, where maxcorr is measured
against members ALREADY HELD in the pack. It is computed one incoming stream at a time and is
structurally blind to how much the incoming streams duplicate EACH OTHER.

ARM 211 added 9 members that all cleared it -- and all 9 come from ONE author (y94/yadoy666),
so they plausibly share preprocessing, folds and feature code. If they are mutually redundant,
9 gated members carry far less than 9 members of information, and the +0.16e-6 result needs no
further explanation.

Prediction, stated before the numbers are computed: median WITHIN-group |rho| for ext_members15
is HIGHER than the gate's 0.99 ceiling on against-pack maxcorr, and higher than for
ext_members16 (6 members spread over 3 authors).
"""
import numpy as np, pathlib, itertools
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parent.parent

def load(d):
    out = {}
    for p in sorted((ROOT / "data" / d).glob("oof_*.npy")):
        out[p.stem[4:]] = np.load(p)
    return out

def spear(a, b):
    return np.corrcoef(rankdata(a), rankdata(b))[0, 1]

print("=" * 74)
print("w44d  WITHIN-GROUP redundancy of the gated import groups")
print("=" * 74)

summary = {}
for d in ["ext_members15", "ext_members16", "ext_members14", "ext_members11", "ext_members12"]:
    m = load(d)
    if len(m) < 2:
        continue
    names = list(m)
    R = {}
    for a, b in itertools.combinations(names, 2):
        R[(a, b)] = spear(m[a], m[b])
    v = np.array(list(R.values()))
    summary[d] = (len(names), np.median(v), v.min(), v.max())
    print(f"\n--- {d}   n={len(names)}   authors={len(set(x.split('_')[0] for x in names))}")
    print(f"    within-group spearman: median {np.median(v):.5f}  "
          f"min {v.min():.5f}  max {v.max():.5f}")
    print(f"    pairs above the gate's 0.99 ceiling: "
          f"{(v > 0.99).sum()} / {len(v)}  ({(v>0.99).mean():.0%})")
    # effective number of independent members: participation ratio of the corr-matrix eigenvalues
    M = np.eye(len(names))
    for (a, b), r in R.items():
        i, j = names.index(a), names.index(b)
        M[i, j] = M[j, i] = r
    ev = np.linalg.eigvalsh(M)
    ev = ev[ev > 0]
    n_eff = ev.sum() ** 2 / (ev ** 2).sum()
    print(f"    EFFECTIVE independent members (participation ratio): "
          f"{n_eff:.2f} of {len(names)}   ({n_eff/len(names):.0%})")
    summary[d] = summary[d] + (n_eff,)

print("\n" + "=" * 74)
print("THE DOSE WAS NEVER WHAT THE MEMBER COUNT SAID")
print("=" * 74)
print(f"\n{'group':<16}{'n':>3}{'median rho':>12}{'n_eff':>8}{'delta e-6':>11}{'per n_eff':>11}")
DELTA = {"ext_members11": 10.98, "ext_members12": 3.87,
         "ext_members14": -2.41, "ext_members15": 0.16}
for d, s in summary.items():
    n, med, _, _, neff = s
    dd = DELTA.get(d)
    ds = f"{dd:+.2f}" if dd is not None else "pending"
    pn = f"{dd/neff:+.2f}" if dd is not None else "-"
    print(f"{d:<16}{n:>3}{med:>12.5f}{neff:>8.2f}{ds:>11}{pn:>11}")
