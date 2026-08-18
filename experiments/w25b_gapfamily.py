"""w25b — is the (LB - CV) gap a CONSTANT, or is it structured by transform family?

Every LB prediction this workspace has ever registered assumes LB ~= CV + gap, with the gap
treated as a common shift that cancels in any paired comparison. w25a's full-history table
makes that assumption checkable for the first time across all six transform families at once,
and it appears to fail: the logit files sit ~80e-6 above the h3 files on gap, which is ~5x the
17.2e-6 paired slice sd that w17a measured as the entire budget for paired disagreement.

If the gap really is family-structured, that is NOT a harmless bookkeeping detail. h3 is the
mix WITHOUT logit and ens4 is the mix WITH it, both WANTED files are h3, and a family gap
would mean the public LB's persistent preference for ens4 has a mechanism rather than being
the slice draw w16s closed it as.

Two questions, and the second is the one with teeth:

  Q1  Group the 58 scored files by transform family. Is between-family gap variance large
      against the within-family spread?

  Q2  MATCHED PAIRS ONLY. For every member set where both the h3 mix and the ens4 mix of the
      SAME set are scored, record sign(dCV) and sign(dLB). Slice noise is symmetric, so under
      the null of no mechanism the LB sign is a coin flip per pair. Count them.

The honest limit, stated before the numbers: a public-LB-only analysis cannot separate "the
public slice happens to like logit" from "the test set likes logit". Both produce this table.
What it CAN do is price the first explanation, and 10 identical coin flips prices badly.

    .venv/bin/python experiments/w25b_gapfamily.py
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

t = pd.read_csv("experiments/w25a_cvlb_full.csv").dropna(subset=["cv"]).copy()


def family(s: str) -> str:
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit"):
        if s.endswith("_" + f):
            return f
    if s.endswith("_w") or s.endswith("_w2") or s in ("blend156w", "blend156w2"):
        return "w"
    return "ens4"          # a bare stem is the rank-average of all four transforms


def memberset(s: str) -> str:
    """Strip the trailing transform tag; what is left identifies the member set + build."""
    for f in ("_wh3", "_h3", "_hybrid", "_rankraw", "_rescale", "_logit", "_w2", "_w"):
        if s.endswith(f):
            return s[: -len(f)]
    return s


t["fam"] = t.stem.map(family)
t["set"] = t.stem.map(memberset)
t["gap6"] = t.gap * 1e6

print("=== Q1: (LB - CV) gap by transform family, e-6 ===")
q1 = t.groupby("fam").agg(n=("gap6", "size"), mean=("gap6", "mean"), sd=("gap6", "std"),
                          lo=("gap6", "min"), hi=("gap6", "max"), cv_mean=("cv", "mean"))
print(q1.sort_values("mean", ascending=False).round(2).to_string())

# The gap is confounded with CV level (low-CV files have big gaps: see the pub74/86/88 stacks),
# so re-read the family effect as a residual from the CV->LB regression on CV >= 0.97, which
# is where every decision-relevant file lives.
h = t[t.cv >= 0.97].copy()
sl, ic = np.polyfit(h.cv, h.lb, 1)
h["resid6"] = (h.lb - (sl * h.cv + ic)) * 1e6
print(f"\n=== Q1b: residual from the CV>=0.97 OLS line (slope {sl:.3f}), by family, e-6 ===")
q1b = h.groupby("fam").agg(n=("resid6", "size"), mean=("resid6", "mean"), sd=("resid6", "std"))
q1b["se"] = q1b["sd"] / np.sqrt(q1b["n"])
print(q1b.sort_values("mean", ascending=False).round(2).to_string())

print("\n=== Q2: MATCHED h3-vs-ens4 pairs (same member set, both mixes scored) ===")
piv = t.pivot_table(index="set", columns="fam", values=["cv", "lb"], aggfunc="first")
rows = []
for s in piv.index:
    try:
        cv_h, cv_e = piv[("cv", "h3")][s], piv[("cv", "ens4")][s]
        lb_h, lb_e = piv[("lb", "h3")][s], piv[("lb", "ens4")][s]
    except KeyError:
        continue
    if any(pd.isna(v) for v in (cv_h, cv_e, lb_h, lb_e)):
        continue
    rows.append(dict(set=s, cv_h3=cv_h, cv_ens4=cv_e, dcv6=(cv_h - cv_e) * 1e6,
                     lb_h3=lb_h, lb_ens4=lb_e, dlb6=(lb_h - lb_e) * 1e6))
p = pd.DataFrame(rows).sort_values("cv_h3", ascending=False)
print(p.assign(cv_h3=lambda d: d.cv_h3.map("{:.7f}".format),
               cv_ens4=lambda d: d.cv_ens4.map("{:.7f}".format),
               dcv6=lambda d: d.dcv6.map("{:+.2f}".format),
               dlb6=lambda d: d.dlb6.map("{:+.1f}".format)).to_string(index=False))

n = len(p)
cv_h3_wins = int((p.dcv6 > 0).sum())
lb_h3_wins = int((p.dlb6 > 0).sum())
lb_ties = int((p.dlb6 == 0).sum())
lb_e_wins = n - lb_h3_wins - lb_ties
print(f"\npairs {n}   CV prefers h3 in {cv_h3_wins}/{n}   "
      f"LB prefers h3 in {lb_h3_wins}/{n}, ties {lb_ties}, prefers ens4 in {lb_e_wins}/{n}")
print(f"mean dCV {p.dcv6.mean():+.2f}e-6   mean dLB {p.dlb6.mean():+.1f}e-6")
# Sign test on the LB direction, treating pairs as independent -- which they are NOT (see the
# caveat below). This is an UPPER bound on how surprising the streak is, not a p-value.
from scipy.stats import binomtest
bt = binomtest(lb_e_wins, n, 0.5)
print(f"sign test on LB direction (INDEPENDENCE ASSUMED, and it is false): p {bt.pvalue:.2e}")
print("\nCAVEAT, and it is the whole ballgame: these pairs share member sets and are all read")
print("off the SAME fixed public slice, so they are not independent draws. One slice-level")
print("quirk in the logit component produces every row. The sign test above is the p-value")
print("under an independence null that does not hold; the true p is larger and unknown.")

json.dump(dict(n_pairs=n, cv_h3_wins=cv_h3_wins, lb_ens4_wins=lb_e_wins, lb_ties=lb_ties,
               mean_dcv=float(p.dcv6.mean()), mean_dlb=float(p.dlb6.mean()),
               signtest_p_indep=float(bt.pvalue),
               fam_resid=q1b["mean"].to_dict()),
          open("experiments/w25b_gapfamily.json", "w"), indent=1)
p.to_csv("experiments/w25b_pairs.csv", index=False)
t.to_csv("experiments/w25b_families.csv", index=False)
print("\nwrote experiments/w25b_{pairs,families}.csv + .json")
