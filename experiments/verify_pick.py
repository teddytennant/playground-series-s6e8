"""Prove the deadline-pick CSVs are what their names say, from the files on disk.

`audit.py` checks that every candidate has the right ids, is finite, and is a distinct
ranking. That catches a corrupt file. It does NOT catch a **mislabelled** one -- a
`blend159av.csv` that is actually a stale build, or an `_h3` that quietly still contains
logit. The whole deadline argument is a comparison between two file names, so the names
being accurate is load-bearing, and nothing was checking it.

There is an exact check available for free. `blend_lab.build` writes the ensemble as

    ens4 = mean of rk(stack_k) over k in (logit, hybrid, rankraw, rescale)

and `make_h3` writes the same average over (hybrid, rankraw, rescale) only. Every one of
those single-transform stacks is itself written out as a CSV. So each composite file can be
**recomputed from its own parts** and compared. AUC reads only the ordering, so the test is
on ranks, not floats: spearman must be 1.0 to floating-point noise.

Two failures this is built to catch, in decreasing order of nastiness:

  - a composite that does not match its parts (stale build, wrong member set, wrong name);
  - an `_h3` that matches the FOUR-transform average better than the three-transform one,
    i.e. logit is still in it -- which would make the entire ens4-vs-h3 contrast, the sole
    basis of the deadline pick, a comparison of a file against itself.

    verify_pick.py                     # every set that has a composite on disk
    verify_pick.py blend159av          # one set, verbosely
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import SUB, TARGET  # noqa: E402

KINDS4 = ("logit", "hybrid", "rankraw", "rescale")
KINDS3 = ("hybrid", "rankraw", "rescale")
TOL = 1e-9


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def load(name):
    p = os.path.join(SUB, f"{name}.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def check_set(base, verbose=False):
    """Return a row of results for one member set, or None if it has no composite."""
    ens = load(base)
    h3 = load(f"{base}_h3")
    if ens is None and h3 is None:
        return None

    parts = {k: load(f"{base}_{k}") for k in KINDS4}
    missing = [k for k, v in parts.items() if v is None]
    if missing:
        return dict(set=base, note=f"missing parts: {','.join(missing)}")

    ids = None
    for k, df in parts.items():
        if ids is None:
            ids = df["id"].to_numpy()
        elif not np.array_equal(ids, df["id"].to_numpy()):
            return dict(set=base, note=f"id order differs in {k}")

    r = {k: rk(parts[k][TARGET].to_numpy()) for k in KINDS4}
    rebuilt4 = np.mean([r[k] for k in KINDS4], 0)
    rebuilt3 = np.mean([r[k] for k in KINDS3], 0)

    out = dict(set=base)
    for label, got, want4, want3 in (("ens4", ens, rebuilt4, rebuilt3),
                                     ("h3", h3, rebuilt3, rebuilt4)):
        if got is None:
            out[label] = np.nan
            continue
        if not np.array_equal(got["id"].to_numpy(), ids):
            out[label] = np.nan
            out[f"{label}_note"] = "id order differs from parts"
            continue
        v = got[TARGET].to_numpy()
        s_own = spearman(v, want4)
        s_other = spearman(v, want3)
        out[label] = s_own
        out[f"{label}_alt"] = s_other
        if verbose:
            print(f"  {base}_{label}: vs own recipe {s_own:.12f}   "
                  f"vs the other recipe {s_other:.12f}")
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    bases = sorted({os.path.basename(p)[:-4].split("_")[0]
                    for p in glob.glob(os.path.join(SUB, "blend*_logit.csv"))})
    if only:
        bases = [b for b in bases if b == only] or [only]

    rows = []
    for b in bases:
        r = check_set(b, verbose=bool(only))
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit("no member set on disk has both a composite and its four parts")

    print(f"=== composite vs its own parts, {len(rows)} member sets ===")
    print("spearman(file, recomputed). 1.000000 = the name is accurate.\n")
    print(f"{'set':14s} {'ens4 vs 4-avg':>14s} {'(vs 3-avg)':>12s} "
          f"{'h3 vs 3-avg':>13s} {'(vs 4-avg)':>12s}")

    bad = []
    for r in rows:
        if "note" in r:
            print(f"{r['set']:14s}  -- {r['note']}")
            continue
        f = lambda x: "     -" if x is None or (isinstance(x, float) and np.isnan(x)) \
            else f"{x:.6f}"
        print(f"{r['set']:14s} {f(r.get('ens4')):>14s} {f(r.get('ens4_alt')):>12s} "
              f"{f(r.get('h3')):>13s} {f(r.get('h3_alt')):>12s}")
        for label in ("ens4", "h3"):
            own, alt = r.get(label), r.get(f"{label}_alt")
            if own is None or (isinstance(own, float) and np.isnan(own)):
                continue
            if own < 1 - 1e-6:
                bad.append(f"{r['set']}_{label}: only {own:.6f} against its own recipe")
            if alt is not None and alt >= own:
                bad.append(f"{r['set']}_{label}: matches the OTHER recipe at least as "
                           f"well ({alt:.6f} >= {own:.6f}) -- name may be wrong")

    print()
    if bad:
        print("!! FAILURES")
        for b in bad:
            print(f"  {b}")
        raise SystemExit(1)
    print("every composite on disk reproduces exactly from its own single-transform "
          "parts, and matches its own recipe strictly better than the other one.")
    print("the ens4-vs-h3 contrast is therefore a real contrast: h3 genuinely excludes "
          "logit and ens4 genuinely includes it.")


if __name__ == "__main__":
    main()
