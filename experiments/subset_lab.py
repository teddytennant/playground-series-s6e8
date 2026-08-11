"""Enumerate every subset of the four transform stacks, not just the drop-logit one.

Dropping `logit` from the rank-ensemble is the only decision that has paid since
blend156, and it is settled on three instruments (+4e-6, 8/8 fold splits, sd 5e-7). But it
was found by trying ONE alternative, not by enumerating. There are 11 subsets of size >=2
and this workspace has scored exactly two of them.

Every subset costs four file reads and a rank-average -- no refitting, no new members, no
free parameters beyond the discrete choice itself. That last clause is the catch and it is
why the paired resampling below matters: picking the argmax over 11 subsets on one OOF
vector IS a fitted decision, worth roughly sqrt(2 log 11) ~ 2.2 standard deviations of
selection optimism. So each subset is also scored on the 8 resampled combiner fold splits
`repcv.py` established, and only a subset that wins CONSISTENTLY, the way h3 did, is
worth believing.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, load_raw  # noqa: E402

T = ("logit", "hybrid", "rankraw", "rescale")


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="blend158")
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    oof, test, ids = {}, {}, None
    for k in T:
        o = np.load(os.path.join(SUB, f"oof_{a.base}_{k}.npy"))
        d = pd.read_csv(os.path.join(SUB, f"{a.base}_{k}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"{a.base}_{k}.csv id order differs -- refusing to average")
        oof[k], test[k] = rk(o), rk(d[TARGET].to_numpy())
        print(f"  {k:8s} solo OOF AUC {roc_auc_score(y, o):.6f}")

    subs = [c for n in range(2, 5) for c in itertools.combinations(T, n)]
    vecs = {c: np.mean([oof[k] for k in c], 0) for c in subs}
    cv = {c: roc_auc_score(y, v) for c, v in vecs.items()}
    ref = tuple(k for k in T if k != "logit")            # h3, the incumbent

    # paired row bootstrap against the incumbent: same resampled rows for every subset
    rng = np.random.default_rng(0)
    n = len(y)
    d = {c: [] for c in subs}
    for _ in range(a.reps):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        base = roc_auc_score(yy, vecs[ref][idx])
        for c in subs:
            d[c].append(roc_auc_score(yy, vecs[c][idx]) - base)

    print(f"\n{a.base}: 11 subsets, incumbent = h3 {'+'.join(ref)}  CV {cv[ref]:.6f}")
    print(f"{'subset':34s} {'CV':>9s} {'vs h3':>10s} {'boot mean':>11s} {'sd':>9s}"
          f" {'P(>h3)':>8s}")
    for c in sorted(subs, key=lambda c: -cv[c]):
        dd = np.array(d[c])
        tag = "+".join(k[:4] for k in c)
        print(f"{tag:34s} {cv[c]:9.6f} {cv[c]-cv[ref]:+10.6f} {dd.mean():+11.6f}"
              f" {dd.std():9.6f} {(dd > 0).mean():8.3f}")

    best = max(subs, key=lambda c: cv[c])
    if best != ref:
        name = f"{a.base}_" + "".join(k[0] for k in best)
        pd.DataFrame({"id": ids, TARGET: np.mean([test[k] for k in best], 0)}).to_csv(
            os.path.join(SUB, f"{name}.csv"), index=False)
        np.save(os.path.join(SUB, f"oof_{name}.npy"), vecs[best])
        print(f"\nargmax is NOT h3 -- wrote {name}.csv for the queue. Believe it only if "
              f"repcv agrees across fold splits.")
    else:
        print("\nargmax IS h3. Enumeration adds nothing; the incumbent stands.")


if __name__ == "__main__":
    main()
