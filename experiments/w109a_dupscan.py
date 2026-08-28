"""Does the shipped pack carry duplicate columns, and does carrying them cost anything?

`agent/stack.py:load_members` enrols every `oof_*/test_*` pair it finds, and the drop lists
name only members that were retired for provenance. Nothing has ever dropped a member for
being a function of other members already in the pack -- so `xgb_latcat_avg3`, which is
EXACTLY the probability mean of `xgb_latcat`, `_s17` and `_s23`, ships alongside all three.
The +2e-6 "1.4% pass-through" that closes ANGLE INDEX row 7's member arm was measured in a
configuration that dropped the seeds and kept the average; that is not what ships.

Registered in experiments/w109_prereg.txt before any number here existed.

    w109a_dupscan.py --scan          # Q1 only: the census. Minutes.
    w109a_dupscan.py --scan --arms   # Q1 + Q2: three cross-fitted arms. Under an hour.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_lab as BL  # noqa: E402  -- pins the BLAS thread count on import

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, TARGET, get_folds, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402

# experiments/w36b_run.sh verbatim -- the recipe that produced the deadline pick.
DROP = ("golem_a,golem_f,lgbm_tuned_lat,lgbm_tuned_lat_frac,"
        "lat_ctraw_r400,lat_ctfixte_r400,om_cat")
# load_all() prepends ext_members{,2} to whatever --extra-dirs names, and w36b_run.sh
# went through load_all -- so the shipped pack is those two plus the eight below.
BASEDIRS = ("ext_members", "ext_members2")
XDIRS = ("ext_members3,ext_members4,ext_members6,ext_members7pin,"
         "ext_members8,ext_members10,ext_members11,ext_members12")
EXPECT = 199
KINDS = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")
SEEDS = ("xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23")
AVG = "xgb_latcat_avg3"
BOLT = "bolt_xgb_d7_alt2"     # byte-identical to bolt_xgb_d7_alt1, oof AND test
# A/B/C are the arms w109_prereg.txt registered and the P-rules score. D/E are EXPLORATORY,
# added after Q1's census -- which the prereg registered as descriptive with no decision
# attached -- turned up an exact duplicate pair the latcat question knew nothing about.
# They are reported, never scored against a bar written after the fact.
ARMS = {"A_shipped": (), "B_seeds_out": SEEDS, "C_avg_out": (AVG,),
        "D_bolt_dedup": (BOLT,), "E_clean": (BOLT, AVG)}
REGISTERED = ("B_seeds_out", "C_avg_out")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w109a_dupscan.json")


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def census(names, O):
    """Q1: which columns are near-duplicates, and is any one a function of the others?"""
    n = len(names)
    S = (O - O.mean(0)) / O.std(0)
    R = (S.T @ S) / len(O)
    np.fill_diagonal(R, 0.0)

    iu, ju = np.triu_indices(n, 1)
    r = R[iu, ju]
    pairs = [(float(r[k]), names[iu[k]], names[ju[k]])
             for k in np.argsort(-r) if r[k] >= 0.9995]
    print(f"\nQ1 CENSUS -- {n} members, {n*(n-1)//2:,} pairs")
    print(f"  pairs at r >= 0.9995: {len(pairs)}")
    for v, a, b in pairs:
        print(f"    {v:.6f}  {a:28s} {b}")

    np.fill_diagonal(R, 1.0)
    ev, V = np.linalg.eigh(R)
    print(f"  correlation-matrix eigenvalues: max {ev[-1]:.4f}, "
          f"min {ev[0]:.3e}, {int((ev < 1e-3).sum())} below 1e-3")
    tight = []
    for k in range(min(4, n)):
        if ev[k] >= 1e-2:
            break
        w = V[:, k]
        top = np.argsort(-np.abs(w))[:6]
        tight.append({"eigenvalue": float(ev[k]),
                      "loadings": [[names[i], float(w[i])] for i in top]})
        print(f"  near-dependency, eigenvalue {ev[k]:.3e}:")
        for i in top:
            print(f"      {w[i]:+.4f}  {names[i]}")
    return {"n_members": n,
            "pairs_ge_0.9995": [[v, a, b] for v, a, b in pairs],
            "eig_min": float(ev[0]), "eig_max": float(ev[-1]),
            "n_eig_below_1e-3": int((ev < 1e-3).sum()),
            "near_dependencies": tight}


def arm(label, keep, names, y, mats, folds):
    """Cross-fit every transform on the frozen folds for one column subset."""
    t0 = time.time()
    per = {}
    for k in KINDS:
        Z, Zt = mats[k]
        Z = Z[:, keep]
        s = Z.std(0)
        s[s <= 0] = 1.0                      # --standardize, scale from the OOF side
        Z = (Z / s).astype(Z.dtype)
        mo = np.zeros(len(y))
        for itr, iva in folds:
            mo[iva] = (LogisticRegression(max_iter=5000, C=1.0)
                       .fit(Z[itr], y[itr]).decision_function(Z[iva]))
        per[k] = mo
        print(f"  [{label}] stack_{k:8s} {roc_auc_score(y, mo):.10f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    h3 = roc_auc_score(y, np.mean([rk(per[k]) for k in H3], 0))
    ens4 = roc_auc_score(y, np.mean([rk(per[k]) for k in KINDS], 0))
    print(f"  [{label}] n={len(keep)}  h3 {h3:.10f}  ens4 {ens4:.10f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    return {"n": len(keep), "h3": h3, "ens4": ens4,
            "per_transform": {k: roc_auc_score(y, v) for k, v in per.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--arms", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = tuple(os.path.join(DATA, d)
                  for d in BASEDIRS + tuple(XDIRS.split(",")))
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(DROP.split(",")))
    if len(names) != EXPECT:
        raise SystemExit(f"pack is {len(names)} members, expected {EXPECT} -- "
                         "this is not the shipped recipe, refusing to report")
    print(f"{len(names)} members loaded in {time.time()-t0:.0f}s")
    for m in SEEDS + (AVG, BOLT):
        if m not in names:
            raise SystemExit(f"{m} absent from the pack -- the premise is wrong")

    res = {"pack": EXPECT, "drop": DROP, "extra_dirs": XDIRS}
    if a.scan:
        res["census"] = census(names, O.astype(np.float64))
    if not a.arms:
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\nwrote {OUT}  [{time.time()-t0:.0f}s total]")
        return

    mats = {}
    for k in KINDS:
        Z, Zt = transform(O, T, k)
        mats[k] = (Z.astype(np.float32), Zt.astype(np.float32))
    del O, T
    print(f"transforms built [{time.time()-t0:.0f}s]", flush=True)

    folds = get_folds(y)
    idx = {m: i for i, m in enumerate(names)}
    res["arms"] = {}
    for label, drop in ARMS.items():
        keep = np.array([i for m, i in idx.items() if m not in drop])
        keep.sort()
        res["arms"][label] = arm(label, keep, names, y, mats, folds)
        json.dump(res, open(OUT, "w"), indent=1)

    base = res["arms"]["A_shipped"]
    print("\nQ2 -- vs A_shipped, primary h3, secondary ens4")
    for label, r in res["arms"].items():
        if label == "A_shipped":
            continue
        dh, de = (r["h3"] - base["h3"]) * 1e6, (r["ens4"] - base["ens4"]) * 1e6
        agree = "agree" if dh * de > 0 else "DISAGREE -> P3 null"
        tag = "registered" if label in REGISTERED else "EXPLORATORY"
        print(f"  {label:12s} n={r['n']}  h3 {dh:+8.3f}e-6   ens4 {de:+8.3f}e-6   "
              f"[{agree}] [{tag}]")

    dh = {k: (v["h3"] - base["h3"]) * 1e6 for k, v in res["arms"].items() if k != "A_shipped"}
    de = {k: (v["ens4"] - base["ens4"]) * 1e6 for k, v in res["arms"].items() if k != "A_shipped"}
    # the P-rules score ONLY the registered arms; D/E cannot make or break P1.
    p1 = all(abs(dh[k]) < 4.0 for k in REGISTERED)
    cand = [k for k in REGISTERED if dh[k] > 4.0 and dh[k] * de[k] > 0]
    res["verdict"] = {"P1_null": bool(p1), "P2_candidates": cand,
                      "registered_arms": list(REGISTERED),
                      "d_h3_e6": dh, "d_ens4_e6": de}
    print(f"\nP1 null (both REGISTERED arms within +-4.0e-6 on h3): {p1}")
    print(f"P2 candidates: {cand or 'none'}")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}  [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
