"""Build the population w15a_crossteam had no members of: pairs of solutions that are
MATCHED IN QUALITY and GENUINELY DIVERSE.

The audit in w17h needs to separate "the partner is a different team" from "the partner is
worse".  Every pair w15a measured confounded the two: its three foreign partners were 193,
549 and 577e-6 below our file on CV.  The teams the resulting 53-84e-6 floor gets applied to
sit at deficit ~0.

A rank-average over a member subset is quality-mismatched and weak.  The right object is
what another team would actually have built from the same public pool: a cross-fitted
logistic stack over a DISJOINT half of the members, on the frozen SKF5 seed42 folds.  Two
such stacks share no member at all, so neither is a perturbation of the other, and both land
near full stack quality because the pack is collinear (RESEARCH: median pairwise 0.98).

Writes experiments/w17i_syn_<tag>.npy, which w17h_floorpop.py picks up automatically.

    w17i_disjoint.py --seeds 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))
sys.path.insert(0, HERE)
from common import ROOT  # noqa: E402
from stack_lab import build, crossfit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--transform", default="hybrid")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()

    names, Z, Zt, y = build(a.transform, drop=[], refresh=False)
    k = len(names)
    print(f"{k} members on the '{a.transform}' scale | {Z.shape[0]:,} rows", flush=True)
    del Zt

    # resumable: this box kills long background jobs, so each half is checkpointed to disk
    # and a re-run picks up where the last one stopped.
    cache = os.path.join(HERE, "w17i_full_auc.json")
    if os.path.exists(cache):
        full_auc = json.load(open(cache))["full_auc"]
        print(f"FULL {k}-member stack (cached): {full_auc:.8f}\n", flush=True)
    else:
        full_auc, _ = crossfit(Z, y, a.C)
        json.dump({"full_auc": float(full_auc)}, open(cache, "w"))
        print(f"FULL {k}-member stack, cross-fitted on the frozen folds: {full_auc:.8f}\n",
              flush=True)

    out = {"transform": a.transform, "C": a.C, "n_members": k,
           "full_auc": float(full_auc), "pairs": []}
    for s in range(a.seeds):
        rng = np.random.default_rng(1700 + s)
        perm = rng.permutation(k)
        ia, ib = np.sort(perm[: k // 2]), np.sort(perm[k // 2:])
        assert not set(ia.tolist()) & set(ib.tolist()), "halves must be disjoint"
        rec = {"seed": 1700 + s, "n_a": len(ia), "n_b": len(ib)}
        for tag, idx in [("a", ia), ("b", ib)]:
            fp = os.path.join(HERE, f"w17i_syn_s{s}{tag}.npy")
            if os.path.exists(fp):
                from sklearn.metrics import roc_auc_score as _ras
                auc = float(_ras(y, np.load(fp)))
                rec[f"auc_{tag}"] = auc
                print(f"  s{s}{tag}: cached -> {auc:.8f}", flush=True)
                continue
            t0 = time.time()
            auc, mo = crossfit(Z, y, a.C, idx=idx)
            np.save(fp, mo)
            rec[f"auc_{tag}"] = float(auc)
            print(f"  s{s}{tag}: {len(idx)} members -> {auc:.8f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rec["deficit"] = abs(rec["auc_a"] - rec["auc_b"])
        rec["deficit_vs_full"] = full_auc - 0.5 * (rec["auc_a"] + rec["auc_b"])
        print(f"  s{s}: deficit between halves {rec['deficit']*1e6:.1f}e-6 | "
              f"mean half is {rec['deficit_vs_full']*1e6:.1f}e-6 below the full stack\n",
              flush=True)
        out["pairs"].append(rec)

    json.dump(out, open(os.path.join(HERE, "w17i_disjoint.json"), "w"), indent=1)
    print("wrote w17i_disjoint.json")


if __name__ == "__main__":
    main()
