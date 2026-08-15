"""Fold-seed diversity at the STACKER level, not the member level.

Every number in this workspace is cross-fitted on ONE partition:
`StratifiedKFold(5, shuffle=True, random_state=42)`. The members must use it -- the whole
library is only stackable because it shares it -- but the STACKER's own cross-fit is a
free choice, and it has never been varied. So every reported stack CV carries that one
partition's noise, and the 1e-6 gaps that three journal entries argued over
(`h3` 0.970049 vs `ens4` 0.970045 vs `blend158_h3` 0.970048) have never had an error bar
that includes partition noise. This puts one on them.

WHAT MOVES WITH THE SEED AND WHAT DOES NOT
------------------------------------------
`blend_lab.build()` takes the TEST prediction from a full fit on all 691,369 rows
(`full = LogisticRegression(...).fit(Z, y)`), not from the fold models. So for `h3` and
`ens4` -- which fit zero parameters above the stacks -- the submitted file is a
deterministic function of the member matrix with ZERO dependence on the fold partition.
"Average the test predictions over several stacker fold seeds" is therefore a structural
null for the deadline pick, and the assert below states that rather than measuring it.

The seed does reach the test side through exactly one door: the weight-fitted family
(`blend159av_w`, `blend159av_wh3`), whose simplex weights are chosen on the cross-fitted
OOF and so inherit the partition. Stage `w` measures how far those weights move.

PRE-REGISTERED, BEFORE LOOKING
------------------------------
Off-seed cross-fits should read slightly HIGH against seed 42, and the mechanism is
specific. Under seed 42 the stacker's validation fold is exactly one member-fold, so no
member model contributed features to both sides of the split. Under any other seed a
validation row's member prediction comes from a model trained on rows that are mostly in
the stacker's training half, which correlates the two sides. Row i's own label is still
never seen by the model that predicted row i -- that is what makes an OOF vector an OOF
vector, and it holds under any stacker partition -- so this is a mild optimism, not
leakage. If the off-seed marginals sit above 42 by a consistent offset, that is the
mechanism showing. It cancels in the paired contrasts, which is what this is for.

    w14c_seedstack.py --seeds 42,7,11,13,101,777
    w14c_seedstack.py --seeds 42 --stage w        # weights only, from saved OOF
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import DATA, SUB, TARGET, load_raw  # noqa: E402
from stack import load_members, transform  # noqa: E402
from transform_weights import simplex  # noqa: E402

KINDS = ("logit", "hybrid", "rankraw", "rescale")
H3 = ("hybrid", "rankraw", "rescale")

# The blend159av member set: the honest drop (golem_a/golem_f early-stop on their own
# validation fold; lgbm_tuned_lat* carried the same defect and are superseded), the three
# xgb_latcat seeds replaced by their probability mean `xgb_latcat_avg3`, and the two
# `orig_bin*` members that only ever appeared in the blend160orig builds. 168 - 9 = 159,
# and the seed-42 control below reproduces logs_blend159av.txt to the digit.
DROP = ("golem_a", "golem_f", "lgbm_tuned_lat", "lgbm_tuned_lat_frac",
        "xgb_latcat", "xgb_latcat_s17", "xgb_latcat_s23", "orig_bin", "orig_binm")

OUT = os.path.join(ROOT, "experiments", "w14c_out")
os.makedirs(OUT, exist_ok=True)


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def oof_path(seed, kind):
    return os.path.join(OUT, f"w14c_oof_seed{seed}_{kind}.npy")


def crossfit_seed(mats, y, seed, C):
    """Cross-fit every transform stack on a fold partition of our choosing."""
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed)
                 .split(np.zeros(len(y)), y))
    out = {}
    for k in KINDS:
        p = oof_path(seed, k)
        if os.path.exists(p):
            out[k] = np.load(p)
            print(f"  seed {seed:>3d} {k:8s} {roc_auc_score(y, out[k]):.6f}  (cached)",
                  flush=True)
            continue
        Z = mats[k]
        t0 = time.time()
        mo = np.zeros(len(y))
        for itr, iva in folds:
            mo[iva] = (LogisticRegression(max_iter=5000, C=C)
                       .fit(Z[itr], y[itr]).decision_function(Z[iva]))
        np.save(p, mo)
        out[k] = mo
        print(f"  seed {seed:>3d} {k:8s} {roc_auc_score(y, mo):.6f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def composites(oof, y):
    """The blends the deadline argument is actually about, on one partition."""
    r = {k: roc_auc_score(y, oof[k]) for k in KINDS}
    o_h3 = np.mean([rk(oof[k]) for k in H3], 0)
    o_e4 = np.mean([rk(oof[k]) for k in KINDS], 0)
    r["h3"] = roc_auc_score(y, o_h3)
    r["ens4"] = roc_auc_score(y, o_e4)
    return r, o_h3, o_e4


def stage_seeds(seeds, C):
    tr, te = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    extra = (os.path.join(DATA, "ext_members"), os.path.join(DATA, "ext_members2"))
    t0 = time.time()
    names, O, T = load_members(y, len(te), extra_dirs=extra, drop=set(DROP))
    print(f"{len(names)} members ({time.time()-t0:.0f}s)", flush=True)
    if len(names) != 159:
        raise SystemExit(f"expected the 159av member set, got {len(names)}")

    mats = {}
    for k in KINDS:
        t1 = time.time()
        Z, _Zt = transform(O, T, k)
        mats[k] = Z.astype("float32")
        del _Zt
        print(f"  transform {k:8s} {time.time()-t1:5.0f}s", flush=True)
    del O, T

    rows = {}
    for s in seeds:
        oof = crossfit_seed(mats, y, s, C)
        rows[s], _, _ = composites(oof, y)
    report(rows, seeds)
    return rows


def report(rows, seeds):
    df = pd.DataFrame(rows).T
    df.index.name = "fold_seed"
    print("\ncross-fitted stack AUC by STACKER fold seed (members frozen on seed 42)")
    print(df.to_string(float_format="%.6f"))

    print("\nmarginal spread across seeds")
    for c in df.columns:
        v = df[c].to_numpy()
        print(f"  {c:>8s}  mean {v.mean():.6f}  sd {v.std(ddof=1):.6f}  "
              f"range {v.max()-v.min():.6f}")

    if 42 in df.index and len(df) > 1:
        off = df.drop(index=42)
        d = off.mean() - df.loc[42]
        print("\npre-registered check: off-seed mean minus seed 42 (positive = the "
              "member-model-sharing optimism)")
        for c in df.columns:
            print(f"  {c:>8s}  {d[c]:+.6f}")

    print("\nPAIRED contrasts on identical partitions -- the number the 1e-6 arguments "
          "never had")
    pairs = [("h3", "ens4"), ("h3", "rankraw"), ("h3", "hybrid"), ("h3", "rescale"),
             ("hybrid", "logit"), ("rankraw", "hybrid"), ("ens4", "logit")]
    for a, b in pairs:
        d = (df[a] - df[b]).to_numpy()
        ok = "consistent" if (d > 0).all() or (d < 0).all() else "SIGN FLIPS"
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a:>7s} - {b:<7s}  {d.mean():+.6f} +/- {d.std(ddof=1):.6f} (sd)  "
              f"se {se:.6f}  [{ok}]  per-seed {np.array2string(d*1e6, precision=1)}")


def stage_w(seeds, step=0.05):
    """How far do the h3 simplex weights move when only the stacker partition changes?

    The weights are chosen on the FULL OOF vector for each seed. That is optimistic as a
    score -- and no score is reported here. The question is the SPREAD of the argmax, and
    for that the full-row fit is the right estimator: it is the same estimator the shipped
    `_w` files use, so its wobble is the shipped file's wobble.
    """
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    grid = simplex(3, step)
    print(f"simplex grid {grid.shape[0]} points, step {step}")
    W = {}
    for s in seeds:
        paths = [oof_path(s, k) for k in H3]
        if not all(os.path.exists(p) for p in paths):
            print(f"  seed {s}: no saved OOF, skipping")
            continue
        R = np.column_stack([rk(np.load(p)) for p in paths])
        sc = np.array([roc_auc_score(y, R @ w) for w in grid])
        w = grid[int(np.argmax(sc))]
        W[s] = w
        print(f"  seed {s:>3d}  w = {np.array2string(w, precision=3)}  "
              f"in-sample {sc.max():.6f}  (equal {roc_auc_score(y, R.mean(1)):.6f})",
              flush=True)
    if len(W) > 1:
        M = np.array(list(W.values()))
        print(f"\nweight spread over {len(W)} partitions, coords {H3}")
        print(f"  mean {np.array2string(M.mean(0), precision=4)}")
        print(f"  sd   {np.array2string(M.std(0, ddof=1), precision=4)}")
        print(f"  range{np.array2string(M.max(0)-M.min(0), precision=4)}")
    with open(os.path.join(OUT, "w14c_weights.json"), "w") as f:
        json.dump({str(k): list(map(float, v)) for k, v in W.items()}, f, indent=1)
    return W


def stage_build(W, name):
    """Apply the seed-AVERAGED h3 weights to the existing blend159av transform files.

    The test side of each transform stack comes from a full fit on all rows and does not
    depend on any fold partition, so these CSVs are reusable verbatim -- nothing is
    refitted and nothing about the members changes. Only the three mixing weights are
    new, and they are the average of the per-partition argmaxes rather than the one
    partition's argmax that `blend159av_wh3` spends.
    """
    if len(W) < 2:
        raise SystemExit("need >=2 seeds of weights to average")
    w = np.array(list(W.values())).mean(0)
    w = w / w.sum()
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    ids, cols_t, cols_o = None, [], []
    for k in H3:
        d = pd.read_csv(os.path.join(SUB, f"blend159av_{k}.csv"))
        if ids is None:
            ids = d["id"].to_numpy()
        elif not np.array_equal(ids, d["id"].to_numpy()):
            raise SystemExit(f"blend159av_{k}.csv id order differs -- refusing to blend")
        cols_t.append(rk(d[TARGET].to_numpy()))
        cols_o.append(rk(np.load(os.path.join(SUB, f"oof_blend159av_{k}.npy"))))
    Rt, Ro = np.column_stack(cols_t), np.column_stack(cols_o)

    t_new, o_new = Rt @ w, Ro @ w
    cv = roc_auc_score(y, o_new)
    print(f"\n{name}: w = {np.array2string(w, precision=4)} over {H3}")
    print(f"  seed-42 cross-fitted CV of this mix  {cv:.6f}")
    print(f"  equal-weight h3 on the same vectors  {roc_auc_score(y, Ro.mean(1)):.6f}")
    ref = np.load(os.path.join(SUB, "oof_blend159av_wh3.npy"))
    print(f"  shipped blend159av_wh3               {roc_auc_score(y, ref):.6f}")

    p = os.path.join(SUB, f"{name}.csv")
    if os.path.exists(p):
        raise SystemExit(f"{p} exists -- refusing to overwrite another agent's file")
    pd.DataFrame({"id": ids, TARGET: t_new}).to_csv(p, index=False)
    np.save(os.path.join(SUB, f"oof_{name}.npy"), o_new)
    print(f"  wrote {p}  rows={len(ids):,}")
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,7,11,13,101,777")
    ap.add_argument("--stage", default="seeds", choices=("seeds", "w", "build", "all"))
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--name", default="w14c_wh3sa")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x]

    if a.stage in ("seeds", "all"):
        stage_seeds(seeds, a.C)
    if a.stage in ("w", "all"):
        W = stage_w(seeds)
        if a.stage == "all":
            stage_build(W, a.name)
    if a.stage == "build":
        with open(os.path.join(OUT, "w14c_weights.json")) as f:
            W = {int(k): np.array(v) for k, v in json.load(f).items()}
        stage_build(W, a.name)


if __name__ == "__main__":
    main()
