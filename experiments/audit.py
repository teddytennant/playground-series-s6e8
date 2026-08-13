"""End-to-end audit of the submission queue: integrity, distinctness, CV, and CV->LB.

Consolidation instrument. Answers four questions in one pass, with no model fitting:

1.  **Integrity.** Does every candidate CSV in `submissions/` have exactly the test's
    296,302 ids, in the test's order, with finite values in [0, 1] and no NaN? A file
    that fails this scores zero no matter how good the model is, and the failure is
    invisible until Kaggle rejects it.
2.  **Distinctness.** Which files are byte-identical *as rankings*? Scoring here is
    deterministic on a fixed public slice, so two files with the same row ordering are
    the same submission and one of them is a wasted slot. Compared on rank vectors, not
    on floats, because AUC only sees the order.
3.  **CV.** Recompute each candidate's cross-fitted OOF AUC from its own `oof_*.npy`
    rather than trusting the number written in the journal, and check the CSV and the
    OOF vector actually come from the same build (rank correlation of the two is not
    checkable directly, so we check the OOF exists and is the right length).
4.  **CV -> LB.** For every file that has already been scored, pair its recomputed CV
    with its public score and report the gap. The gap is the number this workspace has
    been collecting all week; this prints the whole map at once so the relationship can
    be read rather than remembered.

    audit.py                 # audit everything in submissions/
    audit.py blend158_h3     # audit one candidate verbosely
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
from common import DATA, ID, SUB, TARGET  # noqa: E402

N_TEST = 296_302

# Public scores keyed by the submission FILE name, read off `kaggle competitions
# submissions -v`. Kept here rather than re-fetched so the audit runs offline; a file
# absent from this map is simply reported as unscored.
#
# This dict went 10 points stale on 2026-08-13 and the audit then reported six files as
# "never submitted" that had been submitted that afternoon -- which, in a workspace whose
# one rule for spending a slot is "is this file genuinely different?", is the exact error
# that wastes one. Do not hand-edit it again. Refresh with:
#
#     kaggle competitions submissions -c playground-series-s6e8 -v \
#       | python experiments/lb_refresh.py
#
# which writes experiments/lb_scores.json; that file, when present, is overlaid on top of
# the literals below so the map cannot silently fall behind the leaderboard again.
LB = {
    "stack_pub74_logit.csv": 0.97081,
    "stack_pub88_mine_logit.csv": 0.97081,
    "stack_pub86_hybrid.csv": 0.97080,
    "stack_pub149_hybrid.csv": 0.97099,
    "stack_pub151_hybrid.csv": 0.97099,
    "stack_pub151_rankraw.csv": 0.97102,
    "stack_pub151_fixed_rankraw.csv": 0.97103,
    "blend150fx_hybrid.csv": 0.97099,
    "blend150fx_rankraw.csv": 0.97102,
    "blend150fx_rescale.csv": 0.97102,
    "blend150fx_logit.csv": 0.97103,
    "blend150fx.csv": 0.97104,
    "blend150sx_hybrid.csv": 0.97101,
    "blend150sx_rankraw.csv": 0.97102,
    "blend150sx_logit.csv": 0.97104,
    "blend150sx.csv": 0.97104,
    "blend156_rankraw.csv": 0.97104,
    "blend156_rescale.csv": 0.97105,
    "blend156.csv": 0.97106,
    "blend153.csv": 0.97104,
    # --- 2026-08-13, the ten-submission day: first h3 readings and the fully-crossed 158 ---
    "blend158_h3.csv": 0.97105,
    "blend158.csv": 0.97106,
    "blend158_logit.csv": 0.97106,
    "blend158_hybrid.csv": 0.97103,
    "blend158_rankraw.csv": 0.97104,
    "blend158_rescale.csv": 0.97105,
    "blend159av_h3.csv": 0.97105,
    "blend159av.csv": 0.97106,
    "blend160origm_h3.csv": 0.97105,
    "blend156w.csv": 0.97105,
}

_OVERLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lb_scores.json")
if os.path.exists(_OVERLAY):
    with open(_OVERLAY) as _f:
        LB.update(json.load(_f))


def rk(v):
    return (rankdata(v) - 0.5) / len(v)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None

    te_ids = pd.read_csv(os.path.join(DATA, "test.csv"), usecols=[ID])[ID].to_numpy()
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].astype(int).to_numpy()
    print(f"reference: {len(te_ids):,} test ids, {len(y):,} train rows, "
          f"base rate {y.mean():.10f}")
    if len(te_ids) != N_TEST:
        print(f"  !! expected {N_TEST:,} test rows")

    paths = sorted(glob.glob(os.path.join(SUB, "*.csv")))
    if only:
        paths = [p for p in paths if os.path.basename(p).startswith(only)]

    rows, rank_hash = [], {}
    for p in paths:
        name = os.path.basename(p)
        stem = name[:-4]
        d = pd.read_csv(p)
        bad = []

        if list(d.columns) != [ID, TARGET]:
            bad.append(f"columns={list(d.columns)}")
        if len(d) != len(te_ids):
            bad.append(f"rows={len(d):,}")
        elif not np.array_equal(d[ID].to_numpy(), te_ids):
            same_set = set(d[ID]) == set(te_ids)
            bad.append("id order differs" if same_set else "id SET differs")

        v = d[TARGET].to_numpy(dtype=np.float64)
        if not np.isfinite(v).all():
            bad.append(f"{(~np.isfinite(v)).sum()} non-finite")
        elif v.min() < 0 or v.max() > 1:
            bad.append(f"range [{v.min():.4g}, {v.max():.4g}]")
        n_tie = len(v) - len(np.unique(v))

        # CV from the matching cross-fitted OOF vector, recomputed not remembered.
        cv = np.nan
        op = os.path.join(SUB, f"oof_{stem}.npy")
        if os.path.exists(op):
            o = np.load(op)
            if len(o) != len(y):
                bad.append(f"oof len {len(o):,}")
            else:
                cv = roc_auc_score(y, o)

        # Identity as a RANKING: AUC cannot see anything else.
        h = hashlib.sha1(rankdata(v).astype(np.int64).tobytes()).hexdigest()[:12] \
            if len(v) == len(te_ids) else "-"
        rank_hash.setdefault(h, []).append(stem)

        rows.append(dict(name=stem, cv=cv, lb=LB.get(name, np.nan), n_tie=n_tie,
                         rhash=h, bad="; ".join(bad)))

    df = pd.DataFrame(rows)

    print(f"\n=== integrity: {len(df)} files ===")
    broken = df[df.bad != ""]
    if len(broken) == 0:
        print("  all files: correct columns, correct id order, finite, in [0,1]")
    else:
        for _, r in broken.iterrows():
            print(f"  !! {r['name']:28s} {r['bad']}")

    dups = {h: v for h, v in rank_hash.items() if len(v) > 1 and h != "-"}
    print(f"\n=== distinctness: {len(rank_hash)} distinct rankings ===")
    if not dups:
        print("  every file is a distinct ranking")
    for h, v in dups.items():
        print(f"  !! identical ranking [{h}]: {', '.join(v)}")

    print("\n=== CV -> LB ===")
    m = df.dropna(subset=["cv", "lb"]).sort_values("cv")
    print(f"{'file':28s} {'CV':>9s} {'LB':>9s} {'LB-CV':>9s}")
    for _, r in m.iterrows():
        print(f"{r['name']:28s} {r['cv']:9.6f} {r['lb']:9.5f} {r['lb'] - r['cv']:+9.6f}")
    if len(m) > 2:
        c, l = m.cv.to_numpy(), m.lb.to_numpy()
        print(f"\n  n={len(m)}  pearson {np.corrcoef(c, l)[0, 1]:+.3f}  "
              f"spearman {np.corrcoef(rankdata(c), rankdata(l))[0, 1]:+.3f}")
        b, a = np.polyfit(c, l, 1)
        print(f"  LB = {a:.6f} + {b:.3f} * CV     (slope 1.0 would mean CV gains "
              f"transfer 1:1)")
        print(f"  gap LB-CV: mean {np.mean(l - c):+.6f}  sd {np.std(l - c):.6f}  "
              f"range [{np.min(l - c):+.6f}, {np.max(l - c):+.6f}]")
        # How much of the LB spread is even resolvable? LB is rounded to 5dp.
        print(f"  LB spans {l.max() - l.min():.5f} over a CV span of {c.max() - c.min():.6f}"
              f"; LB quantisation is 1e-5")

    print("\n=== full CV ranking (all files with an OOF vector) ===")
    r = df.dropna(subset=["cv"]).sort_values("cv", ascending=False)
    print(f"{'file':28s} {'CV':>9s} {'LB':>9s}  sent")
    for _, x in r.iterrows():
        lb = f"{x['lb']:9.5f}" if not np.isnan(x['lb']) else f"{'-':>9s}"
        print(f"{x['name']:28s} {x['cv']:9.6f} {lb}  {'yes' if not np.isnan(x['lb']) else 'NO'}")

    unsent = r[r.lb.isna()]
    print(f"\n{len(unsent)} scored-CV files never submitted; best unsent CV "
          f"{unsent.cv.max():.6f} vs best sent {r.dropna(subset=['lb']).cv.max():.6f}"
          if len(unsent) else "\nevery file with a CV has been submitted")

    df.sort_values("cv", ascending=False).to_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_results.csv"),
        index=False)


if __name__ == "__main__":
    main()
