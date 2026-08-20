"""w39 consolidation -- END-TO-END INTEGRITY AUDIT of the files that decide the competition.

Every entry since w23 argues about which file to send and which to select. Nothing has ever
checked that the CSVs themselves are structurally sound: right rows, right ids in the right
ORDER, finite, non-degenerate, and byte-identical to what w26d_queueprice priced. A queue-head
file with shuffled ids scores ~0.5 and no amount of CV work rescues it.

Audits: every priority-1 queued file (the 13 the send order will draw tomorrow), plus every
sent file that currently holds the best public score, plus the two check_selection WANTED files.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")

ss = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
REF_IDS = ss["id"].values
N = len(REF_IDS)
print(f"reference: sample_submission {N} rows, id {REF_IDS[0]}..{REF_IDS[-1]}\n")

q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
md5map = dict(zip(q.file, q.md5))
cvmap = dict(zip(q.file, q.cv))

targets = list(q[q.priority == 1].sort_values("send_rank").file)
# the two files Kaggle would auto-select today, and the WANTED pair
for extra in ["w29_ad194stdcorr.csv", "w27_ad190stdcorr.csv", "w23_ad187stdcorr.csv"]:
    if extra not in targets:
        targets.append(extra)

rows = []
vecs = {}
for f in targets:
    p = os.path.join(SUB, f)
    r = {"file": f, "exists": os.path.exists(p)}
    if not r["exists"]:
        rows.append(r)
        continue
    with open(p, "rb") as fh:
        r["md5"] = hashlib.md5(fh.read()).hexdigest()
    r["md5_ok"] = (md5map.get(f) is None) or (r["md5"] == md5map[f])
    d = pd.read_csv(p)
    r["rows"] = len(d)
    r["cols"] = ",".join(d.columns)
    r["rows_ok"] = len(d) == N
    r["ids_exact_order"] = bool(len(d) == N and np.array_equal(d["id"].values, REF_IDS))
    v = d["addicted_label"].values.astype(np.float64)
    r["n_nan"] = int(np.isnan(v).sum())
    r["n_inf"] = int(np.isinf(v).sum())
    r["min"], r["max"] = float(np.nanmin(v)), float(np.nanmax(v))
    r["n_unique"] = int(len(np.unique(v)))
    r["mean"] = float(np.nanmean(v))
    r["cv"] = cvmap.get(f, float("nan"))
    vecs[f] = v
    rows.append(r)

t = pd.DataFrame(rows)
bad = t[~(t.exists & t.md5_ok.fillna(False) & t.rows_ok.fillna(False)
          & t.ids_exact_order.fillna(False) & (t.n_nan == 0) & (t.n_inf == 0)
          & (t.n_unique > 1000))]
pd.set_option("display.width", 200)
print(t[["file", "exists", "md5_ok", "rows", "ids_exact_order", "n_nan", "n_inf",
         "n_unique", "min", "max", "mean"]].to_string(index=False))
t.to_csv(os.path.join(HERE, "w39a_audit.csv"), index=False)

print(f"\nFAILURES: {len(bad)}")
if len(bad):
    print(bad.to_string(index=False))

# Spearman of the queue head against what the board already holds -- a shuffled or
# mis-joined file shows up here as a correlation nowhere near 1.
head = "w36_ad199stdcorr.csv"
if head in vecs:
    from scipy.stats import spearmanr
    print(f"\nrank correlation of the queue head {head} against:")
    for f in ["w29_ad194stdcorr.csv", "w27_ad190stdcorr.csv", "w23_ad187stdcorr.csv"]:
        if f in vecs:
            rho = spearmanr(vecs[head], vecs[f]).statistic
            print(f"  {f:32s} rho {rho:.6f}")
