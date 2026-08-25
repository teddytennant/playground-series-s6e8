"""w25a — the FULL CV<->LB table over the entire live send history, rebuilt from scratch.

The angle handed for this slot is consolidation: "check the CV-to-LB gap across every
experiment so far, and make sure the strongest submission is the one selected." The last
time this account cut the CV->LB relation over the whole history was w17a (46 files,
2026-08-17 early). Waves 18-25 added the whole 187-member-pack line -- w20_ad187_h3,
w20_ad187_rankraw, w20_ad187, w21_ad187corr, w21_ad187corr_ens4, w23_ad187stdcorr -- which
is exactly the top-of-CV region where the selection decision lives, and none of it is in
w17a's fit.

Every LB number is read LIVE from the API, never from a stale local table. Every CV number
is recomputed HERE from the stored OOF vector against data/train.csv, never copied from the
journal, so a transcription error in either column cannot survive.

    .venv/bin/python experiments/w25a_cvlb_full.py
"""
from __future__ import annotations

import io
import json
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from w91a_subdate import parse_sub_dates  # w91: one owner of the timestamp format

COMP = "playground-series-s6e8"

raw = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "300"],
    capture_output=True, text=True,
).stdout
sub = pd.read_csv(io.StringIO(raw))
sub["date"] = parse_sub_dates(sub["date"])
bad = sub[sub["status"] != "SubmissionStatus.COMPLETE"]
print(f"live submissions: {len(sub)}   non-COMPLETE: {len(bad)}")
sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].copy()
sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)

# Scores are deterministic, so a stem sent twice must print the same number. Verify rather
# than assume -- a mismatch would mean the file on disk changed under a reused name.
g = sub.groupby("stem")["publicScore"].agg(["max", "min", "count"])
inc = g[g["max"] != g["min"]]
print(f"stems sent >1x with DIFFERING scores: {len(inc)}")
if len(inc):
    print(inc)

y = pd.read_csv("data/train.csv", usecols=["addicted_label"]).addicted_label.values

rows = []
for stem, r in g.iterrows():
    p = f"submissions/oof_{stem}.npy"
    try:
        oof = np.load(p)
    except FileNotFoundError:
        rows.append(dict(stem=stem, cv=np.nan, lb=r["max"], n_sent=int(r["count"])))
        continue
    rows.append(dict(stem=stem, cv=roc_auc_score(y, oof), lb=r["max"], n_sent=int(r["count"])))

t = pd.DataFrame(rows).sort_values("cv", ascending=False, na_position="last").reset_index(drop=True)
t["gap"] = t["lb"] - t["cv"]

have = t.dropna(subset=["cv"]).copy()
print(f"\nscored stems: {len(t)}   with a local OOF vector: {len(have)}   "
      f"CV unavailable: {len(t) - len(have)}")

pd.set_option("display.width", 200)
print("\n=== every scored file with a recomputed CV, best CV first ===")
print(have.assign(cv=lambda d: d.cv.map("{:.10f}".format),
                  gap=lambda d: (d.gap * 1e6).map("{:+.1f}".format)).to_string(index=False))

# --- the relation itself -----------------------------------------------------------
x = have["cv"].values
z = have["lb"].values
sl, ic = np.polyfit(x, z, 1)
res = z - (sl * x + ic)
from scipy.stats import spearmanr, pearsonr
rho, prho = spearmanr(x, z)
rp, pp = pearsonr(x, z)
print(f"\nOLS  LB = {sl:.4f} * CV + {ic:.6f}")
print(f"     slope {sl:.3f}   residual sd {res.std(ddof=2)*1e6:.2f}e-6   n {len(x)}")
print(f"Spearman rho {rho:+.3f} (p {prho:.2e})   Pearson r {rp:+.3f} (p {pp:.2e})")
print(f"mean gap (LB - CV) {have.gap.mean()*1e6:+.1f}e-6   sd {have.gap.std()*1e6:.1f}e-6")

# The decision region is what matters: the fit above is dominated by the low-CV public-stack
# forks. Re-cut on the leaders only.
for thr in (0.97000, 0.9700400, 0.9700900):
    h = have[have.cv >= thr]
    if len(h) < 4:
        continue
    s2, i2 = np.polyfit(h.cv.values, h.lb.values, 1)
    r2 = h.lb.values - (s2 * h.cv.values + i2)
    rr, pr2 = spearmanr(h.cv.values, h.lb.values)
    print(f"\nCV >= {thr:.7f}: n {len(h):2d}  slope {s2:+8.3f}  resid sd {r2.std(ddof=2)*1e6:5.2f}e-6"
          f"  Spearman {rr:+.3f} (p {pr2:.3f})  LB range {h.lb.min():.5f}-{h.lb.max():.5f}")

out = "experiments/w25a_cvlb_full.csv"
t.to_csv(out, index=False)
json.dump(dict(n_scored=int(len(t)), n_with_cv=int(len(have)), slope=float(sl),
               resid_sd=float(res.std(ddof=2)), spearman=float(rho), spearman_p=float(prho),
               mean_gap=float(have.gap.mean()), gap_sd=float(have.gap.std())),
          open("experiments/w25a_cvlb_full.json", "w"), indent=1)
print(f"\nwrote {out}")
