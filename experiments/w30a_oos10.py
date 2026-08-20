"""w30a — score the 08-20 ten-file drain against the FROZEN w26e LB model.

WHY THIS IS WORTH A SCRIPT
--------------------------
Every previous out-of-sample test of this workspace's CV->LB predictor was one or two
files at a time, which cannot separate "the model is right" from "that file drew well":
the paired slice sd is ~8.2e-6, one reporting step is 10e-6, so a single file is a coin
flip against any hypothesis. Today's drain sent TEN files whose predictions were written
by w26d_queueprice.py BEFORE upload (they are quoted verbatim in each submission message,
so they cannot be revised after the fact), against a model fitted on 60 files none of
which are these ten. That is a real held-out sample and it can resolve family offsets that
n=3 in-sample could not.

Reports, per file and per family: predicted LB, actual LB, residual, and the z of each
family's mean residual against the model's own 8.35e-6 residual sd.

    .venv/bin/python experiments/w30a_oos10.py
"""
from __future__ import annotations

import csv, io, json, os, subprocess, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
SENT_DAY = "2026-08-20"
M = json.load(open(os.path.join(HERE, "w26e_famfix.json")))
RESID = M["resid_sd_new"]
STEP = 1e-5           # the public board's reporting granularity

q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
q["stem"] = q.file.str.replace(".csv", "", regex=False)

r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                    "--page-size", "500"], capture_output=True, text=True, timeout=300)
rows = list(csv.DictReader(io.StringIO(r.stdout)))
if not rows:
    sys.exit("kaggle returned nothing")
sub = pd.DataFrame(rows)
sub = sub[sub.date.str.startswith(SENT_DAY) & (sub.publicScore != "")].copy()
sub["stem"] = sub.fileName.str.replace(".csv", "", regex=False)
sub["lb"] = sub.publicScore.astype(float)
if len(sub) != 10:
    print(f"⚠ expected 10 scored files on {SENT_DAY}, found {len(sub)}")

d = sub[["stem", "lb"]].merge(q[["stem", "cv", "fam", "std", "pred_lb"]], on="stem")
d["resid_e6"] = (d.lb - d.pred_lb) * 1e6
d = d.sort_values("resid_e6", ascending=False)

print(f"HELD-OUT: {len(d)} files sent {SENT_DAY}, priced by w26d BEFORE upload,")
print(f"under a model fitted on {M['n']} files none of which are these.\n")
print(f"{'file':26s} {'fam':8s} {'CV':>14s} {'pred':>9s} {'actual':>8s} {'resid e-6':>10s}")
for _, x in d.iterrows():
    print(f"{x.stem:26s} {x.fam:8s} {x.cv:14.10f} {x.pred_lb:9.6f} {x.lb:8.5f} {x.resid_e6:+10.2f}")

# Rounding to the 1e-5 grid puts a uniform(-5,+5)e-6 term on every residual: sd 2.89e-6.
tot = float(np.sqrt(RESID ** 2 + (STEP * 1e6) ** 2 / 12))
m, n = d.resid_e6.mean(), len(d)
print(f"\npooled: mean {m:+.2f}e-6  sd {d.resid_e6.std(ddof=1):.2f}  "
      f"vs model {RESID:.2f} (+grid -> {tot:.2f})")
print(f"        z of the mean = {m / (tot / np.sqrt(n)):+.2f}   "
      f"{'BIASED' if abs(m / (tot / np.sqrt(n))) > 2 else 'unbiased'}")
print(f"        residual sd ratio observed/model = {d.resid_e6.std(ddof=1) / tot:.2f}")

print(f"\nBY FAMILY (the term fitted on the fewest points is the one most likely wrong)")
print(f"{'fam':10s} {'n':>2s} {'mean resid':>11s} {'z':>7s}")
for fam, g in d.groupby("fam"):
    mm, k = g.resid_e6.mean(), len(g)
    z = mm / (tot / np.sqrt(k))
    print(f"{fam:10s} {k:2d} {mm:+11.2f} {z:+7.2f}  {'*' if abs(z) > 2 else ''}")

json.dump(dict(n=len(d), mean_resid=float(m), sd_resid=float(d.resid_e6.std(ddof=1)),
               model_sd=RESID, total_sd=tot,
               by_fam={f: dict(n=int(len(g)), mean=float(g.resid_e6.mean()))
                       for f, g in d.groupby("fam")}),
          open(os.path.join(HERE, "w30a_oos10.json"), "w"), indent=1)
d.to_csv(os.path.join(HERE, "w30a_oos10.csv"), index=False)
print("\nwrote experiments/w30a_oos10.{csv,json}")
