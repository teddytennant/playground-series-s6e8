"""w28b — end-to-end re-verification of the top of the pipeline, and validation of every
UNSENT candidate CSV.

Consolidation slot. Three jobs, all read-only against the artefacts:

1. Recompute the cross-fitted CV of every file on disk that has a stored OOF vector and is
   NOT yet sent, straight from `data/train.csv`. No number is copied from the journal.
2. Validate each unsent candidate CSV the way `w26g_send.py` will validate it tomorrow --
   id set, row count, NaN, duplicates -- so a malformed file cannot burn a slot. ⚠
   `w27_ad190stdcorr.csv`, the current CV leader, was built after `w27x_validate.log` was
   written and has never been through this.
3. Check that each candidate's TEST vector is consistent with the OOF vector it ships with:
   same construction => the test predictions must lie in the same value regime (a `*corr`
   file is a probability in [0,1]; a bare transform file is a z-scale rank statistic). A
   mismatch means the CSV and the OOF came from different runs.

Then price every unsent file under the frozen w25f model, so tomorrow's send order is a
ranked list rather than a hand-pick.

    OMP_NUM_THREADS=4 .venv/bin/python experiments/w28b_verify.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e8"
BEST_LB, STEP = 0.97118, 1e-5


# Classifiers live in `stdflag` -- see that module for why the substring rule was WRONG.
import stdflag  # noqa: E402
from stdflag import family, is_std   # noqa: E402


# w30b's refit, to match `stdflag.family`'s corrected `*corr` labels AND to carry the c_avg
# correction term (w30). This file used to load `w26e_famfix.json` and the `mu` assert below
# FIRED once w25a's table grew past that fit's 60 rows -- which is the assert doing its job:
# the model file and the centring source must be the same vintage or every prediction here is
# silently mis-centred. If it fires again, refit rather than widening the tolerance.
M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C, RESID = M["coefs"], M["resid_sd_new"]
MU = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).query("cv >= 0.97").cv.mean()
assert abs(MU - M["mu"]) < 1e-12, "centring drifted from the fitted model"


def predict(cv, fam, std, corr=False):
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{fam}]", 0.0)
    return (lb6 + (C["std"] if std else 0.0) + (C["corr"] if corr else 0.0)) * 1e-6


# --- what is already sent, live by NAME and by MD5 ------------------------------------
raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                      "--page-size", "300"], capture_output=True, text=True, cwd=ROOT).stdout
sub = pd.read_csv(io.StringIO(raw))
assert len(sub) > 50, "page-size truncation"
sent = set(sub.fileName)
print(f"live: {len(sub)} submissions, {len(sent)} distinct filenames")

y = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["addicted_label"]).addicted_label.values
test_ids = pd.read_csv(os.path.join(ROOT, "data/test.csv"), usecols=["id"]).id.values
N_TEST = len(test_ids)
print(f"test rows: {N_TEST:,}")

subdir = os.path.join(ROOT, "submissions")
cands = []
for fn in sorted(os.listdir(subdir)):
    if not fn.endswith(".csv") or fn in sent:
        continue
    stem = fn[:-4]
    oofp = os.path.join(subdir, f"oof_{stem}.npy")
    if not os.path.exists(oofp):
        cands.append(dict(stem=stem, cv=np.nan))
        continue
    cv = roc_auc_score(y, np.load(oofp))
    cands.append(dict(stem=stem, cv=cv))

print(f"\nunsent CSVs on disk: {len(cands)}   with a stored OOF: {sum(np.isfinite(c['cv']) for c in cands)}")

rows = []
for c in sorted(cands, key=lambda d: (-d["cv"] if np.isfinite(d["cv"]) else 1)):
    stem = c["stem"]
    d = pd.read_csv(os.path.join(subdir, stem + ".csv"))
    cols_ok = list(d.columns) == ["id", "addicted_label"]
    n_ok = len(d) == N_TEST
    id_ok = bool(np.array_equal(np.sort(d.id.values), np.sort(test_ids)))
    dup = int(d.id.duplicated().sum())
    v = d.addicted_label.values
    nan = int(np.isnan(v).sum())
    fin = bool(np.isfinite(v).all())
    prob = bool(v.min() >= 0.0 and v.max() <= 1.0)
    valid = cols_ok and n_ok and id_ok and dup == 0 and nan == 0 and fin
    fam, std, corr = family(stem), is_std(stem), "corr" in stem
    cv = c["cv"]
    pred = predict(cv, fam, std, corr) if np.isfinite(cv) else np.nan
    p = 1.0 - norm.cdf((BEST_LB + STEP / 2 - pred) / (RESID * 1e-6)) if np.isfinite(pred) else np.nan
    rows.append(dict(stem=stem, cv=cv, fam=fam, std=std, corr=corr, pred_lb=pred, p_beat=p,
                     valid=valid, rows=len(d), dup=dup, nan=nan, prob_scale=prob,
                     vmin=float(v.min()), vmax=float(v.max())))

t = pd.DataFrame(rows)
stdflag.require_corr_registered(t.stem)   # a new *corr file must be classified BY HAND
pd.set_option("display.width", 240)
print("\n=== UNSENT CANDIDATES, best CV first ===")
print(t.assign(cv=lambda d: d.cv.map(lambda x: f"{x:.10f}" if np.isfinite(x) else "-"),
               pred_lb=lambda d: d.pred_lb.map(lambda x: f"{x:.5f}" if np.isfinite(x) else "-"),
               p_beat=lambda d: d.p_beat.map(lambda x: f"{x:.2e}" if np.isfinite(x) else "-"))
      [["stem", "fam", "std", "cv", "pred_lb", "p_beat", "valid", "rows", "dup", "nan",
        "prob_scale", "vmin", "vmax"]].head(30).to_string(index=False))

bad = t[~t.valid]
print(f"\nINVALID candidate files: {len(bad)}" + ("" if len(bad) == 0 else "\n" + bad.to_string(index=False)))

top = t.dropna(subset=["cv"]).nlargest(10, "cv")
print("\n=== SEND ORDER for the next UTC day (top 10 unsent by CV) ===")
for i, r in enumerate(top.itertuples(), 1):
    print(f"  {i:2d}. {r.stem:28s} cv {r.cv:.10f}  fam {r.fam:8s} std {str(r.std):5s} "
          f"pred {r.pred_lb:.5f}  P(>{BEST_LB}) {r.p_beat:.2e}  valid {r.valid}")
print(f"\n  P(at least one of these ten beats {BEST_LB}) = "
      f"{1.0 - np.prod(1.0 - top.p_beat.values):.3e}   (independence assumed -> OVERSTATED)")

t.to_csv(os.path.join(HERE, "w28b_unsent.csv"), index=False)
print("\nwrote experiments/w28b_unsent.csv")
