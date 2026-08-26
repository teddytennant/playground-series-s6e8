"""w28a — refresh the CV<->LB table over the WHOLE send history and test the w25f model
OUT OF SAMPLE on everything sent since it was fitted.

The angle for this slot is consolidation: re-verify the best pipeline, check the CV-to-LB gap
across every experiment so far, and make sure the strongest submission is selected. w25a cut
that table once (2026-08-18, 71 stems). Since then the account has sent 18 more files -- eight
on 08-18 and ten on 08-19 -- and, more importantly, w27 slot 3 found and fixed a BUG in the
`standardised` flag (`w26d_queueprice.is_std`) that had been handing every post-w23 build
+27.43e-6 of predicted LB. Every prediction made between w25f and that fix was wrong by 3.3
reporting steps. This script is the audit of what the model actually does now.

⚠ WHICH MODEL THIS AUDITS (w92, 2026-08-26). The model here is the SUPERSEDED w25f/w26e
ancova. The live pricer has been `w53a_pricer` on `w30b_corrterm.json` since w30, centred at
the pinned MU 0.9700571395217258 that `w57c_muguard` and `w75b_muguard` re-verify every run.
Read this file for what the 08-19 model does out of sample; read `w82a_pricecal` for the LIVE
predictor measured against its own published predictions, and `w75a_erarefresh` for the era
term. Nothing here re-prices, re-ranks or re-centres anything.

Two things it deliberately does NOT do:

1. It does NOT refit w25f. The coefficients AND the centring constant MU are read frozen from
   the model's own JSON, and the 60 fit stems from `w28a_w25f_fitrows.json`, a snapshot of
   `w25a_cvlb_full.csv` as w25f read it (git 24dbd0a).
   🔴 MU USED TO BE RE-DERIVED FROM THE LIVE `w25a_cvlb_full.csv`, and w30's legitimate
   2026-08-22 refresh of that registry (71 rows -> 91, cv>=0.97 mean +4.914e-6) turned line 99
   into an AssertionError. This module was dead from 2026-08-22 10:12 until w92, and the w91
   date repair could not have revived it -- it had a second, independent break one line down.
   A frozen model's centring is a property of the model, not of a mutable CSV.
2. It does NOT overwrite `w25a_cvlb_full.csv`. ⚠ That file is the MU pin's source; re-running
   `w25a_cvlb_full.py` would silently re-centre the LIVE pricer and every stored prediction
   and gate downstream would shift. Output goes to `w28a_cvlb_full.csv`.

    .venv/bin/python experiments/w28a_cvlb_refresh.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e8"


# --- classifiers ---
# Classifiers live in `stdflag` -- see that module for why the substring rule was WRONG.
import stdflag  # noqa: E402
from stdflag import family, is_std   # noqa: E402
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from w91a_subdate import parse_sub_dates  # w91: one owner of the timestamp format


# --- live LB -------------------------------------------------------------------------
raw = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "300"],
    capture_output=True, text=True, cwd=ROOT).stdout
sub = pd.read_csv(io.StringIO(raw))
assert len(sub) > 50, f"page-size truncation: only {len(sub)} rows (the CLI default is 50)"
sub["date"] = parse_sub_dates(sub["date"])
print(f"live submissions: {len(sub)}   non-COMPLETE: {(sub.status != 'SubmissionStatus.COMPLETE').sum()}")
sub = sub[sub.status == "SubmissionStatus.COMPLETE"].copy()
sub["stem"] = sub.fileName.str.replace(r"\.csv$", "", regex=True)

g = sub.groupby("stem").agg(lb=("publicScore", "max"), lo=("publicScore", "min"),
                            n_sent=("publicScore", "size"), first=("date", "min"))
inc = g[g.lb != g.lo]
print(f"stems sent >1x with DIFFERING scores: {len(inc)}   (scores are deterministic; >0 is a defect)")
if len(inc):
    print(inc)

# --- CV recomputed from the stored OOF vector, never copied from the journal ----------
y = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["addicted_label"]).addicted_label.values
rows = []
for stem, r in g.iterrows():
    p = os.path.join(ROOT, f"submissions/oof_{stem}.npy")
    cv = roc_auc_score(y, np.load(p)) if os.path.exists(p) else np.nan
    rows.append(dict(stem=stem, cv=cv, lb=r.lb, n_sent=int(r.n_sent), first=r.first))
t = pd.DataFrame(rows)
t["gap"] = t.lb - t.cv
t["fam"] = t.stem.map(family)
t["std"] = t.stem.map(is_std)

# ⛔ THE FIT MEMBERSHIP COMES FROM A FROZEN SNAPSHOT, NOT FROM THE LIVE REGISTRY.
# `w25a_cvlb_full.csv` is mutable — w30 refreshed it on 2026-08-22 from 71 rows to 91 and
# moved the cv>=0.97 mean by +4.914e-6. Reading fit membership off it marked 78 stems as
# "in w25f's fit" when 60 were, and re-deriving MU off it broke this module outright.
_FIT = json.load(open(os.path.join(HERE, "w28a_w25f_fitrows.json")))
fit_rows = set(_FIT["stems"])
stdflag.require_corr_registered(t.stem)   # a new *corr file must be classified BY HAND
t["in_w25f"] = t.stem.isin(fit_rows)
have = t.dropna(subset=["cv"]).sort_values("cv", ascending=False).reset_index(drop=True)
print(f"\nscored stems: {len(t)}   with a local OOF vector: {len(have)}   "
      f"CV unavailable: {len(t) - len(have)}   in w25f's fit: {t.in_w25f.sum()}")

# --- the frozen w25f model -----------------------------------------------------------
# The model is w26e's REFIT (w25f with the two `*corr` labels corrected to h3), because the
# labels used here are `stdflag.family`, which applies CORR_MAP. Pairing w25f's coefficients
# with these labels would be neither model. See stdflag.family's docstring.
M = json.load(open(os.path.join(HERE, "w26e_famfix.json")))
C, RESID = M["coefs_new"], M["resid_sd_new"]
MU = float(M["mu"])                         # the model's OWN centring, read from its JSON
assert len(fit_rows) == M["n"] == 60, "the frozen fit snapshot is not w26e's 60 rows"
assert abs(_FIT["mu"] - MU) < 1e-12, "the frozen snapshot no longer reproduces the model's mu"


def predict(cv, fam, std):
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{fam}]", 0.0)
    return (lb6 + (C["standardised"] if std else 0.0)) * 1e-6


have["pred"] = [predict(r.cv, r.fam, r.std) for r in have.itertuples()]
have["res6"] = (have.lb - have.pred) * 1e6

# GATE: reproduce w25f's own residual sd on its own 60 rows.
f = have[have.in_w25f & (have.cv >= 0.97)]
dof = len(f) - len(C)
rsd = float(np.sqrt((f.res6.values ** 2).sum() / dof))
ok = abs(rsd - RESID) < 0.5
print(f"GATE: w26e rows n={len(f)} residual sd {rsd:.2f}e-6 (dof {dof}) vs stored "
      f"{RESID:.2f}e-6 -- {'PASS' if ok else 'FAIL'}")
assert ok, "predict() is not w26e's refit; nothing below is readable"

print("\n=== EVERY SCORED FILE, best CV first ===")
pd.set_option("display.width", 220)
show = have.assign(cv=lambda d: d.cv.map("{:.10f}".format),
                   gap=lambda d: (d.gap * 1e6).map("{:+.0f}".format),
                   pred=lambda d: d.pred.map("{:.5f}".format),
                   res6=lambda d: d.res6.map("{:+.1f}".format),
                   first=lambda d: d["first"].dt.strftime("%m-%d"))
print(show[["stem", "fam", "std", "in_w25f", "cv", "lb", "pred", "res6", "first"]].to_string(index=False))

# --- OUT OF SAMPLE -------------------------------------------------------------------
new = have[~have.in_w25f & (have.cv >= 0.97)]
print(f"\n=== OUT OF SAMPLE: {len(new)} files sent AFTER w25f was fitted ===")
print(show.loc[new.index, ["stem", "fam", "std", "cv", "lb", "pred", "res6"]].to_string(index=False))
if len(new):
    r = new.res6.values
    print(f"\n  n={len(r)}  mean residual {r.mean():+.2f}e-6  sd {r.std(ddof=1):.2f}e-6  "
          f"max|res| {np.abs(r).max():.1f}e-6   (fitted sd {RESID:.2f}e-6, slice floor 8.21e-6)")
    print("  ⚠ these residuals are NOT independent: every file is scored on the SAME fixed "
          "public slice\n    and most share member sets, so the honest n is far below the count.")

# --- the bar for a new file ----------------------------------------------------------
BEST, STEP = 0.97118, 1e-5
need = (BEST + STEP / 2) * 1e6
print(f"\nCV a NEW file needs for an even-money shot at beating {BEST:.5f}:")
print(f"  {'family':>8s} {'unstandardised':>17s} {'standardised':>17s}")
bars = {}
for fam in ("h3", "ens4", "hybrid", "rankraw", "rescale", "logit"):
    base = (need - C["const"] - C.get(f"fam[{fam}]", 0.0)) / C["cv_e6"] * 1e-6 + MU
    std = base - C["standardised"] / C["cv_e6"] * 1e-6
    bars[fam] = dict(unstd=base, std=std)
    print(f"  {fam:>8s} {base:17.10f} {std:17.10f}")
print("  Every file built here since w23 is standardised. Read the RIGHT column.")

best_cv = have.cv.max()
best_stem = have.loc[have.cv.idxmax(), "stem"]
print(f"\nbest CV among SENT files: {best_cv:.10f}  ({best_stem})")

have.drop(columns=["first"]).to_csv(os.path.join(HERE, "w28a_cvlb_full.csv"), index=False)
json.dump(dict(n_sent_stems=int(len(t)), n_with_cv=int(len(have)), mu=float(MU),
               gate_resid_sd=rsd, oos_n=int(len(new)),
               oos_mean=float(new.res6.mean()) if len(new) else None,
               oos_sd=float(new.res6.std(ddof=1)) if len(new) > 1 else None,
               bars=bars, best_sent_cv=float(best_cv), best_sent_stem=best_stem),
          open(os.path.join(HERE, "w28a_cvlb_refresh.json"), "w"), indent=1)
print("\nwrote experiments/w28a_cvlb_full.csv and w28a_cvlb_refresh.json")
