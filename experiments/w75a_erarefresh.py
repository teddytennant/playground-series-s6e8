"""w75a — re-estimate the CV->LB ERA TERM on the full held-out sample, and re-run the
WANTED slot-1 argmax against the refreshed number.

Bars in experiments/w75_prereg.txt, committed 5244407 BEFORE this file existed.

WHY. `w46c_predlb.ERA_SHIFT` is -29.82e-6 with se 4.37e-6 and it is estimated on FIVE files
first scored on 2026-08-21. It is the only input to the WANTED slot-1 argmax still resting on
five points. The era-deflated margin RESEARCH:2110 publishes is +5.7e-6 = the 21.7e-6 raw CV
lead minus ERA_SHIFT/slope; it crosses zero at era = -40.3e-6, which is 2.4 se away. Three
send days have since added ~30 held-out ad>=195 files and nobody has folded them in.

⚠ TWO `wave()` FUNCTIONS, DIFFERENT MEANINGS. `stdflag.wave` is the BUILD wave (w36 -> 36);
`w46c_predlb.wave` is the ad-PACK SIZE (ad199 -> 199). The era gate is the pack size. Always
qualify the call.

WRITES ONLY w75a_erarefresh.{json,csv}. Touches no model, no pick, no plan, no submission.

    .venv/bin/python experiments/w75a_erarefresh.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
COMP = "playground-series-s6e8"

import stdflag                                    # noqa: E402
import w46c_predlb as W46                         # noqa: E402

FAIL = []


def check(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    if not ok:
        FAIL.append(msg)


# =====================================================================================
# GATE A — reproduce w46c's own five ERA_ROWS from this pipeline.
# Pinned to the artefact being EXTENDED, not to its ancestor (w74 section 2).
# =====================================================================================
print("=== GATE A: reproduce w46c_predlb's n=5 era estimate ===")
y = pd.read_csv(os.path.join(ROOT, "data/train.csv"), usecols=["addicted_label"]).addicted_label.values


def cv_of(stem):
    p = os.path.join(ROOT, f"submissions/oof_{stem}.npy")
    return float(roc_auc_score(y, np.load(p))) if os.path.exists(p) else np.nan


DOC_PRED = {"w36_ad199std_rescale": 0.971201, "w36_ad199std_hybrid": 0.971172,
            "w36_ad199stdcorr": 0.971212, "w36_ad199std": 0.971199,
            "w34_ad195stdcorr": 0.971184}
res5 = []
for stem, cv_hard, lb in W46.ERA_ROWS:
    cv = cv_of(stem)
    check(abs(cv - cv_hard) < 1e-9, f"{stem:22s} CV from OOF {cv:.10f} vs hardcoded (d={cv-cv_hard:+.1e})")
    raw = W46._raw(cv_hard, stem)
    check(abs(raw - DOC_PRED[stem]) < 0.5e-6, f"{stem:22s} raw pred {raw:.6f} vs docstring {DOC_PRED[stem]:.6f}")
    res5.append((lb - raw) / 1e-6)
res5 = np.array(res5)
check(abs(res5.mean() - (-29.82)) < 0.05, f"n=5 mean residual {res5.mean():+.3f}e-6 vs -29.82")
check(abs(res5.std(ddof=1) - 9.76) < 0.05, f"n=5 sd {res5.std(ddof=1):.3f}e-6 vs 9.76")

# =====================================================================================
# GATE B — independent second anchor: reproduce w28a on its own FROZEN board.
# =====================================================================================
print("\n=== GATE B: reproduce w28a_cvlb_refresh.json on the frozen 77-row board ===")
w28 = pd.read_csv(os.path.join(HERE, "w28a_cvlb_full.csv"))
w28j = json.load(open(os.path.join(HERE, "w28a_cvlb_refresh.json")))
M26 = json.load(open(os.path.join(HERE, "w26e_famfix.json")))
C26, RESID26 = M26["coefs_new"], M26["resid_sd_new"]
old25 = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv"))
# MU26 IS READ FROM THE MODEL'S OWN JSON, NOT RE-DERIVED FROM w25a_cvlb_full.csv.
# w28a re-derived it from that CSV and asserted it matched w26e's stored value. It did, then.
# It does NOT now: the CSV was REGENERATED after w28a ran (60 -> 78 rows above the cv>=0.97
# floor) and the centring moved 0.9700522254 -> 0.9700571395. So w28a_cvlb_refresh.py can no
# longer reproduce its own output and its own assert would fire. Proof that this is the ONLY
# drifted input, and not a second defect: with the frozen mu every stored per-row `pred` in
# w28a_cvlb_full.csv reproduces to 3.3e-16 and gate_resid_sd/oos_mean to 1e-9, checked below.
# A CONSTANT RE-DERIVED FROM A MUTABLE FILE IS NOT FROZEN, however loudly the file says it is.
MU26 = M26["mu"]
_mu_live = float(old25[old25.cv >= 0.97].cv.mean())


def pred26(cv, fam, std):
    lb6 = C26["const"] + C26["cv_e6"] * (cv - MU26) * 1e6 + C26.get(f"fam[{fam}]", 0.0)
    return (lb6 + (C26["standardised"] if std else 0.0)) * 1e-6


w28 = w28.assign(p=[pred26(r.cv, r.fam, r.std) for r in w28.itertuples()])
w28 = w28.assign(r6=(w28.lb - w28.p) * 1e6)
f28 = w28[w28.in_w25f & (w28.cv >= 0.97)]
rsd28 = float(np.sqrt((f28.r6.values ** 2).sum() / (len(f28) - len(C26))))
n28 = w28[~w28.in_w25f & (w28.cv >= 0.97)]
check(abs(MU26 - w28j["mu"]) < 1e-12, f"mu (frozen, from w26e_famfix.json) {MU26:.13f}")
print(f"  note: the same constant re-derived from w25a_cvlb_full.csv today is {_mu_live:.13f} "
      f"({(_mu_live - MU26)*1e6:+.2f}e-6 adrift, n {(old25.cv >= 0.97).sum()} vs w26e's {M26['n']})"
      f" — w28a's own re-derivation would now FAIL")
check(abs(_mu_live - MU26) > 1e-12, "the mu drift is real and is recorded, not silently absorbed")
check(float((w28.p - w28.pred).abs().max()) < 1e-12,
      "every stored per-row pred in w28a_cvlb_full.csv reproduces under the frozen mu")
check(abs(rsd28 - w28j["gate_resid_sd"]) < 1e-6, f"gate_resid_sd {rsd28:.6f} vs {w28j['gate_resid_sd']:.6f}")
check(len(n28) == w28j["oos_n"], f"oos_n {len(n28)} vs {w28j['oos_n']}")
check(abs(n28.r6.mean() - w28j["oos_mean"]) < 1e-6, f"oos_mean {n28.r6.mean():+.6f} vs {w28j['oos_mean']:+.6f}")
check(abs(n28.r6.std(ddof=1) - w28j["oos_sd"]) < 1e-6, f"oos_sd {n28.r6.std(ddof=1):.6f} vs {w28j['oos_sd']:.6f}")
check(abs(w28.cv.max() - w28j["best_sent_cv"]) < 1e-12, f"best_sent_cv {w28.cv.max():.10f}")
check(w28.loc[w28.cv.idxmax(), "stem"] == w28j["best_sent_stem"], f"best_sent_stem {w28.loc[w28.cv.idxmax(),'stem']}")

if FAIL:
    print(f"\n⛔ {len(FAIL)} GATE FAILURE(S) — nothing below is readable.")
    for m in FAIL:
        print("   -", m)
    sys.exit(1)
print("\n✅ both gates clean; the estimator is w46c's and w28a's.\n")

# =====================================================================================
# THE LIVE BOARD
# =====================================================================================
raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "500"],
                     capture_output=True, text=True, cwd=ROOT,
                     env={**os.environ, "KAGGLE_CONFIG_DIR": "/home/nixos/.kaggle"}).stdout
sub = pd.read_csv(io.StringIO(raw))
assert len(sub) >= 131, f"page-size truncation or auth failure: {len(sub)} rows"
sub["date"] = pd.to_datetime(sub["date"])
n_incomplete = int((sub.status != "SubmissionStatus.COMPLETE").sum())
sub = sub[sub.status == "SubmissionStatus.COMPLETE"].copy()
sub["stem"] = sub.fileName.str.replace(r"\.csv$", "", regex=True)
print(f"live submissions {len(sub)} complete (+{n_incomplete} not complete), "
      f"{sub.stem.nunique()} distinct stems")

g = sub.groupby("stem").agg(lb=("publicScore", "max"), lo=("publicScore", "min"),
                            n_sent=("publicScore", "size"), first=("date", "min")).reset_index()

# ---- P5 DETERMINISM ----------------------------------------------------------------
print("\n=== P5: determinism ===")
disagree = g[g.lb != g.lo]
check(len(disagree) == 0, f"stems sent >1x with DIFFERING public scores: {len(disagree)}")
if len(disagree):
    print(disagree.to_string(index=False))

# ---- CV recomputed from the stored OOF vector, never parsed from a description -------
stdflag.require_corr_registered(g.stem)
g["cv"] = [cv_of(s) for s in g.stem]
g["fam"] = g.stem.map(stdflag.family)
g["std"] = g.stem.map(stdflag.is_std)
g["pack"] = g.stem.map(W46.wave)                 # ad-PACK size, not the build wave
g["build"] = g.stem.map(stdflag.wave)
fit_stems = set(old25[old25.cv >= 0.97].stem)
g["in_fit"] = g.stem.isin(fit_stems)
have = g.dropna(subset=["cv"]).copy()
nocv = g[g.cv.isna()]
print(f"\nsent stems {len(g)}   with a stored OOF vector {len(have)}   without {len(nocv)}")

# ---- P4 COMPLETENESS ----------------------------------------------------------------
print("\n=== P4: completeness — the hole this angle exists to close ===")
arg_raw = have.loc[have.cv.idxmax(), "stem"]
check(arg_raw == "w36_ad199stdcorr",
      f"(a) raw-CV argmax over ALL {len(have)} sent stems, CV from OOF: {arg_raw} "
      f"({have.cv.max():.10f})")
check(bool((nocv.lb <= 0.97110).all()),
      f"(b) all {len(nocv)} OOF-less sent stems have public LB <= 0.97110 "
      f"(max {nocv.lb.max() if len(nocv) else float('nan'):.5f})")
print("  OOF-less stems:", ", ".join(f"{r.stem}({r.lb:.5f})" for r in nocv.itertuples()))
top = have.sort_values("cv", ascending=False).head(8)
print("\n  top 8 by recomputed CV:")
for r in top.itertuples():
    print(f"    {r.cv:.10f}  lb {r.lb:.5f}  pack {r.pack}  {r.stem}")

# ---- P1 ERA TERM, REFRESHED ---------------------------------------------------------
print("\n=== P1: the era term on the full held-out sample ===")
have["raw_pred"] = [W46._raw(r.cv, r.stem) for r in have.itertuples()]
have["res6"] = (have.lb - have.raw_pred) * 1e6
era_stems = {s for s, _, _ in W46.ERA_ROWS}
pop = have[(~have.in_fit) & (have.cv >= 0.97) & (have.pack.fillna(0) >= W46.ERA_MIN_AD)
           & (~have.stem.isin(era_stems))]
r = pop.res6.values
era_new = float(r.mean())
se_new = float(r.std(ddof=1) / np.sqrt(len(r)))
print(f"  held-out ad>={W46.ERA_MIN_AD} files not in w30b's fit and not among the five: n={len(r)}")
print(f"  mean residual {era_new:+.3f}e-6   sd {r.std(ddof=1):.3f}   naive se {se_new:.3f}")
print(f"  first-sent dates {pop['first'].min():%m-%d} .. {pop['first'].max():%m-%d}"
      f"   (w30b's board ends {sub[sub.stem.isin(fit_stems)].date.max():%m-%d})")
check(abs(era_new - (-29.82)) <= 8.74,
      f"|{era_new:+.3f} - (-29.82)| = {abs(era_new + 29.82):.3f} <= 8.74e-6 (2 x the n=5 se)")
print("  ⚠ these residuals are NOT independent: one fixed public slice, heavily shared member")
print("    sets. The honest n is far below the count and the naive se is far too small.")
by_day = pop.groupby(pop["first"].dt.strftime("%m-%d")).res6.agg(["size", "mean"])
print("\n  by first-sent day (a day-level read, closer to the honest unit):")
print(by_day.to_string())
day_mean = float(by_day["mean"].mean())
day_se = float(by_day["mean"].std(ddof=1) / np.sqrt(len(by_day))) if len(by_day) > 1 else float("nan")
print(f"  day-mean-of-means {day_mean:+.3f}e-6, se over {len(by_day)} days {day_se:.3f}")

# ---- P2 / P3  DOES IT MOVE WANTED? --------------------------------------------------
print("\n=== P2/P3: era-deflated argmax with the REFRESHED term ===")
SLOPE = W46.C["cv_e6"]
SLOT1 = "w36_ad199stdcorr"


def deflated(term_lb6):
    d = have.copy()
    shift = np.where(d.pack.fillna(0) >= W46.ERA_MIN_AD, term_lb6 / SLOPE * 1e-6, 0.0)
    d["dcv"] = d.cv + shift          # term is negative -> ad>=195 files are pushed DOWN
    return d.sort_values("dcv", ascending=False)


rows = []
for label, term in (("w46c n=5", W46.ERA_SHIFT), ("w75a refreshed", era_new),
                    ("w75a day-mean", day_mean)):
    d = deflated(term)
    a, b = d.iloc[0], d.iloc[1]
    rows.append(dict(basis=label, term_lb6=term, cv_units=term / SLOPE,
                     argmax=a.stem, runner_up=b.stem, margin_e6=(a.dcv - b.dcv) * 1e6))
    print(f"  {label:15s} term {term:+7.2f}e-6 = {term/SLOPE:+6.2f}e-6 of CV -> "
          f"argmax {a.stem:22s} margin {(a.dcv-b.dcv)*1e6:+7.3f}e-6 over {b.stem}")
d_new = deflated(era_new)
check(d_new.iloc[0].stem == SLOT1, f"P2 argmax under the refreshed term is {d_new.iloc[0].stem}")
check(float((d_new.iloc[0].dcv - d_new.iloc[1].dcv) * 1e6) > 0, "P3 margin > 0")

# where does the margin die? sweep the term until slot 1 loses.
kill = None
for t in np.arange(0.0, -120.0, -0.05):
    d = deflated(float(t))
    if d.iloc[0].stem != SLOT1:
        kill = float(t)
        break
print(f"\n  slot 1 stops being the era-deflated argmax at era = {kill:+.2f}e-6"
      f"  ({(kill - era_new)/max(se_new,1e-9):+.1f} naive se from the refreshed estimate,"
      f"  {(kill - (-29.82))/4.37:+.1f} se from the n=5 one)")

# ---- the consolidation table the brief asks for -------------------------------------
print("\n=== CV<->LB across the whole send history (the brief's number) ===")
h97 = have[have.cv >= 0.97]
for label, d in (("all with OOF", have), ("cv>=0.97", h97),
                 ("cv>=0.97, pack>=195", h97[h97.pack.fillna(0) >= 195]),
                 ("cv>=0.97, pack<=194", h97[h97.pack.fillna(0) <= 194])):
    if len(d) < 3:
        continue
    rho, p = spearmanr(d.cv, d.lb)
    print(f"  {label:22s} n={len(d):3d}  Spearman(CV,LB) {rho:+.3f} (p={p:.1e})  "
          f"mean gap {(d.lb - d.cv).mean()*1e6:+.0f}e-6")

out = have.drop(columns=["first"]).sort_values("cv", ascending=False)
out.to_csv(os.path.join(HERE, "w75a_erarefresh.csv"), index=False)
json.dump(dict(n_sent_stems=int(len(g)), n_with_oof=int(len(have)), n_without_oof=int(len(nocv)),
               nocv_stems=list(nocv.stem), nocv_lb_max=float(nocv.lb.max()) if len(nocv) else None,
               raw_argmax=arg_raw, raw_argmax_cv=float(have.cv.max()),
               era_n5=float(W46.ERA_SHIFT), era_new=era_new, era_new_n=int(len(r)),
               era_new_sd=float(r.std(ddof=1)), era_new_se_naive=se_new,
               era_day_mean=day_mean, era_day_se=day_se, era_days=int(len(by_day)),
               slope_cv_e6=float(SLOPE), deflated=rows, kill_era_lb6=kill,
               spearman_cv_lb_all=float(spearmanr(have.cv, have.lb)[0]),
               spearman_cv_lb_97=float(spearmanr(h97.cv, h97.lb)[0]),
               failures=len(FAIL)),
          open(os.path.join(HERE, "w75a_erarefresh.json"), "w"), indent=1)

print(f"\nFAILURES {len(FAIL)}")
for m in FAIL:
    print("   -", m)
print("wrote experiments/w75a_erarefresh.{csv,json}")
sys.exit(1 if FAIL else 0)
