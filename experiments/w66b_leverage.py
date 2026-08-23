"""w66b — is w66a's calibration slope a LEVERAGE ARTEFACT of three extrapolated files?

⚠⚠ DECLARED POST-HOC. Nothing here is registered in w66_prereg.txt. It exists because w66a's
PART A sample turned out to contain three files (`stack_pub88_mine_logit`, `stack_pub74_logit`,
`stack_pub86_hybrid`) sitting 300-430e-6 of predicted LB below the other sixteen, and a
calibration slope fitted through three far points and a tight cluster is a slope fitted through
three points. Read every number below as a robustness check on w66a, never as a confirmation of
it: a post-hoc subset that agrees is weak evidence, and a post-hoc subset that DISAGREES is the
finding.

Three arms per part, and the reading rule is fixed before the arms are printed:
  * if the ALL arm and the TIGHT arm agree to within their overlapping intervals, w66a's slope
    is a property of the model and not of the three points;
  * if the TIGHT arm's interval contains 1.0 while ALL's excludes it, w66a measured extrapolation
    failure only, and its headline must be restated as such.

    .venv/bin/python experiments/w66b_leverage.py
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from common import SUB, TARGET, load_raw                       # noqa: E402
from w16b_cellweight import fast_auc                           # noqa: E402
import w53a_pricer as PR                                       # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402
from w66a_pricercal import gls, load_board, ROUND_VAR, U, FIT_TABLE   # noqa: E402 — SHARED

COMP = "playground-series-s6e8"
CV_FLOOR = 0.9699          # the three outliers sit at 0.96964..0.96968; the rest at >=0.96996
OUTLIERS = ("stack_pub88_mine_logit", "stack_pub74_logit", "stack_pub86_hybrid")


def main():
    beta_c = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    LB = load_board()
    fit_stems = set(pd.read_csv(FIT_TABLE).stem)
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    fresh = set(sub[sub.date.str[:10] == "2026-08-23"].fileName.str.replace(r"\.csv$", "", regex=True))

    # ---------------------------------------------------------------- PART A, three arms -----
    oos = sorted(k for k in LB if k not in fit_stems
                 and os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                 and PR.family(k) != "member")
    names = sorted(set(oos) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}
    E = W63A.fit(names, LB, cv, V, y, beta_c)
    Sxx, col = E["Sxx"], E["col"]
    del V

    print("=" * 96)
    print("PART A — the out-of-sample calibration slope, on three nested samples")
    print("=" * 96)

    def arm(keys, tag):
        js = [col[k] for k in keys]
        pred = np.array([PR.predict_lb(cv[k], k) for k in keys])
        z = (np.array([LB[k] for k in keys]) - pred) / U
        px = (pred - pred.mean()) / U
        X = np.column_stack([np.ones(len(keys)), px])
        Om = Sxx[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
        G = gls(X, z, Om)
        c, se = float(G["b"][1]), float(G["se_scaled"][1])
        lo, hi = 1 + c - 1.96 * se, 1 + c + 1.96 * se
        print(f"\n  {tag}  n={len(keys)}  pred spread {px.max()-px.min():7.1f}e-6  "
              f"chi2/dof {G['chi2']/G['dof']:.3f} scale {G['scale']:.3f}")
        print(f"    beta_total = {1+c:7.4f}   se {se:.4f}   95% CI [{lo:7.4f}, {hi:7.4f}]   "
              f"excludes 1.0: {not (lo <= 1.0 <= hi)}   excludes 0: {not (lo <= 0 <= hi)}")
        # leverage: which rows carry the slope
        iOm = G["iOm"]
        M = np.linalg.inv(X.T @ iOm @ X) @ (X.T @ iOm)
        w = M[1]                       # the slope is a linear functional of z; these are weights
        top = sorted(zip(keys, w), key=lambda r: -abs(r[1]))[:4]
        print(f"    slope weight |w| top 4: " +
              ", ".join(f"{k.split('_')[0]}..{k[-10:]} {v:+.4f}" for k, v in top))
        print(f"    share of sum|w| in the 3 extrapolated files: "
              f"{sum(abs(v) for k, v in zip(keys, w) if k in OUTLIERS)/np.abs(w).sum():.3f}")
        return dict(tag=tag, n=len(keys), beta=1 + c, se=se, ci=[lo, hi],
                    excl1=bool(not (lo <= 1.0 <= hi)), excl0=bool(not (lo <= 0 <= hi)))

    tight = [k for k in oos if cv[k] >= CV_FLOOR]
    fr = [k for k in oos if k in fresh]
    A_all = arm(oos, "ALL   ")
    A_tight = arm(tight, "TIGHT ")
    A_fresh = arm(fr, "FRESH ")

    # ---------------------------------------------------------------- PART B, two arms -------
    print("\n" + "=" * 96)
    print("PART B — w53a's own design refitted by GLS, on all files and on the tight range")
    print("=" * 96)
    fams_ok = set(PR.FAMS) | {"h3"}
    bn = sorted(set(k for k in LB
                    if os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                    and PR.family(k) != "member" and PR.family(k) in fams_ok) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in bn}
    cvb = {k: fast_auc(y, V[k]) for k in bn}
    Eb = W63A.fit(bn, LB, cvb, V, y, beta_c)
    Sb, colb = Eb["Sxx"], Eb["col"]
    del V
    ic = PR.NAMES.index("cv6")

    def armb(keys, tag):
        keys = [k for k in keys if k in LB]
        fams = set(PR.family(k) for k in keys)
        js = [colb[k] for k in keys]
        cv6 = np.array([(cvb[k] - PR.MU) / U for k in keys])
        lb6 = np.array([LB[k] / U for k in keys])
        X = np.vstack([PR._design(c6, PR.family(k), PR.is_std(k), PR.is_corr(k), PR.new_era(k))
                       for c6, k in zip(cv6, keys)])
        keep = [i for i in range(X.shape[1]) if np.ptp(X[:, i]) > 0 or i == 0]
        drop = [PR.NAMES[i] for i in range(X.shape[1]) if i not in keep]
        Xk = X[:, keep]
        Om = Sb[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
        G = gls(Xk, lb6, Om)
        k_ic = keep.index(ic)
        b, se = float(G["b"][k_ic]), float(G["se_scaled"][k_ic])
        print(f"\n  {tag}  n={len(keys)}  families {len(fams)}  cv6 range "
              f"{cv6.min():+8.1f}..{cv6.max():+8.1f}  chi2/dof {G['chi2']/G['dof']:.3f}")
        if drop:
            print(f"    dropped constant columns: {drop}")
        print(f"    GLS cv6 = {b:+.4f}  se {se:.4f}  95% CI [{b-1.96*se:+.4f}, {b+1.96*se:+.4f}] "
              f"  (frozen OLS {PR.BETA[ic]:+.4f})")
        return dict(tag=tag, n=len(keys), cv6=b, se=se, ci=[b - 1.96 * se, b + 1.96 * se])

    B_all = armb(bn, "ALL   ")
    B_tight = armb([k for k in bn if cvb[k] >= CV_FLOOR], "TIGHT ")

    # ---------------------------------------------------------------- the reading ------------
    print("\n" + "=" * 96)
    print("THE READING, BY THE RULE FIXED IN THIS FILE'S DOCSTRING BEFORE THE ARMS RAN")
    print("=" * 96)
    overlap = not (A_tight["ci"][1] < A_all["ci"][0] or A_all["ci"][1] < A_tight["ci"][0])
    artefact = (not A_tight["excl1"]) and A_all["excl1"]
    print(f"  ALL   beta_total {A_all['beta']:.4f} CI [{A_all['ci'][0]:.4f}, {A_all['ci'][1]:.4f}]")
    print(f"  TIGHT beta_total {A_tight['beta']:.4f} CI [{A_tight['ci'][0]:.4f}, {A_tight['ci'][1]:.4f}]")
    print(f"  FRESH beta_total {A_fresh['beta']:.4f} CI [{A_fresh['ci'][0]:.4f}, {A_fresh['ci'][1]:.4f}]")
    print(f"\n  intervals overlap: {overlap}")
    print(f"  LEVERAGE ARTEFACT (tight contains 1.0 while all excludes it): {artefact}")
    if artefact:
        print("  ⚠⚠ w66a's PART A headline must be RESTATED as extrapolation failure only.")
    else:
        print("  => w66a's PART A slope survives dropping the three extrapolated files.")
    print(f"\n  PART B: cv6 ALL {B_all['cv6']:+.4f} +- {B_all['se']:.4f}, "
          f"TIGHT {B_tight['cv6']:+.4f} +- {B_tight['se']:.4f}, frozen {PR.BETA[ic]:+.4f}")

    # ---- what it costs: the CV needed to top the public board, at each slope ---------------
    top = float(json.load(open(os.path.join(HERE, "w63a_setprice.json")))
                .get("top_public", float("nan"))) if os.path.exists(
                    os.path.join(HERE, "w63a_setprice.json")) else float("nan")
    best_cv = max(cvb[k] for k in bn if k in LB)
    best_lb = max(LB.values())
    print("\n" + "=" * 96)
    print("WHAT THE SLOPE COSTS — the CV needed to gain one reporting step (1e-5) of public LB")
    print("=" * 96)
    for nm, s in (("frozen OLS base", float(PR.BASE_SLOPE)),
                  ("frozen OLS era", float(PR.ERA_SLOPE)),
                  ("GLS all", B_all["cv6"]), ("GLS tight", B_tight["cv6"])):
        print(f"  {nm:16s} slope {s:+.4f}/e-6  ->  need {10.0/s:8.2f}e-6 of CV per LB step")
    print(f"\n  account best LB {best_lb:.5f}; best scored CV {best_cv:.10f}")

    json.dump(dict(part_a=dict(all=A_all, tight=A_tight, fresh=A_fresh),
                   part_b=dict(all=B_all, tight=B_tight),
                   overlap=bool(overlap), artefact=bool(artefact),
                   frozen_cv6=float(PR.BETA[ic]), cv_floor=CV_FLOOR),
              open(os.path.join(HERE, "w66b_leverage.json"), "w"), indent=1)
    print(f"\nwrote experiments/w66b_leverage.json")


if __name__ == "__main__":
    main()
