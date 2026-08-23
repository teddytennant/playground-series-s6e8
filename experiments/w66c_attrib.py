"""w66c — PART A and PART B disagree, and this decides which reading is right.

⚠⚠ DECLARED POST-HOC, like w66b. Not registered in w66_prereg.txt.

w66b left two findings that look contradictory:
  * PART A, restricted to the pricer's operating range: the frozen OLS pricer's predicted LB is
    calibrated 1:1 out of sample (beta_total 0.987 on 16 files, 1.136 on the ten 08-23 sends).
  * PART B: refitting w53a's OWN design by GLS moves cv6 from +1.833 to +0.718, and the 95% CI
    excludes the frozen value on BOTH the all-112 and the tight-109 arms.

Both cannot be a statement about PREDICTION. The reconciliation on offer is that the design is
collinear -- cv6 against the family dummies and the std/corr flags -- so two fits can predict
almost identically while ATTRIBUTING the same fitted value to different columns. That is not a
harmless disagreement: `w53a_pricer.cv_needed` and every "how much CV buys an LB step" statement
in this workspace is a COUNTERFACTUAL query, and a counterfactual reads the attribution, not the
fit. `w26g_send.hijack_risk` gates the send path on `pred_lb`, which reads the fit.

Three arms, and the reading rule is fixed here before they run:
  A. FIT EQUIVALENCE. On the 112 scored files, how far apart are the OLS and GLS fitted values?
     If max|difference| is small against the 7.7e-6 residual sd, the two models are
     observationally near-equivalent and the disagreement is attribution, not accuracy.
  B. HONEST HELD-OUT. Refit the SAME design by GLS on the SAME 93 fit-table rows -- never on
     the 19 -- then score the frozen OLS pricer and the GLS refit on the 19 out-of-sample files.
     Lower held-out RMSE wins. This is the only arm that can say one pricer is BETTER.
  C. THE COUNTERFACTUAL. `cv_needed` under each, for the one query the workspace actually asks.

    .venv/bin/python experiments/w66c_attrib.py
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
from w66a_pricercal import gls, load_board, ROUND_VAR, U, FIT_TABLE   # noqa: E402

COMP = "playground-series-s6e8"


def design(keys, cv):
    return np.vstack([PR._design((cv[k] - PR.MU) / U, PR.family(k), PR.is_std(k),
                                 PR.is_corr(k), PR.new_era(k)) for k in keys])


def main():
    beta_c = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    LB = load_board()
    t93 = pd.read_csv(FIT_TABLE)
    fit_stems = set(t93.stem)

    fams_ok = set(PR.FAMS) | {"h3"}
    scored = sorted(k for k in LB
                    if os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                    and PR.family(k) != "member" and PR.family(k) in fams_ok)
    names = sorted(set(scored) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}
    E = W63A.fit(names, LB, cv, V, y, beta_c)
    Sxx, col = E["Sxx"], E["col"]
    del V

    infit = [k for k in scored if k in fit_stems]
    oos = [k for k in scored if k not in fit_stems]
    print(f"{len(scored)} scored files with an OOF and a fitted family: "
          f"{len(infit)} in the 93-row fit table, {len(oos)} out of it\n")

    # ------------------------------------------------------------------ ARM A: fit equivalence
    print("=" * 96)
    print("ARM A — are the OLS and GLS fits observationally distinguishable on the 112?")
    print("=" * 96)
    js = [col[k] for k in scored]
    X = design(scored, cv)
    lb6 = np.array([LB[k] / U for k in scored])
    Om = Sxx[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
    G = gls(X, lb6, Om)
    f_gls, f_ols = X @ G["b"], X @ G["b_ols"]
    d = f_gls - f_ols
    print(f"  in-sample RMSE  OLS {np.sqrt(np.mean((lb6-f_ols)**2)):.3f}e-6   "
          f"GLS {np.sqrt(np.mean((lb6-f_gls)**2)):.3f}e-6")
    print(f"  fitted-value difference: mean {d.mean():+.3f}, sd {d.std(ddof=1):.3f}, "
          f"max|.| {np.abs(d).max():.3f} e-6   (w53a residual sd {PR.RESID_SD:.3f}e-6)")
    print(f"  corr(fitted_OLS, fitted_GLS) = {np.corrcoef(f_ols, f_gls)[0,1]:.6f}")
    print(f"  => the disagreement is {'ATTRIBUTION' if np.abs(d).max() < 3*PR.RESID_SD else 'ACCURACY'}, "
          f"on the max|difference| vs 3x residual sd test fixed in the docstring")

    # ------------------------------------------------------------------ ARM B: honest held-out
    print("\n" + "=" * 96)
    print("ARM B — refit by GLS on the SAME 93 rows, then score BOTH on the 19 held-out files")
    print("=" * 96)
    jf = [col[k] for k in infit]
    Xf = design(infit, cv)
    yf = np.array([LB[k] / U for k in infit])
    Omf = Sxx[np.ix_(jf, jf)] + np.eye(len(jf)) * ROUND_VAR
    Gf = gls(Xf, yf, Omf)
    print(f"  refit on {len(infit)} of the 93 rows (a fit-table row with no OOF on disk cannot "
          f"enter the covariance)")
    print(f"  {'term':14s} {'frozen OLS':>11s} {'GLS(93)':>11s} {'se':>9s}")
    for i, nm in enumerate(PR.NAMES):
        print(f"  {nm:14s} {PR.BETA[i]:+11.4f} {float(Gf['b'][i]):+11.4f} "
              f"{float(Gf['se_scaled'][i]):9.4f}")

    Xo = design(oos, cv)
    yo = np.array([LB[k] / U for k in oos])
    p_ols = np.array([PR.predict_lb(cv[k], k) / U for k in oos])
    p_gls = Xo @ Gf["b"]
    for tag, p in (("frozen OLS", p_ols), ("GLS(93)  ", p_gls)):
        r = yo - p
        print(f"\n  held-out {tag}: RMSE {np.sqrt(np.mean(r**2)):8.2f}e-6   "
              f"MAE {np.abs(r).mean():8.2f}e-6   bias {r.mean():+8.2f}e-6")
    # and on the 16 that are not 300e-6 extrapolations
    tight = [i for i, k in enumerate(oos) if cv[k] >= 0.9699]
    print(f"\n  restricted to the {len(tight)} held-out files inside the pricer's range:")
    for tag, p in (("frozen OLS", p_ols), ("GLS(93)  ", p_gls)):
        r = (yo - p)[tight]
        print(f"    {tag}: RMSE {np.sqrt(np.mean(r**2)):7.2f}e-6   MAE {np.abs(r).mean():7.2f}e-6   "
              f"bias {r.mean():+7.2f}e-6")

    # ------------------------------------------------------------------ ARM C: counterfactual
    print("\n" + "=" * 96)
    print("ARM C — the counterfactual the workspace actually asks: CV needed to top the board")
    print("=" * 96)
    best_lb = max(LB.values())
    best_cv = max(cv[k] for k in scored)
    stem = max(scored, key=lambda k: cv[k])
    target = best_lb + 1e-5
    print(f"  best scored CV {best_cv:.10f} ({stem}, family {PR.family(stem)}), "
          f"account best LB {best_lb:.5f}")
    print(f"  target: one reporting step above, {target:.5f}\n")
    for nm, s in (("frozen OLS base", float(PR.BASE_SLOPE)), ("frozen OLS era", float(PR.ERA_SLOPE)),
                  ("GLS(93) base", float(Gf["b"][PR.NAMES.index("cv6")])),
                  ("GLS(93) era", float(Gf["b"][PR.NAMES.index("cv6")]
                                        + Gf["b"][PR.NAMES.index("era:cv6")]))):
        gap6 = (target - PR.predict_lb(best_cv, stem)) / U
        print(f"  {nm:16s} slope {s:+.4f}/e-6 -> {10.0/s:7.2f}e-6 of CV per LB step; "
              f"the {gap6:+.1f}e-6 gap to target needs {gap6/s:+8.1f}e-6 more CV")

    json.dump(dict(n_scored=len(scored), n_infit=len(infit), n_oos=len(oos),
                   fit_diff_max=float(np.abs(d).max()), fit_diff_sd=float(d.std(ddof=1)),
                   rmse_in_ols=float(np.sqrt(np.mean((lb6-f_ols)**2))),
                   rmse_in_gls=float(np.sqrt(np.mean((lb6-f_gls)**2))),
                   gls93=Gf["b"].tolist(), gls93_se=Gf["se_scaled"].tolist(), terms=PR.NAMES,
                   heldout=dict(
                       ols_rmse=float(np.sqrt(np.mean((yo-p_ols)**2))),
                       gls_rmse=float(np.sqrt(np.mean((yo-p_gls)**2))),
                       ols_rmse_tight=float(np.sqrt(np.mean(((yo-p_ols)[tight])**2))),
                       gls_rmse_tight=float(np.sqrt(np.mean(((yo-p_gls)[tight])**2)))),
                   ), open(os.path.join(HERE, "w66c_attrib.json"), "w"), indent=1)
    print("\nwrote experiments/w66c_attrib.json")


if __name__ == "__main__":
    main()
