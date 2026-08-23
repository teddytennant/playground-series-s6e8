"""w66a — is the queue pricer calibrated OUT OF SAMPLE, and do its own standard errors mean
anything?  Registered in experiments/w66_prereg.txt, committed d7db059 BEFORE this file existed.

The issued angle is consolidation: "check the CV-to-LB gap across every experiment so far".
w28a cut that table on 2026-08-19 over 89 stems. Forty files and three pricer versions later,
two questions are open and neither has ever been measured.

PART A. `w53a_pricer` is frozen on the 93 rows of `w52b_cvlb93.csv`. Twenty-eight scored stems
are absent from that table and nineteen of them carry an OOF, TEN being the 2026-08-23 sends
whose predictions went into the submission descriptions before the scores existed. That is a
genuine out-of-sample calibration sample. The calibration slope beta_total = 1 + c from a GLS
of (LB - pred) on (pred - mean pred) answers "does a predicted e-6 buy a realised e-6".

⛔ THE RECORDED STRINGS ARE NOT USED. Only 10 of the 42 description-recorded predictions
reproduce under the live pricer (08-21's four are off by -28e-6): they were written by three
pricer versions. Every `pred` here is RECOMPUTED from the live frozen model, on a CV RECOMPUTED
from the .npy on disk. w64: a stamp is not a comparison.

PART B. w53a fits by plain `np.linalg.lstsq` and `w30b_corrterm.json` reports its ses as if the
93 LB values were independent draws. They are not: they are ONE draw of ONE public slice, and
w64 measured those errors correlating at ~0.9999. OLS stays unbiased under that; its SES DO
NOT. This refits w53a's own design by GLS under Sxx = St + Sp plus the LB rounding term and
compares coefficient-by-coefficient and se-by-se.

The covariance estimator is IMPORTED from w63a_setprice, not re-implemented.

    .venv/bin/python experiments/w66a_pricercal.py
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
import w53a_pricer as PR                                       # noqa: E402 — the FROZEN model
import w63a_setprice as W63A                                   # noqa: E402 — IMPORTED, not copied
import check_selection as CS                                   # noqa: E402 — for R1

COMP = "playground-series-s6e8"
U = 1e-6
LB_STEP = 1e-5                        # the public LB is reported to five decimals
ROUND_VAR = (LB_STEP / U) ** 2 / 12.0  # 8.333 in units of 1e-12; sd 2.887e-6

FIT_TABLE = os.path.join(HERE, "w52b_cvlb93.csv")
WANTED_LOCK = {"w36_ad199stdcorr.csv", "w23_ad187stdcorr.csv"}   # R1

# registered bars
P2B_LO, P2B_HI = 0.55, 1.20           # P2b
P3_BAR = 12.0                         # P3, e-6
P4A_BAR = 2.0                         # P4a
P4B_LO, P4B_HI = 0.01, 0.50           # P4b
P5_BAR = 0.30                         # P5 and R2 limb 1
R2_RATIO_BAR = 1.0                    # R2 limb 2
INJECT = (-0.45, 0.0, +0.45)          # P6
N_PERM = 2000                         # P7

FAILURES = 0
RESULTS = {}


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def verdict(tag: str, ok: bool, detail: str) -> bool:
    RESULTS[tag] = bool(ok)
    print(f"  {'✅' if ok else '🔴'} {tag:5s} {detail}")
    return ok


def load_board():
    """The live scored board. Same read as w63a/w64a: one page, asserted not truncated."""
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores are not deterministic"
    return agg["max"].to_dict()


# =============================================================================================
# P1 — THE GATE. Re-derive w53a's frozen fit from its own table, in this process.
# =============================================================================================
def gate_p1():
    print("=" * 96)
    print("GATE P1 — the frozen pricer must be reproducible from w52b_cvlb93.csv IN THIS PROCESS")
    print("=" * 96)
    t = pd.read_csv(FIT_TABLE)
    X = np.vstack([PR._design(r.cv6, r.fam, r["std"], r["corr"], r.era) for _, r in t.iterrows()])
    y = t.lb6.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    dof = len(t) - X.shape[1]
    sd = float(np.sqrt(float(r @ r) / dof))
    db = float(np.max(np.abs(beta - PR.BETA)))
    dsd = abs(sd - PR.RESID_SD)
    print(f"  {len(t)} rows, {X.shape[1]} columns, dof {dof}")
    print(f"  max |re-derived BETA - PR.BETA| = {db:.3e}")
    print(f"  |re-derived resid_sd - PR.RESID_SD| = {dsd:.3e}   (both {sd:.6f})")
    print(f"  frozen cv6 = {PR.COEFS['cv6']:+.4f}/e-6   BASE_SLOPE {PR.BASE_SLOPE:+.4f}   "
          f"ERA_SLOPE {PR.ERA_SLOPE:+.4f} ({100*PR.ERA_SLOPE/PR.BASE_SLOPE:.1f}% of base)")
    ok = verdict("P1", db < 1e-9 and dsd < 1e-9, f"max coef diff {db:.2e}, sd diff {dsd:.2e}")
    if not ok:
        fail("P1: the frozen pricer does not re-derive -- everything below is VOID")
    return t, X, y, dof


# =============================================================================================
# the shared GLS solver. One implementation; PART A and PART B both call it.
# =============================================================================================
def gls(X, z, Om):
    """GLS with the one-sided chi2/dof interval scaling, plus OLS at the SAME variance scale.

    The OLS arm uses s2 = mean(diag(Om)) so the se RATIO reported downstream is a statement
    about the CORRELATION STRUCTURE and not about two different variance units.
    """
    iOm = np.linalg.inv(Om)

    def solve(W):
        XtW = X.T @ W
        V = np.linalg.inv(XtW @ X)
        return V @ (XtW @ z), V

    b_g, V_g = solve(iOm)
    s2 = float(np.mean(np.diag(Om)))
    b_o, V_o = solve(np.eye(len(z)) / s2)
    resid = z - X @ b_g
    chi2 = float(resid @ iOm @ resid)
    dof = len(z) - X.shape[1]
    # ⚠ ONE-SIDED. An se set by the covariance MODEL is only trustworthy if the scatter agrees;
    # a model that happens to fit well must never be allowed to BUY a narrower interval.
    scale = math.sqrt(max(chi2 / dof, 1.0))
    return dict(b=b_g, V=V_g, se=np.sqrt(np.diag(V_g)), b_ols=b_o,
                se_ols=np.sqrt(np.diag(V_o)), chi2=chi2, dof=dof, scale=scale,
                se_scaled=np.sqrt(np.diag(V_g)) * scale, iOm=iOm)


# =============================================================================================
# PART A — the out-of-sample calibration slope
# =============================================================================================
def part_a(y, beta_c, LB):
    print("\n" + "=" * 96)
    print("PART A — OUT-OF-SAMPLE CALIBRATION OF THE FROZEN PRICER")
    print("=" * 96)
    fit_stems = set(pd.read_csv(FIT_TABLE).stem)
    oos = sorted(k for k in LB if k not in fit_stems
                 and os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                 and PR.family(k) != "member")
    # WANTED must be in the covariance build (w63a.fit indexes it); it is NOT in the regression.
    names = sorted(set(oos) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}
    E = W63A.fit(names, LB, cv, V, y, beta_c)
    Sxx, col = E["Sxx"], E["col"]
    del V

    js = [col[k] for k in oos]
    pred = np.array([PR.predict_lb(cv[k], k) for k in oos])
    lb = np.array([LB[k] for k in oos])
    z = (lb - pred) / U
    px = (pred - pred.mean()) / U
    X = np.column_stack([np.ones(len(oos)), px])

    fresh = [k for k in oos if k in FRESH10]
    print(f"\n  {len(oos)} scored stems absent from the {len(fit_stems)}-row fit table and "
          f"carrying an OOF")
    print(f"  of which {len(fresh)} were sent 2026-08-23 with their prediction recorded BEFORE "
          f"the score existed")
    print(f"  pred spread {px.min():+.1f} .. {px.max():+.1f} e-6 (sd {px.std(ddof=1):.1f}); "
          f"rounding sd {math.sqrt(ROUND_VAR):.3f}e-6")
    print(f"\n  {'stem':34s} {'fam':8s} {'CV':>13s} {'pred':>9s} {'LB':>9s} {'z e-6':>9s}")
    for k, p, l, zz in sorted(zip(oos, pred, lb, z), key=lambda r: -r[1]):
        print(f"  {k:34s} {PR.family(k):8s} {cv[k]:13.10f} {p:9.6f} {l:9.5f} {zz:+9.2f}")

    def run(S):
        Om = S[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
        return gls(X, z, Om)

    A = run(Sxx)
    c, se = float(A["b"][1]), float(A["se_scaled"][1])
    alpha, ase = float(A["b"][0]), float(A["se_scaled"][0])
    lo, hi = 1 + c - 1.96 * se, 1 + c + 1.96 * se
    print(f"\n  chi2 = {A['chi2']:.1f} on {A['dof']} dof (chi2/dof {A['chi2']/A['dof']:.3f}) "
          f"=> intervals scaled by {A['scale']:.3f}")
    print(f"  alpha (mean OOS residual) = {alpha:+.3f} e-6, se {ase:.3f}")
    print(f"  c     = {c:+.4f}  se {float(A['se'][1]):.4f} -> {se:.4f} scaled")
    print(f"  beta_total = 1 + c = {1+c:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    verdict("P2a", (1 + c) < 1.0, f"beta_total = {1+c:.4f} {'<' if 1+c < 1 else '>='} 1.0")
    verdict("P2b", P2B_LO <= (1 + c) <= P2B_HI,
            f"beta_total {1+c:.4f} in [{P2B_LO}, {P2B_HI}]? ")
    p2c = verdict("P2c", lo > 0.0 or hi < 0.0, f"CI [{lo:.4f}, {hi:.4f}] excludes 0")
    verdict("P2d", lo <= 1.0 <= hi, f"CI [{lo:.4f}, {hi:.4f}] contains 1.0")
    verdict("P3", abs(alpha) <= P3_BAR, f"|alpha| = {abs(alpha):.3f} <= {P3_BAR} e-6")
    if not p2c:
        print("  ⚠⚠ P2c FAILED — PART A HAS NO RESOLUTION. Per w66_prereg §5, P2a/P2b/P2d "
              "above are VOID, not confirmations.")

    # ---- P6 POWER CONTROL ------------------------------------------------------------------
    Om = Sxx[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
    power = {}
    for inj in INJECT:
        Ai = gls(X, z + inj * px, Om)
        rec = float(Ai["b"][1])
        power[f"{inj:+.2f}"] = dict(recovered=rec, err=rec - (c + inj),
                                    detected=bool(abs(rec) > 1.96 * float(Ai["se_scaled"][1])))
    print("\n  P6 POWER CONTROL — inject a known calibration deviation, re-fit, must recover it:")
    for k, v in power.items():
        print(f"    injected {k}  ->  recovered {v['recovered']:+.4f}  (error {v['err']:+.2e})  "
              f"detected at 2sigma: {v['detected']}")
    p6 = (all(abs(v["err"]) < 1e-9 for v in power.values())
          and power[f"{-0.45:+.2f}"]["detected"])
    verdict("P6", p6, "recovery exact and the -0.45 injection is detected")
    if not p6:
        fail("P6: the instrument cannot see a -0.45 calibration deviation -- P2d is uninformative")

    # ---- P7 NEGATIVE CONTROL ---------------------------------------------------------------
    rng = np.random.default_rng(66)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        pp = rng.permutation(px)
        Xp = np.column_stack([np.ones(len(js)), pp])
        null[i] = float(gls(Xp, z, Om)["b"][1])
    p_perm = float((np.abs(null) >= abs(c)).mean())
    print(f"\n  P7 NEGATIVE CONTROL — {N_PERM} permutations of the pred column:")
    print(f"    null centre {null.mean():+.4f}, null sd {null.std(ddof=1):.4f}, "
          f"observed |c| = {abs(c):.4f}, permutation p = {p_perm:.4f}")
    verdict("P7", abs(null.mean()) < 0.05,
            f"permutation null is centred at {null.mean():+.4f} (|.| < 0.05)")

    # ---- P9 SENSITIVITY: Omega' = Sp + s^2 (Sxx - Sp) + R -----------------------------------
    # Omega is the covariance of (LB - CV); pred applies a slope >1 to CV, so Omega understates
    # the dispersion of (LB - pred). Rebuilding with the transfer part scaled by BASE_SLOPE^2
    # is the conservative arm: the sign of c must survive and the se must WIDEN.
    Sp = E.get("Sp")
    if Sp is None:
        Om2 = Sxx[np.ix_(js, js)] * PR.BASE_SLOPE ** 2 + np.eye(len(js)) * ROUND_VAR
        note = "Sp not exported by w63a.fit -- the whole of Sxx is scaled, which is STRICTLY " \
               "more conservative than scaling only its transfer part"
    else:
        Spj = Sp[np.ix_(js, js)]
        Om2 = Spj + (Sxx[np.ix_(js, js)] - Spj) * PR.BASE_SLOPE ** 2 + np.eye(len(js)) * ROUND_VAR
        note = "transfer part scaled by BASE_SLOPE^2"
    A2 = gls(X, z, Om2)
    c2, se2 = float(A2["b"][1]), float(A2["se_scaled"][1])
    print(f"\n  P9 SENSITIVITY ({note}):")
    print(f"    c = {c2:+.4f}  se {se2:.4f}   (base c {c:+.4f}, se {se:.4f})")
    verdict("P9", (np.sign(c2) == np.sign(c) or abs(c2) < 1e-12) and se2 > se,
            f"sign preserved and se widened ({se2:.4f} > {se:.4f})")

    return dict(n=len(oos), n_fresh=len(fresh), oos=oos, alpha=alpha, alpha_se=ase,
                c=c, c_se=se, beta_total=1 + c, ci=[lo, hi], chi2=A["chi2"], dof=A["dof"],
                scale=A["scale"], power=power, perm_p=p_perm, perm_sd=float(null.std(ddof=1)),
                perm_centre=float(null.mean()), c_sens=c2, se_sens=se2,
                pred=pred.tolist(), lb=lb.tolist(), z=z.tolist()), cv


# =============================================================================================
# PART B — w53a's own design, refitted by GLS under the shared public slice
# =============================================================================================
def part_b(y, beta_c, LB):
    print("\n" + "=" * 96)
    print("PART B — DO w30b's STANDARD ERRORS MEAN ANYTHING? THE SAME DESIGN, UNDER Sxx + R")
    print("=" * 96)
    names = sorted(k for k in LB
                   if os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))
                   and PR.family(k) != "member")
    fams_ok = set(PR.FAMS) | {"h3"}
    names = [k for k in names if PR.family(k) in fams_ok]
    names = sorted(set(names) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}
    E = W63A.fit(names, LB, cv, V, y, beta_c)
    Sxx, col = E["Sxx"], E["col"]
    del V

    use = [k for k in names if k in LB]
    js = [col[k] for k in use]
    cv6 = np.array([(cv[k] - PR.MU) / U for k in use])
    lb6 = np.array([LB[k] / U for k in use])
    X = np.vstack([PR._design(c6, PR.family(k), PR.is_std(k), PR.is_corr(k), PR.new_era(k))
                   for c6, k in zip(cv6, use)])
    # ⚠ The regression is on LB in e-6 with the FULL design (an intercept lives in the design),
    # so this is w53a's own model, refitted -- not a different parameterisation of it.
    Om = Sxx[np.ix_(js, js)] + np.eye(len(js)) * ROUND_VAR
    B = gls(X, lb6, Om)

    ic = PR.NAMES.index("cv6")
    i0 = PR.NAMES.index("const")
    print(f"\n  {len(use)} scored files with an OOF and a family in the fitted design "
          f"({len(set(PR.family(k) for k in use))} families)")
    print(f"  chi2 = {B['chi2']:.1f} on {B['dof']} dof (chi2/dof {B['chi2']/B['dof']:.3f}) "
          f"=> scale {B['scale']:.3f}")
    print(f"\n  {'term':14s} {'frozen OLS':>11s} {'w30b se':>9s} {'GLS':>11s} {'se':>9s} "
          f"{'OLS(here)':>11s} {'se':>9s} {'se ratio':>9s}")
    w30b_se = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))["ses"]
    ratios = {}
    for i, nm in enumerate(PR.NAMES):
        r = float(B["se"][i] / B["se_ols"][i])
        ratios[nm] = r
        key = {"const": "const", "cv6": "cv_e6"}.get(nm, nm)
        w = w30b_se.get(key)
        print(f"  {nm:14s} {PR.BETA[i]:+11.4f} {('%9.4f' % w) if w else '        -':>9s} "
              f"{float(B['b'][i]):+11.4f} {float(B['se'][i]):9.4f} "
              f"{float(B['b_ols'][i]):+11.4f} {float(B['se_ols'][i]):9.4f} {r:9.3f}")

    d_cv = float(B["b"][ic]) - float(PR.BETA[ic])
    print(f"\n  GLS cv6 - frozen cv6 = {d_cv:+.4f} per e-6")
    verdict("P4a", ratios["const"] > P4A_BAR,
            f"se ratio on const = {ratios['const']:.3f} > {P4A_BAR}")
    verdict("P4b", P4B_LO <= ratios["cv6"] <= P4B_HI,
            f"se ratio on cv6 = {ratios['cv6']:.4f} in [{P4B_LO}, {P4B_HI}]")
    verdict("P5", abs(d_cv) < P5_BAR, f"|GLS cv6 - {PR.BETA[ic]:.4f}| = {abs(d_cv):.4f} < {P5_BAR}")

    r2 = abs(d_cv) > P5_BAR or ratios["cv6"] > R2_RATIO_BAR
    print(f"\n  R2: refit-under-GLS required?  {'YES' if r2 else 'NO'}  "
          f"(|dcv| {abs(d_cv):.4f} > {P5_BAR}: {abs(d_cv) > P5_BAR}; "
          f"se ratio {ratios['cv6']:.4f} > {R2_RATIO_BAR}: {ratios['cv6'] > R2_RATIO_BAR})")
    return dict(n=len(use), terms=PR.NAMES, frozen=PR.BETA.tolist(), gls=B["b"].tolist(),
                gls_se=B["se"].tolist(), ols=B["b_ols"].tolist(), ols_se=B["se_ols"].tolist(),
                ratios=ratios, chi2=B["chi2"], dof=B["dof"], scale=B["scale"],
                d_cv=d_cv, r2_fires=bool(r2))


FRESH10: set = set()


def main() -> None:
    global FRESH10
    write = "--no-write" not in sys.argv
    beta_c = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    LB = load_board()
    print(f"live board: {len(LB)} distinct scored files, coupling beta {beta_c:.6f}\n")

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    FRESH10 = set(sub[sub.date.str[:10] == "2026-08-23"].fileName.str.replace(
        r"\.csv$", "", regex=True))

    gate_p1()
    A, _ = part_a(y, beta_c, LB)
    B = part_b(y, beta_c, LB)

    # ---- R1. The consolidation run asserts the selection is UNCHANGED. ---------------------
    print("\n" + "=" * 96)
    print("R1 — THE SELECTION IS UNCHANGED BY THIS RUN, BY CONSTRUCTION")
    print("=" * 96)
    print(f"  check_selection.WANTED = {sorted(CS.WANTED)}")
    r1 = set(CS.WANTED) == WANTED_LOCK
    verdict("R1", r1, f"WANTED is the locked pair {sorted(WANTED_LOCK)}")
    if not r1:
        fail("R1: WANTED moved -- a consolidation run must not change the selection")
    print("  Predicted LB is strictly increasing in CV within a family, so no recalibration of "
          "the pricer\n  can reorder a CV ranking. R2 is about PRICES, not about SELECTION.")

    print("\n" + "=" * 96)
    print(f"REGISTERED PREDICTIONS: {sum(RESULTS.values())} of {len(RESULTS)} confirmed")
    print("=" * 96)
    for k, v in RESULTS.items():
        print(f"  {'✅' if v else '🔴'} {k}")
    print(f"\nFAILURES {FAILURES}")

    if write:
        out = os.path.join(HERE, "w66a_pricercal.json")
        json.dump(dict(part_a=A, part_b=B, results=RESULTS, failures=FAILURES,
                       frozen_cv6=float(PR.BETA[PR.NAMES.index("cv6")]),
                       base_slope=PR.BASE_SLOPE, era_slope=PR.ERA_SLOPE),
                  open(out, "w"), indent=1)
        print(f"\nwrote {out}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
