"""w67a — CAN the 08-24 sends decide the ABOVE-RANGE slope? Power, registered before the data.

w66 section 9 item 3 left R2 open and said the question "answers itself from the 08-24 sends
onward if someone keeps score". This file does not keep score -- the scores do not exist yet.
It asks the prior question, which w66 did not ask: **would keeping score decide anything?**

THE ESTIMAND (w67_prereg section 2, by reference, not restated). Above FIT_CV_MAX the frozen
OLS pricer and the GLS(93) refit differ in ONE quantity, the cv slope. Holding every other
term at what w53a already says:

    lb_hat(k) = ANCHOR(k) + s * excess(k)
    ANCHOR(k) = the FROZEN pricer's own prediction for k with its cv CLAMPED to FIT_CV_MAX
    excess(k) = (cv[k] - FIT_CV_MAX) * 1e6                      [ > 0 by construction ]

Every sendable above-range file is ad216/ad217, hence era, so the contrast is the ERA slope:
H_OLS s = +1.4425, H_GLS s = +0.9028, SEPARATION 0.5397 per e-6.

⚠ THE POINT. An se computed under IID would be a lie here. The nine files are nine transforms
of TWO packs (ad216, ad217); their public-LB errors are almost the same error. The covariance
is therefore taken from `w63a_setprice.fit` -- IMPORTED, never re-implemented, exactly as w66a
does -- and the IID number is reported only as the foil it is.

READ-ONLY. This module touches nothing on the send path: it does not import `w26d_queueprice`,
does not write the queue, and does not move WANTED. It writes exactly one artefact,
`w67a_aboveslope.json`, which freezes the two hypotheses' per-file predictions so that a run
after the 08-24 window can score them without re-deriving anything.

    .venv/bin/python experiments/w67a_aboveslope.py
"""
from __future__ import annotations

import contextlib, io, json, math, os, subprocess, sys

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
import check_selection as CS                                   # noqa: E402 — for P9

# ⚠ THE VETO IS IMPORTED, NEVER RE-LISTED. `w48e_order` prints its whole send report and reads
# the live board at import time, so the import is silenced -- but it is an IMPORT, so the veto
# this module tests is by construction the veto the SENDER enforces. Re-typing the list here
# would be the w66 section 6 defect (a rule enforced in one consumer is not enforced) in its
# purest form. Importing does NOT write: w48e's writes are under `--write`, and w26d's under
# `__main__` (w66 section 10).
with contextlib.redirect_stdout(io.StringIO()):
    import w48e_order as W48E                                  # noqa: E402

U = 1e-6
LB_STEP = 1e-5                         # the public LB is reported to five decimals
ROUND_VAR = (LB_STEP / U) ** 2 / 12.0  # 8.333 in units of 1e-12; sd 2.887e-6
QUEUE = os.path.join(HERE, "w23b_sendqueue.csv")
WANTED_LOCK = {"w36_ad199stdcorr.csv", "w23_ad187stdcorr.csv"}   # P9

# registered bars -- w67_prereg section 3. Typed here ONCE and never restated in prose.
P1_BAR = 1e-9
P2_N = 9                               # STRICT
P3_SEP, P3_TOL = 0.5397, 1e-4
P4_LO, P4_HI = 0.15, 0.35              # WEAK (declared contamination)
P5_LO, P5_HI = 1.0, 6.0                # FULL WEIGHT
R_NORES, R_CLEAN = 1.96, 3.92          # P6 buckets
P6_PREDICTED = "a"                     # WEAK
P7_BAR = 1e-9
P8_SIGMA = 3.0
N_PERM = 2000
PERM_SEED = 67

FAILURES = 0
RESULTS = {}


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print("  !! " + msg, flush=True)


def verdict(tag: str, ok: bool, detail: str) -> bool:
    RESULTS[tag] = bool(ok)
    print(f"  {'✅' if ok else '🔴'} {tag}: {detail}", flush=True)
    return bool(ok)


def gls(X, z, Om):
    """GLS with the one-sided chi2/dof interval scaling. Same construction as w66a.gls.

    ⚠ ONE-SIDED by design: a covariance MODEL that happens to fit well must never be allowed to
    BUY a narrower interval than the scatter supports.
    """
    iOm = np.linalg.inv(Om)
    XtW = X.T @ iOm
    V = np.linalg.inv(XtW @ X)
    b = V @ (XtW @ z)
    resid = z - X @ b
    chi2 = float(resid @ iOm @ resid)
    dof = max(len(z) - X.shape[1], 1)
    scale = math.sqrt(max(chi2 / dof, 1.0))
    return dict(b=b, se=np.sqrt(np.diag(V)), se_scaled=np.sqrt(np.diag(V)) * scale,
                chi2=chi2, dof=dof, scale=scale)


# =============================================================================================
# P1 — THE GATE. A run that cannot reproduce the frozen fit may not reason about its slope.
# =============================================================================================
def gate_p1():
    print("=" * 96)
    print("P1 GATE — re-derive the frozen pricer's BETA from its own 93-row table, in-process")
    print("=" * 96)
    t = pd.read_csv(PR.FIT_TABLE if hasattr(PR, "FIT_TABLE")
                    else os.path.join(HERE, "w52b_cvlb93.csv"))
    X = np.array([PR._design(r.cv6, r.fam, r.std, r.corr, r.era) for r in t.itertuples()])
    b, *_ = np.linalg.lstsq(X, t.lb6.to_numpy(), rcond=None)
    d = float(np.abs(b - PR.BETA).max())
    print(f"  rows {len(t)}   max |re-derived BETA - PR.BETA| = {d:.3e}")
    verdict("P1", d < P1_BAR, f"BETA re-derives to {d:.3e} < {P1_BAR:.0e}")
    if d >= P1_BAR:
        fail("P1: the frozen fit does not reproduce -- nothing below is trustworthy")
    return d


# =============================================================================================
# P2/P3 — the sample and the separation
# =============================================================================================
def select_above():
    print("\n" + "=" * 96)
    print("P2/P3 — THE SAMPLE ABOVE FIT_CV_MAX, AND WHAT THE TWO MODELS DISAGREE ABOUT")
    print("=" * 96)
    q = pd.read_csv(QUEUE)
    q["stem"] = q.file.str.replace(r"\.csv$", "", regex=True)
    q = q[~q.sent.astype(bool)]
    above = q[q.cv > PR.FIT_CV_MAX].copy()
    above["fam"] = [PR.family(s) for s in above.stem]
    # ⚠ w66 section 6: NEVER let a `member` row into a fit or a max(CV). Enforced HERE because a
    # rule enforced in one consumer is not enforced -- the consumers are enumerated from the RULE.
    memb = above[above.fam == "member"]
    above = above[above.fam != "member"].sort_values("cv", ascending=False)
    assert len(memb) > 0, "the member filter removed nothing -- it is not being exercised"
    print(f"  queue {len(q)} unsent; {len(above) + len(memb)} above FIT_CV_MAX "
          f"({PR.FIT_CV_MAX:.10f}); {len(memb)} dropped as family `member`: "
          f"{sorted(memb.stem)}")

    era = [bool(PR.new_era(s)) for s in above.stem]
    print(f"\n  {'stem':26s} {'fam':8s} {'CV':>13s} {'excess e-6':>11s} {'era':>5s}")
    for r, e in zip(above.itertuples(), era):
        print(f"  {r.stem:26s} {PR.family(r.stem):8s} {r.cv:13.10f} "
              f"{(r.cv - PR.FIT_CV_MAX) * 1e6:11.2f} {str(e):>5s}")

    ok2 = len(above) == P2_N and all(era)
    verdict("P2", ok2, f"{len(above)} sendable above-range files (need exactly {P2_N}), "
                       f"all era: {all(era)}")

    gls93 = json.load(open(os.path.join(HERE, "w66c_attrib.json")))
    terms, b = gls93["terms"], gls93["gls93"]
    s_gls = float(b[terms.index("cv6")] + b[terms.index("era:cv6")])
    s_ols = float(PR.ERA_SLOPE)
    sep = abs(s_ols - s_gls)
    print(f"\n  H_OLS era slope {s_ols:+.4f}   H_GLS era slope {s_gls:+.4f}   "
          f"SEPARATION {sep:.4f} per e-6")
    verdict("P3", abs(sep - P3_SEP) < P3_TOL,
            f"separation {sep:.4f}, registered {P3_SEP:.4f}, |diff| {abs(sep - P3_SEP):.2e}")
    return above, s_ols, s_gls, sep


# =============================================================================================
# V — VALIDITY. Asked BEFORE power, because power on an unobservable sample is not power.
# =============================================================================================
def validity(above):
    """Can any of these nine ever be SENT? If not, the instrument is void whatever its se.

    ⚠ THIS IS THE SECTION THE PREREG DID NOT HAVE. w67_prereg registered P4-P6 (power) and
    never asked whether the sample is observable. It is not: every file above FIT_CV_MAX is
    vetoed by `w48e_order.VETO` as ARM 216/217 -- CV inflated by `hboyang_mix` -- or is a
    `member` row that can never be sent. So `FIT_CV_MAX` equalling the best CV on record is
    NOT a coincidence to be noted (w66 section 5); it is a FIXED POINT of the veto.
    """
    print("\n" + "=" * 96)
    print("V — VALIDITY: is this sample OBSERVABLE? (asked before power, not after)")
    print("=" * 96)
    rows = []
    for r in above.itertuples():
        rows.append(dict(stem=r.stem, cv=float(r.cv), vetoed=r.stem in W48E.VETO,
                         fam=PR.family(r.stem)))
    n_vet = sum(x["vetoed"] for x in rows)
    free = [x["stem"] for x in rows if not x["vetoed"] and x["fam"] != "member"]
    print(f"  {len(rows)} above-range candidates; {n_vet} carry an explicit VETO in "
          f"w48e_order.VETO ({len(W48E.VETO)} vetoed overall)")
    for x in rows:
        tail = "VETOED — " + W48E.VETO[x["stem"]][:52] if x["vetoed"] else "free"
        print(f"    {'⛔' if x['vetoed'] else '  '} {x['stem']:26s} cv {x['cv']:.10f} {tail}")
    print(f"\n  SENDABLE, NON-VETOED, NON-member files above FIT_CV_MAX: {len(free)}")
    ok = len(free) == 0
    verdict("V1", ok, f"the above-range sample is EMPTY of sendable files ({len(free)} free) "
                      f"— the instrument is VOID, not underpowered")
    return dict(n_above=len(rows), n_vetoed=n_vet, n_free=len(free), free=free, rows=rows)


# =============================================================================================
# P4/P5/P6 — the power, IID as the foil and CORRELATED as the answer
# =============================================================================================
def power(above, y, beta_c, sep):
    print("\n" + "=" * 96)
    print("P4/P5/P6 — POWER. Would keeping score decide anything?")
    print("=" * 96)
    stems = list(above.stem)
    # ANCHOR: the frozen pricer's own prediction with cv CLAMPED to FIT_CV_MAX. Everything the
    # two models AGREE on lives here; `excess` carries the whole of what they disagree about.
    anchor = np.array([PR.predict_lb(PR.FIT_CV_MAX, k) for k in stems])
    excess = np.array([(c - PR.FIT_CV_MAX) * 1e6 for c in above.cv])
    X = np.column_stack([np.ones(len(stems)), excess])

    names = sorted(set(stems) | set(W63A.WANTED))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}
    # None of the nine is scored yet; fit() only needs the SCORED subset for its gamma term,
    # and WANTED is scored. The covariance itself is built from the OOF vectors, not from LB.
    LB = _board_subset(names)
    E = W63A.fit(names, LB, cv, V, y, beta_c)
    Sxx, col = E["Sxx"], E["col"]
    del V

    js = [col[k] for k in stems]
    S = Sxx[np.ix_(js, js)]
    Om_corr = S + np.eye(len(js)) * ROUND_VAR
    # per-point IID sd in e-6: the pricer's own residual sd plus LB rounding.
    # ⚠ PR.RESID_SD IS ALREADY IN e-6 -- w53a fits lb6/cv6, which w52b_cvlb93.csv stores
    # pre-scaled (lb6 971040.0, not 0.97104). Multiplying by 1e6 here inflated the IID foil by
    # a million and silently flipped P4 and P5. Asserted, so it cannot come back.
    assert 1.0 < PR.RESID_SD < 100.0, (PR.RESID_SD, "RESID_SD is not in e-6 -- units moved")
    sd_iid = math.sqrt(PR.RESID_SD ** 2 + ROUND_VAR)
    Om_iid = np.eye(len(js)) * sd_iid ** 2

    # The se of s_hat does NOT depend on the response, only on the design and Omega. Use a
    # noiseless response so `scale` is exactly 1 and the se reported is the MODEL se.
    def se_of(Om, s_true):
        z = s_true * excess
        G = gls(X, z, Om)
        return float(G["se"][1]), float(G["b"][1])

    se_iid, _ = se_of(Om_iid, 1.4425)
    se_corr, _ = se_of(Om_corr, 1.4425)
    print(f"  per-point IID sd {sd_iid:.3f}e-6 (RESID_SD {PR.RESID_SD:.3f} + rounding "
          f"{math.sqrt(ROUND_VAR):.3f})")
    print(f"  correlated Omega: mean diag {np.diag(Om_corr).mean():.2f}, "
          f"mean off-diag corr {_meancorr(Om_corr):.4f}")
    print(f"\n  se(s_hat)  IID {se_iid:.4f}   CORRELATED {se_corr:.4f}   "
          f"ratio {se_corr / se_iid:.3f}")

    verdict("P4", P4_LO <= se_iid <= P4_HI,
            f"IID se {se_iid:.4f} in [{P4_LO}, {P4_HI}] (WEAK — declared contamination)")
    ratio = se_corr / se_iid
    verdict("P5", P5_LO <= ratio <= P5_HI,
            f"se ratio {ratio:.3f} in [{P5_LO}, {P5_HI}] (FULL WEIGHT)")

    R = sep / se_corr
    bucket = "a" if R < R_NORES else ("b" if R < R_CLEAN else "c")
    label = {"a": "NO RESOLUTION", "b": "ONE-SIDED", "c": "CLEAN"}[bucket]
    print(f"\n  R = SEPARATION / se_corr = {sep:.4f} / {se_corr:.4f} = {R:.3f}  ->  "
          f"bucket ({bucket}) {label}")
    print(f"  95% CI half-width on s_hat = {1.96 * se_corr:.3f}; the two hypotheses are "
          f"{sep:.3f} apart")
    verdict("P6", bucket == P6_PREDICTED,
            f"bucket ({bucket}) {label}, predicted ({P6_PREDICTED}) (WEAK)")

    # ---- HOW MUCH MARGIN DOES THE VERDICT HAVE? --------------------------------------------
    # ⚠ se_corr above is the NOISELESS MODEL se: the response was synthesised on the line, so
    # chi2 = 0 and `gls`'s one-sided scale clamps to 1. Real data cannot make the se SMALLER
    # (the scaling is one-sided by construction) and will make it larger whenever the scatter
    # disagrees with Sxx. R = SEPARATION / se_corr is therefore an UPPER BOUND on the
    # resolution, and the bucket must be reported with the misfit that would demote it.
    scale_to_b = sep / R_CLEAN / se_corr
    scale_to_a = sep / R_NORES / se_corr
    print(f"\n  ⚠ R is an UPPER BOUND — se_corr is the noiseless MODEL se and `gls` scales it "
          f"by\n    sqrt(max(chi2/dof, 1)), which is ONE-SIDED. Margin on the verdict:")
    print(f"      chi2/dof >= {scale_to_b ** 2:.3f} (se x {scale_to_b:.3f}) demotes (c) -> (b)")
    print(f"      chi2/dof >= {scale_to_a ** 2:.3f} (se x {scale_to_a:.3f}) demotes (b) -> (a)")
    return dict(scale_to_b=float(scale_to_b), scale_to_a=float(scale_to_a),stems=stems, anchor=anchor, excess=excess, X=X, Om_corr=Om_corr,
                Om_iid=Om_iid, se_iid=se_iid, se_corr=se_corr, R=R, bucket=bucket,
                sd_iid=sd_iid, cv={k: cv[k] for k in stems})


def _board_subset(names):
    """The live board, restricted to `names`. Same read as w66a.load_board."""
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c",
                          "playground-series-s6e8", "-v", "--page-size", "500"],
                         capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores not deterministic"
    return {k: v for k, v in agg["max"].to_dict().items() if k in set(names)}


def _meancorr(Om):
    d = np.sqrt(np.diag(Om))
    C = Om / np.outer(d, d)
    iu = np.triu_indices(len(d), 1)
    return float(C[iu].mean())


# =============================================================================================
# P7/P8 — the controls. An se is evidence about power only if the estimator is sound.
# =============================================================================================
def controls(P, s_ols, s_gls):
    print("\n" + "=" * 96)
    print("P7/P8 — POWER CONTROL AND NEGATIVE CONTROL")
    print("=" * 96)
    worst = 0.0
    for s_true in (s_ols, s_gls):
        G = gls(P["X"], s_true * P["excess"], P["Om_corr"])
        err = abs(float(G["b"][1]) - s_true)
        worst = max(worst, err)
        print(f"  P7 inject s_true {s_true:+.4f}  ->  recovered {float(G['b'][1]):+.10f}  "
              f"(error {err:.2e})")
    verdict("P7", worst < P7_BAR, f"both slopes recovered to {worst:.2e} < {P7_BAR:.0e}")

    rng = np.random.default_rng(PERM_SEED)
    resid = rng.standard_normal(len(P["excess"])) * P["sd_iid"]
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        xp = rng.permutation(P["excess"])
        Xp = np.column_stack([np.ones(len(xp)), xp])
        null[i] = float(gls(Xp, resid, P["Om_corr"])["b"][1])
    m, sd = float(null.mean()), float(null.std(ddof=1))
    tstat = abs(m) / (sd / math.sqrt(N_PERM))
    print(f"  P8 permutation null over {N_PERM}: mean {m:+.5f}  sd {sd:.5f}  "
          f"|t| {tstat:.2f}")
    verdict("P8", tstat < P8_SIGMA, f"null centred at 0 (|t| {tstat:.2f} < {P8_SIGMA})")
    return dict(p7_worst=worst, null_mean=m, null_sd=sd, null_t=tstat)


# =============================================================================================
# P9 — the send path does not move
# =============================================================================================
def p9():
    print("\n" + "=" * 96)
    print("P9 — THE SEND PATH DOES NOT MOVE")
    print("=" * 96)
    print(f"  check_selection.WANTED = {sorted(CS.WANTED)}")
    ok = set(CS.WANTED) == WANTED_LOCK
    verdict("P9", ok, f"WANTED is the locked pair {sorted(WANTED_LOCK)}")
    if not ok:
        fail("P9: WANTED moved -- w67 is read-only and must not touch the selection")
    return ok


def main() -> None:
    write = "--no-write" not in sys.argv
    beta_c = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))
                   ["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    gate_p1()
    above, s_ols, s_gls, sep = select_above()
    VAL = validity(above)
    P = power(above, y, beta_c, sep)
    C = controls(P, s_ols, s_gls)
    p9()

    # ---- FREEZE the two hypotheses, per file, so a post-08-24 run can score without re-deriving
    frozen = []
    for k, a, e in zip(P["stems"], P["anchor"], P["excess"]):
        frozen.append(dict(stem=k, cv=float(P["cv"][k]), excess_e6=float(e),
                           anchor=float(a),
                           lb_hat_ols=float(a + s_ols * e * U),
                           lb_hat_gls=float(a + s_gls * e * U),
                           sep_e6=float(sep * e)))
    print("\n" + "=" * 96)
    print("FROZEN PER-FILE PREDICTIONS (scored by a run AFTER the 08-24 window)")
    print("=" * 96)
    print(f"  {'stem':26s} {'excess':>8s} {'H_OLS lb':>10s} {'H_GLS lb':>10s} "
          f"{'gap e-6':>8s} {'steps':>6s}")
    for f in frozen:
        print(f"  {f['stem']:26s} {f['excess_e6']:8.2f} {f['lb_hat_ols']:10.6f} "
              f"{f['lb_hat_gls']:10.6f} {f['sep_e6']:8.2f} "
              f"{f['sep_e6'] / (LB_STEP / U):6.2f}")
    nres = sum(1 for f in frozen if f["sep_e6"] >= LB_STEP / U)
    print(f"\n  {nres} of {len(frozen)} files have the two hypotheses more than ONE LB "
          f"reporting step ({LB_STEP / U:.0f}e-6) apart.")

    print("\n" + "=" * 96)
    print("THE GOVERNING VERDICT")
    print("=" * 96)
    if VAL["n_free"] == 0:
        print(f"  The instrument has bucket ({P['bucket']}) resolution (R = {P['R']:.3f}) and "
              f"NO VALIDITY.")
        print(f"  All {VAL['n_above']} above-range files are vetoed or `member`; ZERO can be "
              f"sent.")
        print("  w66 section 9 item 3 — \"this answers itself from the 08-24 sends onward\" — "
              "is FALSE.")
        print("  The 2026-08-24 plan sends ten files, every one BELOW FIT_CV_MAX, and no later")
        print("  day can differ while the veto binds. R2 is NOT answerable from the send queue.")
    else:
        print(f"  {VAL['n_free']} sendable above-range file(s) exist: {VAL['free']}")

    print("\n" + "=" * 96)
    print(f"REGISTERED PREDICTIONS: {sum(RESULTS.values())} of {len(RESULTS)} confirmed")
    print("=" * 96)
    for k, v in RESULTS.items():
        print(f"  {'✅' if v else '🔴'} {k}")
    print(f"\nFAILURES {FAILURES}")

    if write:
        out = os.path.join(HERE, "w67a_aboveslope.json")
        json.dump(dict(frozen=frozen, s_ols=s_ols, s_gls=s_gls, separation=sep,
                       se_iid=P["se_iid"], se_corr=P["se_corr"], R=P["R"],
                       bucket=P["bucket"], sd_iid=P["sd_iid"],
                       chi2dof_demote_to_b=P["scale_to_b"] ** 2,
                       chi2dof_demote_to_a=P["scale_to_a"] ** 2,
                       fit_cv_max=PR.FIT_CV_MAX, controls=C,
                       n_resolvable=int(nres), validity=VAL,
                       results=RESULTS, failures=FAILURES),
                  open(out, "w"), indent=1)
        print(f"\nwrote {out}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
