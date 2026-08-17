"""w19a — is the paired CV->LB instrument calibrated? Fit the per-file transfer sd tau by ML.

Slot 4's next-run items 1 and 2, which are one question. Two numbers in this workspace disagree:

  w17a  over 46 sent files, within tight families, rms(dLB - dCV) = 7.4e-6 against 9.4e-6
        predicted by slice draw + grid rounding ALONE. Ratio 0.79 — residuals TIGHTER than
        the model, no variance left to allocate to anything else.
  w18b  blend159av_rankraw printed 0.97102 against a registered paired 0.97104 (LAW-IF sd
        7.344e-6) and sat z -3.58 against its own family. Residual LOOSER than the model, and
        slot 4 concluded every conditioned number in the click price is sharp the same way.

Slot 4 proposed pooling the four registered paired predictions. Four points cannot separate
those. But EVERY within-family pair of sent files is the same instrument evaluated once, and
there are ~500 of them, so the question is a variance-components fit and not a tally:

    LB_k = round_G( cv_k + g_fam(k) + d_k + e_k + u_k )      G = 1e-5
    (d + e) ~ N(0, S_x)   S_x = S_t + S_p, LAW-IF, closed form, no draws  [w17h, w18a]
    u_k     ~ N(0, tau^2) iid per file   <- THE PARAMETER. tau = 0 is w17a; tau > 0 is w18b.

Within a family g cancels, so each pair contributes the exact pmf of a rounded Gaussian
difference and tau is fitted on ~500 pairs by composite ML.

THREE THINGS THIS FILE DOES THAT THE OBVIOUS VERSION GETS WRONG.

1. ROUNDING IS EXACT, NOT A VARIANCE. For a pair the reported difference is
   round(v_a) - round(v_b), and with the per-file marginal sd (558e-6) 56x the grid the phase
   of v_b is uniform mod G, so

       P(dLB = mG) = integral phi(D; dcv, s) * tri((mG - D)/G) dD = [I(mG+G) - 2I(mG) + I(mG-G)]/G

   with I the antiderivative of the normal cdf. That is EXACT (and, pleasingly, identical to
   convolving with two independent U(-G/2, G/2) — the additive-uniform picture is right for a
   PAIR even though rounding is not additive noise). What is wrong is replacing the triangular
   by a Gaussian of matched variance G^2/6: the triangular is bounded at +/-G and the Gaussian
   is not, and the near-duplicate pairs that carry all the information about tau sit exactly
   where that difference is the whole answer. Refit both ways; the gap is registered as P5.

2. THE COMPOSITE LIKELIHOOD'S CURVATURE SE IS A LIE. Pairs sharing a file are correlated, and
   with ~50 files and ~500 pairs each file appears in ~20 of them. Uncertainty comes from a
   PARAMETRIC BOOTSTRAP of the whole file-level system (draw x ~ N(0, S_x) with the full K x K
   covariance, u ~ N(0, tau^2 I), round, rebuild every pair, refit), which reproduces that
   correlation by construction. Registered as P4.

3. THE TAIL DIAGNOSTIC IS BOOTSTRAP-CALIBRATED, NOT GAUSSIAN. Standardised residuals of a
   ROUNDED quantity are not normal even when the model is perfectly true — they are discrete,
   and for the tight pairs violently so. Comparing their kurtosis to 3 would manufacture a
   finding. The observed statistics are located in the fitted model's OWN bootstrap null.

Pre-registration with P1-P7: experiments/w19_prereg.txt, written before any number existed.

    .venv/bin/python experiments/w19a_transfer.py
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402
from w17b_famfix import family  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
F = 0.20
G = 1e-5
U = 1e-6                       # everything below is in units of 1e-6


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def _I(z, mu, s):
    """Antiderivative of Phi((u-mu)/s): int_{-inf}^{z} Phi((u-mu)/s) du."""
    t = (z - mu) / s
    return s * (t * norm.cdf(t) + norm.pdf(t))


def pmf_exact(m, mu, s, grid=G / U):
    """P(round(v_a)-round(v_b) = m*G) for v_a-v_b ~ N(mu, s^2). Exact triangular convolution."""
    x = m * grid
    p = (_I(x + grid, mu, s) - 2.0 * _I(x, mu, s) + _I(x - grid, mu, s)) / grid
    return np.clip(p, 1e-300, None)


def pmf_gauss(m, mu, s, grid=G / U):
    """The approximation P5 exists to price: rounding as an extra Gaussian variance G^2/6."""
    se = np.sqrt(s ** 2 + grid ** 2 / 6.0)
    x = m * grid
    p = norm.cdf((x + 0.5 * grid - mu) / se) - norm.cdf((x - 0.5 * grid - mu) / se)
    return np.clip(p, 1e-300, None)


def fit_tau(m, dcv, var0, pmf=pmf_exact, hi=40.0):
    """Composite ML for tau over pairs. var0 = LAW-IF paired variance, units 1e-12."""
    def nll(tau):
        return -np.log(pmf(m, dcv, np.sqrt(var0 + 2.0 * tau ** 2))).sum()
    r = minimize_scalar(nll, bounds=(0.0, hi), method="bounded",
                        options={"xatol": 1e-4})
    # the bounded optimiser cannot report the tau=0 boundary cleanly; check it explicitly
    return (0.0, nll(0.0)) if nll(0.0) <= r.fun else (float(r.x), float(r.fun))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=19_2026)
    a = ap.parse_args()
    out = {}

    # ------------------------------------------------------------------ live board
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub = sub[sub["publicScore"].notna()]
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree — scores are not deterministic"
    LB = agg["max"].to_dict()
    print(f"live: {len(sub)} scored submissions, {len(agg)} distinct files (page size 200)")

    names = sorted(k for k in LB if os.path.exists(os.path.join(SUB, f"oof_{k}.npy")))
    K = len(names)
    print(f"      {K} of them carry a stored OOF vector -> that is the file set")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = np.array([fast_auc(y, V[k]) for k in names])
    fam = np.array([family(k) for k in names])
    foreign = np.array([k.startswith("stack_pub") for k in names])
    print(f"      families: " + ", ".join(f"{f}={int((fam==f).sum())}"
                                          for f in sorted(set(fam))))
    print(f"      foreign (stack_pub*, other teams' files): {int(foreign.sum())}")

    # ------------------------------------------------------------------ LAW-IF, closed form
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.empty((K, n1))
    B = np.empty((K, n0))
    for i, k in enumerate(names):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    C1, C0 = np.cov(A), np.cov(B)
    del A, B
    m_t = N_TEST
    n_pub = int(round(N_TEST * F))
    pi1 = n1 / n
    S_t = (1.0 - m_t / n) * (C1 / (m_t * pi1) + C0 / (m_t * (1 - pi1)))
    S_p = (1.0 - n_pub / m_t) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1)))
    S_x = (S_t + S_p) / U ** 2                       # units 1e-12, i.e. sd in 1e-6
    print(f"\n=== LAW-IF over {K} files, {n1:,} pos / {n0:,} neg, ZERO draws ===")
    print(f"  single-file public-slice sd {np.sqrt(np.diag(S_p)).mean()*1e6:.1f}e-6 "
          f"(drawn out of the test); it is COMMON to all {K} files and cancels in every pair")

    # ------------------------------------------------------------------ the pairs
    rows = []
    for i, j in itertools.combinations(range(K), 2):
        if fam[i] != fam[j]:
            continue
        u = np.zeros(K)
        u[i], u[j] = 1.0, -1.0
        var0 = float(u @ S_x @ u)
        dcv = (cv[i] - cv[j]) / U
        dlb = (LB[names[i]] - LB[names[j]]) / U
        rows.append(dict(a=names[i], b=names[j], fam=fam[i], dcv=dcv, dlb=dlb,
                         resid=dlb - dcv, sd0=np.sqrt(var0), var0=var0,
                         m=int(round(dlb / (G / U))),
                         foreign=bool(foreign[i] or foreign[j])))
    P = pd.DataFrame(rows)
    print(f"\n=== {len(P)} within-family pairs ===")
    print(f"  LAW-IF paired sd: min {P.sd0.min():.2f}  median {P.sd0.median():.2f}  "
          f"max {P.sd0.max():.2f} e-6")
    print(f"  observed rms(dLB - dCV) {np.sqrt((P.resid**2).mean()):.2f}e-6 against "
          f"{np.sqrt((P.var0 + (G/U)**2/6).mean()):.2f}e-6 predicted at tau=0 "
          f"(slice draw + rounding)  -> ratio {np.sqrt((P.resid**2).mean()/(P.var0 + (G/U)**2/6).mean()):.3f}")
    print("  (w17a read that ratio as 0.79 on a smaller set with simulated sds)")

    # ------------------------------------------------------------------ P1 GATE vs w17a
    print("\n=== P1 GATE: LAW-IF paired sd vs w17a's simulated paired sd, shared pairs ===")
    w17 = pd.read_csv(os.path.join(HERE, "w17a_pairs.csv"))
    key = {(r.a, r.b): r.sd / U for r in w17.itertuples()}
    key.update({(r.b, r.a): r.sd / U for r in w17.itertuples()})
    sh = [(r.sd0, key[(r.a, r.b)]) for r in P.itertuples() if (r.a, r.b) in key]
    ratio = np.array([x / z for x, z in sh])
    err = np.abs(ratio - 1.0)
    p1 = bool(len(sh) > 50 and np.median(err) < 0.05 and err.max() < 0.25)
    print(f"  {len(sh)} shared pairs; |ratio-1| median {np.median(err)*100:.2f}%  "
          f"max {err.max()*100:.2f}%   (registered median<5%, max<25%)")
    print(f"  P1 {'PASSED' if p1 else '*** FAILED — everything below is void ***'}")
    out["P1"] = dict(passed=p1, n=len(sh), median_err=float(np.median(err)),
                     max_err=float(err.max()))

    mm, dcvv, vv = P.m.to_numpy(), P.dcv.to_numpy(), P.var0.to_numpy()

    # ------------------------------------------------------------------ P2 the fit
    tau, nll = fit_tau(mm, dcvv, vv)
    tau_g, _ = fit_tau(mm, dcvv, vv, pmf=pmf_gauss)
    print(f"\n=== P2: composite ML for the per-file transfer sd tau ===")
    print(f"  tau_hat = {tau:.3f}e-6      (exact triangular rounding)")
    print(f"  tau_hat = {tau_g:.3f}e-6    (rounding as Gaussian variance G^2/6)   <- P5 compares")

    grid = np.linspace(0.0, max(12.0, 3 * tau + 6), 121)
    prof = np.array([-np.log(pmf_exact(mm, dcvv, np.sqrt(vv + 2 * t ** 2))).sum() for t in grid])
    lo_hi = grid[prof <= prof.min() + 1.92]          # 95% profile interval, chi2_1
    print(f"  95% profile interval [{lo_hi.min():.3f}, {lo_hi.max():.3f}]e-6   "
          f"(deviance vs tau=0: {2*(prof[0]-prof.min()):.2f})")
    # naive curvature se, for P4
    h = 0.05
    f0 = -np.log(pmf_exact(mm, dcvv, np.sqrt(vv + 2 * tau ** 2))).sum()
    fp = -np.log(pmf_exact(mm, dcvv, np.sqrt(vv + 2 * (tau + h) ** 2))).sum()
    fmn = -np.log(pmf_exact(mm, dcvv, np.sqrt(vv + 2 * max(tau - h, 0) ** 2))).sum()
    curv = (fp - 2 * f0 + fmn) / h ** 2
    se_naive = float(1.0 / np.sqrt(curv)) if curv > 0 else float("nan")
    print(f"  naive composite-likelihood curvature se: {se_naive:.3f}e-6")

    # split fits, for the record
    print("\n  splits (same fit, subsets of the pairs):")
    for tag, sel in (("all", np.ones(len(P), bool)),
                     ("ours only (no stack_pub*)", ~P.foreign.to_numpy()),
                     ("tight pairs, LAW-IF sd < 5e-6", P.sd0.to_numpy() < 5.0),
                     ("loose pairs, LAW-IF sd >= 5e-6", P.sd0.to_numpy() >= 5.0)):
        if sel.sum() < 20:
            continue
        t, _ = fit_tau(mm[sel], dcvv[sel], vv[sel])
        print(f"    {tag:34s} n={int(sel.sum()):4d}   tau_hat {t:6.3f}e-6")
    for f in sorted(set(P.fam)):
        sel = (P.fam == f).to_numpy()
        if sel.sum() < 20:
            continue
        t, _ = fit_tau(mm[sel], dcvv[sel], vv[sel])
        print(f"    family {f:28s} n={int(sel.sum()):4d}   tau_hat {t:6.3f}e-6")

    # ------------------------------------------------------------------ diagnostics, observed
    sd_fit = np.sqrt(vv + 2 * tau ** 2 + (G / U) ** 2 / 6.0)
    z = P.resid.to_numpy() / sd_fit
    stat_obs = dict(maxabsz=float(np.abs(z).max()), kurt=float((z ** 4).mean() / (z ** 2).mean() ** 2),
                    rms=float(np.sqrt((z ** 2).mean())))
    worst = P.iloc[int(np.argmax(np.abs(z)))]
    print(f"\n=== observed residual diagnostics (located in the bootstrap null below) ===")
    print(f"  max |z| {stat_obs['maxabsz']:.2f}  at {worst.a} vs {worst.b} "
          f"(resid {worst.resid:+.1f}e-6, sd {sd_fit[int(np.argmax(np.abs(z)))]:.2f})")
    print(f"  kurtosis {stat_obs['kurt']:.2f}   rms z {stat_obs['rms']:.3f}")

    # where does w18b's send sit, as a proper paired z?
    print("\n  w18b's print, re-cut as a LAW-IF paired z (it was quoted as -3.58 against a "
          "family sd on n=4):")
    for r in P.itertuples():
        if "blend159av_rankraw" in (r.a, r.b) and r.fam == "rankraw":
            s = np.sqrt(r.var0 + 2 * tau ** 2 + (G / U) ** 2 / 6.0)
            print(f"    {r.a:26s} vs {r.b:26s} resid {r.resid:+6.1f}e-6  sd {s:5.2f}  "
                  f"z {r.resid/s:+5.2f}")

    # ------------------------------------------------------------------ P3/P4 parametric bootstrap
    print(f"\n=== P3/P4: parametric bootstrap of the WHOLE file system, {a.boot} draws ===")
    print("    (draw x ~ N(0, S_x) over all files with the full covariance, u ~ N(0,tau^2 I),")
    print("     round on the real grid, rebuild every pair, refit tau — so the pair-sharing")
    print("     correlation and the rounding discreteness are both in the null by construction)")
    ev, EV = np.linalg.eigh(S_x)
    L = EV * np.sqrt(np.clip(ev, 0.0, None))
    gap = {}
    for f in sorted(set(fam)):
        sel = fam == f
        gap[f] = float(np.mean([LB[names[i]] - cv[i] for i in np.flatnonzero(sel)]) / U)
    base = np.array([cv[i] / U + gap[fam[i]] for i in range(K)])
    ia = np.array([names.index(x) for x in P.a])
    ib = np.array([names.index(x) for x in P.b])
    rng = np.random.default_rng(a.seed)
    bt, bmax, bkurt, brms = [], [], [], []
    for _ in range(a.boot):
        v = base + L @ rng.standard_normal(K) + tau * rng.standard_normal(K)
        lbs = np.round(v / (G / U)) * (G / U)
        dl = lbs[ia] - lbs[ib]
        mb = np.round(dl / (G / U)).astype(int)
        t, _ = fit_tau(mb, dcvv, vv)
        bt.append(t)
        zb = (dl - dcvv) / np.sqrt(vv + 2 * tau ** 2 + (G / U) ** 2 / 6.0)
        bmax.append(np.abs(zb).max())
        bkurt.append((zb ** 4).mean() / (zb ** 2).mean() ** 2)
        brms.append(np.sqrt((zb ** 2).mean()))
    bt = np.array(bt)
    se_boot = float(bt.std(ddof=1))
    bias = float(bt.mean() - tau)
    print(f"  tau_hat over the bootstrap: mean {bt.mean():.3f} (bias {bias:+.3f}), "
          f"sd {se_boot:.3f}e-6, 2.5/97.5 pct [{np.percentile(bt,2.5):.3f}, "
          f"{np.percentile(bt,97.5):.3f}]")
    p4 = bool(se_boot / se_naive > 1.3)
    print(f"  P4: boot se / naive se = {se_boot/se_naive:.2f}   "
          f"(registered > 1.3)  {'CONFIRMED' if p4 else 'FALSIFIED'}")
    q = float((np.array(bmax) <= stat_obs["maxabsz"]).mean())
    qk = float((np.array(bkurt) <= stat_obs["kurt"]).mean())
    qr = float((np.array(brms) <= stat_obs["rms"]).mean())
    print(f"  P3: observed max|z| {stat_obs['maxabsz']:.2f} sits at bootstrap percentile "
          f"{q*100:.1f}   (registered > 90th)  {'CONFIRMED' if q > 0.90 else 'FALSIFIED'}")
    print(f"      kurtosis {stat_obs['kurt']:.2f} at percentile {qk*100:.1f};  "
          f"rms z {stat_obs['rms']:.3f} at percentile {qr*100:.1f}")

    p2 = bool(tau < 3.0 and lo_hi.max() < 6.0)
    p5 = bool(abs(tau - tau_g) > 1.0)
    print(f"\n  P2 {'CONFIRMED' if p2 else '*** FALSIFIED ***'}   "
          f"(tau_hat {tau:.3f} < 3.0 and upper {lo_hi.max():.3f} < 6.0)")
    print(f"  P5 {'CONFIRMED' if p5 else 'FALSIFIED'}   "
          f"(|exact - gaussian-rounding| = {abs(tau-tau_g):.3f}e-6, registered > 1.0)")

    out.update(dict(
        K=K, n_pairs=int(len(P)), tau=tau, tau_gaussround=tau_g,
        prof_lo=float(lo_hi.min()), prof_hi=float(lo_hi.max()),
        dev_vs_zero=float(2 * (prof[0] - prof.min())),
        se_naive=se_naive, se_boot=se_boot, boot_bias=bias,
        boot_q_maxabsz=q, boot_q_kurt=qk, boot_q_rms=qr, stat_obs=stat_obs,
        P2=p2, P3=bool(q > 0.90), P4=p4, P5=p5,
        rms_resid=float(np.sqrt((P.resid ** 2).mean())),
        rms_pred_tau0=float(np.sqrt((P.var0 + (G / U) ** 2 / 6).mean())),
        boot=a.boot, seed=a.seed,
    ))
    json.dump(out, open(os.path.join(HERE, "w19a_transfer.json"), "w"), indent=1)
    P.to_csv(os.path.join(HERE, "w19a_pairs.csv"), index=False)
    np.save(os.path.join(HERE, "w19a_Sx.npy"), S_x)
    json.dump({"names": names, "cv": cv.tolist(), "lb": {k: LB[k] for k in names},
               "fam": fam.tolist()}, open(os.path.join(HERE, "w19a_files.json"), "w"), indent=1)
    print("\nwrote w19a_transfer.json, w19a_pairs.csv, w19a_Sx.npy, w19a_files.json")


if __name__ == "__main__":
    main()
