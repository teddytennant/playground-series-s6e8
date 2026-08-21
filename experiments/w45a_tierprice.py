"""w45a — reprice the final-selection click on the LIVE auto-selection tiers.

Pre-registration: experiments/w45_prereg.txt, committed dda1dca BEFORE this file existed.

WHY. The six prior pricings all condition on auto-slot 1 being held by
w16e_aonly / w16q_ens4avg / w16t_cellens4. Those three score 0.97108 and are in
NEITHER live tier. w36_ad199stdcorr -- the WANTED file -- landed 2026-08-21
00:07 UTC at public 0.97118 and is now inside auto-slot 1's four-way tie. The
quantity changed; this is not a re-estimate of the same one.

MATH. Identical to w18a, whose P6 proved the joint conditioning collapses to a
scalar map because the pseudo-test draw and the public-slice draw are the same
sampling operation at two sizes, so S_t and S_p are exactly proportional:

    S_t = (1 - m/n)  [C1/m1 + C0/m0]        pool -> test
    S_p = (1 - f)    [C1/p1 + C0/p0]        test -> public slice
    C1  = cov over pool POSITIVES of F0_k(s_i);  C0 = cov over NEGATIVES of 1 - F1_k(s_j)
    E[private dev | observed public dev x] = (beta*I + (1-beta) M) x,  M = S_t (S_t+S_p)^-1
    beta = -0.2477, w17d's measured coupling (exact partition -0.25).

Non-additivity residual: w17d measured |corr| 0.9924..0.9971 between private and
public deviations at fixed test. We do not have per-pair corr for these new
pairs, so the residual is carried at the CONSERVATIVE end (0.9924) and the
sensitivity to the optimistic end is printed.

TAU. w19c/w19d: tau is a per-file transfer sd the board cannot identify. Swept.

    .venv/bin/python experiments/w45a_tierprice.py
"""
from __future__ import annotations

import io, itertools, json, os, subprocess, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw          # noqa: E402
from w16b_cellweight import fast_auc              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, GRID, U = 296_302, 0.20, 1e-5, 1e-6

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
TIER1 = ["w36_ad199stdcorr", "w29_ad194stdcorr", "w27_ad190stdcorr", "w21_ad187corr_ens4"]
TIER2 = ["w36_ad199std", "w34_ad195stdcorr", "w27_ad190std", "w21_ad187corr",
         "w22_ad187corr_rankraw"]
NAMES = sorted(set(TIER1 + TIER2 + list(WANTED)))
FAM = {"w36_ad199stdcorr": "h3c", "w29_ad194stdcorr": "h3c", "w27_ad190stdcorr": "h3c",
       "w21_ad187corr_ens4": "ens4c", "w23_ad187stdcorr": "h3c", "w36_ad199std": "ens4",
       "w34_ad195stdcorr": "h3c", "w27_ad190std": "ens4", "w21_ad187corr": "h3c",
       "w22_ad187corr_rankraw": "rankraw"}


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def emax(mu, sd, rho):
    """E[max(X,Y)] for a bivariate normal (Clark). Exact, no simulation."""
    th = np.sqrt(max(sd[0] ** 2 + sd[1] ** 2 - 2 * rho * sd[0] * sd[1], 1e-300))
    a = (mu[0] - mu[1]) / th
    return mu[0] * norm.cdf(a) + mu[1] * norm.cdf(-a) + th * norm.pdf(a)


def main() -> None:
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree — scores are not deterministic"
    LB = agg["max"].to_dict()
    print(f"live board: {len(sub)} scored submissions, {len(agg)} distinct files")

    # --- GATE A: the tiers are what the prereg says they are, re-derived from the live board
    top = agg["max"].sort_values(ascending=False)
    t1v, t2v = top.iloc[0], sorted(set(top.values))[-2]
    live1 = sorted(top[top == t1v].index)
    live2 = sorted(top[top == t2v].index)
    ok1, ok2 = live1 == sorted(TIER1), live2 == sorted(TIER2)
    print(f"  auto-slot 1 public {t1v:.5f}: {live1}\n    matches prereg: {ok1}")
    print(f"  auto-slot 2 public {t2v:.5f}: {live2}\n    matches prereg: {ok2}")
    if not (ok1 and ok2):
        print("*** TIERS MOVED SINCE THE PREREG — everything below is void. ***")
        sys.exit(2)

    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}
    print(f"\n{'file':24s} {'CV':>14s} {'public':>9s} {'z=LB-CV':>9s}  fam")
    for k in NAMES:
        print(f"  {k:22s} {cv[k]:.10f} {LB[k]:9.5f} {(LB[k]-cv[k])/U:+9.1f}  {FAM[k]}")

    # ---------------------------------------------------------------- LAW-IF, once
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.empty((len(NAMES), n1)); B = np.empty((len(NAMES), n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    C1, C0 = np.cov(A), np.cov(B)
    m, n_pub = N_TEST, int(round(N_TEST * F))
    pi1 = n1 / n
    S_t = (1.0 - m / n) * (C1 / (m * pi1) + C0 / (m * (1 - pi1)))
    S_p = (1.0 - n_pub / m) * (C1 / (n_pub * pi1) + C0 / (n_pub * (1 - pi1)))
    St, Sp = S_t / U ** 2, S_p / U ** 2
    Sx = St + Sp
    M = St @ np.linalg.inv(Sx)

    # GATE B: w18a's P6 — M must be a scalar times I, or the scalar-map reading is wrong
    at, ap = (1 - m / n) / m, (1 - n_pub / m) / n_pub
    w_closed = at / (at + ap)
    offI = float(np.abs(M - np.eye(len(NAMES)) * np.diag(M).mean()).max())
    gamma = beta + (1 - beta) * float(np.diag(M).mean())
    print(f"\n=== LAW-IF ({n1:,} pos / {n0:,} neg, zero draws) ===")
    print(f"  max |M - w*I| = {offI:.3e}   w = {np.diag(M).mean():.9f} "
          f"(closed form from row counts {w_closed:.9f})")
    print(f"  gamma = beta + (1-beta)w = {beta:+.4f} + {1-beta:.4f}*{np.diag(M).mean():.4f} "
          f"= {gamma:+.6f}")
    assert offI < 1e-6, "M is not w*I — w18a's P6 does not hold on this file set"

    col = {k: i for i, k in enumerate(NAMES)}

    def uvec(x, z):
        u = np.zeros(len(NAMES)); u[col[x]], u[col[z]] = 1.0, -1.0
        return u

    # ---------------------------------------------------------------- conditioning (GLOBAL, CLOSED)
    def solve(tau, corr):
        """Return (mu, cov, resid_sd) with a per-file transfer sd tau (in 1e-6) added."""
        Stt = St + tau ** 2 * np.eye(len(NAMES))
        Sxx = Stt + Sp
        Mm = Stt @ np.linalg.inv(Sxx)
        Kmap = beta * np.eye(len(NAMES)) + (1 - beta) * Mm
        Cpri = (1 - beta) ** 2 * (Stt - Mm @ Sxx @ Mm.T)
        Cpri = 0.5 * (Cpri + Cpri.T)
        z = np.array([LB[k] - cv[k] for k in NAMES]) / U
        D = np.ones((len(NAMES), 1))
        iS = np.linalg.inv(Sxx)
        Vg = np.linalg.inv(D.T @ iS @ D)
        gh = Vg @ (D.T @ iS @ z)
        xh = z - D @ gh
        mu = Kmap @ xh
        cov = Cpri + Kmap @ D @ Vg @ D.T @ Kmap.T
        # non-additivity residual, uniform, from w17d's measured corr band
        s_pri = abs(beta) * np.sqrt(np.diag(Sp)) / abs(corr)
        rsd = s_pri * np.sqrt(max(1 - corr ** 2, 0.0))
        return mu, cov, rsd, float(gh[0])

    CORR_CONS, CORR_OPT = 0.9923594211711959, 0.9971372069496918
    mu, cov, rsd, ghat = solve(0.0, CORR_CONS)
    print(f"  free common gap G = {ghat:+.2f}e-6 (GLS); residual sd per file "
          f"{rsd.min():.2f}..{rsd.max():.2f}e-6 at corr {CORR_CONS:.4f}")

    iw = [col[k] for k in WANTED]

    def price(mu, cov, rsd, autofile, second=None):
        """cost of NOT clicking = E[max over WANTED] - E[max over auto picks]."""
        mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
        rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
        ew = emax(mw, sw, rw)
        if second is None:
            ia = col[autofile]
            ea = mu[ia] + cv[autofile] / U
        else:
            ia = [col[autofile], col[second]]
            ma = np.array([mu[i] + cv[NAMES[i]] / U for i in ia])
            sa = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in ia])
            ra = cov[ia[0], ia[1]] / np.sqrt(cov[ia[0], ia[0]] * cov[ia[1], ia[1]])
            ea = emax(ma, sa, ra)
        return float(ew - ea)

    def pjoint(mu, cov, rsd, x):
        # Degenerate when x IS one of the WANTED files: the contrast is the file against
        # itself, so the event is not defined. Return None rather than a meaningless ~0.5.
        if x in WANTED:
            return None
        mus = np.array([(cv[x] - cv[k]) / U + float(uvec(x, k) @ mu) for k in WANTED])
        cc = np.empty((2, 2))
        for i, ki in enumerate(WANTED):
            for j, kj in enumerate(WANTED):
                cc[i, j] = uvec(x, ki) @ cov @ uvec(x, kj)
        for i, ki in enumerate(WANTED):
            cc[i, i] += rsd[col[x]] ** 2 + rsd[col[ki]] ** 2
        return float(multivariate_normal(mean=-mus, cov=cc, allow_singular=True).cdf([0.0, 0.0]))

    print("\n=== LIMIT 1: cost of NOT clicking, per branch of the auto-slot-1 tie ===")
    print("  positive = clicking is worth this much on the private slice.  units 1e-6")
    print(f"  {'auto file':22s} {'dCV':>8s} {'dLB':>6s} {'cost':>8s} {'P(auto beats BOTH)':>19s}")
    l1 = {}
    for x in TIER1:
        c = price(mu, cov, rsd, x)
        pj = pjoint(mu, cov, rsd, x)
        l1[x] = dict(cost=c, p_joint=pj,
                     dcv=(cv[x] - cv["w36_ad199stdcorr"]) / U,
                     dlb=(LB[x] - LB["w36_ad199stdcorr"]) / U)
        tag = "   <-- IS the WANTED file" if x in WANTED else ""
        pjs_ = f"{pj:19.3f}" if pj is not None else f"{'n/a (self)':>19s}"
        print(f"  {x:22s} {l1[x]['dcv']:+8.1f} {l1[x]['dlb']:+6.1f} {c:+8.2f} {pjs_}{tag}")
    avg = float(np.mean([l1[x]["cost"] for x in TIER1]))
    print(f"  uniform-tiebreak average over the four branches: {avg:+.2f}e-6")

    print("\n=== LIMIT 2: one draw from each tier (Kaggle's documented default) ===")
    l2 = {}
    for x in TIER1:
        for zz in TIER2:
            l2[f"{x}+{zz}"] = price(mu, cov, rsd, x, zz)
    vals = np.array(list(l2.values()))
    worst = max(l2, key=l2.get); best = min(l2, key=l2.get)
    print(f"  20 pairs: {vals.min():+.2f} .. {vals.max():+.2f}e-6,  mean {vals.mean():+.2f}e-6")
    print(f"    cheapest  {best:45s} {l2[best]:+8.2f}")
    print(f"    dearest   {worst:45s} {l2[worst]:+8.2f}")
    nowant = [v for k, v in l2.items() if not k.startswith("w36_ad199stdcorr+")]
    print(f"  restricted to the 15 pairs that MISS the wanted file: mean {np.mean(nowant):+.2f}e-6")

    print("\n=== TAU SWEEP (w19c's sign-flip test), limit 1 ===")
    print(f"  {'tau':>6s} " + " ".join(f"{x[:16]:>17s}" for x in TIER1))
    taus = [0.0, 0.5, 1.0, 1.72, 2.5, 3.0, 5.0]
    sweep = {}
    for t in taus:
        mt, ct, rt, _ = solve(t, CORR_CONS)
        row = [price(mt, ct, rt, x) for x in TIER1]
        sweep[t] = row
        print(f"  {t:6.2f} " + " ".join(f"{v:+17.2f}" for v in row))
    flip = any(np.sign(sweep[t][i]) != np.sign(sweep[0.0][i]) and abs(sweep[t][i]) > 0.05
               for t in taus if t <= 3.0 for i in (1, 2, 3))
    cheap3 = min(sweep[3.0][i] for i in (1, 2, 3))

    print("\n=== corr sensitivity (the residual is the only assumed quantity) ===")
    mo, co, ro, _ = solve(0.0, CORR_OPT)
    for x in TIER1:
        print(f"  {x:22s} conservative {price(mu,cov,rsd,x):+7.2f}   "
              f"optimistic {price(mo,co,ro,x):+7.2f}")

    # ---------------------------------------------------------------- registered predictions
    print("\n=== the registered predictions (w45_prereg.txt) ===")
    non = [l1[x]["cost"] for x in TIER1 if x not in WANTED]
    p1 = bool(min(non) >= 10.0 and max(non) <= 45.0)
    print(f"  P1 non-wanted branches {np.round(non,2)}  band [10,45]  -> "
          f"{'CONFIRMED' if p1 else '*** FALSIFIED ***'}  (tighter registered guess was [15,30])")
    p2 = bool(abs(l1["w36_ad199stdcorr"]["cost"]) <= 1.0)
    print(f"  P2 the wanted branch costs {l1['w36_ad199stdcorr']['cost']:+.3f}e-6  band |.|<=1  -> "
          f"{'CONFIRMED' if p2 else '*** FALSIFIED ***'}")
    p3 = bool(avg >= 10.0)
    print(f"  P3 uniform-tiebreak average {avg:+.2f}e-6  band >=10  -> "
          f"{'CONFIRMED' if p3 else '*** FALSIFIED ***'}")
    pjs = [l1[x]["p_joint"] for x in TIER1 if x not in WANTED]
    p4 = bool(max(pjs) <= 0.20)
    print(f"  P4 P(auto beats both) {np.round(pjs,4)}  band <=0.20  -> "
          f"{'CONFIRMED' if p4 else '*** FALSIFIED ***'}")
    p5 = bool((not flip) and cheap3 >= 8.0)
    print(f"  P5 no sign flip on tau in [0,3]: {not flip}; cheapest branch at tau=3 "
          f"{cheap3:+.2f}e-6 (band >=8)  -> {'CONFIRMED' if p5 else '*** FALSIFIED ***'}")

    out = dict(tiers=dict(slot1=TIER1, slot2=TIER2), wanted=list(WANTED),
               cv={k: cv[k] for k in NAMES}, lb={k: LB[k] for k in NAMES},
               beta=beta, gamma=gamma, gap=ghat, limit1=l1, limit2=l2,
               uniform_avg=avg, tau_sweep={str(k): v for k, v in sweep.items()},
               p1=p1, p2=p2, p3=p3, p4=p4, p5=p5)
    with open(os.path.join(HERE, "w45a_tierprice.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote experiments/w45a_tierprice.json")


if __name__ == "__main__":
    main()
