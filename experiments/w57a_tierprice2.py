"""w57a — re-price the final-selection click on the LIVE auto-selection tiers.

Pre-registration: experiments/w57_prereg.txt, committed 90a289d BEFORE this file existed.

WHY THIS FILE EXISTS, AND WHY IT IS NOT w45a RE-RUN.
`w45a_tierprice.py` hard-codes the tier membership it was registered against and refuses to
run when the board moves (its GATE A). That gate fired this run: the ten files sent
2026-08-22 12:37-12:38 UTC changed BOTH tiers.

    auto-slot 1  public 0.97118   4 files -> 5   (+ w40_ad211stdcorr)
    auto-slot 2  public 0.97117   5 files -> 7   (+ w38_ad202stdcorr, + w40_ad211std)

So every click price in RESEARCH.md (+0.00 / +15.77 / +21.74 / +35.15e-6) prices a board
configuration that no longer exists -- the same staleness w45a itself was written to fix,
one board-day later. A hard-coded tier goes stale on EVERY send day, so this module derives
the tiers LIVE and records the membership it ran on into its own JSON. A future run compares
that record against the live board instead of trusting the number.

⚠⚠ AND w45a's LIMIT-2 MODEL IS WRONG NOW, IN A WAY THAT MATTERS.
w45a draws auto-pick 1 from tier 1 and auto-pick 2 from tier 2. That is only right when
tier 1 holds exactly one file. `0.97118` is a ROUNDED display value: the true public scores
inside the tie are distinct (AUC over a 59k-row slice does not tie exactly), so Kaggle's
"best two by public score" takes the top TWO BY TRUE SCORE, and with five files in tier 1
BOTH come from tier 1. Tier 2 is never reached. Both models are reported:

    MODEL A  w45a's: one draw from tier 1, one from tier 2.  Retained ONLY so the published
             +15.77e-6 has a like-for-like successor. It is not the live mechanism.
    MODEL B  correct: the auto-selected pair is a 2-subset of tier 1.

MATH. Identical to w45a/w18a -- same beta, same LAW-IF conditioning, same Clark E[max], same
conservative non-additivity corr, same tau sweep. Nothing about the estimator changed; only
the file set it is evaluated on. Reusing the math verbatim is the point: a re-price that also
re-derived the estimator could not be compared against the number it supersedes.

    .venv/bin/python experiments/w57a_tierprice2.py
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
from stdflag import family                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST, F, GRID, U = 296_302, 0.20, 1e-5, 1e-6

WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")
PICK = "w36_ad199stdcorr"                 # the WANTED file that sits inside tier 1

# w45a's published figures, for the like-for-like comparison R3/R4/R7 are registered against.
W45A = dict(uniform_limit1=21.74, limit2_mean=15.77, wanted_branch=0.00, n_tier1=4)

CORR_CONS, CORR_OPT = 0.9923594211711959, 0.9971372069496918


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
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    # ⚠ w54: the API defaults to 50 rows and truncates SILENTLY. Guard on the page size.
    assert len(sub) < 500, "hit the page size -- raise it, the window is truncated"
    sub = sub[sub["status"] == "SubmissionStatus.COMPLETE"].dropna(subset=["publicScore"])
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree -- scores are not deterministic"
    LB = agg["max"].to_dict()

    # --- tiers derived LIVE. No hard-coded membership: that is what went stale in w45a.
    top = agg["max"].sort_values(ascending=False)
    vals = sorted(set(top.values), reverse=True)
    t1v, t2v = vals[0], vals[1]
    TIER1 = sorted(top[top == t1v].index)
    TIER2 = sorted(top[top == t2v].index)
    NAMES = sorted(set(TIER1 + TIER2 + list(WANTED)))
    print(f"live board: {len(sub)} scored submissions, {len(agg)} distinct files")
    print(f"  auto-slot 1 public {t1v:.5f}: {len(TIER1)} files {TIER1}")
    print(f"  auto-slot 2 public {t2v:.5f}: {len(TIER2)} files {TIER2}")
    print(f"  (w45a was registered on a {W45A['n_tier1']}-file tier 1 -- that price is void)")

    assert PICK in TIER1, "the WANTED pick has fallen OUT of tier 1 -- reprice from scratch"
    assert len(TIER1) >= 2, "tier 1 holds <2 files; MODEL B degenerates, use MODEL A"

    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}
    print(f"\n{'file':24s} {'CV':>14s} {'public':>9s} {'dCV vs pick':>12s}  fam  tier")
    for k in NAMES:
        t = "1" if k in TIER1 else ("2" if k in TIER2 else "-")
        print(f"  {k:22s} {cv[k]:.10f} {LB[k]:9.5f} {(cv[k]-cv[PICK])/U:+12.2f}  "
              f"{family(k):<8s} {t}")

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
    M = St @ np.linalg.inv(St + Sp)

    # GATE B: w18a's P6 -- M must be a scalar times I, or the scalar-map reading is wrong.
    offI = float(np.abs(M - np.eye(len(NAMES)) * np.diag(M).mean()).max())
    gamma = beta + (1 - beta) * float(np.diag(M).mean())
    print(f"\n=== LAW-IF ({n1:,} pos / {n0:,} neg, zero draws) ===")
    print(f"  max |M - w*I| = {offI:.3e}   w = {np.diag(M).mean():.9f}")
    print(f"  gamma = {gamma:+.6f}")
    r6 = bool(offI < 1e-6)
    assert r6, "M is not w*I -- w18a's P6 does not hold on this file set"

    col = {k: i for i, k in enumerate(NAMES)}

    def uvec(x, z):
        u = np.zeros(len(NAMES)); u[col[x]], u[col[z]] = 1.0, -1.0
        return u

    def solve(tau, corr):
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
        s_pri = abs(beta) * np.sqrt(np.diag(Sp)) / abs(corr)
        rsd = s_pri * np.sqrt(max(1 - corr ** 2, 0.0))
        return mu, cov, rsd, float(gh[0])

    mu, cov, rsd, ghat = solve(0.0, CORR_CONS)
    print(f"  free common gap G = {ghat:+.2f}e-6 (GLS); residual sd per file "
          f"{rsd.min():.2f}..{rsd.max():.2f}e-6")

    iw = [col[k] for k in WANTED]

    def emax_set(mu, cov, rsd, files):
        """E[max private] over a 1- or 2-file set, on the same scale as the CVs."""
        ia = [col[f] for f in files]
        ma = np.array([mu[i] + cv[NAMES[i]] / U for i in ia])
        if len(ia) == 1:
            return float(ma[0])
        sa = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in ia])
        ra = cov[ia[0], ia[1]] / np.sqrt(cov[ia[0], ia[0]] * cov[ia[1], ia[1]])
        return float(emax(ma, sa, ra))

    def price(mu, cov, rsd, files):
        """cost of NOT clicking = E[max over WANTED] - E[max over the auto picks]."""
        mw = np.array([mu[i] + cv[NAMES[i]] / U for i in iw])
        sw = np.array([np.sqrt(cov[i, i] + rsd[i] ** 2) for i in iw])
        rw = cov[iw[0], iw[1]] / np.sqrt(cov[iw[0], iw[0]] * cov[iw[1], iw[1]])
        return float(emax(mw, sw, rw) - emax_set(mu, cov, rsd, files))

    def pjoint(mu, cov, rsd, x):
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

    # ---------------------------------------------------------------- LIMIT 1
    print("\n=== LIMIT 1: cost of NOT clicking, per branch of the auto-slot-1 tie ===")
    print("  positive = clicking is worth this much on the private slice.  units 1e-6")
    print(f"  {'auto file':22s} {'dCV':>8s} {'dLB':>6s} {'cost':>8s} {'P(auto beats BOTH)':>19s}")
    l1 = {}
    for x in TIER1:
        c = price(mu, cov, rsd, [x])
        pj = pjoint(mu, cov, rsd, x)
        l1[x] = dict(cost=c, p_joint=pj, dcv=(cv[x] - cv[PICK]) / U,
                     dlb=(LB[x] - LB[PICK]) / U)
        tag = "   <-- IS the WANTED pick" if x == PICK else ""
        new = "  [NEW today]" if x in ("w40_ad211stdcorr",) else ""
        pjs_ = f"{pj:19.3f}" if pj is not None else f"{'n/a (self)':>19s}"
        print(f"  {x:22s} {l1[x]['dcv']:+8.2f} {l1[x]['dlb']:+6.1f} {c:+8.2f} {pjs_}{tag}{new}")
    avg1 = float(np.mean([l1[x]["cost"] for x in TIER1]))
    print(f"  uniform-tiebreak average over the {len(TIER1)} branches: {avg1:+.2f}e-6"
          f"   (w45a on 4 branches: {W45A['uniform_limit1']:+.2f})")

    # ---------------------------------------------------------------- MODEL A (w45a's)
    print("\n=== MODEL A (w45a's, RETAINED ONLY FOR COMPARABILITY -- not the live mechanism) ===")
    print("    one draw from tier 1 + one from tier 2. Wrong while tier 1 holds >= 2 files.")
    lA = {f"{x}+{z}": price(mu, cov, rsd, [x, z]) for x in TIER1 for z in TIER2}
    vA = np.array(list(lA.values()))
    print(f"  {len(lA)} pairs: {vA.min():+.2f} .. {vA.max():+.2f}e-6,  mean {vA.mean():+.2f}e-6"
          f"   (w45a: {W45A['limit2_mean']:+.2f})")

    # ---------------------------------------------------------------- MODEL B (correct)
    print("\n=== MODEL B (CORRECT): the auto pair is a 2-subset of tier 1 ===")
    lB = {}
    for a, b in itertools.combinations(TIER1, 2):
        lB[f"{a}+{b}"] = price(mu, cov, rsd, [a, b])
    vB = np.array(list(lB.values()))
    hit = [k for k in lB if PICK in k.split("+")]
    p_pick = len(hit) / len(lB)
    print(f"  {len(lB)} pairs: {vB.min():+.2f} .. {vB.max():+.2f}e-6,  "
          f"uniform mean {vB.mean():+.2f}e-6")
    for k in sorted(lB, key=lB.get, reverse=True):
        mark = "  <-- contains the pick" if PICK in k.split("+") else ""
        print(f"    {k:47s} {lB[k]:+8.2f}{mark}")
    print(f"  P(pick inside the auto 2) = {len(hit)}/{len(lB)} = {p_pick:.3f}"
          f"   (4-file tier was 2/4 = 0.500)")
    miss = [v for k, v in lB.items() if PICK not in k.split("+")]
    print(f"  restricted to the {len(miss)} pairs that MISS the pick: mean {np.mean(miss):+.2f}e-6")

    # named tiebreaks, for the branches a human can reason about
    by_date = sub.groupby("stem")["date"].min().to_dict()
    order = sorted(TIER1, key=lambda s: by_date[s])
    named = {
        "earliest-first": order[:2],
        "latest-first": order[-2:],
        "filename A-Z": sorted(TIER1)[:2],
        "filename Z-A": sorted(TIER1, reverse=True)[:2],
    }
    print("\n  named tiebreaks (MODEL B):")
    for nm, pr in named.items():
        print(f"    {nm:16s} {'+'.join(pr):47s} {price(mu,cov,rsd,pr):+8.2f}"
              f"{'   <-- contains the pick' if PICK in pr else ''}")

    # ---------------------------------------------------------------- TAU / corr sensitivity
    print("\n=== TAU SWEEP (w19c's sign-flip test), MODEL B uniform mean ===")
    taus = [0.0, 0.5, 1.0, 1.72, 2.5, 3.0, 5.0]
    sweep = {}
    for t in taus:
        mt, ct, rt, _ = solve(t, CORR_CONS)
        vals_t = [price(mt, ct, rt, list(p)) for p in itertools.combinations(TIER1, 2)]
        sweep[t] = float(np.mean(vals_t))
        print(f"  tau {t:5.2f}   MODEL B mean {sweep[t]:+8.2f}e-6")
    flip = any(np.sign(sweep[t]) != np.sign(sweep[0.0]) and abs(sweep[t]) > 0.05 for t in taus)

    mo, co, ro, _ = solve(0.0, CORR_OPT)
    vB_opt = float(np.mean([price(mo, co, ro, list(p)) for p in itertools.combinations(TIER1, 2)]))
    print(f"\n=== corr sensitivity ===\n  MODEL B mean: conservative {vB.mean():+.2f}   "
          f"optimistic {vB_opt:+.2f}e-6")

    # ---------------------------------------------------------------- registered predictions
    print("\n=== the registered predictions (w57_prereg.txt) ===")
    r1 = bool(abs(l1[PICK]["cost"]) <= 1.0)
    print(f"  R1 wanted branch costs {l1[PICK]['cost']:+.3f}e-6  band |.|<=1  -> "
          f"{'CONFIRMED' if r1 else '*** FALSIFIED ***'}")
    NEW = "w40_ad211stdcorr"
    r2 = bool(NEW in l1 and 0.0 < l1[NEW]["cost"] <= 6.0)
    print(f"  R2 new branch {NEW} costs {l1[NEW]['cost']:+.3f}e-6  band (0,6]  -> "
          f"{'CONFIRMED' if r2 else '*** FALSIFIED ***'}")
    r3 = bool(15.0 <= avg1 <= 21.0 and avg1 < W45A["uniform_limit1"])
    print(f"  R3 limit-1 uniform avg {avg1:+.2f}e-6  band [15,21] AND "
          f"< {W45A['uniform_limit1']}  -> {'CONFIRMED' if r3 else '*** FALSIFIED ***'}")
    r4 = bool(5.0 <= vB.mean() <= 12.0 and vB.mean() < W45A["limit2_mean"])
    print(f"  R4 MODEL B uniform {vB.mean():+.2f}e-6  band [5,12] AND "
          f"< {W45A['limit2_mean']}  -> {'CONFIRMED' if r4 else '*** FALSIFIED ***'}")
    r5 = bool(abs(p_pick - 0.400) < 1e-9)
    print(f"  R5 P(pick in auto 2) {p_pick:.3f}  == 0.400  -> "
          f"{'CONFIRMED' if r5 else '*** FALSIFIED ***'}")
    print(f"  R6 GATE B max|M-wI| {offI:.2e} < 1e-6  -> {'CONFIRMED' if r6 else '*** FALSIFIED ***'}")
    r7 = bool(avg1 < W45A["uniform_limit1"] and vB.mean() < W45A["limit2_mean"])
    print(f"  R7 the day's 10 sends were NET FAVOURABLE on the click price  -> "
          f"{'CONFIRMED' if r7 else '*** FALSIFIED -- the sends made the exposure WORSE ***'}")

    out = dict(read_utc_note="tiers derived live; compare `tiers` against the board before quoting",
               tiers=dict(slot1=TIER1, slot2=TIER2, slot1_public=float(t1v),
                          slot2_public=float(t2v)),
               wanted=list(WANTED), pick=PICK, n_scored=int(len(sub)),
               cv={k: cv[k] for k in NAMES}, lb={k: LB[k] for k in NAMES},
               beta=beta, gamma=gamma, gap=ghat,
               limit1=l1, uniform_limit1=avg1,
               model_a=dict(pairs=lA, mean=float(vA.mean())),
               model_b=dict(pairs=lB, mean=float(vB.mean()), p_pick=p_pick,
                            mean_missing_pick=float(np.mean(miss)),
                            named={k: v for k, v in named.items()},
                            mean_optimistic=vB_opt),
               tau_sweep={str(k): v for k, v in sweep.items()}, tau_flip=flip,
               supersedes="w45a_tierprice (4-file tier 1, stale since the 08-22 sends)",
               w45a_published=W45A,
               r1=r1, r2=r2, r3=r3, r4=r4, r5=r5, r6=r6, r7=r7)
    with open(os.path.join(HERE, "w57a_tierprice2.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote experiments/w57a_tierprice2.json")


if __name__ == "__main__":
    main()
