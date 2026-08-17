"""w19b — the FULL calibration of the paired CV->LB instrument, with the control that decides it.

w19a asked slot 4's question as posed ("is there a per-file transfer variance tau?") and got
tau_hat = 0.000e-6, 95% profile upper 1.7e-6. But its own splits refused to hold still:

    tight pairs (LAW-IF sd < 5e-6)   tau_hat 0.000
    loose pairs (LAW-IF sd >= 5e-6)  tau_hat 3.562
    all                              tau_hat 0.000     rms resid / rms predicted = 1.66

An additive per-file term cannot make that pattern, so this file fits the general model

    dLB = round_G( beta * dcv + N(0, lambda^2 sigma_ab^2 + 2 tau^2) )

with exact triangular rounding as in w19a.  beta is the parameter that matters most: every
paired prediction registered in this workspace, and w18a's click price, assumes
E[A_k(test)] = cv_k exactly, i.e. beta = 1.  lambda is a multiplicative error in the LAW-IF sd.

THE CONTROL, AND IT IS THE POINT OF THE FILE.  Composite ML over 194 pairs drawn from 51 files
is not 194 independent observations, the pairs are dominated by near-duplicates, and the
response is quantised on a 1e-5 grid.  An estimator like that can manufacture structure out of
a perfectly correct model.  So every statistic below is located in a PARAMETRIC BOOTSTRAP OF
THE NULL: draw the whole 51-file system from (beta=1, lambda=1, tau=0) with the full LAW-IF
covariance, round on the real grid, rebuild all 194 pairs, and refit exactly the same way.
If the observed beta_hat, lambda_hat and their band-by-band pattern sit inside that null, the
instrument is calibrated and the pattern is the estimator, not the board.  Nothing here is
read as a finding until it clears its own null.

beta and lambda are POST-HOC — w19_prereg.txt registered tau alone.  Labelled as such, and the
out-of-sample test is registered in w19c_shipprereg.txt before this slot's upload.

    .venv/bin/python experiments/w19b_calib.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
G = 1e-5
U = 1e-6
GU = G / U

BND = dict(beta=(-1.0, 3.0), lam=(0.05, 6.0), tau=(0.0, 40.0))
X0 = dict(beta=1.0, lam=1.0, tau=0.5)


def _I(z, mu, s):
    t = (z - mu) / s
    return s * (t * norm.cdf(t) + norm.pdf(t))


def pmf(m, mu, s):
    x = m * GU
    return np.clip((_I(x + GU, mu, s) - 2.0 * _I(x, mu, s) + _I(x - GU, mu, s)) / GU, 1e-300, None)


def nll(theta, m, dcv, var0, free):
    th = dict(X0, beta=1.0, lam=1.0, tau=0.0)
    th.update(dict(zip(free, theta)))
    if th["lam"] <= 0.01 or th["tau"] < 0:
        return 1e12
    return -np.log(pmf(m, th["beta"] * dcv, np.sqrt(th["lam"] ** 2 * var0 + 2.0 * th["tau"] ** 2))).sum()


def fit(m, dcv, var0, free):
    if not free:
        return {}, nll([], m, dcv, var0, [])
    r = minimize(nll, [X0[f] for f in free], args=(m, dcv, var0, free),
                 method="L-BFGS-B", bounds=[BND[f] for f in free])
    return dict(zip(free, r.x)), float(r.fun)


def profile(par, grid, m, dcv, var0, free_all):
    others = [f for f in free_all if f != par]
    out = []
    for g in grid:
        def f(x):
            th = dict(zip(others, x))
            th[par] = g
            return nll([th[k] for k in free_all], m, dcv, var0, free_all)
        r = minimize(f, [X0[o] for o in others], method="L-BFGS-B",
                     bounds=[BND[o] for o in others])
        out.append(float(r.fun))
    return np.array(out)


def interval(grid, prof, drop=1.92):
    ok = grid[prof <= prof.min() + drop]
    return float(ok.min()), float(ok.max())


def stats(m, dcv, var0, sel_mine, bands):
    """Every statistic this file reads, computed identically on real and simulated boards."""
    s = {}
    s["beta_all"], s["lam_all"] = (lambda t: (t["beta"], t["lam"]))(fit(m, dcv, var0, ["beta", "lam"])[0])
    th, _ = fit(m[sel_mine], dcv[sel_mine], var0[sel_mine], ["beta", "lam"])
    s["beta_ours"], s["lam_ours"] = th["beta"], th["lam"]
    s["tau_all"] = fit(m, dcv, var0, ["tau"])[0]["tau"]
    for i, sel in enumerate(bands):
        s[f"lam_b{i}"] = fit(m[sel], dcv[sel], var0[sel], ["lam"])[0]["lam"]
        s[f"beta_b{i}"] = fit(m[sel], dcv[sel], var0[sel], ["beta", "lam"])[0]["beta"]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=19_2026)
    a = ap.parse_args()

    P = pd.read_csv(os.path.join(HERE, "w19a_pairs.csv"))
    S_x = np.load(os.path.join(HERE, "w19a_Sx.npy"))
    meta = json.load(open(os.path.join(HERE, "w19a_files.json")))
    names, fam = meta["names"], np.array(meta["fam"])
    cv = np.array(meta["cv"])
    LB = meta["lb"]
    K = len(names)
    m = P.m.to_numpy()
    dcv = P.dcv.to_numpy()
    var0 = P.var0.to_numpy()
    mine = ~P.foreign.to_numpy()
    print(f"w19a artefacts: {K} files, {len(P)} within-family pairs ({int(mine.sum())} ours-only), "
          f"LAW-IF S_x reused")

    q = np.quantile(np.abs(dcv[mine]), [0, 1 / 3, 2 / 3, 1.0])
    bands = [mine & (np.abs(dcv) >= q[i]) & (np.abs(dcv) <= q[i + 1]) for i in range(3)]

    # ------------------------------------------------------------------ nested ladder
    print("\n=== nested model ladder, exact rounding, composite ML over the pairs ===")
    print(f"  {'model':26s} {'set':6s} {'beta':>7s} {'lambda':>7s} {'tau':>6s} {'2*dNLL':>8s}")
    LAD = [("null beta=1 lam=1 tau=0", []), ("tau only (w19a)", ["tau"]), ("lambda only", ["lam"]),
           ("beta only", ["beta"]), ("beta + lambda", ["beta", "lam"]),
           ("beta + lambda + tau", ["beta", "lam", "tau"])]
    ladder = {}
    for tag, sel in (("all", np.ones(len(P), bool)), ("ours", mine)):
        base = nll([], m[sel], dcv[sel], var0[sel], [])
        for name, free in LAD:
            th, f = fit(m[sel], dcv[sel], var0[sel], free)
            ladder[f"{tag}|{name}"] = dict(theta=th, dev=2 * (base - f))
            print(f"  {name:26s} {tag:6s} {th.get('beta',1.0):7.3f} {th.get('lam',1.0):7.3f} "
                  f"{th.get('tau',0.0):6.3f} {2*(base-f):8.1f}")

    obs = stats(m, dcv, var0, mine, bands)
    print(f"\n  |dCV| bands (ours-only, terciles): "
          + "  ".join(f"[{q[i]:.1f},{q[i+1]:.1f}] n={int(bands[i].sum())}" for i in range(3)))
    print(f"  band lambda  " + "  ".join(f"{obs[f'lam_b{i}']:6.3f}" for i in range(3)))
    print(f"  band beta    " + "  ".join(f"{obs[f'beta_b{i}']:6.3f}" for i in range(3)))

    # ------------------------------------------------------------------ profiles
    print("\n=== profile intervals (chi2_1), OURS-ONLY pairs, beta+lambda+tau ===")
    FREE = ["beta", "lam", "tau"]
    th_min, _ = fit(m[mine], dcv[mine], var0[mine], FREE)
    prof = {}
    for par, grid in (("beta", np.linspace(-0.5, 3.0, 71)), ("lam", np.linspace(0.1, 3.0, 59)),
                      ("tau", np.linspace(0.0, 12.0, 49))):
        pr = profile(par, grid, m[mine], dcv[mine], var0[mine], FREE)
        lo, hi = interval(grid, pr)
        prof[par] = dict(lo=lo, hi=hi, hat=float(th_min[par]))
        print(f"  {par:5s} hat {th_min[par]:7.4f}   95% [{lo:.3f}, {hi:.3f}]"
              f"   contains {'1' if par!='tau' else '0'}? "
              f"{'yes' if (lo <= (0 if par=='tau' else 1) <= hi) else 'NO'}")

    # ------------------------------------------------------------------ THE NULL CONTROL
    print(f"\n=== THE CONTROL: {a.boot} whole-system draws from the NULL "
          f"(beta=1, lambda=1, tau=0) ===")
    print("    full LAW-IF covariance across all 51 files, real grid, all 194 pairs rebuilt,")
    print("    every statistic above recomputed by the identical estimator.")
    ev, EV = np.linalg.eigh(S_x)
    L = EV * np.sqrt(np.clip(ev, 0.0, None))
    gap = {f: float(np.mean([LB[names[i]] / U - cv[i] / U for i in np.flatnonzero(fam == f)]))
           for f in sorted(set(fam))}
    base = np.array([cv[i] / U + gap[fam[i]] for i in range(K)])
    ia = np.array([names.index(x) for x in P.a])
    ib = np.array([names.index(x) for x in P.b])
    rng = np.random.default_rng(a.seed)
    null = {k: [] for k in obs}
    for _ in range(a.boot):
        v = base + L @ rng.standard_normal(K)
        lbs = np.round(v / GU) * GU
        mb = np.round((lbs[ia] - lbs[ib]) / GU).astype(int)
        for k, val in stats(mb, dcv, var0, mine, bands).items():
            null[k].append(val)
    print(f"  {'statistic':12s} {'observed':>9s} {'null mean':>10s} {'null sd':>8s} "
          f"{'null 2.5%':>10s} {'null 97.5%':>11s} {'pctile':>7s}")
    loc = {}
    for k in ["beta_all", "beta_ours", "lam_all", "lam_ours", "tau_all",
              "lam_b0", "lam_b1", "lam_b2", "beta_b0", "beta_b1", "beta_b2"]:
        v = np.array(null[k])
        p = float((v <= obs[k]).mean())
        loc[k] = dict(obs=float(obs[k]), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                      lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), pctile=p)
        flag = "  <-- OUTSIDE" if (p < 0.025 or p > 0.975) else ""
        print(f"  {k:12s} {obs[k]:9.3f} {v.mean():10.3f} {v.std(ddof=1):8.3f} "
              f"{np.percentile(v,2.5):10.3f} {np.percentile(v,97.5):11.3f} {p*100:6.1f}%{flag}")
    outside = [k for k, d in loc.items() if d["pctile"] < 0.025 or d["pctile"] > 0.975]
    print(f"\n  statistics outside their own null's central 95%: "
          f"{', '.join(outside) if outside else 'NONE'}  of {len(loc)} looked at")
    print(f"  Sidak-corrected two-sided threshold for {len(loc)} statistics: "
          f"reject only below {(1-0.95**(1/len(loc)))/2*100:.2f}% or above "
          f"{(1-(1-0.95**(1/len(loc)))/2)*100:.2f}% percentile")
    surv = [k for k, d in loc.items()
            if d["pctile"] < (1 - 0.95 ** (1 / len(loc))) / 2
            or d["pctile"] > 1 - (1 - 0.95 ** (1 / len(loc))) / 2]
    print(f"  surviving multiplicity: {', '.join(surv) if surv else 'NONE'}")
    print("  -> " + ("the parametric structure w19b found is REPRODUCED BY THE CORRECT MODEL; "
                     "it is the estimator, not the board." if not outside else
                     "at least one statistic is real and needs a mechanism."))

    # a single omnibus: does the null reproduce the ALL-vs-OURS beta split that looked so bad?
    d_obs = obs["beta_all"] - obs["beta_ours"]
    d_null = np.array(null["beta_all"]) - np.array(null["beta_ours"])
    print(f"\n  omnibus, the split that motivated this file: beta_all - beta_ours = {d_obs:+.3f}"
          f"   null {d_null.mean():+.3f} +/- {d_null.std(ddof=1):.3f}"
          f"   two-sided p = {float(np.mean(np.abs(d_null - d_null.mean()) >= abs(d_obs - d_null.mean()))):.3f}")

    json.dump(dict(ladder=ladder, obs={k: float(v) for k, v in obs.items()},
                   profile=prof, null=loc, outside=outside,
                   bands=[[float(q[i]), float(q[i + 1]), int(bands[i].sum())] for i in range(3)],
                   n_pairs=int(len(P)), n_ours=int(mine.sum()), boot=a.boot, seed=a.seed),
              open(os.path.join(HERE, "w19b_calib.json"), "w"), indent=1)
    print("\nwrote w19b_calib.json")


if __name__ == "__main__":
    main()
