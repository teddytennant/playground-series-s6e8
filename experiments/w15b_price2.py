"""w15b step 4e -- the CORRECTED price ladder. Supersedes w15b_price.py's inversion.

THE ERROR IN w15b_price.py, and it matters
------------------------------------------
That script priced the gap as `A*(p_tau) - A*(p_hat)`: how much the CEILING rises when an
orthogonal signal of size tau exists. That is the wrong difference. Adding orthogonal
signal to the truth does two things at once:

  * it raises the ceiling      A*(p_tau)                  (what a competitor who has it gets)
  * it LOWERS our own score    auc_under(p_tau, p_hat)    (our ranking is now missing something)

and the leader's advantage over us is the sum of both, not just the first. w15b_power.py
computed the correct difference incidentally and the discrepancy is 7x: at tau = 0.36 the
ceiling rises +168e-6 but the pack's own score falls by ~1069e-6, so the true advantage is
**+1237e-6**, not the +180e-6 the old ladder reported.

THE SECOND ERROR, which the first one hides
-------------------------------------------
A field p_tau = expit(logit(p_hat) + tau*z) is INCONSISTENT WITH OUR OWN OBSERVED AUC for
any tau > 0. If the truth really had that much orthogonal noise in it, we would be scoring
0.968922, not the 0.970024 we actually measure. Any candidate truth has to reproduce what
we observe.

THE CORRECT CONSTRUCTION
------------------------
Index the family of truths CONSISTENT WITH OUR MEASUREMENT by how much orthogonal signal
they contain:

    p*(k, tau) = expit( k * logit(p_hat) + tau * z ),   z ~ N(0,1) independent

For each tau, solve for k such that

    auc_under(p*, p_hat) == our observed OOF AUC          <-- the consistency constraint

i.e. the sharpness k rises just enough to keep our own score where it actually is while the
orthogonal component tau grows. Then

    gap(tau) = A*(p*) - observed AUC

is exactly the maximum advantage a competitor holding that signal could have over us. At
tau = 0 the solution is k = 1 and gap = 0 -- we are Bayes-optimal and nobody can beat us,
which is the correct boundary condition and a check on the solver.

Note AUC is rank-only and both p_hat and k*logit(p_hat) are monotone in logit(p_hat), so
`auc_under(p*, p_hat)` does not depend on k through the ranking -- only through the truth.
That also means the Jensen mismatch between expit(k*lp) and E_z[p*] is irrelevant here: it
cannot change a ranking.

Inverting gap(tau) onto the leaderboard gaps gives the number this run exists to produce.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import TARGET, load_raw  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15b_price import bayes_auc, expit, logit  # noqa: E402

SEED = 20260815
NREP = int(os.environ.get("W15B_NREP2", "4"))
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, 0.90]
TARGETS = {
    "+18e-6  rayk anti-student (author's low)": 18e-6,
    "+36e-6  rayk anti-student (author's high)": 36e-6,
    "+50e-6  one CV noise floor": 50e-6,
    "+110e-6 gap to LB rank 2 (0.97117)": 110e-6,
    "+180e-6 gap to MILANFX (0.97124)": 180e-6,
    "+321e-6 same at the measured 56% CV->LB pass-through": 321e-6,
}


class RankedAUC:
    """auc_under(p_true, s) with the ranking of `s` FIXED and pre-sorted.

    Only the probability field changes as we search over k, so the O(n log n) sort is paid
    once and each evaluation is O(n). Ties in `s` are handled exactly via tie-group sums.
    """

    def __init__(self, s: np.ndarray):
        self.o = np.argsort(s, kind="stable")
        ss = s[self.o]
        self.bnd = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1], True])[:-1]

    def __call__(self, p_true: np.ndarray) -> float:
        ps = p_true[self.o]
        qs = 1.0 - ps
        gp = np.add.reduceat(ps, self.bnd)
        gq = np.add.reduceat(qs, self.bnd)
        gpq = np.add.reduceat(ps * qs, self.bnd)
        below = np.r_[0.0, np.cumsum(gq)[:-1]]
        num = float(np.sum(gp * below) + 0.5 * (np.sum(gp * gq) - np.sum(gpq)))
        den = float(ps.sum() * qs.sum() - np.sum(ps * qs))
        return num / den


def solo_auc(z: np.ndarray, pt: np.ndarray) -> float:
    o = np.argsort(z, kind="stable")
    ps, qs = pt[o], 1 - pt[o]
    below = np.r_[0.0, np.cumsum(qs)[:-1]]
    return float(np.sum(ps * below)) / float(ps.sum() * qs.sum() - np.sum(ps * qs))


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    p = np.load(os.path.join(ROOT, "experiments", "w15b_phat.npy"))
    lp = logit(p)
    a_obs = roc_auc_score(y, p)
    ru = RankedAUC(p)
    print(f"calibrated pack p-hat: observed OOF AUC {a_obs:.6f}   A*(p-hat) "
          f"{bayes_auc(p):.6f}")
    print("consistency constraint: every candidate truth must return our own score to "
          f"{a_obs:.6f}\n", flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    print(f"  {'tau':>5}  {'k solved':>9}  {'our AUC':>9}  {'ceiling A*':>11}"
          f"  {'GAP':>10}  {'solo AUC of z':>13}")
    for tau in TAUS:
        Gs, Ks, As, Ss = [], [], [], []
        for _ in range(1 if tau == 0 else NREP):
            z = rng.standard_normal(len(p))

            def ours(k):
                return ru(expit(k * lp + tau * z))

            lo, hi = 1.0, 1.0
            if tau > 0:
                # our score falls as tau grows; raise k until it is restored
                while ours(hi) < a_obs and hi < 6.0:
                    hi *= 1.25
                for _ in range(45):
                    mid = 0.5 * (lo + hi)
                    if ours(mid) < a_obs:
                        lo = mid
                    else:
                        hi = mid
            k = 0.5 * (lo + hi)
            pt = expit(k * lp + tau * z)
            Ks.append(k); As.append(ru(pt))
            Gs.append(bayes_auc(pt) - a_obs); Ss.append(solo_auc(z, pt))
        k, ao, g, s = np.mean(Ks), np.mean(As), np.mean(Gs), np.mean(Ss)
        gsd = float(np.std(Gs, ddof=1)) if len(Gs) > 1 else 0.0
        rows.append(dict(tau=tau, k=float(k), our_auc=float(ao), gap=float(g),
                         gap_sd=gsd, solo=float(s)))
        print(f"  {tau:5.2f}  {k:9.5f}  {ao:9.6f}  {ao + g:11.6f}  {g:+10.6f}"
              f"  {s:13.4f}", flush=True)

    print("\n=== INVERTED: what the missing predictor must look like ===")
    gs = np.array([r["gap"] for r in rows])
    ts = np.array([r["tau"] for r in rows])
    ss = np.array([r["solo"] for r in rows])
    inv = {}
    for name, tgt in TARGETS.items():
        if tgt > gs.max():
            print(f"  {name:54s}  beyond the ladder")
            continue
        t = float(np.interp(tgt, gs, ts)); so = float(np.interp(tgt, gs, ss))
        inv[name] = dict(tau=t, solo=so, gap=tgt)
        print(f"  {name:54s}  tau {t:.4f}   **solo AUC {so:.4f}**")

    with open(os.path.join(ROOT, "experiments", "w15b_price2.json"), "w") as f:
        json.dump(dict(observed=float(a_obs), ladder=rows, inverted=inv), f, indent=2)
    print("\nwrote experiments/w15b_price2.json")


if __name__ == "__main__":
    main()
