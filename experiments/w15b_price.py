"""w15b step 4 -- price the leader's 18e-5 in units of MISSING SIGNAL.

The identity this rests on
--------------------------
If labels are y_i ~ Bernoulli(p_i) independently, then for any score s the expected
Mann-Whitney numerator and denominator are exactly

    E[#pos * #neg * AUC] = sum_{i!=j} p_i (1-p_j) [ 1{s_i>s_j} + 0.5*1{s_i=s_j} ]
    E[#pos * #neg]       = sum_{i!=j} p_i (1-p_j)

so the achievable AUC of ANY score against a Bernoulli(p) label field is a closed-form
functional of p and s alone -- no labels needed. Setting s = p gives the BAYES AUC of the
probability field p. Call it A*(p). It is computable in O(n log n) by sorting.

Two things fall out, and the second is the point of this script.

1. A CALIBRATION IDENTITY, free of charge. If our calibrated pack score p-hat really is
   P(y|x), then A*(p-hat) must equal the pack's OBSERVED OOF AUC. It is a genuine check:
   the two are computed from disjoint information (one from the score distribution alone,
   one from the labels alone). Agreement validates the calibration that step 2 leans on.

2. THE PRICE OF THE GAP. A*(p) is a strictly increasing functional of the dispersion of p.
   Our pack sits at 0.970049 CV / 0.97106 LB; MILANFX sits at 0.97124 LB. Whatever the
   leaders have, in this framework it is extra dispersion in their probability field --
   signal about y that our p-hat does not carry. So ask the question backwards:

       how much extra independent signal must exist for the Bayes AUC to reach the
       leader's score, and what would a feature carrying that signal look like?

   Model the missing signal as an independent logit-scale term:  p_tau = expit(logit(p-hat)
   + tau*z),  z ~ N(0,1) independent of everything we have. Sweep tau, and for each report
   (a) A*(p_tau), the Bayes AUC once that signal exists, and (b) the STANDALONE AUC of z
   itself against labels drawn from p_tau -- i.e. how good a predictor the missing feature
   would have to be on its own. (b) is what makes the answer communicable: "they would
   need a feature worth 0.62 solo AUC" is a statement group 2 can act on, where "they
   would need tau = 0.31" is not.

   Independence is the CONSERVATIVE choice. A missing signal correlated with what we
   already have buys less AUC per unit of dispersion than an orthogonal one, so the tau
   this script reports is a LOWER bound on what the leaders would need. That is the right
   direction: if even the cheapest possible version of the missing signal is implausibly
   large, the "they are inside our columns" branch is dead.

Pure numpy over saved vectors. No model is fitted.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
from common import SUB, TARGET, get_folds, load_raw  # noqa: E402

BLEND = os.environ.get("W15B_BLEND", "blend159av_h3")
SEED = 20260815
NREP = int(os.environ.get("W15B_NREP", "8"))


def bayes_auc(p: np.ndarray) -> float:
    """A*(p): the AUC obtained by ranking on p when y ~ Bernoulli(p), in expectation.

    numerator   = sum_{i!=j} p_i (1-p_j) [1{p_i>p_j} + 0.5 1{p_i=p_j}]
    denominator = (sum p)(sum 1-p) - sum p(1-p)
    Ties are handled exactly by grouping equal values.
    """
    p = np.asarray(p, dtype=np.float64)
    q = 1.0 - p
    o = np.argsort(p, kind="stable")
    ps, qs = p[o], q[o]
    # group boundaries over equal p
    bnd = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1], True])
    gp = np.add.reduceat(ps, bnd[:-1])        # sum of p per tie-group
    gq = np.add.reduceat(qs, bnd[:-1])        # sum of (1-p) per tie-group
    gpq = np.add.reduceat(ps * qs, bnd[:-1])  # sum of p(1-p) per tie-group
    # strictly-below mass of (1-p)
    below = np.r_[0.0, np.cumsum(gq)[:-1]]
    num = float(np.sum(gp * below) + 0.5 * (np.sum(gp * gq) - np.sum(gpq)))
    den = float(p.sum() * q.sum() - np.sum(p * q))
    return num / den


def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_oof(score, y, folds):
    out = np.empty(len(y))
    for tr_i, va_i in folds:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(score[tr_i], y[tr_i])
        out[va_i] = ir.predict(score[va_i])
    return np.clip(out, 1e-6, 1 - 1e-6)


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    raw = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    a_obs = roc_auc_score(y, raw)
    print(f"pack {BLEND}: observed OOF AUC {a_obs:.6f}   n={len(y):,}", flush=True)

    p = calibrate_oof(raw, y, folds)
    a_cal = roc_auc_score(y, p)
    a_self = bayes_auc(p)
    print(f"cross-fitted isotonic p-hat: observed AUC {a_cal:.6f}   "
          f"analytic A*(p-hat) {a_self:.6f}")
    print(f"  ** calibration identity check: A* - observed = {a_self - a_cal:+.6f} **")
    print(f"  mean p-hat {p.mean():.6f} vs base rate {y.mean():.6f}   "
          f"Brier {np.mean((y - p) ** 2):.6f}")
    print(f"  var(p-hat) {p.var():.6f}   E[p(1-p)] {np.mean(p * (1 - p)):.6f}", flush=True)

    # ------------------------------------------------------------------ the tau ladder
    print("\n=== how much INDEPENDENT extra signal buys how much Bayes AUC ===")
    print("  p_tau = expit(logit(p-hat) + tau*z),  z ~ N(0,1) independent")
    print("  'solo AUC of z' = the standalone AUC the missing feature would have\n")
    rng = np.random.default_rng(SEED)
    taus = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.20, 1.80]
    lp = logit(p)
    rows = []
    for tau in taus:
        A, solo = [], []
        reps = 1 if tau == 0.0 else NREP
        for _ in range(reps):
            z = rng.standard_normal(len(p))
            pt = expit(lp + tau * z)
            A.append(bayes_auc(pt))
            # standalone value of z: rank by z alone against a Bernoulli(p_tau) field.
            # same identity, s = z: sum_{i!=j} pt_i (1-pt_j) 1{z_i > z_j} / denominator
            o = np.argsort(z, kind="stable")
            pts, qts = pt[o], 1 - pt[o]
            below = np.r_[0.0, np.cumsum(qts)[:-1]]
            num = float(np.sum(pts * below))
            den = float(pt.sum() * (1 - pt).sum() - np.sum(pt * (1 - pt)))
            solo.append(num / den)
        Am, Asd = float(np.mean(A)), float(np.std(A, ddof=1) if len(A) > 1 else 0.0)
        sm = float(np.mean(solo))
        rows.append(dict(tau=tau, bayes_auc=Am, sd=Asd, solo_auc_of_z=sm,
                         gain=Am - a_self))
        print(f"  tau {tau:4.2f}   A*(p_tau) {Am:.6f} +-{Asd:.6f}   "
              f"gain over p-hat {Am - a_self:+.6f}   solo AUC of z {sm:.4f}", flush=True)

    # invert: what tau reaches the two targets
    print("\n=== inverting the ladder onto the leaderboard gap ===")
    ts = np.array([r["tau"] for r in rows])
    ga = np.array([r["gain"] for r in rows])
    so = np.array([r["solo_auc_of_z"] for r in rows])
    targets = {
        "+0.000180 (leader LB gap, 1:1 pass-through)": 180e-6,
        "+0.000321 (same gap at the measured 56% pass-through)": 321e-6,
        "+0.000050 (one noise floor)": 50e-6,
        "+0.000005 (the h3/ens4 margin)": 5e-6,
    }
    inv = {}
    for name, g in targets.items():
        if g > ga.max():
            print(f"  {name:52s}  beyond the ladder")
            continue
        t = float(np.interp(g, ga, ts))
        s = float(np.interp(g, ga, so))
        inv[name] = dict(tau=t, solo_auc=s, gain=g)
        print(f"  {name:52s}  tau {t:.4f}   missing feature solo AUC {s:.4f}")

    out = dict(blend=BLEND, auc_observed=float(a_obs), auc_calibrated=float(a_cal),
               bayes_auc_of_phat=float(a_self), ladder=rows, inverted=inv)
    with open(os.path.join(ROOT, "experiments", "w15b_price.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/w15b_price.json")


if __name__ == "__main__":
    main()
