"""w15b step 4d -- is the price ladder an artefact of the calibrator?

w15b_price.py's headline ("the leader's 18e-5 costs an orthogonal predictor of solo AUC
~0.53") is derived from ONE calibrated probability field. w15b_calib.py showed the
calibrator choice moves the BASELINE A* by ~340e-6 across the sweep -- about twice the
effect being priced. So the obvious objection is that the ladder is a calibrator artefact.

It is not, and the reason is visible in w15b_calib.py's last column: Var(p) moves only
0.2% across every arm, while A* moves 340e-6. The ladder inverts a DISPERSION question,
not a level question, so it should be near-invariant. This script checks that rather than
asserting it: re-derive the whole ladder on three fields spanning the calibrator sweep
(under-, correctly, and over-dispersed) and compare the inverted answers.

The quantity that must be stable is the GAIN column (A*(p_tau) - A*(p_hat)) and the solo
AUC it inverts to -- not A* itself.
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
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from w15b_calib import binned_calibrator  # noqa: E402
from w15b_price import bayes_auc, expit, logit  # noqa: E402

BLEND = "blend159av_h3"
SEED = 20260815
TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80]
TARGETS = {"+110e-6 (LB rank 2)": 110e-6, "+180e-6 (MILANFX)": 180e-6,
           "+321e-6 (pass-through adj.)": 321e-6,
           "+18e-6 (rayk anti-student, author's low)": 18e-6,
           "+36e-6 (rayk anti-student, author's high)": 36e-6}


def ladder(p, nrep=6):
    lp = logit(p)
    a0 = bayes_auc(p)
    rng = np.random.default_rng(SEED)
    gains, solos = [], []
    for tau in TAUS:
        G, S = [], []
        for _ in range(1 if tau == 0 else nrep):
            z = rng.standard_normal(len(p))
            pt = expit(lp + tau * z)
            G.append(bayes_auc(pt) - a0)
            o = np.argsort(z, kind="stable")
            pts, qts = pt[o], 1 - pt[o]
            below = np.r_[0.0, np.cumsum(qts)[:-1]]
            S.append(float(np.sum(pts * below)) /
                     float(pt.sum() * (1 - pt).sum() - np.sum(pt * (1 - pt))))
        gains.append(float(np.mean(G)))
        solos.append(float(np.mean(S)))
    return a0, np.array(gains), np.array(solos)


def main():
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    raw = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))

    fields = {}
    for nb in (50, 100, 2000):
        p = np.empty(len(y))
        for tr_i, va_i in folds:
            f = binned_calibrator(raw[tr_i], y[tr_i], nb)
            p[va_i] = f(raw[va_i])
        fields[f"binned B={nb}"] = np.clip(p, 1e-6, 1 - 1e-6)
    p = np.empty(len(y))
    for tr_i, va_i in folds:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(raw[tr_i], y[tr_i])
        p[va_i] = ir.predict(raw[va_i])
    fields["isotonic (over-dispersed)"] = np.clip(p, 1e-6, 1 - 1e-6)

    out = {}
    for name, p in fields.items():
        a0, g, s = ladder(p)
        print(f"\n=== {name} ===  A* {a0:.6f}   Var(p) {p.var():.6f}   "
              f"observed {roc_auc_score(y, p):.6f}")
        print("    tau   gain      solo AUC")
        for t, gg, ss in zip(TAUS, g, s):
            print(f"   {t:4.2f}  {gg:+.6f}   {ss:.4f}")
        inv = {}
        for tname, tgt in TARGETS.items():
            if tgt <= g.max():
                inv[tname] = dict(tau=float(np.interp(tgt, g, TAUS)),
                                  solo=float(np.interp(tgt, g, s)))
        out[name] = dict(a_star=a0, var=float(p.var()), inverted=inv)

    print("\n\n=== INVERSION STABILITY ACROSS CALIBRATORS (the thing that must hold) ===")
    print(f"{'target':44s} " + "  ".join(f"{n:>22s}" for n in fields))
    for tname in TARGETS:
        cells = []
        for n in fields:
            v = out[n]["inverted"].get(tname)
            cells.append(f"tau {v['tau']:.3f} solo {v['solo']:.4f}" if v else "  beyond ladder  ")
        print(f"{tname:44s} " + "  ".join(f"{c:>22s}" for c in cells))

    with open(os.path.join(ROOT, "experiments", "w15b_price_robust.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote experiments/w15b_price_robust.json")


if __name__ == "__main__":
    main()
