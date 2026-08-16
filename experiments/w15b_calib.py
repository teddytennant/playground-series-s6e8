"""w15b step 4b -- validate the calibration the price ladder stands on.

w15b_price.py found A*(p-hat) - observed AUC = +117e-6 for a cross-fitted isotonic
p-hat. For a PERFECTLY calibrated score that difference is identically zero: the two
sides use disjoint information (score distribution vs labels) but the Bernoulli identity
forces them equal. 117e-6 is 2.3 noise floors, so it is not rounding -- either the
estimator is wrong or the calibration is over-dispersed, and the price ladder's baseline
depends on which.

Three arms, in the order that isolates the cause:

  A. IN-SAMPLE isotonic. Calibrated on the same rows it scores, so it is calibrated by
     construction. If A* - observed is not ~0 here, `bayes_auc` itself is wrong. This is
     the code control, and it must pass before anything else is read.

  B. CROSS-FITTED isotonic (what price.py used). The gap here minus the gap in A is the
     out-of-fold over-dispersion of PAVA -- it chases noise into blocks at the extremes,
     and clipping those to 1e-6 / 1-1e-6 makes them maximally dispersed.

  C. CROSS-FITTED BINNED-PAVA at several bin counts. Equal-count bins, bin means, PAVA on
     the bin means, linear interpolation between bin centres. B bins of ~n/B rows each
     cannot produce a 1e-6 block unless a whole bin is pure, so the extreme-value
     pathology is bounded by the bin size. Sweeping B shows the gap closing, and the
     largest B whose gap is within the noise floor is the calibrator the ladder should use.

Also reported: Var(p-hat) per arm. The price ladder is a dispersion argument, so seeing
the dispersion move with the calibrator is the honest way to state its uncertainty.
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
from w15b_price import bayes_auc  # noqa: E402

BLEND = os.environ.get("W15B_BLEND", "blend159av_h3")


def pava(yv: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators on already-ordered (value, weight) pairs."""
    v = yv.astype(np.float64).copy()
    ww = w.astype(np.float64).copy()
    idx = list(range(len(v)))
    vals, wts, sizes = [], [], []
    for i in idx:
        vals.append(v[i]); wts.append(ww[i]); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-1] + wts[-2]
            vals[-2] = (vals[-1] * wts[-1] + vals[-2] * wts[-2]) / w2
            wts[-2] = w2
            sizes[-2] += sizes[-1]
            vals.pop(); wts.pop(); sizes.pop()
    out = np.empty(len(v))
    k = 0
    for val, sz in zip(vals, sizes):
        out[k:k + sz] = val
        k += sz
    return out


def binned_calibrator(s_fit: np.ndarray, y_fit: np.ndarray, nbins: int):
    """Fit equal-count binned means + PAVA; return an interpolating callable."""
    o = np.argsort(s_fit, kind="stable")
    ss, yy = s_fit[o], y_fit[o]
    edges = np.linspace(0, len(ss), nbins + 1).astype(int)
    lo, hi = edges[:-1], edges[1:]
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    cnt = (hi - lo).astype(np.float64)
    csum_y = np.r_[0.0, np.cumsum(yy)]
    csum_s = np.r_[0.0, np.cumsum(ss)]
    ybar = (csum_y[hi] - csum_y[lo]) / cnt
    xbar = (csum_s[hi] - csum_s[lo]) / cnt
    fit = pava(ybar, cnt)
    return lambda q: np.interp(q, xbar, fit)


def report(name, p, y, out):
    a_obs = roc_auc_score(y, p)
    a_star = bayes_auc(p)
    out[name] = dict(observed=float(a_obs), bayes=float(a_star),
                     gap=float(a_star - a_obs), var=float(p.var()),
                     brier=float(np.mean((y - p) ** 2)))
    print(f"  {name:34s} observed {a_obs:.6f}   A* {a_star:.6f}   "
          f"A*-obs {a_star - a_obs:+.6f}   var {p.var():.6f}   "
          f"Brier {np.mean((y - p) ** 2):.6f}", flush=True)


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    folds = get_folds(y)
    raw = np.load(os.path.join(SUB, f"oof_{BLEND}.npy"))
    print(f"pack {BLEND}: raw OOF AUC {roc_auc_score(y, raw):.6f}\n", flush=True)
    out = {}

    print("=== A. in-sample isotonic (CODE CONTROL: gap must be ~0) ===")
    ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    ir.fit(raw, y)
    report("in-sample isotonic", np.clip(ir.predict(raw), 1e-6, 1 - 1e-6), y, out)

    print("\n=== A2. in-sample binned-PAVA (second code control) ===")
    for nb in (200, 2000):
        f = binned_calibrator(raw, y, nb)
        report(f"in-sample binned B={nb}", np.clip(f(raw), 1e-6, 1 - 1e-6), y, out)

    print("\n=== B. cross-fitted isotonic (what price.py used) ===")
    p = np.empty(len(y))
    for tr_i, va_i in folds:
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(raw[tr_i], y[tr_i])
        p[va_i] = ir.predict(raw[va_i])
    report("cross-fitted isotonic", np.clip(p, 1e-6, 1 - 1e-6), y, out)

    print("\n=== C. cross-fitted binned-PAVA, bin-count sweep ===")
    best = None
    for nb in (50, 100, 200, 500, 1000, 2000, 5000, 20000):
        p = np.empty(len(y))
        for tr_i, va_i in folds:
            f = binned_calibrator(raw[tr_i], y[tr_i], nb)
            p[va_i] = f(raw[va_i])
        p = np.clip(p, 1e-6, 1 - 1e-6)
        name = f"cross-fitted binned B={nb}"
        report(name, p, y, out)
        if best is None or abs(out[name]["gap"]) < abs(out[best[0]]["gap"]):
            best = (name, nb, p.copy())

    print(f"\nbest-calibrated arm: {best[0]}   gap {out[best[0]]['gap']:+.6f}")
    np.save(os.path.join(ROOT, "experiments", "w15b_phat.npy"), best[2])
    out["chosen"] = dict(name=best[0], nbins=best[1])
    with open(os.path.join(ROOT, "experiments", "w15b_calib.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote experiments/w15b_phat.npy, w15b_calib.json")


if __name__ == "__main__":
    main()
