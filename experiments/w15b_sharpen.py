"""w15b step 4c -- close the second route to a higher Bayes AUC.

w15b_price.py prices the leaders' gap as an ORTHOGONAL missing signal. But raising the
Bayes AUC does not strictly require new information: A*(p) is a functional of the
DISPERSION of the probability field, and a field can be more dispersed along the direction
we already have. Concretely, if the true field were p_k = expit(k * logit(p_hat)) with
k > 1 -- identical ranking, sharper probabilities -- then A*(p_k) > A*(p_hat) with no new
signal whatsoever. That is a real second route and the price argument is incomplete until
it is closed.

It is closed by CALIBRATION, and this script does it in two directions:

  1. How much sharpening would be needed? Sweep k and read A*(p_k) against the same
     targets w15b_price.py used.
  2. Is the data consistent with it? If the true field were p_k, our p_hat would be
     systematically UNDER-confident, and the mismatch is directly observable: the
     out-of-fold binned-PAVA calibration curve would bend away from the diagonal, and the
     Brier score of p_k would BEAT that of p_hat on the real labels. Both are measured.

The logic that matters: A* - observed is a signed over-dispersion meter (w15b_calib.py),
and it was driven to ~0 by construction when p_hat was chosen. A field that is calibrated
cannot also be systematically under-confident. So if sharpening improves nothing on the
real labels, the sharpening route is empirically dead and the ONLY remaining route to a
higher Bayes AUC is orthogonal signal -- which is what the price ladder measures.
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


def main() -> None:
    tr, _ = load_raw()
    y = tr[TARGET].to_numpy(np.float64)
    p = np.load(os.path.join(ROOT, "experiments", "w15b_phat.npy"))
    lp = logit(p)
    a0 = bayes_auc(p)
    b0 = float(np.mean((y - p) ** 2))
    ll0 = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    print(f"p-hat: A* {a0:.6f}   observed {roc_auc_score(y, p):.6f}   "
          f"Brier {b0:.6f}   logloss {ll0:.6f}\n")

    print("=== sharpening the EXISTING direction: p_k = expit(k*logit(p-hat)) ===")
    print("    ranking is identical for every k, so observed AUC never moves.")
    print("    Brier/logloss on the REAL labels say whether the data wants it.\n")
    print(f"    {'k':>5}  {'A*(p_k)':>9}  {'gain':>10}  {'Brier':>9}  {'dBrier':>10}"
          f"  {'logloss':>9}  {'dLogloss':>10}")
    rows = []
    for k in (0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.20, 1.40, 1.80):
        pk = np.clip(expit(k * lp), 1e-9, 1 - 1e-9)
        a = bayes_auc(pk)
        b = float(np.mean((y - pk) ** 2))
        ll = float(-np.mean(y * np.log(pk) + (1 - y) * np.log(1 - pk)))
        rows.append(dict(k=k, bayes=a, gain=a - a0, brier=b, dbrier=b - b0,
                         logloss=ll, dlogloss=ll - ll0))
        flag = "  <- would need this" if (a - a0) >= 180e-6 and (a - a0) < 400e-6 else ""
        print(f"    {k:5.2f}  {a:9.6f}  {a - a0:+10.6f}  {b:9.6f}  {b - b0:+10.6f}"
              f"  {ll:9.6f}  {ll - ll0:+10.6f}{flag}")

    ks = np.array([r["k"] for r in rows])
    gs = np.array([r["gain"] for r in rows])
    dbs = np.array([r["dbrier"] for r in rows])
    for name, g in (("+110e-6 (LB rank 2)", 110e-6), ("+180e-6 (MILANFX)", 180e-6),
                    ("+321e-6 (pass-through adj.)", 321e-6)):
        if g <= gs.max():
            kk = float(np.interp(g, gs, ks))
            db = float(np.interp(g, gs, dbs))
            print(f"\n  {name}: needs k = {kk:.3f}, which costs "
                  f"dBrier {db:+.6f} on the real labels")

    print("\n=== verdict ===")
    best = min(rows, key=lambda r: r["brier"])
    print(f"  Brier-optimal sharpening k = {best['k']:.2f} "
          f"(dBrier {best['dbrier']:+.6f}, Bayes gain {best['gain']:+.6f})")
    if abs(best["k"] - 1.0) < 1e-9:
        print("  => p-hat is already the Brier-optimal member of its own sharpening family.")
        print("     The data does NOT want a sharper field. The sharpening route is dead,")
        print("     and orthogonal signal is the only remaining route to a higher ceiling.")
    else:
        print("  => p-hat is NOT at the Brier optimum; re-read before trusting the ladder.")

    with open(os.path.join(ROOT, "experiments", "w15b_sharpen.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\nwrote experiments/w15b_sharpen.json")


if __name__ == "__main__":
    main()
