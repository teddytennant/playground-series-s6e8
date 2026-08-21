"""w46c — the CV->LB predictor, ERA-CORRECTED. Import this, not w26d's inline copy.

⚠⚠ WHY THIS MODULE EXISTS. w30b's predictor is BROKEN on every file now in the send queue.

w30b was fitted on 78 scored files through 2026-08-20 and its residual sd is 7.76e-6. Five
files first scored on 2026-08-21 were therefore a clean held-out test, the first since w30a.
Every one of them came in BELOW prediction:

    w36_ad199std_rescale  pred 0.971201  actual 0.97116   -40.9e-6
    w36_ad199std_hybrid   pred 0.971172  actual 0.97114   -32.5e-6
    w36_ad199stdcorr      pred 0.971212  actual 0.97118   -32.4e-6
    w36_ad199std          pred 0.971199  actual 0.97117   -29.2e-6
    w34_ad195stdcorr      pred 0.971184  actual 0.97117   -14.1e-6
                                          mean -29.82e-6, sd 9.76e-6

Mean residual -29.82e-6 against a claimed sd of 7.76e-6 over n=5: z = -8.6. This is not a
draw. For scale, w26d's own report on w36_ad199stdcorr -- written into the permanent Kaggle
submission log at 00:07 UTC today -- reads "predicted LB 0.971212 with P(beats the 0.97118
account best) 1.00e+00". It scored 0.97118. A probability published as 1.00 for an event that
did not happen.

IT IS NOT MODEL FORM, AND IT IS NOT DRIFT. Two checks:
  * In-sample residual by CV quintile is FLAT (+1.24 / +0.05 / -1.07 / +0.02 / -0.22e-6), so
    the linear-in-CV slope is not failing at the top of the range and this is not an
    extrapolation artefact.
  * By day first sent, the ten days 08-10..08-20 run -5.73..+3.44e-6 with no trend, then
    08-21 steps to -29.82e-6. A step, not a slope.

WHAT IT IS. The break falls exactly between the ad194 wave (08-20, residual +0.13e-6) and the
ad195/ad199 wave (08-21, -29.82e-6) -- which is exactly where JOURNAL w44 section 6
independently closed the member-import line on a FLAT dose-response. Those are the same fact
seen from two sides: past ~ad194 an imported member still raises cross-fitted CV but no longer
raises true test performance, so CV runs ahead of LB and a predictor linear in CV over-reads.
w44 section 6 closed the line as "stop building"; it ALSO invalidates the pricer, and that
second consequence went unrecorded for a run.

⚠ THE SCOPE IS TOTAL. Every one of the 77 unsent files is from the ad195/ad197/ad199/ad202/
ad211 waves. There is not one file in the queue that the uncorrected predictor prices
correctly.

CORRECTION APPLIED. A single era level term for ad >= 195, estimated at -29.82e-6 with
se 9.76/sqrt(5) = 4.37e-6, and a predictive sd of sqrt(9.76^2 + 4.37^2) = 10.69e-6 that
carries the uncertainty in that term. Five points is a thin estimate and the module says so:
`ERA_N` is exported so callers can report it.
"""
from __future__ import annotations

import json, os, re
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))

_M = json.load(open(os.path.join(HERE, "w30b_corrterm.json")))
C = _M["coefs"]
SD_OLD = _M["resid_sd_new"]                     # 7.756e-6, valid for ad <= 194
_t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv"])
MU = _t[_t.cv >= 0.97].cv.mean()

# The five 2026-08-21 held-out rows, verbatim. Kept as data so the estimate is auditable and
# so a later run can extend it rather than re-derive it.
ERA_ROWS = [("w36_ad199std_rescale", 0.9701219373, 0.97116),
            ("w36_ad199std_hybrid",  0.9701231283, 0.97114),
            ("w36_ad199stdcorr",     0.9701400060, 0.97118),
            ("w36_ad199std",         0.9701323250, 0.97117),
            ("w34_ad195stdcorr",     0.9701247949, 0.97117)]
ERA_MIN_AD = 195
ERA_N = len(ERA_ROWS)


def family(stem):
    from stdflag import family as _f
    return _f(stem)


def is_std(stem):
    from stdflag import is_std as _s
    return _s(stem)


def is_corr(stem):
    return "corr" in stem


def wave(stem):
    """The adNNN pack size in the stem, or None for pre-ad builds (blend*, stack_pub*)."""
    m = re.search(r"ad(\d{3})", stem)
    return int(m.group(1)) if m else None


def new_era(stem):
    w = wave(stem)
    return w is not None and w >= ERA_MIN_AD


def _raw(cv, stem):
    lb6 = C["const"] + C["cv_e6"] * (cv - MU) * 1e6 + C.get(f"fam[{family(stem)}]", 0.0)
    if is_std(stem):
        lb6 += C["std"]
    if is_corr(stem):
        lb6 += C["corr"]
    return lb6 * 1e-6


_res = np.array([(lb - _raw(cv, s)) / 1e-6 for s, cv, lb in ERA_ROWS])
ERA_SHIFT = float(_res.mean())                                   # -29.82e-6
ERA_SE = float(_res.std(ddof=1) / np.sqrt(ERA_N))                # 4.37e-6
SD_NEW = float(np.sqrt(_res.std(ddof=1) ** 2 + ERA_SE ** 2))     # 10.69e-6


def predict_lb(cv, stem):
    p = _raw(cv, stem)
    return p + ERA_SHIFT * 1e-6 if new_era(stem) else p


def pred_sd(stem):
    return (SD_NEW if new_era(stem) else SD_OLD) * 1e-6


def scenario_probs(cv, stem):
    """(point prediction, P(rounds to >= 0.97119), P(rounds to exactly 0.97118))."""
    p, s = predict_lb(cv, stem), pred_sd(stem)
    p19 = float(1.0 - norm.cdf((0.971185 - p) / s))
    p18 = float(norm.cdf((0.971185 - p) / s) - norm.cdf((0.971175 - p) / s))
    return p, p19, p18


if __name__ == "__main__":
    print(f"ERA_SHIFT {ERA_SHIFT:+.2f}e-6  se {ERA_SE:.2f}  n {ERA_N}  "
          f"sd_new {SD_NEW:.2f}e-6 (was {SD_OLD:.2f})")
    print(f"{'stem':26s} {'raw':>10s} {'corrected':>10s} {'P19':>7s} {'P18':>7s}")
    for s, cv, lb in ERA_ROWS:
        p, a, b = scenario_probs(cv, s)
        print(f"  {s:24s} {_raw(cv,s):.6f} {p:.6f} {a:7.3f} {b:7.3f}   actual {lb:.5f}")
    q = json.load(open(os.path.join(HERE, "w45b_unsent_cv.json")))
    n_new = sum(new_era(r["stem"]) for r in q)
    print(f"\nunsent queue: {n_new} of {len(q)} files are ad>={ERA_MIN_AD} — "
          f"i.e. in the regime where the UNCORRECTED predictor is known broken.")
    for r in sorted(q, key=lambda r: -r["cv"])[:6]:
        p, a, b = scenario_probs(r["cv"], r["stem"])
        print(f"  {r['stem']:26s} raw {_raw(r['cv'],r['stem']):.6f} -> {p:.6f}  "
              f"P19 {a:.3f}  P18 {b:.3f}")
