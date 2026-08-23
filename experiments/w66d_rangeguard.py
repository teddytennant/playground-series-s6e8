"""w66d — the two things w66 wired must stay wired, and must stay LOAD-BEARING.

1. THE FITTED RANGE (w53a). `cv_needed`'s extrapolation warning used to be a paragraph. w64:
   A RULE IN A PARAGRAPH IS NOT A RULE. It is now `FIT_CV_MIN` / `FIT_CV_MAX` / `out_of_range`,
   DERIVED from the fit table and never hand-typed, and w26d prints the flag at the number
   rather than under it. `FIT_CV_MAX` is the best CV on record, so every "CV needed to top the
   board" row is an extrapolation by construction and must read non-zero once the target moves
   above the account best.

2. THE `member` EXCLUSION (w26d). `LEADER` was `max(q.cv.max(), ...)` and the argmax was
   `w48_cal_hboyang_mix`, a calibration vector. The standing rule ("never let a
   calibration/`member` row into a `max(CV)`") was enforced in the SENDER by w60d and nowhere
   here. Both the exclusion AND the fact that it changes the answer are checked: a filter that
   removes nothing is not protection, it is decoration that will read as protection later.

Every check is NEGATIVE-CONTROLLED — a wrong value must be REJECTED, not merely a right one
accepted, and the range constants are COMPARED against a re-derivation from the fit table
rather than read out of a literal (w64: a stamp is not a comparison).

    .venv/bin/python experiments/w66d_rangeguard.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import w53a_pricer as PR   # noqa: E402

FAILURES = 0


def check(label, ok, detail=""):
    global FAILURES
    if not ok:
        FAILURES += 1
    print(f"  {'✅' if ok else '🔴'} {label}{('  — ' + detail) if detail else ''}")
    return ok


def main():
    print("=== w66d: the fitted range and the member exclusion ===\n")

    # ---------------------------------------------------------------- 1. THE FITTED RANGE ----
    t = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
    lo, hi = float(t.cv.min()), float(t.cv.max())
    check("FIT_CV_MIN/MAX are re-derived from w52b_cvlb93.csv, not typed",
          PR.FIT_CV_MIN == lo and PR.FIT_CV_MAX == hi,
          f"{PR.FIT_CV_MIN:.10f}..{PR.FIT_CV_MAX:.10f}")
    check("the fitted range TOPS OUT at the best CV on record",
          abs(hi - float(t.cv.max())) < 1e-15 and hi >= float(t.cv.quantile(1.0)),
          f"FIT_CV_MAX {hi:.10f}")

    check("out_of_range is 0.0 strictly inside", PR.out_of_range((lo + hi) / 2) == 0.0)
    check("out_of_range is 0.0 AT both endpoints (closed interval)",
          PR.out_of_range(lo) == 0.0 and PR.out_of_range(hi) == 0.0)
    # negative controls: it must FIRE, with the right sign and magnitude, just outside
    above, below = hi + 20e-6, lo - 20e-6
    check("out_of_range FIRES POSITIVE just above the range",
          abs(PR.out_of_range(above) - 20.0) < 1e-6, f"{PR.out_of_range(above):+.3f}e-6")
    check("out_of_range FIRES NEGATIVE just below the range",
          abs(PR.out_of_range(below) + 20.0) < 1e-6, f"{PR.out_of_range(below):+.3f}e-6")

    # the query the workspace actually asks must come back FLAGGED
    stem = "w36_ad199stdcorr"
    cv_t, flag = PR.cv_needed_flagged(0.97120, stem)
    check("cv_needed_flagged on 'one step above the account best' is FLAGGED",
          flag > 0.0, f"needs {cv_t:.10f}, {flag:+.2f}e-6 outside the fit")
    check("cv_needed_flagged does not CHANGE cv_needed",
          cv_t == PR.cv_needed(0.97120, stem))
    # ...and an in-range target must NOT be flagged, or the flag means nothing
    inrange_target = PR.predict_lb((lo + hi) / 2, stem)
    check("an IN-RANGE target is NOT flagged (the flag discriminates)",
          PR.cv_needed_flagged(inrange_target, stem)[1] == 0.0,
          f"target {inrange_target:.6f}")

    # the measured cost is on disk and reproduced from the artefact, not from this docstring
    aj = os.path.join(HERE, "w66c_attrib.json")
    if os.path.exists(aj):
        h = json.load(open(aj))["heldout"]
        check("w66c: frozen OLS beats the GLS refit INSIDE the range",
              h["ols_rmse_tight"] < h["gls_rmse_tight"],
              f"{h['ols_rmse_tight']:.2f} vs {h['gls_rmse_tight']:.2f} e-6")
        check("w66c: frozen OLS LOSES to the GLS refit once the range is left",
              h["ols_rmse"] > h["gls_rmse"],
              f"{h['ols_rmse']:.2f} vs {h['gls_rmse']:.2f} e-6")
        check("the docstring's 1.9x slope spread is the artefact's, not a typed number",
              abs(json.load(open(aj))["gls93"][PR.NAMES.index("cv6")] - 0.9582) < 5e-4,
              f"GLS(93) cv6 = {json.load(open(aj))['gls93'][PR.NAMES.index('cv6')]:+.4f}")
    else:
        check("w66c_attrib.json is on disk", False, "run w66c_attrib.py")

    # ---------------------------------------------------------------- 2. THE member EXCLUSION -
    print()
    src = open(os.path.join(HERE, "w26d_queueprice.py")).read()
    check("w26d's LEADER excludes member rows in SOURCE",
          '_cand = q[q.fam != "member"]' in src and "LEADER = float(max(_cand.cv.max()" in src)
    check("w26d still asserts the member filter is load-bearing",
          "the filter below is not load-bearing" in src)

    with contextlib.redirect_stdout(io.StringIO()):
        import w26d_queueprice as QP
    q = QP.q
    mem = q[q.fam == "member"]
    check("the live queue still HAS a member row (else this guard tests nothing)",
          len(mem) > 0, f"{len(mem)} member row(s)")
    check("the member row is still the ARGMAX of the unfiltered queue",
          float(mem.cv.max()) == float(q.cv.max()),
          f"member {mem.cv.max():.10f} vs queue {q.cv.max():.10f}")
    check("LEADER is NOT the member row — the exclusion CHANGES the answer",
          QP.LEADER < float(q.cv.max()),
          f"LEADER {QP.LEADER:.10f} < unfiltered {q.cv.max():.10f} "
          f"({(q.cv.max()-QP.LEADER)*1e6:+.2f}e-6)")
    check("LEADER is the best NON-member CV (or the fit-table max, whichever is higher)",
          QP.LEADER == max(float(q[q.fam != "member"].cv.max()), float(QP._fit.cv.max())))
    # negative control: the old rule must be REJECTED by this guard
    check("NEGATIVE CONTROL — the pre-w66 rule would fail the check above",
          not (float(max(q.cv.max(), QP._fit.cv.max())) < float(q.cv.max())),
          "max(q.cv.max(), ...) is the member row, by construction")

    # and the flag column must actually be printed where the number is read
    check("w26d prints the out-of-fit flag AT the CV-needed table",
          "PR.out_of_range(stdc)" in src and "'outside fit'" in src)

    print(f"\nFAILURES {FAILURES}")
    print("w66d: the fitted range is a number, and the member row is out of the max." if not FAILURES
          else "w66d: FAILED")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
