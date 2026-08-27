"""Evaluate the w96b pre-registered rule against w96a's numbers. Prints BUILD or NO-BUILD.

The rule is `experiments/w96b_prereg.txt`, committed at 2eda33b while
`logs_w96a_teprior_folds.txt` still held zero AUC lines. This file only reads it back and
applies it; it deliberately contains no threshold that is not quoted there verbatim.

Nothing here writes anything anywhere, including on import.
"""
from __future__ import annotations
import json
import math
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w96a_teprior_folds.json")
MIN_MEAN_E6 = 25.0     # rule C
MIN_POSITIVE = 4       # rule B
K_SE = 2.0             # rule A


def var(v):
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)


def main():
    if not os.path.exists(SRC):
        print(f"NO-BUILD: {SRC} does not exist -- w96a has not run.")
        return 2
    res = json.load(open(SRC))
    folds = res.get("folds", {})
    S = [f["signal_e6"] for f in folds.values() if "signal_e6" in f]
    N = [f["null_e6"] for f in folds.values() if "null_e6" in f]
    print(f"w96a: {len(folds)} fold slot(s), {len(S)} complete   C1={res.get('C1', 'not reached')}")
    for k in sorted(folds, key=int):
        f = folds[k]
        s, n = f.get("signal_e6"), f.get("null_e6")
        print(f"  fold {k}: signal {s if s is None else round(s, 3)!s:>10}e-6   "
              f"null {n if n is None else round(n, 3)!s:>10}e-6")
    if len(S) < 5 or len(N) < 5:
        print(f"NO-BUILD (NOT YET DECIDABLE): the rule reads all five folds; "
              f"{len(S)} signal / {len(N)} null are present. Re-run when the job finishes.")
        return 2
    if res.get("C1") != "PASS":
        print("NO-BUILD: C1 did not pass. No arm comparison off an unverified variant.")
        return 1

    mS, mN = sum(S) / len(S), sum(N) / len(N)
    se = math.sqrt(var(S) / len(S) + var(N) / len(N))
    a = (mS - mN) > K_SE * se
    b = sum(1 for x in S if x > 0) >= MIN_POSITIVE
    c = mS >= MIN_MEAN_E6
    print(f"\n  SIGNAL mean {mS:+.3f}e-6  sd {math.sqrt(var(S)):.3f}e-6")
    print(f"  NULL   mean {mN:+.3f}e-6  sd {math.sqrt(var(N)):.3f}e-6")
    print(f"  A SEPARATION  {mS - mN:+.3f}e-6 vs {K_SE:.0f}se = {K_SE * se:.3f}e-6"
          f"      {'PASS' if a else 'FAIL'}")
    print(f"  B CONSISTENCY {sum(1 for x in S if x > 0)}/5 folds positive, need {MIN_POSITIVE}"
          f"                {'PASS' if b else 'FAIL'}")
    print(f"  C MAGNITUDE   mean {mS:+.3f}e-6 vs floor {MIN_MEAN_E6:.0f}e-6"
          f"          {'PASS' if c else 'FAIL'}")
    if a and b and c:
        print("\nBUILD. Launch experiments/w96c_build_cachewin.py per the prereg.")
        return 0
    if a and b and not c:
        print("\nNO-BUILD: REAL BUT SMALL. A and B hold, so the effect is not noise -- it is "
              "simply not worth a rebuild four days out. Record it and stop (prereg, "
              "'a real +10e-6 on one member four days out is a fact, not a project').")
        return 1
    print("\nNO-BUILD. Do not re-run with other folds, seeds or presets to move a number "
          "across a threshold (prereg, final paragraph of THE RULE).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
