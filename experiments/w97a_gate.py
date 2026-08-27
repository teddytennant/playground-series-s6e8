"""w97a: the w97 prereg's rule, in code.

`experiments/w97_prereg.txt` §3 states five conditions. This applies them and returns
ENROL or NO-ENROL. It exists so the rule cannot be re-read after the table is on screen --
the whole failure mode w97_prereg.txt §5 is written to forbid.

    .venv/bin/python experiments/w97a_gate.py                 # report
    .venv/bin/python experiments/w97a_gate.py --confirm       # exit 0 only on ENROL

Exit codes: 0 ENROL, 1 NO-ENROL, 2 NOT YET DECIDABLE (inputs missing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")

WIN = "lgbm_teprior_windowed"
GLB = "lgbm_teprior_global"

G2_FLOOR = 1.0e-6          # prereg §3 G2
G3_CEIL = 10.0e-6          # prereg §3 G3
G1_MIN_POSITIVE = 4        # prereg §3 G1, of 5 reps
SD_RATIO = (0.95, 1.45)    # prereg §3 G4, the standing member gate


def load(out: str):
    csv = os.path.join(EXP, out + ".csv")
    js = os.path.join(EXP, out + ".json")
    smry = os.path.join(ROOT, "oof_w96", f"summary_{WIN}.json")
    missing = [p for p in (csv, js, smry) if not os.path.exists(p)]
    if missing:
        print("NOT YET DECIDABLE -- missing:")
        for p in missing:
            print("   ", os.path.relpath(p, ROOT))
        sys.exit(2)
    return pd.read_csv(csv), json.load(open(js)), json.load(open(smry))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="w97a_teprior_value")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()

    df, js, smry = load(a.out)
    for c in (WIN, GLB):
        if c not in df.columns:
            print(f"NOT YET DECIDABLE -- arm {c!r} absent from {a.out}.csv")
            sys.exit(2)

    # d(windowed) - d(global): both arms are held-out AUC on the SAME rows against the
    # SAME pack, so the pack term cancels exactly and the difference is already paired.
    delta = (df[WIN] - df[GLB]).to_numpy()
    n = len(delta)
    mean, sd = float(delta.mean()), float(delta.std(ddof=1)) if n > 1 else float("nan")
    npos = int((delta > 0).sum())
    ratio = float(smry.get("sd_test_over_sd_oof", float("nan")))

    print(f"reps {n}   Delta per rep: " + " ".join(f"{d:+.2e}" for d in delta))
    print(f"mean {mean:+.3e}  sd {sd:.3e}  positive {npos}/{n}")
    print(f"registered prediction was [0, +3e-6], modal +1e-6\n")

    g0 = bool(js.get("gate", {}).get("passed", False))
    g1 = npos >= G1_MIN_POSITIVE
    g2 = mean >= G2_FLOOR
    g3 = mean <= G3_CEIL
    g4 = SD_RATIO[0] <= ratio <= SD_RATIO[1]

    gj = js.get("gate", {})
    print(f"G0 INSTRUMENT  cat4 {gj.get('mean', float('nan')):+.6f} "
          f"vs published {gj.get('published', float('nan')):+.6f}"
          f"      {'PASS' if g0 else 'FAIL'}")
    print(f"G1 CONSISTENCY {npos}/{n} reps positive, need {G1_MIN_POSITIVE}"
          f"              {'PASS' if g1 else 'FAIL'}")
    print(f"G2 MAGNITUDE   mean {mean:+.3e} vs floor {G2_FLOOR:+.1e}"
          f"        {'PASS' if g2 else 'FAIL'}")
    print(f"G3 CEILING     mean {mean:+.3e} vs ceiling {G3_CEIL:+.1e}"
          f"      {'PASS' if g3 else 'FAIL'}")
    print(f"G4 PROVENANCE  sd_test/sd_oof {ratio:.4f} in [{SD_RATIO[0]}, {SD_RATIO[1]}]"
          f"   {'PASS' if g4 else 'FAIL'}")

    ok = g0 and g1 and g2 and g3 and g4
    print()
    if ok:
        print("-> ENROL. Copy oof_/test_/summary_" + WIN + " into oof/ and NOTHING else.")
        print("   The global twin stays out: oof/ is globbed, not allow-listed (w96 §10).")
        print("   This does NOT change SELECT_THESE.md and does NOT trigger a submission")
        print("   (prereg §4).")
    else:
        print("-> NO-ENROL.")
        if not g0:
            print("   G0 failed, so nothing in the table is comparable to RESEARCH.md and")
            print("   the other four verdicts should not be quoted at all.")
        if g0 and not g3:
            print("   The ceiling is what failed: re-run at reps 5..9 BEFORE believing it")
            print("   (prereg §3 G3). That is the only re-run this document permits.")
        if g0 and g3 and not (g1 and g2):
            print("   Modal outcome. The encoding is not distinguishable from one more")
            print("   redundant member. Do NOT switch to d(windowed) or d(both) to rescue")
            print("   it -- the twin was built at cost to prevent exactly that (§5).")
    if a.confirm:
        sys.exit(0 if ok else 1)
    return ok


if __name__ == "__main__":
    main()
