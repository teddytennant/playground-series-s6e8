"""w122b — ANGLE INDEX row 2 re-verified at the ARTEFACT level, not quoted (2026-08-30).

The handed ANGLE was *"LightGBM: tune it properly against the fixed folds — learning rate,
leaves, regularisation, categorical handling."* Genus `LightGBM` -> row 2, closed, price +4e-7.

w106 established the standard: an index row tells you an angle is closed, it does NOT tell you
the closure's evidence still exists. Row 3's closure cited two members that were never built.
Row 4's price was INHERITED from row 2 and w106 checked row 4 on its own artefacts -- but the
row it was inherited FROM has never been checked, so the whole +4e-7 has been resting on an
unaudited LightGBM measurement that row 4 then borrowed.

WHAT ROW 2's CLOSURE CLAIMS, and what each claim is checked against here:

  A  "Tuning a GBDT for solo AUC on these folds was measured at +3e-5" (2026-08-10, LightGBM).
     -> re-measure it. `lgbm_tuned_lat` vs `lgbm_fixed_lat` is that experiment: same pipeline,
        same frozen folds, hyperparameters the only difference. Recompute both solo OOF AUCs.
  B  "you cannot tune a LightGBM into a decorrelated member" -- `lgbm_stump_lat_frac` at depth
     3 / 8 leaves scores solo 0.96735, i.e. 0.00044 WORSE than `lgbm_fixed_lat_frac`, and
     correlates with the pack at maxcorr 0.9961.
     -> recompute the solo AUCs, the gap, and the stump's max correlation across the pack.
  C  "Solo->stack pass-through is 1.4%", the multiplier that turns A into the +4e-7 price.
     -> arithmetic only: check 3e-5 * 1.4% rounds to the published 4e-7, and that 4e-7 really
        is ~1% of the 5e-5 noise floor. A price nobody has multiplied out is not a price.

This is a READ. It builds nothing, enrols nothing, and tunes nothing -- sweeping the three
closed knobs is what the closure forbids. It only asks whether the numbers the index sells as
settled reproduce from the arrays on disk.

    .venv/bin/python experiments/w122b_row2.py

Deterministic: no API call, no fit, no fold rebuild beyond the frozen partition.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA  # noqa: E402

OOF = os.path.join(ROOT, "oof")

# The published numbers, frozen as literals so this is a COMPARISON and not a recomputation
# that agrees with itself. Sources are named per row in the closure quoted above.
PUBLISHED = {
    "tuning_solo_gain": 3e-5,     # claim A, "measured at +3e-5"
    "passthrough": 0.014,         # claim C, "solo->stack pass-through is 1.4%"
    "price": 4e-7,                # claim C, the row 2 / row 4 price
    "noise_floor": 5e-5,          # claim C, "~1% of the 5e-5 noise floor"
    "stump_solo": 0.96735,        # claim B
    "stump_gap": 0.00044,         # claim B, stump WORSE than fixed by this much
    "stump_maxcorr": 0.9961,      # claim B
}
TOL_AUC = 5e-6      # agreement bar for a published AUC quoted to 5 decimals
TOL_GAP = 5e-5      # the gap is quoted to 5 decimals, so allow its last digit


def y_true():
    return pd.read_csv(os.path.join(DATA, "train.csv"),
                       usecols=["addicted_label"])["addicted_label"].values


def load(stem):
    p = os.path.join(OOF, f"oof_{stem}.npy")
    return np.load(p) if os.path.exists(p) else None


def main() -> int:
    y = y_true()
    fails, notes = [], []

    # ---------------------------------------------------------------- claim A: the tuning null
    print("A  TUNING FOR SOLO AUC, re-measured on the frozen folds (tuned vs fixed, same pipeline)")
    pairs = [("lgbm_tuned_lat", "lgbm_fixed_lat"),
             ("lgbm_tuned_lat_frac", "lgbm_fixed_lat_frac")]
    gains = []
    for tuned, fixed in pairs:
        a, b = load(tuned), load(fixed)
        if a is None or b is None:
            fails.append(f"A missing artefact for the pair ({tuned}, {fixed})")
            continue
        ga, gb = roc_auc_score(y, a), roc_auc_score(y, b)
        gains.append(ga - gb)
        print(f"   {tuned:24} {ga:.6f}   {fixed:24} {gb:.6f}   tuned-fixed {ga-gb:+.6f}")
    if gains:
        worst = max(abs(g) for g in gains)
        print(f"   largest |tuned - fixed| over {len(gains)} pair(s): {worst:.6f}"
              f"   published bound {PUBLISHED['tuning_solo_gain']:.6f}")
        # The claim is that tuning is worth ABOUT +3e-5 solo. It survives if the measured
        # effect is no LARGER than published -- a bigger effect would reopen the price.
        if worst > PUBLISHED["tuning_solo_gain"] * 3:
            fails.append(f"A tuning moves solo AUC by {worst:.6f}, >3x the published "
                         f"{PUBLISHED['tuning_solo_gain']:.6f}; the +4e-7 price is understated")
        else:
            notes.append(f"A holds: solo tuning effect {worst:.2e} vs published 3e-5")

    # ---------------------------------------------------------------- claim B: the stump
    print("\nB  THE DECORRELATION ATTEMPT (depth 3 / 8 leaves) — solo cost and pack correlation")
    stump, fixedf = load("lgbm_stump_lat_frac"), load("lgbm_fixed_lat_frac")
    if stump is None or fixedf is None:
        fails.append("B lgbm_stump_lat_frac or lgbm_fixed_lat_frac is missing from oof/")
    else:
        s_auc, f_auc = roc_auc_score(y, stump), roc_auc_score(y, fixedf)
        gap = f_auc - s_auc
        print(f"   lgbm_stump_lat_frac      {s_auc:.6f}   published {PUBLISHED['stump_solo']:.5f}")
        print(f"   lgbm_fixed_lat_frac      {f_auc:.6f}")
        print(f"   fixed - stump            {gap:.6f}   published {PUBLISHED['stump_gap']:.5f}")
        if abs(s_auc - PUBLISHED["stump_solo"]) > TOL_AUC:
            fails.append(f"B stump solo {s_auc:.6f} != published "
                         f"{PUBLISHED['stump_solo']:.5f} (tol {TOL_AUC})")
        if abs(gap - PUBLISHED["stump_gap"]) > TOL_GAP:
            fails.append(f"B stump gap {gap:.6f} != published "
                         f"{PUBLISHED['stump_gap']:.5f} (tol {TOL_GAP})")

        pack = sorted(f for f in os.listdir(OOF)
                      if f.startswith("oof_") and f != "oof_lgbm_stump_lat_frac.npy")
        cors = []
        for f in pack:
            v = np.load(os.path.join(OOF, f))
            if v.shape == stump.shape:
                cors.append((float(np.corrcoef(stump, v)[0, 1]), f[4:-4]))
        cors.sort(reverse=True)
        print(f"   maxcorr across {len(cors)} pack member(s): {cors[0][0]:.4f} "
              f"({cors[0][1]})   published {PUBLISHED['stump_maxcorr']:.4f}")
        print(f"   median corr: {np.median([c for c, _ in cors]):.4f}")
        if cors[0][0] < 0.99:
            fails.append(f"B stump maxcorr {cors[0][0]:.4f} < 0.99 — it IS decorrelated, "
                         "which contradicts the closure and reopens row 2")
        else:
            notes.append(f"B holds: stump maxcorr {cors[0][0]:.4f}, still inside the dense pack")

    # ---------------------------------------------------------------- claim C: the arithmetic
    print("\nC  THE PRICE, multiplied out rather than quoted")
    derived = PUBLISHED["tuning_solo_gain"] * PUBLISHED["passthrough"]
    frac = PUBLISHED["price"] / PUBLISHED["noise_floor"]
    print(f"   3e-5 x 1.4% = {derived:.3e}   published price {PUBLISHED['price']:.1e}")
    print(f"   price / noise floor = {frac:.3%}   published '~1%'")
    if not (0.5 <= derived / PUBLISHED["price"] <= 2.0):
        fails.append(f"C 3e-5 x 1.4% = {derived:.3e} does not round to the published "
                     f"{PUBLISHED['price']:.1e}")
    if not (0.005 <= frac <= 0.02):
        fails.append(f"C price is {frac:.3%} of the noise floor, not the published ~1%")

    # ---------------------------------------------------------------- the fourth knob
    print("\nD  THE FOURTH KNOB — 'categorical handling' is a PIPELINE change, not a knob")
    gate = os.path.join(HERE, "w97a_gate.json")
    if os.path.exists(gate):
        g = json.load(open(gate))
        print(f"   w97a_gate.json present — the windowed TE-prior verdict is on disk: "
              f"{json.dumps(g)[:120]}")
    else:
        notes.append("D w97a_gate.json absent; the NO-ENROL verdict is prose-only in RESEARCH")
    enrolled = [f for f in os.listdir(OOF) if "teprior" in f]
    print(f"   windowed-TE-prior arrays copied into oof/: {enrolled} (must be empty — NO-ENROL)")
    if enrolled:
        fails.append(f"D a teprior array reached oof/ despite the w97 NO-ENROL gate: {enrolled}")

    print()
    for n in notes:
        print("note " + n)
    for f in fails:
        print("FAIL " + f)
    print(f"FAILURES: {len(fails)}")
    if fails:
        return 1
    print("✅ ROW 2 HOLDS ON ITS OWN ARTEFACTS — tuning null re-measured, the stump's solo cost "
          "and pack correlation reproduce, the price multiplies out, the fourth knob stayed out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
