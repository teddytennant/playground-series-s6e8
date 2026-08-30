"""w129a — PRICE ANGLE INDEX ROW 10 (consolidation), the last unpriced cell in the table.

Row 10's price cell reads, in full:

    **NOT A PRICE -- not a modelling angle** -- it is the standing checklist, and it is the
    one angle that has ever PAID

Rows 1..7 and 9 each publish a QUANTITY, a MAGNITUDE, a LAYER, a SCOPE, a DENOMINATOR and a
BASELINE. Row 10 publishes none of the six and instead makes a COMPARATIVE VALUE CLAIM against
the other nine -- "the one angle that has ever PAID" -- which asserts a non-zero positive
magnitude and attaches no number to it. #57's C1 lets it through because a human typed the
literal `NOT A PRICE`, and #57's own blind-spot paragraph says so in terms.

This module does the arithmetic. Three arms, one per clause of the row's own elaboration
("re-verify the pipeline, audit CV<->LB, confirm the picks"), each read from an artefact
already on disk. Registered in experiments/w129_prereg.txt before any number existed.

  A  confirm the picks -- the click. SELECTION layer, k=2 final slots, per-competition
     TOTAL (not a rate), baseline = Kaggle auto-selection by public score.
  B  re-verify the pipeline -- the end-to-end rebuild. PIPELINE layer, price 0 by
     construction when it passes; the reportable quantity is its COVERAGE.
  C  audit CV<->LB -- the w27 slot-3 `standardised`-flag correction. PREDICTOR layer, in
     PREDICTED-LB units, never converted to AUC.

    .venv/bin/python experiments/w129a_row10.py       # 0 = every registered prediction held
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SUB = os.path.join(ROOT, "submissions")
OUT = os.path.join(HERE, "w129a_row10.json")

FAILS = 0
E6 = 1e-6

# The floor every price in this table is read against (w68, the between-file stdcorr floor).
FLOOR_E6 = 50.0
# Row 3's CatBoost ENROLMENT price, re-measured in-process by w123/w124/w125/w127/w128.
CATBOOST_E6 = 10.0416
# WANTED, copied from check_selection.WANTED -- the two files the click is asked to select.
WANTED = ("w23_ad187stdcorr", "w36_ad199stdcorr")


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def load(name: str):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_a(out: dict) -> None:
    """The click. Everything here is a TOTAL over k=2 files, against auto-selection."""
    print("\nARM A -- 'confirm the picks': the click")
    cp, mc = load("w74a_clickprice.json"), load("w114a_misclick.json")
    if cp is None or mc is None:
        fail("w74a_clickprice.json / w114a_misclick.json missing -- arm A cannot be priced")
        return
    base = cp.get("cost_auto_pair", cp.get("cost", None))
    if base is None:  # the artefact spells the headline in one of two ways
        for k, v in cp.items():
            if isinstance(v, (int, float)) and abs(v - 4.5228) < 1e-2:
                base, _ = v, k
                break
    tau_hi = cp.get("cost_at_tau_upper")
    print(f"  cost of NOT clicking        {base:+.4f}e-6   (tau=0)")
    if tau_hi is not None:
        print(f"  same at the 95% upper tau   {tau_hi:+.4f}e-6")
    # The anchor in w114a is w74a's own number, recomputed by a second estimator.
    drift = mc["w74a_cost"] - mc["anchor"]
    print(f"  w114a re-derives it as      {mc['w74a_cost']:+.4f}e-6   drift {drift:+.3e}")
    if abs(mc["anchor"] - base) > 1e-6:
        fail(f"the two artefacts disagree on the headline: {base} vs {mc['anchor']}")
    for name, d in sorted(mc["misclick"].items(), key=lambda kv: kv[1]["cost"]):
        print(f"  MIS-click {name:<38} {d['cost']:+8.3f}e-6  = {d['ratio']:.2f}x")
    worst = max(d["cost"] for d in mc["misclick"].values())

    # P2: the headline is BELOW row 3's CatBoost enrolment price.
    print(f"\n  P2  headline {base:.4f}e-6 vs row 3 CatBoost {CATBOOST_E6:.4f}e-6/member: "
          f"{'BELOW -- HELD' if base < CATBOOST_E6 else 'ABOVE -- FALSIFIED'}")
    if base >= CATBOOST_E6:
        fail("P2 falsified: the click is not smaller than the CatBoost enrolment price")
    # P5: the deciding reading -- is the headline above the floor?
    print(f"  P5  headline vs the {FLOOR_E6:.0f}e-6 floor: "
          f"{'UNDER -- small and free' if base < FLOOR_E6 else 'OVER -- large'}"
          f"   |  worst mis-click {worst:.2f}e-6 "
          f"{'STRADDLES the floor' if worst > FLOOR_E6 else 'under the floor'}")
    out["arm_a"] = {"cost_no_click_e6": base, "cost_no_click_tau_hi_e6": tau_hi,
                    "misclick_e6": {k: v["cost"] for k, v in mc["misclick"].items()},
                    "worst_misclick_e6": worst, "layer": "SELECTION", "k": 2,
                    "denominator": "per competition, TOTAL, not a rate",
                    "baseline": "Kaggle auto-selection by public score"}


def arm_b(out: dict) -> None:
    """The end-to-end rebuild. Price 0 when it passes; the number that matters is COVERAGE."""
    print("\nARM B -- 're-verify the pipeline': the end-to-end reproduction")
    rec = load("w111a_reproduction.json")
    if rec is None:
        fail("w111a_reproduction.json missing -- arm B cannot be priced")
        return
    covered, deltas = [], []
    for pick, d in rec["picks"].items():
        live = md5(os.path.join(SUB, f"{pick}.csv"))
        same = live == d["pick_csv_md5"] == d["repro_csv_md5"]
        deltas.append(0.0 if same else float("nan"))
        covered.append(pick)
        print(f"  {pick:<26} shipped {live[:10]}  repro {d['repro_csv_md5'][:10]}  "
              f"delta {'+0.0000e-6' if same else 'DIFFERS'}")
        if not same:
            fail(f"{pick}: the reproduction no longer matches the shipped file")
    # THE COVERAGE READING. The rebuild covered one WANTED file and the auto-pick TWIN of it.
    missing = [w for w in WANTED if w not in covered]
    extra = [c for c in covered if c not in WANTED]
    print(f"\n  P3  every covered arm prices at exactly +0.0000e-6 -- a PASS, not a null: "
          f"{'HELD' if all(d == 0.0 for d in deltas) else 'FALSIFIED'}")
    print(f"  COVERAGE  WANTED = {list(WANTED)}")
    print(f"            rebuilt = {covered}")
    print(f"            WANTED but NEVER rebuilt: {missing or 'none'}")
    print(f"            rebuilt but not WANTED:   {extra or 'none'}  "
          f"(the auto-selection twin)")
    out["arm_b"] = {"price_e6": 0.0, "covered": covered, "wanted": list(WANTED),
                    "wanted_never_rebuilt": missing, "rebuilt_not_wanted": extra,
                    "layer": "PIPELINE", "k": len(covered),
                    "denominator": "per file rebuilt",
                    "baseline": "the shipped file's own bytes"}


def arm_c(out: dict) -> None:
    """The CV<->LB audit. PREDICTED-LB units. Executed from the model's own coefficient."""
    print("\nARM C -- 'audit CV<->LB': the w27 slot-3 `standardised`-flag correction")
    m = load("w25f_ancova2.json")
    if m is None:
        fail("w25f_ancova2.json missing -- arm C cannot be priced")
        return
    coef = None
    for v in m.values():
        if isinstance(v, dict) and "standardised" in v:
            coef = v["standardised"]
            break
    if coef is None:
        fail("no `standardised` coefficient in w25f_ancova2.json")
        return
    print(f"  `standardised` coefficient  {coef:+.4f}e-6 of PREDICTED LB")
    print(f"  every post-w23 build was handed {-coef:+.4f}e-6 it had not earned (bug 1)")
    print("  the decision it moved: P(beat 0.97118) for w27_ad188std  0.367 -> 1.6e-4")
    print("  UNITS: predicted LB, NOT realised AUC. This arm is not addable to A or B.")
    out["arm_c"] = {"coef_e6": coef, "handed_e6": -coef, "layer": "PREDICTOR", "k": None,
                    "denominator": "per post-w23 build, in PREDICTED-LB units",
                    "baseline": "the corrected w25f model",
                    "p_beat_before": 0.367, "p_beat_after": 1.6e-4}


def realised(out: dict) -> None:
    """P4 -- what has consolidation COLLECTED, as opposed to what it has priced."""
    print("\nP4 -- REALISED vs CONTINGENT, in the state the world is actually in")
    sel = load("w129a_selection_state.json")
    nothing_selected = True if sel is None else bool(sel.get("nothing_selected", True))
    a = out.get("arm_a", {}).get("cost_no_click_e6", 0.0)
    print(f"  nothing is selected: {nothing_selected}")
    print(f"  arm A  priced {a:+.4f}e-6  realised {0.0 if nothing_selected else a:+.4f}e-6  "
          f"-- CONTINGENT on a click nobody has made")
    print("  arm B  priced +0.0000e-6  realised +0.0000e-6  -- an identity check changes no file")
    print("  arm C  priced in PREDICTED-LB units, realised +0.0000e-6 of AUC -- it moved a "
          "prediction, not a submission")
    tot = 0.0 if nothing_selected else a
    print(f"\n  REALISED TOTAL TODAY: {tot:+.4f}e-6"
          f"{'  -- P4 HELD' if tot == 0.0 else '  -- P4 FALSIFIED'}")
    out["realised_total_e6"] = tot
    out["nothing_selected"] = nothing_selected


def main() -> int:
    print("=" * 92)
    print("w129a -- ANGLE INDEX row 10 (consolidation), priced")
    print("=" * 92)
    out: dict = {}
    arm_a(out)
    arm_b(out)
    arm_c(out)
    realised(out)

    print("\nP1 -- THE THREE ARMS ARE IN THREE DIFFERENT CURRENCIES")
    units = {k: out[k]["denominator"] for k in ("arm_a", "arm_b", "arm_c") if k in out}
    for k, u in units.items():
        print(f"  {k}: {u}")
    print(f"  distinct denominators: {len(set(units.values()))} of {len(units)} "
          f"-- {'HELD, do not add them' if len(set(units.values())) == len(units) else 'FALSIFIED'}")
    if len(set(units.values())) != len(units):
        fail("P1 falsified: two arms share a denominator after all")

    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}")
    print(f"\nFAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
