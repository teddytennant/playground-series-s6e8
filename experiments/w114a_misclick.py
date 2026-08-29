"""w114 (2026-08-29) — price the MIS-CLICK, not just the missing click.

Seven pricings (w15i .. w45, w57, w62, w74) all answer one question: what does it cost if
Teddy never clicks and Kaggle auto-selects on public score? The live answer is +4.5228e-6
(`w74a_clickprice.json`). Every one of them assumes that IF the click happens it lands on
`check_selection.WANTED`.

`check_selection.py` is the instrument that tells a human what to click, and in its
NOTHING-IS-SELECTED branch it prints, inside a green tick box, three lines below the correct
`Wanted:` line:

    ## ✅ RESOLVED 2026-08-17 (w21 slot 7). WANTED HAS MOVED.  ##
    WANTED is now {w21_ad187corr.csv, w20_ad187_h3.csv}.

That was true on 08-17 and has been false since 08-19 (w28) and again since 08-21 (w36b).
Both named files are SENT, so Kaggle will happily offer them: the wrong click is reachable.
A second block further down annotates `w16i_schemeavg` and `blend159av_h3` as "current
WANTED", which has been false since 08-17.

This module prices those two reachable wrong pairs on the SAME estimator as w74a, so the
mis-click hazard can be compared against the +4.5228e-6 the click chain exists to protect.
It prices; it does not edit anything and it does not move WANTED.

GATE R  reproduce `w74a_clickprice.json`'s auto-pair cost on w74a's OWN recorded board.
C1      the price of WANTED against itself is 0.
C2      CV recomputed from every OOF vector matches w74a's recorded CV.
C3      a pair that DOMINATES wanted on CV must price NEGATIVE — the estimator is not
        sign-locked to "wanted always wins".
"""
from __future__ import annotations

import json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import TARGET, load_raw                        # noqa: E402
from w74a_clickprice import build, _load, CORR_CONS, WANTED  # noqa: E402
import check_selection as cs                                # noqa: E402 — SHARED client

COMP = "playground-series-s6e8"
U = 1e-6
FAILURES = 0

# The two wrong pairs `check_selection.py` can currently talk a reader into, named exactly as
# its own output names them. Stems, not filenames.
MISCLICK = {
    "the ✅ RESOLVED box's literal instruction": ("w21_ad187corr", "w20_ad187_h3"),
    "the slot-6 notice's 'current WANTED' annotation": ("w16i_schemeavg", "blend159av_h3"),
}
# C3's control pair: w40_ad211stdcorr outranks w23_ad187stdcorr on CV by +22.4e-6, so
# {pick, ad211} must price NEGATIVE. ⛔ This is a CONTROL ON THE ESTIMATOR, not a proposal:
# w40_ad211 is w40d-INELIGIBLE for WANTED (check_selection.WANTED_INELIGIBLE) and WANTED
# does not move.
CONTROL_PAIR = ("w36_ad199stdcorr", "w40_ad211stdcorr")


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def live_lb() -> dict:
    """publicScore by stem, read live through check_selection's OWN client snippet.

    ⚠ SHARED, not re-derived. The first cut of this function wrote its own `import kaggle`
    snippet and died on ModuleNotFoundError: the API client lives in a separate uv tool
    interpreter (`check_selection.KAGGLE_PY`), not in `.venv`, and its page size is 200
    because this account passed the CLI's default 50 on 08-17.
    """
    p = subprocess.run([cs.KAGGLE_PY, "-c", cs.SNIPPET], capture_output=True, text=True)
    if p.returncode != 0:
        print("live board read FAILED:\n" + p.stderr.strip())
        sys.exit(3)
    out = {}
    for _ref, fn, sc in json.loads(p.stdout)["successful"]:
        if sc is not None:
            out[fn[:-4] if fn.endswith(".csv") else fn] = float(sc)
    return out


def main() -> int:
    global FAILURES
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    ref = json.load(open(os.path.join(HERE, "w74a_clickprice.json")))

    # ------------------------------------------------------------------------------ GATE R
    print("=" * 92)
    print("GATE R — reproduce w74a_clickprice.json's auto-pair cost on w74a's OWN board")
    print("=" * 92)
    rNAMES = sorted(ref["cv"])
    rLB = {k: float(v) for k, v in ref["lb"].items()}
    rV, rcv = _load(rNAMES, y)
    for k in rNAMES:                                                            # C2
        if abs(rcv[k] - float(ref["cv"][k])) > 5e-10:
            fail(f"C2: CV of {k} does not reproduce from its OOF vector")
    rE = build(rNAMES, rLB, rcv, rV, y, beta)
    rmu, rcov, rrsd, _ = rE["solve"](0.0, CORR_CONS)
    got = rE["price"](rmu, rcov, rrsd, list(ref["auto_pair"]))
    rec = float(ref["cost_auto_pair"])
    print(f"  {len(rNAMES)} files · auto pair {'+'.join(ref['auto_pair'])}")
    print(f"  cost of NOT clicking  {got:+.9f}e-6   recorded {rec:+.9f}e-6   "
          f"|d| {abs(got - rec):.3e}   (bar 1e-9)")
    if abs(got - rec) > 1e-9:
        fail(f"GATE R: {got:+.12f} != recorded {rec:+.12f}")
    self_price = rE["price"](rmu, rcov, rrsd, list(WANTED))                      # C1
    print(f"  C1  price of WANTED against itself {self_price:+.3e}e-6   (bar 1e-9)")
    if abs(self_price) > 1e-9:
        fail(f"C1: self-price is {self_price:+.3e}, not 0")
    if FAILURES:
        print("\n⛔ GATE R FAILED — the estimator is not w74a's. REFUSING to price anything.")
        return 1
    print("  ✅ GATE R PASSED — this is w74a's estimator, unmodified.")

    # ---------------------------------------------------------------- the augmented board
    # Pricing a pair requires its OOF on the board, and adding names refits the GLS common
    # gap, so every price moves a little. The drift on the ANCHOR is printed rather than
    # assumed away: a re-price that cannot reproduce its own anchor is not comparable.
    extra = sorted({s for p in MISCLICK.values() for s in p} | {CONTROL_PAIR[1]})
    lb = live_lb()
    missing = [k for k in extra if k not in lb]
    if missing:
        print(f"\n⛔ not on the live board, cannot price: {missing}")
        return 1
    NAMES = sorted(set(rNAMES) | set(extra))
    LB = dict(rLB)
    for k in extra:
        LB[k] = lb[k]
    V, cv = _load(NAMES, y)
    E = build(NAMES, LB, cv, V, y, beta)
    mu, cov, rsd, _ = E["solve"](0.0, CORR_CONS)
    anchor = E["price"](mu, cov, rsd, list(ref["auto_pair"]))
    print("\n" + "=" * 92)
    print(f"THE AUGMENTED BOARD — {len(NAMES)} files ({len(rNAMES)} + {len(extra)})")
    print("=" * 92)
    print(f"  ANCHOR   cost of not clicking, auto pair   {anchor:+.4f}e-6"
          f"   vs {rec:+.4f}e-6 on w74a's board   drift {anchor - rec:+.4f}e-6")

    rows = []
    print("\n  cost of clicking the WRONG pair  =  E[max over WANTED] - E[max over that pair]")
    for label, pair in MISCLICK.items():
        c = E["price"](mu, cov, rsd, list(pair))
        dcv = [(k, (cv[k] - cv[WANTED[1]]) / U) for k in pair]
        rows.append((label, pair, c))
        print(f"\n  {label}")
        print(f"    click {' + '.join(pair)}")
        for k, d in dcv:
            print(f"      {k:<22} CV {cv[k]:.10f}  LB {LB[k]:.5f}   {d:+7.2f}e-6 vs the pick")
        print(f"    ⚠ COST OF THIS MIS-CLICK  {c:+.4f}e-6"
              f"    = {c / anchor:.2f}x the cost of not clicking at all")

    c3 = E["price"](mu, cov, rsd, list(CONTROL_PAIR))                            # C3
    print(f"\n  C3  control pair {'+'.join(CONTROL_PAIR)} (CV-dominant) prices {c3:+.4f}e-6"
          "   — must be NEGATIVE")
    if not (c3 < 0):
        fail(f"C3: a CV-dominant pair priced {c3:+.4f}e-6, not negative — the estimator "
             "is sign-locked to 'wanted always wins' and no price above can be trusted")

    out = dict(gate_r=True, anchor=anchor, w74a_cost=rec, drift=anchor - rec,
               self_price=self_price, c3=c3, c3_pair=list(CONTROL_PAIR),
               n_names=len(NAMES),
               misclick={"+".join(p): dict(label=lab, cost=c, ratio=c / anchor)
                         for lab, p, c in rows},
               cv={k: cv[k] for k in NAMES}, lb={k: LB[k] for k in NAMES},
               failures=FAILURES)
    json.dump(out, open(os.path.join(HERE, "w114a_misclick.json"), "w"), indent=1)
    print(f"\nFAILURES {FAILURES}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
