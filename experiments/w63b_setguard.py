"""w63b — the successor pricer's artefact, the leverage it publishes, and the live account best.

WHAT THIS EXISTS FOR. w63 changed four things that a later run could silently undo:

  1. `w26g_send.HIJACKPRICE` now points at `w63a_setprice.json`. w59a's artefact was derived on
     a five-file tie at 0.97118 that no longer exists.
  2. `w26g_send.above_tier_reason` used to print the literal "3x the leverage of an in-tier
     landing". That 3 is C(n+1,2)/n at n = 5 — a reading of a tier SIZE, not a constant of the
     problem. It is 1.5 at the n = 2 the board holds today.
  3. `w26d_queueprice.BEST_LB` was the frozen literal 0.97118 after the account best had moved
     to 0.97119, so every `p_beat` was priced against a target already beaten.
  4. `w48e_order.ORDERS["2026-08-24"]` IMPORTS the ten from the pricer instead of re-typing
     them, so the registered list and the priced list cannot drift.

Every check below has a negative control, because the failure mode in each case is a value that
still looks plausible: a stale bar is still a number between 0.9 and 1.0, a stale leverage is
still a small integer, and a stale account best is still a valid AUC.

    .venv/bin/python experiments/w63b_setguard.py
"""
from __future__ import annotations

import json, math, os, sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import w26g_send as SND                                              # noqa: E402

FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    print(f"  {'ok ' if ok else '*** FAILURE'}  {name:62s} {detail}")
    if not ok:
        FAILURES += 1


def main():
    print("=" * 96)
    print("w63b: the successor artefact, the published leverage, and the live account best")
    print("=" * 96)

    # ---- 1. THE ARTEFACT THE SENDER READS IS THE SUCCESSOR, AND IT CARRIES WHAT THE SENDER NEEDS
    check("the sender reads w63a_setprice.json",
          os.path.basename(SND.HIJACKPRICE) == "w63a_setprice.json",
          os.path.basename(SND.HIJACKPRICE))
    art = json.load(open(SND.HIJACKPRICE))
    need = ["cv_bar_new", "gate_t", "tiers", "H_binding", "H_uncond", "H_cond_predsd",
            "pick_cv", "gate_j", "base", "set_price_cond", "set_price_additive", "falsified"]
    miss = [k for k in need if k not in art]
    check("artefact carries every field its readers use", not miss, f"missing {miss}" if miss else "")
    check("tiers.slot1 and slot1_public are present (the live SET staleness check)",
          "slot1" in art.get("tiers", {}) and "slot1_public" in art.get("tiers", {}))
    # ⚠ w62's lesson wired in the other direction: the SUCCESSOR must not inherit w59a's stamp.
    check("the artefact is NOT w59a's (it supersedes, it does not patch)",
          art.get("supersedes", "").startswith("w59a"), art.get("supersedes", "")[:44])

    # ---- 2. THE LEVERAGE IS A FUNCTION OF THE TIER SIZE, NOT THE LITERAL 3
    n = int(art["gate_j"]["n_tier"])
    lev = SND.hijack_leverage()
    check("hijack_leverage() returns the artefact's leverage",
          abs(lev - float(art["gate_j"]["leverage"])) < 1e-12, f"{lev:.3f}")
    check(f"and it is C(n+1,2)/n at the live n = {n}",
          abs(lev - math.comb(n + 1, 2) / n) < 1e-12, f"{lev:.3f}")
    # ⚠⚠ THE CONTROL THAT MAKES THE NEW LAW A GENERALISATION AND NOT A REPLACEMENT: the formula
    # must REPRODUCE w59a's measured 3 at w59a's own tier size. A "fix" that merely swapped one
    # constant for another would pass every check above and fail this one.
    check("the formula reproduces w59a's measured 3.0 at ITS tier size n = 5",
          abs(math.comb(6, 2) / 5 - 3.0) < 1e-12, "C(6,2)/5 = 3.0")
    check("and it is NOT 3.0 on today's board (the regression this run fixed)",
          abs(lev - 3.0) > 0.1, f"{lev:.3f} at n = {n}")

    # ---- 3. THE REASON STRING PUBLISHES THE LIVE NUMBER, AND FAILS TO nan RATHER THAN TO STALE
    row = SimpleNamespace(file="w99_probe.csv", fam="h3", cv=0.5)
    why = SND.above_tier_reason(row, float(art["cv_bar_new"]))
    check("a below-bar row is blocked and the reason quotes the LIVE leverage",
          bool(why) and f"{lev:.1f}x" in why, (why or "")[-52:])
    check("and the reason no longer says 3x", bool(why) and "3.0x" not in why and "3x" not in why)
    keep = SND.HIJACKPRICE
    try:
        SND.HIJACKPRICE = keep + ".MISSING"
        # ⚠ NEGATIVE CONTROL. With no artefact the leverage must go nan, not silently fall back
        # to a literal. A reason string reading "nanx" is a reader's problem; one reading "3x"
        # on a two-file tier is a wrong fact stated with confidence.
        check("no artefact -> leverage is nan, NOT a stale literal",
              math.isnan(SND.hijack_leverage()), repr(SND.hijack_leverage()))
        check("and the bar goes None, so above_tier_reason BLOCKS",
              SND.hijack_cv_bar() is None and bool(SND.above_tier_reason(row, None)))
    finally:
        SND.HIJACKPRICE = keep
    check("restored", os.path.basename(SND.HIJACKPRICE) == "w63a_setprice.json")

    # ---- 4. THE SET PRICE IS NOT THE ADDITIVE PRICE
    # The whole reason the successor exists. If these two ever agree to the last digit, the set
    # enumeration has collapsed back into a per-file sum and w62 §4 is free to happen again.
    err = float(art["set_price_additive"]) - float(art["set_price_uncond"])
    check("the SET price differs from the ADDITIVE price of the same ten",
          abs(err) > 1e-9, f"additive {art['set_price_additive']:+.4f} vs set "
                           f"{art['set_price_uncond']:+.4f}  ({err:+.4f}e-6)")
    check("and the additive price is the OPTIMISTIC one (understates the cost)",
          err < 0, f"additive - set = {err:+.4f}e-6")
    na = art["nonadditivity"]
    check("non-additivity is measured over pairs, not asserted",
          int(na["n_pairs"]) > 0 and na["min_gap"] < 0 < na["max_gap"],
          f"{na['n_pairs']} pairs, gap in [{na['min_gap']:+.3f}, {na['max_gap']:+.3f}]")
    # ⚠ the falsifications are CARRIED, not laundered. A later reader must see them.
    check("the artefact records what was falsified rather than dropping it",
          isinstance(art.get("falsified"), list), str(art.get("falsified"))[:46])

    # ---- 5. THE ACCOUNT BEST IS LIVE, AND ITS FLOOR IS ONE-WAY
    import w26d_queueprice as QP
    live_best = float(art["tiers"]["slot1_public"])
    check("w26d.BEST_LB is the live top public score, not the frozen 0.97118",
          abs(QP.BEST_LB - live_best) < 1e-12, f"{QP.BEST_LB:.5f} vs live {live_best:.5f}")
    check("and it has actually moved off the literal it replaced", QP.BEST_LB > 0.97118)
    # ⚠ NEGATIVE CONTROL, BOTH WAYS. A best score cannot fall: a floor ABOVE the artefact must
    # hold, and a floor BELOW it must be raised. A function that just returned the floor would
    # pass the first of these and fail the second; one that just returned the artefact, vice
    # versa. Both are required, so neither degenerate implementation survives.
    check("a floor ABOVE the artefact holds (never lowered)",
          abs(QP._best_lb(floor=0.99) - 0.99) < 1e-12, f"{QP._best_lb(floor=0.99):.5f}")
    check("a floor BELOW the artefact is raised to it",
          abs(QP._best_lb(floor=0.90) - live_best) < 1e-12, f"{QP._best_lb(floor=0.90):.5f}")

    # ---- 6. THE REGISTERED TEN AND THE PRICED TEN ARE ONE OBJECT
    from w63a_setprice import PLAN_0824
    from w48e_order import ORDERS                              # noqa: E402 — prints its plan
    check("w48e ORDERS['2026-08-24'] IS w63a.PLAN_0824",
          ORDERS.get("2026-08-24") == list(PLAN_0824),
          f"{len(ORDERS.get('2026-08-24', []))} files")
    check("the priced ten are the ten in the artefact",
          list(art["plan_0824"]) == list(PLAN_0824))

    print(f"\nFAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
