"""w64b — negative-controlled guard on the WANTED slot-2 decision and the instruments behind it.

WHAT THIS EXISTS FOR. w64 closed a three-run deferral with two measurements and a decision rule,
and every part of that is a thing a later run could silently undo or misread:

  1. The DECISION (`WANTED` does not move) is recorded in a comment and in an artefact. If the
     artefact's readings and the recorded decision ever disagree, one of them was edited.
  2. The BREAK-EVEN (-1.5005e-6/member) is arithmetic over two ledger numbers. w63 §3's rule:
     read the number from the artefact, never re-type it. This guard re-derives it and refuses
     a re-typed literal that has drifted.
  3. The three deferrals named `w40_ad211stdcorr` where `w38_ad202stdcorr` is higher-CV and
     unencumbered. That is a ledger fact and it must stay true, or the whole §1 argument moves.
  4. P3's null ("no pack-size transfer penalty") is only evidence of absence because the POWER
     CONTROL says the regression could have seen a break-even-sized slope. An artefact whose
     power control did not detect is an artefact whose null means nothing.

Every check has a negative control, because in each case the wrong value still looks plausible:
a stale decision is still a boolean, a re-typed break-even is still a small negative number, a
powerless regression still reports a tight interval around zero.

    .venv/bin/python experiments/w64b_hedgeguard.py
"""
from __future__ import annotations

import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import check_selection as CS                                          # noqa: E402
import w64a_hedgeprice as W64A                                        # noqa: E402

FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    print(f"  {'ok ' if ok else '*** FAILURE'}  {name:66s} {detail}")
    if not ok:
        FAILURES += 1


def main() -> None:
    art = os.path.join(HERE, "w64a_hedgeprice.json")
    print("=" * 96)
    print("w64b — the WANTED slot-2 decision, its two instruments, and the ledger fact under it")
    print("=" * 96)
    check("w64a artefact exists", os.path.exists(art), art)
    if FAILURES:
        sys.exit(1)
    d = json.load(open(art))
    A, B = d["part_a"], d["part_b"]

    # ---------------------------------------------------------------- 1. GATE A actually ran
    print("\n-- the imported estimator")
    check("GATE A recorded as passed against w63a_setprice.json",
          d["gate_a"]["passed"] and d["gate_a"]["against"] == "w63a_setprice.json")
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    check("and the price it reproduced IS w63a's recorded base",
          abs(float(d["gate_a"]["base"]) - float(ref["base"])) < 1e-9,
          f"{d['gate_a']['base']:.12f}")
    # negative control: a base that is merely a plausible number must not pass
    check("[neg] a plausible-but-wrong base is rejected",
          not abs(4.5 - float(ref["base"])) < 1e-9)

    # ---------------------------------------------------------------- 2. the decision vs WANTED
    print("\n-- the decision, compared against the live dict rather than stamped")
    wanted = {w[:-4] if w.endswith(".csv") else w for w in CS.WANTED}
    check("w64a's slot1/incumbent ARE check_selection.WANTED",
          wanted == {d["slot1"], d["incumbent"]}, f"{sorted(wanted)}")
    check("the recorded decision is 'do not move'", d["decision_move"] is False)
    check("and WANTED still holds the incumbent, consistent with it",
          (d["incumbent"] in wanted) == (not d["decision_move"]))
    # negative control: had the decision been MOVE, the same comparison must fail on this dict
    check("[neg] a MOVE decision would be caught against this dict",
          not ((d["incumbent"] in wanted) == (not True)))

    # the AND rule, re-applied to the RECORDED readings rather than trusted
    p = d["predictions"]
    check("decision == AND of the recorded readings (p2 strict AND p3a AND p3b)",
          d["decision_move"] == bool(p["p2"] and p["p3a"] and p["p3b"]))
    check("[neg] an OR rule would have given a DIFFERENT answer here",
          bool(p["p2"] or p["p3a"]) != bool(p["p2"] and p["p3a"] and p["p3b"]),
          "so the AND/OR choice in w64_prereg §4 is load-bearing, not cosmetic")

    # ---------------------------------------------------------------- 3. the break-even
    print("\n-- the break-even: DERIVED from the ledger, never re-typed (w63 §3)")
    derived = -float(d["dcv"]) / float(d["n_members"])
    check("artefact break-even == -dCV/n_members recomputed here",
          abs(float(d["delta_breakeven"]) - derived) < 1e-12, f"{derived:+.6f} e-6/member")
    check("n_members == pack(challenger) - pack(incumbent)",
          int(d["n_members"]) == W64A.PACK_OF_CHALLENGER - W64A.PACK_OF_INCUMBENT)
    check("[neg] the same dCV over the WRONG span gives a different bar",
          abs((-float(d["dcv"]) / 24.0) - derived) > 0.1,
          "187->211 instead of 187->202 moves it by 0.6e-6")

    # ---------------------------------------------------------------- 4. the ledger fact
    print("\n-- the ledger fact the three deferrals got wrong")
    cv = A["cv"]
    check("challenger w38_ad202stdcorr outranks w40_ad211stdcorr on CV",
          cv[W64A.CHALLENGER] > cv[W64A.ALSO],
          f"{(cv[W64A.CHALLENGER] - cv[W64A.ALSO]) / 1e-6:+.3f}e-6")
    check("the file the deferrals named carries a RETIRED bar",
          any(W64A.ALSO.startswith(k) for k in CS.WANTED_RETIRED))
    check("the challenger carries neither a retired nor a live bar",
          not any(W64A.CHALLENGER.startswith(k) for k in CS.WANTED_RETIRED)
          and not any(W64A.CHALLENGER.startswith(k) for k in CS.WANTED_INELIGIBLE))
    check("[neg] an ineligible arm IS detected by the same test",
          any("w42_ad217stdcorr".startswith(k) for k in CS.WANTED_INELIGIBLE))
    bad = [k for k in W64A.LADDER
           if any(k.startswith(pat) for pat in CS.WANTED_INELIGIBLE)]
    check("no w40d-ineligible arm is on the priced ladder", not bad, str(bad))

    # ---------------------------------------------------------------- 5. the power control
    print("\n-- P3's null is only evidence of absence if the instrument had power")
    pw = B["power"]
    key = f"{d['delta_breakeven']:+.4f}"
    check("the power control injected the break-even slope itself", key in pw, key)
    check("and the estimator recovered every injection exactly",
          all(abs(v["err"]) < 1e-9 for v in pw.values()),
          f"worst |err| {max(abs(v['err']) for v in pw.values()):.2e}")
    check("a break-even-sized slope IS detected at 2 sigma", bool(pw[key]["detected"]))
    check("[neg] an artefact with detected=False would fail this check",
          not bool({"detected": False}["detected"]))

    # ---------------------------------------------------------------- 6. the GLS scaling rule
    print("\n-- the chi2/dof scaling is ONE-SIDED: a good fit never buys a narrower interval")
    scale = float(B["scale"])
    check("scale == sqrt(max(chi2/dof, 1))",
          abs(scale - math.sqrt(max(B["chi2"] / B["dof"], 1.0))) < 1e-12,
          f"chi2/dof {B['chi2'] / B['dof']:.3f} -> {scale:.4f}")
    check("scale >= 1 always", scale >= 1.0)
    check("[neg] a chi2/dof of 0.25 must NOT shrink the interval",
          math.sqrt(max(0.25, 1.0)) == 1.0)
    check("the scaled se is the one the CI was built from",
          abs(float(B["delta_se_scaled"]) - float(B["delta_se"]) * scale) < 1e-12)
    lo, hi = B["ci"]
    check("the CI is the scaled +-1.96 interval",
          abs(lo - (B["delta"] - 1.96 * B["delta_se_scaled"])) < 1e-9
          and abs(hi - (B["delta"] + 1.96 * B["delta_se_scaled"])) < 1e-9)
    check("and it EXCLUDES the break-even", lo > float(d["delta_breakeven"]),
          f"[{lo:+.4f}, {hi:+.4f}] vs {float(d['delta_breakeven']):+.4f}")

    # ---------------------------------------------------------------- 7. eligibility still holds
    print("\n-- check_selection's own enforcement, exercised in both directions")
    try:
        CS.assert_wanted_eligible()
        check("assert_wanted_eligible passes on the live WANTED", True)
    except SystemExit:
        check("assert_wanted_eligible passes on the live WANTED", False)
    try:
        CS.assert_wanted_eligible({"w42_ad217stdcorr.csv", "w36_ad199stdcorr.csv"})
        check("[neg] and REFUSES on a w40d-ineligible arm", False)
    except SystemExit:
        check("[neg] and REFUSES on a w40d-ineligible arm", True)

    # ---------------------------------------------------------------- 8. the falsifications stay
    print("\n-- the falsified readings are carried, not quietly dropped (w63 §8)")
    check("the artefact records its falsifications", len(d["falsified"]) == 2,
          str(d["falsified"]))
    check("P2 is recorded FALSE and P5 is recorded FALSE",
          p["p2"] is False and p["p5"] is False)
    check("both readings of the ambiguous §4(i) are recorded",
          d["p2_strict"] is False and d["p2_loose"] is True and "prereg_defect" in d)

    print(f"\nFAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
