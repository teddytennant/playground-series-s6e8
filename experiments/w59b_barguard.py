"""w59b — the regression test for the w59 hijack CV bar. It EXERCISES the bar, not just its name.

w56b's idiom: a guard that is only asserted to exist is a guard that gets edited out. This
FIRES the bar on synthetic rows either side of it and checks the fail-safe, so a future run
that quietly loosens it, hard-codes it, or points it at a stale artefact fails here.

What it checks:
  1. `hijack_cv_bar()` reads the bar LIVE from w59a_hijackprice.json (not a constant in source).
  2. That bar equals w59a's `cv_bar_new`, which is `H_binding` below the pick, and `H_binding`
     is the STRICTER of the unconditional and conditional break-evens.
  3. It is STRICTER than the superseded `wanted_cv_bar()` — the whole point of w59. If a future
     run makes the bar looser than the WANTED bar again, that is the w58 error returning.
  4. `above_tier_reason` FIRES on a row just below the bar and PASSES a row just above it.
  5. FAIL-SAFE: with the artefact unreadable the bar is None and `above_tier_reason` BLOCKS.
     w55's lesson — a row the rule cannot be evaluated on is not covered by it.
  6. The artefact was produced against the LIVE tier (its GATE T passed and its tier matches
     w57a's), so the bar is not quoted off a dead board.

    .venv/bin/python experiments/w59b_barguard.py
"""
from __future__ import annotations

import json, os, sys
from types import SimpleNamespace

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import w26g_send as SND                                                     # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else '⛔ '} {name:52s} {detail}")
    if not ok:
        FAIL.append(name)


def main() -> None:
    art = json.load(open(os.path.join(HERE, "w59a_hijackprice.json")))
    bar = SND.hijack_cv_bar()
    print("=== w59b: the hijack CV bar is wired, live, strict, and fails safe ===")

    check("bar is read and is a plausible AUC", bar is not None and 0.9 < bar < 1.0,
          f"{bar!r}")
    check("bar == w59a cv_bar_new (live artefact, not a source constant)",
          bar is not None and abs(bar - float(art["cv_bar_new"])) < 1e-15,
          f"{bar:.10f}" if bar else "")
    Hb, Hu, Hc = art["H_binding"], art["H_uncond"], art["H_cond_predsd"]
    check("H_binding is the STRICTER of the two break-evens",
          abs(Hb - min(Hu, Hc)) < 1e-12, f"binding {Hb:.2f} = min({Hu:.2f}, {Hc:.2f})")
    check("bar == pick CV - H_binding",
          bar is not None and abs(bar - (art["pick_cv"] - Hb * 1e-6)) < 1e-12,
          f"pick {art['pick_cv']:.10f} - {Hb:.2f}e-6")

    old = SND.wanted_cv_bar()
    check("bar is STRICTER than the superseded WANTED bar (the w59 finding)",
          bar is not None and old is not None and bar > old,
          f"{bar:.10f} > {old:.10f}  (+{(bar-old)/1e-6:.2f}e-6)")

    check("artefact was produced on the live tier (its GATE T passed)",
          art.get("gate_t") == "PASS", str(art.get("gate_t")))
    prev = json.load(open(os.path.join(HERE, "w57a_tierprice2.json")))["tiers"]
    check("artefact tier == w57a tier",
          sorted(art["tiers"]["slot1"]) == sorted(prev["slot1"]),
          f"{len(art['tiers']['slot1'])} files @ {art['tiers']['slot1_public']}")

    # ---- 4. FIRE IT. A row is a namespace shaped like the itertuples row the sender passes.
    def row(cv, stem="w99_probe", fam="h3"):
        return SimpleNamespace(file=f"{stem}.csv", fam=fam, cv=cv)

    below = SND.above_tier_reason(row(bar - 1e-9), bar)
    above = SND.above_tier_reason(row(bar + 1e-9), bar)
    check("FIRES on a row 1e-9 BELOW the bar", bool(below), (below or "")[:58])
    check("PASSES a row 1e-9 ABOVE the bar", above is None, str(above)[:58])

    # the real rows this changed, by name, so the test breaks if the bar drifts back
    for stem, cv in (("w27_ad188stdcorr", 0.9701168076), ("w36_ad197stdcorr", 0.9701286299)):
        check(f"blocks {stem} (passed the OLD bar)",
              bool(SND.above_tier_reason(row(cv, stem), bar)), f"cv {cv:.10f}")
    # ⚠ RE-POINTED w60, NOT SOFTENED — and the reason lives here, at the check. This row used to
    # be `w40_ad211std_h3` (cv 0.9701331846), asserting the bar does not OVER-block a file that
    # clears it. w60 added `w40_ad211` to `check_selection.WANTED_INELIGIBLE` — w40d_prereg
    # registered ARM 211 as never WANTED-eligible before the arm existed and w56 left it out of
    # the dict it built to enforce that — so ad211 is now refused ABOVE the tier on the
    # eligibility half of the two-part test, before the CV bar is ever consulted. The property
    # under test is unchanged ("a file that clears the bar is still admitted"); only the subject
    # moved, to an es-CLEARED pack. ⛔ Do not point it back at ad211 to make this pass.
    for stem, cv in (("w38_ad202std_h3", 0.9701330214),):
        check(f"still admits {stem} (eligible, over the bar)",
              SND.above_tier_reason(row(cv, stem), bar) is None, f"cv {cv:.10f}")
    # ⚠ RE-POINTED AGAIN w61, NOT SOFTENED, and the reason lives here. w60 pointed this at
    # `w40_ad211std_h3` to assert ad211 was refused on ELIGIBILITY rather than on CV. w61 then
    # ran the four-base matched-control test w60_prereg registered — h3 +0.163, ens4 +0.381,
    # rescale +0.946, rankraw -2.353 e-6, all inside the ±4e-6 rebuild floor — and RETIRED that
    # bar (check_selection.WANTED_RETIRED). Keeping the old assertion would assert the
    # retirement did not happen. The property under test is unchanged — "a still-barred arm is
    # refused on eligibility BEFORE the CV bar is consulted" — and it now runs on the harder
    # subject: `w42_ad217stdcorr` clears the CV bar by 49.4e-6, more than any file on disk, so
    # this is what breaks first if the two halves of the test ever swap order. ⛔ Do not point
    # it back at ad211.
    for stem, cv in (("w42_ad217stdcorr", 0.9701788311219954),):
        blocked = SND.above_tier_reason(row(cv, stem), bar)
        check(f"blocks {stem} on w40d-ineligibility, not on CV",
              bool(blocked) and "ineligible" in blocked, f"cv {cv:.10f} (clears the bar by 49e-6)")
    # ...and the retired arm is no longer refused on eligibility. Paired with the line above so
    # a future edit cannot satisfy one by breaking the other.
    for stem, cv in (("w40_ad211std_h3", 0.9701331846),):
        why = SND.above_tier_reason(row(cv, stem), bar)
        check(f"and NO LONGER blocks {stem} on eligibility (w61 retirement)",
              not (why and "ineligible" in why), f"cv {cv:.10f} -> {str(why)[:40]}")

    # ---- 5. FAIL-SAFE. Point the module at a missing artefact; the bar must vanish and BLOCK.
    keep = SND.HIJACKPRICE
    try:
        SND.HIJACKPRICE = os.path.join(HERE, "w59a_hijackprice.MISSING.json")
        gone = SND.hijack_cv_bar()
        check("bar is None when the artefact is unreadable", gone is None, repr(gone))
        check("and above_tier_reason then BLOCKS rather than waving through",
              bool(SND.above_tier_reason(row(0.99), gone)),
              (SND.above_tier_reason(row(0.99), gone) or "")[:52])
    finally:
        SND.HIJACKPRICE = keep

    # ---- 6. a member row is still refused on family alone, before any CV comparison
    check("fam=member still refused before the CV test",
          "cross-fitted" in (SND.above_tier_reason(row(0.99, "w48_cal_x", "member"), bar) or ""))

    print(f"\nFAILURES: {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print("w59b: the measured bar is wired, strict, and fails safe.")


if __name__ == "__main__":
    main()
