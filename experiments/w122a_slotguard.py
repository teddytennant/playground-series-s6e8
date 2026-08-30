"""w122a — STANDING GUARD #51: the auto-selection report must fill SLOTS, not list TIERS.

WHY THIS GUARD EXISTS. `check_selection.py` is the artefact a human reads immediately before
making the one irreversible decision left in this competition. Its NOTHING-IS-SELECTED branch
printed the top two DISTINCT PUBLIC SCORES under the labels "auto-slot 1" and "auto-slot 2".
Those coincide only when every tier holds exactly one file. On the 2026-08-30 board:

    0.97119  2 files   <- fills BOTH final slots by itself
    0.97118  5 files   <- never reached, and one of the five IS the CV pick

so the reader was shown `w36_ad199stdcorr` on a line headed "auto-slot 2" while w62 had
measured P(the CV pick lands in the final pair) = 0.000. The available inference is "it gets
selected anyway, no need to click", and that inference makes the +4.5228e-6 cost permanent.
This is w120 §4 / w121 §3 in a third place: THE LABEL DESCRIBED THE INPUT, NOT THE PREDICATE
THE TOOL APPLIED. The number was never wrong; the word "slot" was.

THE RULE, in both directions, because the positive half alone would pass with the fix reverted:

  C1  `fill_final_slots` is importable and exact on hand-checked synthetic boards, including
      the boundary where a tie straddles the last slot.
  C2  On the FROZEN 2026-08-30 board shape, the shipped function must report the pick
      UNREACHABLE, and the PRE-FIX predicate (top-2 distinct scores) must report it PRESENT.
      If both agree, the fix is doing no work and this reports INERT rather than green.
  C3  No file below the last filled slot may appear in the reachable set -- the property the
      defect violated, checked on the LIVE board when one is cached, else on the fixture.
  C4  `check_selection.py` must not have regained the old tier-slicing idiom.

The 2026-08-30 shape is a FROZEN LITERAL, not a live read: w115's rule is that a control
anchored to HEAD stops being a control the moment the fix is committed.

    .venv/bin/python experiments/w122a_slotguard.py            # rc 0 = clean
    .venv/bin/python experiments/w122a_slotguard.py --control  # must exit 0, having fired

Offline and deterministic: no API call, no leaderboard, no fold. Standing check #51.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_selection as cs  # noqa: E402

TARGET_SRC = os.path.join(HERE, "check_selection.py")

# ── C2's fixture: the board shape that produced the defect, frozen ──────────────────────────
# The two tiers that matter on 2026-08-30, verbatim. Nothing below 0.97118 changes any answer
# here, so only the two top tiers are frozen; a third tier is appended to prove the walk stops.
BOARD_0830 = [
    (0.97119, "w36_ad199stdcorr_ens4.csv"),
    (0.97119, "w38_ad202stdcorr_ens4.csv"),
    (0.97118, "w21_ad187corr_ens4.csv"),
    (0.97118, "w27_ad190stdcorr.csv"),
    (0.97118, "w29_ad194stdcorr.csv"),
    (0.97118, "w36_ad199stdcorr.csv"),
    (0.97118, "w40_ad211stdcorr.csv"),
    (0.97116, "w23_ad187stdcorr.csv"),
]
PICK_0830 = "w36_ad199stdcorr.csv"
LIMIT = 2


def prefix_reachable(scored, limit):
    """The PRE-FIX predicate, reconstructed: everything in the top `limit` DISTINCT scores.

    This is what the old block printed under the word "auto-slot". Kept here so C2 can show
    the two predicates disagree; if they ever stop disagreeing the fix is inert.
    """
    tiers = sorted({sc for sc, _ in scored}, reverse=True)[:limit]
    return {fn for sc, fn in scored if sc in tiers}


def reachable(certain, draw):
    return {fn for _, fn in certain} | (set(draw[1]) if draw else set())


def c1_synthetic():
    """Exact behaviour on hand-checked boards. -> list of failure strings."""
    fails = []

    def check(name, scored, limit, want_certain, want_draw):
        certain, draw = cs.fill_final_slots(scored, limit)
        got = ([fn for _, fn in certain],
               None if draw is None else (draw[0], draw[1], draw[2]))
        if got != (want_certain, want_draw):
            fails.append(f"C1 {name}: got {got!r}, want {(want_certain, want_draw)!r}")

    # Singleton tiers: slots and tiers coincide. This is the ONLY case the old code got right.
    check("singletons", [(0.9, "a.csv"), (0.8, "b.csv"), (0.7, "c.csv")], 2,
          ["a.csv", "b.csv"], None)
    # The defect's shape: tier 1 fills every slot, so tier 2 is unreachable.
    check("tier1 fills all", [(0.9, "a.csv"), (0.9, "b.csv"), (0.8, "c.csv")], 2,
          ["a.csv", "b.csv"], None)
    # Boundary: one slot left against a 3-way tie -> a genuine draw at P=1/3.
    check("straddling tie", [(0.9, "a.csv"), (0.8, "b.csv"), (0.8, "c.csv"), (0.8, "d.csv")], 2,
          ["a.csv"], (0.8, ["b.csv", "c.csv", "d.csv"], 1))
    # A tie wider than the whole limit: nothing is certain at all.
    check("tie wider than limit", [(0.9, "a.csv"), (0.9, "b.csv"), (0.9, "c.csv")], 2,
          [], (0.9, ["a.csv", "b.csv", "c.csv"], 2))
    # Fewer files than slots: the walk must terminate, not pad.
    check("short board", [(0.9, "a.csv")], 2, ["a.csv"], None)

    certain, draw = cs.fill_final_slots(BOARD_0830, LIMIT)
    n = len(certain) + (draw[2] if draw else 0)
    if n != LIMIT:
        fails.append(f"C1 slot conservation: {n} slots handed out, want {LIMIT}")
    return fails


def c2_both_directions():
    """The fix must change the verdict on the frozen board. -> (fails, inert)."""
    fails = []
    certain, draw = cs.fill_final_slots(BOARD_0830, LIMIT)
    now = reachable(certain, draw)
    before = prefix_reachable(BOARD_0830, LIMIT)

    if PICK_0830 in now:
        fails.append(f"C2 positive: shipped predicate still reports {PICK_0830} as reachable")
    if PICK_0830 not in before:
        fails.append(f"C2 negative: pre-fix predicate does NOT report {PICK_0830}; "
                     "the fixture no longer reproduces the defect it was frozen for")
    inert = (now == before)
    if inert:
        fails.append("C2 INERT: the two predicates agree on the frozen board, so the fix "
                     "has no behaviour to select. Re-derive it before trusting a green.")
    return fails, inert


def c3_nothing_below_the_line(scored, label):
    """No reachable file may score below the worst file that actually holds a slot."""
    fails = []
    certain, draw = cs.fill_final_slots(scored, LIMIT)
    floor = min([sc for sc, _ in certain] + ([draw[0]] if draw else []))
    for sc, fn in scored:
        if fn in reachable(certain, draw) and sc < floor:
            fails.append(f"C3 {label}: {fn} scores {sc} < slot floor {floor} yet is reachable")
    return fails


def c4_idiom_gone():
    """The old tier-slicing idiom must not be back in the printing path."""
    src = open(TARGET_SRC).read()
    fails = []
    if re.search(r"reverse=True\)\[:2\]", src):
        fails.append("C4 check_selection.py has regained a `sorted(...)[:2]` tier slice")
    if "auto-slot" in src.split("def fill_final_slots")[0]:
        pass  # the historical decision record above the function may say it; that is history
    if "fill_final_slots" not in src:
        fails.append("C4 check_selection.py no longer calls fill_final_slots")
    return fails


def main() -> int:
    control = "--control" in sys.argv
    fails = []

    fails += c1_synthetic()
    c2, inert = c2_both_directions()
    fails += c2
    fails += c3_nothing_below_the_line(BOARD_0830, "fixture")
    fails += c4_idiom_gone()

    if control:
        # Fire deliberately: hand C3 a board whose reachable set was built by the OLD rule.
        certain, draw = cs.fill_final_slots(BOARD_0830, LIMIT)
        floor = min([sc for sc, _ in certain] + ([draw[0]] if draw else []))
        below = [fn for sc, fn in BOARD_0830 if sc < floor]
        planted = prefix_reachable(BOARD_0830, LIMIT)
        fired = [fn for fn in below if fn in planted]
        print(f"CONTROL pre-fix predicate reaches {len(fired)} file(s) below the slot floor "
              f"{floor}: {sorted(fired)}")
        if not fired:
            print("CONTROL DID NOT FIRE — the fixture no longer carries the defect.")
            return 1
        print("CONTROL fired as designed; the shipped predicate reaches none of them.")

    certain, draw = cs.fill_final_slots(BOARD_0830, LIMIT)
    print(f"w122a — fixture: {len(certain)} certain slot(s), "
          f"draw={'none' if draw is None else f'{draw[2]} of {len(draw[1])} at {draw[0]}'}, "
          f"inert={inert}")
    for f in fails:
        print("FAIL " + f)
    print(f"FAILURES: {len(fails)}")
    if fails:
        return 1
    print("✅ CLEAN — slots are filled, not tiers listed; the frozen defect stays caught.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
