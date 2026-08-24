"""w76b — standing guard: the non-additivity readings a later run could mistake for a rule.

w63a_setprice.json carries a `nonadditivity` block and a POST-HOC partition (78/78, 0/26, 1/1)
that w63 §4 turned into a candidate rule. w76a tested that rule out of sample and FALSIFIED half
of it: same-side => optimistic held 205/205, but straddle => pessimistic broke on 41 of 120
pairs. The live hazard is not arithmetic — it is that a later run reads either artefact and
treats the partition as established, exactly as w63a's own summary invites.

This guard is one file read and no refit. It pins four things:

  C1  the SENDER'S REGIME direction, on both boards. Every candidate the sender actually queues
      is weaker than the auto pair, so "neither helps" is the cell that governs the set price.
      It is 78/78 positive on w63a's board and 190/190 on w76a's. If it ever flips, the additive
      price stops being optimistic in the regime w60 was burned in, and the set price would be
      anti-conservative in exactly that direction.
  C2  w63a's own additive_error is POSITIVE, i.e. the day it priced was optimistic — the same
      direction as C1, read off an independent field.
  C3  w76a still records P3 as FALSIFIED. A later run must not be able to cite w76a as having
      confirmed the iff.  (The w75b C4 move: pin the finding so it cannot be inherited silently.)
  C4  w63a's nonadditivity block is unmoved, so the numbers C1 and C3 are about are the numbers
      still on disk.

    .venv/bin/python experiments/w76b_addguard.py            # rc 0
    .venv/bin/python experiments/w76b_addguard.py --selftest # planted defects, each must fire
"""
from __future__ import annotations

import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
W63A = os.path.join(HERE, "w63a_setprice.json")
W76A = os.path.join(HERE, "w76a_addtest.json")

W63A_NEITHER_N, W63A_NEITHER_POS = 78, 78
W63A_NONADD = dict(n_pairs=105, n_joint_gt_add=79)
FAILURES = 0


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def check(a: dict, b: dict) -> None:
    # C1 — both boards, the sender's regime
    nb = b["cells"]["neither helps"]
    if nb["n_neg"] != 0 or nb["n"] != nb["n_pos"]:
        fail(f"C1: w76a's 'neither helps' cell is no longer all-positive "
             f"({nb['n_pos']}/{nb['n']}, {nb['n_neg']} negative) -- the additive price has "
             f"stopped being optimistic in the regime the sender queues into")
    else:
        print(f"  C1a w76a  'neither helps' {nb['n_pos']}/{nb['n']} positive, "
              f"min gap {nb['min_gap']:+.4f}e-6")
    if not (a["nonadditivity"]["n_joint_gt_add"] == W63A_NONADD["n_joint_gt_add"]
            and a["nonadditivity"]["n_pairs"] == W63A_NONADD["n_pairs"]):
        fail(f"C1b/C4: w63a's nonadditivity block moved: {a['nonadditivity']['n_pairs']} pairs, "
             f"{a['nonadditivity']['n_joint_gt_add']} positive (pinned "
             f"{W63A_NONADD['n_pairs']}/{W63A_NONADD['n_joint_gt_add']})")
    else:
        print(f"  C4  w63a  nonadditivity unmoved: {W63A_NONADD['n_pairs']} pairs, "
              f"{W63A_NONADD['n_joint_gt_add']} with joint > add")

    # C2 — the same direction from an independent field
    ae = float(a["additive_error"])
    if ae <= 0:
        fail(f"C2: w63a's additive_error is {ae:+.4f}e-6 -- not optimistic. C1 and C2 disagree "
             f"and one of the two artefacts is wrong.")
    else:
        print(f"  C2  w63a  additive_error {ae:+.4f}e-6 (optimistic), agrees with C1")

    # C3 — the falsification cannot be inherited as a confirmation
    fl = list(b.get("falsified", []))
    if b["predictions"].get("p3") is not False or not any("P3" in s for s in fl):
        fail(f"C3: w76a no longer records P3 as FALSIFIED (p3={b['predictions'].get('p3')}, "
             f"falsified={fl}). w63 §4's iff is NOT established and nothing may cite it as such.")
    else:
        print(f"  C3  w76a  P3 still recorded FALSIFIED: {fl}")


def selftest() -> None:
    import copy
    a0, b0 = json.load(open(W63A)), json.load(open(W76A))
    planted = [
        ("C1 the sender's regime flips", lambda a, b: b["cells"]["neither helps"].update(n_neg=3)),
        ("C2 the additive error turns pessimistic", lambda a, b: a.update(additive_error=-1.0)),
        ("C3 the falsification is laundered into a confirmation",
         lambda a, b: (b["predictions"].update(p3=True), b.update(falsified=[]))),
        ("C4 w63a's nonadditivity block is regenerated",
         lambda a, b: a["nonadditivity"].update(n_pairs=99)),
    ]
    global FAILURES
    print("=" * 90)
    print("SELFTEST — each planted defect must make this guard FAIL")
    print("=" * 90)
    bad = 0
    for name, mut in planted:
        a, b = copy.deepcopy(a0), copy.deepcopy(b0)
        mut(a, b)
        FAILURES = 0
        print(f"\n  planted: {name}")
        check(a, b)
        if FAILURES == 0:
            print(f"  *** THE GUARD DID NOT FIRE ON '{name}' -- it does not check what it claims")
            bad += 1
        else:
            print(f"  -> fired, {FAILURES} failure(s). GOOD.")
    FAILURES = 0
    print(f"\n  {len(planted)} defects planted, {len(planted) - bad} caught.")
    if bad:
        sys.exit(1)
    print("  ✅ SELFTEST PASSED — the guard fires on every defect it claims to cover.")


def main() -> None:
    if "--selftest" in sys.argv:
        selftest()
        return
    print("=" * 90)
    print("w76b — non-additivity guard (w63a + w76a artefacts, one read each, no refit)")
    print("=" * 90)
    check(json.load(open(W63A)), json.load(open(W76A)))
    print(f"\n  FAILURES {FAILURES}")
    if FAILURES:
        print("⛔ w76b FAILED. Read experiments/w76_prereg.txt and JOURNAL's w76 entry before "
              "touching either artefact.")
        sys.exit(1)
    print("  ✅ all four checks pass.")


if __name__ == "__main__":
    main()
