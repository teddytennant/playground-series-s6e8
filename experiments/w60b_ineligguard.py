"""w60 (2026-08-22) — the WANTED-ineligibility registry must be COMPLETE, not merely present.

WHY THIS EXISTS, AND WHY w56b DID NOT COVER IT
-----------------------------------------------
`w56b_wantedguard.py` asks three good questions — is the guard there, does live WANTED pass it,
does it FIRE — and it passed every day since 2026-08-22 while `WANTED_INELIGIBLE` was MISSING
`w40_ad211`, the arm w40d_prereg.txt was written about and the source of the very sentence the
dict exists to enforce. w56b could not have caught it: its case list is hand-typed, and it was
typed from the two arms w56 was looking at. A guard whose cases come from the same place as the
bug cannot see the bug.

⚠ THE LESSON IS NARROWER THAN w54/w55/w56's "a rule in a paragraph is not a rule", and nastier:
**when you move a rule into an enforcing structure, enumerate its cases FROM THE RULE, not from
the instances in front of you.** A partly-populated enforcer reads exactly like a complete one.

SO THIS GUARD DERIVES THE CASES FROM THE PREREGISTRATIONS, IN BOTH DIRECTIONS
-----------------------------------------------------------------------------
  FORWARD   every prereg on disk that DECLARES an arm WANTED-ineligible must have a live key.
            The declaring files are enumerated in MAPPING below, with the clause each one is
            required to still contain — so deleting the clause is a failure, not a pass.
  SWEEP     every `*prereg*.txt` in experiments/ is scanned for the declaration pattern, and a
            declaring file that MAPPING does not know about is a FAILURE. This is the half that
            catches the NEXT omission: a future arm registered as ineligible and never keyed.
  REVERSE   every key in WANTED_INELIGIBLE must be backed by a declaring prereg on disk, so a
            key cannot be invented (or silently retired) without the registration moving too.
  EXERCISE  `assert_wanted_eligible` and `w26g_send.above_tier_reason` are both made to FIRE on
            a real ad211 file by name. w54's lesson: a guard nobody has seen fire is decorative.

Exit 0 = registry complete and both enforcement paths bite. Exit 1 = a registered rule is not
being enforced somewhere. Run on any run that touches `check_selection.py`, `w26g_send.py`, or
adds a prereg with an ineligibility clause.

    .venv/bin/python experiments/w60b_ineligguard.py
"""
from __future__ import annotations

import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TARGET = os.path.join(HERE, "check_selection.py")

# Every preregistration that DECLARES an arm ineligible for CV-based selection -> the
# WANTED_INELIGIBLE key(s) that must exist because of it, and a substring of the clause that
# must still be in the file. The clause is quoted so that softening the prereg fails here too:
# retiring an entry is allowed (the dict says how), quietly deleting the rule is not.
MAPPING = {
    "w40d_prereg.txt": (
        {"w40_ad211"},
        "ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV.",
    ),
    # w42b's clause is written about the ext_members16 POOL, so it covers BOTH arms built on
    # it — ARM 217 and ARM 216 (which is ARM 217 minus one member; w50_prereg §5 says removing a
    # member does not discharge the clause).
    "w42b_prereg.txt": (
        {"w42_ad217", "w50_ad216"},
        "NO ARM BUILT ON ext_members16 IS ELIGIBLE FOR check_selection.WANTED ON CV ALONE.",
    ),
    # w51 is the ARM 216 es-read and quotes w42b's clause as the thing it is trying to
    # discharge. Found by the SWEEP below, not by hand — which is the sweep's whole point.
    "w51_prereg.txt": (
        {"w42_ad217", "w50_ad216"},
        "NO ARM BUILT ON ext_members16 IS ELIGIBLE FOR check_selection.WANTED ON CV ALONE.",
    ),
    "w50_prereg.txt": (
        {"w50_ad216"},
        "NOT WANTED-eligible, whatever d_five turns out to be.",
    ),
    # w60 restates w40d's clause verbatim in the section that ADDS the key. It maps to the same
    # arm; it is listed so the SWEEP below does not read the restatement as an unkeyed rule.
    "w60_prereg.txt": (
        {"w40_ad211"},
        "ARM 211 IS NOT ELIGIBLE FOR check_selection.WANTED, WHATEVER ITS CV.",
    ),
}

# A DECLARATION, for the sweep. BOTH halves are required on the same line:
#   * the standalone caps word ELIGIBLE — `WANTED_INELIGIBLE` does not match it (no space
#     before ELIGIBLE), so prose that merely *mentions* the dict is not read as a new rule;
#   * the literal `check_selection.WANTED`, so the line has to be about THE SELECTION SET.
# ⚠ The first cut used only the caps word and flagged w59_prereg's "P5/P8 measure ELIGIBLE
# files with cross-fitted CVs" as a registration. A sweep that cries wolf gets its MAPPING
# padded to silence it, which is how a completeness check turns back into a hand-typed list.
DECLARE = re.compile(r"(?<= )ELIGIBLE(?= )")
SCOPE = "check_selection.WANTED"


def _fail(msg):
    print(f"  ⛔ {msg}")
    return 1


def main() -> int:
    fails = 0

    spec = importlib.util.spec_from_file_location("_check_selection_w60", TARGET)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        print(f"⛔ importing check_selection.py raised SystemExit({exc.code}) — live WANTED "
              f"names a barred arm. Fix WANTED, not this guard.")
        return 1
    reg = mod.WANTED_INELIGIBLE
    print(f"registry: {len(reg)} entries — {', '.join(sorted(reg))}\n")

    # ---- FORWARD -----------------------------------------------------------------------
    print("FORWARD — every declaring prereg has a live key, and still contains its clause:")
    declared = set()
    for fn, (keys, clause) in sorted(MAPPING.items()):
        path = os.path.join(HERE, fn)
        if not os.path.exists(path):
            fails += _fail(f"{fn} is GONE. A registration cannot be enforced if it is deleted.")
            continue
        src = open(path).read()
        if clause not in src:
            fails += _fail(f"{fn} no longer contains its clause {clause!r:.70} — the rule was "
                           f"softened in the prereg rather than retired in the dict.")
            continue
        for k in keys:
            declared.add(k)
            if k not in reg:
                fails += _fail(f"{fn} declares {k} ineligible and WANTED_INELIGIBLE has no such "
                               f"key. THIS IS THE w60 BUG. Add the key, quoting the clause.")
            else:
                print(f"  ✅ {fn:<20} -> {k}")

    # ---- SWEEP -------------------------------------------------------------------------
    print("\nSWEEP — no prereg declares an arm ineligible without MAPPING knowing about it:")
    unknown = []
    for path in sorted(glob.glob(os.path.join(HERE, "*prereg*.txt"))):
        fn = os.path.basename(path)
        hits = [ln.strip() for ln in open(path)
                if DECLARE.search(ln) and SCOPE in ln]
        if not hits:
            continue
        if fn in MAPPING:
            print(f"  ✅ {fn:<20} {len(hits)} declaration line(s), mapped")
        else:
            unknown.append((fn, hits))
    for fn, hits in unknown:
        fails += _fail(f"{fn} contains an ineligibility DECLARATION that MAPPING does not "
                       f"cover — this is the next omission, caught:")
        for h in hits[:3]:
            print(f"       {h[:110]}")
    if not unknown:
        print(f"  ✅ swept {len(glob.glob(os.path.join(HERE, '*prereg*.txt')))} prereg files, "
              f"no unmapped declaration")

    # ---- REVERSE -----------------------------------------------------------------------
    print("\nREVERSE — every key is backed by a registration on disk:")
    for k in sorted(reg):
        if k in declared:
            print(f"  ✅ {k:<12} backed")
        else:
            fails += _fail(f"{k} is in WANTED_INELIGIBLE but no prereg in MAPPING declares it. "
                           f"Either the registration moved or the key was invented.")

    # ---- EXERCISE ----------------------------------------------------------------------
    # A guard nobody has seen fire is decorative (w54). Both enforcement paths, by name.
    print("\nEXERCISE — both enforcement paths bite on a real ad211 file:")
    fired = False
    try:
        mod.assert_wanted_eligible({"w40_ad211stdcorr.csv", "w36_ad199stdcorr.csv"})
    except SystemExit as exc:
        fired = exc.code == 3
    if fired:
        print("  ✅ assert_wanted_eligible fires on w40_ad211stdcorr.csv (exit 3)")
    else:
        fails += _fail("assert_wanted_eligible did NOT fire on w40_ad211stdcorr.csv.")

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "agent"))
    import w26g_send as S  # noqa: E402

    class Row:  # the duck-type `above_tier_reason` reads: .fam, .file, .cv
        def __init__(self, file, fam, cv):
            self.file, self.fam, self.cv = file, fam, cv

    bar = 0.9701294160          # w59a's conditional bar; ad211stdcorr is 8.0e-6 ABOVE it
    r = Row("w40_ad211stdcorr.csv", "h3", 0.9701374732624002)
    why = S.above_tier_reason(r, bar)
    if why and "ineligible" in why:
        print(f"  ✅ above_tier_reason blocks w40_ad211stdcorr ABOVE the CV bar: {why[:72]}...")
    else:
        fails += _fail(f"above_tier_reason waved w40_ad211stdcorr through above the tier "
                       f"(returned {why!r}). This was live for two days.")

    # ...and the control: an ELIGIBLE arm over the bar must still be allowed through, or the
    # fix has broken the lever w59 §7 is trying to arm rather than protecting it.
    ok = Row("w36_ad199stdcorr_ens4.csv", "ens4", 0.9701366)
    why_ok = S.above_tier_reason(ok, bar)
    if why_ok is None:
        print("  ✅ and w36_ad199stdcorr_ens4 (eligible, over the bar) is STILL allowed above it")
    else:
        fails += _fail(f"the fix over-blocks: an eligible ad199 arm was refused ({why_ok!r})")

    print(f"\nFAILURES: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
