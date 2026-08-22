"""w56 (2026-08-22) — regression test for the WANTED-ineligibility guard.

Same shape as `w54a_vetoexpiry.py` and `w55a_unpriced.py`, and for the same reason:
prose in a journal does not survive, a test that re-reads the module does. w54 found
the ad216/ad217 VETO was only a sort order; w55 found the tier rule was only a printed
paragraph; w40d's WANTED-ineligibility rule was only a line in RESEARCH.md until w56.

Three assertions:
  1. the guard is still present and still wired in `check_selection.py`;
  2. the current WANTED passes it;
  3. it actually FIRES on an ineligible arm — exercised, not asserted.

Exit 0 = guard intact. Exit 1 = the guard is gone or has stopped working.
Run this on any run that touches `check_selection.py` or moves WANTED.
"""
from __future__ import annotations

import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "experiments", "check_selection.py")

# The exact strings that make the rule a rule. If a future edit removes any of them the
# rule has silently reverted to a paragraph.
REQUIRED = [
    "WANTED_INELIGIBLE = {",
    "def assert_wanted_eligible(",
    "\nassert_wanted_eligible()",          # the CALL, not just the definition
    "raise SystemExit(3)",
    "w50_ad216",
    "w42_ad217",
]


def main() -> int:
    with open(TARGET) as fh:
        src = fh.read()

    print(f"reading {os.path.relpath(TARGET, ROOT)}")
    missing = [q for q in REQUIRED if q not in src]
    if missing:
        print("⛔ THE WANTED GUARD IS GONE. Missing from check_selection.py:")
        for q in missing:
            print(f"     {q!r}")
        print("   w40d's rule is back to being a paragraph. Restore it before selecting.")
        return 1
    print(f"  ✅ all {len(REQUIRED)} guard strings present, including the call site")

    # Import the module WITHOUT running main() — the import itself runs the guard on
    # the live WANTED, which is assertion 2.
    spec = importlib.util.spec_from_file_location("_check_selection", TARGET)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit as exc:
        print(f"⛔ importing check_selection.py raised SystemExit({exc.code}) — the live")
        print(f"   WANTED = names a w40d-ineligible arm. That is the guard doing its job;")
        print(f"   fix WANTED, do not weaken the guard.")
        return 1
    print(f"  ✅ live WANTED passes: {sorted(mod.WANTED)}")

    print(f"  registry ({len(mod.WANTED_INELIGIBLE)} entries):")
    for pat, why in sorted(mod.WANTED_INELIGIBLE.items()):
        print(f"     {pat:<12} {why}")

    # Assertion 3 — exercise it. A guard nobody has seen fire is a guard nobody knows
    # works; this is the whole lesson of w54/w55.
    fired = False
    try:
        mod.assert_wanted_eligible({"w42_ad217stdcorr.csv", "w23_ad187stdcorr.csv"})
    except SystemExit as exc:
        fired = exc.code == 3
    if not fired:
        print("⛔ THE GUARD DID NOT FIRE on w42_ad217stdcorr.csv. It is decorative.")
        return 1
    print("  ✅ fires on w42_ad217stdcorr.csv (exit 3), as it must")

    fired = False
    try:
        mod.assert_wanted_eligible({"w50_ad216stdcorr.csv"})
    except SystemExit as exc:
        fired = exc.code == 3
    if not fired:
        print("⛔ THE GUARD DID NOT FIRE on w50_ad216stdcorr.csv. It is decorative.")
        return 1
    print("  ✅ fires on w50_ad216stdcorr.csv (exit 3), as it must")

    print("\nWANTED guard intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
