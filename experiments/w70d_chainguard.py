"""w70d — THE SEND-CHAIN SMOKE TEST. Would have caught w70 section 1 in a few seconds.

WHY THIS EXISTS
---------------
On 2026-08-23 the send path was DEAD and **all fourteen standing guards passed.** w69 shipped
`w69_ad208stdcorr.csv` into `submissions/` without a `stdflag.CORR_MAP` entry;
`require_corr_registered()` is an ASSERT reached at **import** time by `w26d_queueprice`, so
`w26d`, `w48e_order` and `w26g_send` all raised on import and the registered 2026-08-24 ten
could not have been priced or sent. Nothing noticed, because **not one of the fourteen guards
imports any of those three modules.**

⚠⚠ **A GUARD SUITE SAYS NOTHING ABOUT A MODULE NONE OF ITS MEMBERS IMPORTS.** The fix is not a
cleverer assertion, it is COVERAGE: import every module the send actually executes, in the order
the send executes them, and fail on anything that raises. It is the cheapest possible test and
it subsumes every import-time assert those modules already carry — `require_corr_registered`,
`stdflag.gate`, w26d's two pricer GATEs and w48e's veto/length asserts all run for free.

⛔ THIS IS A SMOKE TEST, NOT A DRY RUN. It imports; it does not plan, price to disk or send.
`w48e_order.py` writes nothing without `--write`, and this passes no argv through to it.

WHAT IT DOES NOT COVER, STATED SO NOBODY READS MORE INTO A PASS THAN IS THERE: a module can
import cleanly and still be wrong at call time. Pair a green run here with the four-command
chain's own dry run (`w48e_order.py --day <DAY>` then `w26g_send.py --n 10`) before a window.

    .venv/bin/python experiments/w70d_chainguard.py
"""
from __future__ import annotations

import importlib, os, sys, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

# In the order the four-command send chain reaches them. `stdflag` and `check_selection` are
# first because they are what the others assert against.
CHAIN = [
    ("stdflag",          "family/is_member labels and CORR_MAP; require_corr_registered"),
    ("check_selection",  "WANTED / WANTED_RETIRED / WANTED_INELIGIBLE, read by the sender"),
    ("w23b_sendqueue",   "step 1 — the ranked queue"),
    ("w53a_pricer",      "the LB model, FIT_CV_MIN/MAX"),
    ("w26d_queueprice",  "the priced queue — where w70 §1's assert actually fired"),
    ("w48e_order",       "step 2 — the registered day; exits 2 on an unregistered one"),
    ("w70c_plan0825",    "the 08-25 derivation w48e reads"),
    ("w26g_send",        "steps 3 and 4 — the gates and the send"),
]

FAILURES = 0


def main() -> None:
    global FAILURES
    print("=" * 92)
    print("w70d  SEND-CHAIN IMPORT SMOKE TEST")
    print("=" * 92 + "\n")
    stdout = sys.stdout
    for name, what in CHAIN:
        try:
            # These modules print at import. Swallow it so one line per module is readable;
            # the traceback below still goes to the real stdout if anything raises.
            with open(os.devnull, "w") as null:
                sys.stdout = null
                importlib.import_module(name)
            sys.stdout = stdout
            print(f"  ✅ {name:<20} {what}")
        except BaseException:                      # SystemExit included — w48e exits 2
            sys.stdout = stdout
            FAILURES += 1
            print(f"  🔴 {name:<20} {what}\n"
                  f"     ⛔ THE SEND PATH IS BROKEN AT THIS MODULE. Everything after it is "
                  f"unreachable.\n")
            traceback.print_exc()
            break                                  # later modules import this one; stop here
    sys.stdout = stdout

    # VACUITY CHECK — a pass over an empty or truncated list is not a pass.
    if not FAILURES and len(CHAIN) != 8:
        FAILURES += 1
        print(f"  *** FAILURE: the chain list is {len(CHAIN)} modules, not the 8 registered")

    print(f"\n  {len(CHAIN) - FAILURES} of {len(CHAIN)} reached.   FAILURES {FAILURES}")
    if FAILURES:
        print("\n  ⛔ Do NOT open a send window until this exits 0.")
        sys.exit(1)
    print("  Every module the send executes imports cleanly, and every import-time assert "
          "they carry\n  ran as a side effect. This is a SMOKE TEST — still dry-run the "
          "four-command chain before a window.")


if __name__ == "__main__":
    main()
