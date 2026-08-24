"""w80f -- STANDING GUARD: ext_members17 is NOT adopted, and its verdicts are NOT amended.

    .venv/bin/python experiments/w80f_packguard.py
    .venv/bin/python experiments/w80f_packguard.py --selftest

WHAT STATE THIS PROTECTS. w80/w81 adopted nothing, so at first sight there is nothing to
guard. That is wrong: the NON-adoption is itself state, and it is state a later run can
destroy quietly. Three instruments have now been pointed at data/ext_members17 and every one
of them came back licensing NOTHING:

    w80a_weakscreen.json   REGISTERED    REFUSE -- malformed (P1)
    w80b_foldsig.json      REGISTERED    INDETERMINATE
    w80e_permsig.json      REGISTERED    VOID (G2 power 0.450 < 0.500)

and two POST-HOC files exist alongside them (w80a_posthoc.json, w80c_foldsig_diag.json,
w80d_poscalib.json) carrying numbers that look adoptable and are not. The failure mode this
guard exists for is a later run reading the +10.14e-6 out of a "REGISTERED": false file,
deciding the fold-signature thread came out fine, and putting a pack column on the send path.

C1  the three REGISTERED verdicts are unchanged (nobody amended a verdict after the fact)
C2  every POST-HOC artefact still carries "REGISTERED": false
C3  w80e's gating controls are unchanged -- FPR 0.000, POWER 0.450 -- so nobody "fixed" the
    VOID by retuning ALPHA/M/the seed block, which prereg3 forbids by name
C4  no pack path appears anywhere on the send path (the sender, the queue, the price files)
C5  w80c's void chi2(4) thresholds are not referenced by any live script other than w80c
    itself and the files that document them as void
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Re-typed here ON PURPOSE, so that mutating an artefact cannot also mutate the thing the
# artefact is checked against. These are the values the three registered runs produced.
VERDICTS = {
    "w80a_weakscreen.json": ("verdict", "REFUSE -- malformed (P1)"),
    "w80b_foldsig.json":    ("verdict", "INDETERMINATE"),
    "w80e_permsig.json":    ("verdict", "VOID"),
}
POSTHOC = ["w80a_posthoc.json", "w80c_foldsig_diag.json", "w80d_poscalib.json"]
W80E_FPR, W80E_POWER = 0.0, 0.45
PACK_TOKENS = ("ext_members17", "weakscreen", "permsig", "foldsig", "poscalib")
SEND_PATH_SRC = ["w26g_send.py", "w48e_order.py", "w23b_sendqueue.py"]

FAILURES = []


def chk(tag: str, ok: bool, msg: str) -> None:
    if not ok:
        FAILURES.append(tag)
        print(f"  *** {tag} FAILED: {msg}")
    else:
        print(f"  {tag} ok")


def run(mut=None) -> int:
    global FAILURES
    FAILURES = []
    mut = mut or {}

    def load(name):
        d = json.load(open(os.path.join(HERE, name)))
        d.update(mut.get(name, {}))
        return d

    # C1 -- registered verdicts unchanged
    for name, (key, want) in VERDICTS.items():
        got = load(name).get(key)
        chk(f"C1[{name}]", got == want, f"verdict is {got!r}, registered value is {want!r}")

    # C2 -- post-hoc files still declare themselves post-hoc
    for name in POSTHOC:
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            chk(f"C2[{name}]", False, "post-hoc artefact is missing")
            continue
        d = load(name)
        chk(f"C2[{name}]", d.get("REGISTERED") is False,
            f'"REGISTERED" is {d.get("REGISTERED")!r}, must be false')

    # C3 -- the VOID was not "fixed" by retuning
    e = load("w80e_permsig.json")
    chk("C3a", abs(e.get("G1_fpr", -1) - W80E_FPR) < 1e-12,
        f"w80e FPR is {e.get('G1_fpr')}, registered run measured {W80E_FPR}")
    chk("C3b", abs(e.get("G2_power", -1) - W80E_POWER) < 1e-12,
        f"w80e POWER is {e.get('G2_power')}, registered run measured {W80E_POWER}")
    rd = e.get("registered_design", {})
    chk("C3c", rd.get("alpha") == 0.01 and rd.get("M") == 199 and rd.get("seed_base") == 90000,
        f"prereg3's ALPHA/M/seed block moved: {rd.get('alpha')}/{rd.get('M')}/{rd.get('seed_base')}")
    chk("C3d", e.get("G2_power", 1.0) < rd.get("g2_power", 0.0),
        "w80e no longer reads as a power failure -- the VOID has been worked around")

    # C4 -- nothing about the pack is on the send path
    for src in SEND_PATH_SRC:
        p = os.path.join(HERE, src)
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        hit = [t for t in PACK_TOKENS if t in txt]
        chk(f"C4[{src}]", not hit, f"send-path source references the pack thread: {hit}")

    # C5 -- w80c's void chi2(4) numbers are not quoted by a live script
    for f in sorted(os.listdir(HERE)):
        if not f.endswith(".py") or f in ("w80c_foldsig_diag.py", "w80d_poscalib.py",
                                          os.path.basename(__file__)):
            continue
        txt = open(os.path.join(HERE, f)).read()
        if "9.4877" in txt or "0.7107" in txt:
            chk(f"C5[{f}]", False, "quotes w80c's VOID chi2(4) thresholds")
    chk("C5", not any(t.startswith("C5[") for t in FAILURES),
        "a live script quotes w80c's void chi2(4) thresholds")

    print(f"\n  FAILURES {len(FAILURES)}: {FAILURES}")
    return 1 if FAILURES else 0


def selftest() -> int:
    """Each mutation must be caught, and by the EXPECTED tag -- declared here, not discovered."""
    cases = [
        ("verdict flipped to an adoption",
         {"w80b_foldsig.json": {"verdict": "MATCHED"}}, "C1[w80b_foldsig.json]"),
        ("post-hoc file promoted to registered",
         {"w80d_poscalib.json": {"REGISTERED": True}}, "C2[w80d_poscalib.json]"),
        ("power retuned upward to clear the gate",
         {"w80e_permsig.json": {"G2_power": 0.90}}, "C3b"),
        ("alpha loosened to manufacture power",
         {"w80e_permsig.json": {"registered_design": {"alpha": 0.05, "M": 199,
                                                      "seed_base": 90000, "g2_power": 0.5}}},
         "C3c"),
        ("gate bar lowered under the measured power",
         {"w80e_permsig.json": {"registered_design": {"alpha": 0.01, "M": 199,
                                                      "seed_base": 90000, "g2_power": 0.2}}},
         "C3d"),
    ]
    bad = 0
    print("  SELFTEST -- every mutation must be caught by its DECLARED tag\n")
    for desc, mut, want in cases:
        rc = run(mut)
        got = FAILURES
        hit = want in got
        print(f"    [{'PASS' if rc == 1 and hit else 'FAIL'}] {desc:44s} rc={rc} "
              f"expected {want!r} in {got}")
        if not (rc == 1 and hit):
            bad += 1
        print()
    print(f"  SELFTEST {'PASSED' if bad == 0 else f'FAILED on {bad} case(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("=" * 78)
    print("w80f -- STANDING GUARD: ext_members17 is NOT adopted")
    print("=" * 78)
    sys.exit(run())
