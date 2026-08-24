"""w77b — the standing guard for what w77 established about the hijack bar.

Five controls, all file reads, NO refit and NO network. Runs in well under a second.

C1  w63a's bracket is STILL the 16.5e-6 hole with the same two ends. If a later run silently
    re-derives the artefact and the hole closes or moves, the w77 numbers stop applying and
    this fires rather than letting them be quoted at a board they no longer describe.

C2  w77a's P6 STAYS VOID. w77a reported the enlarged cost curve crossing `base` ELEVEN times.
    It does not — that is the scored/unscored crevasse opened by a pathological GLS gap, and
    w77c records it as such. ⚠ THIS IS THE ONE THAT MATTERS. `w77a_bracket.json` on disk reads
    `falsified: ["P6", ...]` and a later run reading only that field would conclude "the break-
    even is not a well-defined scalar and the live bar is under-specified" — a false and
    alarming claim, resting on a broken instrument. Same shape as w76b's C3, and it is the
    SECOND time a falsification on this account has needed pinning against its own artefact.

C3  w77c's Q2 STAYS FALSIFIED. The LIVE GLS common gap (+1024.16e-6) is OUTSIDE the hull of
    the eight z values it averages ([+1039.99, +1076.07]). No later run gets to cite the live
    estimator as well-behaved just because the enlarged one is worse.

C4  w77d STAYS EXPLORATORY. It measures a bar 5.889e-6 STRICTER than the sender's, off a
    construction designed after the answer was visible. Nobody may adopt it without its own
    pre-registration, so this asserts w77d still disclaims itself and still records that bar.

C5  THE LIVE BAR STAYS PUT, read through `w26g_send.HIJACKPRICE` — NEVER by re-typing the path
    (w63 §9: a guard that names the artefact's path instead of the constant the code reads is
    testing a different file the moment the code moves). If a later run adopts w77d's number,
    this fires and points at the missing pre-registration.

    .venv/bin/python experiments/w77b_bracketguard.py
    .venv/bin/python experiments/w77b_bracketguard.py --selftest
"""
from __future__ import annotations

import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/agent")
sys.path.insert(0, HERE)

W63A_BRACKET_WIDTH = 16.491624447501074
W63A_HI, W63A_LO = "w40_ad211stdcorr", "w40_ad211std_rescale"
LIVE_BAR = 0.9701288617          # displayed form; compared at 1e-9
W77D_BAR = 0.9701347511
Q2_HULL = (1039.99, 1076.07)

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
        for path, val in mut.get(name, []):
            t = d
            for k in path[:-1]:
                t = t[k]
            t[path[-1]] = val
        return d

    w63a, w77a, w77c, w77d = (load(f) for f in ("w63a_setprice.json", "w77a_bracket.json",
                                                "w77c_glssweep.json", "w77d_holefill.json"))

    b = w63a["bracket"]["uncond"]
    chk("C1", abs(float(b["width"]) - W63A_BRACKET_WIDTH) < 1e-9
        and b["hi"] == W63A_HI and b["lo"] == W63A_LO,
        f"w63a's bracket is now {b['hi']}..{b['lo']} at {float(b['width']):.6f}e-6 — the w77 "
        f"numbers describe a hole that no longer exists. Re-run w77a before quoting them.")

    chk("C2", "P6" in w77a["falsified"] and w77c.get("p6_void") is True
        and "artefact" in str(w77c.get("p6_void_reason", "")),
        "w77a still records P6 FALSIFIED but w77c no longer voids it — the eleven crossings "
        "would be read as a real non-monotone cost curve and as evidence the live bar is "
        "under-specified. They are neither.")

    live = w77c["live"]
    chk("C3", w77c["predictions"]["q2"] is False and live["in_hull"] is False
        and abs(live["zmin"] - Q2_HULL[0]) < 0.01 and abs(live["zmax"] - Q2_HULL[1]) < 0.01,
        f"w77c no longer records the LIVE gh {live['gh']:.2f} as outside its own hull "
        f"[{live['zmin']:.2f}, {live['zmax']:.2f}] — Q2's falsification has been lost.")

    chk("C4", w77d.get("exploratory") is True and w77d.get("registers_nothing") is True
        and w77d.get("writes_a_bar") is False and w77a.get("writes_a_bar") is False
        and abs(float(w77d["bar_would_be"]) - W77D_BAR) < 1e-9,
        "w77d stopped disclaiming itself, or its recorded bar moved. It is a post-hoc "
        "construction and may not be adopted without its own pre-registration (w59a).")

    # C5 reads the bar through the SENDER'S OWN constant, never by re-typing the path (w63 §9),
    # and then through `load` so a planted defect reaches the same dict the sender would see.
    import w26g_send as SEND                                    # noqa: E402
    sender_file = os.path.basename(SEND.HIJACKPRICE)
    read = load(sender_file) if sender_file == "w63a_setprice.json" else json.load(
        open(SEND.HIJACKPRICE))
    chk("C5", sender_file == "w63a_setprice.json"
        and abs(float(read["cv_bar_new"]) - LIVE_BAR) < 1e-9,
        f"the sender reads {sender_file} at bar {float(read.get('cv_bar_new', float('nan'))):.10f}, "
        f"not the {LIVE_BAR:.10f} w77 measured against. If the bar moved on purpose, the run that "
        f"moved it owed a pre-registration and this constant an update.")
    return len(FAILURES)


def main() -> None:
    if "--selftest" in sys.argv:
        print("SELFTEST — each planted defect must fire its OWN control and no other\n")
        cases = [
            ("C1", {"w63a_setprice.json": [(["bracket", "uncond", "width"], 3.0)]}),
            ("C2", {"w77c_glssweep.json": [(["p6_void"], False)]}),
            ("C3", {"w77c_glssweep.json": [(["predictions", "q2"], True)]}),
            ("C4", {"w77d_holefill.json": [(["exploratory"], False)]}),
            ("C5", {"w63a_setprice.json": [(["cv_bar_new"], W77D_BAR)]}),
        ]
        bad = 0
        for want, mut in cases:
            run(mut)
            ok = FAILURES == [want]
            print(f"  planted {want:3s} -> fired {FAILURES or 'nothing'}  "
                  f"{'✅' if ok else '⛔ WRONG CONTROL'}\n")
            bad += (not ok)
        print(f"selftest {'PASSED' if not bad else 'FAILED'} — five controls, "
              f"five separately-plantable defects.")
        sys.exit(1 if bad else 0)

    print("w77b — the hijack-bracket guard")
    n = run()
    print(f"\nFAILURES {n}")
    if n:
        print("⛔ Something w77 established has been overwritten or laundered. Read JOURNAL w77.")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
