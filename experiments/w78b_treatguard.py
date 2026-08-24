"""w78b — the standing guard for what w78 established, which is mostly that w77's alarms do not
generalise. Five controls, all file reads, NO refit and NO network. Well under a second.

C1  EVERY VERDICT ON DISK RECOMPUTES FROM ITS OWN NUMBERS. `w78a_treatment.json` carries a
    `stats` table of (measured, comparator, bar) for all thirteen predictions, and this
    recomputes each one and compares against the recorded `confirmed`/`falsified` lists. w77 §4
    is the reason: a `falsified` field that cannot be re-derived from the artefact it sits in
    reads as rigour and is not. This is the third run in a row that a verdict on this account
    has needed pinning, so it is now mechanical rather than editorial.

C2  ⚠ THE ONE THAT MATTERS. NO Sum|w| HEALTH BAR IS ADOPTED, AND NONE MAY BE. w77 §5 concluded
    "out-of-hull is not the alarm; Sum|w| is", and a later run acting on that sentence would
    install a threshold on `fit`. w78 P5a re-tested it on a different family of designs and got
    rho -0.335, not +0.957. Worse, the decision rule w78 itself registered mechanically outputs
    a bar BELOW the live fit's own Sum|w|, so installing it would refuse every send from day
    one. This fires if `sumw_check_adopted` flips or if that arithmetic stops holding.

C3  THE TREATMENT MIXTURE IS A FACT, NOT A DEFECT. w63a's live bracket really does have a
    SCORED hi end and an UNSCORED lo end, and a later run will notice that and be tempted to
    call the bar broken. The measured offset between the two treatments is 0.15e-6 against a
    registered 1.0e-6 bar with no consistent sign, so it explains nothing. This asserts the
    mixture is still recorded AND still measured as inert AND that R1 stayed un-adopted.

C4  THE BAR IS INSENSITIVE TO THE GLS AND SENSITIVE TO THE HOLE. Four constructions spanning
    28.9e-6 of common gap agree on H_binding to 0.36e-6 once the hole is filled, and the
    filled/empty split is 5.6e-6. That is the finding a successor to w77d actually needs, and
    it is the one thing here worth quoting.

C5  THE COUNTEREXAMPLE SURVIVES. `live_18` has the LARGEST Sum|w| of every fit measured and the
    SMALLEST |gh - mean z|. If that row is ever edited into agreement with w77c's sweep, the
    reason C2 exists has been laundered out of the artefact.

    .venv/bin/python experiments/w78b_treatguard.py
    .venv/bin/python experiments/w78b_treatguard.py --selftest
"""
from __future__ import annotations

import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

P2A_BAR = 1.0                    # the registered treatment-offset bar w78 P2a failed against
GH_SPREAD_MIN = 25.0             # C4: the four constructions must still disagree this much
H_FILLED_MAX = 1.0               # C4: ...and still agree this closely once the hole is filled
H_SPLIT_MIN = 5.0                # C4: ...with the empty/filled split at least this wide
CMP = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
       "<": lambda a, b: a < b, "==": lambda a, b: a == b}

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

    a = load("w78a_treatment.json")

    # ---- C1: recompute all thirteen verdicts from `stats` -------------------------------------
    rec = {t: (t in a["confirmed"]) for t in a["stats"]}
    bad = []
    for tag, (val, op, bar) in a["stats"].items():
        if tag not in a["confirmed"] and tag not in a["falsified"]:
            bad.append(f"{tag} appears in neither list")
        elif CMP[op](float(val), float(bar)) != rec[tag]:
            bad.append(f"{tag} records {'CONFIRMED' if rec[tag] else 'FALSIFIED'} but "
                       f"{val:.4f} {op} {bar} is {CMP[op](float(val), float(bar))}")
    extra = (set(a["confirmed"]) | set(a["falsified"])) - set(a["stats"])
    chk("C1", not bad and not extra and a["failures"] == 0 and a["gate_a"]["worst"] == 0.0,
        "; ".join(bad) or f"tags {sorted(extra)} carry a verdict with no recomputable number, "
                          f"or GATE A no longer reproduces w63a exactly "
                          f"(worst {a['gate_a']['worst']:.3e}, failures {a['failures']}).")

    # ---- C2: no Sum|w| bar, and the arithmetic that forbids one ------------------------------
    live8 = next(r for r in a["fits"] if r["arm"] == "live_8")
    r2 = float(a["decisions"]["R2_sumw_bar"])
    chk("C2", a.get("sumw_check_adopted") is False and r2 < live8["sumw"]
        and "P5a falsified" in str(a.get("sumw_reason", "")),
        f"a Sum|w| health bar has been adopted, or R2's output {r2:.2f} is no longer below the "
        f"live fit's own {live8['sumw']:.2f}. w77 §5's sentence 'Sum|w| is the alarm' is a "
        f"property of w77c's nested sweep and NOT of `fit` in general — w78 P5a got rho "
        f"{a['p5_rho']:+.3f}. Installing a threshold on it refuses every send.")

    # ---- C3: the mixture is recorded and measured inert ---------------------------------------
    off = abs(float(a["p2_median_offset"]))
    chk("C3", a["design"]["bracket_mixed"] is True and off < P2A_BAR
        and a["decisions"]["R1_adopt_bracket_guard"] is False
        and "P2a" in a["falsified"] and "P2b" in a["falsified"],
        f"w63a's bracket is no longer recorded as MIXED, or the measured treatment offset "
        f"{off:.4f}e-6 has grown past the registered {P2A_BAR}e-6 bar, or R1 has been adopted. "
        f"The mixture is real and it explains nothing; do not re-derive the bar because of it.")

    # ---- C4: the finding worth quoting --------------------------------------------------------
    R = a["robust"]
    gh = [r["gh"] for r in R]
    he = [r["H"] for r in R if r["hole"] == "EMPTY"]
    hf = [r["H"] for r in R if r["hole"] == "FILLED"]
    chk("C4", len(R) == 4 and (max(gh) - min(gh)) > GH_SPREAD_MIN
        and (max(hf) - min(hf)) < H_FILLED_MAX and (min(he) - max(hf)) > H_SPLIT_MIN,
        f"the four constructions no longer split on the HOLE: gh spread "
        f"{max(gh) - min(gh):.2f}e-6, filled H spread {max(hf) - min(hf):.3f}e-6, "
        f"empty-minus-filled {min(he) - max(hf):.3f}e-6. The claim that w77d's 5.6e-6 "
        f"correction is about the empty bracket and not about the GLS rests on exactly this.")

    # ---- C5: the counterexample -------------------------------------------------------------
    fits = a["fits"]
    dev = {r["arm"]: abs(r["gh"] - r["zmean"]) for r in fits}
    big = max(fits, key=lambda r: r["sumw"])["arm"]
    small = min(dev, key=dev.get)
    chk("C5", big == "live_18" and small == "live_18" and float(a["p5_rho"]) < 0.0,
        f"the counterexample is gone: largest Sum|w| is `{big}`, smallest |gh - mean z| is "
        f"`{small}`, rho {float(a['p5_rho']):+.3f}. w78's whole case against a Sum|w| health "
        f"check is that those two are the SAME arm.")
    return len(FAILURES)


def main() -> None:
    if "--selftest" in sys.argv:
        print("SELFTEST — each planted defect must fire its OWN control and no other\n")
        cases = [
            ("C1", {"w78a_treatment.json": [(["stats", "P5a", 0], 0.99)]}),
            ("C2", {"w78a_treatment.json": [(["sumw_check_adopted"], True)]}),
            ("C3", {"w78a_treatment.json": [(["design", "bracket_mixed"], False)]}),
            ("C4", {"w78a_treatment.json": [(["robust", 3, "H"], 9.9)]}),
            ("C5", {"w78a_treatment.json": [(["fits", 1, "sumw"], 0.01)]}),
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

    print("w78b — the treatment/Sum|w| guard")
    n = run()
    print(f"\nFAILURES {n}")
    if n:
        print("⛔ Something w78 established has been overwritten or laundered. Read JOURNAL w78.")
    sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
