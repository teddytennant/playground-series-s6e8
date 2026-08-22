"""w57c -- PIN `w46c_predlb.MU`, the pricer's centring constant, and prove the pin fires.

⚠⚠ THE LANDMINE, FOUND BY w57 ON 2026-08-22 BY STEPPING ON IT.

`w46c_predlb.py:59` defines the pricer's centring constant as a MEASUREMENT off a data file:

    MU = _t[_t.cv >= 0.97].cv.mean()          # _t = w25a_cvlb_full.csv

`w25a_cvlb_full.py` regenerates that file from the live board, and it looks like a read-only
refresh -- an audit chore. It is not. Rewriting the table moves MU, and MU is the origin the
whole M2 design is expressed around:

    cv6 = (cv - MU) * 1e6            and the era term is an INTERACTION, era:cv6

`w53a_pricer` FITS on `w52b_cvlb93.csv`'s PRECOMPUTED `cv6` column -- frozen at the old MU --
while PREDICTING through `(cv - MU) * 1e6` at the new one. The two desync silently in the fit
and loudly at the gate. Measured this run:

    MU before  0.9700571395217258   (78 rows with cv >= 0.97)
    MU after   0.9700678992717492   (93 rows)          shift +10.76e-6
    w53a gate  "predict_flags() does not reproduce the M2 fit (21.8591 vs 7.7200e-6)"

`w26d_queueprice` imports `w53a_pricer`, and `w26g_send.py` imports `w26d_queueprice`, so for
as long as the table was stale-refreshed THE SENDER WOULD NOT IMPORT. w57 restored the table
with `git checkout` rather than re-centring the pricer the day before a registered ten-file
send whose slot 1 carries pre-registered LB thresholds.

WHY A PIN AND NOT A COMMENT. w53a's gate catches the desync, but only AFTER something has
already rewritten the table -- and it reports the symptom (a residual sd) rather than the
cause (someone ran an audit script). This pin names the cause, at import, before any pricing
happens. It is the same lesson this workspace has now paid for six times: a rule that lives in
a paragraph is not a rule. w54, w55, w56 each paid a run for one instance of it on the send
path; w56 found the fifth on the SELECTION path; this is the sixth, and the first where the
dangerous action is disguised as routine maintenance.

RETIRING THE PIN IS A DELIBERATE ACT. If the pricer is genuinely re-centred, that means: refit
M2, regenerate `w52b_cvlb93.csv`'s `cv6`, re-derive `w52b_joint.json` and `w52d_predlb.json`,
and update MU_PINNED here -- all in the SAME commit. Do not edit this constant to make a run
pass.

    .venv/bin/python experiments/w57c_muguard.py        # exit 0 = pin holds and fires correctly
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# The value the entire live M2 pricer is fitted and published at.
MU_PINNED = 0.9700571395217258
TOL = 1e-12

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else '*** FAIL ***':12s} {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def main() -> None:
    print("w57c — the MU pin\n")

    # ---- 1. the pin itself
    import w46c_predlb as W46C
    live_mu = float(W46C.MU)
    check("w46c_predlb.MU is at the pinned value",
          abs(live_mu - MU_PINNED) < TOL,
          f"live {live_mu!r} vs pinned {MU_PINNED!r} (shift "
          f"{(live_mu - MU_PINNED) * 1e6:+.3f}e-6)")

    # ---- 2. the published artefacts agree with the pin. If any of these drift apart the
    #        pricer is being read at one centring and fitted at another.
    for f, key in (("w52b_joint.json", "mu"), ("w52d_predlb.json", "mu")):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            check(f"{f} present", False, "missing")
            continue
        import json
        v = float(json.load(open(p))[key])
        check(f"{f}['{key}'] agrees with the pin", abs(v - MU_PINNED) < TOL,
              f"{v!r}")

    # ---- 3. THE COUPLING IS REAL, demonstrated rather than asserted. Recompute MU the way
    #        w46c does, from the table on disk, and show it reproduces the pin. If someone
    #        has refreshed the table, this is the line that says so in plain terms.
    t = pd.read_csv(os.path.join(HERE, "w25a_cvlb_full.csv")).dropna(subset=["cv", "lb"])
    recomputed = float(t[t.cv >= 0.97].cv.mean())
    check("w25a_cvlb_full.csv still reproduces MU",
          abs(recomputed - MU_PINNED) < TOL,
          f"{len(t[t.cv >= 0.97])} rows with cv>=0.97 give {recomputed!r}")

    # ---- 4. the pricer's own gate is intact and the module imports. This is the thing that
    #        would take the SENDER down, so it is checked here too and not merely assumed.
    try:
        import w53a_pricer as PR
        ok_import, why = True, f"RESID_SD {PR.RESID_SD:.4f}e-6"
    except Exception as e:                                    # noqa: BLE001
        ok_import, why = False, f"{type(e).__name__}: {str(e)[:90]}"
    check("w53a_pricer imports (w26d -> w26g depend on it)", ok_import, why)

    # ---- 5. ⚠ WATCH THE PIN FIRE. A guard nobody has seen fire is a guard nobody knows
    #        works (w54/w55/w56). Rebuild w53a's gate arithmetic at a DELIBERATELY shifted MU
    #        and require it to blow up by a wide margin.
    if ok_import:
        import w53a_pricer as PR
        _t = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
        # The exact MU the 08-22 refresh produced (93 rows with cv>=0.97), not a round number:
        # this replays the real incident rather than a synthetic perturbation.
        shifted = 0.9700678992717492

        def gate_sd_at(mu):
            r = np.array([
                row.lb6 - float(PR._design((row.cv - mu) * 1e6, row.fam, row["std"],
                                           row["corr"], row.era) @ PR.BETA)
                for _, row in _t.iterrows()])
            return float(np.sqrt(float(r @ r) / PR.DOF))

        sd_at_pin, sd_shifted = gate_sd_at(MU_PINNED), gate_sd_at(shifted)
        check("at the pinned MU the pricer reproduces its own fit",
              abs(sd_at_pin - PR.RESID_SD) < 1e-6, f"{sd_at_pin:.4f}e-6")
        # Observed on the real incident: 7.7200 -> 21.8591e-6, a factor of 2.83. The bar is
        # set at 2x -- far above any fitting noise, comfortably below the measured effect.
        check("the 08-22 MU shift BREAKS the fit (the pin is load-bearing)",
              sd_shifted > 2.0 * PR.RESID_SD,
              f"residual sd {sd_at_pin:.4f} -> {sd_shifted:.4f}e-6 "
              f"({sd_shifted / PR.RESID_SD:.1f}x)")

    print()
    if FAIL:
        print(f"*** {len(FAIL)} CHECK(S) FAILED: {FAIL}")
        print("If the pricer was re-centred ON PURPOSE, refit M2, regenerate w52b_cvlb93.csv's")
        print("cv6, re-derive w52b_joint/w52d_predlb, and move MU_PINNED — in the SAME commit.")
        raise SystemExit(1)
    print("MU pin holds, its artefacts agree, and the pin is load-bearing.")


if __name__ == "__main__":
    main()
