"""w53a — the M2 pricer, WIRED ONTO THE SEND PATH. Supersedes w46c_predlb for every caller.

w52 §8.4 left one registered task: `w26d_queueprice.predict()` still added `W46.ERA_SHIFT`
(M1, the flat era LEVEL dummy) and then ASSERTED that it reproduced w30b's residual sd, so
delegating it to M2 would trip that assert **inside `w48e_order.py`, on the critical send
path**. w52 deferred the rewire "to a run with slots". This run has ZERO slots (10/10 already
sent for the 08-22 UTC day), and that is the SAFER side of the trade, not the worse one: the
08-23 window opens in ~11h, and a rewire that breaks the ritual is discovered now, by a dry
run, instead of tomorrow with the window open. That is exactly the failure w51 §3 had to fix.

WHAT CHANGES. w46c's M1 adds a flat -29.82e-6 to every ad>=195 file. w47b's designed 08-22
experiment refuted that form on all three of its registered rules, and w52b's leave-one-day-out
comparison put M2 (era x cv interaction, no level term) ahead of it on the full 12-day LODO
(8.69 vs 9.59e-6) and on the era slice (8.77 vs 12.82e-6). The era files do not sit a fixed
distance below the line; they convert CV to LB at a SHALLOWER SLOPE — +1.443 per e-6 against
the base +1.833, i.e. 79% of it.

ONE DELIBERATE DIFFERENCE FROM w52d, AND IT IS NOT COSMETIC. w52d drops the single `wh3` row
from the fit; it had to, because w52b's LODO holds out whole days and holding out 08-15 left
family `wh3` with no rows at all. A PRICER has no such constraint — it never holds a day out —
and dropping the row does real damage here, because `_design` maps an unseen family onto the
h3 REFERENCE LEVEL silently. There is exactly one `wh3` file in the unsent queue, and w30b
prices fam[wh3] at -4.41e-6, so under w52d-as-pricer that file would quietly be handed
+4.41e-6 it has not earned. w53a therefore fits all 93 rows, keeps fam[wh3], and — the part
that matters more than the coefficient — ASSERTS that every family it is asked to price is in
the fitted design. A silent fallback to the reference level is how a pricer lies.

The w52d-vs-w53a coefficient difference is reported by `__main__` so it is on the record rather
than assumed small.

    .venv/bin/python experiments/w53a_pricer.py
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w46c_predlb as W46C   # noqa: E402  — for MU, ERA_MIN_AD and the stem classifiers only

# The classifiers are SHARED, not re-implemented. stdflag derives `is_std` from the wave number
# because a filename is not provenance (w28); `new_era`/`wave` come from w46c. Nothing about the
# era LEVEL term is imported — only the ad>=195 boundary, which M2 and M1 agree on.
MU = W46C.MU
ERA_MIN_AD = W46C.ERA_MIN_AD
new_era, wave, family, is_std, is_corr = (W46C.new_era, W46C.wave, W46C.family,
                                          W46C.is_std, W46C.is_corr)

# ---------------------------------------------------------------------------------- THE FIT
_t = pd.read_csv(os.path.join(HERE, "w52b_cvlb93.csv"))
assert len(_t) == 93 and _t.stem.is_unique, (len(_t), "w52b_cvlb93.csv is not the 93-row table")

FAMS = sorted(f for f in _t.fam.unique() if f != "h3")     # h3 is the reference level
NAMES = ["const", "cv6"] + [f"fam[{f}]" for f in FAMS] + ["std", "corr", "era:cv6"]


def _design(cv6, fam, std, corr, era):
    """One design row. M2: an era x cv INTERACTION and no era level term."""
    if fam not in FAMS and fam != "h3":
        raise KeyError(
            f"family {fam!r} is not in the fitted design {['h3'] + FAMS}. Refusing to price it: "
            f"an unknown family silently collapses onto the h3 reference level, which is a "
            f"price that looks real and is not. Add the file's family to the fit table first.")
    return np.array([1.0, cv6] + [1.0 if fam == f else 0.0 for f in FAMS]
                    + [float(std), float(corr), float(era) * cv6])


_X = np.vstack([_design(r.cv6, r.fam, r["std"], r["corr"], r.era)
                   for _, r in _t.iterrows()])
_y = _t.lb6.values
BETA, *_ = np.linalg.lstsq(_X, _y, rcond=None)
COEFS = dict(zip(NAMES, BETA))
_r = _y - _X @ BETA
DOF = len(_t) - _X.shape[1]
RESID_SD = float(np.sqrt(float(_r @ _r) / DOF))            # in-sample, dof-corrected

# ⚠ The predictive sd for an ERA file is NOT the in-sample residual sd. w52c measured M2's
# HELD-OUT error on the era slice under leave-one-day-out at 8.77e-6, and that is the honest
# number to hand to a probability. For pre-era files the in-sample sd is the better estimate
# (78 of the 93 rows are pre-era and none of them is an extrapolation).
SD_ERA = 8.77
SD_OLD = W46C.SD_OLD                                       # 7.756e-6, w30b's, valid for ad<=194

BASE_SLOPE = float(COEFS["cv6"])
ERA_SLOPE = float(COEFS["cv6"] + COEFS["era:cv6"])


def predict_lb(cv, stem):
    """Predicted public LB in absolute AUC, for a stem whose flags are derived, not passed."""
    return predict_flags(cv, family(stem), is_std(stem), is_corr(stem), new_era(stem))


def predict_flags(cv, fam, std, corr, era):
    """Same, for callers that hold the flags explicitly (w26d's legacy signature)."""
    return float(_design((cv - MU) * 1e6, fam, std, corr, era) @ BETA) * 1e-6


def pred_sd(stem):
    return (SD_ERA if new_era(stem) else SD_OLD) * 1e-6


def cv_needed(target_lb, stem):
    """The CV this stem's family/flags would need in order to predict `target_lb`.

    ⚠ Anything materially above the best era CV on record is an EXTRAPOLATION beyond the fitted
    range. Read it as 'the gap is this large in CV units', not as a promise."""
    intercept = float(_design(0.0, family(stem), is_std(stem), is_corr(stem), new_era(stem))
                      @ BETA)
    slope = ERA_SLOPE if new_era(stem) else BASE_SLOPE
    return MU + ((target_lb * 1e6 - intercept) / slope) * 1e-6


# --------------------------------------------------------------------------------- THE GATE
# w26d has carried a self-check since w25f and it has caught two real parameterisation slips
# (raw-vs-centred CV, ddof). It must MOVE to M2, not be deleted — w52 §8's new standing warning
# exists because deleting it is the tempting shortcut. Re-predict the rows this module was
# fitted on through the PUBLIC entry point and require the fitted residual sd back.
_gate_r6 = np.array([_t.lb6.values[i]
                     - predict_flags(r.cv, r.fam, r["std"], r["corr"], r.era) * 1e6
                     for i, (_, r) in enumerate(_t.iterrows())])
GATE_SD = float(np.sqrt(float(_gate_r6 @ _gate_r6) / DOF))
assert abs(GATE_SD - RESID_SD) < 1e-6, (
    f"predict_flags() does not reproduce the M2 fit ({GATE_SD:.4f} vs {RESID_SD:.4f}e-6); "
    f"nothing priced through this module is readable")

# A second, independent gate: the WANTED deadline pick is the one row whose true LB this
# workspace cares about, and w52d's own sanity check was that it prices to 0.971181 against an
# actual 0.97118. If a refit ever moves that, the pricer has changed under the decision.
WANTED_STEM, WANTED_CV, WANTED_LB = "w36_ad199stdcorr", 0.9701400060, 0.97118
_w = predict_lb(WANTED_CV, WANTED_STEM)
assert abs(_w - WANTED_LB) < 5e-6, (_w, "M2 no longer reproduces the WANTED pick's actual LB")


if __name__ == "__main__":
    print(f"w53a M2 pricer — n={len(_t)} (w52d used {len(_t) - 1}, dropping the wh3 row)")
    print(f"  in-sample resid sd {RESID_SD:.3f}e-6 (dof {DOF})   held-out era sd {SD_ERA:.2f}e-6"
          f"   pre-era sd {SD_OLD:.2f}e-6")
    print(f"  base cv slope {BASE_SLOPE:+.4f}/e-6   era cv slope {ERA_SLOPE:+.4f} "
          f"({100 * ERA_SLOPE / BASE_SLOPE:.1f}% of base)")
    print(f"  GATE reproduce-own-fit: {GATE_SD:.4f} vs {RESID_SD:.4f}e-6  PASS")
    print(f"  GATE {WANTED_STEM}: pred {_w:.6f} vs actual {WANTED_LB:.5f}  PASS")

    w52d = json.load(open(os.path.join(HERE, "w52d_predlb.json")))["coefs"]
    print(f"\n  COEFFICIENTS, w53a (93 rows) vs w52d (92, no wh3) — the cost of keeping the row")
    print(f"  {'term':14s} {'w53a':>12s} {'w52d':>12s} {'delta e-6':>11s}")
    for k in NAMES:
        a, b = COEFS[k], w52d.get(k)
        d = "n/a" if b is None else f"{a - b:+11.4f}"
        print(f"  {k:14s} {a:12.4f} {'—' if b is None else f'{b:12.4f}'} {d:>11s}")

    # What the rewire actually does to the queue, file by file. This is the number w52 §4
    # asserted was small (w46b priced the whole free-rider ordering at +0.47e-6) without
    # re-measuring it under M2. Measure it.
    import stdflag                                                            # noqa: E402
    q = pd.read_csv(os.path.join(HERE, "w23b_sendqueue.csv"))
    q = q[~q.sent].dropna(subset=["cv"]).copy()
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    q["fam"] = q.stem.map(family)
    q["m1"] = [W46C.predict_lb(r.cv, r.stem) for r in q.itertuples()]
    q["m2"] = [predict_lb(r.cv, r.stem) for r in q.itertuples()]
    q["d6"] = (q.m2 - q.m1) * 1e6
    era = q[q.stem.map(new_era)]
    print(f"\n  REPRICE of the {len(q)} unsent files: {len(era)} are era (ad>={ERA_MIN_AD})")
    print(f"    era files move {era.d6.min():+.2f} .. {era.d6.max():+.2f}e-6 "
          f"(mean {era.d6.mean():+.2f})")
    print(f"    pre-era files move {q[~q.stem.map(new_era)].d6.abs().max():.2f}e-6 at most")
    r1 = q.sort_values("m1", ascending=False).stem.tolist()
    r2 = q.sort_values("m2", ascending=False).stem.tolist()
    moved = sum(a != b for a, b in zip(r1, r2))
    print(f"    ORDERING: {moved} of {len(q)} positions change under the rewire; "
          f"top file {'UNCHANGED' if r1[0] == r2[0] else f'{r1[0]} -> {r2[0]}'}")
    print(f"    top 5 by M1: {r1[:5]}")
    print(f"    top 5 by M2: {r2[:5]}")

    json.dump(dict(n=len(_t), coefs=COEFS, resid_sd=RESID_SD, dof=DOF, sd_era=SD_ERA,
                   sd_old=SD_OLD, base_slope=BASE_SLOPE, era_slope=ERA_SLOPE, mu=MU,
                   supersedes="w46c_predlb (M1 level dummy); w52d (same M2 minus the wh3 row)",
                   wanted=dict(stem=WANTED_STEM, cv=WANTED_CV, pred=_w, actual=WANTED_LB),
                   reprice=dict(n_unsent=len(q), n_era=len(era),
                                era_mean_e6=float(era.d6.mean()),
                                era_min_e6=float(era.d6.min()),
                                era_max_e6=float(era.d6.max()),
                                positions_moved=int(moved), top_m1=r1[0], top_m2=r2[0])),
              open(os.path.join(HERE, "w53a_pricer.json"), "w"), indent=1)
    print(f"\nwrote {os.path.join(HERE, 'w53a_pricer.json')}")
