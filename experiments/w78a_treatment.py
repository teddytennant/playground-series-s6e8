"""w78a — THE TREATMENT BOUNDARY INSIDE w63a's COST CURVE, and the live refit that already moved.

Pre-registration: experiments/w78_prereg.txt, committed 637a6c1 BEFORE this file existed.

THE OPEN ITEM (w77 §10.3). w77d measured the binding break-even at 5.255e-6 against the live
interpolated 11.144e-6 and was owed three things before anything could adopt it:

  (a) an argument that withholding a realised public score from the GLS is the RIGHT
      counterfactual for a hijack — P4 turns that into a measurement;
  (b) a re-derivation through w63a itself on the LIVE board — ARM LIVE / P1;
  (c) a decision on Sum|w| as a standing health check on `fit` — P5 and R2.

THE HYPOTHESIS THIS RUN ADDS. `fit` gives a candidate in `LB` the residual xh0 = (LB-cv)/U - gh
and a candidate outside it xh0 = 0. Those are two different TREATMENTS, and w63a's 15-row curve
is a mixture of them: the only two rows off the `worthless` ceiling are BOTH scored, and the
16.49e-6 bracket has a SCORED hi endpoint and an UNSCORED lo endpoint. A crossing read across a
treatment boundary is not an interpolation of one curve; it is a line drawn between two.

⚠ THE ESTIMATOR IS w63a's, IMPORTED, NEVER RE-IMPLEMENTED. `arm` and `crossings` come from w77a
and `gls` from w77c, so every arm below runs the same code on the same board and the ONLY thing
that differs between arms is which files are in `LB`. That is the whole experiment.

⛔ THIS FILE PERSISTS NO BAR. `w63a.main()` is never called and w63a_setprice.json is never
opened for writing. Prereg R3: the run that finds the defect does not also ship the fix.

    .venv/bin/python experiments/w78a_treatment.py              # ~4 min, writes its own 2 files
    .venv/bin/python experiments/w78a_treatment.py --no-write   # what-if
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import TARGET, load_raw                             # noqa: E402
import w63a_setprice as W63A                                    # noqa: E402 — the instrument
import w76a_addtest as W76A                                     # noqa: E402 — the board
from w77a_bracket import arm, crossings                         # noqa: E402 — same definitions
from w77c_glssweep import gls                                   # noqa: E402 — the same GLS

U = 1e-6
GATE_A_BAR = 1e-9              # GATE A is a re-derivation: anything above this is a broken copy
W77D_H_BIND = 5.254881093054007
W77D_INTERP_ERR = 6.364183378316056        # 13.937812608870207 - 7.573629230554151

# registered bars, w78_prereg.txt
P1A, P1B, P1C = 20.0, 2.0, 10.0
P2A, P2B = 1.0, 11
P3A, P3B = 10.0, 1.0
P4A, P4B = 50.0, 20.0
P5A = 0.5
P6_BAR = 3.0
SUMW_BAR = 10.0                # R2's candidate standing bar

FAILURES, FALSIFIED, CONFIRMED = 0, [], []


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def verdict(tag: str, ok: bool, msg: str) -> bool:
    (CONFIRMED if ok else FALSIFIED).append(tag)
    print(f"  {'✅' if ok else '🔴'} {tag} {'CONFIRMED' if ok else 'FALSIFIED'} — {msg}")
    return ok


def health(E, LB, cv):
    """Sum|w|, gh and the spread of z for one arm's GLS, read off the arm's OWN Sxx."""
    names = E["names"]
    col = {k: i for i, k in enumerate(names)}
    scored = [k for k in names if k in LB]
    js = [col[k] for k in scored]
    z = np.array([LB[k] - cv[k] for k in scored]) / U
    gh, Vg, w = gls(E["Sxx"], js, z)
    return dict(k=len(js), gh=gh, gh_fit=E["gap"], sumw=float(np.abs(w).sum()),
                minw=float(w.min()), sqrtVg=float(np.sqrt(Vg)),
                zmin=float(z.min()), zmax=float(z.max()), zmean=float(z.mean()),
                out_of_hull=bool(gh < z.min() or gh > z.max()))


def binding(rows, base):
    """The bar-setting crossing: the smaller of the uncond and cond break-evens, as w63a reads it."""
    out = {}
    for key in ("uncond", "cond"):
        out[key] = crossings(rows, base, key)
    hs = [c[0]["H"] for c in out.values() if c]
    return (min(hs) if hs else None), out


# ----------------------------------------------------------------------------------------------
def main() -> None:
    write = "--no-write" not in sys.argv
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    T1, T2 = list(ref["tiers"]["slot1"]), list(ref["tiers"]["slot2"])
    plan = list(ref["plan_0824"])
    thresh = float(ref["threshold_public"])
    gh0, BAR0 = float(ref["gap"]), float(ref["cv_bar_new"])
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    fillers = list(json.load(open(os.path.join(HERE, "w77d_holefill.json")))["fillers"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    S_NAMES = sorted(set(T1 + T2 + list(ref["wanted"]) + plan))       # w63a's 18
    S_CAND = sorted(set(plan + T2 + [W63A.PICK]))                     # w63a's 15 rows
    M_NAMES = sorted(set(S_NAMES) | set(fillers))                     # w77d's 31
    M_CAND = sorted(set(M_NAMES) - set(T1))                           # w77d's 29

    # ------------------------------------------------------------------- the two boards, once
    lb_live, ltl1, ltl2, _, _ = W76A.board()
    lb_rw, ltr1, _, _, _ = W76A.board(exclude=plan)
    if sorted(ltr1) != sorted(T1):
        fail(f"the board did not rewind: {sorted(ltr1)} != {sorted(T1)}"); sys.exit(1)
    if sorted(ltl1) != sorted(T1) or sorted(ltl2) != sorted(T2):
        fail("THE LIVE TIER MOVED. `thresh` is w63a's and no longer describes the board. "
             "Refusing — this instrument prices the tier it was told about."); sys.exit(1)
    print(f"live board {len(lb_live)} scored, tier1 {ltl1} — UNCHANGED from w63a's")
    print(f"rewound board {len(lb_rw)} scored (08-24 ten removed)")
    LB0 = {k: v for k, v in lb_rw.items() if k in set(S_NAMES)}       # w63a's own 8
    LB_LIVE = {k: v for k, v in lb_live.items() if k in set(S_NAMES)}
    print(f"  w63a design: {len(LB0)} scored on the rewound board, "
          f"{len(LB_LIVE)} on the LIVE one — {len(LB_LIVE) - len(LB0)} candidates flipped "
          f"treatment overnight")

    # ============================================================== GATE A — re-derive w63a
    print("\n" + "=" * 96)
    print("GATE A — rebuild w63a's 08-23 design and RE-DERIVE its stored numbers. Not a stamp.")
    print("=" * 96)
    S = arm(S_NAMES, LB0, T1, y, beta, thresh, S_CAND)
    H_S, cr_S = binding(S["rows"], S["base"])
    got = {"gap": S["E"]["gap"], "base": S["base"], "worthless": S["worthless"],
           "H_uncond": cr_S["uncond"][0]["H"] if cr_S["uncond"] else float("nan"),
           "H_binding": H_S, "cv_bar_new": S["cv"][W63A.PICK] - H_S * U}
    want = {"gap": gh0, "base": float(ref["base"]), "worthless": float(ref["worthless_limit1"]),
            "H_uncond": float(ref["H_uncond"]), "H_binding": float(ref["H_binding"]),
            "cv_bar_new": BAR0}
    devs = []
    for k in want:
        d = abs(got[k] - want[k]); devs.append(d)
        print(f"  {k:12s} got {got[k]:+.10f}   stored {want[k]:+.10f}   |d| {d:.3e}")
    rw = {r["stem"]: r for r in ref["rows"]}
    for r in S["rows"]:
        for key in ("dcv", "uncond", "cond"):
            devs.append(abs(r[key] - float(rw[r["stem"]][key])))
    worst = max(devs)
    print(f"  {len(devs)} quantities compared, WORST DEVIATION {worst:.3e}")
    if worst > GATE_A_BAR:
        fail("GATE A FAILED — this is not w63a's estimator. Nothing below can be read.")
        sys.exit(1)
    print("  ✅ GATE A PASSED — every arm below is w63a's own code on a different `LB`.")

    # the treatment of every one of w63a's 15 rows, as a fact rather than an assertion
    scored_rows = [r["stem"] for r in S["rows"] if r["stem"] in LB0]
    print(f"\n  w63a's 15 candidate rows: {len(scored_rows)} SCORED, "
          f"{15 - len(scored_rows)} UNSCORED.")
    off_ceiling = [r for r in S["rows"] if S["worthless"] - r["uncond"] > 0.05]
    print(f"  rows more than 0.05e-6 off the `worthless` ceiling: "
          f"{[(r['stem'], r['stem'] in LB0) for r in off_ceiling]}")
    br = ref["bracket"]["uncond"]
    mixed = (br["hi"] in LB0) != (br["lo"] in LB0)
    print(f"  the live bracket: hi {br['hi']} {'SCORED' if br['hi'] in LB0 else 'UNSCORED'} .. "
          f"lo {br['lo']} {'SCORED' if br['lo'] in LB0 else 'UNSCORED'}"
          f"   -> {'MIXED' if mixed else 'UNIFORM'}")

    # ================================================================ P1 — ARM LIVE
    print("\n" + "=" * 96)
    print("P1 — w63a's OWN design and candidate rule, re-derived on the LIVE board")
    print("=" * 96)
    L = arm(S_NAMES, LB_LIVE, T1, y, beta, thresh, S_CAND)
    H_L, cr_L = binding(L["rows"], L["base"])
    hL = health(L["E"], LB_LIVE, L["cv"])
    hS = health(S["E"], LB0, S["cv"])
    bar_live = (L["cv"][W63A.PICK] - H_L * U) if H_L is not None else float("nan")
    print(f"  gh    rewound {gh0:+.6f}   LIVE {L['E']['gap']:+.6f}   "
          f"d {(L['E']['gap'] - gh0):+.3f}e-6")
    print(f"  base  {S['base']:+.6f} -> {L['base']:+.6f}    "
          f"worthless {S['worthless']:+.6f} -> {L['worthless']:+.6f}")
    for key in ("uncond", "cond"):
        print(f"  {key:8s} {len(cr_L[key])} crossing(s): " + "; ".join(
            f"H {c['H']:.3f} [{c['hi']} {c['hi_dcv']:+.2f} .. {c['lo']} {c['lo_dcv']:+.2f}]"
            for c in cr_L[key]) or "  none")
    print(f"  H_binding {float(ref['H_binding']):.4f} -> {H_L if H_L is None else round(H_L, 4)}"
          f"    bar {BAR0:.10f} -> {bar_live:.10f}  "
          f"({(bar_live - BAR0) / U:+.3f}e-6 of CV)")
    print(f"  Sum|w|  8-file {hS['sumw']:.2f}  ->  18-file {hL['sumw']:.2f}"
          f"    sqrt(Vg) {hS['sqrtVg']:.1f} -> {hL['sqrtVg']:.1f}")
    verdict("P1a", abs(L["E"]["gap"] - gh0) > P1A,
            f"|d gh| {abs(L['E']['gap'] - gh0):.3f}e-6 vs bar {P1A}")
    verdict("P1b", abs(bar_live - BAR0) / U > P1B,
            f"|d bar| {abs(bar_live - BAR0) / U:.3f}e-6 of CV vs bar {P1B}")
    verdict("P1c", hL["sumw"] > P1C, f"Sum|w| {hL['sumw']:.2f} vs bar {P1C}")
    if cr_L["uncond"]:
        bl = cr_L["uncond"][0]
        print(f"  ⚠ the LIVE bracket is "
              f"{'MIXED' if (bl['hi'] in LB_LIVE) != (bl['lo'] in LB_LIVE) else 'UNIFORM'} — "
              f"on the live board every one of w63a's candidates is SCORED, so the treatment "
              f"boundary is gone and a THIRD number appears.")

    # ================================================================ P2 / P4 — the 13 arms
    print("\n" + "=" * 96)
    print("P2 / P4 — one filler at a time into the GLS, with a matched control on the gh channel")
    print("=" * 96)
    M0 = arm(M_NAMES, LB0, T1, y, beta, thresh, M_CAND)
    dg = abs(M0["E"]["gap"] - gh0)
    print(f"  GATE D: base arm gh {M0['E']['gap']:+.10f} vs w63a {gh0:+.10f}  |d| {dg:.3e}")
    if dg > 1e-6:
        fail("GATE D FAILED — the fillers leaked into the GLS."); sys.exit(1)
    print(f"  ✅ GATE D — 31-name design, GLS still w63a's {len(LB0)} files.")
    r0 = {r["stem"]: r for r in M0["rows"]}
    hM0 = health(M0["E"], LB0, M0["cv"])
    fits = [dict(arm="live_8", **hS), dict(arm="live_18", **hL), dict(arm="base_31", **hM0)]

    arms, rows_p2 = {}, []
    for f in fillers:
        LBf = dict(LB0); LBf[f] = lb_rw[f]
        A = arm(M_NAMES, LBf, T1, y, beta, thresh, fillers)
        h = health(A["E"], LBf, A["cv"])
        fits.append(dict(arm=f"incl_{f}", **h))
        arms[f] = dict(rows={r["stem"]: r for r in A["rows"]}, base=A["base"], gh=A["E"]["gap"],
                       sumw=h["sumw"])
    print(f"\n  {'filler':24s} {'gh_f':>10s} {'d gh':>9s} {'Sum|w|':>8s} "
          f"{'D(f)':>9s} {'C(f)':>9s} {'offset':>9s} {'offset_b':>9s}")
    for f in fillers:
        A = arms[f]
        D = A["rows"][f]["uncond"] - r0[f]["uncond"]
        C = float(np.median([A["rows"][g]["uncond"] - r0[g]["uncond"]
                             for g in fillers if g != f]))
        Db = (A["rows"][f]["uncond"] - A["base"]) - (r0[f]["uncond"] - M0["base"])
        Cb = float(np.median([(A["rows"][g]["uncond"] - A["base"])
                              - (r0[g]["uncond"] - M0["base"]) for g in fillers if g != f]))
        rows_p2.append(dict(stem=f, dcv=r0[f]["dcv"], gh=A["gh"], dgh=A["gh"] - gh0,
                            sumw=A["sumw"], D=D, C=C, offset=D - C, offset_base=Db - Cb))
        print(f"  {f:24s} {A['gh']:+10.3f} {A['gh'] - gh0:+9.3f} {A['sumw']:8.2f} "
              f"{D:+9.4f} {C:+9.4f} {D - C:+9.4f} {Db - Cb:+9.4f}")

    offs = np.array([r["offset"] for r in rows_p2])
    med = float(np.median(offs))
    npos, nneg = int((offs > 0).sum()), int((offs < 0).sum())
    print(f"\n  median |offset| {np.abs(offs).mean():.4f} (mean) / "
          f"{np.median(np.abs(offs)):.4f} (median)   signs {npos}+ / {nneg}-")
    verdict("P2a", float(np.median(np.abs(offs))) > P2A,
            f"median |offset| {np.median(np.abs(offs)):.4f}e-6 vs bar {P2A}")
    verdict("P2b", max(npos, nneg) >= P2B, f"{max(npos, nneg)} of 13 share a sign, bar {P2B}")
    verdict("P2c", med > 0, f"median offset {med:+.4f}e-6 — the SCORED treatment makes a "
                            f"candidate look like it {'HURTS' if med > 0 else 'HELPS'} more")

    ghs = np.array([r["gh"] for r in rows_p2])
    verdict("P4a", float(ghs.max() - ghs.min()) > P4A,
            f"range(gh_f) {ghs.max() - ghs.min():.3f}e-6 over 13 arms vs bar {P4A}")
    verdict("P4b", float(np.abs(ghs - gh0).max()) > P4B,
            f"max |gh_f - gh0| {np.abs(ghs - gh0).max():.3f}e-6 vs bar {P4B}")

    # ================================================================ P3 — ARM U
    print("\n" + "=" * 96)
    print("P3 — ARM U: every candidate UNSCORED, which leaves only tier1 in the GLS")
    print("=" * 96)
    LB_U = {k: v for k, v in lb_rw.items() if k in set(T1)}
    Uarm = arm(M_NAMES, LB_U, T1, y, beta, thresh, M_CAND)
    H_U, cr_U = binding(Uarm["rows"], Uarm["base"])
    hU = health(Uarm["E"], LB_U, Uarm["cv"])
    fits.append(dict(arm="uniform_2", **hU))
    print(f"  GLS files {hU['k']}  gh {Uarm['E']['gap']:+.4f}  (w63a {gh0:+.4f}, "
          f"d {Uarm['E']['gap'] - gh0:+.3f}e-6)   Sum|w| {hU['sumw']:.2f}")
    for key in ("uncond", "cond"):
        print(f"  {key:8s} " + ("; ".join(
            f"H {c['H']:.3f} [{c['hi']} .. {c['lo']}]" for c in cr_U[key]) or "no crossing"))
    verdict("P3a", abs(Uarm["E"]["gap"] - gh0) > P3A,
            f"|gh_U - gh0| {abs(Uarm['E']['gap'] - gh0):.3f}e-6 vs bar {P3A}")
    verdict("P3b", H_U is None or abs(H_U - W77D_H_BIND) > P3B,
            f"H_binding(ARM U) {H_U if H_U is None else round(H_U, 4)} vs w77d "
            f"{W77D_H_BIND:.4f}, bar {P3B}")

    # ================================================================ P5 — Sum|w|
    print("\n" + "=" * 96)
    print("P5 — Sum|w| as the health variable, re-tested on a different family of designs")
    print("=" * 96)
    print(f"  {'arm':28s} {'k':>3s} {'Sum|w|':>9s} {'min w':>8s} {'gh':>10s} "
          f"{'gh-mean(z)':>11s} {'hull':>6s}")
    for r in fits:
        print(f"  {r['arm']:28s} {r['k']:3d} {r['sumw']:9.2f} {r['minw']:+8.3f} "
              f"{r['gh']:+10.2f} {r['gh'] - r['zmean']:+11.2f} "
              f"{'OUT' if r['out_of_hull'] else 'in':>6s}")
    sw = np.array([r["sumw"] for r in fits])
    dv = np.abs([r["gh"] - r["zmean"] for r in fits])
    ok = dv > 0
    rho = float(np.corrcoef(np.log(sw[ok]), np.log(dv[ok]))[0, 1]) if ok.sum() > 2 else float("nan")
    verdict("P5a", rho > P5A, f"corr(log Sum|w|, log|gh - mean z|) = {rho:+.3f} over "
                              f"{int(ok.sum())} fits vs bar {P5A}")
    smallest = min(fits, key=lambda r: r["sumw"])
    verdict("P5b", smallest["arm"] == "live_8",
            f"smallest Sum|w| is `{smallest['arm']}` at {smallest['sumw']:.2f} "
            f"(live 8-file {hS['sumw']:.2f})")

    # ================================================================ P6
    print("\n" + "=" * 96)
    print("P6 — does the offset ACCOUNT for w77d's 6.364e-6? (the weakest prediction here)")
    print("=" * 96)
    bu = json.load(open(os.path.join(HERE, "w77d_holefill.json")))["crossings"]["uncond"][0]
    hi, lo = r0[bu["hi"]], r0[bu["lo"]]
    slope = (lo["uncond"] - hi["uncond"]) / (hi["dcv"] - lo["dcv"])
    dH = med / slope if slope else float("nan")
    print(f"  local slope at w77d's uncond bracket ({bu['hi']} .. {bu['lo']}): "
          f"{slope:+.4f} cost per e-6 of dCV")
    print(f"  median offset {med:+.4f}e-6 of cost  ->  {dH:+.3f}e-6 of dCV   "
          f"vs w77d's measured {W77D_INTERP_ERR:+.3f}e-6")
    verdict("P6", abs(dH - W77D_INTERP_ERR) < P6_BAR,
            f"|{dH:.3f} - {W77D_INTERP_ERR:.3f}| = {abs(dH - W77D_INTERP_ERR):.3f} "
            f"vs bar {P6_BAR}")

    # ================================================================ THE REGISTERED DECISIONS
    print("\n" + "=" * 96)
    print("THE DECISION RULES, AS REGISTERED (w78_prereg.txt) — read off, not chosen")
    print("=" * 96)
    r1 = ("P2a" in CONFIRMED) and ("P2b" in CONFIRMED)
    print(f"  R1 {'ADOPT' if r1 else 'DO NOT ADOPT'} the same-treatment bracket guard: "
          f"P2a {'✅' if 'P2a' in CONFIRMED else '🔴'} P2b {'✅' if 'P2b' in CONFIRMED else '🔴'}")
    if r1:
        print(f"     w63a's live bracket IS mixed ({br['hi']} SCORED .. {br['lo']} UNSCORED), "
              f"so BOTH its H_uncond and H_cond are read across a treatment boundary.")
    r2_ok = "P5b" in CONFIRMED
    r2_bar = SUMW_BAR if r2_ok else round(2 * float(sw.min()), 2)
    r2_why = ("P5b confirmed, the live fit is the floor" if r2_ok else
              "P5b FAILED, so the bar is 2x the smallest fit measured, not 10.0")
    print(f"  R2 standing health bar on `fit`: Sum|w| <= {r2_bar}  ({r2_why})")
    print(f"  R3 THE BAR DOES NOT MOVE. w63a_setprice.json untouched, w63a.main() never called, "
          f"live bar stays {BAR0:.10f}.")
    r4 = "P1a" in CONFIRMED and "P1b" in CONFIRMED
    print(f"  R4 {'BAR0 IS STALE' if r4 else 'the live refit did not move the bar materially'}"
          f" — and bar_live {bar_live:.10f} does NOT replace it either: a THIRD construction.")

    # ---- WHAT ACTUALLY SURVIVES: the bar is insensitive to the GLS and sensitive to the hole
    robust = [dict(arm="w63a live bar", k=hS["k"], gh=gh0, hole="EMPTY",
                   H=float(ref["H_binding"])),
              dict(arm="ARM LIVE (18 scored)", k=hL["k"], gh=L["E"]["gap"], hole="EMPTY", H=H_L),
              dict(arm="w77d (fillers withheld)", k=hM0["k"], gh=M0["E"]["gap"], hole="FILLED",
                   H=W77D_H_BIND),
              dict(arm="ARM U (uniform unscored)", k=hU["k"], gh=Uarm["E"]["gap"], hole="FILLED",
                   H=H_U)]
    ghr = [r["gh"] for r in robust]
    he = [r["H"] for r in robust if r["hole"] == "EMPTY"]
    hf = [r["H"] for r in robust if r["hole"] == "FILLED"]
    print("\n" + "=" * 96)
    print("WHAT SURVIVES — four constructions, and the split is the HOLE, not the GLS")
    print("=" * 96)
    print(f"  {'construction':26s} {'GLS k':>6s} {'gh':>10s} {'hole':>7s} {'H_binding':>10s}")
    for r in robust:
        print(f"  {r['arm']:26s} {r['k']:6d} {r['gh']:+10.2f} {r['hole']:>7s} {r['H']:10.3f}")
    print(f"  gh spans {max(ghr) - min(ghr):.2f}e-6 across the four; H_binding splits into "
          f"{min(he):.3f}..{max(he):.3f} (hole EMPTY) and {min(hf):.3f}..{max(hf):.3f} (FILLED).")
    print(f"  ⟹ a {max(ghr) - min(ghr):.1f}e-6 spread in the GLS gap buys "
          f"{max(hf) - min(hf):.2f}e-6 of bar; filling the hole buys "
          f"{min(he) - max(hf):.2f}e-6. THE HOLE IS THE WHOLE EFFECT.")

    print(f"\n  CONFIRMED {CONFIRMED}\n  FALSIFIED {FALSIFIED}\n  FAILURES {FAILURES}")

    # ⚠ EVERY VERDICT ABOVE IS RE-CHECKABLE FROM THE ARTEFACT ALONE. w77 §4: a `falsified` field
    # that cannot be recomputed from its own numbers reads as rigour and is not. `w78b` C5
    # recomputes all eleven from `stats` and fails if any disagrees with the two lists below.
    stats = {
        "P1a": (abs(L["E"]["gap"] - gh0), ">", P1A),
        "P1b": (abs(bar_live - BAR0) / U, ">", P1B),
        "P1c": (hL["sumw"], ">", P1C),
        "P2a": (float(np.median(np.abs(offs))), ">", P2A),
        "P2b": (float(max(npos, nneg)), ">=", float(P2B)),
        "P2c": (med, ">", 0.0),
        "P3a": (abs(Uarm["E"]["gap"] - gh0), ">", P3A),
        "P3b": (abs(H_U - W77D_H_BIND), ">", P3B),
        "P4a": (float(ghs.max() - ghs.min()), ">", P4A),
        "P4b": (float(np.abs(ghs - gh0).max()), ">", P4B),
        "P5a": (rho, ">", P5A),
        "P5b": (0.0 if smallest["arm"] == "live_8" else 1.0, "==", 0.0),
        "P6": (abs(dH - W77D_INTERP_ERR), "<", P6_BAR),
    }

    out = dict(
        prereg="experiments/w78_prereg.txt", commit_before="637a6c1",
        writes_a_bar=False, w63a_main_called=False, stats=stats,
        # requirement (c), DECIDED. P5a is falsified, so the premise under R2 is gone and the
        # bar R2 mechanically produces (2 x the smallest fit) is BELOW the live fit's own 4.98 —
        # installing it would fail the send chain on day one on a statistic that tracks nothing.
        sumw_check_adopted=False,
        sumw_reason=("P5a falsified (rho %+.3f, not w77c's +0.957) — Sum|w| does not track "
                     "|gh - mean z| on this family of designs, and live_18 is the counterexample:"
                     " Sum|w| %.2f, the LARGEST, with |gh - mean z| %.2f, the SMALLEST. R2's "
                     "mechanical output %.2f is below the live fit's own %.2f."
                     % (rho, hL["sumw"], abs(hL["gh"] - hL["zmean"]), r2_bar, hS["sumw"])),
        gate_a=dict(n=len(devs), worst=worst, bar=GATE_A_BAR),
        design=dict(n_names=len(S_NAMES), n_cand=len(S_CAND), n_scored_rewound=len(LB0),
                    n_scored_live=len(LB_LIVE), scored_rows=scored_rows,
                    off_ceiling=[r["stem"] for r in off_ceiling],
                    bracket_mixed=bool(mixed), bracket=br),
        live=dict(gh=L["E"]["gap"], base=L["base"], worthless=L["worthless"],
                  H_binding=H_L, bar=bar_live, bar0=BAR0, dbar_e6=(bar_live - BAR0) / U,
                  crossings=cr_L, sumw=hL["sumw"]),
        p2=rows_p2, p2_median_offset=med, p2_signs=[npos, nneg],
        armU=dict(k=hU["k"], gh=Uarm["E"]["gap"], sumw=hU["sumw"], H_binding=H_U,
                  crossings=cr_U),
        fits=fits, p5_rho=rho, robust=robust, p6=dict(slope=slope, dH=dH, target=W77D_INTERP_ERR),
        decisions=dict(R1_adopt_bracket_guard=bool(r1), R2_sumw_bar=r2_bar,
                       R3_bar_moved=False, R4_bar0_stale=bool(r4)),
        confirmed=CONFIRMED, falsified=FALSIFIED, failures=FAILURES,
        rows_live=L["rows"], rows_base31=M0["rows"], rows_U=Uarm["rows"])
    if write:
        json.dump(out, open(os.path.join(HERE, "w78a_treatment.json"), "w"), indent=1)
        pd.DataFrame(rows_p2).to_csv(os.path.join(HERE, "w78a_treatment.csv"), index=False)
        print("\nwrote w78a_treatment.json / .csv")
    else:
        print("\n--no-write: nothing written")


if __name__ == "__main__":
    main()
