"""w77a — THE 16.5e-6 BRACKET HOLE, tested by ENLARGING THE DESIGN rather than by choosing files.

Pre-registration: experiments/w77_prereg.txt, committed 39e4aaf BEFORE this file existed.

THE OPEN ITEM (w63 §5a, carried forward by w64..w76 §7.5):

    w40_ad211stdcorr        dCV  -2.5327   only() +2.2764   HELPS   (< base +4.5228)
    .............. 16.4916e-6 of dCV WITH NO CANDIDATE IN IT ..............
    w40_ad211std_rescale    dCV -19.0243   only() +5.5246   HURTS

`H_uncond = 13.9378` is a straight line drawn across that hole and `H_cond_predsd = 11.1443`
sets the sender's live CV bar. w63 §10.5 named the fix and forbade the cheap version of it:
price against EVERY scored file on the board — a set nobody chose — and check first whether
the estimator survives a design that big.

⚠ THE ESTIMATOR IS IMPORTED FROM w63a, NEVER RE-IMPLEMENTED, and the board reconstruction is
imported from w76a. Both arms therefore run the SAME code on the SAME board with the SAME
tier, and the only thing that differs between them is HOW MANY FILES ARE IN THE DESIGN. That
is the whole experiment: everything else is held fixed so any movement is attributable.

⛔ THIS FILE PERSISTS NO BAR. w59a's rule is that a run may not enlarge the bracket and persist
a bar derived from it in the same run — that is choosing the bar by choosing the bracket. This
run enlarges the bracket, so it writes ONLY w77a_bracket.{json,csv} and never w63a_setprice.json,
the queue, the plan, the sender or a submission. `w63a.main()` is never called.

    .venv/bin/python experiments/w77a_bracket.py              # ~8 min, writes its own 2 files
    .venv/bin/python experiments/w77a_bracket.py --no-write   # what-if
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import SUB, TARGET, load_raw                       # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402 — the instrument
import w76a_addtest as W76A                                    # noqa: E402 — the board rewind

U = 1e-6
TIE = 1e-9              # prereg resolution bar, in e-6 units
COND_BAR = 1e12         # prereg P2: cond(Sxx) must stay under this
ONLY_BAR = 1e4          # prereg P2: |only(x)| must stay under this
HOLE_MIN = 3            # prereg P3
GATE_R_BAR = 1.0e-6     # w63a's own GATE R tolerance, in e-6 units
P7_BAR = 5.0            # prereg P7, in e-6

FAILURES, FALSIFIED = 0, []


def fail(msg: str) -> None:
    global FAILURES
    FAILURES += 1
    print(f"  *** FAILURE: {msg}")


def falsify(tag: str, msg: str) -> None:
    FALSIFIED.append(tag)
    print(f"  🔴 {tag} FALSIFIED — {msg}")


# ----------------------------------------------------------------------------------------------
# The three quantities, written once and used by both arms. These are w63a's own definitions
# (w63a_setprice.py:323 `worthless`, :330 `only`, :393 `crossings`). GATE A is what proves it.
# ----------------------------------------------------------------------------------------------
def arm(NAMES, LB, TIER1, y, beta, thresh, cands):
    """Fit the estimator on `NAMES` and read off base, worthless, and the per-candidate cost."""
    V, cv = W63A._load(NAMES, y)
    E = W63A.fit(NAMES, LB, cv, V, y, beta)
    price = E["price"]
    base = price(sorted(TIER1))
    worthless = float(np.mean([price([d]) for d in TIER1]))

    def only(x, pfun=None):
        p = pfun or price
        return float(np.mean([p([x, d]) for d in TIER1 if d != x]))

    rows = []
    for x in sorted(cands, key=lambda k: -cv[k]):
        rows.append(dict(stem=x, cv=cv[x], dcv=(cv[x] - cv[W63A.PICK]) / U,
                         uncond=only(x), cond=only(x, E["cond_price"]([x], thresh))))
    return dict(E=E, cv=cv, base=base, worthless=worthless, rows=rows, only=only, price=price)


def crossings(rows, base, key):
    """w63a's method VERBATIM: adjacent brackets only, every further crossing reported."""
    rs = sorted(rows, key=lambda r: -r["dcv"])
    out = []
    for a, b in zip(rs, rs[1:]):
        if (a[key] < base) != (b[key] < base) and a[key] != b[key]:
            t = (base - a[key]) / (b[key] - a[key])
            out.append(dict(H=-(a["dcv"] + t * (b["dcv"] - a["dcv"])),
                            hi=a["stem"], hi_dcv=a["dcv"], lo=b["stem"], lo_dcv=b["dcv"],
                            width=a["dcv"] - b["dcv"]))
    return out


# ----------------------------------------------------------------------------------------------
def main() -> None:
    write = "--no-write" not in sys.argv
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    T1, T2 = list(ref["tiers"]["slot1"]), list(ref["tiers"]["slot2"])
    plan = list(ref["plan_0824"])
    thresh = float(ref["threshold_public"])
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()

    # ---------------------------------------------------------------- the one shared board
    lb_all, lt1, lt2, _, _ = W76A.board(exclude=plan)
    if sorted(lt1) != sorted(T1) or sorted(lt2) != sorted(T2):
        fail(f"rewound tier {sorted(lt1)}/{sorted(lt2)} != w63a's — the board did not rewind")
        sys.exit(1)
    S_NAMES = sorted(set(T1 + T2 + list(ref["wanted"]) + plan))
    print(f"board rewound to w63a's 08-23 state: {len(lb_all)} scored files, tier1 {lt1}")

    # ARM L's population, MECHANICALLY defined: every scored stem with an OOF vector, plus the
    # 15 already in the design. No file is chosen for where it sits, and the ones that cannot
    # be priced are named rather than guessed (w59a: "cannot price them, do not guess").
    have = [k for k in sorted(lb_all) if os.path.exists(os.path.join(SUB, f"oof_{k}.npy"))]
    noof = [k for k in sorted(lb_all) if k not in set(have)]
    L_NAMES = sorted(set(S_NAMES) | set(have))
    print(f"  ARM S design {len(S_NAMES)} files ({sum(k in lb_all for k in S_NAMES)} scored)")
    print(f"  ARM L design {len(L_NAMES)} files ({sum(k in lb_all for k in L_NAMES)} scored)")
    print(f"  ⚠ {len(noof)} scored stems have NO OOF vector and are EXCLUDED, not guessed:")
    print(f"    {', '.join(noof)}")

    S_CAND = sorted(set(plan + T2 + [W63A.PICK]))          # w63a's own row population
    L_CAND = sorted(set(L_NAMES) - set(T1))                 # NAMES minus the status quo pair
    LB_S = {k: v for k, v in lb_all.items() if k in set(S_NAMES)}
    LB_L = {k: v for k, v in lb_all.items() if k in set(L_NAMES)}

    # ============================================================================== ARM S
    print("\n" + "=" * 96)
    print("GATE A (P1) — ARM S must RE-DERIVE w63a_setprice.json. A comparison, not a stamp.")
    print("=" * 96)
    S = arm(S_NAMES, LB_S, T1, y, beta, thresh, S_CAND)
    worst, nchk = 0.0, 0

    def cmp(label, got, want, bar=1e-9):
        nonlocal worst, nchk
        nchk += 1
        worst = max(worst, abs(got - want))
        if abs(got - want) > bar:
            fail(f"GATE A: {label} {got:+.12f} != w63a's {want:+.12f}")

    cmp("base", S["base"], float(ref["base"]))
    cmp("worthless_limit1", S["worthless"], float(ref["worthless_limit1"]))
    rmap = {r["stem"]: r for r in ref["rows"]}
    if sorted(rmap) != sorted(r["stem"] for r in S["rows"]):
        fail("GATE A: ARM S's row population is not w63a's")
    for r in S["rows"]:
        cmp(f"uncond[{r['stem']}]", r["uncond"], float(rmap[r["stem"]]["uncond"]))
        cmp(f"cond[{r['stem']}]", r["cond"], float(rmap[r["stem"]]["cond"]))
        cmp(f"cv[{r['stem']}]", r["cv"], float(rmap[r["stem"]]["cv"]), 5e-10)
    crS = {k: crossings(S["rows"], S["base"], k) for k in ("uncond", "cond")}
    for k in ("uncond", "cond"):
        cmp(f"H[{k}]", crS[k][0]["H"], float(ref["H_" + ("uncond" if k == "uncond"
                                                         else "cond_predsd")]))
        cmp(f"bracket[{k}].width", crS[k][0]["width"], float(ref["bracket"][k]["width"]))
    print(f"  {nchk} recorded quantities re-derived; worst absolute deviation {worst:.3e} (bar 1e-9)")
    if FAILURES:
        print("\n⛔ GATE A FAILED — this is not w63a's instrument. Claiming nothing. REFUSING.")
        sys.exit(1)
    print("  ✅ GATE A PASSED (P1). Everything below is w63a's estimator, on a bigger design.")

    # ============================================================================== ARM L
    print("\n" + "=" * 96)
    print(f"ARM L — the same estimator, the same board, {len(L_NAMES)} files instead of {len(S_NAMES)}")
    print("=" * 96)
    L = arm(L_NAMES, LB_L, T1, y, beta, thresh, L_CAND)

    # ------------------------------------------------------------------ P2, numerical health
    condno = float(np.linalg.cond(L["E"]["Sxx"]))
    biggest = max(abs(r["uncond"]) for r in L["rows"])
    print(f"\n=== P2 GATE C: cond(Sxx) = {condno:.3e} (bar {COND_BAR:.0e}), "
          f"max|only| = {biggest:.3f} (bar {ONLY_BAR:.0e})")
    p2 = condno < COND_BAR and biggest < ONLY_BAR and np.isfinite(L["base"])
    if not p2:
        falsify("P2", "the enlarged design is numerically degenerate — the principled route "
                      "is dead for numerical reasons, and that IS the answer to w63 §10.5")
        print("\n⛔ Nothing below P2 can be read. Writing the artefact with p2=false and stopping.")
    else:
        print("  ✅ P2 CONFIRMED — the enlarged design is numerically sound.")

    hole_lo = float(ref["bracket"]["uncond"]["lo_dcv"])
    hole_hi = float(ref["bracket"]["uncond"]["hi_dcv"])
    inside = [r for r in L["rows"] if hole_lo < r["dcv"] < hole_hi]
    res = dict(p2=bool(p2))

    if p2:
        # -------------------------------------------------------------- P3, does the hole fill
        print(f"\n=== P3 HOLE FILL: candidates strictly inside w63a's bracket "
              f"({hole_lo:+.4f}, {hole_hi:+.4f}) ===")
        print(f"  {'stem':30s} {'dCV':>9s} {'uncond':>9s} {'cond':>9s}  verdict")
        for r in sorted(inside, key=lambda r: -r["dcv"]):
            print(f"  {r['stem']:30s} {r['dcv']:+9.3f} {r['uncond']:+9.4f} {r['cond']:+9.4f}"
                  f"  {'HELPS' if r['uncond'] < L['base'] else 'HURTS'}")
        p3 = len(inside) >= HOLE_MIN
        print(f"  n = {len(inside)} distinct files inside the hole (registered bar {HOLE_MIN})")
        if p3:
            print("  ✅ P3 CONFIRMED — the hole is populated by files nobody chose.")
        else:
            falsify("P3", f"only {len(inside)} candidate(s) inside the hole; the enlarged design "
                          "does not resolve the interpolation and P5/P7 are void")

        # ------------------------------------------------- P4, does GATE R survive the design
        dR = abs(L["base"] - S["base"])
        dW = abs(L["worthless"] - S["worthless"])
        print(f"\n=== P4 GATE R SURVIVAL — the registered question ===")
        print(f"  base        ARM S {S['base']:+.9f}   ARM L {L['base']:+.9f}   |d| {dR:.3e}")
        print(f"  worthless   ARM S {S['worthless']:+.9f}   ARM L {L['worthless']:+.9f}   |d| {dW:.3e}")
        print(f"  GLS gap     ARM S {S['E']['gap']:+.4f}e-6  ARM L {L['E']['gap']:+.4f}e-6")
        print(f"  gamma       ARM S {S['E']['gamma']:+.8f}   ARM L {L['E']['gamma']:+.8f}")
        print(f"  GATE R as w63a codes it (bar {GATE_R_BAR:.1e}): "
              f"{'FAILS' if max(dR, dW) > GATE_R_BAR else 'PASSES'} on the enlarged design")
        p4 = max(dR, dW) > GATE_R_BAR
        if p4:
            print("  ✅ P4 CONFIRMED — GATE R does not survive, exactly as w63 §10.5 suspected.")
        else:
            falsify("P4", "GATE R SURVIVES: `base` is design-invariant to <1e-6, which is a "
                          "BETTER outcome than predicted — the enlarged design can be adopted "
                          "wholesale without re-deriving the reference constant")

        # ---------------------------------------------------------- P5/P6/P7, the break-even
        print("\n=== P5/P6/P7 THE MEASURED BREAK-EVEN, no interpolation across a hole ===")
        crL = {k: crossings(L["rows"], L["base"], k) for k in ("uncond", "cond")}
        for k in ("uncond", "cond"):
            print(f"  {k:8s} {len(crL[k])} sign change(s) walking down in CV")
            for c in crL[k]:
                print(f"      H = {c['H']:7.3f}e-6   bracket {c['hi']} {c['hi_dcv']:+.2f} .. "
                      f"{c['lo']} {c['lo_dcv']:+.2f}  (width {c['width']:.3f})")
        p6 = all(len(crL[k]) == 1 for k in ("uncond", "cond"))
        if p6:
            print("  ✅ P6 CONFIRMED — the crossing is unique, so `the break-even` names a scalar.")
        else:
            falsify("P6", "the cost curve crosses `base` more than once, so `the break-even` is "
                          "NOT a well-defined scalar and every bar derived from one, INCLUDING "
                          "THE LIVE ONE, is under-specified")

        HL = crL["uncond"][0]["H"] if crL["uncond"] else None
        HLc = crL["cond"][0]["H"] if crL["cond"] else None
        p5 = p7 = None
        if HL is None:
            falsify("P5", "ARM L brackets no crossing at all")
        else:
            p5 = bool(hole_lo < -HL < hole_hi)
            print(f"\n  P5: H_L = {HL:.3f}e-6 -> dCV {-HL:+.3f}, w63a's bracket "
                  f"({hole_lo:+.3f}, {hole_hi:+.3f})")
            if p5:
                print("  ✅ P5 CONFIRMED — the measured crossing is inside the interpolated bracket.")
            else:
                falsify("P5", "the measured crossing is OUTSIDE w63a's bracket — the interpolated "
                              "bar is on the wrong side of a file that was already in the design")
            err = abs(HL - float(ref["H_uncond"]))
            p7 = bool(err < P7_BAR)
            print(f"\n  P7: |H_L - w63a's interpolated {float(ref['H_uncond']):.3f}| = {err:.3f}e-6 "
                  f"(bar {P7_BAR})")
            print("  ✅ P7 CONFIRMED — the straight line was good to under a third of the hole."
                  if p7 else "")
            if not p7:
                falsify("P7", f"the interpolation was off by {err:.3f}e-6, more than a third of "
                              f"the {hole_hi-hole_lo:.1f}e-6 hole it was drawn across")

        # ------------------------------------------------------ what it would mean for the bar
        bar_L = W63A.PICK and (L["cv"][W63A.PICK] - min(h for h in (HL, HLc) if h is not None) * U)
        print(f"\n=== WHAT THIS WOULD DO TO THE BAR (reported, NOT written) ===")
        print(f"  live bar   (w63a, interpolated) {float(ref['cv_bar_new']):.10f}"
              f"   from H_binding {float(ref['H_binding']):.3f}e-6")
        print(f"  ARM L bar  (measured)           {bar_L:.10f}"
              f"   from H_binding {min(h for h in (HL, HLc) if h is not None):.3f}e-6")
        print(f"  difference {(bar_L - float(ref['cv_bar_new']))/U:+.3f}e-6 of CV — "
              f"{'STRICTER' if bar_L > float(ref['cv_bar_new']) else 'LOOSER'} than the live bar")
        print("  ⛔ NOT WRITTEN. A run that enlarges the bracket may not persist the bar it "
              "implies\n     (w59a). The successor is a separate pre-registered run.")

        res.update(p3=bool(p3), p4=bool(p4), p5=p5, p6=bool(p6), p7=p7,
                   n_inside=len(inside), inside=[r["stem"] for r in
                                                 sorted(inside, key=lambda r: -r["dcv"])],
                   base_S=S["base"], base_L=L["base"], worthless_S=S["worthless"],
                   worthless_L=L["worthless"], gate_r_dev=max(dR, dW),
                   gap_S=S["E"]["gap"], gap_L=L["E"]["gap"],
                   gamma_S=S["E"]["gamma"], gamma_L=L["E"]["gamma"],
                   H_L_uncond=HL, H_L_cond=HLc, crossings_L=crL,
                   bar_L_reported_not_written=bar_L,
                   bar_live_w63a=float(ref["cv_bar_new"]))

    out = dict(
        purpose="feasibility of the ENLARGED design named in w63 §10.5; PERSISTS NO BAR",
        prereg="experiments/w77_prereg.txt @ 39e4aaf",
        board="live board minus w63a.plan_0824 == w63a's own 08-23 board",
        n_scored=len(lb_all), tiers=ref["tiers"],
        arm_s=dict(n_names=len(S_NAMES), n_cand=len(S_CAND), names=S_NAMES),
        arm_l=dict(n_names=len(L_NAMES), n_cand=len(L_CAND), names=L_NAMES),
        excluded_no_oof=noof, cond_Sxx_L=condno,
        gate_a=dict(passed=True, worst=worst, n_checks=nchk),
        hole=dict(lo_dcv=hole_lo, hi_dcv=hole_hi, width=hole_hi - hole_lo),
        predictions=res, falsified=FALSIFIED, failures=FAILURES,
        writes_a_bar=False)
    if write:
        json.dump(out, open(os.path.join(HERE, "w77a_bracket.json"), "w"), indent=1)
        pd.DataFrame(L["rows"]).to_csv(os.path.join(HERE, "w77a_bracket.csv"), index=False)
        print(f"\nwrote w77a_bracket.json ({len(L['rows'])} ARM L rows in w77a_bracket.csv)")
    print(f"\nFAILURES {FAILURES} · FALSIFIED {FALSIFIED or 'none'}")


if __name__ == "__main__":
    main()
