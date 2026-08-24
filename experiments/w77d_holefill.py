"""w77d — EXPLORATORY. Can the bracket hole be filled WITHOUT enlarging the GLS fit?

⚠⚠ THIS IS NOT A TEST AND REGISTERS NOTHING. It was designed after w77a and w77c had already
been read, so its numbers are a PROPOSAL for a successor to pre-register and re-derive, not
evidence. Anything downstream that cites it as a measurement is citing a post-hoc construction.

WHAT w77a AND w77c ESTABLISHED
  * 13 scored files sit inside w63a's 16.5e-6 bracket hole (w77a P3, CONFIRMED).
  * Pricing against all 122 of them destroys the estimator: `fit` puts every scored file into
    one GLS common-gap fit, the board's files are near-duplicate ranking vectors (median
    max-pairwise-correlation to an earlier file 0.999924), and the GLS weights reach sum|w| =
    2001 with gh landing 5,634e-6 from the equal-weight mean (w77c).

THE OBSERVATION THIS FILE ACTS ON
  `fit` derives its GLS set as `scored = [k for k in NAMES if k in LB]`. DESIGN MEMBERSHIP and
  GLS MEMBERSHIP are the same switch only because they are fed the same dict. A file passed in
  NAMES but withheld from LB becomes exactly what w63a already makes of an unsent candidate:
  xh0 = 0, priced through the coupling, contributing nothing to the common gap.

  And for the HIJACK question that is arguably the RIGHT treatment, not a concession. The
  question is "what if X had landed a reporting step higher"; conditioning on the public score
  X actually got is conditioning on the branch that did not happen.

  So: enlarge the CANDIDATE set to fill the hole, hold the GLS fit at the 8 files w63a already
  uses. GATE D checks the GLS really is untouched, by reproducing w63a's stored `gap` exactly.

⛔ WRITES NO BAR. w77d_holefill.json only.
"""
from __future__ import annotations

import json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)
from common import TARGET, load_raw                            # noqa: E402
import w63a_setprice as W63A                                   # noqa: E402
import w76a_addtest as W76A                                    # noqa: E402
from w77a_bracket import arm, crossings                        # noqa: E402 — same definitions

U = 1e-6


def main() -> None:
    ref = json.load(open(os.path.join(HERE, "w63a_setprice.json")))
    T1, T2 = list(ref["tiers"]["slot1"]), list(ref["tiers"]["slot2"])
    plan = list(ref["plan_0824"])
    thresh = float(ref["threshold_public"])
    beta = float(json.load(open(os.path.join(HERE, "w17d_coupling.json")))["coupling_beta_median"])
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    lb_all, lt1, _, _, _ = W76A.board(exclude=plan)
    assert sorted(lt1) == sorted(T1), "the board did not rewind"

    S_NAMES = sorted(set(T1 + T2 + list(ref["wanted"]) + plan))
    LB_S = {k: v for k, v in lb_all.items() if k in set(S_NAMES)}      # w63a's own 8 scored
    fill = list(json.load(open(os.path.join(HERE, "w77a_bracket.json")))["predictions"]["inside"])
    M_NAMES = sorted(set(S_NAMES) | set(fill))
    M_CAND = sorted(set(M_NAMES) - set(T1))
    print(f"ARM M: {len(M_NAMES)} design files = w63a's {len(S_NAMES)} + {len(fill)} hole-fillers")
    print(f"  GLS fit held at w63a's {len(LB_S)} scored files; the {len(fill)} fillers enter with "
          f"xh0 = 0,\n  the same treatment w63a gives its ten unsent plan candidates.")

    M = arm(M_NAMES, LB_S, T1, y, beta, thresh, M_CAND)

    # ------------------------------------------------------------------------------ GATE D
    print("\n=== GATE D: the GLS common gap must be UNTOUCHED by the enlargement ===")
    dg = abs(M["E"]["gap"] - float(ref["gap"]))
    print(f"  gh  ARM M {M['E']['gap']:+.10f}   w63a {float(ref['gap']):+.10f}   |d| {dg:.3e}")
    if dg > 1e-6:
        print("⛔ GATE D FAILED: the fillers leaked into the GLS. The construction does not do\n"
              "   what it claims and nothing below can be read. REFUSING.")
        sys.exit(1)
    print("  ✅ GATE D PASSED — same gap, same Vg, only the candidate set grew.")

    print(f"\n  base       w63a {float(ref['base']):+.6f}   ARM M {M['base']:+.6f}   "
          f"|d| {abs(M['base']-float(ref['base'])):.3e}")
    print(f"  worthless  w63a {float(ref['worthless_limit1']):+.6f}   ARM M {M['worthless']:+.6f}"
          f"   |d| {abs(M['worthless']-float(ref['worthless_limit1'])):.3e}")

    print(f"\n=== the cost curve with the hole filled ({len(M['rows'])} candidates) ===")
    print(f"  {'stem':30s} {'dCV':>9s} {'uncond':>9s} {'cond':>9s}  in-hole verdict")
    for r in M["rows"]:
        print(f"  {r['stem']:30s} {r['dcv']:+9.3f} {r['uncond']:+9.4f} {r['cond']:+9.4f}"
              f"  {'FILL' if r['stem'] in set(fill) else '    '}"
              f"    {'HELPS' if r['uncond'] < M['base'] else 'HURTS'}")

    print("\n=== the break-even, MEASURED rather than interpolated ===")
    out_cr = {}
    for key in ("uncond", "cond"):
        cr = crossings(M["rows"], M["base"], key)
        out_cr[key] = cr
        print(f"  {key:8s} {len(cr)} crossing(s)")
        for c in cr:
            print(f"      H = {c['H']:7.3f}e-6  bracket {c['hi']} {c['hi_dcv']:+.2f} .. "
                  f"{c['lo']} {c['lo_dcv']:+.2f}  (width {c['width']:.3f})")
    Hs = [c[0]["H"] for c in out_cr.values() if c]
    H_bind = min(Hs) if Hs else None
    print(f"\n  w63a's interpolated bracket was {float(ref['bracket']['uncond']['width']):.3f}e-6 "
          f"wide with nothing in it.")
    if H_bind is not None:
        bar = M["cv"][W63A.PICK] - H_bind * U
        print(f"  ARM M binding H = {H_bind:.3f}e-6 -> CV bar {bar:.10f}")
        print(f"  live bar (w63a) {float(ref['cv_bar_new']):.10f} from H_binding "
              f"{float(ref['H_binding']):.3f}e-6")
        print(f"  difference {(bar - float(ref['cv_bar_new']))/U:+.3f}e-6 of CV — "
              f"{'STRICTER' if bar > float(ref['cv_bar_new']) else 'LOOSER'}")
    else:
        bar = None
        print("  no crossing bracketed")
    print("\n  ⛔ NOT WRITTEN AND NOT ADOPTED. This construction was designed after the answer was\n"
          "     visible. A successor must pre-register it, state why withholding a real public\n"
          "     score from the GLS is the right counterfactual for a hijack, and re-derive the\n"
          "     bar through w63a itself before anything on the send path moves.")

    json.dump(dict(exploratory=True, registers_nothing=True, writes_a_bar=False,
                   n_names=len(M_NAMES), n_cand=len(M_CAND), fillers=fill,
                   gate_d=dict(passed=True, gap=M["E"]["gap"], w63a_gap=float(ref["gap"]), dev=dg),
                   base=M["base"], base_w63a=float(ref["base"]),
                   worthless=M["worthless"], worthless_w63a=float(ref["worthless_limit1"]),
                   crossings=out_cr, H_binding=H_bind,
                   bar_would_be=bar, bar_live=float(ref["cv_bar_new"]),
                   rows=M["rows"]),
              open(os.path.join(HERE, "w77d_holefill.json"), "w"), indent=1)
    pd.DataFrame(M["rows"]).to_csv(os.path.join(HERE, "w77d_holefill.csv"), index=False)
    print("\nwrote w77d_holefill.json / .csv")


if __name__ == "__main__":
    main()
