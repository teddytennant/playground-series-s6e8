"""w88a — the send gate is PER FILE and it is applied SIXTY MORE TIMES. Price the calendar.

WHAT NOTHING ELSE ASKS
----------------------
`w26g_send.hijack_risk` refuses any filler whose P(landing above the auto-selection tier)
exceeds `P_MAX = 0.02`, and `w87a` C7 replays that refusal test over all 60 registered files
and reports 0 refusals. Both are true and both are PER FILE. Nobody has ever asked what
sixty independent 2%-tolerated draws add up to, and the answer is not 2%.

The exposure this measures is the one thing the filler programme could still cost us: while
`check_selection` reports nothing selected, Kaggle auto-selects the best two by PUBLIC score,
so a filler that lands at or above the tier IS a final entry — a CV-inferior one, installed
by the public slice, which is the failure this whole workspace is organised against.

THE TWO LANDINGS, PRICED IN THE WORKSPACE'S OWN UNIT (both from `w79a_barfill.json`)
  ABOVE the tier   X takes auto-slot 1 alone: `worthless_limit1 - base` per landing.
  AT the tier      X joins tier 1 and dilutes the draw: the same increment divided by
                   `gate_j.leverage` = C(n+1,2)/n, which is w63a's GATE J identity and the
                   reason the multiplier is 1.5 on today's two-file tier and not the 3.0 it
                   was at n = 5. ⛔ Do not re-type either number: they are read, not derived.

THE ANSWER, 2026-08-25: about 0.2e-6 of expected private AUC over all six remaining days —
roughly a twentieth of the 4.52e-6 click price the workspace already calls "not a large
number". The per-file gate is vindicated in aggregate, and that is a question CLOSED, not a
problem opened. ⚠ REPORTED AND NOT ASSERTED: the family-wise PROBABILITY of an above-tier
landing (2.9% at the sender's sd) does exceed the sender's own per-file `P_MAX`. That is not
a contradiction — a probability is not a price, and 60 draws of a 1e-6 loss is still 1e-6.

THE CEILING IS EXTERNALLY ANCHORED, ON PURPOSE. G3 asserts the priced exposure stays under
`base` — the cost of nobody clicking. That number was fixed by w74a long before this file
existed and is the one price in the workspace already judged worth acting on, so it cannot be
tuned to make today's answer pass. Today it passes with ~20x of room.

⛔ THIS ADOPTS NOTHING. It does not re-price, re-rank or re-order a single send. If the
   exposure ever breaches the ceiling the remedy is a human reading WHY, not an edit here.
⛔ THE SENSITIVITY ROW AT w82a's sd IS A REPORT. w82a's own docstring forbids using its
   residuals to re-price anything, and the DO-NOT list forbids per-day and per-family
   corrections from it. G3 is asserted at the SENDER's `PRED_SD` — the constant that actually
   gates — and the wider sd is printed beside it so a reader can see which way it moves.

CONTROLS (w72 §5.3: a control that can only fail is not a control)
  C1 +- a planted registration 20e-6 ABOVE the tier drives the exposure past the ceiling and
        G3 FIRES; removing it passes again. Both directions, on the real calendar.
  C2 -  a registered stem with NO `pred_lb` FAILS (G1) instead of being silently dropped from
        the sum. w55/w58 shipped that exact hole twice, one level up each time.
  C3    every per-file probability is `w26g_send.hijack_risk` ITSELF, not a second copy of the
        formula, and C3 asserts the two agree to 1e-15 over all 60 (w87 §3: two copies of a
        ruling disagreed inside an hour).
  C4    the live tier is `w26g_send.auto_tier` over `w26g_send.api_submissions`, which refuses
        at its own page size (w86 §2) — the only truncation test that detects itself.
  C5    drift between the LIVE tier and the tier each day was registered against is REPORTED,
        never asserted (w87a C2's idiom); the tier is monotone non-decreasing, so drift can
        only make a registered margin WIDER.

    .venv/bin/python experiments/w88a_calexposure.py         # 0 = ok, 1 = a guard failed
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import w26g_send as SND                                                     # noqa: E402

OUT = os.path.join(HERE, "w88a_calexposure.json")
QUEUEPRICE = os.path.join(HERE, "w26d_queueprice.csv")
UNPRICED = os.path.join(HERE, "w55a_unpriced.json")
FILLERS = os.path.join(HERE, "w85a_fillers.csv")
PLANS = os.path.join(HERE, "w72a_plan_*.json")
U = 1e-6

# w82a's out-of-sample residual sd over EVERY priced send (n = 62, all families). Printed as a
# sensitivity row only -- see the docstring. The gate is the sender's PRED_SD.
W82A = os.path.join(HERE, "w82a_pricecal.json")


def price_units():
    """`base`, and what one landing of each kind costs ON TOP of it. Read, never re-derived.

    ⚠ EVERY NUMBER IN `w79a_barfill.json` IS ALREADY IN e-6. w63a prints them as
    `{base:+.4f}e-6` and stores the mantissa, so this module works in e-6 throughout and
    NEVER divides by U again. The first cut did, and reported an exposure of 214337e-6 --
    which passed G3 anyway, because the ceiling was scaled by the same mistake. A units bug
    that cancels inside the comparison is invisible in the verdict and wrong in the report.
    """
    d = json.load(open(SND.HIJACKPRICE))
    base = float(d["base"])                       # e-6
    above = float(d["worthless_limit1"]) - base   # e-6, per ABOVE-tier landing
    lev = float(d["gate_j"]["leverage"])
    return base, above, above / lev, lev


def registered(fail_on_missing=True, plant=None):
    """Every file registered for a day that has not been sent yet, with its published pred_lb.

    `plant` appends a synthetic (day, stem, pred_lb) row -- C1's lever.
    """
    q = pd.read_csv(QUEUEPRICE)
    pl = {s: v for s, v in zip(q["stem"], q["pred_lb"])}
    fam = {s: f for s, f in zip(q["stem"], q["fam"])}
    cv = {s: c for s, c in zip(q["stem"], q["cv"])}
    # the two member sources, so a certified filler is priced rather than silently skipped
    try:
        w55 = json.load(open(UNPRICED))["rows"]
    except OSError:
        w55 = {}
    try:
        f85 = pd.read_csv(FILLERS)
        pl85 = {f[:-4]: v for f, v in zip(f85["file"], f85["pred_lb"])}
    except OSError:
        pl85 = {}

    out, missing = [], []
    for p in sorted(glob.glob(PLANS)):
        d = json.load(open(p))
        for stem in d["plan"]:
            v = pl.get(stem)
            src = "w26d"
            if v is None or (isinstance(v, float) and math.isnan(v)):
                if stem + ".csv" in w55:
                    v, src = float(w55[stem + ".csv"]["reg_lb"]), "w55a"
                elif stem in pl85:
                    v, src = float(pl85[stem]), "w85a"
                else:
                    v, src = None, "MISSING"
                    missing.append((d["day"], stem))
            out.append(dict(day=d["day"], stem=stem, pred_lb=v, src=src,
                            fam=fam.get(stem), cv=cv.get(stem),
                            reg_tier=float(d["tier"])))
    if plant is not None:
        out.append(dict(day=plant[0], stem=plant[1], pred_lb=plant[2], src="PLANT",
                        fam="rankraw", cv=0.97, reg_tier=out[0]["reg_tier"] if out else None))
        if plant[2] is None:
            missing.append((plant[0], plant[1]))
    return pd.DataFrame(out), missing


def expose(df, tier, sd, above_cost, at_cost):
    """Family-wise landing probabilities and the priced exposure, over the whole calendar."""
    p_above, p_at = [], []
    for v in df["pred_lb"]:
        z_hi = (tier + SND.STEP / 2 - float(v)) / sd
        z_lo = (tier - SND.STEP / 2 - float(v)) / sd
        pa = 0.5 * (1.0 - math.erf(z_hi / math.sqrt(2.0)))
        p_above.append(pa)
        p_at.append(0.5 * (1.0 + math.erf(z_hi / math.sqrt(2.0)))
                    - 0.5 * (1.0 + math.erf(z_lo / math.sqrt(2.0))))
    any_above = 1.0 - math.prod(1.0 - p for p in p_above)
    any_at = 1.0 - math.prod(1.0 - p for p in p_at)
    return dict(p_above=p_above, p_at=p_at,
                any_above=any_above, any_at=any_at,
                e_above=sum(p_above), e_at=sum(p_at),
                exposure_e6=sum(p_above) * above_cost + sum(p_at) * at_cost,
                max_pf=max(p_above) if p_above else 0.0)


def main():
    fails = []
    base, above_cost, at_cost, lev = price_units()
    rows = SND.api_submissions()                                            # C4
    tier = SND.auto_tier(rows)
    if tier is None:
        print("⛔ could not read the auto-selection tier from the board — refusing to price.")
        return 1
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    print(f"live tier {tier:.5f} over {len(rows)} submissions; "
          f"sender PRED_SD {SND.PRED_SD/U:.2f}e-6, P_MAX {SND.P_MAX}")
    print(f"prices from {os.path.basename(SND.HIJACKPRICE)}: base {base:+.4f}e-6, "
          f"ABOVE-tier landing {above_cost:+.4f}e-6, AT-tier landing {at_cost:+.4f}e-6 "
          f"(leverage {lev:.1f}, n_tier {json.load(open(SND.HIJACKPRICE))['gate_j']['n_tier']})")

    df, missing = registered()

    # ---------------------------------------------------------------------- G1
    if missing:
        fails.append(f"G1 {len(missing)} registered file(s) have no pred_lb: {missing[:5]}")
        print(f"⛔ G1 {len(missing)} registered file(s) unpriceable — they cannot be gated "
              f"against the tier at all: {missing[:5]}")
    else:
        print(f"✅ G1 all {len(df)} registered files carry a pred_lb "
              f"({df['src'].value_counts().to_dict()})")

    live = expose(df, tier, SND.PRED_SD, above_cost, at_cost)

    # ---------------------------------------------------------------------- C3
    worst = max(abs(p - SND.hijack_risk(v, tier))
                for p, v in zip(live["p_above"], df["pred_lb"]))
    if worst > 1e-15:
        fails.append(f"C3 per-file probabilities disagree with w26g_send.hijack_risk ({worst:.3e})")
        print(f"⛔ C3 my P(above) differs from the sender's own by {worst:.3e}")
    else:
        print(f"✅ C3 every per-file P(above) is w26g_send.hijack_risk itself (max |d| {worst:.1e})")

    # ---------------------------------------------------------------------- G2
    over = [(s, p) for s, p in zip(df["stem"], live["p_above"]) if p > SND.P_MAX]
    if over:
        fails.append(f"G2 {len(over)} registered file(s) exceed the sender's per-file P_MAX: {over}")
        print(f"⛔ G2 {len(over)} registered file(s) over P_MAX — the sender would refuse them: {over}")
    else:
        print(f"✅ G2 every registered file is under the sender's per-file P_MAX "
              f"(worst {live['max_pf']:.4f} vs {SND.P_MAX})")

    # ---------------------------------------------------------------------- the table
    df = df.assign(margin_e6=(tier - df["pred_lb"]) / U, p_above=live["p_above"],
                   p_at=live["p_at"])
    print(f"\nper day ({len(df)} registered files, days > {today} or today with slots left):")
    g = df.groupby("day").agg(n=("stem", "size"), min_margin_e6=("margin_e6", "min"),
                              sum_p_above=("p_above", "sum"), sum_p_at=("p_at", "sum"))
    print(g.to_string(float_format=lambda x: f"{x:10.4f}"))
    print("\nthe five tightest margins on the whole remaining calendar:")
    print(df.nsmallest(5, "margin_e6")[["day", "stem", "fam", "pred_lb", "margin_e6",
                                        "p_above", "p_at"]].to_string(index=False))

    # ---------------------------------------------------------------------- G3
    print(f"\nFAMILY-WISE over all {len(df)} remaining sends, at the sender's "
          f"{SND.PRED_SD/U:.2f}e-6:")
    print(f"  P(at least one lands ABOVE the tier) {live['any_above']:.4f}   "
          f"E[#] {live['e_above']:.3f}")
    print(f"  P(at least one lands AT    the tier) {live['any_at']:.4f}   "
          f"E[#] {live['e_at']:.3f}")
    print(f"  PRICED EXPOSURE {live['exposure_e6']:+.4f}e-6   ceiling = base {base:+.4f}e-6")
    # C6: G3 is not implied by G2. The worst calendar the PER-FILE gate admits is every file
    # sitting just under P_MAX, and that calendar breaches the ceiling -- so the aggregate
    # test has bite that the per-file test does not.
    _pa = SND.P_MAX
    _pat = max(lv for lv in live["p_at"]) if live["p_at"] else 0.0
    _worst = len(df) * (_pa * above_cost + _pat * at_cost)
    print(f"  ℹ C6 the worst calendar G2 admits ({len(df)} files at P_MAX, each with the "
          f"tightest live P(at) {_pat:.3f}) prices at {_worst:+.4f}e-6, "
          f"{'ABOVE' if _worst > base else 'below'} the ceiling -- so G3 is "
          f"{'not ' if _worst > base else ''}implied by G2")
    if live["any_above"] > SND.P_MAX:
        print(f"  ⚠ REPORTED, NOT ASSERTED: the family-wise probability {live['any_above']:.4f} "
              f"exceeds the sender's PER-FILE P_MAX {SND.P_MAX}. A probability is not a price; "
              f"G3 is on the price.")
    if not (live["exposure_e6"] < base):
        fails.append(f"G3 priced calendar exposure {live['exposure_e6']:.4f}e-6 >= base {base:.4f}e-6")
        print(f"⛔ G3 the remaining calendar now costs more than nobody clicking. Read WHY "
              f"before editing anything.")
    else:
        print(f"✅ G3 exposure is {base/live['exposure_e6']:.1f}x under the ceiling")

    # ------------------------------------------------------- sensitivity, REPORTED ONLY
    sens = {}
    try:
        sd82 = float(json.load(open(W82A))["resid_sd_e6"]) * U
        w = expose(df, tier, sd82, above_cost, at_cost)
        sens = dict(sd_e6=sd82 / U, any_above=w["any_above"], any_at=w["any_at"],
                    exposure_e6=w["exposure_e6"])
        flips = [(s_, p2) for s_, p1, p2 in zip(df["stem"], live["p_above"], w["p_above"])
                 if p1 <= SND.P_MAX < p2]
        sens["flips"] = [[s_, p2] for s_, p2 in flips]
        print(f"\nℹ SENSITIVITY (reported, adopts nothing) at w82a's out-of-sample sd "
              f"{sd82/U:.2f}e-6: P(any above) {w['any_above']:.4f}, P(any at) {w['any_at']:.4f}, "
              f"exposure {w['exposure_e6']:+.4f}e-6 — still "
              f"{base/w['exposure_e6']:.1f}x under the ceiling.")
        # THE ONE LINE A LATER RUN NEEDS. The two sds are not the same estimand -- 8.77e-6 is
        # w52c's HELD-OUT era-slice RMSE and every registered file is an era file, while w82a's
        # 13.00e-6 is the realised residual over every priced send of every family -- so neither
        # dominates and G3 is asserted at the one that actually gates. But the PER-FILE verdict
        # on the tightest files is not robust to the choice, and the price is what says that
        # does not matter here.
        if flips:
            print(f"  ⚠ {len(flips)} registered file(s) that G2 ADMITS at {SND.PRED_SD/U:.2f}e-6 "
                  f"would be REFUSED at {sd82/U:.2f}e-6: "
                  + ", ".join(f"{s_} P {p2:.4f}" for s_, p2 in flips))
            print(f"    ⛔ NOT A REASON TO EDIT THE CALENDAR. Both sds price the whole "
                  f"calendar at well under the ceiling ({live['exposure_e6']:.4f} and "
                  f"{w['exposure_e6']:.4f}e-6 vs {base:.4f}e-6); dropping every flipped file "
                  f"buys ~{sum(p * above_cost for _, p in flips):.3f}e-6 at the wider sd and "
                  f"costs a send-path edit. Measured, then left alone.")
        else:
            print(f"  ✅ no registered file's per-file verdict depends on which sd is used")
    except (OSError, KeyError, ValueError, ZeroDivisionError) as e:
        print(f"ℹ sensitivity row unavailable ({e}) — G3 is unaffected, it is asserted at "
              f"the sender's PRED_SD.")

    # ---------------------------------------------------------------------- C5
    drift = sorted({(d, t) for d, t in zip(df["day"], df["reg_tier"]) if t != tier})
    if drift:
        print(f"\nℹ C5 tier drift since registration (REPORTED — the tier is monotone "
              f"non-decreasing, so drift widens every registered margin): {drift}")
    else:
        print(f"\n✅ C5 every day was registered against the live tier {tier:.5f}")

    # ---------------------------------------------------------------------- C1, C2
    print("\n-- controls --")
    _, miss2 = registered(plant=("2026-08-27", "w88a_control_unpriced", None))
    if any(s == "w88a_control_unpriced" for _, s in miss2):
        print("✅ C2 an unpriced registration is caught by G1, not silently dropped")
    else:
        fails.append("C2 an unpriced registration was NOT caught")
        print("⛔ C2 an unpriced registration slipped through G1")

    # ⚠ ONE planted hijacker CANNOT fire G3, and that is arithmetic, not a weak guard: an
    # above-tier landing costs `above_cost` and the ceiling is `base`, so the plant has to be
    # ceil(base/above_cost) files deep before the price can reach it. The first cut planted one,
    # read the pass as a fault, and would have had me lower the ceiling to make a control go
    # green -- fitting the bar to the control is the same error as fitting it to the answer.
    need = int(math.ceil(base / above_cost))
    plants = [("2026-08-27", f"w88a_control_hijacker{i}", tier + 20 * U) for i in range(need)]
    dfp, _ = registered()
    for pl in plants:
        dfp.loc[len(dfp)] = dict(day=pl[0], stem=pl[1], pred_lb=pl[2], src="PLANT",
                                 fam="rankraw", cv=0.97, reg_tier=tier)
    lp = expose(dfp, tier, SND.PRED_SD, above_cost, at_cost)
    if lp["exposure_e6"] >= base:
        print(f"✅ C1+ {need} planted +20e-6 registrations ({need} = ceil(base/above_cost), the "
              f"shallowest plant that CAN reach the ceiling) drive the exposure to "
              f"{lp['exposure_e6']:+.4f}e-6 and G3 FIRES")
    else:
        fails.append(f"C1 the planted hijackers did NOT trip G3 ({lp['exposure_e6']:.4f}e-6)")
        print(f"⛔ C1 {need} planted hijackers left G3 passing at {lp['exposure_e6']:.4f}e-6 — "
              f"the guard cannot detect the thing it exists for")
    dfm1, _ = registered()
    for pl in plants[:-1]:
        dfm1.loc[len(dfm1)] = dict(day=pl[0], stem=pl[1], pred_lb=pl[2], src="PLANT",
                                   fam="rankraw", cv=0.97, reg_tier=tier)
    lm1 = expose(dfm1, tier, SND.PRED_SD, above_cost, at_cost)
    if lm1["exposure_e6"] < base:
        print(f"✅ C1- one fewer plant ({need - 1}) passes at {lm1['exposure_e6']:+.4f}e-6, so the "
              f"control brackets the ceiling rather than swamping it")
    else:
        fails.append(f"C1- the {need-1}-plant calendar also fired G3 — the control swamps the bar")
    if live["exposure_e6"] < base:
        print(f"✅ C1- and the real calendar passes at {live['exposure_e6']:+.4f}e-6")

    json.dump(dict(day=today, tier=tier, n_registered=int(len(df)),
                   pred_sd_e6=SND.PRED_SD / U, p_max=SND.P_MAX,
                   base_e6=base / U, above_landing_e6=above_cost / U,
                   at_landing_e6=at_cost / U, leverage=lev,
                   any_above=live["any_above"], any_at=live["any_at"],
                   e_above=live["e_above"], e_at=live["e_at"],
                   exposure_e6=live["exposure_e6"], sensitivity=sens,
                   tightest=df.nsmallest(5, "margin_e6")[["day", "stem", "margin_e6",
                                                          "p_above"]].to_dict("records"),
                   failures=fails,
                   note="ADOPTS nothing -- measurement only; see the module docstring"),
              open(OUT, "w"), indent=1)

    print(f"\n{'⛔' if fails else '✅'} GUARDS {len(fails)} failure(s) -> {OUT}")
    for f in fails:
        print("   -", f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
