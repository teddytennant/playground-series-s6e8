"""w46b — the JOINT cost of a ten-file drain, and whether SEND ORDER is a free lever.

w46a prices each candidate ALONE. Its P4 then added nine such deltas and got -94.89e-6,
driving the cost of not clicking to -73e-6, which is impossible: the cost is bounded below by
cost(best available auto-pick) ~= +2.6e-6. THE SUM IS NOT THE JOINT. Nine files each ~certain
to land at 0.97119 each remove THE SAME +21.74e-6 baseline; adding the deltas counts that
removal nine times. w46a's P4 verdict "below +12e-6" is right, its arithmetic is not, and this
file replaces it.

Two things only the joint can say:

 1. The real number. Once ANY ONE of the near-certain files clears the tier, the tie dissolves
    and the cost is that file's own, so the drain's value saturates almost immediately. The
    marginal file after the second is worth ~nothing on this axis.

 2. SEND ORDER IS A FREE LEVER, and w46a missed it. Under the LATEST-FIRST tiebreak -- one of
    the two branches w45a could not rule out -- Kaggle's pick inside a tie is the NEWEST
    submission. w46a's send list is ordered best-CV-FIRST, which under that rule makes the
    WORST-CV file of the ten the newest and therefore the pick. Reversing the order costs
    nothing and is strictly better in that branch.

RESIDUAL CORRELATION. w30b's residual sd is 7.76e-6 against an independently simulated
slice+grid noise floor of 8.70e-6 -- i.e. the residual is essentially pure public-slice draw,
and the slice draw is SHARED by files whose predictions correlate 0.99+. So the near-truth is
the COMMON branch. Both extremes are run and reported as a bracket; do not read the
independent branch as the answer.

    .venv/bin/python experiments/w46b_jointdrain.py
"""
from __future__ import annotations

import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260821)
NDRAW = 200_000
# ⚠ per-file predictive sd from w46c (era-corrected): 10.70e-6 for ad>=195, 7.76e-6 below.
# The first run of this file used w30b's flat 7.76e-6 and is preserved as
# w46b_jointdrain.UNCORRECTED.json.
from w46c_predlb import pred_sd, new_era, ERA_SHIFT  # noqa: E402
CV_W = 0.9701400060          # w36_ad199stdcorr, the better WANTED file
GAMMA_MAP = 1.0918           # w45a: with dLB = 0 the LAW-IF map degenerates to this scalar

A = json.load(open(os.path.join(HERE, "w46a_sendhazard.json")))
ROW = {r["stem"]: r for r in A["rows"]}
BASE = A["base"]
TIER1 = A["tiers"]["slot1"]
SEND = A["send_order"]

_fallback = [0]


def cost_of(stem, bucket):
    """Cost of not clicking when `stem` is the sole auto-pick, given it landed in `bucket`
    (19 = >= 0.97119, 18 = exactly 0.97118). Read out of w46a's per-file LAW-IF solves."""
    r = ROW[stem]
    d = r["detail"]
    if bucket == 19 and "s19" in d:
        return d["s19"]["new"]["uniform"]          # all three tiebreaks agree: x is alone
    if bucket == 18 and "s18" in d:
        return d["s18"]["new"]["latest"]           # latest-first picks x inside the tie => cost(x)
    _fallback[0] += 1
    return GAMMA_MAP * (CV_W - r["cv"]) / 1e-6     # w45a's validated closed form


def simulate(order, common):
    """order: the send sequence, EARLIEST first. common: share the slice draw across files."""
    k = len(order)
    pred = np.array([ROW[s]["pred"] for s in order])
    sd = np.array([pred_sd(s) for s in order])
    if common:
        # one shared standardised slice draw, scaled by each file's own predictive sd
        e = RNG.normal(0.0, 1.0, size=(NDRAW, 1)) * sd[None, :]
    else:
        e = RNG.normal(0.0, 1.0, size=(NDRAW, k)) * sd[None, :]
    score = np.round(pred[None, :] + e, 5)

    c19 = np.array([cost_of(s, 19) for s in order])
    c18 = np.array([cost_of(s, 18) for s in order])
    old = np.array([BASE["limit1"][t] for t in TIER1])

    out = {t: np.empty(NDRAW) for t in ("uniform", "latest", "earliest")}
    for d in range(NDRAW):
        sc = score[d]
        topnew = sc.max()
        top = max(0.97118, topnew)
        at = sc >= top - 1e-9                       # new files sitting in the top tier
        newc = np.where(sc >= 0.971185, c19, c18)[at]
        idx = np.flatnonzero(at)
        if top > 0.97118 + 1e-9:                    # our new files own the tier outright
            out["uniform"][d] = newc.mean()
            out["latest"][d] = newc[np.argmax(idx)]     # newest = latest in the send order
            out["earliest"][d] = newc[np.argmin(idx)]
        else:                                       # they joined the existing four-way tie
            allc = np.concatenate([old, newc])
            out["uniform"][d] = allc.mean()
            # the existing tie's newest member is the WANTED file (00:07 today); anything we
            # send tomorrow is newer still, so latest-first prefers our newest sent file.
            out["latest"][d] = newc[np.argmax(idx)] if len(newc) else BASE["latest"]
            # the existing tie's oldest is w21_ad187corr_ens4 (08-17), older than anything new.
            out["earliest"][d] = BASE["earliest"]
    return {t: (float(v.mean()), float(np.percentile(v, 5)), float(np.percentile(v, 95)))
            for t, v in out.items()}


def show(title, res):
    print(f"  {title:34s} " + "  ".join(
        f"{t}: {res[t][0]:+7.2f} [{res[t][1]:+6.2f},{res[t][2]:+6.2f}]"
        for t in ("uniform", "latest", "earliest")))


def main() -> None:
    print(f"w46a's ten, in its best-CV-FIRST order:\n  " + "\n  ".join(
        f"{i+1:2d}. {s:26s} CV {ROW[s]['cv']:.10f}  dCV {ROW[s]['dcv']:+7.1f}  "
        f"pred {ROW[s]['pred']:.6f}  P19 {ROW[s]['p19']:.3f}"
        for i, s in enumerate(SEND)))
    print(f"\nBASELINE, send nothing:            uniform {BASE['uniform']:+7.2f}  "
          f"latest {BASE['latest']:+7.2f}  earliest {BASE['earliest']:+7.2f}   (e-6)")
    print(f"w46a P4's SUM-OF-DELTAS claim:     uniform {BASE['uniform'] - 94.89:+7.2f}  "
          f"<-- impossible, the cost floor is ~+2.6e-6. Superseded by the rows below.")

    fwd = list(SEND)
    rev = list(reversed(SEND))
    print(f"\n=== JOINT, {NDRAW:,} draws, COMMON slice residual (the near-truth branch) ===")
    print("  positive = clicking is still worth this much.  [5th, 95th] pct over the slice draw.")
    rc_f = simulate(fwd, common=True); show("ten, best-CV-FIRST (w46a order)", rc_f)
    rc_r = simulate(rev, common=True); show("ten, best-CV-LAST  (reversed)", rc_r)
    rc_2 = simulate(fwd[:2], common=True); show("just the top TWO", rc_2)
    rc_1 = simulate(fwd[:1], common=True); show("just the top ONE", rc_1)

    print(f"\n=== JOINT, independent residuals (the other extreme — NOT the answer) ===")
    ri_f = simulate(fwd, common=False); show("ten, best-CV-FIRST (w46a order)", ri_f)
    ri_r = simulate(rev, common=False); show("ten, best-CV-LAST  (reversed)", ri_r)

    print(f"\n=== WHAT THE JOINT SAYS THAT THE SUM COULD NOT ===")
    sat = rc_1["uniform"][0] - rc_f["uniform"][0]
    print(f"  1. SATURATION. One file gets {BASE['uniform'] - rc_1['uniform'][0]:+.2f}e-6 of the "
          f"{BASE['uniform'] - rc_f['uniform'][0]:+.2f}e-6 the whole ten get. The other nine are "
          f"worth {sat:+.2f}e-6 between them on this axis.")
    dl = rc_f["latest"][0] - rc_r["latest"][0]
    print(f"  2. SEND ORDER. latest-first branch: best-CV-FIRST {rc_f['latest'][0]:+.2f}e-6 vs "
          f"best-CV-LAST {rc_r['latest'][0]:+.2f}e-6 — reversing is worth {dl:+.2f}e-6, free.")
    print(f"     uniform is order-invariant by construction ({rc_f['uniform'][0]:+.2f} vs "
          f"{rc_r['uniform'][0]:+.2f}); earliest-first is pinned by the 08-17 file.")
    worst_f = max(rc_f[t][0] for t in rc_f)
    worst_r = max(rc_r[t][0] for t in rc_r)
    print(f"  3. THE BRACKET, which is what actually decides: worst tiebreak branch is "
          f"{worst_f:+.2f}e-6 sending best-CV-first, {worst_r:+.2f}e-6 sending best-CV-last, "
          f"against {max(BASE['uniform'], BASE['latest'], BASE['earliest']):+.2f}e-6 sending nothing.")
    print(f"  4. AND IT IS ALL MOOT IF TEDDY CLICKS: the click sets every branch to 0.")
    print(f"\n  closed-form fallback fired {_fallback[0]} times "
          f"(scenarios w46a skipped at p < 1e-4)")

    json.dump(dict(ndraw=NDRAW, send_order=SEND, baseline=BASE,
                   common=dict(forward=rc_f, reversed=rc_r, top2=rc_2, top1=rc_1),
                   independent=dict(forward=ri_f, reversed=ri_r),
                   order_value_latest=dl, saturation=sat),
              open(os.path.join(HERE, "w46b_jointdrain.json"), "w"), indent=1, default=float)
    print("\nwrote experiments/w46b_jointdrain.json")


if __name__ == "__main__":
    main()
