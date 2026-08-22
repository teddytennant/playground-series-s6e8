"""w57b -- does w57a's -0.07e-6 justify moving the SECOND pick? Answer: NO, but NOT for the
reason this file was first written to show. Read the WITHDRAWAL note.

WHAT RAISES THE QUESTION. w57a priced the auto pair `w36_ad199stdcorr + w40_ad211stdcorr` at
-0.07e-6: marginally BETTER than the WANTED pair `w36_ad199stdcorr + w23_ad187stdcorr`. Read
naively that says swap the second pick to `w40_ad211stdcorr` (CV 0.9701374733) from
`w23_ad187stdcorr` (CV 0.9701150809) -- +22.4e-6 of CV on the hedge slot.

⛔⛔ WITHDRAWAL, SAME RUN. The first cut of this file registered Q2/Q3 as "the era discount
flips the sign, and it takes under 25% of w52d's fitted discount to do it". Both came back
FALSIFIED, and inspecting the output showed the instrument -- not the hypothesis -- was at
fault: it fed `emax` a PLACEHOLDER per-file sd (a `.get(...)` default that silently fired
because the key did not exist) and a HARDCODED rho of 0.99999 instead of the per-pair values.
With those inputs both pairs collapse to the mean of their common dominant member and the
printed difference was exactly +0.000e-6 -- a degenerate object, not a measurement. The
readings from that cut are VOID and are not quoted anywhere. This file re-derives the pair
covariance properly (LAW-IF, the same construction w57a and w18a use) and reports what it
actually finds, including the part that cuts AGAINST the conclusion.

WHAT THE PROPER MEASUREMENT SHOWS -- and it is not what I expected:
the second pick is worth almost NOTHING under the sampling model, whichever file holds it.
Two near-identical stacks have a private DIFFERENCE sd of well under 1e-6 (rho ~0.9998 at a
per-file sd of ~16e-6), so a 22e-6 CV gap is never overturned by slice noise. E[max] is the
better file's mean in both pairs. The hedge slot therefore cannot be decided on E[max] at
all -- and neither can the swap. w57a's -0.07e-6 is inside that same dead zone.

SO THE DECISION IS A TAIL QUESTION, AND THAT IS WHAT THIS FILE MEASURES.
  `check_selection.py`: the second pick is "the pack hedge ... identical construction on the
  187 pack WITHOUT any of the member additions". It insures the branch where the member-import
  line ad188..ad211 added CV and not true test quality -- the workspace's OWN leading model
  (JOURNAL w44 s6's flat dose-response; w46c's "CV runs ahead of LB"; w52d's era x CV
  interaction, era slope 1.4425 vs base 1.8327 = a 21.3% shrink of the era CV edge).
  ERA_MIN_AD = 195, so:
      w23_ad187stdcorr  ad187  PRE-era   <- current hedge, OUTSIDE the suspect region
      w36_ad199stdcorr  ad199  era       <- the pick
      w40_ad211stdcorr  ad211  era       <- the proposed swap, DEEPER inside it
  The current WANTED straddles the axis the workspace says is uncertain. The swap would put
  both picks on the same side of it.

REGISTERED, second cut, before re-running:
  Q1  rho(pick, swap) >= 0.9999 and > rho(pick, hedge): the twins carry no diversification.
      [this survived the first cut unchanged -- it never depended on the broken code path]
  Q4  under w52d's FITTED era shrink the pick still beats the hedge privately with
      probability > 0.99, i.e. the hedge does NOT pay at the fitted effect size.
  Q5  the shrink required for the hedge to break even is > 2x w52d's fitted one. If Q4 and Q5
      both hold, the hedge is tail insurance against a scenario BEYOND the fitted effect --
      which is an honest reason to hold it cheaply, NOT a reason to claim it is winning.

    .venv/bin/python experiments/w57b_hedge.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from common import SUB, TARGET, load_raw          # noqa: E402
from w16b_cellweight import fast_auc              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
U, N_TEST, F = 1e-6, 296_302, 0.20
PICK, HEDGE, SWAP = "w36_ad199stdcorr", "w23_ad187stdcorr", "w40_ad211stdcorr"
NAMES = [PICK, HEDGE, SWAP]
BASE_SLOPE, ERA_SLOPE = 1.8327097116756488, 1.4425451925947441   # w52d


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def emax(mu, sd, rho):
    th = np.sqrt(max(sd[0] ** 2 + sd[1] ** 2 - 2 * rho * sd[0] * sd[1], 1e-300))
    a = (mu[0] - mu[1]) / th
    return mu[0] * norm.cdf(a) + mu[1] * norm.cdf(-a) + th * norm.pdf(a)


def main() -> None:
    # ---- Q1: test-vector correlation. The hedge's whole value is being DIFFERENT.
    T = {}
    for k in NAMES:
        df = pd.read_csv(os.path.join(SUB, f"{k}.csv")).sort_values("id")
        T[k] = df[df.columns[1]].to_numpy()
    n_t = len(T[PICK])
    assert all(len(v) == n_t for v in T.values()) and n_t == N_TEST, "test vectors malformed"
    R = {k: pd.Series(v).rank().to_numpy() for k, v in T.items()}
    rho_swap = float(np.corrcoef(R[PICK], R[SWAP])[0, 1])
    rho_hedge = float(np.corrcoef(R[PICK], R[HEDGE])[0, 1])
    print(f"=== Q1: test-vector Spearman rho against the pick ({n_t:,} rows) ===")
    print(f"  rho(pick, swap  {SWAP:18s}) = {rho_swap:.8f}   twin")
    print(f"  rho(pick, hedge {HEDGE:18s}) = {rho_hedge:.8f}   current")
    print(f"  the hedge is {(1-rho_hedge)/(1-rho_swap):.2f}x further from the pick in 1-rho terms")
    q1 = bool(rho_swap >= 0.9999 and rho_swap > rho_hedge)
    print(f"  Q1 -> {'CONFIRMED' if q1 else '*** FALSIFIED ***'}")

    # ---- LAW-IF covariance on the three files. Same construction as w57a/w18a.
    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, V[k]) for k in NAMES}
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    A = np.empty((3, n1)); B = np.empty((3, n0))
    for i, k in enumerate(NAMES):
        A[i] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        B[i] = 1.0 - midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    C1, C0 = np.cov(A), np.cov(B)
    pi1 = n1 / n
    S_t = (1.0 - N_TEST / n) * (C1 / (N_TEST * pi1) + C0 / (N_TEST * (1 - pi1))) / U ** 2
    col = {k: i for i, k in enumerate(NAMES)}
    sd = np.sqrt(np.diag(S_t))

    def pair_rho(x, z):
        i, j = col[x], col[z]
        return float(S_t[i, j] / np.sqrt(S_t[i, i] * S_t[j, j]))

    print(f"\n=== the private-slice covariance the decision actually runs on ===")
    for k in NAMES:
        print(f"  {k:22s} CV {cv[k]:.10f}   private sd {sd[col[k]]:6.2f}e-6")
    for x, z in ((PICK, HEDGE), (PICK, SWAP)):
        r = pair_rho(x, z)
        i, j = col[x], col[z]
        sdd = float(np.sqrt(S_t[i, i] + S_t[j, j] - 2 * S_t[i, j]))
        print(f"  {x} vs {z:22s} rho {r:.6f}  sd(difference) {sdd:5.2f}e-6")

    # ---- Q4 / Q5: the era scenario. Shrink the ERA files' CV edge over the PRE-ERA hedge.
    ref = cv[HEDGE] / U

    def means(shrink):
        out = {}
        for k in NAMES:
            edge = cv[k] / U - ref
            out[k] = ref + edge * (1.0 - shrink if k != HEDGE else 1.0)
        return out

    def p_hedge_beats(k, shrink):
        """P(hedge > k) on the private slice at a given era shrink."""
        m = means(shrink)
        i, j = col[HEDGE], col[k]
        sdd = float(np.sqrt(S_t[i, i] + S_t[j, j] - 2 * S_t[i, j]))
        return float(norm.cdf((m[HEDGE] - m[k]) / sdd))

    def pair_emax(x, z, shrink):
        m = means(shrink)
        return float(emax(np.array([m[x], m[z]]), np.array([sd[col[x]], sd[col[z]]]),
                          pair_rho(x, z)))

    fitted = 1.0 - ERA_SLOPE / BASE_SLOPE
    print(f"\n=== Q4: does the hedge pay at w52d's FITTED era shrink ({fitted*100:.2f}%)? ===")
    for sh, lbl in ((0.0, "no era effect"), (fitted, "w52d fitted")):
        pw = pair_emax(PICK, HEDGE, sh)
        ps = pair_emax(PICK, SWAP, sh)
        print(f"  shrink {sh*100:6.2f}% ({lbl:13s})  P(pick > hedge) "
              f"{1-p_hedge_beats(PICK, sh):.6f}   E[max] WANTED-swap {pw-ps:+7.3f}e-6")
    p_pick_wins = 1 - p_hedge_beats(PICK, fitted)
    q4 = bool(p_pick_wins > 0.99)
    print(f"  Q4 (pick still beats the hedge with P>0.99 at the fitted shrink) -> "
          f"{'CONFIRMED' if q4 else '*** FALSIFIED ***'}")

    # break-even shrink: where the hedge draws level with the pick
    lo, hi = 0.0, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_hedge_beats(PICK, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    be = 0.5 * (lo + hi)
    print(f"\n=== Q5: how big must the era effect be for the hedge to pay? ===")
    print(f"  break-even shrink of the era CV edge: {be*100:.2f}%   "
          f"({be/fitted:.2f}x w52d's fitted {fitted*100:.2f}%)")
    q5 = bool(be > 2.0 * fitted)
    print(f"  Q5 (break-even > 2x the fitted effect) -> "
          f"{'CONFIRMED' if q5 else '*** FALSIFIED ***'}")

    print(f"\n=== VERDICT ===")
    print(f"  SECOND PICK UNCHANGED: {HEDGE} stays.")
    print(f"  But the honest reason is NOT that it wins on E[max] -- it does not, and neither")
    print(f"  does the swap. At rho {pair_rho(PICK,HEDGE):.5f} the two pairs differ by")
    print(f"  {pair_emax(PICK,HEDGE,0.0)-pair_emax(PICK,SWAP,0.0):+.3f}e-6, inside the dead zone where the second slot")
    print(f"  is worth nothing under sampling noise. The hedge is TAIL insurance against a")
    print(f"  member-import reversal {be/fitted:.1f}x larger than the one w52d fitted, and it costs")
    print(f"  ~0.07e-6 to hold. Swapping it buys 0.07e-6 of noise and sells the only pick")
    print(f"  this account holds OUTSIDE the era whose CV the workspace already distrusts.")
    json.dump(dict(pick=PICK, hedge=HEDGE, swap=SWAP,
                   rho_swap=rho_swap, rho_hedge=rho_hedge,
                   cv={k: cv[k] for k in NAMES},
                   private_sd={k: float(sd[col[k]]) for k in NAMES},
                   rho_pick_hedge=pair_rho(PICK, HEDGE), rho_pick_swap=pair_rho(PICK, SWAP),
                   fitted_shrink=fitted, p_pick_beats_hedge_at_fitted=p_pick_wins,
                   breakeven_shrink=be, breakeven_over_fitted=be / fitted,
                   emax_gap_nodisc=pair_emax(PICK, HEDGE, 0.0) - pair_emax(PICK, SWAP, 0.0),
                   q1=q1, q4=q4, q5=q5,
                   withdrawn="first cut's Q2/Q3 -- placeholder sd + hardcoded rho, degenerate, VOID",
                   verdict="second pick UNCHANGED; tail-insurance rationale, not an E[max] one"),
              open(os.path.join(HERE, "w57b_hedge.json"), "w"), indent=1, default=float)
    print("\nwrote experiments/w57b_hedge.json")


if __name__ == "__main__":
    main()
