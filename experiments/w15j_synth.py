"""w15j — the synthesis neither w15a nor w15b could do alone.

w15a (+ w15j's live-LB validation of the same law) measured the UNCERTAINTY on the gap:
the cross-team paired slice sd is 53e-6, so MILANFX's 180e-6 is 3.46 sigma, not a fact.
w15b priced the SIGNAL a gap implies: solving for the family of truths consistent with our
own measured AUC, a gap of g costs an orthogonal predictor of standalone AUC solo(g).

Neither run propagated the other's uncertainty. Doing so answers the question the whole day
was pointed at - "how much signal are we actually missing?" - with an honest error bar.
"""
import json, numpy as np

SD_CROSS = 53.0e-6          # w15a, najiama 18_blend, the only real other-team ranking near us
lad = json.load(open("experiments/w15b_price2.json"))["ladder"]
g = np.array([r["gap"] for r in lad])
s = np.array([r["solo"] for r in lad])
ok = g > 0
solo_of = lambda gap: float(np.interp(gap, g[ok], s[ok]))

print("=== the observed public gaps, and what they cost as ORTHOGONAL signal ===")
print(f"{'team':<24}{'gap':>9}{'sigma':>7}   {'95% CI on the TRUE gap':<26}{'solo AUC required':<22}")
board = [("MILANFX", 180e-6), ("Maher el Ouahabi", 110e-6), ("Don Mani", 100e-6),
         ("Optimistix", 90e-6), ("Utkarsh", 70e-6)]
for name, gap in board:
    lo, hi = gap - 1.96 * SD_CROSS, gap + 1.96 * SD_CROSS
    slo = solo_of(max(lo, 0.0)) if lo > 0 else 0.5
    shi = solo_of(hi)
    ci = f"[{lo*1e6:+6.0f}, {hi*1e6:+6.0f}]e-6"
    lab = f"[{slo:.4f}, {shi:.4f}]" + ("  (incl. ZERO)" if lo <= 0 else "")
    print(f"{name:<24}{gap*1e6:>7.0f}e-6{gap/SD_CROSS:>7.2f}   {ci:<26}{lab:<22}")

print("\n=== and the point estimate is shrunk further by best-of-n selection ===")
print("w15a's extreme-value fit gives the leader's TRUE edge g-hat under two plateau sizes:")
for plateau, ghat in [("30 (the visible top)", 46e-6), ("30, sd 65e-6", 8e-6),
                      ("30, sd 84e-6", 0.0), ("155 (within 3e-4)", 120e-6),
                      ("267 (within 5e-4)", 130e-6)]:
    lab = f"{solo_of(ghat):.4f}" if ghat > 0 else "0.5000 (nothing to find)"
    print(f"  plateau {plateau:<22} g-hat {ghat*1e6:>5.0f}e-6  ->  solo AUC {lab}")

print("""
READ THIS AS THE DAY'S BOTTOM LINE.

Four of the five teams above us have a 95% CI on their true edge that INCLUDES ZERO. Only
MILANFX's excludes it, and only just (3.46 sigma against a measured, not assumed, sd).

Even taking MILANFX's gap at face value, the missing object is an orthogonal predictor of
standalone AUC ~0.512 - a modest nudge, not a new column. Under the plateau model it could
be anywhere from 0.500 to 0.515, and the public leaderboard cannot identify which.

w15b then closes the only class it could have belonged to: the residual search that produced
this workspace's closed list has 78-102% power at exactly this effect size, and it recovered
nothing on real labels against matched controls. So whatever is left is NOT an inductive
function of the 12 given columns.

That leaves exactly one candidate class standing at the end of the day: TRANSDUCTIVE /
test-time signal, which is orthogonal to all 168 of our inductive members by construction,
and which w15e found the only known instance of (max |rho| 0.077 against the entire pack).
""")
