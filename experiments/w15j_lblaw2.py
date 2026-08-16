"""w15j — quantitative validation of w15a's slice-noise law on LIVE leaderboard data.

v1 (w15j_lblaw.py) established the ordering: spearman(1-rho, |dLB|) = +0.73, p=4e-20 over
112 CV-matched pairs. This version gets the SCALE right, which v1 did not:

  1. For a zero-mean gaussian difference X, E|X| = sd(X)*sqrt(2/pi) = 0.7979*sd(X).
     v1 compared sd(|X|) against the law's sd(X) - wrong statistic, off by 1/0.603.
  2. The LB is quantised to 1e-5. In the high-rho buckets the true gap is far under one
     grid step, so most pairs read |dLB| = 0 exactly and the moment estimator is floored.
     Handled by forward-simulating the law THROUGH the quantiser instead of inverting it.

Under H0 the two files have the same full-test AUC and dLB ~ N(0, sd_law), rounded to the
1e-5 grid.  We report observed E|dLB| against simulated E|dLB| under exactly that null.
"""
import numpy as np, pandas as pd

SD_SINGLE = 0.0005673015477911883   # w15a, slice of 59,260 rows (f=0.20)
GRID = 1e-5
RNG = np.random.default_rng(20260815)

P = pd.read_csv("experiments/w15j_lblaw_pairs.csv")
Q = P[P.dcv < 5e-6].copy()
print(f"CV-matched pairs (|dCV| < 5e-6): {len(Q)}  of {len(P)} total")

edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
lbl = ["<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", ">1e-2"]
Q["bucket"] = pd.cut(1 - Q.rho, bins=edges, labels=lbl, include_lowest=True)

print("\n=== observed E|dLB| vs the law pushed through the 1e-5 quantiser ===")
print(f"{'bucket':<12}{'n':>4}{'mean(1-rho)':>13}{'law sd(X)':>12}"
      f"{'pred E|dLB|':>13}{'obs E|dLB|':>12}{'ratio':>8}{'z':>7}")
out = []
for b, sl in Q.groupby("bucket", observed=True):
    n = len(sl)
    sd_law = SD_SINGLE * np.sqrt(2 * (1 - sl.rho.values))          # per-pair, not pooled
    # forward-simulate: draw dLB under H0, quantise both endpoints to the grid
    reps = 4000
    draws = RNG.normal(0.0, 1.0, size=(reps, n)) * sd_law
    # both files' scores are rounded independently to the grid, so the difference of the
    # rounded scores is what the board shows; model as rounding the pair's two levels
    base = RNG.uniform(0, GRID, size=(reps, n))                    # unknown grid offset
    q = np.abs(np.round((base + draws) / GRID) - np.round(base / GRID)) * GRID
    pred = q.mean(1)                                               # E|dLB| per rep
    obs = sl.dlb.mean()
    z = (obs - pred.mean()) / pred.std(ddof=1)
    ratio = obs / pred.mean() if pred.mean() > 0 else np.nan
    print(f"{str(b):<12}{n:>4}{(1-sl.rho).mean():>13.3g}{sd_law.mean():>12.3g}"
          f"{pred.mean():>13.3g}{obs:>12.3g}{ratio:>8.2f}{z:>7.1f}")
    out.append(dict(bucket=str(b), n=n, obs=obs, pred=pred.mean(), ratio=ratio, z=z))

O = pd.DataFrame(out)
print("\nThe law is a one-parameter prediction with NOTHING fitted to the leaderboard:")
print(f"  sd_single = {SD_SINGLE:.4g} came from resampling LABELLED rows (w15a), never from LB.")
print(f"  median |ratio - 1| across buckets = {np.nanmedian(np.abs(O.ratio - 1)):.2f}")
O.to_csv("experiments/w15j_lblaw2.csv", index=False)

# --- what the law says about the thing the day was sent to explain ---
print("\n=== the reframe, applied to the board ===")
for name, rho, gap in [("MILANFX  0.97124", 0.9958, 180e-6),
                       ("rank2 el Ouahabi 0.97117", 0.9958, 110e-6),
                       ("Don Mani 0.97116", 0.9958, 100e-6),
                       ("Optimistix 0.97115", 0.9958, 90e-6),
                       ("Utkarsh 0.97113", 0.9958, 70e-6)]:
    sd = SD_SINGLE * np.sqrt(2 * (1 - rho))
    print(f"  {name:<26} gap {gap*1e6:5.0f}e-6   sd {sd*1e6:5.1f}e-6   = {gap/sd:4.2f} sigma")
print("  (rho 0.9958 = w15a's measured cross-team value for najiama's 18_blend, the only")
print("   other-team file with a real published ranking near the top. It is a stand-in.)")
