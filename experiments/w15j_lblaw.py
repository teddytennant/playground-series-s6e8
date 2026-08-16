"""w15j — validate w15a's cross-team noise law against LIVE leaderboard readings.

w15a's central reframe of the day: the resolving power of the public slice is not a
constant "5e-5 noise floor", it is a function of how correlated the two files are,
    sd(paired slice gap) = sd_single * sqrt(2 * (1 - rho)).
That law was established by resampling LABELLED rows (simulation). It has never been
checked against the leaderboard itself.

This account holds 36 scored files spanning rho 0.97..0.99999. That is a live test.

Design: take pairs of our own SENT files whose cross-fitted CV differs by < 5e-6, so the
true full-test gap is ~0 and any observed LB difference is slice draw + quantisation.
Bucket those pairs by test-space Spearman rho and compare the observed spread of the LB
difference against what the law predicts.

Scale-free version (the one that does not depend on the unknown public-slice fraction):
does |LB difference| grow monotonically with (1 - rho) among CV-matched pairs?
"""
import io, json, subprocess, itertools
import numpy as np, pandas as pd
from scipy.stats import rankdata

COMP = "playground-series-s6e8"
CVMATCH = 5e-6          # pairs closer than this in CV have ~zero true gap
SD_SINGLE = 0.0005673015477911883   # w15a, f=0.20 slice of 59,260 rows

raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
                     capture_output=True, text=True).stdout
sub = pd.read_csv(io.StringIO(raw))
sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
lb = sub.groupby("stem")["publicScore"].max()

audit = pd.read_csv("experiments/audit_results.csv").rename(columns={"name": "stem"})
cv = audit.set_index("stem")["cv"]

stems = [s for s in lb.index if s in cv.index]
print(f"scored files with a CV on the frozen folds: {len(stems)}")

# load test-space ranks once
R = {}
for s in stems:
    p = f"submissions/{s}.csv"
    try:
        d = pd.read_csv(p)
    except FileNotFoundError:
        print(f"  (no local file for {s}, skipped)"); continue
    col = [c for c in d.columns if c != "id"][0]
    R[s] = rankdata(d[col].values).astype(np.float32)
stems = sorted(R)
print(f"loaded {len(stems)} local prediction files, {len(R[stems[0]])} rows each")

# pairwise spearman via pearson on ranks
M = np.vstack([R[s] for s in stems])
M = (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
RHO = (M @ M.T) / M.shape[1]

rows = []
for i, j in itertools.combinations(range(len(stems)), 2):
    a, b = stems[i], stems[j]
    dcv = abs(cv[a] - cv[b])
    rows.append(dict(a=a, b=b, rho=RHO[i, j], dcv=dcv,
                     dlb=abs(lb[a] - lb[b])))
P = pd.DataFrame(rows)
P.to_csv("experiments/w15j_lblaw_pairs.csv", index=False)
print(f"\nall pairs: {len(P)}")

Q = P[P.dcv < CVMATCH].copy()
print(f"CV-matched pairs (|dCV| < {CVMATCH:.0e}): {len(Q)}")

# --- the scale-free test: |dLB| vs (1-rho), CV-matched pairs ---
edges = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
lbl = ["<1e-5", "1e-5..1e-4", "1e-4..1e-3", "1e-3..1e-2", ">1e-2"]
Q["bucket"] = pd.cut(1 - Q.rho, bins=edges, labels=lbl, include_lowest=True)
g = Q.groupby("bucket", observed=True).agg(
    n=("dlb", "size"), mean_1mrho=("rho", lambda v: (1 - v).mean()),
    mean_dlb=("dlb", "mean"), max_dlb=("dlb", "max"), sd_dlb=("dlb", "std"))
g["law_sd_gap"] = SD_SINGLE * np.sqrt(2 * g["mean_1mrho"])
print("\n=== CV-matched pairs: observed |LB difference| vs the law ===")
print(g.to_string(float_format=lambda v: f"{v:.3g}"))

if len(Q) > 3:
    from scipy.stats import spearmanr
    r, p = spearmanr(1 - Q.rho, Q.dlb)
    print(f"\nspearman( 1-rho , |dLB| ) over {len(Q)} CV-matched pairs = {r:+.4f}  p={p:.3g}")

# --- the top-cluster endpoint: zero-or-few-parameter files, all at one LB value ---
top = [s for s in stems if cv[s] > 0.970045]
print(f"\n=== top cluster (CV > 0.970045): {len(top)} files ===")
for s in sorted(top, key=lambda x: -cv[x]):
    print(f"  {s:22s} CV {cv[s]:.7f}  LB {lb[s]:.5f}")
tl = np.array([lb[s] for s in top])
sub_rho = [RHO[stems.index(a), stems.index(b)] for a, b in itertools.combinations(top, 2)]
print(f"  LB sd across the cluster: {tl.std(ddof=1):.3g}   distinct LB values: {sorted(set(tl))}")
print(f"  pairwise rho within cluster: min {min(sub_rho):.7f} median {np.median(sub_rho):.7f}")
print(f"  law's predicted sd(gap) at median rho: "
      f"{SD_SINGLE*np.sqrt(2*(1-np.median(sub_rho))):.3g}  (LB quantisation is 1e-5)")
