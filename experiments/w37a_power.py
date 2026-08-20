"""w37a — POWER CALCULATION for the es-bias line, run BEFORE any slot is spent.

w36f fitted `offset = a + b*oof_auc` on THREE clean points (2 params, 1 dof) and read the
es-on-val inflation off the residual of the dirty members. That number (+2.7e-4) is a sketch.

This script asks one question and answers it with numbers that already exist on disk:
  How much does the prediction error at the quarantined members' AUCs shrink if we spend
  N of tomorrow's near-worthless queue-drain slots submitting RAW MEMBER test vectors?

No new member, no build, no network. Pure design.
"""
import numpy as np, pandas as pd, json, itertools

W = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"

# ---------- 1. our own 91 (cv, lb) points ----------
ours = pd.read_csv(W + "experiments/w25a_cvlb_full.csv").dropna(subset=["cv","gap"])
print(f"our own submitted files: {len(ours)}")
print(f"  cv span      {ours.cv.min():.7f} .. {ours.cv.max():.7f}  = {(ours.cv.max()-ours.cv.min())*1e4:.2f}e-4")
print(f"  gap mean     {ours.gap.mean():.7f}   sd {ours.gap.std():.3e}")

# fit gap ~ cv on ours alone
X = np.c_[np.ones(len(ours)), ours.cv.values]
y = ours.gap.values
beta_o, *_ = np.linalg.lstsq(X, y, rcond=None)
res_o = y - X @ beta_o
s2_o = res_o @ res_o / (len(ours) - 2)
cov_o = s2_o * np.linalg.inv(X.T @ X)
print(f"\nOUR-POINTS-ONLY fit: offset = {beta_o[0]:.6f} + {beta_o[1]:.6f} * oof")
print(f"  slope se {np.sqrt(cov_o[1,1]):.4f}   residual sd {np.sqrt(s2_o):.3e}")
print(f"  -> slope is {abs(beta_o[1]/np.sqrt(cov_o[1,1])):.2f} sigma from zero "
      f"({'RESOLVED' if abs(beta_o[1])>2*np.sqrt(cov_o[1,1]) else 'UNRESOLVED — as the journal said'})")

# ---------- 2. the w36f clean 3-point line ----------
es = json.load(open(W + "experiments/w36f_esbias.json"))
clean = [r for r in es["rows"] if r["clean"]]
dirty = [r for r in es["rows"] if not r["clean"]]
print(f"\nw36f clean calibration points: {len(clean)}")
for r in clean:
    print(f"  {r['member']:22s} oof {r['oof']:.6f}  lb {r['lb']:.5f}  offset {r['offset']:.6f}")

# ---------- 3. the design question ----------
# residual sd of a single LB reading. The LB grid is 1e-5; the journal's own pair-sd is 5-9e-6.
# Use OUR measured residual sd, which is the only honestly-estimated one here (89 dof).
SIGMA = np.sqrt(s2_o)
print(f"\nsingle-reading sd used for the design: {SIGMA:.3e}  (from our 91 points, 89 dof)")

TARGETS = {"omid_tabm": 0.9675077199, "zwr_realmlp": 0.9691277390, "mkt_mlp": 0.9414220911}

def pred_se(oof_pts, x0, sigma=SIGMA):
    """se of the FITTED LINE at x0 given calibration abscissae oof_pts."""
    Xc = np.c_[np.ones(len(oof_pts)), np.asarray(oof_pts)]
    try:
        C = sigma**2 * np.linalg.inv(Xc.T @ Xc)
    except np.linalg.LinAlgError:
        return np.inf
    v = np.array([1.0, x0])
    return float(np.sqrt(v @ C @ v))

now = [r["oof"] for r in clean]
print("\n---------- CURRENT INSTRUMENT (3 clean points) ----------")
print(f"  abscissae: {['%.6f'%v for v in now]}  span {(max(now)-min(now))*1e3:.2f}e-3")
for n, x0 in TARGETS.items():
    se = pred_se(now, x0)
    print(f"  se of predicted offset at {n:12s} (oof {x0:.6f}): {se:.3e}")

# candidates we could submit tomorrow. oof AUCs are measured and on disk.
CAND = {
    "mkt_realmlp": 0.9581337349,   # ext_members13, CLEAN (max_epochs reached 5/5), unused
    "ram_hgb":     0.9680258266,   # already a point via the author's title; OUR OWN send removes
    "ram_lgb":     0.9682590393,   # the attribution risk and is an independent reading
}
print("\n---------- WITH mkt_realmlp ADDED (1 slot) ----------")
plus = now + [CAND["mkt_realmlp"]]
print(f"  abscissae span {(max(plus)-min(plus))*1e3:.2f}e-3   dof {len(plus)-2}")
for n, x0 in TARGETS.items():
    se0, se1 = pred_se(now, x0), pred_se(plus, x0)
    print(f"  {n:12s}: {se0:.3e} -> {se1:.3e}   ({se0/se1:.2f}x tighter)")

print("\n---------- WITH mkt_realmlp + our own re-reads of both ram members (3 slots) ----------")
plus3 = now + [CAND["mkt_realmlp"], CAND["ram_hgb"], CAND["ram_lgb"]]
print(f"  abscissae span {(max(plus3)-min(plus3))*1e3:.2f}e-3   dof {len(plus3)-2}")
for n, x0 in TARGETS.items():
    se0, se1 = pred_se(now, x0), pred_se(plus3, x0)
    print(f"  {n:12s}: {se0:.3e} -> {se1:.3e}   ({se0/se1:.2f}x tighter)")

# ---------- 4. what has to be true for this to matter ----------
SHORTFALL = dirty[0]["shortfall"]   # zwr_realmlp, +2.73e-4
print(f"\n---------- IS THE ANSWER EVER IN DOUBT? ----------")
print(f"  the effect being measured (zwr_realmlp shortfall): {SHORTFALL:.3e}")
for lbl, pts in (("now", now), ("+1", plus), ("+3", plus3)):
    se = pred_se(pts, TARGETS["zwr_realmlp"])
    tot = np.sqrt(se**2 + SIGMA**2)   # line error + the dirty member's own reading error
    print(f"  {lbl:4s}: total se on the shortfall {tot:.3e}  -> effect is {SHORTFALL/tot:.1f} sigma")

# ================= 5. THE PART THAT IS ACTUALLY UNDER-POWERED =================
# The clean line is fine. What rests on ONE reading is the DIRTY side: zwr_realmlp is the
# only usable shortfall (tam_lkup's came out negative = mis-attributed LB). "es-on-val is
# worth +2.7e-4" is therefore n=1, and the question that decides whether the 15 quarantined
# members can be RESCUED by deflation is not the mean but the SPREAD across members.
#
# We hold the test vector of every quarantined member on disk. Submitting it ourselves gives
# an exact LB for exactly that vector -- no attribution ambiguity, which is the failure that
# killed tam_lkup. Each send is one independent shortfall reading.
print("\n" + "="*70)
print("THE DIRTY SIDE: what n independent shortfall readings buy")
print("="*70)
DIRTY_ON_DISK = {   # oof AUC on OUR frozen folds, measured, from the vet CSVs
    "om_xgb2":        0.9687331700, "om_flamlxgb":    0.9679800817,
    "om_tabtrans":    0.9674689212, "om_cnn":         0.9677056335,
    "om_flamllgb":    0.9663407223, "omid_tabm":      0.9675077199,
    "dm_cat":         0.9667001254, "dm_lgb":         0.9663919918,
    "dkv_xgb":        0.9644659266, "dkv_lgb":        0.9639537087,
    "dkv_cb":         0.9627085344, "kava_cat":       0.9637552596,
    "ravi_cb1c":      0.9639441019, "ravi_realmlp1c": 0.9646676405,
    "mkt_mlp":        0.9414220911,
}
print(f"  quarantined members with a test vector on disk: {len(DIRTY_ON_DISK)}")
print(f"  their oof span: {min(DIRTY_ON_DISK.values()):.6f} .. {max(DIRTY_ON_DISK.values()):.6f}")

print("\n  se of the LINE at each candidate, on the CURRENT 3 clean points vs +mkt_realmlp:")
plus1 = now + [CAND["mkt_realmlp"]]
for n, x0 in sorted(DIRTY_ON_DISK.items(), key=lambda kv: -kv[1]):
    a, b = pred_se(now, x0), pred_se(plus1, x0)
    flag = "  <-- line unusable, se > the 2.7e-4 effect" if a > SHORTFALL else ""
    print(f"    {n:16s} oof {x0:.6f}   {a:.3e} -> {b:.3e}{flag}")

print("\n  se on the MEAN es-on-val inflation, as dirty readings accumulate")
print("  (line error is COMMON to all readings and does not average down; the per-reading")
print("   error does. Using the tightest line se, 1.96e-05, as the common floor.)")
LINE = 1.961e-05
for n in (1, 2, 3, 4, 5, 6, 8):
    se = np.sqrt(LINE**2 + SIGMA**2 / n)
    print(f"    n={n}: se {se:.3e}   effect/se = {SHORTFALL/se:5.1f} sigma")

print("\n  BUT THE MEAN IS NOT THE QUESTION. To deflate a member you need the SPREAD:")
print("  with n readings the sd across members is estimated on n-1 dof. n=1 gives NOTHING,")
print("  n=2 gives a number with 1 dof, n>=5 gives a spread you can actually act on.")
print(f"  Detectable spread (chi-square, 95%, per-reading noise {SIGMA:.2e}):")
for n in (3, 4, 5, 6, 8):
    # chi-square based: sd_true detectable when (n-1)s^2/sigma^2 exceeds chi2_{0.95,n-1}
    from scipy.stats import chi2
    crit = chi2.ppf(0.95, n - 1)
    det = SIGMA * np.sqrt(max(crit / (n - 1) - 1, 0) + 1) * np.sqrt(crit / (n-1))
    print(f"    n={n}: can resolve a between-member sd above ~{det:.2e}"
          f"  ({det/SHORTFALL*100:.0f}% of the effect)")
