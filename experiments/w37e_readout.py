"""w37e — READ OUT the es-bias calibration sends and EVALUATE the w37c rules automatically.

Written and committed before any of the seven LB scores existed. Run it after the sends land:

    .venv/bin/python experiments/w37e_readout.py

It applies R1, R2 and R3 from `w37c_prereg.py` as written. The script decides; a run reading
its output does not get to re-derive the rule from the numbers. That is the whole point --
w36 §2 records a slot in which a 4-of-5 sign gate was drafted, fired, and then disagreed with
the workspace's own 5-of-5 standing criterion, and the only reason it was not resolved
post-hoc in favour of the preferred answer is that the rule had been fixed first.
"""
import numpy as np, pandas as pd, subprocess, io, json, sys

W = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"
COMP = "playground-series-s6e8"
GRID  = 1e-5
SIGMA = 3.764e-05        # per-reading sd, our 87 (cv,lb) points, 85 dof (w37a)
SD_RESOLVABLE = 8.93e-05 # between-member sd resolvable at n=5 (w37a chi-square)

pre = pd.read_csv(W + "experiments/w37c_prereg.csv")

# --- pull the live scores. PAGE SIZE IS NOT OPTIONAL: this account passed 50 submissions on
# --- 2026-08-17 and the CLI default silently truncates (RESEARCH.md, w17 slot 2).
out = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                      "--page-size", "500"], capture_output=True, text=True, timeout=600)
subs = pd.read_csv(io.StringIO(out.stdout))
if len(subs) == 500:
    sys.exit("submission list came back exactly at the page size -- it is truncated, refusing.")
subs = subs.dropna(subset=["publicScore"])
lb = subs.groupby("fileName").publicScore.max()

pre["lb"] = pre.file.map(lb)
landed = pre.dropna(subset=["lb"]).copy()
print(f"{len(landed)} of {len(pre)} calibration sends have landed\n")
if landed.empty:
    sys.exit("nothing to read yet.")
landed["shortfall"] = landed.pred_H0 - landed.lb

print(f"{'member':16s} {'role':13s} {'oof':>9} {'pred_H0':>9} {'actual':>9} {'shortfall':>11} {'steps':>7}")
for r in landed.itertuples():
    print(f"{r.member:16s} {r.role:13s} {r.oof_auc:9.6f} {r.pred_H0:9.6f} {r.lb:9.5f} "
          f"{r.shortfall:11.3e} {r.shortfall/GRID:7.1f}")

# ---------------- R1: the audit comes first, because a failure withdraws a claim ------------
print("\n" + "="*72 + "\nR1  CLEAN AUDIT -- ram_hgb against the title LB the line is fitted on\n" + "="*72)
a = landed[landed.member == "ram_hgb"]
if a.empty:
    print("  not landed yet. R2 and R3 below are PROVISIONAL until it does.")
else:
    got, want = a.lb.iloc[0], 0.96945
    d = abs(got - want)
    print(f"  our send {got:.5f}   author's title {want:.5f}   |diff| {d/GRID:.1f} steps")
    if d <= 1.5 * GRID:
        print("  ✅ PASS. The title LB belongs to that vector; the w36f line's foundation holds.")
    else:
        print("  ⛔ FAIL. The title LB does not belong to that vector -- the `tam_lkup` failure")
        print("     mode, now caught on a point the line is FITTED on. Per R1 the +2.7e-4")
        print("     es-on-val figure is WITHDRAWN from RESEARCH.md pending a refit that drops")
        print("     every title-sourced point and keeps only self-submitted ones.")

# ---------------- R2: does the linear form survive extrapolation? ---------------------------
print("\n" + "="*72 + "\nR2  CLEAN ANCHOR -- mkt_realmlp, 9.9e-3 below the fitted range\n" + "="*72)
b = landed[landed.member == "mkt_realmlp"]
if b.empty:
    print("  not landed yet.")
else:
    se = 6.359e-04 * 0 + 3.706e-05   # se of the line at 0.958 is dominated by extrapolation;
    # recompute honestly from the three fitted abscissae rather than quoting a target's value
    Xc = np.c_[np.ones(3), [0.9680258266, 0.9682590393, 0.9701182875]]
    C = SIGMA**2 * np.linalg.inv(Xc.T @ Xc)
    v = np.array([1.0, b.oof_auc.iloc[0]])
    se = float(np.sqrt(v @ C @ v))
    d = b.shortfall.iloc[0]
    print(f"  predicted {b.pred_H0.iloc[0]:.6f}  actual {b.lb.iloc[0]:.5f}  miss {d:.3e}"
          f"   line se here {se:.3e}  ->  {abs(d)/se:.2f} sigma")
    if abs(d) <= 2 * se:
        print("  ✅ The linear offset form survives a 10e-3 extrapolation. The low-AUC dirty")
        print("     readings (ravi_*, dkv_*, kava_*) are interpretable.")
    else:
        print("  ⚠ The offset is NOT linear in AUC over this range. Per R2, restrict R3 to the")
        print("     dirty members above oof 0.966 and report the rest as unpriced.")

# ---------------- R3: the number that has never existed -- the SPREAD -----------------------
print("\n" + "="*72 + "\nR3  DIRTY READINGS -- mean, and the between-member spread\n" + "="*72)
d = landed[landed.role == "DIRTY"]
n = len(d)
print(f"  n = {n} of 5 planned")
if n == 0:
    print("  none landed yet.")
else:
    m, s = d.shortfall.mean(), (d.shortfall.std(ddof=1) if n > 1 else np.nan)
    LINE = 1.961e-05
    se_m = np.sqrt(LINE**2 + SIGMA**2 / n)
    print(f"  mean shortfall {m:.3e}  ({m/GRID:.1f} steps)   se {se_m:.3e}  -> {m/se_m:.1f} sigma")
    print(f"  w36f's single reading was +2.729e-04; this mean is "
          f"{'CONSISTENT' if abs(m-2.729e-4) < 2*se_m else 'INCONSISTENT'} with it.")
    if n < 2 or not np.isfinite(s):
        print("  spread: needs n>=2. UNDECIDED.")
    else:
        print(f"  sd across members {s:.3e}  = {s/m*100:.0f}% of the mean  ({n-1} dof)")
        if n < 5:
            print(f"  ⚠ n={n} < 5. w37a says a between-member sd is only resolvable above")
            print(f"     ~{SD_RESOLVABLE:.2e} at n=5; below that this number is not yet a verdict.")
        if s < 0.33 * m:
            print("  ✅ R3 -> es-on-val is a roughly COMMON bias. A deflation arm is worth")
            print("     building: shrink each quarantined member's OOF column until its AUC")
            print("     drops by the mean shortfall, then refit the pack with all 15 in.")
        else:
            print("  ⛔ R3 -> the bias is MEMBER-SPECIFIC. No single deflation constant is")
            print("     right, and THE QUARANTINE STANDS. This was pre-registered as at least")
            print("     as likely as the other branch. It costs nothing; it closes a line.")
    if any(d.shortfall < 0):
        neg = d[d.shortfall < 0].member.tolist()
        print(f"  ⚠ NEGATIVE shortfall on {neg}. On a SELF-SUBMITTED vector this cannot be the")
        print("     mis-attribution that explained tam_lkup -- the LB provably belongs to these")
        print("     predictions. It means the line is wrong at that AUC, not the member.")

landed.to_csv(W + "experiments/w37e_readout.csv", index=False)
print(f"\nwrote experiments/w37e_readout.csv")
