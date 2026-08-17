"""w17 slot 3 ship — the paired CV->LB instrument tested in the OTHER contaminated family.

Slot 2 found that w17b's published per-family CV->LB gaps are contaminated wherever a foreign
`stack_pub*` file sits in the group, and named TWO such families: logit and hybrid.  It then
tested the logit one out of sample (`blend159av_logit`, predicted 0.97106, printed 0.97106) and
left hybrid as an open item -- next-run item 4, "re-run w17b_famfix with stack_pub* EXCLUDED,
not merely added".  This closes it with a live test rather than a table edit.

The hybrid family is the better test of the two, for a reason the logit family could not offer:
its group gap is genuinely noisy (five comparable sent files at +1001.7 / +993.9 / +975.8 /
+971.7 / +965.7e-6, sd ~15e-6 against the logit blends' 9.6e-6), and there is no single obvious
decontamination.  Three defensible choices of "which files are comparable" name THREE DIFFERENT
grid values, and the paired form names a fourth position.  So the outcome discriminates instead
of confirming.

SHIP: blend159av_hybrid, CV 0.9700291725, the highest-CV hybrid file this account holds and
never sent.  NOT a leaderboard candidate -- 26.5e-6 below the CV pick, and the hybrid transform
has never once beaten h3 here.  It is sent as an instrument test, which is what the slot is for.

RIVAL MODELS, all registered before the upload:
  A  PAIRED (registered)   LB(blend158_hybrid) + dCV, the reference's own rounding integrated
                           out, paired slice sd measured on THIS pair.
  B  GROUP GAP, all 6 sent hybrid files (the contaminated form, what w17b would print)
  C  GROUP GAP, 5 files -- drop only stack_pub86_hybrid, which is 340e-6 below on CV
  D  GROUP GAP, our 3 blend files only (the strictest decontamination)
  E  A's point value, but with the paired sd from w17h's LAW-IF instead of from simulation.
     E cannot move the mode; it is registered to test whether the influence-function law is
     usable OPERATIONALLY, i.e. whether it reproduces a simulated paired sd it never saw.

    .venv/bin/python experiments/w17j_hybridpair.py --reps 2000
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402
from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
SEED, F, GRID = 20260817, 0.20, 1e-5

SHIP = "blend159av_hybrid"
REF = "blend158_hybrid"
HYBRID_SENT = ["blend158_hybrid", "stack_pub151_hybrid", "stack_pub149_hybrid",
               "blend150sx_hybrid", "blend150fx_hybrid", "stack_pub86_hybrid"]
OURS = ["blend158_hybrid", "blend150sx_hybrid", "blend150fx_hybrid"]
FAR = "stack_pub86_hybrid"          # 340e-6 below the rest on CV


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def dist_from(point_delta, ref_lb, sd, rng):
    """Marginalise the reference's own +/-5e-6 rounding and the paired slice draw."""
    t_ref = ref_lb + rng.uniform(-GRID / 2, GRID / 2, 400_000)
    dd = rng.normal(point_delta, sd, 400_000)
    vals = np.round(np.round((t_ref + dd) / GRID) * GRID, 5)
    u, c = np.unique(vals, return_counts=True)
    return {float(x): float(k / c.sum()) for x, k in zip(u, c) if k / c.sum() >= 0.0005}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    LB = sub.groupby(sub["fileName"].str.replace(r"\.csv$", "", regex=True))["publicScore"].max()
    assert SHIP not in LB.index, f"{SHIP} already sent — an identical resend is forbidden"
    print(f"live: {len(sub)} submissions on the account; {SHIP} never sent")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    names = sorted(set(HYBRID_SENT + [SHIP]))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, v) for k, v in V.items()}

    print("\n=== the hybrid family ===")
    print(f"{'file':28s} {'CV':>14s} {'LB':>9s} {'gap':>11s}")
    gaps = {k: LB[k] - cv[k] for k in HYBRID_SENT}
    for k in HYBRID_SENT:
        print(f"  {k:26s} {cv[k]:.10f} {LB[k]:>9.5f} {gaps[k]*1e6:>+9.1f}e-6")
    print(f"  {SHIP:26s} {cv[SHIP]:.10f} {'never':>9s}   <- the ship")

    g_all = float(np.mean([gaps[k] for k in HYBRID_SENT]))
    five = [k for k in HYBRID_SENT if k != FAR]
    g_5 = float(np.mean([gaps[k] for k in five]))
    s_5 = float(np.std([gaps[k] for k in five], ddof=1))
    g_ours = float(np.mean([gaps[k] for k in OURS]))
    s_ours = float(np.std([gaps[k] for k in OURS], ddof=1))
    print(f"\n  all 6 (contaminated)          {g_all*1e6:+.1f}e-6")
    print(f"  5, dropping {FAR:22s} {g_5*1e6:+.1f}e-6  sd {s_5*1e6:.1f}e-6")
    print(f"  our 3 blends only             {g_ours*1e6:+.1f}e-6  sd {s_ours*1e6:.1f}e-6")
    print(f"  -> the one far file moves the family gap by {(g_all-g_5)*1e6:+.1f}e-6")

    # ------------------------------------------------------- paired slice sd, simulated
    print(f"\n=== paired slice sd for ({SHIP}, {REF}), {a.reps} draws ===")
    pk = {k: prep(V[k], y) for k in (SHIP, REF)}
    dpool = cv[SHIP] - cv[REF]
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    d = np.empty(a.reps)
    for i in range(a.reps):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[:n_pub]] = True
        d[i] = subset_auc(pk[SHIP], m) - subset_auc(pk[REF], m)
    sd_sim = float((d - dpool).std(ddof=1))
    rho = float(np.corrcoef(rankdata(V[SHIP]), rankdata(V[REF]))[0, 1])
    print(f"  dCV {dpool*1e6:+.3f}e-6   rho {rho:.6f}   simulated paired slice sd "
          f"{sd_sim*1e6:.3f}e-6")

    # ------------------------------------------- the same sd from LAW-IF, no simulation
    pos, neg = y == 1, y == 0
    n1p, n0p = n_pub * float(y.mean()), n_pub * (1 - float(y.mean()))
    Fp, Fn = {}, {}
    for k in (SHIP, REF):
        Fp[k] = midrank_cdf(np.sort(V[k][neg]), V[k][pos])
        Fn[k] = midrank_cdf(np.sort(V[k][pos]), V[k][neg])
    var_if = ((Fp[SHIP] - Fp[REF]).var(ddof=1) / n1p
              + (Fn[REF] - Fn[SHIP]).var(ddof=1) / n0p)
    sd_if = float(np.sqrt(var_if * (1.0 - n_pub / n)))
    print(f"  LAW-IF sd (zero simulation, w17h)              {sd_if*1e6:.3f}e-6"
          f"   ratio {sd_if/sd_sim:.3f}")
    print(f"  -> LAW-IF is {'USABLE' if abs(sd_if/sd_sim - 1) < 0.10 else 'NOT usable'} "
          f"here (registered: within 10%)")

    # ------------------------------------------------------------------- the four models
    rr = np.random.default_rng(SEED + 1)
    pA = dist_from(dpool, float(LB[REF]), sd_sim, rr)
    modeA = max(pA, key=pA.get)
    print(f"\n=== MODEL A (registered): paired against {REF} at {LB[REF]:.5f} ===")
    for x in sorted(pA, key=lambda z: -pA[z]):
        print(f"    P(prints {x:.5f}) = {pA[x]:.3f}")
    pE = dist_from(dpool, float(LB[REF]), sd_if, np.random.default_rng(SEED + 2))
    modeE = max(pE, key=pE.get)

    preds = {}
    for tag, g in [("B", g_all), ("C", g_5), ("D", g_ours)]:
        preds[tag] = float(np.round((cv[SHIP] + g) / GRID) * GRID)
    print(f"\n=== the rival models ===")
    print(f"  A paired (registered)                    {modeA:.5f}   P {pA[modeA]:.3f}")
    print(f"  B group gap, all 6 (contaminated)        {preds['B']:.5f}")
    print(f"  C group gap, 5 (drop the far file)       {preds['C']:.5f}")
    print(f"  D group gap, our 3 blends only           {preds['D']:.5f}")
    print(f"  E paired with the LAW-IF sd              {modeE:.5f}   P {pE[modeE]:.3f}")
    vals = sorted({round(v, 5) for v in (modeA, preds['B'], preds['C'], preds['D'])})
    print(f"  -> {len(vals)} distinct predicted values: "
          f"{', '.join(f'{v:.5f}' for v in vals)}"
          f"   {'DISCRIMINATING' if len(vals) > 1 else 'NOT discriminating'}")

    # ------------------------------------------------------------------- tier risk
    print(f"\n=== tier risk, priced BEFORE the send ===")
    p8 = sum(v for k, v in pA.items() if k >= 0.97108 - 1e-9)
    print(f"  auto-slot 1 tier is 0.97108 (3-way). P(this file reaches it) = {p8:.4f}")
    print(f"  -> the send {'CANNOT' if p8 < 1e-3 else 'COULD'} change the auto-pick tiers.")

    # ------------------------------------------------------------------- artifact audit
    print(f"\n=== artifact audit ===")
    ss = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    df = pd.read_csv(os.path.join(SUB, f"{SHIP}.csv"))
    assert len(df) == N_TEST and (df["id"].to_numpy() == ss["id"].to_numpy()).all()
    v = df["addicted_label"].to_numpy()
    assert np.isfinite(v).all()
    print(f"  {len(df):,} rows, ids match sample_submission in order, all finite, "
          f"{len(np.unique(v)):,} distinct")
    rk = rankdata(v)
    ident = []
    for fn in sorted(os.listdir(SUB)):
        if not fn.endswith(".csv") or fn[:-4] == SHIP or fn[:-4] not in LB.index:
            continue
        o = pd.read_csv(os.path.join(SUB, fn))["addicted_label"].to_numpy()
        if len(o) == N_TEST and np.array_equal(rk, rankdata(o)):
            ident.append(fn)
    print(f"  rank-identical to an already-sent file? {ident if ident else 'no'}")
    assert not ident, "rank-identical to a sent file — that send would be pointless"

    json.dump(dict(ship=SHIP, ref=REF, reps=a.reps, cv=cv, lb_ref=float(LB[REF]),
                   dcv=dpool, rho=rho, sd_sim=sd_sim, sd_if=sd_if,
                   sd_if_ratio=sd_if / sd_sim, gap_all6=g_all, gap_5=g_5, gap_ours=g_ours,
                   gap_5_sd=s_5, gap_ours_sd=s_ours,
                   pred_A=modeA, pred_A_dist=pA, pred_B=preds["B"], pred_C=preds["C"],
                   pred_D=preds["D"], pred_E=modeE, p_reaches_tier=p8),
              open(os.path.join(HERE, "w17j_hybridpair.json"), "w"), indent=1)
    print(f"\nwrote w17j_hybridpair.json")


if __name__ == "__main__":
    main()
