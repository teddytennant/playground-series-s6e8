"""w17e — the slot-2 ship, and it is a test of the paired instrument in the regime slot 1 LEFT OPEN.

Slot 1 (w17a/w17b) established that within the TIGHT families the observed CV->LB scatter is
fully explained by slice draw plus grid rounding (rms 7.4e-6 observed against 9.4e-6 predicted,
ratio 0.79) and that the apparent excess over all within-family pairs is "entirely hybrid and
logit (ratio 1.97)". It then tested the instrument out of sample ONCE, on a tight-family pair at
dCV ~ 0 (`w14a_repro159av_h3`, predicted 0.97105, printed 0.97105).

The loose regime has never been tested out of sample, and it is the only place the instrument is
known not to fit. `blend159av_logit` is the highest-CV logit file this workspace holds and has
never been sent. Its nearest sent sibling is `blend158_logit` (CV 4.0e-6 lower, LB 0.97106).

⚠ AND THE PUBLISHED LOGIT GAP IS CONTAMINATED, which this file found while setting up the test.
w17b's logit group gap of +1103.4e-6 (resid sd 32.1e-6, n=4) averages three comparable blend
files at +1099.0/+1085.4/+1080.3e-6 with `stack_pub88_mine_logit` at +1150.4e-6 — a foreign
public stack 30e-5 below the others on CV. Drop it and the logit gap is +1088.2e-6 with sd
9.7e-6, i.e. the "loose" logit family is not loose at all; one incomparable member made it look
that way. This is wave-summary §4.7 again (a number measured on one population and quoted for
another) and it is the THIRD time this wave.

THREE NAMED RIVAL MODELS, each falsifiable, each predicting a different grid value:

  A  PAIRED (the instrument slot 1 validated, and this file's registered prediction)
       LB = LB(blend158_logit) + dCV, with the reference's own rounding integrated out and
       the paired slice sd measured on this exact pair.
  B  PUBLISHED GROUP GAP  LB = CV + 1103.4e-6   (w17b's table, stack_pub88 included)
  C  FITTED SLOPE  the three blend logit files give dLB/dCV ~ 2.8 rather than 1; extrapolate it.

A is the workspace's current model and it is the one on the line here. If A prints and B/C do
not, the paired instrument has now been confirmed out of sample in both regimes. If B or C
prints, the loose-family excess slot 1 measured is real, A is wrong outside the tight families,
and every LB statement this workspace makes about a transform contrast needs re-deriving.

    .venv/bin/python experiments/w17e_logitpair.py --reps 2000
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import DATA, SUB, TARGET, load_raw  # noqa: E402
from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
SEED, F, GRID = 20260817, 0.20, 1e-5

SHIP = "blend159av_logit"
REF = "blend158_logit"
# ⚠ FIVE, not four. `stack_pub74_logit` is a SENT file that w17b never saw: the Kaggle CLI's
# submission list is PAGINATED and the account passed the page length on 2026-08-17, so the
# default page silently drops the oldest rows. Every API-reading script here now passes
# --page-size 200. w17b's logit family had 4 members; it has 5.
LOGIT_SENT = ["blend158_logit", "blend150sx_logit", "blend150fx_logit",
              "stack_pub88_mine_logit", "stack_pub74_logit"]
FOREIGN = ("stack_pub88_mine_logit", "stack_pub74_logit")   # public stacks, not our blends


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "500"],
                         capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    LB = sub.groupby(sub["fileName"].str.replace(r"\.csv$", "", regex=True))["publicScore"].max()
    assert SHIP not in LB.index, f"{SHIP} has already been sent — an identical resend is forbidden"
    print(f"live: {len(sub)} submissions; {SHIP} never sent")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    names = sorted(set(LOGIT_SENT + [SHIP]))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, v) for k, v in V.items()}

    # ------------------------------------------------- the contaminated group gap
    print(f"\n=== the logit family, and why its published gap is wrong ===")
    print(f"{'file':26s} {'CV':>14s} {'LB':>9s} {'gap':>10s}")
    gaps = {}
    for k in LOGIT_SENT:
        gaps[k] = LB[k] - cv[k]
        print(f"  {k:24s} {cv[k]:.10f} {LB[k]:>9.5f} {gaps[k]*1e6:>+9.1f}e-6")
    print(f"  {SHIP:24s} {cv[SHIP]:.10f} {'--':>9s}   <- never sent")
    g4 = float(np.mean([gaps[k] for k in LOGIT_SENT if k != "stack_pub74_logit"]))
    gall = float(np.mean(list(gaps.values())))
    blends = [k for k in LOGIT_SENT if k not in FOREIGN]
    g3 = float(np.mean([gaps[k] for k in blends]))
    s3 = float(np.std([gaps[k] for k in blends], ddof=1))
    print(f"\n  the 4 w17b saw (its published number)  gap {g4*1e6:+.1f}e-6")
    print(f"  all 5 including the paged-out file     gap {gall*1e6:+.1f}e-6  "
          f"sd {np.std(list(gaps.values()), ddof=1)*1e6:.1f}e-6")
    print(f"  the {len(blends)} comparable blend files            gap {g3*1e6:+.1f}e-6  "
          f"sd {s3*1e6:.1f}e-6")
    print(f"  -> the two FOREIGN public stacks move the family gap by {(gall-g3)*1e6:+.1f}e-6.")
    print(f"     They sit {(cv[blends[-1]]-max(cv[k] for k in FOREIGN))*1e6:.0f}e-6 and more "
          f"below the blends on CV. Drop them and the 'loose' logit family is not loose.")

    # ------------------------------------------------------ the paired slice sd
    print(f"\n=== paired slice sd for ({SHIP}, {REF}), {a.reps} draws ===")
    pk = {k: prep(V[k], y) for k in (SHIP, REF)}
    dpool = cv[SHIP] - cv[REF]
    rng = np.random.default_rng(SEED)
    n_pub = int(round(N_TEST * F))
    d = np.empty(a.reps)
    for i in range(a.reps):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool); m[idx[:n_pub]] = True
        d[i] = subset_auc(pk[SHIP], m) - subset_auc(pk[REF], m)
    sd = float((d - dpool).std(ddof=1))
    print(f"  dCV {dpool*1e6:+.3f}e-6   paired slice sd {sd*1e6:.3f}e-6   "
          f"P(slice reverses the CV ordering) {(np.sign(d) != np.sign(dpool)).mean():.3f}")

    # ------------------------------ model A: paired, integrating the reference's rounding
    # The reference's TRUE public score is uniform on [LB(ref)-5e-6, LB(ref)+5e-6); the ship's
    # true public is that plus a draw of the pair difference.  Marginalise both.
    print(f"\n=== MODEL A (registered): paired against {REF} at {LB[REF]:.5f} ===")
    rr = np.random.default_rng(SEED + 1)
    t_ref = LB[REF] + rr.uniform(-GRID / 2, GRID / 2, 200_000)
    dd = rr.normal(dpool, sd, 200_000)
    vals = np.round((t_ref + dd) / GRID) * GRID
    u, c = np.unique(np.round(vals, 5), return_counts=True)
    pA = {float(x): float(k / c.sum()) for x, k in zip(u, c) if k / c.sum() >= 0.001}
    for x in sorted(pA, key=lambda z: -pA[z]):
        print(f"    P(prints {x:.5f}) = {pA[x]:.3f}")
    modeA = max(pA, key=pA.get)
    lo = min(x for x in pA if pA[x] >= 0.005)
    hi = max(x for x in pA if pA[x] >= 0.005)
    print(f"  MODEL A POINT PREDICTION {modeA:.5f}   "
          f"support (P>=0.005) {lo:.5f}..{hi:.5f}")

    # ------------------------------------------------------------- models B and C
    predB = cv[SHIP] + g4
    print(f"\n=== MODEL B: published 4-file group gap {g4*1e6:+.1f}e-6 ===")
    print(f"  {cv[SHIP]:.7f} + {g4:.7f} = {predB:.7f} -> {round(predB, 5):.5f}")
    X = np.array([cv[k] for k in blends]); Y = np.array([LB[k] for k in blends])
    slope, icpt = np.polyfit(X, Y, 1)
    predC = icpt + slope * cv[SHIP]
    print(f"\n=== MODEL C: slope fitted on the 3 blend logit files ===")
    print(f"  dLB/dCV = {slope:.2f} (the instrument says 1.00) -> {predC:.7f} "
          f"-> {round(predC, 5):.5f}")
    predD = cv[SHIP] + g3
    print(f"\n  (for completeness, the DECONTAMINATED 3-file gap {g3*1e6:+.1f}e-6 "
          f"-> {round(predD, 5):.5f})")

    print(f"\n=== WHAT EACH MODEL IS BETTING ===")
    for tag, v in (("A paired (registered)", modeA), ("B published gap", round(predB, 5)),
                   ("C fitted slope", round(predC, 5)), ("D decontaminated gap", round(predD, 5))):
        print(f"    {tag:24s} {v:.5f}")
    print("  These are not all distinct; where two agree the result cannot separate them, and")
    print("  the honest reading of the outcome is written in the journal BEFORE the upload.")

    out = dict(ship=SHIP, ref=REF, cv={k: float(v) for k, v in cv.items()},
               lb_ref=float(LB[REF]), dcv=float(dpool), paired_sd=sd, reps=int(a.reps),
               gap_all4=g4, gap_blend3=g3, gap_blend3_sd=s3, slope_C=float(slope),
               pred_A=float(modeA), pred_A_dist=pA, pred_B=float(round(predB, 5)),
               pred_C=float(round(predC, 5)), pred_D=float(round(predD, 5)))
    json.dump(out, open(os.path.join(HERE, "w17e_logitpair.json"), "w"), indent=1, sort_keys=True)

    # -------------------------------------------------------- the artifact itself
    print(f"\n=== the artifact ===")
    s = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    f = pd.read_csv(os.path.join(SUB, f"{SHIP}.csv"))
    assert len(f) == N_TEST and f["id"].equals(s["id"]), "id column does not match sample_submission"
    v = f[TARGET].to_numpy(np.float64)
    assert np.isfinite(v).all()
    print(f"  {SHIP}.csv  rows {len(f):,}  ids match  finite  "
          f"distinct {len(np.unique(v)):,}  range [{v.min():.3e}, {v.max():.3f}]")
    from scipy.stats import rankdata
    r = rankdata(v)
    ident = []
    for fn in sorted(os.listdir(SUB)):
        if not fn.endswith(".csv") or fn[:-4] == SHIP or fn[:-4] not in LB.index:
            continue
        o = pd.read_csv(os.path.join(SUB, fn)).set_index("id").reindex(s["id"])[TARGET].to_numpy()
        if np.array_equal(rankdata(o), r):
            ident.append(fn)
    print(f"  rank-identical to an already-SENT file: {ident if ident else 'none — sendable'}")
    assert not ident, "rank-identical to a sent file; sending it would be a forbidden resend"
    print("\nwrote experiments/w17e_logitpair.json")


if __name__ == "__main__":
    main()
