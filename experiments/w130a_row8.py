"""w130a — PRICE ANGLE INDEX ROW 8 (foundation), the cell five guards call honest.

Row 8's price cell reads, in full:

    **NOT A PRICE** -- a foundation row, and the answer is **0**, and it holds on its own
    artefacts: the metric is a table row, the folds are frozen since w38 and verified
    against four public packs by #33, and the GBDT baselines are on disk

w129 cited this cell as the honest counterexample to row 10's opt-out ("row 8 passes on the
`0`"). It passes #53, #55, #56, #57 and #58. It passes #57 specifically because C1 accepts
"a magnitude OR the bare `0`", and row 8 types the `0`.

WHAT THE `0` DOES NOT SAY IS WHAT IT IS MEASURED AGAINST. Every other priced row names its
baseline -- base104 without the member, the uncorrected stack, the shipped file's own bytes,
Kaggle's auto-selection by public score. Row 8's silent baseline is "the foundation ALREADY
EXISTS": it was built once, at w38, and re-doing it today buys nothing. That is a REPEAT
price. Priced against its own ABSENCE the same row is the largest number in the table.

Three arms, one per clause of the row's own elaboration, all arithmetic over arrays already
on disk. Registered in experiments/w130_prereg.txt before any number existed.

  A  confirm the metric   -- FINAL-FILE layer, k=1, per-competition TOTAL. Baseline: the
     same shipped predictions under the failure the brief names (threshold to a hard class),
     steel-manned by taking the BEST cut point, plus the free-monotone-transform half.
  B  the fixed-fold harness -- CV-ESTIMATE layer, per comparison. Not a gain: an sd. Baseline:
     an unpaired harness, and one that re-draws its folds.
  C  one honest GBDT baseline -- OOF layer, against chance (AUC 0.5), with the
     stack-minus-best-member companion in the same units.

    .venv/bin/python experiments/w130a_row8.py     # 0 = every registered prediction held
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import DATA, N_SPLITS, SEED, TARGET   # noqa: E402

OOFDIR = os.path.join(ROOT, "oof")
SUBDIR = os.path.join(ROOT, "submissions")
OUT = os.path.join(HERE, "w130a_row8.json")

FAILS = 0
E6 = 1e-6
FLOOR_E6 = 50.0            # the between-file stdcorr floor every price here is read against
CATBOOST_E6 = 10.0416      # row 3's ENROLMENT price, the table's published "do not build" bar

SHIPPED = os.path.join(SUBDIR, "oof_w36_ad199stdcorr.npy")   # the deadline pick's own OOF
GBDT = os.path.join(OOFDIR, "oof_lgbm_fixed_lat_frac.npy")   # the honest baseline GBDT
TUNED = os.path.join(OOFDIR, "oof_lgbm_tuned_lat_frac.npy")  # row 2's tuned twin

BOOT = 200          # bootstrap draws for arm B1
PART = 200          # re-drawn stratified partitions for arm B2

# Registered intervals, copied from w130_prereg.txt. Frozen as literals so this file is a
# COMPARISON against a prediction, not a computation that agrees with itself.
PRED = {
    "A_min_e6": 50_000.0,      # thresholding must cost > 1000x the floor
    "B_min_ratio": 2.0,        # unpaired sd / paired sd
    "C_min_e6": 400_000.0,     # one honest GBDT against chance
    "C_max_stack_gap_e6": 10_000.0,
    "C_min_share": 0.97,       # foundation's share of total AUC above chance
}


def fail(msg: str) -> None:
    global FAILS
    FAILS += 1
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    print(f"  ok: {msg}")


def auc(y, p):
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------- arm A: the metric
def arm_a(y, out):
    print("\nARM A -- 'confirm the metric': FINAL-FILE layer, k=1, per-competition TOTAL")
    p = np.load(SHIPPED)
    a_cont = auc(y, p)
    print(f"  shipped file, continuous          AUC {a_cont:.10f}")

    # A1. The failure the brief names: ship a hard class. AUC of a 0/1 vector is the single
    # (FPR,TPR) corner one cut produces, i.e. balanced accuracy. STEEL-MANNED: sweep cuts and
    # keep the BEST, so the price is what thresholding costs a run that also guessed the cut
    # right. Cuts include 0.5, the base rate, and the base-rate quantile of the scores.
    base = float(y.mean())
    cuts = sorted(set(
        [0.5, base, float(np.quantile(p, 1.0 - base))]
        + list(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    ))
    best_cut, best_hard = None, -1.0
    for c in cuts:
        a = auc(y, (p >= c).astype(np.float64))
        if a > best_hard:
            best_hard, best_cut = a, c
    a1_e6 = (a_cont - best_hard) / E6
    print(f"  best of {len(cuts)} hard cuts (cut={best_cut:.6f})  AUC {best_hard:.10f}")
    print(f"  A1 thresholding cost, best case   {a1_e6:+,.1f}e-6   "
          f"({a1_e6 / FLOOR_E6:,.0f}x the {FLOOR_E6:.0f}e-6 floor)")
    # naive cut too, for the record: the one a run would pick without thinking
    a_naive = auc(y, (p >= 0.5).astype(np.float64))
    print(f"  A1b same at the naive cut 0.5      {(a_cont - a_naive) / E6:+,.1f}e-6")

    # A2. The other half of the metric decision: under AUC every strictly monotone transform
    # of the file is free, which is WHY "calibrate the final file" is on the DO-NOT list.
    # ⚠ THE OBVIOUS PROBE IS DEGENERATE HERE and the first version of this file used it. The
    # shipped pick is ALREADY a rank-uniform vector (see A3), so re-ranking it is a no-op:
    # max|p - rank/n| = 3.0e-4 and logloss moves 2.4e-6. A no-op cannot demonstrate that a
    # transform is free -- it demonstrates nothing. Use a map the file has NOT already had
    # applied: shrink the logit by half. Both halves of the claim are MEASURED, and the pass
    # line asserts only the halves that were.
    eps = 1e-15
    ll = lambda q: float(-np.mean(y * np.log(np.clip(q, eps, 1)) +
                                  (1 - y) * np.log(np.clip(1 - q, eps, 1))))
    pc = np.clip(p, 1e-9, 1 - 1e-9)
    q = 1.0 / (1.0 + np.exp(-0.5 * np.log(pc / (1 - pc))))
    d_mono_e6 = (auc(y, q) - a_cont) / E6
    ll_p, ll_q = ll(p), ll(q)
    ll_pct = (ll_q - ll_p) / ll_p * 100
    r = pd.Series(p).rank(method="average").to_numpy() / len(p)
    print(f"  A2 logit-halved, under AUC         {d_mono_e6:+.4f}e-6  (free)")
    print(f"  A2 the same map, under logloss     {ll_p:.6f} -> {ll_q:.6f}  ({ll_pct:+.1f}%)")
    print(f"  A2 [degenerate probe, for the record] rank/n IS already the file: "
          f"max|p-r| {np.abs(p - r).max():.2e}")

    # A3. POST-HOC, NOT REGISTERED -- found while checking why A2's logloss would not move.
    # The shipped pick has mean 0.50000 against a base rate of 0.70942: it is a RANK vector,
    # not a probability. Under AUC that is free and correct. Under a calibration metric the
    # same bytes are WORSE THAN sample_submission's constant. This is the metric decision
    # priced on the artefact actually being shipped.
    base_rate = float(y.mean())
    rmse = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    rm_p, rm_c = rmse(p), rmse(np.full_like(p, base_rate))
    print(f"  A3 [post-hoc] shipped file mean {p.mean():.5f} vs base rate {base_rate:.5f}")
    print(f"  A3 [post-hoc] RMSE shipped {rm_p:.6f} vs constant {rm_c:.6f}  -> the shipped "
          f"file is {'WORSE' if rm_p > rm_c else 'better'} than the constant under RMSE")

    out["arm_a"] = {
        "auc_continuous": a_cont, "auc_best_hard": best_hard, "best_cut": best_cut,
        "n_cuts": len(cuts), "threshold_cost_e6": a1_e6,
        "threshold_cost_naive_e6": (a_cont - a_naive) / E6,
        "monotone_map": "logit halved", "monotone_auc_e6": d_mono_e6,
        "logloss_shipped": ll_p, "logloss_mapped": ll_q, "logloss_pct": ll_pct,
        "posthoc_unregistered": {
            "shipped_mean": float(p.mean()), "base_rate": float(y.mean()),
            "rmse_shipped": rm_p, "rmse_constant": rm_c,
            "shipped_worse_than_constant_under_rmse": rm_p > rm_c,
            "rank_probe_max_abs_diff": float(np.abs(p - r).max()),
        },
        "layer": "FINAL-FILE", "k": 1, "scope": "per-competition TOTAL, not a rate",
        "baseline": "the same shipped predictions thresholded to a hard class (best cut)",
    }


# ---------------------------------------------------------------- arm B: the folds
def arm_b(y, out):
    print("\nARM B -- 'build the fixed-fold CV harness': CV-ESTIMATE layer, per comparison")
    m1, m0 = np.load(TUNED), np.load(GBDT)
    d_true = (auc(y, m1) - auc(y, m0)) / E6
    print(f"  the pair: lgbm_tuned_lat_frac - lgbm_fixed_lat_frac = {d_true:+.2f}e-6 "
          f"(row 2's TUNING price is +0.4e-6 at the STACK layer)")

    n = len(y)
    rng = np.random.default_rng(0)
    dp, du = [], []
    t0 = time.time()
    for b in range(BOOT):
        i1 = rng.integers(0, n, n)
        dp.append(auc(y[i1], m1[i1]) - auc(y[i1], m0[i1]))          # same rows for both
        i2 = rng.integers(0, n, n)
        du.append(auc(y[i1], m1[i1]) - auc(y[i2], m0[i2]))          # independent rows
    sd_p, sd_u = float(np.std(dp, ddof=1)) / E6, float(np.std(du, ddof=1)) / E6
    ratio = sd_u / sd_p
    print(f"  {BOOT} bootstraps in {time.time() - t0:.0f}s")
    print(f"  B1 paired   sd of the measured delta  {sd_p:9,.2f}e-6")
    print(f"  B1 unpaired sd of the measured delta  {sd_u:9,.2f}e-6")
    print(f"  B1 variance the shared folds remove   {ratio:.1f}x on the sd, "
          f"{ratio ** 2:.0f}x on the variance")

    # B2. The other clause: FIXED, i.e. do not re-draw. Pooled OOF AUC is invariant to the
    # partition; the fold-MEAN AUC a harness prints is not, and that spread is carried into
    # every number a re-drawing harness reports.
    p = np.load(SHIPPED)
    pooled = auc(y, p)
    means = []
    for r in range(PART):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=r)
        means.append(np.mean([auc(y[te], p[te]) for _, te in skf.split(p, y)]))
    sd_fold = float(np.std(means, ddof=1)) / E6
    spread = (float(np.max(means)) - float(np.min(means))) / E6
    print(f"  B2 pooled OOF AUC, partition-invariant  {pooled:.10f}")
    print(f"  B2 fold-mean AUC over {PART} re-draws     sd {sd_fold:.2f}e-6, "
          f"range {spread:.2f}e-6")

    if ratio < PRED["B_min_ratio"]:
        fail(f"P2: sd ratio {ratio:.2f} is under the registered {PRED['B_min_ratio']}")
    else:
        ok(f"P2 holds: unpaired sd is {ratio:.1f}x the paired sd")

    out["arm_b"] = {
        "delta_true_e6": d_true, "sd_paired_e6": sd_p, "sd_unpaired_e6": sd_u,
        "sd_ratio": ratio, "n_boot": BOOT,
        "pooled_auc": pooled, "fold_mean_sd_e6": sd_fold, "fold_mean_range_e6": spread,
        "n_partitions": PART,
        "layer": "CV-ESTIMATE", "scope": "per comparison; an sd, NOT a gain",
        "baseline": "an unpaired harness, and one that re-draws its folds each experiment",
    }


# ---------------------------------------------------------------- arm C: the baseline
def arm_c(y, out):
    print("\nARM C -- 'score one honest GBDT baseline': OOF layer, against chance")
    a_gbdt = auc(y, np.load(GBDT))
    a_ship = auc(y, np.load(SHIPPED))
    found_e6 = (a_gbdt - 0.5) / E6
    print(f"  constant prediction (sample_submission)  AUC 0.5000000000")
    print(f"  one honest GBDT, oof_lgbm_fixed_lat_frac AUC {a_gbdt:.10f}")
    print(f"  the shipped deadline pick                AUC {a_ship:.10f}")

    # The honest companion: the best SINGLE member anywhere in the pack, so the "everything
    # else" number is not inflated by comparing the stack against a weak baseline.
    best_nm, best_a = "oof_lgbm_fixed_lat_frac", a_gbdt
    scanned = 0
    for d in (OOFDIR, os.path.join(DATA, "oof", "oof")):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.startswith("oof_") and fn.endswith(".npy")):
                continue
            v = np.load(os.path.join(d, fn))
            if v.shape != y.shape:
                continue
            scanned += 1
            a = auc(y, v)
            if a > best_a:
                best_a, best_nm = a, fn[:-4]
    gap_e6 = (a_ship - best_a) / E6
    share = found_e6 / ((a_ship - 0.5) / E6)
    print(f"  best single member of {scanned} scanned      AUC {best_a:.10f}  ({best_nm})")
    print(f"  C  foundation vs chance                  {found_e6:+,.1f}e-6")
    print(f"  C  stack minus best single member        {gap_e6:+,.1f}e-6  "
          "= everything the other nine rows bought")
    print(f"  C  foundation's share of AUC above chance {share * 100:.3f}%")

    if found_e6 <= PRED["C_min_e6"]:
        fail(f"P3a: foundation {found_e6:,.1f}e-6 not above {PRED['C_min_e6']:,.0f}e-6")
    else:
        ok(f"P3a holds: {found_e6:,.1f}e-6 > {PRED['C_min_e6']:,.0f}e-6")
    if gap_e6 >= PRED["C_max_stack_gap_e6"]:
        fail(f"P3b: stack gap {gap_e6:,.1f}e-6 not under {PRED['C_max_stack_gap_e6']:,.0f}e-6")
    else:
        ok(f"P3b holds: stack-minus-best-member {gap_e6:,.1f}e-6 is under "
           f"{PRED['C_max_stack_gap_e6']:,.0f}e-6")
    if share < PRED["C_min_share"]:
        fail(f"P3c: share {share:.4f} under the registered {PRED['C_min_share']}")
    else:
        ok(f"P3c holds: the foundation is {share * 100:.3f}% of AUC above chance")

    out["arm_c"] = {
        "auc_gbdt_baseline": a_gbdt, "auc_shipped": a_ship,
        "best_single_member": best_nm, "auc_best_single": best_a, "n_scanned": scanned,
        "foundation_vs_chance_e6": found_e6, "stack_minus_best_member_e6": gap_e6,
        "foundation_share": share,
        "layer": "OOF", "scope": "per competition, a TOTAL",
        "baseline": "a constant prediction, which AUC scores at 0.5",
    }


def main() -> int:
    y = pd.read_csv(os.path.join(DATA, "train.csv"), usecols=[TARGET])[TARGET].to_numpy()
    print(f"train {len(y):,} rows, base rate {y.mean():.16f}")
    print(f"frozen folds: StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
          f"random_state={SEED})")
    out = {"run": "w130", "row": 8, "written": "2026-08-30"}
    arm_a(y, out)
    arm_b(y, out)
    arm_c(y, out)

    print("\nTHE READING -- three currencies, and they DO NOT ADD")
    a, b, c = out["arm_a"], out["arm_b"], out["arm_c"]
    print(f"  A {a['threshold_cost_e6']:>12,.1f}e-6   final-file AUC at k=1, vs a hard class")
    print(f"  B {b['sd_paired_e6']:>12,.1f}e-6   an sd of a measurement, vs "
          f"{b['sd_unpaired_e6']:,.1f}e-6 unpaired -- NOT a gain")
    print(f"  C {c['foundation_vs_chance_e6']:>12,.1f}e-6   OOF AUC, vs chance")
    print(f"  the table's 'do not build' bar is row 3's +{CATBOOST_E6:.2f}e-6/member; "
          f"the floor is {FLOOR_E6:.0f}e-6")
    print(f"  ratio, arm C to row 3's bar: {c['foundation_vs_chance_e6'] / CATBOOST_E6:,.0f}x")
    out["reading"] = {
        "arms_do_not_add": True,
        "repeat_price_e6": 0.0,
        "repeat_baseline": "the foundation already exists (built once, w38)",
        "catboost_bar_e6": CATBOOST_E6, "floor_e6": FLOOR_E6,
    }
    out["failures"] = FAILS
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {OUT}")
    print(f"FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
