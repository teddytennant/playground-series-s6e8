"""w18b — the CV->LB instrument in a THIRD transform family, and the sharpest test yet of
"decontaminate by PROVENANCE, not by CV distance".

Slot 3's next-run item 3, verbatim: "Scope, don't delete, the 'never a group gap' rule. It needs
a third family. The decontamination criterion should be changed from CV distance to provenance."

WHY RANKRAW IS THE RIGHT THIRD FAMILY, and a harder test than the first two. Six rankraw files
have been sent: four of ours and TWO foreign public stacks (stack_pub151_rankraw,
stack_pub151_fixed_rankraw). In the hybrid family the foreign file that mattered sat 340e-6 below
everything on CV, so "drop the CV-distant file" and "drop the foreign file" pointed at the same
row and slot 3 could only separate them because a THIRD foreign file happened to sit close. Here
the two foreign files sit 11e-6 and 13e-6 below our lowest blend on CV -- i.e. INSIDE our own
spread. CV distance cannot find them at all. If provenance is the right criterion this family is
where it has to show.

THE MODELS, all fixed before the upload (w18c_shipprereg.txt):
  A  PAIRED (registered)  LB(reference) + dCV, the reference's own +/-5e-6 rounding integrated
                          out, paired slice sd from LAW-IF.
  B  GROUP GAP, all 6 sent rankraw files            (contaminated — what w17b would print)
  C  GROUP GAP, 5 — drop the CV-farthest file       (the CV-distance criterion)
  D  GROUP GAP, our 4 blends only                   (the PROVENANCE criterion)

AND THE SD IS NOT SIMULATED. w18a gated LAW-IF against w17d's 6,000-draw paired sds at a median
error of 0.5% over 15 pairs, so this ship uses the closed form as the primary instrument and runs
a short simulation only as a check. That is the operational payoff of w17h's law: a paired
prediction now costs one pass over the OOF vectors instead of thousands of AUC evaluations.

    .venv/bin/python experiments/w18b_rankraw.py
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

SENT = ["blend158_rankraw", "blend156_rankraw", "stack_pub151_fixed_rankraw",
        "blend150fx_rankraw", "stack_pub151_rankraw", "blend150sx_rankraw"]
OURS = ["blend158_rankraw", "blend156_rankraw", "blend150fx_rankraw", "blend150sx_rankraw"]
FOREIGN = ["stack_pub151_fixed_rankraw", "stack_pub151_rankraw"]
CANDIDATES = ["blend160orig_rankraw", "blend159av_rankraw", "blend160origm_rankraw",
              "blend159_rankraw", "w14a_repro159av_rankraw", "blend153_rankraw"]
TIER1 = 0.97108


def midrank_cdf(sorted_ref, s):
    lo = np.searchsorted(sorted_ref, s, side="left")
    hi = np.searchsorted(sorted_ref, s, side="right")
    return (lo + hi) * 0.5 / len(sorted_ref)


def lawif_sd(va, vb, y, n_pub):
    """Paired public-slice sd of AUC(a) - AUC(b), closed form. w17h's LAW-IF, gated by w18a."""
    pos, neg = y == 1, y == 0
    n = len(y)
    pi1 = pos.mean()
    fa = midrank_cdf(np.sort(va[neg]), va[pos]) - midrank_cdf(np.sort(vb[neg]), vb[pos])
    fb = midrank_cdf(np.sort(va[pos]), va[neg]) - midrank_cdf(np.sort(vb[pos]), vb[neg])
    var = fa.var(ddof=1) / (n_pub * pi1) + fb.var(ddof=1) / (n_pub * (1 - pi1))
    return float(np.sqrt(var * (1.0 - n_pub / n)))


def dist_from(point_delta, ref_lb, sd, rng):
    """Marginalise the reference's own +/-5e-6 rounding and the paired slice draw."""
    t_ref = ref_lb + rng.uniform(-GRID / 2, GRID / 2, 400_000)
    dd = rng.normal(point_delta, sd, 400_000)
    vals = np.round(np.round((t_ref + dd) / GRID) * GRID, 5)
    u, c = np.unique(vals, return_counts=True)
    return {float(x): float(k / c.sum()) for x, k in zip(u, c) if k / c.sum() >= 0.0005}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "500"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    LB = sub.groupby("stem")["publicScore"].max()
    print(f"live: {len(sub)} submissions, {len(LB)} distinct files")
    for k in CANDIDATES:
        assert k not in LB.index, f"{k} already sent"

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pub = int(round(N_TEST * F))
    names = sorted(set(SENT + CANDIDATES))
    V = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in names}
    cv = {k: fast_auc(y, V[k]) for k in names}

    print("\n=== the rankraw family as sent ===")
    print(f"{'file':30s} {'CV':>14s} {'LB':>9s} {'gap':>11s}  provenance")
    gaps = {k: float(LB[k] - cv[k]) for k in SENT}
    for k in sorted(SENT, key=lambda z: -cv[z]):
        print(f"  {k:28s} {cv[k]:.10f} {LB[k]:>9.5f} {gaps[k]*1e6:>+9.1f}e-6  "
              f"{'FOREIGN' if k in FOREIGN else 'ours'}")
    far = min(SENT, key=lambda z: cv[z])
    print(f"\n  the CV-farthest sent file is {far} ({(cv[far]-max(cv[k] for k in SENT))*1e6:+.1f}e-6)"
          f" — {'FOREIGN' if far in FOREIGN else 'OURS'}")
    print(f"  the two foreign files sit {(cv[FOREIGN[0]]-min(cv[k] for k in OURS))*1e6:+.1f} and "
          f"{(cv[FOREIGN[1]]-min(cv[k] for k in OURS))*1e6:+.1f}e-6 against our LOWEST blend — "
          f"inside our own spread, so CV distance cannot see them")

    g_all = float(np.mean([gaps[k] for k in SENT]))
    five = [k for k in SENT if k != far]
    g_5 = float(np.mean([gaps[k] for k in five]))
    g_ours = float(np.mean([gaps[k] for k in OURS]))
    s_ours = float(np.std([gaps[k] for k in OURS], ddof=1))
    g_for = float(np.mean([gaps[k] for k in FOREIGN]))
    print(f"\n  B all 6 (contaminated)      {g_all*1e6:+.1f}e-6")
    print(f"  C 5, drop the CV-farthest   {g_5*1e6:+.1f}e-6")
    print(f"  D our 4 blends (provenance) {g_ours*1e6:+.1f}e-6  sd {s_ours*1e6:.1f}e-6")
    print(f"  the 2 foreign files alone   {g_for*1e6:+.1f}e-6  "
          f"-> contamination pulls the group gap by {(g_all-g_ours)*1e6:+.1f}e-6")

    # ------------------------------------------------- pick the ship: highest CV, rank-distinct,
    # tie-broken by how much the four models DISAGREE (a send that cannot discriminate is waste)
    ss = pd.read_csv(os.path.join(ROOT, "data", "sample_submission.csv"))
    sent_ranks = {}
    for fn in sorted(os.listdir(SUB)):
        if fn.endswith(".csv") and fn[:-4] in LB.index:
            sent_ranks[fn[:-4]] = rankdata(pd.read_csv(os.path.join(SUB, fn))["addicted_label"].to_numpy())
    # The paired model's reference is now CHOSEN, not assumed. Every sent file in the family is a
    # legal reference; the sharpest one is whichever has the smallest paired slice sd against the
    # candidate, and LAW-IF prices all of them for the cost of one pass. That decision was
    # unaffordable while paired sds came from 2,000-draw simulations.
    print(f"\n=== candidate screen: reference chosen by LAW-IF paired sd, not by CV rank ===")
    print(f"{'candidate':26s} {'CV':>14s} {'best ref':>26s} {'sd':>7s} {'sd(hi-CV ref)':>13s} "
          f"{'A':>8s} {'B':>8s} {'C':>8s} {'D':>8s} {'nd':>3s}  clash")
    default_ref = max(OURS, key=lambda z: cv[z])
    screen = []
    for k in CANDIDATES:
        rk = rankdata(pd.read_csv(os.path.join(SUB, f"{k}.csv"))["addicted_label"].to_numpy())
        clash = [s for s, r in sent_ranks.items() if np.array_equal(rk, r)]
        sds = {r: lawif_sd(V[k], V[r], y, n_pub) for r in SENT}
        rf = min(sds, key=sds.get)
        pA = dist_from(cv[k] - cv[rf], float(LB[rf]), sds[rf], np.random.default_rng(SEED))
        mA = max(pA, key=pA.get)
        pr = {t: float(np.round((cv[k] + g) / GRID) * GRID)
              for t, g in (("B", g_all), ("C", g_5), ("D", g_ours))}
        nd = len({round(v, 5) for v in (mA, pr["B"], pr["C"], pr["D"])})
        screen.append(dict(name=k, cv=cv[k], ref=rf, sd=sds[rf], sd_default=sds[default_ref],
                           A=mA, pA=pA[mA], **pr, ndist=nd, clash=clash))
        print(f"  {k:24s} {cv[k]:.10f} {rf:>26s} {sds[rf]*1e6:6.2f} "
              f"{sds[default_ref]*1e6:12.2f} {mA:8.5f} {pr['B']:8.5f} {pr['C']:8.5f} "
              f"{pr['D']:8.5f} {nd:3d}  {clash if clash else 'none'}")
    ok = [s for s in screen if not s["clash"]]
    assert ok, "every candidate is rank-identical to something already sent"
    # decision rule, fixed before the numbers: discriminate first, then be as sharp as possible,
    # then prefer the higher CV.
    best = max(ok, key=lambda s: (s["ndist"], round(s["pA"], 3), s["cv"]))
    SHIP, ref = best["name"], best["ref"]
    print(f"\n  -> SHIP {SHIP}: {best['ndist']} distinct predicted values, CV {best['cv']:.10f}, "
          f"reference {ref} (paired sd {best['sd']*1e6:.2f}e-6 against "
          f"{best['sd_default']*1e6:.2f}e-6 for the highest-CV reference)")

    # ------------------------------------------------- the sd, LAW-IF vs a short simulation
    print(f"\n=== paired slice sd for ({SHIP}, {ref}) ===")
    sd_if = best["sd"]
    pk = {k: prep(V[k], y) for k in (SHIP, ref)}
    dpool = cv[SHIP] - cv[ref]
    rng = np.random.default_rng(SEED)
    n = len(y)
    d = np.empty(a.reps)
    for i in range(a.reps):
        idx = rng.choice(n, size=N_TEST, replace=False)
        m = np.zeros(n, dtype=bool)
        m[idx[:n_pub]] = True
        d[i] = subset_auc(pk[SHIP], m) - subset_auc(pk[ref], m)
    sd_sim = float((d - dpool).std(ddof=1))
    print(f"  LAW-IF (closed form, zero draws) {sd_if*1e6:.3f}e-6")
    print(f"  simulated, {a.reps} draws          {sd_sim*1e6:.3f}e-6   ratio {sd_if/sd_sim:.3f}"
          f"   (MC se of the sim itself {1/np.sqrt(2*(a.reps-1))*100:.1f}%)")

    pA = dist_from(dpool, float(LB[ref]), sd_if, np.random.default_rng(SEED + 1))
    modeA = max(pA, key=pA.get)
    print(f"\n=== MODEL A (registered): paired against {ref} at {LB[ref]:.5f} ===")
    print(f"  dCV {dpool*1e6:+.3f}e-6, paired sd {sd_if*1e6:.3f}e-6")
    for x in sorted(pA, key=lambda z: -pA[z]):
        print(f"    P(prints {x:.5f}) = {pA[x]:.3f}")

    print(f"\n=== the four models ===")
    print(f"  A paired, LAW-IF sd (registered)          {modeA:.5f}   P {pA[modeA]:.3f}")
    for t, lab in (("B", "group gap, all 6 (contaminated)"),
                   ("C", f"group gap, 5 (drop CV-far {far[:18]})"),
                   ("D", "group gap, our 4 blends (PROVENANCE)")):
        print(f"  {t} {lab:40s} {best[t]:.5f}")
    vals = sorted({round(v, 5) for v in (modeA, best["B"], best["C"], best["D"])})
    print(f"  -> {len(vals)} distinct values: {', '.join(f'{v:.5f}' for v in vals)}   "
          f"{'DISCRIMINATING' if len(vals) > 1 else 'NOT discriminating'}")

    print(f"\n=== tier risk, priced BEFORE the send ===")
    p8 = sum(v for k, v in pA.items() if k >= TIER1 - 1e-9)
    print(f"  auto-slot 1 is {TIER1:.5f} (3-way tie). P(this file reaches it) = {p8:.5f}")
    print(f"  -> the send {'CANNOT' if p8 < 1e-3 else 'COULD'} move the auto-pick tiers, and the "
          f"file's CV is {(cv[SHIP]-0.970055666257)*1e6:+.1f}e-6 against the CV pick's")

    print(f"\n=== artifact audit ===")
    df = pd.read_csv(os.path.join(SUB, f"{SHIP}.csv"))
    assert len(df) == N_TEST and (df["id"].to_numpy() == ss["id"].to_numpy()).all()
    v = df["addicted_label"].to_numpy()
    assert np.isfinite(v).all()
    print(f"  {len(df):,} rows, ids match sample_submission in order, all finite, "
          f"{len(np.unique(v)):,} distinct, range [{v.min():.6g}, {v.max():.6g}]")
    print(f"  rank-identical to an already-sent file? no (screened above)")

    json.dump(dict(ship=SHIP, ref=ref, reps=a.reps, cv=cv, lb_ref=float(LB[ref]), dcv=dpool,
                   sd_if=sd_if, sd_sim=sd_sim, sd_ratio=sd_if / sd_sim, gap_all6=g_all,
                   gap_5=g_5, gap_ours=g_ours, gap_foreign=g_for, gap_ours_sd=s_ours,
                   cv_far=far, pred_A=modeA, pred_A_dist=pA, pred_B=best["B"],
                   pred_C=best["C"], pred_D=best["D"], n_distinct=len(vals),
                   p_reaches_tier=p8, screen=[{k: (v if k != "clash" else list(v))
                                               for k, v in s.items()} for s in screen]),
              open(os.path.join(HERE, "w18b_rankraw.json"), "w"), indent=1)
    print("\nwrote experiments/w18b_rankraw.json")


if __name__ == "__main__":
    main()
