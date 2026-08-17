"""w17d — the PRIVATE cost of the unmade click, conditioned on the public scores we actually saw.

Slot 1 §9 item 3, verbatim: "Public/private coupling for the three auto-slot-1 files. §6 has the
public inflation but no private penalty."

WHAT IS WRONG WITH THE STANDING PRICE
-------------------------------------
`w16w_reprice.py` draws pseudo-tests out of the labelled pool and reads each file's PRIVATE AUC.
Across draws E[private_k] = cv_k by construction, so WANTED — chosen on CV — wins in expectation
automatically, and the instrument measures only the variance around that. It never conditions on
the public scores that were actually observed, which are the entire reason the auto-pick is what
it is. Slot 1 §6: `w16e_aonly` leads `blend159av_h3` by 30e-6 on public but 5.3e-6 on CV, so
~83% of the lead is slice-specific. Public and private partition ONE 296,302-row test set, so a
file that got lucky on the public slice must give some of it back on the private one.

THE CONDITIONING, done exactly rather than with a kernel
--------------------------------------------------------
Model the real board as one draw of this geometry plus an unknown common additive gap G (the
+1002e-6 CV->LB constant, which is common to all files in a family and cancels from contrasts).
Draw r is consistent with the observed board iff some G satisfies round(P_k + G, 5) == LB_k for
every conditioning file k, i.e.

    G in [LB_k - P_k - 5e-6,  LB_k - P_k + 5e-6)     for all k.

Intersect over k and take the LENGTH of the surviving interval as the draw's weight (flat prior
on G). This integrates the 1e-5 reporting grid exactly instead of approximating it, and it needs
no Gaussian assumption and no partition identity.

Two branches, because one assumption is not decidable from a single fixed slice:
  GLOBAL — one free G. The h3-vs-ens4 group gap of +11.4e-6 is treated as partition noise.
  PERFAM — one free G per transform family. Grants the family displacement as a real train/test
           effect and conditions only on within-family public deviations.
The truth is bracketed by the two. Slot 1's within-family ratio of 0.79 (observed scatter 7.4e-6
against 9.4e-6 predicted by partition+grid alone) supports GLOBAL within a family; it says
nothing about the between-family constant.

Pre-registration with P1-P5 and what falsifies each: experiments/w17d_prereg.txt, written first.

    .venv/bin/python experiments/w17d_coupling.py --reps 6000
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from common import SUB, TARGET, load_raw  # noqa: E402
from w15i_cvlb import prep, subset_auc  # noqa: E402
from w16b_cellweight import fast_auc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
N_TEST = 296_302
SEED, F = 1616, 0.20                       # w16w's seed and protocol, so the gate is exact
GRID = 1e-5
HALF = GRID / 2.0

WANTED = ("w16i_schemeavg", "blend159av_h3")
AUTO1 = ("w16e_aonly", "w16q_ens4avg", "w16t_cellens4")     # the live 0.97108 three-way tie
FAM = {"w16e_aonly": "h3", "w16q_ens4avg": "ens4", "w16t_cellens4": "ens4",
       "w16i_schemeavg": "h3", "blend159av_h3": "h3", "w16n_finegrid": "h3",
       "blend159av": "ens4"}
# w16w's column set, in w16w's order, so the first 500 draws reproduce it exactly.
NAMES = ["blend159av_h3", "blend159av", "w16i_schemeavg", "w16q_ens4avg",
         "w16t_cellens4", "w16e_aonly"]
COND = ["blend159av_h3", "blend159av", "w16i_schemeavg", "w16q_ens4avg",
        "w16t_cellens4", "w16e_aonly"]     # files whose observed public score is conditioned on

GATE_W16W = os.path.join(HERE, "w16w_reprice.json")


def wmean(x, w):
    return float(np.sum(w * x) / np.sum(w))


def wp(x, w):
    return float(np.sum(w * (x > 0)) / np.sum(w))


def interval_weight(P, lb, fams=None):
    """Length of the set of gaps G for which every file's public score rounds to what we saw.

    P   (reps, k) simulated public AUCs.   lb (k,) observed public scores.
    fams None -> one global G.  Otherwise a (k,) array of family labels, one free G per family
    and the weight is the PRODUCT of the per-family interval lengths.
    """
    if fams is None:
        lo = (lb - P - HALF).max(axis=1)
        hi = (lb - P + HALF).min(axis=1)
        return np.maximum(hi - lo, 0.0)
    w = np.ones(P.shape[0])
    for f in sorted(set(fams)):
        c = np.flatnonzero(fams == f)
        lo = (lb[c] - P[:, c] - HALF).max(axis=1)
        hi = (lb[c] - P[:, c] + HALF).min(axis=1)
        w *= np.maximum(hi - lo, 0.0)
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(HERE, "w17d_coupling.json"))
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v", "--page-size", "200"],
                         capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert (sub["status"] == "SubmissionStatus.COMPLETE").all()
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    assert (agg["max"] == agg["min"]).all(), "a file's repeats disagree — scores are not deterministic"
    LB = agg["max"].to_dict()
    print(f"live: {len(sub)} submissions, {len(agg)} distinct files")

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n = len(y)
    vecs = {k: np.load(os.path.join(SUB, f"oof_{k}.npy")).astype(np.float64) for k in NAMES}
    cv = {k: fast_auc(y, v) for k, v in vecs.items()}
    print(f"\n{'file':20s} {'pooled CV':>14s} {'public LB':>10s} {'fam':>5s}")
    for k in NAMES:
        print(f"  {k:18s} {cv[k]:.10f} {LB[k]:>10.5f} {FAM[k]:>5s}")

    pk = {k: prep(vecs[k], y) for k in NAMES}
    col = {nm: i for i, nm in enumerate(NAMES)}
    n_pub = int(round(N_TEST * F))
    print(f"\n=== {a.reps} draws: pseudo-test {N_TEST:,} -> public {n_pub:,} / "
          f"private {N_TEST-n_pub:,} ===")

    rng = np.random.default_rng(SEED)
    PUB = np.zeros((a.reps, len(NAMES)))
    PRI = np.zeros((a.reps, len(NAMES)))
    TOT = np.zeros((a.reps, len(NAMES)))
    for i in range(a.reps):
        idx = rng.choice(n, size=N_TEST, replace=False)        # w16w's exact draw
        mpri = np.zeros(n, dtype=bool); mpri[idx[n_pub:]] = True
        mpub = np.zeros(n, dtype=bool); mpub[idx[:n_pub]] = True
        mtot = mpri | mpub
        for k, nm in enumerate(NAMES):
            PRI[i, k] = subset_auc(pk[nm], mpri)
            PUB[i, k] = subset_auc(pk[nm], mpub)
            TOT[i, k] = subset_auc(pk[nm], mtot)
        if (i + 1) % 500 == 0:
            print(f"  draw {i+1}/{a.reps}", flush=True)

    # ------------------------------------------------------------------ P1: the gate
    print("\n=== P1 GATE: first 500 draws must reproduce w16w_reprice.json ===")
    ok = True
    jw = json.load(open(GATE_W16W))
    cur500 = np.maximum(PRI[:500, col[WANTED[0]]], PRI[:500, col[WANTED[1]]])
    for key, rec in jw["limit2"].items():
        x, z = key.split("+")
        e = np.maximum(PRI[:500, col[x]], PRI[:500, col[z]])
        d = (e - cur500).mean() - rec["d"]
        ok &= abs(d) < 1e-12
        print(f"  limit2 {key:32s} drift {d*1e12:+8.3f}e-12")
    for x, rec in jw["limit1"].items():
        d = (PRI[:500, col[x]] - cur500).mean() + rec["cost_of_not_clicking"]
        ok &= abs(d) < 1e-12
        print(f"  limit1 {x:32s} drift {d*1e12:+8.3f}e-12")
    d = cur500.mean() - jw["e_wanted"]
    ok &= abs(d) < 1e-12
    print(f"  E[max] WANTED{'':25s} drift {d*1e12:+8.3f}e-12")
    print(f"  P1 {'PASSED' if ok else '*** FAILED ***'}")
    if not ok:
        print("  downstream numbers INVALID; not writing json")
        sys.exit(1)

    # ------------------------------------- P2: the coupling, measured not assumed
    print("\n=== P2: public/private coupling, slope of d(private) on d(public) ===")
    print("  AUC is NOT additive over a partition, so -f/(1-f) = -0.2500 is an empirical claim.")
    cpl = {}
    for x, z in itertools.combinations(NAMES, 2):
        dp = PUB[:, col[x]] - PUB[:, col[z]]
        dr = PRI[:, col[x]] - PRI[:, col[z]]
        dt = TOT[:, col[x]] - TOT[:, col[z]]
        # at fixed pseudo-test: regress (private - total) on (public - total)
        u, v = dp - dt, dr - dt
        b = float(np.polyfit(u, v, 1)[0])
        r = float(np.corrcoef(u, v)[0, 1])
        cpl[f"{x}|{z}"] = dict(beta=b, corr=r)
        print(f"  {x:16s} - {z:16s}  beta {b:+.4f}  corr {r:+.5f}")
    bs = np.array([v["beta"] for v in cpl.values()])
    print(f"  -> median beta {np.median(bs):+.4f}   range {bs.min():+.4f}..{bs.max():+.4f}"
          f"   (exact partition -0.2500, w14b measured -0.2517)")
    p2 = bool(-0.26 <= np.median(bs) <= -0.24)
    print(f"  P2 {'CONFIRMED' if p2 else '*** FALSIFIED ***'}")

    # ------------------------------------------------- P3: the variance split
    print("\n=== P3: variance split, w = sigma_t^2/(sigma_t^2 + sigma_p^2) ===")
    ws = {}
    for x, z in itertools.combinations(NAMES, 2):
        dt = TOT[:, col[x]] - TOT[:, col[z]]
        dp = PUB[:, col[x]] - PUB[:, col[z]]
        st = float(dt.std(ddof=1))
        sp = float((dp - dt).std(ddof=1))
        ws[f"{x}|{z}"] = dict(sigma_t=st, sigma_p=sp, w=st ** 2 / (st ** 2 + sp ** 2))
        print(f"  {x:16s} - {z:16s}  sigma_t {st*1e6:5.2f}e-6  sigma_p {sp*1e6:5.2f}e-6  "
              f"w {ws[f'{x}|{z}']['w']:.4f}")
    wmed = float(np.median([v["w"] for v in ws.values()]))
    print(f"  -> median w {wmed:.4f}   predicted from row counts 0.125   break-even f = 0.20")
    p3 = bool(0.10 <= wmed <= 0.15)
    print(f"  P3 {'CONFIRMED' if p3 else '*** FALSIFIED ***'}  "
          f"(w < f => conditioning on public makes the click MORE expensive)")

    # ------------------------------------------------------------ the weights
    lbv = np.array([LB[k] for k in COND])
    Pc = PUB[:, [col[k] for k in COND]]
    famv = np.array([FAM[k] for k in COND])
    W = {"GLOBAL": interval_weight(Pc, lbv, None),
         "PERFAM": interval_weight(Pc, lbv, famv),
         "NONE": np.ones(a.reps)}
    print("\n=== conditioning weights ===")
    for tag in ("GLOBAL", "PERFAM"):
        w = W[tag]
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        print(f"  {tag:7s} draws with weight>0 {int((w > 0).sum()):5d}/{a.reps}   "
              f"ESS {ess:8.1f}")
        if ess < 30:
            print(f"    ⚠ ESS {ess:.1f} is too low to read a mean off — treat as indicative only")

    # --------------------------------------------- the contrasts that matter
    print("\n=== the private contrasts, unconditioned vs conditioned ===")
    print("  positive = the AUTO file beats the CV pick on the private slice")
    rows = []
    for x in AUTO1:
        for z in WANTED:
            d = PRI[:, col[x]] - PRI[:, col[z]]
            rec = dict(auto=x, wanted=z, dcv=cv[x] - cv[z], dlb=LB[x] - LB[z])
            for tag in ("NONE", "GLOBAL", "PERFAM"):
                rec[tag] = wmean(d, W[tag])
                rec["P_" + tag] = wp(d, W[tag])
            rows.append(rec)
    R = pd.DataFrame(rows)
    print(f"{'auto':16s} {'vs CV pick':16s} {'dCV':>8s} {'dLB':>8s} "
          f"{'E none':>9s} {'E GLOBAL':>9s} {'E PERFAM':>9s} {'P none':>7s} {'P GLOB':>7s}")
    for _, r in R.iterrows():
        print(f"  {r['auto']:14s} {r['wanted']:16s} {r['dcv']*1e6:+7.2f} {r['dlb']*1e6:+7.1f} "
              f"{r['NONE']*1e6:+8.3f} {r['GLOBAL']*1e6:+8.3f} {r['PERFAM']*1e6:+8.3f} "
              f"{r['P_NONE']:>7.3f} {r['P_GLOBAL']:>7.3f}")

    # --------------------------------------------------- the repriced click
    print("\n=== the click, repriced. cost of NOT clicking, in private AUC ===")
    out = {"reps": int(a.reps), "seed": SEED, "gate_w16w": True,
           "cv": {k: float(v) for k, v in cv.items()}, "lb": {k: float(LB[k]) for k in NAMES},
           "coupling": cpl, "coupling_beta_median": float(np.median(bs)), "p2": p2,
           "varsplit": ws, "w_median": wmed, "p3": p3,
           "ess": {t: float(W[t].sum() ** 2 / (W[t] ** 2).sum()) for t in W},
           "contrasts": R.to_dict("records"), "limit1": {}, "limit2": {}}
    cur = np.maximum(PRI[:, col[WANTED[0]]], PRI[:, col[WANTED[1]]])

    print("\n  limit 1 — Kaggle auto-selects ONE file")
    print(f"  {'auto file':18s} {'w16w (none)':>13s} {'GLOBAL':>11s} {'PERFAM':>11s} "
          f"{'P(auto better) none -> GLOBAL':>32s}")
    for x in AUTO1:
        d = PRI[:, col[x]] - cur
        rec = {t: -wmean(d, W[t]) for t in ("NONE", "GLOBAL", "PERFAM")}
        rec["P_NONE"], rec["P_GLOBAL"] = wp(d, W["NONE"]), wp(d, W["GLOBAL"])
        out["limit1"][x] = rec
        print(f"  {x:18s} {rec['NONE']*1e6:+12.3f} {rec['GLOBAL']*1e6:+10.3f} "
              f"{rec['PERFAM']*1e6:+10.3f} {rec['P_NONE']:>20.3f} -> {rec['P_GLOBAL']:.3f}")

    print("\n  limit 2 — Kaggle auto-selects TWO of the three tied files")
    for x, z in itertools.combinations(AUTO1, 2):
        e = np.maximum(PRI[:, col[x]], PRI[:, col[z]])
        d = e - cur
        rec = {t: -wmean(d, W[t]) for t in ("NONE", "GLOBAL", "PERFAM")}
        rec["sides"] = f"{FAM[x]}+{FAM[z]}"
        out["limit2"][f"{x}+{z}"] = rec
        print(f"  {x:14s} + {z:16s} [{rec['sides']:9s}] "
              f"none {rec['NONE']*1e6:+7.3f}  GLOBAL {rec['GLOBAL']*1e6:+7.3f}  "
              f"PERFAM {rec['PERFAM']*1e6:+7.3f}")
    for lim in ("limit1", "limit2"):
        for tag in ("NONE", "GLOBAL", "PERFAM"):
            v = [r[tag] for r in out[lim].values()]
            out[f"{lim}_range_{tag}"] = [float(min(v)), float(max(v))]
            print(f"  {lim} range under {tag:6s}: {min(v)*1e6:+7.3f}e-6 .. {max(v)*1e6:+7.3f}e-6")

    a1 = out["limit1"]["w16e_aonly"]
    p4 = bool(2.0e-6 <= a1["GLOBAL"] <= 3.0e-6)
    ps = [out["limit1"][x]["P_GLOBAL"] for x in AUTO1]
    p5 = bool(all(0.30 <= p <= 0.50 for p in ps)
              and all(out["limit1"][x]["P_GLOBAL"] < out["limit1"][x]["P_NONE"] for x in AUTO1))
    out["p4"], out["p5"] = p4, p5
    print(f"\n  P4 {'CONFIRMED' if p4 else '*** FALSIFIED ***'}  "
          f"(w16e_aonly limit-1 GLOBAL {a1['GLOBAL']*1e6:+.3f}e-6 vs w16w's "
          f"{a1['NONE']*1e6:+.3f}e-6; predicted +2.0..+3.0)")
    print(f"  P5 {'CONFIRMED' if p5 else '*** FALSIFIED ***'}  "
          f"(P(auto better) GLOBAL {[round(p,3) for p in ps]}; predicted 0.30..0.50 and falling)")

    print("\n  ⚠ WHAT THIS DOES NOT SAY. WANTED is chosen on CV and nothing here moves it")
    print("  (wave-summary §4.6). This prices a human action. Every figure remains a LOWER")
    print("  bound for w16s's reason: the instrument only ever draws worlds in which the CV")
    print("  ordering is right, so it cannot see WANTED's second slot buying insurance against")
    print("  the whole fitted-correction family failing — and all three auto files carry the")
    print("  same c_avg. GLOBAL vs PERFAM brackets the one assumption a single fixed slice")
    print("  cannot decide; quote the range, not one end of it.")

    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True, default=float)
    R.to_csv(os.path.join(HERE, "w17d_contrasts.csv"), index=False)
    print(f"\nwrote {a.out}, w17d_contrasts.csv")


if __name__ == "__main__":
    main()
