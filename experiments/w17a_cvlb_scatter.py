"""w17a — the SIMULATED half of the CV->LB re-cut: how far can LB sit from CV + gap?

Wave-summary §5.2, the top actionable open item, verbatim:

    "Re-cut the CV->LB relation now that the ladder is falsified (slot 10 §3). The account
     holds 50 scored files. The right object is not a within-family gap constant but a
     residual-scatter estimate: how far can LB sit from CV + gap, given the 10e-6 reporting
     grid? Until that exists, no LB prediction in this workspace is defensible, and several
     'unreachable' claims are unsupported."

Pre-registration, with the predictions and what would falsify each, is in
experiments/w17a_prereg.txt, written before this file was run once.

THE INSTRUMENT (w14b_slicenoise / w15a_crossteam geometry, unchanged)
---------------------------------------------------------------------
Draw a pseudo-test of 296,302 rows out of the 691,369 labelled OOF rows, cut a 20%
(59,260-row) pseudo-public slice, score every stored OOF vector on it. The POOLED ordering
of these vectors IS the CV ordering by construction, so every public/pooled disagreement this
produces is pure slice draw and nothing else.

TWO QUANTITIES COME OUT, AND ONLY ONE OF THEM IS THE RIGHT ONE
--------------------------------------------------------------
  single-file sd  = sd over draws of (slice AUC - pooled AUC) for one file.  ~570e-6 here.
  paired sd       = sd over draws of (slice_i - slice_j) - (pooled_i - pooled_j).  ~18e-6.

The single-file number is NOT comparable to the observed CV->LB scatter, and the first
version of this script wrongly compared against it. Every file's public score is read off
the SAME fixed slice, so the whole ~570e-6 is a shift common to all 50 files and is absorbed
into the gap constant by construction — the same reason the +47.7e-6 small-sample AUC bias is
absorbed. What is left over, and what the residual scatter about CV + gap actually measures,
is the PAIRED quantity. Comparing (A) to the single-file sd would "explain" any scatter at
all and is the shape of test wave-summary §4.2 warns about: one the rule cannot lose.

    .venv/bin/python experiments/w17a_cvlb_scatter.py --reps 500
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
from common import TARGET, load_raw  # noqa: E402

SUB = os.path.join(ROOT, "submissions")
HERE = os.path.join(ROOT, "experiments")
COMP = "playground-series-s6e8"

N_TEST = 296_302
F_PUBLIC = 0.20
N_PUB = int(round(N_TEST * F_PUBLIC))          # 59,260
GRID = 1e-5                                    # the LB reporting grid

# The gate. Stored numbers this run must reproduce before anything downstream is believed.
# Sources: experiments/audit_results.csv (cross-fitted CV) and experiments/w14d_bandmap.log.
GATE = {"blend159av_h3": 0.9700491721182696,
        "w16i_schemeavg": None,               # printed, compared to audit below
        "w16n_finegrid": 0.9700556964750705}
GATE_BANDMAP = ("blend159av_h3", 0.970049)

# The files the deadline pick and the auto-pick tiers are actually about.
LEADERS = ["w16i_schemeavg", "w16n_finegrid", "w16e_aonly", "w16q_ens4avg", "w16t_cellens4",
           "w16h_h3av6", "blend159av_h3", "blendtop3", "w16l_maskw_h3"]


def auc(y, v):
    """Exact ROC AUC with tie handling, on an arbitrary subset."""
    o = np.argsort(v, kind="stable")
    s = v[o]
    yy = y[o]
    n = s.size
    n1 = float(yy.sum())
    n0 = float(n) - n1
    if n1 <= 0 or n0 <= 0:
        return float("nan")
    b = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    ends = np.concatenate((b[1:], [n]))
    avg = (b + ends + 1) * 0.5
    pos = np.add.reduceat(yy, b)
    return (float((avg * pos).sum()) - n1 * (n1 + 1.0) / 2.0) / (n1 * n0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--out", default=os.path.join(HERE, "w17a_cvlb_scatter.json"))
    a = ap.parse_args()

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
                         capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    assert (sub["status"] == "SubmissionStatus.COMPLETE").all(), "a submission is not COMPLETE"
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    agg = sub.groupby("stem")["publicScore"].agg(["max", "min"])
    dup = agg[agg["max"] != agg["min"]]
    print(f"live submissions {len(sub)}, distinct files {len(agg)}, "
          f"files whose repeats disagree {len(dup)} (must be 0 — scores are deterministic)")
    assert len(dup) == 0, dup
    LB = agg["max"].to_dict()

    tr, _ = load_raw()
    y = tr[TARGET].astype(int).to_numpy()
    n_pool = len(y)
    assert n_pool > N_TEST, "pool smaller than the test set — the geometry cannot be drawn"

    # ------------------------------------------------------------------------ the gate
    print(f"\nGATE — pool {n_pool:,} rows")
    for nm, want in GATE.items():
        p = os.path.join(SUB, f"oof_{nm}.npy")
        got = auc(y, np.load(p).astype("float64"))
        if want is None:
            print(f"  {nm:22s} pooled {got:.16f}  (no stored value to gate against)")
            continue
        d = got - want
        print(f"  {nm:22s} pooled {got:.16f}  stored {want:.16f}  drift {d:+.3e}")
        assert abs(d) < 1e-12, f"GATE FAILED on {nm}: drift {d:+.3e}"
    nm, want6 = GATE_BANDMAP
    got6 = round(auc(y, np.load(os.path.join(SUB, f"oof_{nm}.npy")).astype("float64")), 6)
    print(f"  {nm:22s} vs w14d_bandmap.log 6dp {got6:.6f} / {want6:.6f}  "
          f"{'OK' if got6 == want6 else 'FAILED'}")
    assert got6 == want6

    # ------------------------------------------------------------ the file set for the sim
    # CV is taken as the pooled AUC of the stored OOF vector, NOT from audit_results.csv.
    # That csv is missing three of the files the whole auto-pick argument turns on
    # (w16e_aonly, w16q_ens4avg, w16t_cellens4), and the first run of this script silently
    # dropped all three — including both halves of the pair slot 10 called a falsification.
    names = sorted(nm for nm in LB
                   if os.path.exists(os.path.join(SUB, f"oof_{nm}.npy")))
    missing = sorted(set(LB) - set(names))
    print(f"\nsent files with a stored OOF vector: {len(names)} of {len(LB)}")
    print(f"  no OOF vector, excluded: {', '.join(missing) if missing else 'none'}")
    for nm in LEADERS:
        assert nm in names, f"leader {nm} has no OOF vector — the sim would be blind to it"

    V = {nm: np.load(os.path.join(SUB, f"oof_{nm}.npy")).astype("float64") for nm in names}
    for nm, v in V.items():
        assert len(v) == n_pool, f"{nm}: {len(v)} rows != {n_pool}"
    pooled = {nm: auc(y, V[nm]) for nm in names}

    # ------------------------------------------------------------------------ the draws
    print(f"\n=== SIMULATION: {a.reps} draws, pseudo-test {N_TEST:,} -> public {N_PUB:,} ===")
    rng = np.random.default_rng(a.seed)
    pub = {nm: np.empty(a.reps) for nm in names}
    for r in range(a.reps):
        idx = rng.permutation(n_pool)[:N_PUB]
        yy = y[idx]
        for nm in names:
            pub[nm][r] = auc(yy, V[nm][idx])
        if (r + 1) % 100 == 0:
            print(f"  draw {r+1}/{a.reps}", flush=True)

    single = {nm: float((pub[nm] - pooled[nm]).std(ddof=1)) for nm in names}
    bias = {nm: float((pub[nm] - pooled[nm]).mean()) for nm in names}
    single_med = float(np.median(list(single.values())))
    bias_med = float(np.median(list(bias.values())))
    print(f"\n  single-file slice sd: median {single_med*1e6:.1f}e-6   "
          f"common small-sample bias: median {bias_med*1e6:+.1f}e-6")
    print("  BOTH are common to all 50 files (one fixed slice) and are absorbed by the")
    print("  gap constant. Neither is the quantity the CV->LB residual measures.")

    # ------------------------------------------------------------------ the paired quantity
    prs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            dpool = pooled[ni] - pooled[nj]
            dsl = pub[ni] - pub[nj]
            prs.append(dict(a=ni, b=nj, dcv=dpool, dlb=LB[ni] - LB[nj],
                            sd=float((dsl - dpool).std(ddof=1)),
                            p_reverse=float((np.sign(dsl) != np.sign(dpool)).mean())))
    PR = pd.DataFrame(prs)
    paired_med = float(PR["sd"].median())
    print(f"\n  PAIRED slice sd over {len(PR)} pairs: median {paired_med*1e6:.1f}e-6  "
          f"range {PR['sd'].min()*1e6:.1f}–{PR['sd'].max()*1e6:.1f}e-6")
    print(f"  median P(the slice reverses the pooled ordering) {PR['p_reverse'].median():.3f}")

    L = PR[PR["a"].isin(LEADERS) & PR["b"].isin(LEADERS)]
    lead_med = float(L["sd"].median())
    print(f"\n  restricted to the {len(LEADERS)} LEADER files ({len(L)} pairs): "
          f"paired sd median {lead_med*1e6:.1f}e-6  "
          f"range {L['sd'].min()*1e6:.1f}–{L['sd'].max()*1e6:.1f}e-6")
    print(L.assign(dcv_e6=lambda d: d["dcv"] * 1e6, dlb_e6=lambda d: d["dlb"] * 1e6,
                   sd_e6=lambda d: d["sd"] * 1e6)
          .sort_values("sd")[["a", "b", "dcv_e6", "dlb_e6", "sd_e6", "p_reverse"]]
          .round(2).to_string(index=False))

    # ---------------------------------------- the pairs slot 10 called a ladder falsification
    print("\n  THE PAIRS THE FALSIFIED LADDER WAS READ OFF")
    focus = [("w16e_aonly", "w16i_schemeavg"), ("w16e_aonly", "w16n_finegrid"),
             ("w16q_ens4avg", "w16t_cellens4"), ("w16i_schemeavg", "blend159av_h3"),
             ("w16e_aonly", "blend159av_h3")]
    detail = {}
    for x, z in focus:
        row = PR[((PR["a"] == x) & (PR["b"] == z)) | ((PR["a"] == z) & (PR["b"] == x))]
        assert len(row) == 1, f"{x} vs {z} not found — the file set is wrong"
        row = row.iloc[0]
        sgn = 1.0 if row["a"] == x else -1.0
        dcv, dlb, sd = sgn * row["dcv"], sgn * row["dlb"], row["sd"]
        zz = abs(dlb - dcv) / sd
        print(f"    {x:16s} vs {z:16s}  dCV {dcv*1e6:+6.2f}e-6  dLB {dlb*1e6:+6.1f}e-6  "
              f"paired sd {sd*1e6:5.1f}e-6  (dLB-dCV) = {zz:5.2f} sd  "
              f"P(reverse) {row['p_reverse']:.3f}")
        detail[f"{x}|{z}"] = dict(dcv=float(dcv), dlb=float(dlb), sd=float(sd),
                                  z=float(zz), p_reverse=float(row["p_reverse"]))

    tot = float(np.sqrt(paired_med ** 2 + GRID ** 2 / 12.0))
    print(f"\n  paired sd + 1e-5 grid rounding in quadrature: {tot*1e6:.1f}e-6")
    print("  -> this is the number w17b_famfix.py compares the observed scatter against.")

    out = dict(reps=int(a.reps), seed=int(a.seed), n_files=len(names),
               sim_single_sd=single, sim_single_sd_median=single_med,
               sim_single_bias_median=bias_med,
               sim_paired_sd_median=paired_med, sim_paired_sd_leaders=lead_med,
               sim_paired_p_reverse_median=float(PR["p_reverse"].median()),
               sim_paired_plus_grid=tot,
               focus_pairs=detail, pooled={k: float(v) for k, v in pooled.items()})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    PR.to_csv(os.path.join(HERE, "w17a_pairs.csv"), index=False)
    print(f"\nwrote {a.out}, w17a_pairs.csv")


if __name__ == "__main__":
    main()
