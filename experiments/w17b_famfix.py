"""w17b — the EMPIRICAL half of the CV->LB re-cut, compared pair-by-pair against w17a's sim.

Three defects in the first pass of this analysis, all found by reading its own output:

 1. `family()` was w15j's NAME-SUFFIX rule, and every w15/w16 corrected file is named after the
    experiment rather than after the transform it sits on. So `w16b_cellweight` ...
    `w16n_finegrid` (built on the h3 base `blend159av_h3`) and `w16q_ens4avg` / `w16t_cellens4`
    (built on the ens4 base `blend159av`) all fell through to the "ens4" default, mixing the
    single most decision-relevant family in the workspace with its own opposite. Base
    transforms below are read off the live submission descriptions, not remembered.

 2. CV came from `audit_results.csv`, which has no row at all for `w16e_aonly`,
    `w16q_ens4avg` or `w16t_cellens4` and a NaN CV for three more. That silently dropped BOTH
    halves of the pair slot 10 called a falsification. CV is now the pooled AUC of the stored
    OOF vector, taken from w17a (which gates it against audit_results.csv where both exist).

 3. The scatter was compared against the SINGLE-FILE slice sd (~570e-6). Every file's public
    score is read off the SAME fixed slice, so that whole 570e-6 is common to all 50 files and
    is absorbed by the gap constant. The comparison below is per-PAIR, against that pair's own
    simulated paired sd — pairs of near-identical files have a paired sd of 3e-6, pairs of
    different transforms have 30e-6, and a single median hides the difference that matters.

    .venv/bin/python experiments/w17b_famfix.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, "experiments")
COMP = "playground-series-s6e8"
GRID = 1e-5
GRID_SD = GRID / np.sqrt(12.0)          # sd of one uniform rounding to the reporting grid

# Base transform per experiment-named file, read from the live submission descriptions
# (kaggle competitions submissions -v) on 2026-08-17. Not from memory.
BASE = {
    "w15e_antistudent": "h3", "w15f_antistudent_avg": "h3", "w15f_antistudent_cv": "h3",
    "w16b_cellweight": "h3", "w16d_membercell": "h3", "w16e_aonly": "h3",
    "w16f_armavg": "h3", "w16h_h3av6": "h3", "w16i_schemeavg": "h3",
    "w16l_maskw_h3": "h3", "w16m_widegrid": "h3", "w16n_finegrid": "h3",
    "w16q_ens4avg": "ens4", "w16t_cellens4": "ens4",
    "blendtop3": "h3",
}
CORRECTED = set(BASE) - {"blendtop3"}


def family(s: str) -> str:
    if s in BASE:
        return BASE[s]
    for f in ("wh3", "h3", "hybrid", "rankraw", "rescale", "logit", "w2", "w"):
        if s.endswith("_" + f) or (s.endswith(f) and f in ("w", "w2")):
            return f
    return "ens4"


def main() -> None:
    sim = json.load(open(os.path.join(HERE, "w17a_cvlb_scatter.json")))
    PR = pd.read_csv(os.path.join(HERE, "w17a_pairs.csv"))
    pooled = sim["pooled"]

    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v"],
                         capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    lb = sub.groupby("stem")["publicScore"].max()

    # ---- gate the OOF-derived CV against the stored audit CV wherever both exist
    aud = pd.read_csv(os.path.join(HERE, "audit_results.csv")).rename(columns={"name": "stem"})
    aud = aud[aud["cv"].notna()].set_index("stem")["cv"]
    both = [s for s in pooled if s in aud.index]
    drift = max(abs(pooled[s] - aud[s]) for s in both)
    print(f"GATE: {len(both)} files carry both a stored audit CV and an OOF vector; "
          f"max drift {drift:.3e}")
    assert drift < 1e-9, "OOF-derived CV disagrees with audit_results.csv"

    m = pd.DataFrame({"stem": list(pooled)})
    m["cv"] = m["stem"].map(pooled)
    m["lb"] = m["stem"].map(lb)
    m["fam"] = m["stem"].map(family)
    m["grp"] = m["fam"] + np.where(m["stem"].isin(CORRECTED), "+corr", "")
    m["gap"] = m["lb"] - m["cv"]
    assert m["lb"].notna().all()
    print(f"files in the analysis: {len(m)} (every sent file that has a stored OOF vector)")

    print("\n=== (A) EMPIRICAL, family = BASE TRANSFORM, corrected files kept separate ===")
    g = m.groupby("grp").agg(n=("gap", "size"),
                             gap_e6=("gap", lambda v: v.mean() * 1e6),
                             sd_e6=("gap", lambda v: v.std() * 1e6),
                             cv_span_e6=("cv", lambda v: (v.max() - v.min()) * 1e6),
                             lb_span_e6=("lb", lambda v: (v.max() - v.min()) * 1e6))
    print(g.round(2).to_string())
    m["resid"] = m["gap"] - m.groupby("grp")["gap"].transform("mean")
    dof = len(m) - m["grp"].nunique()
    print(f"\n  residual sd about CV + per-group gap: "
          f"{np.sqrt((m['resid']**2).sum()/dof)*1e6:.1f}e-6  (n {len(m)}, dof {dof})")
    print("  -- but this pools groups whose CV spans differ by 100x, so read the per-pair")
    print("     test below instead. It is the one that can fail.")

    # ------------------------------------------------------- the per-pair test
    fam = dict(zip(m["stem"], m["grp"]))
    PR["ga"], PR["gb"] = PR["a"].map(fam), PR["b"].map(fam)
    PR["same"] = PR["ga"] == PR["gb"]
    # residual of the pair: how far the LB difference sits from the CV difference
    PR["r"] = PR["dlb"] - PR["dcv"]
    # predicted sd of that residual under the null: slice draw + two grid roundings
    PR["sd_pred"] = np.sqrt(PR["sd"] ** 2 + 2 * GRID_SD ** 2)
    PR["z"] = PR["r"] / PR["sd_pred"]

    print("\n=== (A) vs (B), PER PAIR: is dLB - dCV the size slice geometry predicts? ===")
    for tag, sel in (("within-group pairs", PR["same"]), ("cross-group pairs", ~PR["same"])):
        q = PR[sel]
        rms = float(np.sqrt((q["r"] ** 2).mean()))
        pred = float(np.sqrt((q["sd_pred"] ** 2).mean()))
        print(f"  {tag:20s} n {len(q):4d}   observed rms(dLB-dCV) {rms*1e6:6.1f}e-6   "
              f"predicted {pred*1e6:6.1f}e-6   ratio {rms/pred:5.2f}")

    print("\n=== does dLB track dCV? (the claim slot 10 declared dead) ===")
    for tag, sel in (("within-group", PR["same"]), ("all pairs", PR["a"].notna())):
        q = PR[sel]
        rho, p = spearmanr(q["dcv"], q["dlb"])
        s = q[q["dlb"].abs() > 1e-9]
        k, n = int((np.sign(s["dcv"]) == np.sign(s["dlb"])).sum()), len(s)
        z = (k / n - 0.5) / np.sqrt(0.25 / n)
        print(f"  {tag:14s} n {len(q):4d}  Spearman {rho:+.3f} (p {p:.3g})   "
              f"grid-separated {n:4d}  sign agreement {k}/{n} = {k/n:.3f}  z {z:+.2f}")

    # ------------------------------------------------------- the outlier scan
    print("\n=== WHICH FILES SIT AWAY FROM THE RELATION? ===")
    print("  mean standardised residual of every pair a file appears in, sign-oriented so")
    print("  positive = this file's public score is HIGH for its CV.")
    rows = []
    for s in m["stem"]:
        za = PR.loc[PR["a"] == s, "z"]
        zb = -PR.loc[PR["b"] == s, "z"]
        v = pd.concat([za, zb])
        rows.append((s, fam[s], float(v.mean()), float(v.std(ddof=1)), len(v)))
    O = pd.DataFrame(rows, columns=["stem", "grp", "zbar", "zsd", "n"]).sort_values(
        "zbar", ascending=False)
    print(O.head(8).round(3).to_string(index=False))
    print("  ...")
    print(O.tail(5).round(3).to_string(index=False))

    print("\n=== THE PAIR SLOT 10 CALLED A FALSIFICATION, in the same units ===")
    for k, v in sim["focus_pairs"].items():
        x, z_ = k.split("|")
        sd_pred = float(np.sqrt(v["sd"] ** 2 + 2 * GRID_SD ** 2))
        print(f"  {x:16s} vs {z_:16s}  dCV {v['dcv']*1e6:+6.2f}e-6  dLB {v['dlb']*1e6:+6.1f}e-6"
              f"  predicted sd {sd_pred*1e6:5.1f}e-6  ->  {(v['dlb']-v['dcv'])/sd_pred:+5.2f} sd")

    out = dict(n=len(m),
               within_rms=float(np.sqrt((PR.loc[PR["same"], "r"] ** 2).mean())),
               within_pred=float(np.sqrt((PR.loc[PR["same"], "sd_pred"] ** 2).mean())),
               groups={str(k): int(v) for k, v in m["grp"].value_counts().items()},
               outliers=O.head(6).to_dict("records"),
               gap_by_group={str(k): float(v) for k, v in m.groupby("grp")["gap"].mean().items()})
    with open(os.path.join(HERE, "w17b_famfix.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    PR.to_csv(os.path.join(HERE, "w17b_pairs.csv"), index=False)
    O.to_csv(os.path.join(HERE, "w17b_outliers.csv"), index=False)
    m.sort_values("cv", ascending=False).to_csv(os.path.join(HERE, "w17b_sent.csv"), index=False)
    print("\nwrote w17b_famfix.json, w17b_pairs.csv, w17b_outliers.csv, w17b_sent.csv")


if __name__ == "__main__":
    main()
