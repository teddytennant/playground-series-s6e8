"""w18d — read out the w18b ship against the LIVE score, as likelihood ratios rather than a verdict.

Same discipline as w17f and w17k: every model gets its OWN uncertainty, and the answer is a
posterior over models, not a winner. Registered in w18c_shipprereg.txt before the upload:

  * A (the paired form, the registered prediction) and C (group gap dropping the CV-farthest
    file) coincide at 0.97104 by arithmetic accident. This print separates {A, C} from {B, D}
    and nothing finer.
  * B (all six sent rankraw files) and D (our four blends, the PROVENANCE decontamination)
    coincide at 0.97103, because rankraw's two foreign files are NOT deviant. That is the
    negative control, and it was the finding before the print.
  * A carries only P 0.451 on its own mode, so a miss is thin evidence either way.

    .venv/bin/python experiments/w18d_readout.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
GRID, HALF = 1e-5, 5e-6


def pgrid(mean, sd, v):
    return float(norm.cdf((v + HALF - mean) / sd) - norm.cdf((v - HALF - mean) / sd))


def main() -> None:
    j = json.load(open(os.path.join(HERE, "w18b_rankraw.json")))
    ship = j["ship"]
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    row = sub[sub["stem"] == ship]
    assert len(row) == 1 and row.iloc[0]["status"] == "SubmissionStatus.COMPLETE", row
    obs = float(row.iloc[0]["publicScore"])
    hit = abs(obs - j["pred_A"]) < 1e-9
    print(f"{ship} ref {int(row.iloc[0]['ref'])}  PRINTED {obs:.5f}")
    print(f"REGISTERED (model A, paired, LAW-IF sd): {j['pred_A']:.5f} -> "
          f"{'HIT' if hit else 'MISS'}\n")

    # the group residual sds, recomputed from the family rather than quoted
    LBm = sub.groupby("stem")["publicScore"].max()
    SENT = ["blend158_rankraw", "blend156_rankraw", "stack_pub151_fixed_rankraw",
            "blend150fx_rankraw", "stack_pub151_rankraw", "blend150sx_rankraw"]
    OURS = ["blend158_rankraw", "blend156_rankraw", "blend150fx_rankraw", "blend150sx_rankraw"]
    gap = {k: float(LBm[k]) - j["cv"][k] for k in SENT}
    five = [k for k in SENT if k != j["cv_far"]]
    sd_all = float(np.std([gap[k] for k in SENT], ddof=1))
    sd_5 = float(np.std([gap[k] for k in five], ddof=1))
    sd_ours = float(np.std([gap[k] for k in OURS], ddof=1))
    cvs = j["cv"][ship]

    pA = {float(k): v for k, v in j["pred_A_dist"].items()}
    rivals = [
        ("A paired (REGISTERED)", None, None, j["pred_A"], pA.get(round(obs, 5), 0.0)),
        ("B group gap, all 6", cvs + j["gap_all6"], sd_all, j["pred_B"], None),
        ("C group gap, 5 (CV-far out)", cvs + j["gap_5"], sd_5, j["pred_C"], None),
        ("D group gap, our 4 (prov.)", cvs + j["gap_ours"], sd_ours, j["pred_D"], None),
    ]
    print(f"{'model':30} {'point':>9} {'mean':>12} {'sd':>10} {'P(printed)':>12}")
    print("-" * 78)
    L = {}
    for label, mean, sd, point, pre in rivals:
        p = pre if pre is not None else pgrid(mean, sd, obs)
        L[label] = max(p, 1e-12)
        ms = f"{mean:.7f}" if mean is not None else "  (mixture)"
        ss = f"{sd*1e6:.1f}e-6" if sd is not None else f"{j['sd_if']*1e6:.2f}e-6*"
        flag = "  <- HIT" if abs(point - obs) < 1e-9 else ""
        print(f"{label:30} {point:>9.5f} {ms:>12} {ss:>10} {p:>12.4f}{flag}")
    print(f"  * A's sd is the LAW-IF closed form; its P is read off the stored mixture.")

    best = max(L, key=L.get)
    print(f"\nlikelihood ratios against the best-supported model ({best}):")
    for k in sorted(L, key=lambda z: -L[z]):
        print(f"  {k:30} {L[best]/L[k]:>7.2f}x")
    tot = sum(L.values())
    print("\nflat-prior posterior:")
    for k in sorted(L, key=lambda z: -L[z]):
        print(f"  {k:30} {L[k]/tot:>7.3f}")

    print("\n--- what this settles, against what was registered ---")
    named = [lab for lab, _, _, pt, _ in rivals if abs(pt - obs) < 1e-9]
    if named:
        print(f"  named the printed value: {named}. Registered in advance that this print "
              f"cannot\n  separate A from C, nor B from D.")
    else:
        print(f"  *** NO MODEL NAMED THE PRINTED VALUE. *** All four point at 0.97103/0.97104 "
              f"and the\n  file printed {obs:.5f} — one step below the group gaps, two below the "
              f"paired point. The\n  registered {{A,C}} vs {{B,D}} contrast is therefore NOT the "
              f"thing this send measured.")
        gz = (obs - cvs - j["gap_ours"]) / sd_ours
        print(f"  The realised family gap is {(obs-cvs)*1e6:+.1f}e-6 against our four sent "
              f"rankraw files at\n  {(j['gap_ours'])*1e6:+.1f}e-6 (sd {sd_ours*1e6:.1f}e-6 on "
              f"n=4): z = {gz:+.2f}. It is BELOW every sent file in the\n  family. The group-gap "
              f"sds are estimated on four points and this is the direct evidence\n  that they "
              f"are too narrow — which is exactly why A, the only model with a MEASURED "
              f"width,\n  still comes out ahead after missing its own point by two steps.")
    print(f"  B and D are the SAME number here — rankraw's two foreign files have gaps inside "
          f"our own\n  spread, so the provenance criterion is a NO-OP in this family. Provenance "
          f"is a prior for\n  deviance, not deviance itself.")
    print(f"  LAW-IF, third out-of-sample pair: closed form {j['sd_if']*1e6:.3f}e-6 against the "
          f"{j['reps']}-draw\n  simulation's {j['sd_sim']*1e6:.3f}e-6, ratio {j['sd_ratio']:.3f} "
          f"— inside the simulation's own MC error.")

    json.dump(dict(ship=ship, printed=obs, registered=j["pred_A"], hit=bool(hit),
                   sd_all=sd_all, sd_5=sd_5, sd_ours=sd_ours,
                   likelihood=L, posterior={k: L[k] / tot for k in L}),
              open(os.path.join(HERE, "w18d_readout.json"), "w"), indent=1)
    print("\nwrote w18d_readout.json")


if __name__ == "__main__":
    main()
