"""w17k — read out the w17j test against the LIVE score, as likelihood ratios rather than a verdict.

`blend159av_hybrid` printed 0.97102. My REGISTERED prediction (model A, the paired form) was
0.97103 and it MISSED. Model D — the group gap over our 3 comparable blend files only, i.e. the
STRICTEST decontamination — put its point exactly on 0.97102.

That matters more than a one-step miss usually would, because it cuts against the rule this
workspace adopted four hours ago. RESEARCH.md currently says, from slot 2's logit test, "use the
PAIRED form, never a group gap" and "a group gap is the wrong estimator for a within-family
contrast even when the group is clean". This is the first out-of-sample test of that rule in a
different family and it went the other way.

Same discipline as w17f: each model gets its OWN uncertainty and the answer is a likelihood
ratio, not a verdict. A point prediction that misses by one grid step in a family whose members
scatter at sd 15e-6 is weak evidence, not a refutation — and by the same token D landing its
point is worth a couple of e-1, not a coronation.

  A  paired slice sd measured on THIS pair (5.759e-6), convolved with the reference file's
     own +/-5e-6 rounding. The only rival here with a MEASURED sd.
  B  the contaminated 6-file group residual sd.
  C  the 5-file (drop the one far file) group residual sd, 15.3e-6.
  D  the decontaminated 3-blend group residual sd, 13.3e-6.
  E  A's construction with the paired sd replaced by w17h's LAW-IF value (5.655e-6). E exists
     to test the LAW, not the point: it should track A almost exactly, and if it does that is
     an out-of-sample confirmation that the influence function reproduces a simulated paired
     sd it never saw.

    .venv/bin/python experiments/w17k_readout.py
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
SHIP = "blend159av_hybrid"
GRID, HALF = 1e-5, 5e-6


def pgrid(mean, sd, v):
    return float(norm.cdf((v + HALF - mean) / sd) - norm.cdf((v - HALF - mean) / sd))


def main() -> None:
    j = json.load(open(os.path.join(HERE, "w17j_hybridpair.json")))
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    row = sub[sub["stem"] == SHIP]
    assert len(row) == 1 and row.iloc[0]["status"] == "SubmissionStatus.COMPLETE", row
    obs = float(row.iloc[0]["publicScore"])
    print(f"{SHIP} ref {int(row.iloc[0]['ref'])}  PRINTED {obs:.5f}")
    print(f"REGISTERED (model A): {j['pred_A']:.5f}  -> "
          f"{'HIT' if abs(obs - j['pred_A']) < 1e-9 else 'MISS'}\n")

    cv = j["cv"][SHIP]
    # A and E are mixtures (reference rounding x paired draw); read their probability straight
    # off the stored distributions rather than re-deriving them.
    pA = {float(k): v for k, v in j["pred_A_dist"].items()}
    rivals = [
        ("A paired (REGISTERED)", None, None, j["pred_A"], pA.get(round(obs, 5), 0.0)),
        ("B group gap, all 6", cv + j["gap_all6"], 20.4e-6, j["pred_B"], None),
        ("C group gap, 5 files", cv + j["gap_5"], j["gap_5_sd"], j["pred_C"], None),
        ("D group gap, our 3", cv + j["gap_ours"], j["gap_ours_sd"], j["pred_D"], None),
    ]
    print(f"{'model':26} {'point':>9} {'mean':>12} {'sd':>10} {'P(printed)':>12}")
    print("-" * 74)
    L = {}
    for label, mean, sd, point, pre in rivals:
        p = pre if pre is not None else pgrid(mean, sd, obs)
        L[label] = max(p, 1e-12)
        ms = f"{mean:.7f}" if mean is not None else "  (mixture)"
        ss = f"{sd*1e6:.1f}e-6" if sd is not None else "5.76e-6*"
        flag = "  <- HIT" if abs(point - obs) < 1e-9 else ""
        print(f"{label:26} {point:>9.5f} {ms:>12} {ss:>10} {p:>12.4f}{flag}")
    print("  * A's sd is the measured paired slice sd; its P is read off the stored mixture.")

    best = max(L, key=L.get)
    print(f"\nlikelihood ratios against the best-supported model ({best}):")
    for k in sorted(L, key=lambda z: -L[z]):
        print(f"  {k:26} {L[best]/L[k]:>7.2f}x")
    tot = sum(L.values())
    print("\nflat-prior posterior:")
    for k in sorted(L, key=lambda z: -L[z]):
        print(f"  {k:26} {L[k]/tot:>7.3f}")

    print(f"\n--- what this does and does not settle ---")
    print(f"  the paired instrument is now 2 hits / 1 miss out of sample "
          f"(0.97105 hit, 0.97106 hit, 0.97103 vs {obs:.5f} MISS).")
    print(f"  D beats A by {L['D group gap, our 3']/L['A paired (REGISTERED)']:.2f}x here. "
          f"That is evidence, not a reversal:")
    print(f"  A still gave the printed value P {L['A paired (REGISTERED)']:.3f}, and one grid "
          f"step in a family scattering at sd {j['gap_ours_sd']*1e6:.1f}e-6 is thin.")
    print(f"  What it DOES kill is the universal form of slot 2's rule. 'Never a group gap' was "
          f"generalised\n  from ONE test in ONE family; in the hybrid family the decontaminated "
          f"group gap wins.")
    print(f"  Both contaminated variants (B, all 6) and the half-decontaminated one (C) failed, "
          f"so the\n  DECONTAMINATION half of slot 2's finding is confirmed a second time and "
          f"more strongly:\n  every foreign stack_pub* file has to come out, not just the far "
          f"one.")

    print(f"\n--- LAW-IF (this slot's proposed law), out of sample on a pair it never saw ---")
    print(f"  simulated paired slice sd {j['sd_sim']*1e6:.3f}e-6 | "
          f"LAW-IF {j['sd_if']*1e6:.3f}e-6 | ratio {j['sd_if_ratio']:.3f}")
    print(f"  model E (A with the LAW-IF sd) predicted {j['pred_E']:.5f}, identical to A, and "
          f"gives the\n  printed value P {pgrid(float(j['lb_ref']) + j['dcv'], float(np.hypot(j['sd_if'], HALF/np.sqrt(3)*np.sqrt(3))), obs):.4f} "
          f"under a normal approximation. The law reproduces the simulation;\n  it does not "
          f"rescue the paired form, because the two agree and BOTH missed.")

    json.dump(dict(ship=SHIP, printed=obs, registered=j["pred_A"],
                   hit=bool(abs(obs - j["pred_A"]) < 1e-9),
                   likelihood={k: L[k] for k in L},
                   posterior={k: L[k] / tot for k in L}),
              open(os.path.join(HERE, "w17k_readout.json"), "w"), indent=1)
    print(f"\nwrote w17k_readout.json")


if __name__ == "__main__":
    main()
