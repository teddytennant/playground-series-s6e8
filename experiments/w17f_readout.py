"""w17f — read out the w17e test against the LIVE score, as likelihood ratios rather than a verdict.

`blend159av_logit` printed 0.97106. Model A (paired) put its point prediction there; B and C said
0.97107 and D said 0.97105. "A was right" is the cheap reading and it is not the one this
workspace is supposed to write — a point prediction with P 0.585 landing is worth about 2:1 to
5:1 against rivals, not a refutation of them. So each model is given its OWN uncertainty and the
likelihood ratio is computed. Rival sds:

  A  the paired slice sd measured on this exact pair (3.978e-6), convolved with the reference
     file's +/-5e-6 rounding.  This is the only one of the four with a MEASURED sd.
  B  w17b's published logit group residual sd, 32.1e-6 on the 4 files it could see.
  C  the fit residual of the 3-point slope regression, which with 3 points and 1 slope has
     1 dof — quoted, but it is barely a distribution at all.
  D  the decontaminated 3-blend group residual sd, 9.6e-6.

    .venv/bin/python experiments/w17f_readout.py
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
SHIP = "blend159av_logit"
GRID, HALF = 1e-5, 5e-6


def pgrid(mean, sd, v):
    """P(a score with this mean and sd rounds to the grid value v)."""
    return float(norm.cdf((v + HALF - mean) / sd) - norm.cdf((v - HALF - mean) / sd))


def main() -> None:
    j = json.load(open(os.path.join(HERE, "w17e_logitpair.json")))
    raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                          "--page-size", "200"], capture_output=True, text=True).stdout
    sub = pd.read_csv(io.StringIO(raw))
    sub["stem"] = sub["fileName"].str.replace(r"\.csv$", "", regex=True)
    row = sub[sub["stem"] == SHIP]
    assert len(row) == 1 and row.iloc[0]["status"] == "SubmissionStatus.COMPLETE", row
    obs = float(row.iloc[0]["publicScore"])
    print(f"{SHIP} ref {int(row.iloc[0]['ref'])}  PRINTED {obs:.5f}\n")

    cv = j["cv"][SHIP]
    # each rival: (label, mean, sd, its registered point prediction)
    Xb = np.array([j["cv"][k] for k in ("blend158_logit", "blend150sx_logit", "blend150fx_logit")])
    Yb = np.array([0.97106, 0.97104, 0.97103])
    resid_c = float(np.sqrt(np.sum((Yb - np.polyval(np.polyfit(Xb, Yb, 1), Xb)) ** 2) / 1))
    rivals = [
        ("A paired (registered)", None, None, j["pred_A"]),
        ("B published group gap", cv + j["gap_all4"], 32.1e-6, j["pred_B"]),
        ("C fitted slope", j["pred_C"], max(resid_c, 1e-9), j["pred_C"]),
        ("D decontaminated gap", cv + j["gap_blend3"], j["gap_blend3_sd"], j["pred_D"]),
    ]
    pA = {float(k): v for k, v in j["pred_A_dist"].items()}
    like = {}
    print(f"{'model':24s} {'point':>9s} {'sd used':>10s} {'P(printed 0.97106)':>20s} {'LR vs A':>9s}")
    for lab, mean, sd, point in rivals:
        p = pA.get(round(obs, 5), 0.0) if mean is None else pgrid(mean, sd, obs)
        like[lab] = p
        sdtxt = "3.98e-6*" if sd is None else f"{sd*1e6:.1f}e-6"
        lr = "" if mean is None else f"{like['A paired (registered)']/p:8.2f}x"
        hit = "  <- HIT" if abs(point - obs) < 1e-9 else ""
        print(f"  {lab:22s} {point:9.5f} {sdtxt:>10s} {p:20.3f} {lr:>9s}{hit}")
    print("   * A's sd is convolved with the reference file's own +/-5e-6 rounding; "
          "it is the only MEASURED one.")

    post = {k: v / sum(like.values()) for k, v in like.items()}
    print(f"\n  flat-prior posterior over the four models:")
    for k in sorted(post, key=lambda z: -post[z]):
        print(f"    {k:24s} {post[k]:.3f}")

    print(f"\n  READ IT HONESTLY. A's point prediction landed and it is the ONLY one of the four")
    print(f"  that did, in a regime slot 1 explicitly said the instrument does not explain. But")
    print(f"  A carried only P {pA.get(round(obs,5),0):.3f} on that value, so the evidence is")
    print(f"  {min(like['A paired (registered)']/v for k, v in like.items() if k[0] != 'A'):.1f}x"
          f"-{max(like['A paired (registered)']/v for k, v in like.items() if k[0] != 'A'):.1f}x, "
          f"not a refutation of the rivals. What it does settle:")
    print("    - the loose-family 'excess mechanism' is not needed to predict this file;")
    print("    - the PAIRED form beat BOTH group-gap forms, the contaminated one (B) and the")
    print("      decontaminated one (D), which named DIFFERENT wrong values on either side. A")
    print("      group gap is the wrong estimator for a within-family contrast even when the")
    print("      group is clean.")
    print(f"    - the tier risk registered at P 0.008 (a 0.97108 print) did NOT occur; the")
    print(f"      auto-slot-1 tie is unchanged at three files.")

    json.dump(dict(observed=obs, likelihood=like, posterior=post, resid_c=resid_c),
              open(os.path.join(HERE, "w17f_readout.json"), "w"), indent=1, sort_keys=True)
    print("\nwrote experiments/w17f_readout.json")


if __name__ == "__main__":
    main()
