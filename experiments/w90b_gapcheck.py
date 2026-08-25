"""w90b — does the 0.97124 anchor's CV advantage convert to public LB at OUR OWN fitted rate?

The standing explanation for the public field being ahead of this account is w51's es-on-val
clause: the popular external member libraries are early-stopped on their validation folds, so
a stack fitted on them is optimistic in OOF and not in test. That predicts a SPECIFIC
signature -- a competitor's public-LB advantage should be SMALLER than their OOF advantage
implies. Until w90 there was no way to look, because no 0.9712x-cluster competitor had put an
OOF vector on a partition this workspace could verify. `atakanaldemir` did, w90a reproduced
their published AUC to 0.000e-6 on our own labels and folds, and their public score is known.

⛔ WHAT THIS DELIBERATELY DOES NOT DO. It does not call `w53a.predict_flags` on their file.
w53a's own docstring says an unknown family collapses onto the h3 reference level and that
"a price that looks real and is not" is how a pricer lies -- and a foreign stack has no family
in our design at all. So this works entirely in DIFFERENCES, where every additive term of the
design cancels, and uses only the fitted SLOPE. The price of that is one assumption, stated
here rather than buried: their file and our pick share a level in the CV->LB map.

⚠ AND THE PUBLIC SCORES ARE QUANTISED. Kaggle prints 5 decimals, so each score carries +-5e-6
of rounding and the OBSERVED gap is an INTERVAL, not a number. Reporting it as 50.0e-6 would
be a fabricated precision.

    .venv/bin/python experiments/w90b_gapcheck.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w53a_pricer as P   # noqa: E402

# GIVEN, all measured elsewhere and none of them produced by this file.
OUR_CV     = 0.9701400060     # w36_ad199stdcorr, our CV pick (w84a_pickargmax, rank 1 of 130)
OUR_LB     = 0.97118          # its realised public score
THEIR_CV   = 0.9701665486     # w90a, computed on OUR labels and OUR frozen folds
THEIR_LB   = 0.97124          # atakanaldemir's stated and dataset-titled public score
HALF_ULP   = 0.5e-5           # Kaggle prints 5 decimals


def main():
    slope_base = P.COEFS["cv6"]
    slope_era = slope_base + P.COEFS["era:cv6"]
    print("w53a M2 coefficients (public LB e-6):")
    for k, v in P.COEFS.items():
        print(f"  {k:16s} {v:+.4f}")
    print(f"\nin-sample residual sd {P.RESID_SD:.3f}e-6 per FILE; a DIFFERENCE of two carries "
          f"{P.RESID_SD * np.sqrt(2):.3f}e-6")

    dcv = (THEIR_CV - OUR_CV) * 1e6
    print(f"\nCV gap (their anchor - our pick), on the SAME verified partition: {dcv:+.2f}e-6")

    obs_lo = (THEIR_LB - HALF_ULP - (OUR_LB + HALF_ULP)) * 1e6
    obs_hi = (THEIR_LB + HALF_ULP - (OUR_LB - HALF_ULP)) * 1e6
    print(f"observed public gap: {THEIR_LB} - {OUR_LB} = [{obs_lo:+.1f}, {obs_hi:+.1f}]e-6 "
          f"after rounding, midpoint {(obs_lo + obs_hi) / 2:+.1f}e-6")

    se = P.RESID_SD * np.sqrt(2)
    print("\nunder the null 'their CV converts at our fitted rate':")
    for tag, s in [("base slope", slope_base)] + ([("era slope", slope_era)] if slope_era else []):
        pred = dcv * s
        # the observed interval and the prediction interval overlap?
        z_lo = (obs_lo - pred) / se
        z_hi = (obs_hi - pred) / se
        rej = "REJECTED" if (z_lo > 2 and z_hi > 2) or (z_lo < -2 and z_hi < -2) else "not rejected"
        print(f"  {tag} {s:+.4f}: predicted gap {pred:+.1f}e-6   observed - predicted in "
              f"[{obs_lo - pred:+.1f}, {obs_hi - pred:+.1f}]e-6 = [{z_lo:+.2f}, {z_hi:+.2f}] se  -> {rej}")

    # ---- THE SENSITIVITY THAT DECIDES WHETHER ANY OF THE ABOVE MEANS ANYTHING ----------
    fams = {k: v for k, v in P.COEFS.items() if k.startswith("fam[")}
    lo, hi = min(fams.values()), max(fams.values())
    hi2 = max(v for k, v in fams.items() if k != "fam[logit]")
    print(f"\nPOWER CONTROL -- the level assumption, priced. Our pick's own level is "
          f"h3 (reference)\n+ std {P.COEFS['std']:+.1f} + corr {P.COEFS['corr']:+.1f}. A foreign "
          f"stack's level is UNMEASURABLE from here, and\nour own family levels span "
          f"[{lo:+.1f}, {hi:+.1f}]e-6 ({hi2:+.1f} excluding the fam[logit] outlier).")
    pred = dcv * slope_era
    print(f"  effect under test        {pred:+.1f}e-6")
    print(f"  unmeasured level term    up to {hi2 - lo:+.1f}e-6 wide")
    print("\nREADING -- AND IT IS A REFUSAL TO READ, WHICH IS THE POINT. es-on-val "
          "contamination\npredicts the observed gap lands BELOW the prediction (their OOF "
          "flatters them, their test\ndoes not). At their level = ours it lands ABOVE. But the "
          "level term this comparison has\nto assume away is as large as the effect it is "
          "measuring, so the SIGN OF THE ANSWER FLIPS\nwithin the assumption. ⛔ This "
          "instrument cannot adjudicate es-on-val, and the honest\noutput is that it cannot, "
          "not a null. What would fix it: their file's family level, which\nis only knowable "
          "by holding several of their files at known CVs -- one file cannot give it.")
    print("\n⛔ NOT AN ACTION. w51's es-on-val clause gates IMPORTS, and nothing here imports, "
          "prices\n   or admits a member. Six days out the chain is not runnable and the "
          "0.97127 cluster stays\n   closed. This is a read of the field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
