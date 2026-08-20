"""w36f -- an instrument for the quarantined es-on-val members that is IMMUNE to their bias.

The open question (w34 §10 item 5): 13 members sit in `ext_members*es/` because they early-stop
or best-epoch on the very fold their OOF reports. Their OOF is inflated, so a stacker fitted on
OOF over-weights them and buys CV the LB will not pay. Refitting the pack with them in cannot
answer this, because the pack refit is scored on the same inflated OOF.

THE IDEA. es-on-val inflates a member's OOF. It does NOT inflate its TEST predictions -- those
come from the same fitted model applied to unseen rows. So for any member whose author
published a public-LB score, the quantity

    offset = LB(their test predictions) - AUC(their OOF, OUR frozen folds)

is inflated-OOF-sensitive in exactly one direction: a member whose OOF is optimistic by b has
an offset SMALLER by b than an honest member of the same true strength. Calibrate the honest
offset on members that passed every gate, then read each dirty member's shortfall.

⚠⚠ POWER. This runs on THREE honest calibration points, because only three objects in this
workspace have both an our-fold OOF AUC and a published LB. Two free parameters, so there is
exactly ONE residual degree of freedom and the fitted line is barely tested. Everything below
is a SKETCH WITH A NUMBER ON IT, not a result. It is written down because the method is sound
and cheap to extend: every future clean import whose kernel title carries `lb-0-XXXXX` adds a
calibration point, and four or five would make this decisive.

WHY THE OFFSET IS NOT A CONSTANT. AUC differences compress towards 1, so a weaker model has a
larger LB-minus-OOF offset than a stronger one. That is why a slope is fitted rather than a
mean. Our own 91 sent submissions cannot supply the slope: they span 2.5e-4 of CV against an LB
grid of 1e-5, so the slope is unidentifiable there (RESEARCH's ladder section, same reason).
The member-level points span 2.1e-3 -- 200 grid steps -- which is what makes them usable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, our-fold OOF AUC, published public LB, clean?, where the LB came from)
PTS = [
    ("ram_hgb",     0.9680258266, 0.96945, True,
     "kernel title `s6e8-histgradientboosting-lb-0-96945`"),
    ("ram_lgb",     0.9682590393, 0.96965, True,
     "kernel title `s6e8-lgbm-lb-0-96965`"),
    ("w29_ad194stdcorr", 0.9701182875, 0.97118, True,
     "our own submission 55634054 -- clean by construction, cross-fitted on the frozen folds"),
    ("zwr_realmlp", 0.9691277390, 0.97009, False,
     "kernel title `s6e8-public-lb-0-97009-single-model-realmlp`"),
    ("tam_lkup",    0.9687560000, 0.97041, False,
     "kernel title `...-lookup-transformer-insights-lb-0-97041` -- ⚠ the title's LB may be the "
     "notebook's BLEND, not this member alone, AND the member is foreign-partition (3 folds). "
     "Reported, not used."),
]


def main():
    d = pd.DataFrame(PTS, columns=["member", "oof", "lb", "clean", "lb_source"])
    d["offset"] = d.lb - d.oof

    cal = d[d.clean]
    A = np.column_stack([np.ones(len(cal)), cal.oof.to_numpy()])
    beta, *_ = np.linalg.lstsq(A, cal.offset.to_numpy(), rcond=None)
    resid = cal.offset.to_numpy() - A @ beta
    dof = len(cal) - 2
    rms = float(np.sqrt((resid ** 2).sum() / max(dof, 1)))

    d["offset_expected"] = beta[0] + beta[1] * d.oof
    d["shortfall"] = d.offset_expected - d.offset          # >0 => OOF looks too good

    print("honest offset line, fitted on the CLEAN points only:")
    print(f"  offset = {beta[0]:+.6f} {beta[1]:+.4f} * oof_auc")
    print(f"  residual dof {dof}, rms {rms:.2e}   <-- the noise floor for every shortfall below")
    print()
    print(d.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    for _, r in d[~d.clean].iterrows():
        n = r.shortfall / rms if rms > 0 else float("nan")
        print(f"  {r.member:14s} shortfall {r.shortfall * 1e4:+.2f}e-4 "
              f"= {n:+.0f}x the residual rms")
        print(f"      -> its honest OOF AUC would be about {r.oof - r.shortfall:.6f} "
              f"(reported {r.oof:.6f})")
    print()
    print("READ THIS BEFORE QUOTING ANY NUMBER ABOVE: 1 residual degree of freedom.")
    d.to_csv(os.path.join(HERE, "w36f_esbias.csv"), index=False)
    json.dump(dict(beta=[float(b) for b in beta], dof=int(dof), rms=rms,
                   rows=d.to_dict("records")),
              open(os.path.join(HERE, "w36f_esbias.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
