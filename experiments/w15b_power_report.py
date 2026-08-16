"""w15b step 5c -- read w15b_power.csv as a matched-control experiment.

The raw `recovered` column is not the answer on its own: the residual search costs AUC even
when there is nothing to find (its tau=0 false-positive arm reads -29e-6 at 25 rounds and
-1084e-6 at 750, purely from fitting noise on top of the pack). The signal-attributable
recovery is the DIFFERENCE against that arm at the SAME frame and the SAME round count:

    recovery = [ recovered(signal, frame, r) - recovered(none, frame, r) ] / injected

which is the matched-control design the workspace requires. Reported per checkpoint,
because the closed list's own instrument (resid_boost2.py) reads its best checkpoint.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "experiments", "w15b_power.csv"))

ctrl = (df[df.kind == "none"]
        .set_index(["frame", "rounds"])["recovered"].rename("toll"))
d = df[df.kind != "none"].join(ctrl, on=["frame", "rounds"])
d["net"] = d.recovered - d.toll
d["recovery_pct"] = 100 * d.net / d.injected

for (kind, tau), g in d.groupby(["kind", "tau"]):
    print(f"\n=== {kind}  tau={tau}   injected {g.injected.iloc[0]:+.6f} ===")
    print(f"  {'frame':5s} {'rounds':>6}  {'raw recovered':>14} {'control toll':>13}"
          f" {'NET':>10} {'recovery':>9}")
    for _, r in g.sort_values(["frame", "rounds"]).iterrows():
        print(f"  {r.frame:5s} {int(r.rounds):>6}  {r.recovered:+14.6f} {r.toll:+13.6f}"
              f" {r.net:+10.6f} {r.recovery_pct:8.1f}%")
    best = g.loc[g.recovery_pct.idxmax()]
    print(f"  -> best checkpoint: {best.frame} r={int(best.rounds)}  "
          f"recovery {best.recovery_pct:.1f}%")

print("\n=== summary: peak control-corrected recovery per injected signal ===")
s = (d.groupby(["kind", "tau", "frame"])["recovery_pct"].max().reset_index()
     .pivot(index=["kind", "tau"], columns="frame", values="recovery_pct"))
print(s.round(1).to_string())
d.to_csv(os.path.join(ROOT, "experiments", "w15b_power_net.csv"), index=False)
print("\nwrote experiments/w15b_power_net.csv")
