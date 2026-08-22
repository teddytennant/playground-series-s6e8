"""w60c — price the two corrected-ens4 arms as HIJACKERS, through w59a's own model.

Not a second pricer. This imports `w59a_hijackprice`, sets its `EXTRA_CAND` what-if hook to the
two files w60a built, and turns `WRITE_ARTEFACT` off so the enlarged bracket cannot persist a
bar the sender would then read (w59a asserts that combination itself). The bar quoted in this
run's output is therefore NOT the live bar and is not meant to be — the live bar stays whatever
the registered `w59a_hijackprice.json` says until a send day puts these files in PLAN_0823.

What it answers, against w60_prereg P5: does an ELIGIBLE file landing above the tier beat the
+7.855e-6 status quo, and does it beat the +3.599e-6 that `w40_ad211std_h3` scored before w60
found ARM 211 is barred from being a final entry at all?

    .venv/bin/python experiments/w60c_lever.py
"""
from __future__ import annotations

import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import w59a_hijackprice as W59  # noqa: E402

NEW = ("w36_ad199stdcorr_ens4", "w38_ad202stdcorr_ens4")

W59.EXTRA_CAND = NEW
W59.WRITE_ARTEFACT = False
out = W59.main()

base = out["base"]
floor = next(r["uncond"] for r in out["rows"] if r["stem"] == out["pick"])
print("\n=== w60 P5: the lever, on files that are ELIGIBLE to be final entries ===")
print(f"  status quo (no hijack, 5-file tier)                  {base:+8.3f}e-6")
print(f"  floor: the PICK itself above the tier (w59 §7 P5)    {floor:+8.3f}e-6")
for x in NEW:
    r = next((r for r in out["rows"] if r["stem"] == x), None)
    if r is None:
        print(f"  ⛔ {x} was not priced — no OOF vector?")
        continue
    # GATE I (w59a): an IN-tier landing is the same quantity at 1/3 the leverage.
    dil = (r["uncond"] - base) / 3.0
    print(f"  {x:24s} dcv {r['dcv']:+7.2f}  above {r['uncond']:+8.3f}e-6"
          f"   in-tier {base + dil:+8.3f}e-6")
print(f"\n  BOTH LANDINGS ARE PRICED because neither is certain: these files carry"
      f"\n  P(above tier) 0.70 / 0.59, so ~3-4 times in 10 they land IN the tier instead and"
      f"\n  dilute it. w59a GATE I says that is the SAME number at 1/3 the leverage, so a file"
      f"\n  that helps above the tier helps inside it too. There is no bad branch here.")
