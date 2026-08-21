"""w50c -- read the ARM 216 build against experiments/w50_prereg.txt R1-R4.

The prereg was committed at ec947af BEFORE the build started and BEFORE any ad216 AUC
existed on disk. This script does not choose a rule; it applies the one that was written.

    .venv/bin/python experiments/w50c_readout.py
"""
from __future__ import annotations

import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

# --- registered constants (w50_prereg.txt Sec.3) ---------------------------------------
LAW_SLOPE, LAW_SE = 0.325, 0.528     # w44b re-fit including ARM 211, e-6 per member
CALL_DROP, CALL_SIGNAL = 10.0, 20.0  # R1 / R2 thresholds
PRED_D_FIVE = 1.6                    # my registered point prediction, e-6
PICK_CV = 0.9701400060               # w36_ad199stdcorr, the WANTED first choice


def base_auc(log: str, pat: str) -> float:
    for line in open(os.path.join(HERE, log)):
        if pat in line and "OOF AUC" in line:
            return float(line.split("OOF AUC")[1].split()[0])
    raise SystemExit(f"no base AUC for {pat} in {log}")


def corr_cv(log: str) -> float | None:
    hits = re.findall(r"arm decile\s+xfit.*?CV\s+([0-9.]+)", open(os.path.join(HERE, log)).read())
    return float(hits[-1]) if hits else None


A211 = base_auc("w40f_build.log", "w40_ad211std_h3")
A216 = base_auc("w50a_build.log", "w50_ad216std_h3")
A217 = base_auc("w42e_build.log", "w42_ad217std_h3")

d_five = (A216 - A211) * 1e6
d_hbo = (A217 - A216) * 1e6
d_tot = (A217 - A211) * 1e6

print("=" * 78)
print("w50c  ARM 216 READOUT -- decomposing ARM 217's +41.71e-6")
print("=" * 78)
print(f"""
  ARM 211 h3 base   {A211:.10f}   (211 members, w40f)
  ARM 216 h3 base   {A216:.10f}   (216 = 217 minus hboyang_mix, w50a)
  ARM 217 h3 base   {A217:.10f}   (217 members, w42e)

  d_five     ARM216 - ARM211  = {d_five:+8.3f}e-6   over 5 members  ({d_five/5:+.3f}e-6/member)
  d_hboyang  ARM217 - ARM216  = {d_hbo:+8.3f}e-6   over 1 member   ({d_hbo:+.3f}e-6/member)
  ------------------------------------------------------------------
  total                         {d_five + d_hbo:+8.3f}e-6   vs {d_tot:+.3f}e-6 measured directly""")
assert abs((d_five + d_hbo) - d_tot) < 1e-9, "decomposition does not close"
print(f"  arithmetic closes exactly (this is a check, not a result)\n")

print(f"  SHARE OF THE +41.71e-6:")
print(f"    hboyang_mix, one member  {d_hbo/d_tot:6.1%}")
print(f"    the other five           {d_five/d_tot:6.1%}")
print(f"    one hboyang is worth {d_hbo/(d_five/5):.1f} members of the other five\n")

z_five = (d_five / 5 - LAW_SLOPE) / LAW_SE
z_hbo = (d_hbo - LAW_SLOPE) / LAW_SE
print(f"  vs w44a's per-member law ({LAW_SLOPE:+.3f}e-6/member, se {LAW_SE:.3f}):")
print(f"    the other five   z {z_five:+6.2f}   -- above the law, significantly")
print(f"    hboyang_mix      z {z_hbo:+6.2f}   -- not a member, an event\n")

print("-" * 78)
print("APPLYING THE REGISTERED RULES (w50_prereg.txt Sec.4)")
print("-" * 78)
if d_five <= CALL_DROP:
    branch, action = "R1 DROP", "hboyang_mix carries ARM 217; no ARM 216 family build."
elif d_five >= CALL_SIGNAL:
    branch, action = "R2 SIGNAL", "build the ARM 216 transform family, register a new day key."
else:
    branch, action = "R3 IN-BETWEEN", "report the number, change nothing, NO family build."
print(f"\n  d_five = {d_five:+.3f}e-6  ->  {branch}")
print(f"  ACTION: {action}")

print(f"\n  MY REGISTERED CALL WAS d_five <= {CALL_DROP:.0f}e-6 (R1), point prediction "
      f"{PRED_D_FIVE:+.1f}e-6.")
if d_five <= CALL_DROP:
    print("  -> CONFIRMED.")
else:
    print(f"  -> ⛔ WRONG. Actual {d_five:+.3f}e-6, {d_five - PRED_D_FIVE:+.1f}e-6 off the point")
    print(f"     prediction and {d_five - CALL_DROP:+.1f}e-6 past the R1 threshold. The other")
    print("     five are NOT ordinary members: they beat w44a's law at z "
          f"{z_five:+.1f}. What survives")
    print("     of the call is the DIRECTION -- hboyang_mix is still the single largest")
    print(f"     term at {d_hbo/d_tot:.0%} -- but 'essentially all of it' is refuted.")

cc = corr_cv("w50a_build.log")
print("\n" + "-" * 78)
print("WHAT IS NOW ON DISK, AND WHAT IT IS NOT")
print("-" * 78)
print(f"""
  w50_ad216std_h3     CV {A216:.10f}
  w50_ad216stdcorr    CV {cc if cc else float('nan'):.10f}   (+{(cc-PICK_CV)*1e6:+.2f}e-6 vs the WANTED pick)

  That is the highest cross-fitted CV ever built in this workspace, and unlike ARM 217 it
  is CLEAN of hboyang_mix. It is ALSO, per w50_prereg Sec.5, still WANTED-INELIGIBLE:
  w42b's clause is about the unread es-on-val status of the imported authors' notebooks,
  and five of the six remain unread after hboyang_mix is dropped. Dropping one member does
  not discharge that clause, and R3 forbids a family build on this number. Discharging it
  is a SEPARATE piece of work and it is the honest next question.
""" if cc else "\n  (w21a corr step had not written its CV when this ran)\n")

json.dump(dict(a211=A211, a216=A216, a217=A217, d_five=d_five, d_hboyang=d_hbo,
               d_total=d_tot, share_hboyang=d_hbo / d_tot, share_five=d_five / d_tot,
               per_member_five=d_five / 5, z_five_vs_law=z_five,
               branch=branch, action=action, call_confirmed=bool(d_five <= CALL_DROP),
               ad216_stdcorr_cv=cc, wanted_eligible=False),
          open(os.path.join(HERE, "w50c_readout.json"), "w"), indent=2)
print(f"wrote {os.path.join(HERE, 'w50c_readout.json')}")
