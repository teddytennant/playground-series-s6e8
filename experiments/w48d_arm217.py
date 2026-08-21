"""w48d -- ARM 217 falsifies its own pre-registration by z +12.6, and one imported member
explains all of it.

WHAT WAS REGISTERED. w44b_heldout.py fitted a per-member law on six import events 190->211 and
pre-registered ARM 217 (+6 members, ext_members16) BEFORE w42e had started building:

    point delta +1.95e-6   95% CI [-4.25, +8.16]e-6   predicted h3 base 0.9701351395
    "REGISTERED CALL: ARM 217 lands as another null. If it does, the import line is closed on
     six consecutive events, and no further arm should be built."

WHAT HAPPENED. w42e finished at 02:03 UTC on 08-21.

    ARM 211 h3 base   0.9701331846      (w40f_build.log, the matched control)
    ARM 217 h3 base   0.9701748950      (w42e_build.log)
    ACTUAL delta     +41.71e-6 over 6 members = +6.95e-6/member

The law's whole fitted history is +0.90e-6/member cumulative over 22 members, and its largest
single event was +5.49e-6/member over two. Six members at +6.95e-6 each is a regime break.
The registered null is dead: z = +12.6 against the registered CI.

⛔ AND THAT IS NOT GOOD NEWS. This file's point is that the falsification is NOT evidence that
importing works. Across all 177 member OOF vectors on this disk the standalone AUC distribution
is median 0.96649, p90 0.96834, p99 0.96923, and the highest of the other 176 is 0.96930. ONE
member sits at 0.9701816 -- above our entire 217-member cross-fitted stack, and +88.6e-6 clear
of the next best of 176. It arrived in ext_members16 and it is in ARM 217 and in nothing else.

A single public vector that alone out-AUCs a 217-member cross-fitted stack is not a base model.
The benign reading is that it is itself a strong public STACK; the malign one is that its "OOF"
was produced without nesting, so it depends on the target at its own rows. Both readings say
the same thing about ARM 217's CV: it is not comparable to the CVs of ad187..ad211, and
w42b_prereg's ⛔ WANTED-ineligibility clause -- written before any of this existed -- was right.

WHAT THIS FILE DOES. Quantifies the outlier, then prices the ONE clean experiment that would
settle it: send hboyang_mix's raw test vector as a w37-style calibration read. Its OOF AUC is
known exactly; the w37 line already established the OOF->LB map for raw member vectors from
five sends on 08-21. If the vector is honest it lands on that map. If its OOF is inflated by
non-nested stacking it lands far below.

    .venv/bin/python experiments/w48d_arm217.py
"""
from __future__ import annotations

import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ARM211, ARM217 = 0.9701331846, 0.9701748950
REG_POINT, REG_LO, REG_HI = 1.95, -4.25, 8.16          # w44b's pre-registration, e-6

# the five w37 calibration reads, oof AUC on our frozen folds -> actual public LB.
# Taken verbatim from the Kaggle submission log; every one of these is a RAW member vector,
# the same object type as hboyang_mix, which is why they are the right calibration class.
CAL = [("mkt_realmlp", 0.9581337349, 0.96285),
       ("dm_cat",      0.9667001254, 0.96813),
       ("omid_tabm",   0.9675077199, 0.96949),
       ("ram_hgb",     0.9680258266, 0.96945),
       ("om_xgb2",     0.9687331700, 0.96993)]
HBOYANG_OOF = 0.9701816

print("=" * 88)
print("w48d  ARM 217 vs ITS PRE-REGISTRATION")
print("=" * 88)
act = (ARM217 - ARM211) * 1e6
pse = (REG_HI - REG_POINT) / 1.96
print(f"\n  registered   {REG_POINT:+.2f}e-6  95% CI [{REG_LO:+.2f}, {REG_HI:+.2f}]  (se {pse:.2f})")
print(f"  ACTUAL       {act:+.2f}e-6 over 6 members = {act/6:+.2f}e-6/member")
print(f"  z vs the registered prediction  {(act-REG_POINT)/pse:+.1f}")
print(f"  ⛔ THE REGISTERED NULL IS FALSIFIED. Recorded as a miss, not re-explained.")
print(f"\n  for scale, the law's own fitted history: +0.90e-6/member cumulative over 22 members,")
print(f"  largest single event +5.49e-6/member over two. This is {act/6/0.90:.1f}x the cumulative rate.")

A = pd.read_csv(os.path.join(HERE, "w48a_member_auc.csv"))
top = A.sort_values("auc", ascending=False)
h = top.iloc[0]
print("\n" + "=" * 88)
print("THE OUTLIER THAT EXPLAINS IT")
print("=" * 88)
print(f"\n  {len(A)} member OOF vectors on disk. standalone AUC:")
for q in (50, 90, 99, 100):
    print(f"    p{q:<3d} {np.percentile(A.auc, q):.7f}")
print(f"\n    #1  {h.member:24s} {h.auc:.7f}   ({h['dir']})")
for _, r in top.iloc[1:5].iterrows():
    print(f"        {r.member:24s} {r.auc:.7f}   ({r['dir']})")
print(f"\n  gap #1 -> #2                     {(h.auc - top.iloc[1].auc)*1e6:+.1f}e-6")
print(f"  spread of #2..#25                {(top.iloc[1].auc - top.iloc[24].auc)*1e6:.1f}e-6")
print(f"  our ARM 217 cross-fitted h3 base  {ARM217:.7f}  -- the member BEATS it by "
      f"{(h.auc-ARM217)*1e6:+.1f}e-6")
print(f"\n  It is in ext_members16 only, which entered the pack only at ARM 217. Every pack")
print(f"  ad187..ad211 -- including the WANTED file's ad199 -- is clean of it.")

print("\n" + "=" * 88)
print("*** PRE-REGISTRATION: the hboyang_mix calibration read (for the 08-23 list) ***")
print("=" * 88)

# ⚠ FIRST ATTEMPT, DISCARDED AND RECORDED AS DISCARDED. A straight lb~oof line on the five
# w37 members gives slope 0.671, residual sd 335e-6 and a 95% interval of +/-791e-6 at
# hboyang's design point. That interval contains every outcome anyone could care about, so it
# decides nothing. Five scattered points extrapolated 14.5e-4 is not a calibration, and the
# rule built on it would have been unfalsifiable. It is left here as a comment because the
# next run will otherwise reinvent it.
#
# WHAT WORKS INSTEAD. Do not fit lb on oof; fit the GAP, lb - oof. Every object this account
# has ever scored has one, the five raw members AND the 83 stacks, so the anchor set is 88
# instead of 5, it spans oof 0.9581..0.9701, and hboyang's design point is INSIDE it rather
# than 14.5e-4 outside.
mem = pd.DataFrame([dict(name=n, oof=o, lb=l, kind="member") for n, o, l in CAL])
stk = pd.read_csv(os.path.join(HERE, "w46c_cvlb_live.csv"))[["stem", "cv", "lb"]]
stk.columns = ["name", "oof", "lb"]; stk["kind"] = "stack"
ALL = pd.concat([mem, stk], ignore_index=True)
ALL["gap6"] = (ALL.lb - ALL.oof) * 1e6
print(f"\n  anchors: {len(mem)} raw members + {len(stk)} stacks = {len(ALL)}, "
      f"oof {ALL.oof.min():.5f}..{ALL.oof.max():.5f}")
print(f"  hboyang_mix oof {HBOYANG_OOF:.7f} sits INSIDE that range.\n")
print(f"  {'oof band':>22s} {'n':>3s} {'mean gap':>10s} {'sd':>8s}")
bands = [(0.9580, 0.9660), (0.9660, 0.9690), (0.9690, 0.96995),
         (0.96995, 0.97010), (0.97010, 0.97015)]
for lo, hi in bands:
    b = ALL[(ALL.oof >= lo) & (ALL.oof < hi)]
    if len(b):
        print(f"  [{lo:.5f},{hi:.5f}) {len(b):3d} {b.gap6.mean():10.1f} {b.gap6.std():8.1f}")
near = ALL[ALL.oof >= 0.97005]
g_lo, g_hi = near.gap6.quantile(0.05), near.gap6.quantile(0.95)
g_mid = near.gap6.median()
print(f"\n  the {len(near)} anchors nearest hboyang (oof>=0.97005): gap median {g_mid:.0f}e-6, "
      f"5-95% [{g_lo:.0f}, {g_hi:.0f}]e-6")
pred = HBOYANG_OOF + g_mid * 1e-6
plo, phi = HBOYANG_OOF + g_lo * 1e-6, HBOYANG_OOF + g_hi * 1e-6
print(f"  -> PREDICTED LB {pred:.5f}   5-95% [{plo:.5f}, {phi:.5f}]")

# the second, fitting-free anchor: it claims a HIGHER oof than anything we own.
BEST_OOF, BEST_LB_ACC = 0.9701400060, 0.97118
print(f"""
  A SECOND ANCHOR THAT NEEDS NO FIT AT ALL, and it is the sharper one.
  Our best cross-fitted OOF ever is {BEST_OOF:.10f} (w36_ad199stdcorr) and it scored
  {BEST_LB_ACC:.5f}. hboyang_mix claims {HBOYANG_OOF:.7f} -- {{:+.1f}}e-6 MORE. The gap
  lb-oof shrinks slowly and stays near +1040e-6 for everything in this band, so a vector that
  really discriminates that well cannot score below our own best. It should MATCH OR BEAT
  {BEST_LB_ACC:.5f}.""".format((HBOYANG_OOF - BEST_OOF) * 1e6))

print(f"""
  READING RULE, fixed here before the file is ever sent:
    lb >= 0.97116  -> HONEST. Its OOF really is that good, ARM 217's +41.7e-6 is real
       conversion, w44's import line was closed too early and w45 section 3 has to be
       revisited on the record. This would be the largest single finding since ad187.
    lb <= 0.97080  -> INFLATED. That is {{:.0f}}e-6 below the nearest-anchor prediction and
       below what our own 0.96996-OOF stacks already score, which no honestly-{HBOYANG_OOF:.5f}
       vector can do. ARM 217's CV is then an imported artefact, its six files are never a
       deadline pick whatever their CV, and hboyang_mix comes out of any future pack.
    in between     -> report the number, change nothing, and do not retrofit a story onto it.

  ⛔ NONE OF THE THREE CHANGES THE WANTED PICK. w36_ad199stdcorr is ad199 and carries no
  ext_members16 stream; w48a re-derived it as the argmax under raw CV, under era-deflated CV,
  and (tied at the cap) under CV-REGION-capped CV.

  WHY 08-23 AND NOT 08-22. The 08-22 ten are registered in w47b_prereg.txt; the five probes
  settle ERA vs CV-REGION at 8 sems and are load-bearing, and the five drain files are worth
  under +0.5e-6 between them. Displacing one would be nearly free -- but the 08-23 list is not
  written yet, so NO prereg has to be amended at all, and by then the ERA/CV-REGION answer is
  in hand and prices this read better. Taking the free option is the whole point.""".format(
      (pred - 0.97080) * 1e6))

json.dump(dict(arm211=ARM211, arm217=ARM217, actual6=float(act), per_member=float(act / 6),
               registered=[REG_POINT, REG_LO, REG_HI], z=float((act - REG_POINT) / pse),
               falsified=True,
               outlier=dict(member=str(h.member), dir=str(h["dir"]), auc=float(h.auc),
                            gap_to_next_e6=float((h.auc - top.iloc[1].auc) * 1e6),
                            n_members=int(len(A))),
               n_anchors=int(len(ALL)), gap_median_e6=float(g_mid),
               hboyang_oof=HBOYANG_OOF, hboyang_pred=float(pred),
               hboyang_band=[float(plo), float(phi)],
               rule=dict(honest_at_or_above=0.97116, inflated_at_or_below=0.97080)),
          open(os.path.join(HERE, "w48d_arm217.json"), "w"), indent=1)
print("\nwrote w48d_arm217.json")
