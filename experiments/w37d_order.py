"""w37d — fix the 08-21 send ORDER explicitly, and say why each slot is spent that way.

The w36 build changed the queue's economics and w37b/c changed what is in it. Neither the
pricer nor `send_rank` defaults get this right on their own:

  * The pricer's p_beat is MARGINAL -- P(file beats the 0.97118 account best). It does not
    condition on the leader having been sent an hour earlier. The five w36_ad199std* siblings
    are near rank-identical to the leader and to each other, so their p_beat of 0.79-0.98
    massively overstates their INCREMENTAL value once slot 1 has landed. Sending all five is
    close to sending one file five times.
  * The w37 calibration sends have p_beat exactly 0, by construction, and are nonetheless the
    highest-information files in the queue: each is a ~7 sigma discriminator on a quantity
    (`RESEARCH.md`'s +2.7e-4 es-on-val figure) that currently rests on ONE reading.

So the order is set by hand, and the reason is written next to each slot.
"""
import pandas as pd, numpy as np

W = "/home/nixos/all-my-repos/ai/kaggle-agents/workspace/playground-series-s6e8/"

ORDER = [
 ("w36_ad199stdcorr.csv",     "CV LEADER 0.9701400060 and check_selection.WANTED. Unsent = "
                              "NOT SELECTABLE, which outranks everything else in the queue."),
 ("w36_ad199std_rescale.csv", "Best non-leader by the pricer (p_beat 0.980). One sibling of the "
                              "199 pack is worth a slot; five are not, they are near-duplicates."),
 ("w37_cal_ram_hgb.csv",      "R1, CLEAN AUDIT. Checks the title-LB that one of the three points "
                              "under the es-bias line rests on. A miss WITHDRAWS a RESEARCH.md "
                              "claim, so it goes early and is read first."),
 ("w37_cal_mkt_realmlp.csv",  "R2, CLEAN ANCHOR. Extrapolation test 9.9e-3 below the fitted "
                              "range; without it five of the dirty readings cannot be priced."),
 ("w36_ad199std.csv",         "Second sibling, p_beat 0.967, different transform family (ens4)."),
 ("w37_cal_om_xgb2.csv",      "R3 dirty reading 1/5. Highest-solo quarantined member, and the "
                              "AUC where the line is tightest."),
 ("w37_cal_omid_tabm.csv",    "R3 dirty reading 2/5. The most aggressive es form on the list "
                              "('Restoring best model' per fold) -- largest expected shortfall."),
 ("w37_cal_dm_cat.csv",       "R3 dirty reading 3/5. CatBoost od_type -- a DIFFERENT mechanism, "
                              "which is what the between-member sd is made of."),
 ("w36_ad199std_hybrid.csv",  "DEQUEUED w36_ad199std_logit and promoted this in its place (w39 "
                              "§4). `logit` carries a +1121e-6 LB-CV gap -- the largest of any "
                              "family over the 78 scored files -- while sitting 82.9e-6 of CV "
                              "below the leader, so on the DEFAULT auto-selection (nothing is "
                              "selected, and the default picks on public score) it displaces the "
                              "CV pick. `hybrid` is the highest-CV file left unqueued "
                              "(0.9701231283) and completes the transform sweep on the 199 pack."),
 ("w34_ad195stdcorr.csv",     "Previous CV leader, still unsent. Journal w36 §8.2 asked for it; "
                              "the 199 build displaced it from the head but not from the day."),
 # --- 08-22 and after, in this order ---
 ("w37_cal_ravi_realmlp1c.csv","R3 dirty reading 4/5. RealMLP additive patience; low-AUC, so it "
                              "also tests whether the anchor really did extend the line's reach."),
 ("w37_cal_dkv_xgb.csv",      "R3 dirty reading 5/5. The MILDEST es form here (early_stopping_"
                              "rounds=150). If the bias is common this one still shows it; if it "
                              "is mechanism-specific, this is the reading that reveals that."),
 ("w36_ad199std_h3.csv",      "Fourth sibling, p_beat 0.791."),
]

q = pd.read_csv(W + "experiments/w26d_queueprice.csv")
for c in ("send_rank", "msg", "why"):
    if c not in q.columns:
        q[c] = np.nan
q["why"] = q["why"].astype(object)
q["priority"] = 0
q["send_rank"] = np.nan

missing = [f for f, _ in ORDER if f not in set(q.file)]
assert not missing, f"not in the queue: {missing}"

for i, (f, why) in enumerate(ORDER, 1):
    m = q.file == f
    q.loc[m, "priority"] = 1
    q.loc[m, "send_rank"] = i
    q.loc[m, "why"] = why

q = q.sort_values(["priority", "send_rank", "pred_lb"], ascending=[False, True, False])

# w40: the write is behind __main__. This script rebuilds send_rank/why from ORDER above and
# blanks anything ORDER omits, so an `import w37d_order` to reuse a name used to silently
# rewrite the send plan. Same defect w39 §1 fixed in w26d_queueprice.py, same fix.
if __name__ != "__main__":
    print("w37d_order imported, not run -- the queue CSV was NOT written.")
else:
    q.to_csv(W + "experiments/w26d_queueprice.csv", index=False)

print(f"{'#':>3} {'file':30s} {'pred_lb':>9} {'p_beat':>8}  why")
for r in q[q.priority == 1].itertuples():
    mark = " <-- 08-21 day ends here" if r.send_rank == 10 else ""
    print(f"{int(r.send_rank):3d} {r.file:30s} {r.pred_lb:9.6f} {r.p_beat:8.2e}  {r.why[:58]}{mark}")
print(f"\n{len(ORDER)} files ordered explicitly; {len(q)-len(ORDER)} left to the pricer behind them.")
