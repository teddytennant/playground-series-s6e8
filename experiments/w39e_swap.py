"""w39 -- swap the one auto-selection hazard out of the 08-21 send order.

`w36_ad199std_logit` held send_rank 9. It is 82.9e-6 of CV below the leader -- four times the
next worst in the priority-1 band -- and w30b still prices it at 0.97119 because the `logit`
family carries a +1121e-6 LB-CV gap against h3's +1025e-6 (w39c). Nothing is selected, so
Kaggle auto-selects on PUBLIC score; a file like that can print top and DISPLACE the CV pick.
w39d prices the removal: the -82.9e-6 tail disappears from every cell of the 2x2, the worst
of 20,000 reps goes -82.90 -> -18.07e-6, P(the best auto-pick is >10e-6 below the CV leader)
falls 0.0059 -> 0.0002 at limit 2 and 0.1711 -> 0.1456 at limit 1, and E[CV of the best
auto-pick] gains +0.11 to +2.34e-6.

Replaced by `w36_ad199std_hybrid`, the highest-CV file left unqueued (0.9701231283, -16.9e-6),
so the slot is still spent -- the brief's economics are honoured, the day still sends ten --
and it completes the transform sweep on the 199-member pack (stdcorr / std / h3 / rescale /
hybrid all read on one member set).

⚠ This is the ONLY reason to prefer one queued file over another that this workspace accepts:
it is not a public-LB hedge and it is not tuned against feedback. It is CV, all the way down.
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "w26d_queueprice.csv")
OUT, IN = "w36_ad199std_logit.csv", "w36_ad199std_hybrid.csv"

MSG = (
    "w39 slot9 w36_ad199std_hybrid — the `hybrid` transform over the same 199-member "
    "standardised pack as the day's leader, cross-fitted CV 0.9701231283 on the frozen folds. "
    "It REPLACES w36_ad199std_logit in this slot, and the reason is auto-selection rather than "
    "public rank. Nothing is selected on this account, so Kaggle will auto-select on best "
    "PUBLIC score. The logit sibling sits 82.9e-6 of CV below the leader yet prices at 0.97119, "
    "because logit carries a +1121e-6 LB-CV gap against h3's +1025e-6 across 78 scored files — "
    "the same shape as blend158_logit, the file the w13 audit named as the whole of this "
    "account's selection exposure. Removing it takes the worst of 20,000 simulated "
    "auto-selections from -82.9e-6 to -18.1e-6 of CV. The brief is right that a spare slot is "
    "free for public RANK; it is not free for the FINAL score while the default picks on "
    "public. Not a deadline candidate: this is 16.9e-6 below the CV leader."
)

q = pd.read_csv(CSV)
assert (q.file == OUT).any() and (q.file == IN).any()
before = q[q.priority == 1].sort_values("send_rank").file.tolist()
rank = float(q.loc[q.file == OUT, "send_rank"].iloc[0])

q.loc[q.file == OUT, ["priority", "send_rank", "msg"]] = [0, float("nan"), float("nan")]
q.loc[q.file == OUT, "why"] = (
    "DEQUEUED w39: the queue's one auto-selection hazard. -82.9e-6 of CV yet priced 0.97119 "
    "on the logit family term; see w39d_logithazard.json."
)
q.loc[q.file == IN, ["priority", "send_rank", "msg"]] = [1, rank, MSG]
q.loc[q.file == IN, "why"] = (
    "Promoted w39 into send_rank 9, replacing the logit sibling. Highest-CV file left unqueued "
    "and it completes the transform sweep on the 199-member pack."
)
q.to_csv(CSV, index=False)

after = q[q.priority == 1].sort_values("send_rank")
print(f"send_rank {rank:.0f}: {OUT} -> {IN}\n")
print(f"{'send':>4}  {'file':28s} {'fam':8s} {'cv':>14s} {'pred':>8s}  {'msg':>3s}")
for r in after.itertuples():
    print(f"{int(r.send_rank):>4}  {r.file[:-4]:28s} {r.fam:8s} {r.cv:.10f} {r.pred_lb:8.5f}  "
          f"{'yes' if isinstance(r.msg, str) else ' - '}")
assert len(after) == len(before) == 13, "the day must still send ten from a 13-file band"
print("\nqueue still 13 deep, ranks 1..13 contiguous:",
      sorted(after.send_rank.tolist()) == list(range(1, 14)))
