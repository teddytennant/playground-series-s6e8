"""w60 (2026-08-22) — the `member` label must SURVIVE A REPRICE, because the send gate reads it.

THE BUG THIS EXISTS TO STOP COMING BACK
----------------------------------------
`w26g_send.above_tier_reason` tests `fam == "member"` FIRST. That branch is the only thing
stopping a raw member prediction vector being installed as a final entry while auto-selection
is live — concretely, `w48_cal_hboyang_mix`, the ARM 217 test, which prices at P(above the
0.97118 tier) = 1.000.

The label was written ONLY by `w48e_order.py`, which runs AFTER `w26d_queueprice.py` regenerates
`fam` from the filename suffix. w59 §9 recorded skipping `w48e_order.py` as safe "because the
gate lives in the sender". w60 re-ran the reprice and `w48_cal_hboyang_mix` came back
`fam=ens4`, `pred_lb` 0.971289, and `above_tier_reason` returned **None** — cleared to ship.

⚠ THE SHAPE, one level up from w58's "a rule enforced on the WRONG COLUMN is not enforced":
**a rule enforced on a column that a DIFFERENT script populates is only enforced if that script
ran.** The fix was to derive the label in `stdflag.family()`, where every other family label is
derived. This test is what keeps it derived.

  1. `stdflag.family()` returns "member" for every registered member file, with NO CSV involved.
  2. The label survives a round trip through the pricer: the live `w26d_queueprice.csv` — the
     artefact the sender actually reads — carries `fam == "member"` on each of them.
  3. `above_tier_reason` FIRES on the live `w48_cal_hboyang_mix` row, by name, on the member
     branch specifically — not on some later test that might be relaxed for other reasons.
  4. The pricer does NOT price a member row through the CV→LB model (w53a refuses an unknown
     family, correctly), and its `p_beat` is blank: a measurement must never sort on P(beat).
  5. A stack file is NOT swept up by the member rule — the classifier has to discriminate.

Exit 0 = the label is derived and biting. Run on any run that touches `stdflag.py`,
`w26d_queueprice.py`, `w26g_send.py`, or re-prices the queue.

    .venv/bin/python experiments/w60d_memberguard.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "agent"))

import stdflag                                                             # noqa: E402
import w26g_send as SND                                                    # noqa: E402

QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
THE_ONE = "w48_cal_hboyang_mix"          # the row that actually costs something if it slips
NOT_MEMBERS = ("w36_ad199stdcorr", "w36_ad199stdcorr_ens4", "w40_ad211std", "blend156_logit")

FAIL = []


def check(name, ok, detail=""):
    print(f"  {'ok ' if ok else '⛔ '} {name:56s} {detail}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    print("=== w60d: the `member` label is DERIVED, and it survives the pricer ===")

    # 1 — derived from the module, no CSV in the loop
    reg = sorted(stdflag.MEMBER_FILES)
    for s in reg:
        check(f"stdflag.family({s}) == 'member'", stdflag.family(s) == "member",
              stdflag.family(s))
    check("the w37_cal_* prefix rule classifies by construction",
          stdflag.family("w37_cal_dm_cat") == "member" and stdflag.is_member("w37_cal_anything"),
          "w37_cal_dm_cat -> member")

    # 5 — and it discriminates: a real stack must NOT be caught by it
    for s in NOT_MEMBERS:
        check(f"{s} is NOT swept up as a member", stdflag.family(s) != "member",
              stdflag.family(s))

    # 2 — the round trip through the pricer, on the artefact the sender reads
    if not os.path.exists(QUEUE):
        check("w26d_queueprice.csv exists", False, "missing")
        print(f"\nFAILURES: {len(FAIL)}")
        return 1
    q = pd.read_csv(QUEUE)
    q["stem"] = q.file.str.replace(".csv", "", regex=False)
    live = q[q.stem.map(stdflag.is_member)]
    check("every member row in the live queue carries fam='member'",
          len(live) > 0 and (live.fam == "member").all(),
          f"{int((live.fam == 'member').sum())}/{len(live)} rows")
    check("and their p_beat is blank (a measurement never sorts on P(beat))",
          live.p_beat.isna().all(), f"{int(live.p_beat.isna().sum())}/{len(live)} NaN")

    # 3 — FIRE IT on the live row, on the member branch specifically
    row = q[q.stem == THE_ONE]
    if row.empty:
        check(f"{THE_ONE} is in the live queue", False, "row absent")
    else:
        r = row.iloc[0]
        why = SND.above_tier_reason(r, SND.hijack_cv_bar())
        check(f"above_tier_reason BLOCKS {THE_ONE}", bool(why), str(why)[:40])
        check("  ...and it blocks on the MEMBER branch, not a later test",
              bool(why) and "fam=member" in why, str(why)[:40])
        # 4 — priced from its own registered instrument, not through the CV->LB model
        check("  ...and its pred_lb is its builder's registered figure (w49a), not a model price",
              abs(float(r.pred_lb) - 0.9712300) < 5e-6, f"{float(r.pred_lb):.6f}")

    print(f"\nFAILURES: {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print("w60d: the member label is derived in stdflag and bites in the sender.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
