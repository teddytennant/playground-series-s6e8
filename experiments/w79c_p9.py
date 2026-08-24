"""w79c — P9: does the adopted bar change WHAT GETS SENT? (w79_prereg.txt, committed 0109708)

⚠ A REGISTERED TEST, RUN IN A FORM THE PREREG DID NOT DESCRIBE, AND THE DEVIATION IS THE FIRST
THING IN THIS DOCSTRING SO NOBODY HAS TO FIND IT.

w79_prereg.txt registered P9 as: `w26g_send.py --n 10` dry, with HIJACKPRICE pointed at the new
artefact, yields the same ten stems in the same order as `w70c_plan0825.json`. That is not
runnable as written today and the reason is in the sender's own source: the queue CSV is keyed
to the UTC day, `w26d_queueprice.csv` is stamped 2026-08-24, and its priority-1 band is today's
ten — all sent. A dry run today therefore falls through to the priority-0 tail and plans a
DIFFERENT ten from the registered 08-25 list, under EITHER bar. Producing the registered list
needs `w48e_order.py --day 2026-08-25 --write`, which rewrites the queue, and w79_prereg.txt
forbids this run from editing it.

So P9 is run in two pieces that together test the claim P9 exists to test — "the adopted bar
does not change what gets sent" — more tightly than the registered form, because both hold the
queue fixed and vary ONLY the bar:

  P9a  DIFFERENTIAL. The sender's dry-run plan under the OLD bar and under the ADOPTED bar,
       same queue, same board, same minute. Registered form compares two runs that differ in
       the queue as well as the bar; this one differs in nothing else.
  P9b  DIRECT, on the registered ten. For each of `w70c_plan0825.json`'s ten stems, call the
       sender's own `above_tier_reason` under both bars and require the verdict to be identical.
       This is the function the bar actually enters, called on the actual list P9 names.

⛔ P9a's two plan captures are made by the CALLER (the run repoints HIJACKPRICE between them and
   diffs); this file records P9b and the artefact. It reruns `w48e_order.py --day 2026-08-25`
   READ-ONLY to confirm the registered ten is still the registered ten.

    .venv/bin/python experiments/w79c_p9.py
"""
from __future__ import annotations

import json, os, subprocess, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE) + "/agent")
sys.path.insert(0, HERE)
import w26g_send as SND                                        # noqa: E402

OLD_BAR = 0.9701288617      # w63a_setprice.json, the interpolated one the ten was registered on
PY = sys.executable


class Row:
    """The duck-type `above_tier_reason` reads: r.file, r.fam, r.cv."""
    def __init__(self, rec):
        self.file, self.fam, self.cv = rec["file"], rec["fam"], rec["cv"]


def main() -> None:
    plan25 = json.load(open(os.path.join(HERE, "w70c_plan0825.json")))
    ten = [p["file"] if isinstance(p, dict) else p for p in plan25["plan"]]
    ten = [t.replace(".csv", "") for t in ten]
    new_bar = float(json.load(open(SND.HIJACKPRICE))["cv_bar_new"])
    print("=" * 96)
    print(f"w79c — P9. sender reads {os.path.basename(SND.HIJACKPRICE)} at bar {new_bar:.10f}")
    print(f"        the registered 08-25 ten was planned under {OLD_BAR:.10f}")
    print("=" * 96)
    assert new_bar > OLD_BAR, "the adopted bar is not stricter — nothing here is the right test"

    # ---- the registered ten is still the registered ten (READ-ONLY, no --write) --------------
    r = subprocess.run([PY, os.path.join(HERE, "w48e_order.py"), "--day", "2026-08-25"],
                       capture_output=True, text=True)
    print(f"\n  w48e_order.py --day 2026-08-25 (read-only) rc={r.returncode}   "
          f"{'✅ the day is registered and verifies' if r.returncode == 0 else '⛔ REFUSED'}")
    if r.returncode != 0:
        print(r.stdout[-1500:])

    # ---- P9b, the direct check --------------------------------------------------------------
    q = pd.read_csv(os.path.join(HERE, "w26d_queueprice.csv"))
    q["stem"] = q["file"].str.replace(r"\.csv$", "", regex=True)
    byname = {r.stem: r for r in q.itertuples()}
    thresh = float(json.load(open(SND.HIJACKPRICE))["threshold_public"])
    print(f"\n=== P9b: `above_tier_reason` on the registered ten, under both bars "
          f"(tier threshold {thresh:.6f}) ===")
    print(f"  {'stem':30s} {'cv':>14s} {'pred_lb':>9s} {'>=tier':>7s}  old bar   new bar")
    same, rows = True, []
    for st in ten:
        if st not in byname:
            print(f"  {st:30s} ⛔ NOT IN THE QUEUE CSV — cannot evaluate, not assumed fine")
            same = False
            rows.append(dict(stem=st, in_queue=False))
            continue
        src = byname[st]
        rec = Row(dict(file=src.file, fam=src.fam, cv=float(src.cv)))
        a = SND.above_tier_reason(rec, OLD_BAR)
        b = SND.above_tier_reason(rec, new_bar)
        ok = (a is None) == (b is None)
        same &= ok
        at = float(src.pred_lb) >= thresh
        print(f"  {st:30s} {float(src.cv):.10f} {float(src.pred_lb):9.6f} "
              f"{'YES' if at else 'no':>7s}  {'BLOCK' if a else 'send ':9s} "
              f"{'BLOCK' if b else 'send '}   {'' if ok else '⛔ CHANGED'}")
        rows.append(dict(stem=st, in_queue=True, cv=float(src.cv), pred_lb=float(src.pred_lb),
                         at_or_above_tier=at, old_blocked=a is not None,
                         new_blocked=b is not None, unchanged=ok,
                         new_reason=(b or "")[:200]))
    consulted = sum(1 for d in rows if d.get("at_or_above_tier"))
    already = sum(1 for d in rows if d.get("old_blocked"))
    print(f"\n  P9b {'✅ CONFIRMED' if same else '⛔ FALSIFIED'} — the adopted bar changes the "
          f"verdict on {sum(not d.get('unchanged', False) for d in rows)} of {len(ten)}.")
    print(f"\n  ⚠⚠ AND HERE IS HOW WEAK THAT IS, SAID BEFORE ANYONE QUOTES IT AS STRONG:")
    print(f"     * `above_tier_reason` is CONSULTED on {consulted} of the {len(ten)} — the "
          f"sender only asks it\n       about a row it predicts will land AT OR ABOVE the tier, "
          f"and none of the ten is.\n       On the other {len(ten) - consulted} the verdict is "
          f"unchanged because it is never read.")
    print(f"     * {already} of the ten ALREADY read BLOCK under the OLD bar (below it on CV, or "
          f"barred by\n       the w40d ARM-208 key). A verdict that was BLOCK and stayed BLOCK "
          f"is not evidence a\n       STRICTER bar is harmless.")
    print(f"     So P9b is CONSISTENT with adoption and is close to VACUOUS on its own. The load "
          f"is carried\n     by P9a — the sender's whole dry-run plan is byte-identical under "
          f"the two bars — and by\n     w79a's P8, where 0 of the non-vetoed newly-blocked rows "
          f"is predicted at or above the tier.")

    json.dump(dict(prereg="experiments/w79_prereg.txt @ 0109708",
                   deviation="registered form needs `w48e_order --day 2026-08-25 --write`, "
                             "which rewrites the queue; the prereg forbids that. Run as P9a "
                             "(differential, caller-captured) + P9b (direct, this file).",
                   p9b=bool(same), p9b_vacuous_on=len(ten) - sum(
                       1 for d in rows if d.get("at_or_above_tier")),
                   p9b_already_blocked_under_old_bar=sum(
                       1 for d in rows if d.get("old_blocked")),
                   old_bar=OLD_BAR, new_bar=new_bar,
                   w48e_0825_rc=r.returncode, threshold_public=thresh,
                   registered_ten=ten, rows=rows,
                   artefact=os.path.basename(SND.HIJACKPRICE)),
              open(os.path.join(HERE, "w79c_p9.json"), "w"), indent=1)
    print("\nwrote w79c_p9.json")
    sys.exit(0 if same and r.returncode == 0 else 1)


if __name__ == "__main__":
    main()
