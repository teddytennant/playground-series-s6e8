"""w62b — the w59 hijack CV bar must go VOID when the tier it was derived on moves.

WHAT THIS EXISTS FOR.
`w26g_send.hijack_cv_bar` carried this line above its staleness test:

    # The artefact must have been produced on the tier we are actually sending against.
    if not d.get("gate_t") == "PASS" or not (0.9 < bar < 1.0):

⚠ THOSE ARE NOT THE SAME CLAIM. `gate_t` is a STAMP recording that w59a's GATE T passed **on
the day the artefact was written**. A frozen stamp cannot notice the board moving afterwards,
and this bar moves with the tier — `hijack_cv_bar`'s own docstring says "H moves with the tier,
so this must be re-derived on every send day". On 2026-08-23 the tier moved from five files at
0.97118 to two at 0.97119, `w59a_hijackprice.py` began refusing to run on its own GATE T, and
`hijack_cv_bar` went on returning 0.9701294160 from the artefact GATE T had just voided.

w62 added the live SET comparison. This exercises it, both ways, with the negative controls
that separate it from a function that always returns the bar and from one that always voids.

⚠ `w59b_barguard.py` calls `hijack_cv_bar()` with no rows and therefore cannot catch a
regression here — that is why this file exists alongside it rather than inside it.

    .venv/bin/python experiments/w62b_barstaleguard.py
"""
from __future__ import annotations

import io, json, os, subprocess, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import w26g_send as S                                            # noqa: E402

COMP = "playground-series-s6e8"
FAILURES = 0


def check(name, ok, detail=""):
    global FAILURES
    print(f"  {'ok ' if ok else '*** FAILURE'}  {name:56s} {detail}")
    if not ok:
        FAILURES += 1


def rows_for(stems, value, filler=("z_filler_a", "z_filler_b")):
    """Submission rows whose TOP public score is exactly `stems` at `value`."""
    out = [dict(fileName=f"{s}.csv", publicScore=f"{value:.5f}") for s in stems]
    out += [dict(fileName=f"{f}.csv", publicScore=f"{value - 1e-5:.5f}") for f in filler]
    return out


def main():
    print("=" * 92)
    print("w62b: the hijack CV bar goes VOID when the auto-slot-1 SET moves")
    print("=" * 92)

    # ⚠⚠ w63: READ THE ARTEFACT THE SENDER READS, NOT A PATH RE-TYPED HERE. This line used to
    # be `json.load(open(os.path.join(HERE, "w59a_hijackprice.json")))`. When w63 re-pointed
    # `w26g_send.HIJACKPRICE` at the successor pricer, that hard-coded path would have gone on
    # testing a file the sender no longer opens — check 1 would have compared the LIVE bar
    # against a SUPERSEDED one and failed for the right reason by accident, and every other
    # check would have silently exercised the wrong artefact. Same shape as the defect this
    # file exists for: the comment states the rule ("the artefact"), the code names a proxy.
    art = json.load(open(S.HIJACKPRICE))
    rec = sorted(art["tiers"]["slot1"])
    recv = float(art["tiers"]["slot1_public"])
    bar = float(art["cv_bar_new"])
    print(f"  artefact: slot1 = {rec} @ {recv:.5f}, cv_bar_new = {bar:.10f}\n")

    # 1. THE MATCHED CASE — the bar is returned unchanged when the live tier IS the recorded one.
    got = S.hijack_cv_bar(rows_for(rec, recv))
    check("MATCHED tier -> the bar is returned", got is not None and abs(got - bar) < 1e-15,
          f"{got}")

    # 2. rows=None keeps the pre-w62 behaviour, so w59b's existing exercises still mean what
    #    they meant. A caller that cannot supply the board is not silently voided.
    check("rows=None -> unchanged (w59b's calls still valid)",
          S.hijack_cv_bar() is not None and abs(S.hijack_cv_bar() - bar) < 1e-15)

    # 3. THE MOVED CASE — a different SET at a different value voids it. ⚠ w63: this used to
    #    name the 08-23 move literally, which stopped being a MOVE the moment the successor
    #    artefact recorded that tier. Derived from the artefact instead, so it is a moved tier
    #    for any artefact this ever runs against.
    moved = sorted(set(rec) ^ {"z_moved_stem"})
    assert sorted(moved) != rec, "the 'moved' tier must differ from the recorded one"
    check("MOVED tier -> VOID", S.hijack_cv_bar(rows_for(moved, recv + 1e-5)) is None,
          f"{moved}")

    # 4. NEGATIVE CONTROL A — a SET comparison, not a COUNT. w61 §5: a queue's BLOCK COUNT was
    #    registered twice as an invariant and was a function of where a loop stopped. Same
    #    cardinality, different membership, must STILL void.
    same_n = sorted(rec[:-1] + ["w42_ad217stdcorr"])
    check("SAME SIZE, different membership -> VOID (set, not count)",
          len(same_n) == len(rec) and S.hijack_cv_bar(rows_for(same_n, recv)) is None,
          f"{len(same_n)} files both ways")

    # 5. NEGATIVE CONTROL B — one file added to the recorded tier voids it. A function that only
    #    compared the tier VALUE would pass this at the same 0.97118 and must not.
    check("recorded tier + one more file at the SAME value -> VOID",
          S.hijack_cv_bar(rows_for(rec + ["w42_ad217std"], recv)) is None)

    # 6. NEGATIVE CONTROL C — one file REMOVED voids it too, so the test is not one-sided.
    check("recorded tier MINUS one file -> VOID",
          S.hijack_cv_bar(rows_for(rec[:-1], recv)) is None)

    # 7. NEGATIVE CONTROL D — the check must not be a function that always voids. 1 and 2 above
    #    already show it returns the bar; make it explicit that BOTH outcomes are reachable from
    #    this file's own inputs, which is what separates a test from a tautology.
    reachable = {S.hijack_cv_bar(rows_for(rec, recv)) is not None,
                 S.hijack_cv_bar(rows_for(moved, 0.97119)) is None}
    check("both outcomes reachable (not a constant function)", reachable == {True})

    # 8. AND THE CONSEQUENCE, paired here so an edit cannot satisfy the bar test by breaking the
    #    thing the bar test is FOR: a None bar must BLOCK an above-tier file, never wave it on.
    r = dict(file="w36_ad199stdcorr_ens4.csv", stem="w36_ad199stdcorr_ens4", fam="ens4",
             cv=0.9701365875, pred_lb=0.971189)
    reason = S.above_tier_reason(pd.Series(r), None)
    check("a VOID bar BLOCKS an above-tier file", isinstance(reason, str) and len(reason) > 0,
          (reason or "")[:60])
    reason_ok = S.above_tier_reason(pd.Series(r), bar)
    check("and the SAME file is admitted when the bar is live", reason_ok is None)

    # ---------------------------------------------------------------- live status, REPORTED
    # ⚠ NOT asserted. Whether the artefact is stale RIGHT NOW is a fact about today, and a guard
    # that asserted it would start failing the moment the pricer chain is correctly re-run.
    try:
        raw = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                              "--page-size", "500"], capture_output=True, text=True).stdout
        sub = pd.read_csv(io.StringIO(raw))
        live = S.live_tier1(sub.dropna(subset=["publicScore"]).to_dict("records"))
        if live:
            lv, lset = live
            stale = sorted(lset) != rec
            print(f"\n  LIVE (reported, not asserted): slot1 = {sorted(lset)} @ {lv:.5f}")
            print(f"  the artefact is {'STALE — the bar is VOID today' if stale else 'CURRENT'}")
    except Exception as e:                                        # noqa: BLE001
        print(f"\n  (live board not read: {e})")

    print(f"\nFAILURES: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
