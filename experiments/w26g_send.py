"""w26g — drain the priced send queue safely, so a Kaggle day costs a slot almost nothing.

WHY THIS EXISTS
---------------
The brief's economics here are the opposite of the simulation competitions: submissions do
not evict each other, the public board shows best-of-all, so an unused daily slot is pure
waste. Ten slots a day need ten files a day. w23b enumerated the queue and w26d priced it
(P(any of the ten beats the account best) = 6.3e-4 -- draining it is free, not promising).
What has never existed is the last mile: a run still had to hand-pick filenames, hand-write
ten messages, and hand-check nothing had already been sent. This does that part.

The two ways a slot can be WASTED rather than merely unproductive, both guarded here:

  1. Re-sending a file that is already on the board. "Scores are deterministic ... resubmitting
     an identical file is genuinely pointless." Guarded by filename AND by md5, because the
     same predictions have been written under more than one stem in this workspace.
  2. Reading a TRUNCATED submission list. The CLI's default page size is 50 and this account
     passed 50 submissions on 2026-08-17; w17 slot 2 found two files hidden from every
     API-reading script here by exactly that. This asks for --page-size 500 and refuses to
     run if the list comes back exactly at the page size.

Sends NOTHING without --go. Refuses to exceed the 10/day cap, counted live from the API in
UTC, which is the day boundary Kaggle uses.

    .venv/bin/python experiments/w26g_send.py                 # plan only, sends nothing
    .venv/bin/python experiments/w26g_send.py --go --n 10     # actually send
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import os
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "agent"))

from common import SUB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = "playground-series-s6e8"
DAILY_CAP = 10
N_TEST = 296_302
PAGE = 500
QUEUE = os.path.join(HERE, "w26d_queueprice.csv")
TIERPRICE = os.path.join(HERE, "w57a_tierprice2.json")

# w58: the constants the ABOVE-TIER gate needs. `WANTED_INELIGIBLE` is the SAME dict
# `check_selection.assert_wanted_eligible` enforces -- imported, never re-typed, so retiring an
# entry there retires it here in the same commit.
import json as _json                                                        # noqa: E402
from check_selection import WANTED as _WANTED, WANTED_INELIGIBLE            # noqa: E402


HIJACKPRICE = os.path.join(HERE, "w59a_hijackprice.json")


def wanted_cv_bar():
    """⛔ SUPERSEDED BY `hijack_cv_bar` (w59). KEPT, NOT DELETED, AS THE DOCUMENTED WRONG ANSWER.

    This returned the CV of the WEAKER of the two WANTED files, on the argument that

        "a file that can be auto-selected displaces one of our two final entries, so the bar it
         has to clear is the bar the entry it displaces already cleared."

    ⚠ THAT ARGUMENT DESCRIBES THE WORLD WHERE WE CLICKED. Nothing is selected, so a hijacker
    does NOT displace a WANTED file -- it displaces A UNIFORM DRAW FROM TIER 1, which is a
    different and much lower-quality object. w59a measured the real break-even at 10.59e-6
    below the pick against this function's 24.93e-6: the bar was TOO LOOSE by 14.3e-6 of CV
    and admitted hijackers that make the exposure strictly WORSE than doing nothing.

    Still called by nothing on the send path. Retained so a future run that re-derives the
    "displaces a final entry" argument finds the measurement that refutes it.
    """
    try:
        cvs = _json.load(open(TIERPRICE))["cv"]
        vals = [float(cvs[w.replace(".csv", "")]) for w in _WANTED]
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return min(vals) if len(vals) == len(_WANTED) else None


def hijack_cv_bar(rows=None):
    """The CV a file must clear before it is allowed to risk landing ABOVE the auto tier (w59).

    MEASURED, not asserted. `w59a_hijackprice.py` prices the real counterfactual: if X lands
    above the tier it takes auto-slot 1 alone and the second auto-pick is one uniform draw from
    the tie X left behind, so the question is whether E[max(X, draw)] beats E[max(draw, draw')].
    That crosses the status quo at H below the pick, and a file HELPS iff its cross-fitted CV is
    within H. Read LIVE from the artefact -- w45a went stale by freezing a live quantity into
    source, and H moves with the tier, so this must be re-derived on every send day.

    ⚠ The bar taken is the CONDITIONAL one. A hijack is by definition the branch where the file
    landed high on the public slice, and w57a fits gamma = -0.092 -- the public-gap to
    private-gap map is NEGATIVE -- so conditioning on that landing LOWERS the file's expected
    private score. The unconditional break-even (12.97e-6) is anti-conservative; the conditional
    one (10.59e-6) is the bar. Both are in the JSON; `H_binding` is the stricter.

    ⚠ w59a GATE I: this is the SAME number as w58's DILUTION break-even D, identically, because
    the pairs a 6th tier member adds to MODEL B are exactly the pairs a hijacker draws from --
    delta_dilution(X) = (cost_hijack(X) - base)/3. One bar governs both landings. The hijack
    simply carries 3x the leverage, in either direction.

    Returns None if the artefact cannot supply it -- callers must then BLOCK, not wave through.

    ⚠⚠ w62: `rows` IS NOT OPTIONAL DECORATION. Until this run the staleness test was
    `d["gate_t"] == "PASS"` and nothing else, and the line above it read "The artefact must have
    been produced on the tier we are actually sending against." THOSE ARE NOT THE SAME CLAIM.
    `gate_t` is a STAMP recording that w59a's GATE T passed **on the day the artefact was
    written**; a frozen stamp cannot notice that the board moved afterwards, and this bar moves
    with the tier -- the docstring above says so itself ("H moves with the tier, so this must be
    re-derived on every send day"). The 08-23 sends moved tier 1 from five files at 0.97118 to
    two at 0.97119 and `w59a_hijackprice.py` now REFUSES to run on its own GATE T, while this
    function went on returning 0.9701294160 from the artefact GATE T just voided. The same shape
    as w58's "a rule enforced on the wrong COLUMN" and w60's "a rule enforced on a column a
    DIFFERENT script populates": the comment states the rule, the code checks a proxy for it.
    So the comparison is made HERE, against the LIVE board, as a SET and not a count (w61 §5).
    ⛔ Do not "fix" a void bar by passing the check -- re-run the pricer chain. Failing safe
    costs at most the choice of one filler; a bar derived on a board that no longer exists is
    how an above-tier file is admitted on a number that prices nothing.
    """
    try:
        d = _json.load(open(HIJACKPRICE))
        bar = float(d["cv_bar_new"])
        # The artefact must have been produced on the tier we are actually sending against.
        if not d.get("gate_t") == "PASS" or not (0.9 < bar < 1.0):
            return None
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if rows is not None:
        live = live_tier1(rows)
        rec = d.get("tiers", {}).get("slot1")
        if live is None or rec is None:
            return None
        lv, lset = live
        if sorted(rec) != sorted(lset):
            print(f"\n⛔ THE w59 HIJACK CV BAR IS VOID: it was derived on auto-slot-1 "
                  f"{sorted(rec)},\n   the live board shows {sorted(lset)} @ {lv:.5f}. "
                  f"w59a_hijackprice.py refuses to run\n   on a moved tier and this bar moves "
                  f"with the tier. Blocking every above-tier file until\n   the pricer chain "
                  f"is re-run. THE FIX IS THE PRICER, NOT THE FLAG.")
            return None
    return bar


def live_tier1(rows):
    """(public value, set of stems) of the top PUBLIC score on the account, or None.

    The auto-selected pair is the best two by public score, so tier 1 is the set of files
    sharing the highest score. Reported as a SET: w61 §5 found two preregistrations that had
    registered a queue's BLOCK COUNT, a quantity that depends on where a loop stopped. A set
    difference is order-independent and is the form these comparisons take here.
    """
    best, stems = None, []
    for r in rows:
        ps = r.get("publicScore")
        if ps in (None, ""):
            continue
        v = float(ps)
        if best is None or v > best:
            best, stems = v, []
        if v == best:
            stems.append(str(r["fileName"]).replace(".csv", ""))
    if best is None:
        return None
    return best, sorted(set(stems))


def auto_tier(rows):
    """The auto-selection tier: the 2nd-highest PUBLIC score on the account, WITH multiplicity.

    Kaggle auto-selects the best two submissions by public score, so a file is capable of being
    auto-selected iff it scores at or above the second-highest score already on the board. With
    five files tied at the top the second-highest is that same tied value, which is why this
    counts duplicates rather than distinct values (w58). Returns None if the board cannot be
    read -- callers must BLOCK, not wave through.
    """
    ps = sorted((float(r["publicScore"]) for r in rows
                 if r.get("publicScore") not in (None, "")), reverse=True)
    return ps[1] if len(ps) >= 2 else None


# w58: the pricer's own residual sd, WIDE branch (ad>=195). The sender owns a constant rather
# than importing w53a, because w57 §4 showed that putting a priced module on the send path is
# how the sender stops importing. `w58a_tiergate.py` ASSERTS this equals w53a's wide branch and
# exits non-zero if the pricer is refitted -- the check lives outside the send path, on purpose.
PRED_SD = 8.77e-6
STEP = 1e-5              # the public LB reports to 5 decimals
P_MAX = 0.02             # tolerated P(a non-final-entry file lands ABOVE the tier)


def hijack_risk(pred_lb, tier):
    """P(this file's true public score lands strictly ABOVE the tier), on the live tier.

    ⚠ A POINT PREDICTION IS NOT A GATE. The first cut of this filter thresholded `pred_lb >=
    tier` and let `w36_ad197std_logit` (pred_lb 0.971177, cv 94e-6 BELOW the pick) through on a
    3e-6 margin against a residual sd of 8.77e-6 -- an 18% chance of the exact hijack the gate
    exists to prevent. The displayed score is rounded to `STEP`, so "above the tier" means above
    `tier + STEP/2` on the underlying scale.
    """
    from math import erf, sqrt
    z = (tier + STEP / 2 - float(pred_lb)) / PRED_SD
    return 0.5 * (1.0 - erf(z / sqrt(2.0)))


def above_tier_reason(r, bar):
    """Why this at-or-above-tier row must not be sent, or None if it is an acceptable final entry.

    THE RULE (w58): while nothing is selected, a file above the tier is not a filler, it is a
    FINAL ENTRY. So it must clear the bar a final entry clears.
    """
    fam = str(getattr(r, "fam", ""))
    stem = str(r.file).replace(".csv", "")
    if fam == "member":
        return ("fam=member — a raw member's OOF AUC is not a cross-fitted stack CV (w53), "
                "so it cannot be compared against the pick at all")
    for pat, why in WANTED_INELIGIBLE.items():
        if stem.startswith(pat):
            return f"w40d-ineligible arm — {why}"
    cv = getattr(r, "cv", None)
    if cv is None or pd.isna(cv):
        return "no cv — cannot be shown to be an acceptable final entry"
    if bar is None:
        return ("the w59 hijack CV bar could not be read from w59a_hijackprice.json — failing "
                "safe. Re-run w57a_tierprice2.py then w59a_hijackprice.py.")
    if float(cv) < bar:
        return (f"cv {float(cv):.10f} < the w59 hijack bar {bar:.10f} — above the tier it would "
                f"take auto-slot 1 from a uniform tier-1 draw and make E[max] WORSE, at 3x the "
                f"leverage of an in-tier landing (w59a GATE I)")
    return None
LOG = os.path.join(HERE, "w26g_sent.csv")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def api_submissions():
    """The live list, with an explicit page size and a truncation guard."""
    r = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMP, "-v",
                        "--page-size", str(PAGE)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or "id,fileName" not in r.stdout and "ref,fileName" not in r.stdout:
        raise SystemExit(f"kaggle submissions failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    rows = list(csv.DictReader(io.StringIO(r.stdout)))
    if len(rows) >= PAGE:
        raise SystemExit(f"submission list came back at exactly the page size ({len(rows)}); "
                         f"it is truncated. Raise PAGE before trusting anything below.")
    return rows


def sent_today(rows):
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    return [r for r in rows if r["date"][:10] == today], today


def check_file(path):
    """A submission that is malformed scores zero and burns the slot. Cheap to check."""
    if not os.path.exists(path):
        return f"missing: {path}"
    d = pd.read_csv(path)
    if list(d.columns) != ["id", "addicted_label"]:
        return f"columns {list(d.columns)} != ['id','addicted_label']"
    if len(d) != N_TEST:
        return f"{len(d):,} rows != {N_TEST:,}"
    v = d["addicted_label"].to_numpy()
    if not pd.Series(v).notna().all():
        return "non-finite values"
    if d["id"].duplicated().any():
        return "duplicate ids"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually submit. Without it, plan only.")
    ap.add_argument("--n", type=int, default=DAILY_CAP, help="max files to send this run")
    ap.add_argument("--tag", default="w26g", help="prefix for the submission message")
    # ⛔ THE VETO IS A DEFERRAL UNTIL SOMETHING MAKES IT A BLOCK (w54). `w48e_order.py` encodes
    # the veto as `priority = -1` and this sender sorts on priority descending, so a vetoed file
    # merely goes to the BACK of the plan. That is safe only while the queue is longer than the
    # slots that remain, and w54a computed the date it stops being true: on 2026-08-22 there were
    # 82 unsent files and 90 slots left before the 08-31 deadline, so a mechanical drain at the
    # cap reaches the first vetoed file on 2026-08-29 and sends all 19 of them by the deadline --
    # including `w50_ad216std_logit`, the logit family term on the highest CV base on disk, which
    # `w48e` calls "the single most dangerous file in submissions/". While `check_selection`
    # reports nothing selected, Kaggle auto-selects on best PUBLIC score, which is precisely the
    # exposure the veto exists to close.
    #
    # So the filter is here, it defaults to ON, and it is deliberately NOT conditional on a live
    # `check_selection` read: an API hiccup must not silently un-veto the queue. Fail safe.
    # Retiring a veto is an evidence decision and belongs in `w48e.VETO`, one entry at a time,
    # with the reason written down -- not in a flag on the send line.
    ap.add_argument("--allow-vetoed", action="store_true",
                    help="send priority<0 files anyway. Only defensible once selection is "
                         "confirmed made; retire the VETO entry in w48e_order.py instead.")
    # Same reasoning one level up, and it is the SAME BUG (w55). w54 established the number
    # that makes a spare slot free or a liability -- "a filler is SAFE iff its predicted public
    # score is < the 0.97118 auto-selection tier" -- and wrote it into RESEARCH.md and into
    # this file's own unfilled-slots message. Nothing enforced it, and four queue rows carry
    # `pred_lb = NaN`, so it could not have been enforced on them anyway. w48e's comment
    # defended those rows with "they keep priority 0 and sort last", which is exactly the
    # argument w54 refuted for the veto: last is reached on ~2026-08-29. Certification belongs
    # in `w55a_unpriced.py`, on evidence, not in a flag on the send line.
    ap.add_argument("--allow-unpriced", action="store_true",
                    help="send rows with no pred_lb anyway. The wrong tool: certify them with "
                         "experiments/w55a_unpriced.py and re-run w48e_order.py --write.")
    # w58 -- THE SAME BUG A THIRD TIME, one level up again. w54 wrote the tier rule, w55
    # enforced only its NaN branch, and a row WITH a `pred_lb` that sits AT OR ABOVE the tier
    # still walked straight through. The 08-23 plan's slot 1, `w48_cal_hboyang_mix`, is exactly
    # that row: pred_lb 0.971230 against a 0.97118 tier, `fam == "member"`, and barred from
    # CV-based selection by `check_selection.WANTED_INELIGIBLE`. While `check_selection` exits
    # 1, Kaggle auto-selects the best TWO by public score, so sending it IS selecting it.
    # w58a prices the hijack at up to +17.95e-6 against a +7.86e-6 status quo -- 2.28x -- and
    # w56's registered conjunction says NO branch of the ARM 217 read can move WANTED, so the
    # send has zero expected value for the private score against a real cost. Unconditional and
    # default-on, exactly like the two filters above.
    ap.add_argument("--allow-above-tier", action="store_true",
                    help="send files predicted at or above the auto-selection tier anyway. "
                         "Only defensible once `check_selection` exits 0 — with the final picks "
                         "chosen, auto-selection does not apply and the tier stops mattering.")
    a = ap.parse_args()

    rows = api_submissions()
    today, daystr = sent_today(rows)
    left = DAILY_CAP - len(today)
    print(f"{len(rows)} submissions on record; {len(today)} already sent on {daystr} (UTC); "
          f"{left} of {DAILY_CAP} slots left today")

    # w58: the tier is LIVE, never a constant -- a hard-coded tier is what went stale in w45a.
    TIER = auto_tier(rows)
    CVBAR = hijack_cv_bar(rows)
    if TIER is None:
        print("\n⛔ could not read the auto-selection tier from the board. Refusing to plan: "
              "the tier rule cannot be evaluated, and w55's lesson is that a row the rule "
              "cannot be evaluated on is not covered by it.")
        sys.exit(2)
    print(f"auto-selection tier {TIER:.5f} (2nd-highest public score, with multiplicity); "
          f"w59 hijack CV bar {CVBAR if CVBAR is None else f'{CVBAR:.10f}'} "
          f"(measured, not the old WANTED bar)")

    sent_names = {r["fileName"] for r in rows}
    sent_md5 = set()
    for n in sent_names:
        p = os.path.join(SUB, n)
        if os.path.exists(p):
            sent_md5.add(md5(p))
    print(f"{len(sent_names)} distinct filenames sent, {len(sent_md5)} of them still on disk "
          f"and fingerprinted")

    q = pd.read_csv(QUEUE)
    # ⚠ THE QUEUE MUST BE KEYED TO TODAY (w53). w49 keyed `w48e_order.py`, the WRITER, to the UTC
    # day so a registered list could never be written for the wrong one. Nothing keyed the READER,
    # and the gap is not theoretical: w53 dry-ran this sender against the 08-22 CSV on the evening
    # of 08-22 and it planned a ten that (a) omitted `w48_cal_hboyang_mix`, slot 1 of the
    # registered 08-23 list, and (b) included two `logit`-family files. Both follow from the same
    # mechanism -- every priority-1 row was already sent, so the plan fell through to the
    # priority-0 tail, where no veto applies because `w48e` enforces the veto as priority -1 and
    # DROPS the `vetoed` column before writing. A stale queue is an UNVETOED queue.
    #
    # A dry run is allowed to look at a queue for another day -- that is how a run at the cap
    # inspects tomorrow's plan -- but it says so loudly. A real send refuses.
    _day = str(q["plan_day"].iloc[0]) if "plan_day" in q.columns and len(q) else None
    if _day != daystr:
        _msg = (f"queue was written for {_day or 'AN UNSTAMPED DAY (pre-w53 artefact)'}, "
                f"not today ({daystr})")
        if a.go:
            print(f"\n⛔ REFUSING TO SEND: {_msg}.\n"
                  f"   Run `w48e_order.py --day {daystr} --write` first. Do not send off a stale "
                  f"queue: its priority-1 band is another day's list, and the files behind it "
                  f"carry a stale day's veto.")
            return 1
        # ⚠ NOT "carries no veto" any more (w54). The priority<0 filter above is unconditional, so
        # a stale CSV's vetoed rows are still blocked. What a stale CSV loses is everything the
        # veto has learned SINCE it was written -- `w48e.VETO` is applied at WRITE time and the
        # `vetoed` column is dropped, so a row vetoed today reads as priority 0 in yesterday's
        # artefact and sails through. Re-write, don't reason about it.
        print(f"\n⚠ DRY RUN AGAINST A QUEUE FOR ANOTHER DAY: {_msg}. The plan below is NOT the "
              f"registered list for today, and it carries only the veto as it stood on "
              f"{_day or 'the day it was written'}.")
    # `priority` pins `check_selection.WANTED` to the head (w28). A deadline pick that is never
    # submitted cannot be selected, and that outranks any public-LB ordering.
    if "priority" not in q.columns:
        q["priority"] = 0
    # `send_rank` (w37) overrides the pred_lb tiebreak inside a priority band. It exists for
    # files whose send ORDER carries information that their predicted LB does not: the w37
    # es-bias calibration sends are deliberately low-scoring member vectors, and the CLEAN
    # anchor among them must land before the DIRTY readings it is there to make interpretable.
    # Blank/absent = default behaviour, so every pre-w37 row is unaffected.
    if "send_rank" not in q.columns:
        q["send_rank"] = float("nan")
    q["send_rank"] = pd.to_numeric(q["send_rank"], errors="coerce").fillna(1e9)
    q = q.sort_values(["priority", "send_rank", "pred_lb"], ascending=[False, True, False])
    _pin = q[q.priority == 1].file.tolist()
    if _pin:
        # Priority 1 is no longer WANTED-only: w37 pins calibration sends into the same band.
        # Label each one, so the plan never implies a measurement file is a deadline pick.
        try:
            from check_selection import WANTED as _W
        except Exception:
            _W = set()
        print("PINNED first (priority 1):")
        for f in _pin:
            print(f"  {f:34s} {'check_selection.WANTED, unsent' if f in _W else 'pinned, NOT a deadline pick'}")
    # A dry run plans the full --n regardless of slots left, so a slot at the cap can still
    # SEE tomorrow's queue and check it is sane. Only a real send is clamped by `left`.
    cap = a.n if not a.go else min(a.n, max(left, 0))
    plan, seen_md5, blocked, unpriced, above = [], set(), [], [], []
    for r in q.itertuples():
        if len(plan) >= cap:
            break
        if r.file in sent_names:
            continue
        if int(getattr(r, "priority", 0)) < 0 and not a.allow_vetoed:
            # Not "skip": BLOCKED. Collected and reported after the plan so it cannot scroll off.
            blocked.append(r.file)
            continue
        # w55: unpriceable. A row with no `pred_lb` is a row the auto-selection tier rule
        # cannot be evaluated on, and its submission message renders as "predicted LB nan ...
        # P(beat) nan", destroying the provenance trail every later run reads back. `w48e`
        # prices such rows from `w55a_unpriced.json` once w55a has CERTIFIED them below the
        # tier on a calibrated instrument; a row that reaches here still NaN has been certified
        # by nothing. Unconditional and default-on, exactly like the veto filter above.
        # NaN `cv` is tolerated ONLY where the row has renounced being a candidate by declaring
        # fam == "member" (w53: a member's published OOF is not a cross-fitted stack CV).
        _why = None
        if pd.isna(getattr(r, "pred_lb", None)):
            _why = "no pred_lb — never certified against the auto-selection tier"
        elif pd.isna(getattr(r, "cv", None)) and str(getattr(r, "fam", "")) != "member":
            _why = "no cv and fam != member — undeclared candidate carrying no CV"
        if _why and not a.allow_unpriced:
            unpriced.append((r.file, _why))
            continue
        # w58: AT OR ABOVE THE TIER. Not a filler -- a final entry. See --allow-above-tier.
        # Two-part test, and the order is the point: first ask whether the row could honestly BE
        # a final entry, and only if it could not, ask how likely it is to become one.
        _pl = getattr(r, "pred_lb", None)
        if not a.allow_above_tier:
            _aw = above_tier_reason(r, CVBAR)
            if _aw:
                _risk = 1.0 if (_pl is None or pd.isna(_pl)) else hijack_risk(_pl, TIER)
                if _risk >= P_MAX:
                    above.append((r.file, float("nan") if _pl is None or pd.isna(_pl)
                                  else float(_pl), _risk, _aw))
                    continue
        p = os.path.join(SUB, r.file)
        m = r.md5 if isinstance(getattr(r, "md5", None), str) else None
        if m is None and os.path.exists(p):
            m = md5(p)
        if m in sent_md5:
            print(f"  skip {r.file}: identical predictions already sent (md5 {m[:8]})")
            continue
        if m in seen_md5:
            print(f"  skip {r.file}: duplicate of another file already in this plan")
            continue
        why = check_file(p)
        if why:
            print(f"  skip {r.file}: {why}")
            continue
        # the md5 on the queue CSV is a claim; verify it rather than trust it
        real = md5(p)
        if m and real != m:
            print(f"  skip {r.file}: md5 on disk {real[:8]} != queue's {m[:8]}, file changed")
            continue
        seen_md5.add(real)
        plan.append(r)

    if blocked:
        print(f"\n  ⛔ {len(blocked)} VETOED file(s) skipped, not sent (w54). Reasons are in "
              f"`w48e_order.py`'s VETO dict:")
        for f in blocked[:8]:
            print(f"       {f}")
        if len(blocked) > 8:
            print(f"       ... and {len(blocked) - 8} more")
        print("     A vetoed file is unsendable while `check_selection` reports nothing "
              "selected.")
        print("     Retire the entry in w48e_order.py on evidence; do NOT pass --allow-vetoed.")

    if unpriced:
        print(f"\n  ⛔ {len(unpriced)} UNPRICEABLE file(s) skipped, not sent (w55). The "
              f"auto-selection tier\n     rule cannot be evaluated on a row with no price, so "
              f"the row is not sendable:")
        for f, why in unpriced[:8]:
            print(f"       {f:34s} {why}")
        if len(unpriced) > 8:
            print(f"       ... and {len(unpriced) - 8} more")
        print("     Certify with `.venv/bin/python experiments/w55a_unpriced.py`, then re-run"
              "\n     `w48e_order.py --day <today> --write`. Do NOT pass --allow-unpriced.")

    if above:
        print(f"\n  ⛔ {len(above)} file(s) with P(landing above the {TIER:.5f} "
              f"AUTO-SELECTION TIER) >= {P_MAX:.0%}\n     skipped, not sent (w58). Nothing is "
              f"selected, so Kaggle auto-picks the best two by PUBLIC\n     score: a file above "
              f"the tier is not a filler, it is a FINAL ENTRY, and must clear\n     the bar a "
              f"final entry clears. Risk is on the LIVE tier at sd {PRED_SD*1e6:.2f}e-6.")
        for f, pl, risk, why in above[:8]:
            print(f"       {f:34s} pred_lb {pl:.6f}  P(above tier) {risk:.3f}")
            print(f"       {'':34s} {why}")
        if len(above) > 8:
            print(f"       ... and {len(above) - 8} more")
        print("     THE FIX IS THE CLICK, not the flag: once `check_selection` exits 0 these "
              "become\n     sendable and --allow-above-tier is the right tool. Until then it "
              "is the wrong one.")

    if not plan:
        print("\nnothing to send: the queue is drained of everything sendable.")
        if blocked:
            print("  ⚠ SLOTS WILL GO UNFILLED. That is the intended trade: the brief's "
                  "\"an extra\n    submission can never hurt\" is FALSE on this account while "
                  "nothing is selected,\n    because Kaggle then auto-selects on best PUBLIC "
                  "score. Build something new and\n    non-vetoed to fill them; do not reach "
                  "for the veto list.")
        return 0

    print(f"\nplan, {len(plan)} file(s):")
    for i, r in enumerate(plan, 1):
        print(f" {i:2d}. {r.file:34s} CV {r.cv:.10f}  pred_lb {r.pred_lb:.6f}  "
              f"P(beat best) {r.p_beat:.2e}")

    if not a.go:
        print("\nDRY RUN. Re-run with --go to send.")
        return 0
    if left <= 0:
        print("\nat the daily cap; refusing to send.")
        return 1

    for r in plan:
        # A `msg` on the queue row replaces the queue-drain boilerplate. Without this, a w37
        # calibration send would be described as a queue-drain attempt and a future run reading
        # the submission history would misread its ~0.958 public score as a huge regression.
        override = getattr(r, "msg", None)
        # ⚠ NaN-safe (w55). The default template formats `cv` and `p_beat`; a `member` row has
        # neither, and would ship "CV nan ... P(beat) nan". `w48e` writes a proper `msg` for
        # every row it certifies, so this branch should never fire -- it is the fail-safe for
        # a certified row that somehow arrives without one, and it refuses rather than lie.
        _has_msg = isinstance(override, str) and override.strip()
        if not _has_msg and pd.isna(getattr(r, "cv", None)):
            print(f"  skip {r.file}: fam={r.fam} row has no cv and no registered msg; "
                  f"re-run w48e_order.py --write so it is described honestly")
            continue
        msg = (f"{a.tag} queue-drain {r.stem} — CV {r.cv:.10f}, family {r.fam}, "
               f"w26d predicted LB {r.pred_lb:.6f} with P(beats the 0.97118 account best) "
               f"{r.p_beat:.2e}. Sent because the brief's economics make an unused slot pure "
               f"waste, NOT because it is expected to move anything: every unsent file is "
               f"below the best already-sent CV and w26d prices the whole queue at 6.3e-4 of "
               f"beating the board. Not a deadline candidate — selection is on CV and this "
               f"is {(0.9701150809 - r.cv)*1e6:.1f}e-6 below the best sent CV.")
        if isinstance(override, str) and override.strip():
            msg = override.strip()
        p = os.path.join(SUB, r.file)
        print(f"\n--> {r.file}", flush=True)
        out = subprocess.run(["kaggle", "competitions", "submit", "-c", COMP, "-f", p,
                              "-m", msg], capture_output=True, text=True, timeout=1200)
        print((out.stdout + out.stderr).strip()[-600:])
        fresh = not os.path.exists(LOG)
        with open(LOG, "a") as f:
            w = csv.writer(f)
            if fresh:
                w.writerow(["utc", "file", "cv", "pred_lb", "p_beat", "md5", "rc"])
            w.writerow([dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        r.file, r.cv, r.pred_lb, r.p_beat, md5(p), out.returncode])

    # `submit` has returned 400 after a 100% upload with nothing registering (RESEARCH.md).
    # Never trust the exit status; re-read the list.
    after, _ = sent_today(api_submissions())
    print(f"\nconfirmed from the API: {len(after)} submissions today (was {len(today)}). "
          f"{DAILY_CAP - len(after)} slots left.")
    for r in after[:len(plan)]:
        print(f"  {r['date']}  {r['fileName']:34s} {r.get('publicScore','')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
