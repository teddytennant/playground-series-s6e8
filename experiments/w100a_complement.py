"""w100a — STANDING GUARD: EVERY UNSENT FILE MUST BE ACCOUNTED FOR, NOT JUST EVERY SLOT.

WHAT CLAIM THIS COVERS, AND WHY NOTHING ELSE COVERS IT. Every guard on this workspace reads
the calendar forwards — from the slots outwards:

    w85c_slotguard    are there enough sendable files for the remaining slots?
    w87a_registrarguard  is every REGISTERED file one the sender would plan?
    w98 §1            does every one of the 40 planned files exist on disk?

All three ask "is the calendar covered". None of them asks the complement: **of the files
sitting unsent in `submissions/`, is every single one accounted for?** A file that is unsent,
unplanned, unvetoed and unrefused is invisible to all of the above. It is not a gap in the
calendar, so no slot instrument sees it; it is not registered, so the registrar guard never
looks at it. It just sits in `submissions/` where `w23b_sendqueue` globs it — and if its CV is
high it sits at the TOP of any CV ordering, waiting for the next run that ranks the queue on
CV to promote it into a send list.

⚠ THAT IS NOT HYPOTHETICAL. `w48e_order.py` says so in its own comment, about a file that had
already happened to it: "anything that ranks the queue on CV puts it near the top ... That is
precisely how `w42_ad217std_logit` got planned into a send list." The 19-file `VETO` exists
because of it. This check asks whether the VETO, the refusals and the calendar between them
still cover the whole unsent set — the thing the VETO's own existence assumes and nothing
tests.

At the time of writing, 231 csvs sit in `submissions/`, 161 are sent, and the 70 unsent
partition 40 planned / 19 vetoed / 3 refused / 4 byte-identical twins of a sent file / 3
registrar headroom / 1 sender-refused, complement EMPTY. The single sender-refused file is
`w48_cal_hboyang_mix`:
highest CV in the queue (0.9701815536), P(above the tier) 1.000, and NOT in `VETO`. Its only
barriers are that its registration is for a past day and that `w26g_send` refuses it live. That
is a real barrier and C3 exercises it rather than repeating the prose — but it is one barrier
where the ad216/ad217 families have two, and a reader who finds this file at the top of a CV
sort deserves to be told which one is holding.

CONTROLS (w72 §5.3: a control that can only fail is not a control; both directions or it is
decoration).
  C1 +   the complement is EMPTY — every unsent file is planned, vetoed, refused, or refused by
         the sender's own live gate.
  C2 +-  FIRED BOTH WAYS. A synthetic unaccounted stem must land in the complement, and
         emptying `VETO` must GROW the complement. If neither moves it, C1 is passing for the
         wrong reason.
  C6 +   the "deliberate spare" bucket is READ from `w87a_registrarguard.json`
         (`headroom_stems` = plannable - registered), never re-typed from RESEARCH's
         "3 spare (gnb_raw, qda_raw, gmm_raw)", and it is refused unless w87a's own day
         stamp is today — headroom moves every send day (the w62b class of bug).
  C5 +   HOW MANY FILES THE VETO ALONE HOLDS — reported, and it is smaller than the veto is.
         15 of the 19 vetoed files are independently refused by the sender's live gate, so the
         `VETO` entry is redundant for them. It is the ONLY barrier for exactly four:

             w50_ad216std_h3          cv 0.9701459152  pred_lb 0.971176  P(above) 0.014
             w50_ad216std_hybrid      cv 0.9701365515  pred_lb 0.971166  P(above) 0.001
             w50_ad216std_rankraw     cv 0.9701134182  pred_lb 0.971143  P(above) 0.000
             w29_ad194stdcorr_rescale cv 0.9701012580  pred_lb 0.971175  P(above) 0.012

         They fall through the sender because their predicted LB sits just UNDER the tier, so
         `hijack_risk` comes back below `P_MAX` and the two-part test never completes. ⚠ The
         first of them carries a CV 5.9e-6 ABOVE the WANTED final pick, on the contaminated
         ad216 arm. Delete the `VETO` entry and that file is sendable, plausible-looking on a
         CV sort, and one 2.2-sigma draw from being auto-selected. This number is the veto's
         real load and it belongs in an artefact, not in a belief.
  C3 +   the sender-refusal leg is COMPUTED by importing `w26g_send.above_tier_reason` at the
         LIVE tier and bar — never restated from RESEARCH prose. A file in that bucket with a
         `None` reason is a hole, not an accounting.
  C4 +   sentness is read from the API, never from the queue's `sent` column. That column is
         VACUOUS — `w23b_sendqueue` only ever writes unsent rows, so every value is False and
         it cannot detect a sent file even in principle. C4 asserts the vacuity rather than the
         disagreement: today all 37 rows read False and all 37 really are unsent, so a
         disagreement count would be 0 and would read as CONFIRMATION of a column that carries
         no information (w98 §5 hit exactly this, on a 66-row snapshot).

Read-only. Writes `w100a_complement.json` and exits non-zero on any failure.
"""
from __future__ import annotations

import hashlib, json, os, sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import w26g_send as S                                                 # noqa: E402
import w48e_order as W                                               # noqa: E402

OUT = os.path.join(HERE, "w100a_complement.json")

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  ⛔ {msg}")


SUB = os.path.join(os.path.dirname(HERE), "submissions")
N_TEST = 296_302          # same constant `w23b_sendqueue.N_TEST` filters on


def disk_frame(api_rows):
    """EVERY csv in `submissions/`, joined to the priced queue, marked sent from the API.

    ⚠ THE UNIVERSE IS THE DIRECTORY, NOT THE QUEUE. The first cut of this check took the
    priced queue as its universe and inherited both of that file's own exclusions: it drops
    byte-identical twins and wrong-row-count files before it is written, and this check then
    reported a complement of zero over a set those files were never in. 231 csvs sit in
    `submissions/`; the queue carries 66 of them. `cv.notna()` cost another 29 — the whole
    certified `w85_cal_*` member pool, which is 26 of the 40 files the calendar still plans.
    A check whose universe is a filtered artefact measures the filter, not the risk.
    """
    have = sorted(f for f in os.listdir(SUB) if f.endswith(".csv"))
    md5s, nrows = {}, {}
    for f in have:
        blob = open(os.path.join(SUB, f), "rb").read()
        md5s[f], nrows[f] = hashlib.md5(blob).hexdigest(), blob.count(b"\n") - 1
    d = pd.DataFrame({"file": have})
    d["stem"] = d.file.str.replace(".csv", "", regex=False)
    d["md5"] = d.file.map(md5s)
    d["rows"] = d.file.map(nrows)
    sent_api = {str(r["fileName"]).replace(".csv", "") for r in api_rows}
    d["sent_api"] = d.stem.isin(sent_api)

    q = pd.read_csv(S.QUEUE)
    q["stem"] = q.file.astype(str).str.replace(".csv", "", regex=False)
    d = d.merge(q[["stem", "cv", "pred_lb", "fam"]], on="stem", how="left")
    d["in_queue"] = d.stem.isin(set(q.stem))
    return d, q, sent_api


def planned_and_refused(today: str):
    """Every file the calendar still offers a slot to, and every file it explicitly refused."""
    planned, refused, days = {}, {}, []
    for fn in sorted(os.listdir(HERE)):
        if not (fn.startswith("w72a_plan_") and fn.endswith(".json")):
            continue
        p = json.load(open(os.path.join(HERE, fn)))
        if p.get("day", "") < today:
            continue
        days.append(p["day"])
        for stem in p.get("plan", []):
            planned.setdefault(stem, p["day"])
        for r in p.get("refused") or []:
            refused.setdefault(r["stem"], r.get("why", "")[:200])
    return planned, refused, days


def headroom_stems(today):
    """The registrar's OWN spare list, read from `w87a_registrarguard.json`.

    ⚠ NOT RE-TYPED. RESEARCH says "60 registered files for 60 slots, 3 spare (gnb_raw,
    qda_raw, gmm_raw)" and copying that sentence into this file would be the prose-only-veto
    mistake at one remove — a second hand-maintained copy of a live set. w87a already computes
    `plannable - registered`; w100 made it publish the stems instead of only the count.
    """
    try:
        d = json.load(open(os.path.join(HERE, "w87a_registrarguard.json")))
    except (OSError, ValueError):
        return None, "w87a_registrarguard.json is unreadable"
    if d.get("failures"):
        return None, f"w87a reports {len(d['failures'])} failure(s) — its headroom is not usable"
    if "headroom_stems" not in d:
        return None, "w87a_registrarguard.json predates `headroom_stems` — re-run w87a"
    # ⚠ FRESHNESS. w87a stamps the UTC day it ran. Headroom is `plannable - registered` and
    # both halves move every send day, so yesterday's spare list is not this day's. w62b
    # exists for exactly this class of bug — an artefact read without its stamp — and the
    # remedy is the same: refuse, do not interpolate.
    if d.get("day") != today:
        return None, (f"w87a_registrarguard.json is stamped {d.get('day')!r}, not today "
                      f"({today}). Re-run w87a; headroom moves every send day.")
    return set(d["headroom_stems"]), None


def classify(d, planned, refused, veto, tier, bar, spare=frozenset()):
    """Assign each unsent file to exactly one bucket; the leftovers are the complement."""
    order = ["planned", "vetoed", "refused", "dup-of-sent", "wrong-rows", "headroom",
             "sender", "COMPLEMENT"]
    buckets = {k: [] for k in order}
    reasons = {}
    sent_md5 = set(d[d.sent_api].md5)
    for r in d[~d.sent_api].itertuples():
        if r.stem in planned:
            buckets["planned"].append(r.stem)
        elif r.stem in veto:
            buckets["vetoed"].append(r.stem)
        elif r.stem in refused:
            buckets["refused"].append(r.stem)
        elif r.md5 in sent_md5:
            buckets["dup-of-sent"].append(r.stem)
        elif r.rows != N_TEST:
            buckets["wrong-rows"].append(r.stem)
        elif r.stem in spare:
            buckets["headroom"].append(r.stem)
        else:
            why = S.above_tier_reason(r, bar) if r.in_queue else None
            risk = (S.hijack_risk(r.pred_lb, tier)
                    if r.in_queue and pd.notna(r.pred_lb) else float("nan"))
            if why is not None and pd.notna(risk) and risk >= S.P_MAX:
                buckets["sender"].append(r.stem)
                reasons[r.stem] = (why, float(risk))
            else:
                buckets["COMPLEMENT"].append(r.stem)
                reasons[r.stem] = (why, float(risk) if pd.notna(risk) else None)
    return buckets, reasons


def main() -> int:
    api = S.api_submissions()
    today = S.sent_today(api)[1]
    tier = S.auto_tier(api)
    bar = S.hijack_cv_bar(api)
    if tier is None or bar is None:
        fail("the live tier or the hijack CV bar could not be read — refusing to accept "
             "an accounting computed against a missing gate")
        return 2

    d, q, sent_api = disk_frame(api)
    planned, refused, days = planned_and_refused(today)
    print(f"\nw100a — {len(api)} submissions on the API, tier {tier:.5f}, hijack CV bar "
          f"{bar:.10f}\n  calendar days still open: {days}")

    # ---- C4: the queue's own `sent` column is NOT the authority ------------------------------
    col_true = int(q.sent.astype(str).str.lower().eq("true").sum())
    qs = q.merge(d[["stem", "sent_api"]], on="stem", how="left")
    disagree = int((qs.sent.astype(str).str.lower().eq("true") != qs.sent_api).sum())
    print(f"\n  C4  sentness read from the API. The queue's `sent` column reads True on "
          f"{col_true} of {len(q)} rows and disagrees with the API on {disagree}.")
    if col_true:
        fail(f"C4: `w23b_sendqueue.csv` is documented to hold unsent rows only, but {col_true} "
             f"rows read sent=True. The queue writer and this check disagree about what the "
             f"file contains — do not trust either until that is resolved.")
    else:
        print("        the column is VACUOUS (all False) and carries no information. A zero "
              "disagreement here is NOT corroboration. OK")

    spare, why_no_spare = headroom_stems(today)
    if spare is None:
        fail(f"C6: the registrar's headroom could not be read — {why_no_spare}. Refusing to "
             f"treat any unplanned file as a deliberate spare on this run.")
        spare = frozenset()
    else:
        print(f"\n  C6  registrar headroom read from w87a: {len(spare)} deliberate spare(s) "
              f"{sorted(spare)}")
    buckets, reasons = classify(d, planned, refused, set(W.VETO), tier, bar, spare)
    n_unsent = int((~d.sent_api).sum())

    print(f"\n  C1  {len(d)} csvs in submissions/, {int(d.sent_api.sum())} sent. "
          f"The {n_unsent} unsent partition as:")
    for k in buckets:
        print(f"        {k:12s} {len(buckets[k]):3d}")
    for stem in buckets["sender"]:
        why, risk = reasons[stem]
        print(f"        sender-refused: {stem} — P(above tier) {risk:.3f} ≥ P_MAX "
              f"{S.P_MAX}; {why[:90]}")
    if buckets["COMPLEMENT"]:
        for stem in buckets["COMPLEMENT"]:
            row = d[d.stem == stem].iloc[0]
            why, risk = reasons[stem]
            fail(f"UNACCOUNTED: {stem} cv={row.cv} pred_lb={row.pred_lb} in_queue="
                 f"{row.in_queue} P(above tier)={risk} — not planned, not vetoed, not "
                 f"refused, not a twin of a sent file, right row count, and the sender "
                 f"would not refuse it (reason={why!r})")
    else:
        print("        complement EMPTY — every unsent file is accounted for. OK")

    # ---- C3: the sender leg must carry a real reason -----------------------------------------
    holes = [s for s in buckets["sender"] if reasons[s][0] is None]
    if holes:
        fail(f"C3: sender-refused files with no reason: {holes}")
    else:
        print(f"\n  C3  every sender-refused file carries an `above_tier_reason` computed at "
              f"the live tier/bar. OK")

    # ---- C2: both directions, or it is decoration --------------------------------------------
    ctl = dict(d.iloc[0])
    ctl.update({"stem": "w000_control_unaccounted", "file": "w000_control.csv",
                "sent_api": False, "cv": 0.97, "pred_lb": 0.9700, "fam": "h3",
                "md5": "0" * 32, "rows": N_TEST, "in_queue": True})
    fake = pd.DataFrame([ctl])
    b_pos, _ = classify(pd.concat([d, fake], ignore_index=True), planned, refused,
                        set(W.VETO), tier, bar, spare)
    if "w000_control_unaccounted" not in b_pos["COMPLEMENT"]:
        fail("C2a: a synthetic unaccounted file did NOT land in the complement — the check "
             "cannot detect the thing it exists for")
    b_neg, _ = classify(d, planned, refused, set(), tier, bar, spare)
    if not (set(buckets["vetoed"]) & set(b_neg["COMPLEMENT"])):
        fail(f"C2b: emptying VETO moved NO vetoed file into the complement — the veto is "
             f"not load-bearing in this accounting and C1 passes for the wrong reason")
    # ⚠ veto-only is the files that FALL OUT OF `vetoed` INTO the complement, not the whole
    # complement of the VETO-less run. The first cut took `b_neg["COMPLEMENT"]` whole and
    # reported 7 where the answer is 4 — it had swept in three files the veto never held.
    veto_only = sorted(set(buckets["vetoed"]) & set(b_neg["COMPLEMENT"]))
    if not FAILURES:
        print(f"  C2  FIRED both ways: synthetic file detected; VETO removal grows the "
              f"complement {len(buckets['COMPLEMENT'])} → {len(b_neg['COMPLEMENT'])}. OK")

    # ---- C5: what the VETO alone is holding, named ------------------------------------------
    print(f"\n  C5  the VETO lists {len(W.VETO)} files; the sender independently refuses "
          f"{len(W.VETO) - len(veto_only)} of them. It is the ONLY barrier for "
          f"{len(veto_only)}:")
    for stem in veto_only:
        row = d[d.stem == stem]
        if row.empty:
            print(f"        {stem:26s} (vetoed but not in the live queue)")
            continue
        row = row.iloc[0]
        print(f"        {stem:26s} cv {row.cv:.10f}  pred_lb {row.pred_lb:.6f}  "
              f"P(above tier) {S.hijack_risk(row.pred_lb, tier):.3f}")

    json.dump({"day": today, "tier": tier, "bar": bar, "n_api": len(api),
               "n_disk": len(d), "n_unsent": n_unsent, "days_open": days,
               "buckets": {k: sorted(v) for k, v in buckets.items()},
               "sender_reasons": {k: [v[0], v[1]] for k, v in reasons.items()},
               "sent_column_disagreements": disagree,
               "control_veto_removed_complement": len(b_neg["COMPLEMENT"]),
               "veto_only": veto_only, "n_veto": len(W.VETO),
               "sent_column_true_rows": col_true,
               "failures": len(FAILURES)},
              open(OUT, "w"), indent=1, sort_keys=True)
    print(f"\n  wrote {os.path.basename(OUT)} — {len(FAILURES)} failures.")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
